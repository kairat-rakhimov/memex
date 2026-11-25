"""``memex build`` — grow a knowledge graph from a seed query.

Usage:

    memex build <kg_id> --query "green hydrogen catalysts" \\
        [--sources arxiv,wikipedia,gdelt] [--docs-per-round 5] \\
        [--max-rounds 40] [--force]

Writes ``kg/<kg_id>/kg.db`` (knowledge graph + full build trace + document
text, single file), ``kg/<kg_id>/build.log`` and
``kg/<kg_id>/agent_trace_*.jsonl``.

Idempotent: re-runs skip already-fetched ``doc_id``s unless ``--force``.
Auto-resumes when ``kg/<kg_id>/kg.db`` already has a loop_state row; the
stored ``base_query`` is used and any conflicting ``--query`` is ignored with
a warning. To start a different topic, pick a new ``kg_id`` (or delete the
directory).

``SAMPLE_MODE=true`` runs the loop against deterministic stubs, offline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

from memex.common.config import Settings
from memex.common.logging_setup import configure_kg_build_logging
from memex.common.trace import TraceWriter
from memex.kg_intelligence.ids import Source
from memex.kg_intelligence.openalex import OpenAlexClient
from memex.kg_intelligence.pipeline import LoopConfig, run_pipeline
from memex.kg_intelligence.sources import (
    AdapterManager,
    ArxivAdapter,
    GdeltAdapter,
    SourceAdapter,
    WikipediaAdapter,
)
from memex.kg_intelligence.sources.common.fetch_cache import FetchCache
from memex.kg_intelligence.state import CorpusState
from memex.observability import TraceStore

VALID_SOURCES: tuple[Source, ...] = ("arxiv", "wikipedia", "gdelt")
_log = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="memex build",
        description="Grow a knowledge graph from a seed query: search, read, extract, "
        "map, resolve — round after round until returns dry up.",
    )
    parser.add_argument("kg_id", help="Identifier for this build (e.g. 'reference', 'demo').")
    parser.add_argument(
        "--query",
        default=None,
        help=(
            "Base query that drives the loop. Required for a fresh build "
            "(no kg.db / no loop_state row). On resume, the stored "
            "base_query is used and any value passed here is ignored with "
            "a warning."
        ),
    )
    parser.add_argument(
        "--sources",
        default=",".join(VALID_SOURCES),
        help="Comma-separated subset of arxiv,wikipedia,gdelt. Default: all three.",
    )
    parser.add_argument(
        "--docs-per-round",
        type=int,
        default=5,
        help="Selector budget: max docs ingested per round across all "
        "sources. Default: 5.",
    )
    parser.add_argument(
        "--per-source-search",
        type=int,
        default=10,
        help="Ranker top-K per source. Default: 10.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=40,
        help=(
            "Hard ceiling on rounds — a build-time cost backstop, not the "
            "real stop (the Strategist halts when the topic is saturated). Quick "
            "builds use ~3; full builds leave the default. Default: 40."
        ),
    )
    parser.add_argument(
        "--kg-root",
        default="kg",
        help="Parent directory for kg_id outputs. Default: kg/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch documents that were already ingested.",
    )
    return parser.parse_args(argv)


def _peek_stored_state(kg_root: Path) -> CorpusState | None:
    """Open ``kg_root/kg.db`` read-only-ish to inspect the loop_state row.

    Returns the stored ``CorpusState`` if present, else ``None``. Closes
    the store before returning so ``run_pipeline`` can open its own
    connection without contention.
    """

    if not (kg_root / "kg.db").exists():
        return None
    store = TraceStore.open(kg_root)
    try:
        blob = store.read_loop_state()
        if blob is None:
            return None
        return CorpusState.model_validate_json(blob)
    finally:
        store.close()


def _parse_sources(raw: str) -> list[Source]:
    requested = [s.strip().lower() for s in raw.split(",") if s.strip()]
    unknown = [s for s in requested if s not in VALID_SOURCES]
    if unknown:
        raise SystemExit(f"Unknown source(s): {unknown}. Choose from {VALID_SOURCES}.")
    if not requested:
        raise SystemExit("--sources cannot be empty.")
    return [s for s in VALID_SOURCES if s in requested]  # type: ignore[misc]


def _build_adapters(
    sources: list[Source], fetch_cache: FetchCache
) -> dict[str, SourceAdapter]:
    """Construct the requested source adapters, probe + register them via
    an ``AdapterManager``, and return the active ones keyed by name."""

    factories: dict[Source, Callable[[FetchCache], SourceAdapter]] = {
        "arxiv": lambda fc: ArxivAdapter(fetch_cache=fc),
        "wikipedia": lambda fc: WikipediaAdapter(fetch_cache=fc),
        "gdelt": lambda fc: GdeltAdapter(fetch_cache=fc),
    }
    manager = AdapterManager()
    manager.probe_and_register([factories[source](fetch_cache) for source in sources])
    _log.info("sources — %s", manager.summary())
    return manager.as_dict()


def _fetch_cache_from_env() -> FetchCache:
    """Resolve cache root from ``MEMEX_FETCH_CACHE``.

    Truthy (anything not in {"", "0", "false", "no"}, case-insensitive) →
    ``Path.cwd() / "cache"``. Unset / empty / `0` → disabled passthrough.
    """

    raw = os.environ.get("MEMEX_FETCH_CACHE", "").strip().lower()
    if raw and raw not in {"0", "false", "no"}:
        return FetchCache(root=Path.cwd() / "cache")
    return FetchCache(root=None)


async def _run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    sources = _parse_sources(args.sources)
    kg_root = Path(args.kg_root) / args.kg_id
    kg_root.mkdir(parents=True, exist_ok=True)

    configure_kg_build_logging(kg_root / "build.log")

    stored = _peek_stored_state(kg_root)
    resuming = stored is not None and bool(stored.base_query)
    if resuming:
        assert stored is not None  # for type narrowing
        if args.query and args.query != stored.base_query:
            _log.warning(
                "ignored --query=%r; resuming kg_id=%r with stored "
                "base_query=%r. To start a different topic, pick a new "
                "kg_id (or delete its directory).",
                args.query, args.kg_id, stored.base_query,
            )
        base_query = stored.base_query
        resume_from_round = stored.round_number
    else:
        if not args.query:
            _log.error(
                "--query is required to start a new build (no loop_state "
                "row found for kg_id=%r).",
                args.kg_id,
            )
            return 2
        base_query = args.query
        resume_from_round = 0

    request_id = f"kg_build_{args.kg_id}_{int(time.time())}"
    trace = TraceWriter(request_id=request_id, log_dir=kg_root)

    fetch_cache = _fetch_cache_from_env()
    if fetch_cache.enabled:
        _log.info("fetch-cache: enabled at %s", fetch_cache.root)
    adapters = _build_adapters(sources, fetch_cache=fetch_cache)
    openalex = (
        OpenAlexClient(mailto=settings.contact if settings.contact and "@" in settings.contact else None)
        if "arxiv" in sources
        else None
    )

    config = LoopConfig(
        per_source_search=args.per_source_search,
        selector_budget=args.docs_per_round,
        max_rounds=args.max_rounds,
    )

    _log.info(
        "kg_build: id=%s query=%r sources=%s selector_budget=%s max_rounds=%s "
        "resume_from_round=%s force=%s sample_mode=%s extract_mode=%s",
        args.kg_id, base_query, sources, config.selector_budget,
        config.max_rounds, resume_from_round, args.force, settings.sample_mode,
        settings.extract_mode,
    )

    outcome = await run_pipeline(
        base_query=base_query,
        kg_root=kg_root,
        adapters=adapters,
        openalex=openalex,
        llm_model=None,
        settings=settings,
        trace=trace,
        config=config,
        resume=True,
        force=args.force,
    )

    summary = {
        "kg_id": args.kg_id,
        "kg_root": str(kg_root),
        "rounds_completed": outcome.rounds_completed,
        "total_docs": outcome.total_docs,
        "kg_entity_count": outcome.kg_entity_count,
        "kg_relation_count": outcome.kg_relation_count,
        "schema_entity_types": outcome.schema_size[0],
        "schema_relation_types": outcome.schema_size[1],
        "stop_reason": outcome.stop_reason,
        "fetch_outcomes": outcome.fetch_outcomes,
        "trace_file": trace.relative_path,
    }
    _log.info("kg_build summary:\n%s", json.dumps(summary, indent=2))
    # Fetch failures are otherwise buried in per-round INFO lines.
    n_failed = outcome.fetch_outcomes.get("fetch_failed", 0)
    if n_failed:
        _log.warning(
            "kg_build DEGRADED: %d fetch(es) failed (%.0f%% of attempted) — by source: %s",
            n_failed,
            100 * outcome.fetch_outcomes.get("fetch_failed_rate", 0.0),
            outcome.fetch_outcomes.get("fetch_failed_by_source", {}),
        )
    if fetch_cache.enabled:
        _log.info(
            "fetch-cache: hits=%d misses=%d served_bytes=%d fetched_bytes=%d",
            fetch_cache.hits,
            fetch_cache.misses,
            fetch_cache.bytes_from_cache,
            fetch_cache.bytes_fetched_fresh,
        )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
