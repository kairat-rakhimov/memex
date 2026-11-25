"""Rebuild harness — re-ingest a whole earlier corpus through the
real per-doc knowledge loop with the current code, for A/B testing ingest-quality
fixes without re-running search / fetch / parse.

Reads every ``selected_and_fetched`` document (text + metadata) from a source
``kg.db`` (e.g. ``kg/green-hydrogen/kg.db``) in its **original round/ranker
order**, then runs the real ``extract_and_consolidate_step`` once as a single
sequential pass (vocab accumulates in doc order, exactly like a live build), and
finishes with **one final full Resolver sweep** over the whole entity set.
Bypasses planner / adapters / fetch / UsefulnessGate (the docs are pre-kept); the
mapping form-gate still runs. Fresh/empty vocab; embeddings recomputed.

The output is a FULL ``kg/<kg_id>/kg.db`` (with provenance) — the A/B engine for
ingest-quality fixes. Compare two rebuilds with ``eval_corpus_compare.py``.

Usage::

    MEMEX_EXTRACT_MODE=session \\
        python experiments/eval_rebuild.py <new_kg_id> --from-kg kg/green-hydrogen/kg.db

``SAMPLE_MODE=true`` runs against deterministic stubs (no LLM) to validate
plumbing offline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic_graph import StepContext

from memex.common.config import Settings
from memex.common.logging_setup import configure_kg_build_logging
from memex.common.trace import TraceWriter, utc_now_iso
from memex.kg_intelligence.agents import AgentCallSinks, resolve
from memex.kg_intelligence.agents.base import build_llm_model
from memex.kg_intelligence.agents.resolver import SWEEP_PERIOD
from memex.kg_intelligence.agents.vocab_resolver import cluster_and_commit
from memex.kg_intelligence.persistence import (
    DocMetadata,
    IngestionMetadata,
    QueryProvenance,
    make_context_excerpt,
    persist_document,
)
from memex.kg_intelligence.pipeline import LoopConfig
from memex.kg_intelligence.round_graph import RoundGraphDeps, RoundGraphState, _RoundDeltas
from memex.kg_intelligence.round_graph.steps import extract_and_consolidate_step
from memex.kg_intelligence.sources.common.base import ChunkedDoc, SourceCandidate
from memex.kg_intelligence.sources.common.chunker import chunk_text
from memex.kg_intelligence.state import CorpusState
from memex.observability import TraceStore

_log = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval_rebuild",
        description=(
            "Rebuild a full reference KG from a source kg.db's fetched corpus "
            "through the real per-doc knowledge loop, for ingest-quality A/B."
        ),
    )
    parser.add_argument("kg_id", help="Identifier for this build (output → kg/<kg_id>/).")
    parser.add_argument(
        "--from-kg",
        required=True,
        help="Source kg.db to pull the fetched corpus (text + order) from.",
    )
    parser.add_argument(
        "--kg-root",
        default="kg",
        help="Parent directory for kg_id outputs. Default: kg/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Build only the first N docs (round/ranker order). For quick checks.",
    )
    parser.add_argument(
        "--no-final-sweep",
        action="store_true",
        help="Skip the final full Resolver sweep (debug only).",
    )
    parser.add_argument(
        "--no-vocab-cluster",
        action="store_true",
        help="Skip the post-hoc VocabResolver predicate-clustering pass (debug).",
    )
    return parser.parse_args(argv)


def _read_corpus_plan(source_db: Path) -> tuple[str, list[tuple[str, str]]]:
    """Read ``(base_query, [(doc_id, source), …])`` from the source build.

    Doc order = original round/ranker order
    (``fetched_docs.round_id, ranker_position``), preserving the vocab-
    accumulation order of the live build. Only ``selected_and_fetched`` docs are
    replayable — the others were never text-persisted."""

    conn = sqlite3.connect(source_db)
    try:
        base_query = conn.execute("SELECT base_query FROM builds LIMIT 1").fetchone()[0]
        rows = conn.execute(
            """
            SELECT doc_id, source
            FROM fetched_docs
            WHERE outcome = 'selected_and_fetched'
            ORDER BY round_id, ranker_position
            """
        ).fetchall()
    finally:
        conn.close()
    return base_query, [(r[0], r[1]) for r in rows]


def _read_stored_doc(source_db: Path, doc_id: str) -> tuple[str, DocMetadata]:
    """Return the stored text + parsed DocMetadata for ``doc_id``."""

    conn = sqlite3.connect(source_db)
    try:
        row = conn.execute(
            "SELECT text, metadata FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise SystemExit(f"doc_id {doc_id!r} not found in {source_db}")
    return row[0], DocMetadata.model_validate_json(row[1])


def _build_chunked_doc(
    doc_id: str, source: str, text: str, stored_meta: DocMetadata
) -> ChunkedDoc:
    """Reconstruct a ChunkedDoc from stored text + metadata, re-chunked by the
    latest ``chunk_text`` (so current chunking applies).

    ``canonical`` is the stored ``doc_id`` (not the original title / arXiv id):
    the step keys off the ``fetched_by_doc_id`` dict key and the harness persists
    under the known ``doc_id`` regardless, so the canonical only needs to be
    internally consistent."""

    candidate = SourceCandidate(
        source=source,  # type: ignore[arg-type]
        canonical=doc_id,
        url=stored_meta.url,
        title=stored_meta.title,
    )
    return ChunkedDoc(
        candidate=candidate,
        cleaned_text=text,
        parse_method="harness",
        chunks=chunk_text(text),
    )


def _persist_and_record(
    store: TraceStore,
    sinks: AgentCallSinks,
    base_query: str,
    doc_id: str,
    fd: ChunkedDoc,
) -> int:
    """Persist the document under its stored doc_id + record a fetched_docs row.

    The ``chunks.fetched_doc_id`` FK requires a real ``fetched_docs`` row, so the
    returned row id is what the step passes to ``record_chunk``."""

    meta = DocMetadata(
        doc_id=doc_id,
        source=fd.candidate.source,
        fetched_at=utc_now_iso(),
        query=QueryProvenance(base=base_query, expanded=""),
        title=fd.candidate.title,
        url=fd.candidate.url,
        context_excerpt=make_context_excerpt(fd.cleaned_text),
        ingestion_metadata=IngestionMetadata(parse_method=fd.parse_method),
    )
    persist_document(store, meta, fd.cleaned_text)
    assert sinks.round_ is not None
    return store.record_fetched_doc(
        sinks.round_,
        source=fd.candidate.source,
        query=fd.candidate.matched_query,
        url=fd.candidate.url,
        title=fd.candidate.title,
        doc_id=doc_id,
        outcome="selected_and_fetched",
        parse_method=fd.parse_method,
        cleaned_text_chars=len(fd.cleaned_text),
    )


async def _run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    source_db = Path(args.from_kg)
    if not source_db.exists():
        raise SystemExit(f"--from-kg path does not exist: {source_db}")

    base_query, corpus_plan = _read_corpus_plan(source_db)
    if args.limit is not None:
        corpus_plan = corpus_plan[: args.limit]
    if not corpus_plan:
        raise SystemExit(f"No selected_and_fetched docs found in {source_db}")

    kg_root = Path(args.kg_root) / args.kg_id
    kg_root.mkdir(parents=True, exist_ok=True)
    configure_kg_build_logging(kg_root / "build.log")

    if (kg_root / "kg.db").exists():
        raise SystemExit(
            f"{kg_root / 'kg.db'} already exists — pick a fresh kg_id "
            "(the harness builds from scratch, it does not resume)."
        )

    llm_model = None if settings.sample_mode else build_llm_model(settings)

    request_id = f"rebuild_{args.kg_id}_{int(time.time())}"
    trace = TraceWriter(request_id=request_id, log_dir=kg_root)
    store = TraceStore.open(kg_root)

    sources = sorted({src for _doc_id, src in corpus_plan})
    _log.info(
        "eval_rebuild: id=%s from=%s docs=%d sources=%s query=%r "
        "sample_mode=%s extract_mode=%s final_sweep=%s",
        args.kg_id, source_db, len(corpus_plan), sources, base_query,
        settings.sample_mode, settings.extract_mode, not args.no_final_sweep,
    )

    try:
        build = store.start_build(
            build_run_id=request_id,
            kg_id=args.kg_id,
            base_query=base_query,
            config={
                "harness": "eval_rebuild",
                "from_kg": str(source_db),
                "docs": len(corpus_plan),
            },
            sources=sources,
            sample_mode=bool(settings.sample_mode),
            resume_from_round=None,
        )
        round_handle = store.start_round(
            build,
            round_number=1,
            entities_before=0,
            relations_before=0,
            schema_entity_types_before=0,
            schema_relation_types_before=0,
        )
        sinks = AgentCallSinks(trace=trace, store=store, round_=round_handle)

        fetched_by_doc_id: dict[str, tuple[ChunkedDoc, int]] = {}
        for doc_id, source in corpus_plan:
            text, stored_meta = _read_stored_doc(source_db, doc_id)
            fd = _build_chunked_doc(doc_id, source, text, stored_meta)
            row_id = _persist_and_record(store, sinks, base_query, doc_id, fd)
            fetched_by_doc_id[doc_id] = (fd, row_id)
            _log.info(
                "loaded %s (%s) %d chars → %d chunks",
                doc_id, source, len(text), len(fd.chunks),
            )

        corpus_state = CorpusState(base_query=base_query)
        corpus_state.round_number = 1
        state = RoundGraphState(
            corpus_state=corpus_state,
            kg_gaps_in=[],
            directives_in=[],
            deltas=_RoundDeltas(),
        )
        state.fetched_by_doc_id = fetched_by_doc_id
        deps = RoundGraphDeps(
            base_query=base_query,
            adapters={},
            openalex=None,
            llm_model=llm_model,
            settings=settings,
            sinks=sinks,
            cfg=LoopConfig(),
            kg_root=kg_root,
            force=False,
        )

        await extract_and_consolidate_step(
            StepContext(state=state, deps=deps, inputs=None)
        )

        # ── final full Resolver sweep over the whole entity set ──────────
        # The per-doc loop ran at round_number=1 (incremental resolve, no
        # sweep). One full-snapshot sweep now resolves cross-round identity
        # better than the live build's periodic sweeps and skips its repeat
        # re-judgment waste. round_number=SWEEP_PERIOD forces do_sweep=True.
        # NOTE: facts embedded during the per-doc loop that name an entity
        # merged ONLY by this sweep carry a slightly stale embedding (minor,
        # semantically near).
        if not args.no_final_sweep:
            committed = await resolve(
                store=store,
                round_number=SWEEP_PERIOD,
                entities_touched_this_round=set(),
                settings=settings,
                sinks=sinks,
                llm_model=llm_model,
                triggered_by="harness_final_sweep",
            )
            _log.info("final sweep: %d additional merges", len(committed))

        # ── post-hoc global VocabResolver pass ──────────────────
        # Predicate and property vocabularies. Runs
        # after the entity sweep so it clusters the final canonical vocabulary.
        if not args.no_vocab_cluster:
            for vocab_kind in ("predicate", "property"):
                vocab_totals = await cluster_and_commit(
                    store=store,
                    kind=vocab_kind,
                    settings=settings,
                    sinks=sinks,
                    llm_model=llm_model,
                    round_number=1,
                )
                _log.info("vocab cluster (%s): %s", vocab_kind, vocab_totals)

        entities = store.count_entities()
        relations = store.count_relations()
        schema_e = store.count_schema_types("entity_type")
        schema_r = store.count_schema_types("relation_type")
        store.finish_round(
            round_handle,
            entities_after=entities,
            relations_after=relations,
            schema_entity_types_after=schema_e,
            schema_relation_types_after=schema_r,
            docs_fetched=len(fetched_by_doc_id),
            docs_skipped=0,
            docs_failed=0,
            stop_criterion_fired="harness_single_round",
            duration_seconds=0.0,
        )
        store.finish_build(
            build,
            stop_reason="harness_single_round",
            rounds_completed=1,
            total_docs=len(fetched_by_doc_id),
            total_entities=entities,
            total_relations=relations,
            schema_entity_types=schema_e,
            schema_relation_types=schema_r,
        )
    finally:
        store.close()

    summary = {
        "kg_id": args.kg_id,
        "kg_db": str(kg_root / "kg.db"),
        "from_kg": str(source_db),
        "docs": len(fetched_by_doc_id),
        "kg_entity_count": entities,
        "kg_relation_count": relations,
        "schema_entity_types": schema_e,
        "schema_relation_types": schema_r,
        "sample_mode": bool(settings.sample_mode),
    }
    _log.info("eval_rebuild summary:\n%s", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
