"""Fixed-corpus KG-build harness for deterministic A/B testing.

Builds a fresh ``kg/<kg_id>/kg.db`` from a PINNED set of 6 documents whose
text is pulled from prior builds' ``documents`` tables. Bypasses the
planner / adapters / fetch / gate — runs ONLY the real per-doc knowledge
loop (``extract_and_consolidate_step``) over the fixed corpus, so the same
inputs flow through extract→map→merge→embed→resolve on every run. temp-0
models + fixed docs make it near-deterministic, which is what makes a
baseline-vs-new A/B comparison meaningful.

Usage:

    python experiments/eval_corpus_build.py <kg_id> \\
        [--source-dbs kg/<build-a>/kg.db,kg/<build-b>/kg.db]

SAMPLE_MODE (env ``SAMPLE_MODE=true``) runs against deterministic stubs so
the plumbing can be validated offline with no LLM call.
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
from memex.kg_intelligence.agents import AgentCallSinks
from memex.kg_intelligence.agents.base import build_llm_model
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

# doc_id → source. Text is read from whichever --source-db holds the doc_id.
FIXED_CORPUS: dict[str, str] = {
    "doc_7ee6b50341019980": "arxiv",
    "doc_f04f9e35a8c0c28d": "arxiv",
    "doc_d52610b6cae85d7a": "wikipedia",
    "doc_9d2aee5ef9c7ccda": "wikipedia",
    "doc_6e6a7c312c74ca77": "gdelt",
    "doc_22e0a54c86a77cfe": "gdelt",
}

DEFAULT_SOURCE_DBS = "kg/smoke_mapper_v2/kg.db,kg/smoke_rrsplit_v1/kg.db"
BASE_QUERY = "fixed-corpus eval harness"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval_corpus_build",
        description=(
            "Build a fresh kg.db from a fixed 6-doc corpus through the real "
            "per-doc knowledge loop. For deterministic A/B testing."
        ),
    )
    parser.add_argument("kg_id", help="Identifier for this build (output → kg/<kg_id>/).")
    parser.add_argument(
        "--source-dbs",
        default=DEFAULT_SOURCE_DBS,
        help=(
            "Comma-separated prior-build kg.db paths to pull doc text from. "
            "Each fixed doc_id is read from the first listed db that holds it. "
            f"Default: {DEFAULT_SOURCE_DBS}"
        ),
    )
    parser.add_argument(
        "--kg-root",
        default="kg",
        help="Parent directory for kg_id outputs. Default: kg/.",
    )
    parser.add_argument(
        "--doc-ids",
        default=None,
        help="Comma-separated subset of the fixed corpus to build (default: all 6).",
    )
    return parser.parse_args(argv)


def _read_stored_doc(source_dbs: list[Path], doc_id: str) -> tuple[str, DocMetadata]:
    """Find ``doc_id`` in the first source db that holds it; return its stored
    text + parsed DocMetadata."""

    for db in source_dbs:
        if not db.exists():
            continue
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT text, metadata FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is not None:
            return row[0], DocMetadata.model_validate_json(row[1])
    raise SystemExit(
        f"doc_id {doc_id!r} not found in any of: {[str(p) for p in source_dbs]}"
    )


def _build_chunked_doc(
    doc_id: str, source: str, text: str, stored_meta: DocMetadata
) -> ChunkedDoc:
    """Reconstruct a ChunkedDoc from stored text + metadata.

    ``canonical`` is set to ``doc_id`` (not the original page title / arXiv id):
    the step keys off ``fetched_by_doc_id``'s dict key, never recomputes
    ``doc_id(source, canonical)``, so the canonical only needs to be internally
    consistent — and the harness persists the document directly under the
    known stored ``doc_id`` regardless."""

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
    doc_id: str,
    fd: ChunkedDoc,
) -> int:
    """Persist the document under its stored doc_id + record a fetched_docs row.

    Mirrors ``helpers._persist_one`` / ``_record_fetch_outcome``: the
    ``chunks.fetched_doc_id`` FK requires a real ``fetched_docs`` row, so the
    returned row id is what the step passes to ``record_chunk``."""

    meta = DocMetadata(
        doc_id=doc_id,
        source=fd.candidate.source,
        fetched_at=utc_now_iso(),
        query=QueryProvenance(base=BASE_QUERY, expanded=""),
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
    corpus = FIXED_CORPUS
    if args.doc_ids:
        wanted = {d.strip() for d in args.doc_ids.split(",") if d.strip()}
        corpus = {k: v for k, v in FIXED_CORPUS.items() if k in wanted}
    source_dbs = [Path(p.strip()) for p in args.source_dbs.split(",") if p.strip()]
    kg_root = Path(args.kg_root) / args.kg_id
    kg_root.mkdir(parents=True, exist_ok=True)
    configure_kg_build_logging(kg_root / "build.log")

    if (kg_root / "kg.db").exists():
        raise SystemExit(
            f"{kg_root / 'kg.db'} already exists — pick a fresh kg_id "
            "(the harness builds from scratch, it does not resume)."
        )

    llm_model = None if settings.sample_mode else build_llm_model(settings)

    request_id = f"eval_corpus_{args.kg_id}_{int(time.time())}"
    trace = TraceWriter(request_id=request_id, log_dir=kg_root)
    store = TraceStore.open(kg_root)

    _log.info(
        "eval_corpus_build: id=%s docs=%d source_dbs=%s sample_mode=%s extract_mode=%s",
        args.kg_id, len(corpus), [str(p) for p in source_dbs],
        settings.sample_mode, settings.extract_mode,
    )

    try:
        build = store.start_build(
            build_run_id=request_id,
            kg_id=args.kg_id,
            base_query=BASE_QUERY,
            config={"harness": "eval_corpus_build", "docs": len(corpus)},
            sources=sorted(set(corpus.values())),
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
        for doc_id, source in corpus.items():
            text, stored_meta = _read_stored_doc(source_dbs, doc_id)
            fd = _build_chunked_doc(doc_id, source, text, stored_meta)
            row_id = _persist_and_record(store, sinks, doc_id, fd)
            fetched_by_doc_id[doc_id] = (fd, row_id)
            _log.info(
                "loaded %s (%s) %d chars → %d chunks",
                doc_id, source, len(text), len(fd.chunks),
            )

        corpus_state = CorpusState(base_query=BASE_QUERY)
        corpus_state.round_number = 1
        state = RoundGraphState(
            corpus_state=corpus_state,
            kg_gaps_in=[],
            directives_in=[],
            deltas=_RoundDeltas(),
        )
        state.fetched_by_doc_id = fetched_by_doc_id
        deps = RoundGraphDeps(
            base_query=BASE_QUERY,
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
        "docs": len(fetched_by_doc_id),
        "kg_entity_count": entities,
        "kg_relation_count": relations,
        "schema_entity_types": schema_e,
        "schema_relation_types": schema_r,
        "sample_mode": bool(settings.sample_mode),
    }
    _log.info("eval_corpus_build summary:\n%s", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
