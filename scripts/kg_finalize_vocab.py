"""Standalone VocabResolver finalization on a finished kg.db.

`memex build` runs the vocab-cluster phase natively at the end of a clean loop —
but with two gaps a stopped/resumed build hits:
  1. A build that's stopped + resumed runs it once per stop, committing via
     INSERT OR IGNORE — so the FIRST stop's verdicts (judged on a PARTIAL vocab)
     are frozen and a later run can only ADD non-conflicting edges, never revise.
  2. A final run that's Ctrl-C'd (sigint) skips it entirely → flat vocab.

This tool runs the SAME `cluster_and_commit` over the COMPLETE final vocabulary,
with `--reset` to clear the prior (partial-vocab) clusters first so the whole
vocab is judged in ONE clean pass — as intended
(post-hoc, once, on the full graph). Non-destructive: the cluster tables
are metadata over the base graph; clearing + rebuilding them never touches
entities / relations / attributes / embeddings.

    python scripts/kg_finalize_vocab.py <kg_id> \
        [--reset] [--kind predicate,property]

Run it only AFTER the build has fully stopped (the db must not be open for
writes by a live `memex build`).
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from memex.common.config import Settings
from memex.common.trace import TraceWriter
from memex.kg_intelligence.agents import AgentCallSinks
from memex.kg_intelligence.agents.base import build_llm_model
from memex.kg_intelligence.agents.vocab_resolver import cluster_and_commit
from memex.observability import TraceStore

_KINDS = ("predicate", "property")
# vocab-resolver kind → (alias table, schema_parents.kind value)
_RESET_MAP = {
    "predicate": ("predicate_aliases", "relation_type"),
    "property": ("property_aliases", "property_type"),
}


def _cluster_counts(store: TraceStore) -> dict[str, int]:
    q = lambda s: store._conn.execute(s).fetchone()[0]  # noqa: E731
    return {
        "predicate_aliases": q("SELECT COUNT(*) FROM predicate_aliases"),
        "property_aliases": q("SELECT COUNT(*) FROM property_aliases"),
        "schema_parents": q("SELECT COUNT(*) FROM schema_parents"),
    }


def _reset(store: TraceStore, kinds: list[str]) -> None:
    for kind in kinds:
        alias_table, parent_kind = _RESET_MAP[kind]
        store._conn.execute(f"DELETE FROM {alias_table}")
        store._conn.execute("DELETE FROM schema_parents WHERE kind = ?", (parent_kind,))
    store._conn.commit()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kg_id")
    ap.add_argument("--kind", default="predicate,property",
                    help="comma-list of predicate|property (default both)")
    ap.add_argument("--reset", action="store_true",
                    help="clear the prior cluster tables for these kinds first "
                         "(re-cluster the WHOLE vocab in one clean pass)")
    ap.add_argument("--kg-root", default="kg")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    settings = Settings.from_env()
    if settings.sample_mode or not settings.api_key:
        raise SystemExit("Need OPENAI_API_KEY set and SAMPLE_MODE off (VocabResolver is a live LLM agent).")
    kinds = [k.strip() for k in args.kind.split(",") if k.strip()]
    for k in kinds:
        if k not in _KINDS:
            raise SystemExit(f"unknown kind {k!r}; choose from {list(_KINDS)}")

    kg_root = Path(args.kg_root) / args.kg_id
    if not (kg_root / "kg.db").exists():
        raise SystemExit(f"no kg.db at {kg_root}")

    llm_model = build_llm_model(settings)
    request_id = f"kg_finalize_vocab_{args.kg_id}_{int(time.time())}"
    trace = TraceWriter(request_id=request_id, log_dir=kg_root)
    store = TraceStore.open(kg_root)

    before = _cluster_counts(store)
    print(f"clusters before: {before}")

    # A finalize round to host the agent calls (round_number after the last build round).
    last_round = store._conn.execute("SELECT COALESCE(MAX(round_number), 0) FROM rounds").fetchone()[0]
    finalize_round = last_round + 1
    n_e, n_r = store.count_entities(), store.count_relations()
    n_se = store.count_schema_types("entity_type")
    n_sr = store.count_schema_types("relation_type")

    build = store.start_build(
        build_run_id=request_id, kg_id=args.kg_id, base_query="(vocab finalize)",
        config={"tool": "kg_finalize_vocab", "reset": args.reset, "kinds": kinds},
        sources=[], sample_mode=False, resume_from_round=None,
    )
    round_handle = store.start_round(
        build, round_number=finalize_round,
        entities_before=n_e, relations_before=n_r,
        schema_entity_types_before=n_se, schema_relation_types_before=n_sr,
    )
    sinks = AgentCallSinks(trace=trace, store=store, round_=round_handle)

    if args.reset:
        _reset(store, kinds)
        print(f"reset: cleared cluster tables for {kinds} → {_cluster_counts(store)}")

    started = time.monotonic()
    try:
        for kind in kinds:
            totals = await cluster_and_commit(
                store=store, kind=kind, settings=settings, sinks=sinks,
                llm_model=llm_model, round_number=finalize_round,
            )
            print(f"vocab finalization ({kind}): {totals}")
    finally:
        store.finish_round(
            round_handle,
            entities_after=n_e, relations_after=n_r,
            schema_entity_types_after=n_se, schema_relation_types_after=n_sr,
            docs_fetched=0, docs_skipped=0, docs_failed=0,
            stop_criterion_fired="vocab_finalize",
            duration_seconds=round(time.monotonic() - started, 3),
        )
        store.finish_build(
            build, stop_reason="vocab_finalize", rounds_completed=finalize_round,
            total_docs=0, total_entities=n_e, total_relations=n_r,
            schema_entity_types=n_se, schema_relation_types=n_sr,
        )

    print(f"clusters after:  {_cluster_counts(store)}")
    store.close()


if __name__ == "__main__":
    asyncio.run(main())
