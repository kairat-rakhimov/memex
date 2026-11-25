"""Per-agent harness for the Q&A layer — prompt tuning without full runs.

Builds a DETERMINISTIC evidence bundle from a seed entity (symbolic only, no
embedding call) and runs one agent in isolation so you can iterate on a prompt
cheaply. The Retriever mode is the one
live tool-using path.

    python experiments/eval_chat.py <kg_id> \\
        --question "..." [--seed-entity "electrolyser"] \\
        [--agent answerer|checker|retriever|full]

`SAMPLE_MODE=true` = offline plumbing (stub agents).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from memex.common.config import Settings
from memex.common.trace import TraceWriter
from memex.kg_intelligence.agents.base import AgentCallSinks
from memex.kg_query import grounding, retrieval
from memex.kg_query.agents import answerer, grounding_checker, retriever
from memex.kg_query.chat import render_markdown, run_chat
from memex.kg_query.deps import QueryDeps


def _deterministic_bundle(deps: QueryDeps, seed: str, limit: int) -> list:
    ents = retrieval.find_entities(deps, seed)
    if not ents:
        return []
    facts = retrieval.entity_facts(deps, ents[0].entity_id, limit=limit)
    return grounding.ground_facts(deps, [(f.kind, f.fact_id) for f in facts])


async def _run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    kg_root = Path(args.kg_root) / args.kg_id
    if not (kg_root / "kg.db").exists():
        print(f"error: no kg.db at {kg_root}/kg.db", file=sys.stderr)
        return 2

    if args.agent == "full":
        result = await run_chat(question=args.question, kg_root=kg_root, settings=settings)
        print(render_markdown(result))
        return 0

    deps = QueryDeps.open(kg_root, settings)
    trace = TraceWriter(request_id=f"eval_chat_{args.kg_id}", log_dir=Path("/tmp"))
    sinks = AgentCallSinks(trace=trace, store=None, round_=None)
    try:
        if args.agent == "retriever":
            res, _ = await retriever.retrieve(
                question=args.question, deps=deps, settings=settings, sinks=sinks,
            )
            print("# Retriever\n")
            print("reasoning:", res.reasoning)
            print("notes:", res.notes)
            print("selected refs:", res.selected)
            bundle = grounding.ground_refs(deps, res.selected)
            print("\n## Grounded bundle\n")
            print(grounding.format_bundle(bundle))
            return 0

        bundle = _deterministic_bundle(deps, args.seed_entity, args.bundle_size)
        print(f"# Deterministic bundle (seed={args.seed_entity!r}, {len(bundle)} facts)\n")
        print(grounding.format_bundle(bundle, with_snippet=False))

        ans, _ = await answerer.answer(
            question=args.question, facts=bundle, settings=settings, sinks=sinks,
        )
        print("\n# Answerer\n")
        print(ans.answer)
        print("\nclaims:")
        for i, c in enumerate(ans.claims, 1):
            print(f"  {i}. {c.text}  [cites {c.support}]")
        print("gaps:", ans.gaps)

        if args.agent == "checker":
            verdict, _ = await grounding_checker.check(
                question=args.question, answer=ans, facts=bundle, settings=settings, sinks=sinks,
            )
            print("\n# GroundingChecker\n")
            print("reasoning:", verdict.reasoning)
            for v in verdict.verdicts:
                print(f"  claim {v.claim_index}: {'OK' if v.supported else 'UNSUPPORTED'} — {v.reasoning}")
            print("revision_request:", verdict.revision_request)
        return 0
    finally:
        deps.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="eval_chat", description="Per-agent Q&A harness.")
    p.add_argument("kg_id")
    p.add_argument("--question", "-q", required=True)
    p.add_argument("--kg-root", default="kg")
    p.add_argument("--agent", choices=["answerer", "checker", "retriever", "full"], default="full")
    p.add_argument("--seed-entity", default="electrolyser", help="Anchor for the deterministic bundle.")
    p.add_argument("--bundle-size", type=int, default=12)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
