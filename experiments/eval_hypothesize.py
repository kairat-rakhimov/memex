"""Per-stage harness for the hypothesis layer.

    # free — inspect deterministic seeds (no LLM), the cheapest QC:
    python experiments/eval_hypothesize.py <kg_id> --seeds-only [--max-seeds 30]

    # one Generator batch over the first seeds (live, reasoning tier):
    python experiments/eval_hypothesize.py <kg_id> --agent generator --max-seeds 8

    # full sweep:
    python experiments/eval_hypothesize.py <kg_id> --agent full --max-seeds 12 --top-n 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from memex.common.config import Settings
from memex.common.trace import TraceWriter
from memex.kg_intelligence.agents.base import AgentCallSinks
from memex.kg_query import seeds as seeds_mod
from memex.kg_query.agents import hypothesis_generator
from memex.kg_query.deps import QueryDeps
from memex.kg_query.grounding import ground_refs
from memex.kg_query.hypothesize import run_hypothesize


async def _run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    kg_root = Path(args.kg_root) / args.kg_id
    if not (kg_root / "kg.db").exists():
        print(f"error: no kg.db at {kg_root}/kg.db", file=sys.stderr)
        return 2

    if args.agent == "full":
        result = await run_hypothesize(
            kg_root=kg_root, settings=settings, max_seeds=args.max_seeds, top_n=args.top_n,
        )
        print(result.report_markdown)
        print("\ndiagnostics:", result.diagnostics, file=sys.stderr)
        return 0

    deps = QueryDeps.open(kg_root, settings)
    try:
        seeds = seeds_mod.generate_seeds(deps, max_seeds=args.max_seeds)
        print(f"# {len(seeds)} seeds — {dict(Counter(s.lane for s in seeds))}\n")
        for i, s in enumerate(seeds):
            print(f"[{i}] [{s.lane}] score={s.score:.2f}  {s.title}")
            print(f"    {s.rationale}")
        if args.seeds_only:
            return 0

        # one generator batch over the first seeds
        batch = [(s, ground_refs(deps, [f"{k}:{i}" for k, i in s.leg_fact_ids])) for s in seeds[: args.batch]]
        trace = TraceWriter(request_id=f"eval_hyp_{args.kg_id}", log_dir=Path("/tmp"))
        sinks = AgentCallSinks(trace=trace, store=None, round_=None)
        gen, _ = await hypothesis_generator.generate(
            base_query=args.query or "the corpus topic", seeds_with_legs=batch,
            settings=settings, sinks=sinks,
        )
        print("\n# Generator\n")
        print("reasoning:", gen.reasoning, "\n")
        for c in gen.candidates:
            tag = "" if c.worthwhile else " (REJECTED)"
            print(f"- seed {c.seed_index} [{c.hypothesis_class}] plaus={c.plausibility:.2f}{tag}")
            if c.worthwhile:
                print(f"    statement: {c.statement}")
                print(f"    mechanism: {c.mechanism}")
                print(f"    legs: {c.leg_refs}")
        return 0
    finally:
        deps.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="eval_hypothesize", description="Per-stage hypothesis harness.")
    p.add_argument("kg_id")
    p.add_argument("--kg-root", default="kg")
    p.add_argument("--agent", choices=["generator", "full"], default="generator")
    p.add_argument("--seeds-only", action="store_true", help="Print seeds and stop (no LLM).")
    p.add_argument("--max-seeds", type=int, default=24)
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--batch", type=int, default=8, help="Seeds for the single Generator batch.")
    p.add_argument("--query", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
