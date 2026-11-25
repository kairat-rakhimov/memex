"""``memex hypothesize`` — hypothesis-generation sweep over a built knowledge graph.

Usage:

    memex hypothesize <kg_id> \\
        [--max-seeds 24] [--top-n 8] [--novelty-max 0.85] \\
        [--lanes bridge,gap,property,attribute_contested,relation_contested] \\
        [--query "<topic override>"] [--out report.md] [--json]

Generates deterministic structural seeds (all lanes), articulates + verifies
hypotheses, ranks by novelty × plausibility × support, and writes a curated
top-N report + appendix over ``kg/<kg_id>/kg.db`` (read-only). Writes the agent
trace + the reasoning trace (``kg/<kg_id>/query_trace.db``).

`SAMPLE_MODE=true` runs against deterministic stubs (offline plumbing).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from memex.common.config import Settings
from memex.common.logging_setup import configure_kg_build_logging
from memex.kg_query.hypothesize import run_hypothesize


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="memex hypothesize",
        description="memex — autonomous hypothesis generation over a built KG.",
    )
    p.add_argument("kg_id")
    p.add_argument("--kg-root", default="kg")
    p.add_argument("--query", default=None, help="Topic override (default: the build's base_query).")
    p.add_argument("--max-seeds", type=int, default=24, help="Seed budget (the LLM fan-out cap). Default: 24.")
    p.add_argument("--top-n", type=int, default=8, help="Hypotheses in the curated section. Default: 8.")
    p.add_argument("--novelty-max", type=float, default=0.85, help="Restatement-similarity floor. Default: 0.85.")
    p.add_argument("--lanes", default=None, help="Comma-separated subset of the 5 seed lanes.")
    p.add_argument("--seed-type", default=None,
                   help="Focus the Explorer on ONE lane (bridge/gap/property/attribute_contested/"
                        "relation_contested). Alias for --lanes with a single lane.")
    p.add_argument("--explore-cap", type=int, default=8,
                   help="How many top-worthiness seeds the Explorer investigates. Default: 8.")
    p.add_argument("--out", default=None, help="Write the report Markdown to this file too.")
    p.add_argument("--json", action="store_true", help="Also print machine-readable JSON.")
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    kg_root = Path(args.kg_root) / args.kg_id
    if not (kg_root / "kg.db").exists():
        print(f"error: no kg.db at {kg_root}/kg.db", file=sys.stderr)
        return 2
    configure_kg_build_logging(kg_root / "query.log")

    lanes = (
        [args.seed_type] if args.seed_type
        else [s.strip() for s in args.lanes.split(",")] if args.lanes
        else None
    )
    result = await run_hypothesize(
        kg_root=kg_root, settings=settings, base_query=args.query,
        max_seeds=args.max_seeds, top_n=args.top_n, novelty_max=args.novelty_max,
        lanes=lanes, explore_cap=args.explore_cap,
    )

    print(result.report_markdown)
    if args.out:
        Path(args.out).write_text(result.report_markdown, encoding="utf-8")
        print(f"\n_(report written to {args.out})_", file=sys.stderr)
    if args.json:
        payload = {
            "base_query": result.base_query,
            "diagnostics": result.diagnostics,
            "trace_file": result.trace_path,
            "query_trace_db": result.query_trace_db,
            "run_db_id": result.run_db_id,
            "hypotheses": [
                {
                    "class": r.candidate.hypothesis_class or r.seed.lane,
                    "statement": r.candidate.statement,
                    "mechanism": r.candidate.mechanism,
                    "scores": {
                        "novelty": r.scores.novelty,
                        "plausibility": r.scores.plausibility,
                        "support": r.scores.support,
                        "composite": r.scores.composite,
                    },
                    "verify_holds": r.verdict.holds,
                    "evidence": [
                        {"ref": f"{f.kind}:{f.fact_id}", "canonical": f.canonical,
                         "source": f.citation.title if f.citation else f.source_doc_id,
                         "url": f.citation.url if f.citation else None}
                        for f in r.surviving_legs
                    ],
                }
                for r in result.ranked
            ],
        }
        print("\n```json")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("```")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
