"""``memex chat`` — grounded Q&A over a built knowledge graph.

Usage:

    memex chat <kg_id> -q "What drives down the levelised cost of hydrogen?" \\
        [--kg-root kg] [--max-revisions 1] [--json]

Runs the single-shot Q&A loop (Retriever → Answerer ↔ GroundingChecker) over
``kg/<kg_id>/kg.db`` (opened read-only) and prints a Markdown answer with an
evidence table + citations. Writes the agent trace to
``kg/<kg_id>/agent_trace_<id>.jsonl`` and the reasoning trace to
``kg/<kg_id>/query_trace.db`` (browse with ``memex ui``).

`SAMPLE_MODE=true` runs the loop against deterministic stubs (offline plumbing
check; retrieval quality is live-only).
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
from memex.kg_query.chat import render_markdown, run_chat


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="memex chat",
        description="Grounded Q&A over a built knowledge graph.",
    )
    parser.add_argument("kg_id", help="Identifier of the built graph (a directory under --kg-root).")
    parser.add_argument("--question", "-q", required=True, help="The question to answer.")
    parser.add_argument("--kg-root", default="kg", help="Parent dir for kg_id. Default: kg/.")
    parser.add_argument(
        "--max-revisions", type=int, default=1,
        help="Max GroundingChecker→Answerer revision cycles. Default: 1.",
    )
    parser.add_argument(
        "--request-limit", type=int, default=8,
        help="Retriever ReAct request cap (the main cost lever). Default: 8.",
    )
    parser.add_argument("--json", action="store_true", help="Also print the result as JSON.")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    kg_root = Path(args.kg_root) / args.kg_id
    if not (kg_root / "kg.db").exists():
        print(f"error: no kg.db at {kg_root}/kg.db", file=sys.stderr)
        return 2

    configure_kg_build_logging(kg_root / "query.log")

    result = await run_chat(
        question=args.question,
        kg_root=kg_root,
        settings=settings,
        max_revisions=args.max_revisions,
        retriever_request_limit=args.request_limit,
    )

    print(render_markdown(result))
    if args.json:
        payload = {
            "question": result.question,
            "answer": result.answer.model_dump(mode="json"),
            "evidence": [
                {
                    "index": i,
                    "kind": f.kind,
                    "fact_id": f.fact_id,
                    "canonical": f.canonical,
                    "raw": f.raw,
                    "corroboration": f.corroboration,
                    "confidence": f.confidence,
                    "source_doc_id": f.source_doc_id,
                    "citation": {
                        "title": f.citation.title,
                        "url": f.citation.url,
                        "source": f.citation.source,
                    } if f.citation else None,
                }
                for i, f in enumerate(result.bundle, start=1)
            ],
            "revised": result.revised,
            "trace_file": result.trace_path,
            "query_trace_db": result.query_trace_db,
            "run_db_id": result.run_db_id,
            "diagnostics": result.diagnostics,
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
