"""Per-agent A/B harness for the VocabResolver.

Reads an EXISTING ``kg.db``'s vocabulary (canonical predicate names + their
embeddings + frequencies) READ-ONLY, builds candidate clusters, runs the LLM
judge on a small sample of clusters, and reports the proposed alias / parent
edges. No commit, no writes — the source db is never
touched. The VocabResolver is a global pass, so its small-scope test is "judge a
few real clusters", not "1–2 docs".

Usage::

    python experiments/eval_vocab_resolver.py \\
        [--from-kg kg/green-hydrogen/kg.db] [--kind predicate] \\
        [--group-limit 12] [--cosine 0.60] [--top-k 8] [--max-group 8]

``SAMPLE_MODE=true`` exercises the plumbing offline (stub marks every member an
alias of its head).
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import build_llm_model
from memex.kg_intelligence.agents.vocab_resolver import (
    COSINE_THRESHOLD,
    MAX_GROUP,
    TOP_K,
    candidate_groups,
    judge_group,
)
from memex.kg_intelligence.embeddings import blob_to_vec

_FREQ_SQL = {
    "predicate": "SELECT predicate, COUNT(*) FROM kg_relations GROUP BY predicate",
    "property": "SELECT property, COUNT(*) FROM kg_entity_properties GROUP BY property",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="eval_vocab_resolver")
    ap.add_argument("--from-kg", default="kg/green-hydrogen/kg.db")
    ap.add_argument("--kind", choices=["predicate", "property"], default="predicate")
    ap.add_argument("--group-limit", type=int, default=12,
                    help="Judge at most N clusters (default 12). 0 = all.")
    ap.add_argument("--cosine", type=float, default=COSINE_THRESHOLD)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--max-group", type=int, default=MAX_GROUP)
    return ap.parse_args(argv)


def _load_vocab(db: Path, kind: str) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Read (names, L2-normalised matrix, freqs) for ``kind`` read-only."""

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        emb_rows = con.execute(
            "SELECT target_id, vector FROM embeddings WHERE target_kind = ?",
            (kind,),
        ).fetchall()
        freq_rows = con.execute(_FREQ_SQL[kind]).fetchall()
    finally:
        con.close()
    freqs = {r[0]: r[1] for r in freq_rows}
    if not emb_rows:
        return [], np.empty((0, 0), dtype=np.float32), freqs
    names = [r[0] for r in emb_rows]
    matrix = np.stack([blob_to_vec(r[1]) for r in emb_rows])
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return names, matrix, freqs


async def _run(args: argparse.Namespace) -> int:
    db = Path(args.from_kg)
    if not db.exists():
        raise SystemExit(f"--from-kg not found: {db}")
    settings = Settings.from_env()

    names, matrix, freqs = _load_vocab(db, args.kind)
    if not names:
        raise SystemExit(
            f"No '{args.kind}' embeddings in {db}. "
            f"(properties have only per-fact 'attribute' embeddings in older builds — "
            f"the property lane needs name-embeddings added first.)"
        )

    groups = candidate_groups(
        names, matrix, freqs,
        cosine_threshold=args.cosine, top_k=args.top_k, max_group=args.max_group,
    )
    clustered = sum(1 + len(m) for _h, m in groups)
    print(
        f"vocab={len(names)} {args.kind}s | {len(groups)} candidate clusters "
        f"covering {clustered} terms (cosine≥{args.cosine}, top_k={args.top_k}, "
        f"max_group={args.max_group}) | sample_mode={settings.sample_mode}"
    )

    judged = groups if args.group_limit == 0 else groups[: args.group_limit]
    if not settings.sample_mode and not settings.api_key:
        raise SystemExit("Need OPENAI_API_KEY (or SAMPLE_MODE=true).")
    llm_model = (
        None if settings.sample_mode else build_llm_model(settings, "reasoning")
    )

    n_alias = n_child = n_unrelated = 0
    for head, members in judged:
        result = await judge_group(
            head, members, freqs, settings=settings, llm_model=llm_model,
        )
        print(f"\n── cluster: {head}  (freq {freqs.get(head, 0)}) ──")
        by_name = {v.name: v for v in result.verdicts}
        for member, score in members:
            v = by_name.get(member)
            if v is None:
                print(f"    {member:40s} sim {score:.2f}  → (no verdict)")
                continue
            tag = {"alias": "ALIAS →", "child_of": "child_of", "unrelated": "—"}[v.relation]
            print(
                f"    {member:40s} sim {score:.2f}  {tag} {v.canonical}"
                f"  ({v.confidence:.2f})  {v.reason}"
            )
            n_alias += v.relation == "alias"
            n_child += v.relation == "child_of"
            n_unrelated += v.relation == "unrelated"

    print(
        f"\n===== SUMMARY ({len(judged)} clusters judged) =====\n"
        f"  alias edges (collapse → vocab shrink): {n_alias}\n"
        f"  child_of edges (parent / rollup):      {n_child}\n"
        f"  unrelated (left standalone):           {n_unrelated}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
