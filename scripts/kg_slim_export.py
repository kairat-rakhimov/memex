"""Slim-export a built `kg.db` into a smaller, query-ready copy.

The working `kg/<id>/kg.db` (e.g. 975 MB) is dominated by two things the query
layer never reads: provenance/trace tables (`kg_mutations`,
`agent_call_details.raw_response`, the build/round/agent rows) and full-precision
embeddings (3072-dim float32 = 71% of the file). This produces a slim copy that
keeps ONLY what chat + hypothesis retrieval need, with embeddings truncated to a
lower Matryoshka dimension (`text-embedding-3-large` is trained so the first N
dims + renormalization is a valid lower-dim embedding).

    python scripts/kg_slim_export.py my-topic \\
        [--out my-topic-slim] [--dim 1024] [--kg-root kg] [--with-trace]

Output lands at `kg/<out>/kg.db`, following the same per-kg layout as builds.
The query layer truncates the query vector to the stored dim, so the slim db is
directly queryable by `memex chat` / `memex hypothesize`.

`--with-trace` also copies the provenance/trace tables (builds, rounds,
agent_calls, agent_call_details, kg_mutations, round_reviews, …) so the build-loop
trace UI (`memex ui`) works on the slim db. The only transform is
the embedding truncation, so the result is the full db at the lower dim.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

from memex.observability.store import apply_schema

# Tables the query layer reads — copied verbatim (embeddings handled separately).
_KEEP_TABLES = [
    "kg_entities",
    "kg_relations",
    "kg_entity_properties",
    "kg_schema",
    "documents",
    "chunks",
    "entity_chunk_refs",
    "entity_aliases",
    "predicate_aliases",
    "entity_type_aliases",
    "property_aliases",
    "schema_parents",
]
# Dropped by default (provenance/trace). With --with-trace these are copied too,
# so the build-loop trace UI (memex ui) works on the slim db.
_TRACE_TABLES = [
    "builds",
    "rounds",
    "agent_calls",
    "agent_call_details",
    "fetched_docs",
    "doc_assessments",
    "extractions",
    "schema_changes",
    "kg_mutations",
    "round_reviews",
    "resolver_verdicts",
    "loop_state",
]

_EMBED_BATCH = 4000


def _truncate(blob: bytes, dim: int) -> bytes:
    """First `dim` dims of a (unit) float32 vector, renormalized to unit norm."""

    vec = np.frombuffer(blob, dtype=np.float32)
    if vec.shape[0] > dim:
        vec = vec[:dim].copy()
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec = vec / n
    return vec.astype(np.float32).tobytes()


def slim_export(src_path: Path, out_path: Path, *, dim: int, with_trace: bool = False) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {out_path}")

    conn = sqlite3.connect(out_path)
    apply_schema(conn)  # creates all tables (provenance ones stay empty)
    conn.execute("PRAGMA foreign_keys = OFF")  # provenance FKs are intentionally orphaned
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute(f"ATTACH DATABASE '{src_path}' AS src")

    tables = _KEEP_TABLES + (_TRACE_TABLES if with_trace else [])
    report: dict[str, int] = {}
    conn.execute("BEGIN")
    for table in tables:
        conn.execute(f"INSERT INTO main.{table} SELECT * FROM src.{table}")
        report[table] = conn.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]

    # embeddings: truncate + renormalize, restamp dimension
    total = conn.execute("SELECT COUNT(*) FROM src.embeddings").fetchone()[0]
    cur = conn.execute(
        "SELECT target_kind, target_id, provider, model, vector, created_at FROM src.embeddings"
    )
    written = 0
    while True:
        rows = cur.fetchmany(_EMBED_BATCH)
        if not rows:
            break
        conn.executemany(
            "INSERT INTO main.embeddings (target_kind, target_id, provider, model, "
            "dimension, vector, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (r[0], r[1], r[2], r[3], dim, _truncate(r[4], dim), r[5])
                for r in rows
            ],
        )
        written += len(rows)
    conn.execute("COMMIT")
    report["embeddings"] = written
    report["embeddings_src"] = total

    conn.execute("DETACH src")
    conn.execute("VACUUM")
    conn.close()
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="kg_slim_export", description="Slim a kg.db for sharing.")
    p.add_argument("kg_id", help="Source build (kg/<kg_id>/kg.db).")
    p.add_argument("--out", default=None, help="Output kg_id. Default: <kg_id>_slim.")
    p.add_argument("--dim", type=int, default=1024, help="Truncated embedding dim. Default: 1024.")
    p.add_argument("--kg-root", default="kg")
    p.add_argument(
        "--with-trace",
        action="store_true",
        help="Also copy provenance/trace tables so the trace UI works on the slim db.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.kg_root)
    src = root / args.kg_id / "kg.db"
    if not src.exists():
        print(f"error: no kg.db at {src}", file=sys.stderr)
        return 2
    out_id = args.out or f"{args.kg_id}_slim"
    out = root / out_id / "kg.db"

    src_mb = src.stat().st_size / 1e6
    report = slim_export(src, out, dim=args.dim, with_trace=args.with_trace)
    out_mb = out.stat().st_size / 1e6

    kept = _KEEP_TABLES + (_TRACE_TABLES if args.with_trace else [])
    print(f"slim-export {args.kg_id} → {out_id}  (dim {args.dim}"
          f"{', +trace' if args.with_trace else ''})")
    for k in kept:
        print(f"  {k:24} {report[k]:>8}")
    print(f"  {'embeddings':24} {report['embeddings']:>8}  (truncated → {args.dim}-dim fp32)")
    print(f"\n  size: {src_mb:.0f} MB → {out_mb:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
