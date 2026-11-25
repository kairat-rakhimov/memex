"""``memex ui`` — read-only browser over build traces and query reasoning.

One server over both the build-loop trace and the query-reasoning trace, with a
top-nav to switch between them. Every graph under ``--kg-root`` shows up in
both views.

    memex ui                       # http://127.0.0.1:8001/
    memex ui --port 9001 --kg-root kg

Never writes to the databases; safe to run against an in-progress build.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

from memex.ui import create_app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="memex ui", description="Browse build and query traces.")
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("UI_PORT", "8001")))
    p.add_argument(
        "--kg-root",
        default=os.environ.get("MEMEX_KG_DIR", "kg"),
        help="Directory of per-kg_id subdirectories to scan (default: kg).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    kg_root = Path(args.kg_root).resolve()
    if not kg_root.exists():
        print(f"warning: kg-root {kg_root} does not exist yet", file=sys.stderr)
    app = create_app(kg_root)
    print(f"memex ui → http://{args.host}:{args.port}/  (kg-root: {kg_root})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0
