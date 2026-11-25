"""``memex fetch-demo`` — download the prebuilt demo graph.

The demo graph is a full build on "green hydrogen production" (Wikipedia +
openly licensed arXiv papers), published as a Hugging Face dataset. It lets you
try ``memex chat`` / ``hypothesize`` / ``ui`` without paying for a build.

    memex fetch-demo                 # → kg/green-hydrogen/kg.db
    memex fetch-demo --kg-id demo --force
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import httpx

DEMO_URL = os.environ.get(
    "MEMEX_DEMO_URL",
    "https://huggingface.co/datasets/krakhimov/memex-green-hydrogen/resolve/main/kg.db",
)
DEMO_KG_ID = "green-hydrogen"
_CHUNK = 1 << 20


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="memex fetch-demo", description=__doc__.splitlines()[0])
    p.add_argument("--kg-id", default=DEMO_KG_ID, help=f"Target kg_id (default: {DEMO_KG_ID}).")
    p.add_argument("--kg-root", default=os.environ.get("MEMEX_KG_DIR", "kg"))
    p.add_argument("--url", default=DEMO_URL, help="Override the download URL.")
    p.add_argument("--force", action="store_true", help="Re-download even if present.")
    return p.parse_args(argv)


def download(url: str, dest: Path) -> str:
    """Stream ``url`` to ``dest`` atomically; return the file's sha256."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".download")
    digest = hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        with tmp.open("wb") as out:
            for block in resp.iter_bytes(_CHUNK):
                out.write(block)
                digest.update(block)
                done += len(block)
                if total:
                    print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    tmp.replace(dest)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dest = Path(args.kg_root) / args.kg_id / "kg.db"
    if dest.exists() and not args.force:
        print(f"{dest} already exists (use --force to re-download).")
        return 0
    print(f"Downloading the demo graph → {dest}")
    try:
        sha = download(args.url, dest)
    except httpx.HTTPError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    print(f"Done (sha256 {sha[:12]}…). Try:\n  memex chat {args.kg_id} -q \"…\"")
    return 0
