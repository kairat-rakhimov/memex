"""``memex serve`` — run the HTTP API (``POST /run``, ``GET /health``)."""

from __future__ import annotations

import argparse

import uvicorn

from memex.common.config import Settings


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    p = argparse.ArgumentParser(prog="memex serve", description="Run the memex HTTP API.")
    p.add_argument("--host", default=settings.host)
    p.add_argument("--port", type=int, default=settings.port)
    args = p.parse_args(argv)
    uvicorn.run(
        "memex.api:app",
        host=args.host,
        port=args.port,
        log_level=settings.log_level.lower(),
    )
    return 0
