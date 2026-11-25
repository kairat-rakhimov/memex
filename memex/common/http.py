"""Shared outbound-HTTP timeout policy.

One helper so every client uses a *granular*, task-sized ``httpx.Timeout`` rather
than a bare float (a bare float applies the same generous value to *connect* as to
*read*). A tight **connect** means a dead / unreachable host fails fast — the
property that matters for an unattended overnight build — while **read** stays
sized to the workload so a slow-but-valid response is never dropped.
"""

from __future__ import annotations

import os

import httpx

PROJECT_URL = "https://github.com/kairat-rakhimov/memex"

# A live host connects in well under a second; a dead/unreachable one should not
# stall an unattended build. 10s is generous headroom without being a hang.
DEFAULT_CONNECT_TIMEOUT_S = 10.0


def http_timeout(
    read: float,
    *,
    connect: float = DEFAULT_CONNECT_TIMEOUT_S,
    write: float = 30.0,
    pool: float = 10.0,
) -> httpx.Timeout:
    """Granular per-task timeout: tight ``connect`` + a task-sized ``read``.
    ``read`` is the value callers size to their workload (PDF download, GROBID
    parse, LLM reasoning call, fast JSON API, …); the rest have sane defaults."""
    return httpx.Timeout(connect=connect, read=read, write=write, pool=pool)


def user_agent() -> str:
    """User-Agent for public APIs. Wikipedia's API policy asks clients for a way
    to reach the operator: set ``MEMEX_CONTACT`` to an address or URL."""
    contact = os.environ.get("MEMEX_CONTACT", "").strip()
    return f"memex/0.1 ({PROJECT_URL}{'; ' + contact if contact else ''})"


__all__ = ["DEFAULT_CONNECT_TIMEOUT_S", "http_timeout", "user_agent"]
