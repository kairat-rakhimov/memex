"""Shared pacing + retry helpers for source adapters.

``AdapterThrottle`` enforces a per-adapter minimum call interval;
``wait_respectful`` and ``concise_retry_log`` are tenacity callbacks for
respectful backoff and one-line retry logging.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)


class AdapterThrottle:
    """Thread-safe ``min_interval`` enforcer; lock held during sleep so
    concurrent callers queue."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


def wait_respectful(
    *,
    min_429: float = 60.0,
    min_other: float = 5.0,
    max_wait: float = 300.0,
) -> Callable[[Any], float]:
    """Tenacity wait callback that floors waits at ``min_429`` after a 429
    (short retries extend the ban) and ``min_other`` for transient
    5xx / network errors."""

    _exp = wait_random_exponential(multiplier=2, max=max_wait)

    def _compute(retry_state: Any) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        is_429 = (
            isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code == 429
        )
        base = _exp(retry_state)
        return max(base, min_429 if is_429 else min_other)

    return _compute


def concise_retry_log(
    log: logging.Logger,
    label: str,
) -> Callable[[Any], None]:
    """Tenacity ``before_sleep`` callback emitting one short line per retry:
    ``<label>: retry #N in Xs after <cause>``."""

    def _emit(retry_state: Any) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        wait = getattr(retry_state.next_action, "sleep", None)
        wait_str = f"{wait:.1f}s" if wait is not None else "?s"
        if isinstance(exc, httpx.HTTPStatusError):
            cause = f"HTTP {exc.response.status_code}"
        elif exc is not None:
            cause = type(exc).__name__
        else:
            cause = "unknown error"
        log.warning(
            "%s: retry #%d in %s after %s",
            label,
            retry_state.attempt_number,
            wait_str,
            cause,
        )

    return _emit


def run_retrying(
    fn: Callable[..., Any],
    *args: Any,
    max_attempts: int,
    retry_predicate: Callable[[BaseException], bool],
    wait: Callable[[Any], float],
    before_sleep: Callable[[Any], None],
) -> Any:
    """Run ``fn(*args)`` under a tenacity ``Retrying`` whose attempt count is
    decided at CALL time — so the cap is per-instance / per-request, not the
    import-time decorator constant. ``max_attempts=1`` ⇒ a single try (zero
    retries). Re-raises the final exception."""

    return Retrying(
        retry=retry_if_exception(retry_predicate),
        stop=stop_after_attempt(max(1, int(max_attempts))),
        wait=wait,
        before_sleep=before_sleep,
        reraise=True,
    )(fn, *args)


__all__ = ["AdapterThrottle", "wait_respectful", "concise_retry_log", "run_retrying"]
