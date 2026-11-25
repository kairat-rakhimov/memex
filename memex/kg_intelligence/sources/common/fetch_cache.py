"""Per-URL byte cache for adapter ``fetch()`` calls.

Stores raw response bytes (PDF / HTML / JSON) keyed by SHA-256 of the
URL so rebuilds skip arxiv / gdelt / wikipedia throttle costs. Raw
bytes only — never parsed text — so parser experiments stay valid
against the same cached bytes. ``root=None`` disables the cache (pure
passthrough, no disk I/O).
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path

_log = logging.getLogger(__name__)


class FetchCache:
    """URL-keyed byte cache with passthrough disabled mode + counters.

    Construct one per process (the build CLI, ``memex build``), pass into
    each adapter's ``__init__``. Adapters wrap their ``self.client.get(...)``
    bytes-returning calls in ``get_or_fetch(url, fetch_fn)``.
    """

    def __init__(self, root: Path | None) -> None:
        self.root = root
        self.hits = 0
        self.misses = 0
        self.bytes_from_cache = 0
        self.bytes_fetched_fresh = 0
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def get_or_fetch(self, url: str, fetch_fn: Callable[[], bytes]) -> bytes:
        """Return cached bytes for ``url`` or call ``fetch_fn`` and cache.

        ``fetch_fn`` must return ``bytes``; any exception it raises
        propagates out unchanged and nothing is written to cache. The
        adapter's existing log-and-skip handling sees the same exception
        it always saw.
        """

        if self.root is None:
            return fetch_fn()

        path = self._path(self._key(url))
        if path.exists():
            data = path.read_bytes()
            self.hits += 1
            self.bytes_from_cache += len(data)
            _log.debug("fetch-cache hit: %s", _shorten(url))
            return data

        data = fetch_fn()
        self._write_atomic(path, data)
        self.misses += 1
        self.bytes_fetched_fresh += len(data)
        _log.debug("fetch-cache miss: %s (%d bytes)", _shorten(url), len(data))
        return data

    @staticmethod
    def _key(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        assert self.root is not None
        return self.root / key[:2] / f"{key}.bin"

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.rename(tmp, path)


def _shorten(url: str, *, head: int = 60) -> str:
    return url if len(url) <= head else url[: head - 1] + "…"
