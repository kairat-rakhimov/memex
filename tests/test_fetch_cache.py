"""Tests for the FetchCache helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.kg_intelligence.sources.common.fetch_cache import FetchCache


def test_disabled_passes_through_every_call(tmp_path: Path) -> None:
    cache = FetchCache(root=None)
    calls = 0

    def fetch() -> bytes:
        nonlocal calls
        calls += 1
        return b"payload"

    assert cache.get_or_fetch("https://x/", fetch) == b"payload"
    assert cache.get_or_fetch("https://x/", fetch) == b"payload"
    assert calls == 2
    assert cache.hits == 0
    assert cache.misses == 0
    assert not (tmp_path / "cache").exists()


def test_miss_then_hit_round_trips_bytes(tmp_path: Path) -> None:
    cache = FetchCache(root=tmp_path / "cache")
    calls = 0

    def fetch() -> bytes:
        nonlocal calls
        calls += 1
        return b"hello world"

    first = cache.get_or_fetch("https://x/a", fetch)
    second = cache.get_or_fetch("https://x/a", fetch)

    assert first == b"hello world"
    assert second == b"hello world"
    assert calls == 1
    assert cache.hits == 1
    assert cache.misses == 1
    assert cache.bytes_from_cache == len(b"hello world")
    assert cache.bytes_fetched_fresh == len(b"hello world")


def test_exception_in_fetch_fn_propagates_and_writes_nothing(tmp_path: Path) -> None:
    cache = FetchCache(root=tmp_path / "cache")

    def boom() -> bytes:
        raise RuntimeError("upstream 503")

    with pytest.raises(RuntimeError, match="upstream 503"):
        cache.get_or_fetch("https://x/b", boom)

    # No .bin files should exist anywhere under the cache root.
    assert not list((tmp_path / "cache").rglob("*.bin"))
    assert cache.hits == 0
    assert cache.misses == 0


def test_eager_mkdir_creates_root(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "cache"
    FetchCache(root=root)
    assert root.is_dir()


def test_different_urls_distinct_cache_entries(tmp_path: Path) -> None:
    cache = FetchCache(root=tmp_path / "cache")
    cache.get_or_fetch("https://x/a", lambda: b"first")
    cache.get_or_fetch("https://x/b", lambda: b"second")

    assert cache.get_or_fetch("https://x/a", lambda: b"should-not-run") == b"first"
    assert cache.get_or_fetch("https://x/b", lambda: b"should-not-run") == b"second"
    assert cache.misses == 2
    assert cache.hits == 2


def test_no_tmp_files_left_after_successful_write(tmp_path: Path) -> None:
    cache = FetchCache(root=tmp_path / "cache")
    cache.get_or_fetch("https://x/a", lambda: b"data")
    assert not list((tmp_path / "cache").rglob("*.tmp"))
