"""Shared utilities for source plugins: protocol + types, chunker,
fetch cache, pacing/retry helpers. Nothing here imports from a specific
source folder."""

from memex.kg_intelligence.sources.common.base import (
    AdapterDescription,
    ChunkedDoc,
    FetchedDoc,
    SourceAdapter,
    SourceCandidate,
)
from memex.kg_intelligence.sources.common.chunker import Chunk, chunk_text
from memex.kg_intelligence.sources.common.fetch_cache import FetchCache
from memex.kg_intelligence.sources.common.throttle import (
    AdapterThrottle,
    concise_retry_log,
    wait_respectful,
)

__all__ = [
    "AdapterDescription",
    "AdapterThrottle",
    "Chunk",
    "ChunkedDoc",
    "FetchCache",
    "FetchedDoc",
    "SourceAdapter",
    "SourceCandidate",
    "chunk_text",
    "concise_retry_log",
    "wait_respectful",
]
