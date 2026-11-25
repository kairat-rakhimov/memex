"""OpenAI-compatible embedding client + in-memory similarity helpers.

Vectors are stored as float32 BLOBs in the ``embeddings`` table and
loaded into numpy matrices for cosine similarity. OpenAI's
``text-embedding-3-large`` returns pre-normalised unit vectors, so
raw dot product == cosine similarity.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from openai import AsyncOpenAI

from memex.common.config import Settings
from memex.common.http import http_timeout

if TYPE_CHECKING:
    from memex.observability.store import TraceStore

_log = logging.getLogger(__name__)

BATCH_SIZE = 100


def vec_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


async def embed_texts(
    texts: list[str],
    settings: Settings,
) -> list[np.ndarray]:
    if not texts:
        return []

    if settings.sample_mode:
        dim = settings.embedding_dimension
        return [np.zeros(dim, dtype=np.float32) for _ in texts]

    client = AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=http_timeout(120.0),
        max_retries=2,
    )

    all_vecs: list[np.ndarray] = [None] * len(texts)  # type: ignore[list-item]
    total_calls = 0
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        resp = await client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        total_calls += 1
        for item in resp.data:
            all_vecs[start + item.index] = np.array(item.embedding, dtype=np.float32)

    _log.debug("embed_texts: %d texts, %d calls", len(texts), total_calls)
    return all_vecs


def load_matrix(
    store: TraceStore,
    kind: str,
) -> tuple[list[str], np.ndarray]:
    rows = store.list_embeddings_by_kind(kind)
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    ids = [r[0] for r in rows]
    vecs = [blob_to_vec(r[1]) for r in rows]
    return ids, np.stack(vecs)


def top_k_similar(
    query: np.ndarray,
    matrix: np.ndarray,
    k: int = 5,
) -> list[tuple[int, float]]:
    if matrix.size == 0 or query.size == 0:
        return []
    scores = matrix @ query
    if len(scores) <= k:
        indices = np.argsort(-scores)
    else:
        indices = np.argpartition(-scores, k)[:k]
        indices = indices[np.argsort(-scores[indices])]
    return [(int(i), float(scores[i])) for i in indices if scores[i] > 0.0]


async def top_k_candidates_by_kind(
    store: TraceStore,
    settings: Settings,
    *,
    raw_names: list[str],
    kind: str = "predicate",
    k: int = 10,
    cap: int = 60,
) -> list[str]:
    """Retrieve a candidate slice of canonical names for the Mapper prompt.

    The above-cutoff fallback: when the canonical
    vocabulary is too large to pass in full, embed each raw surface form
    transiently, take its top-``k`` nearest canonicals (no hard similarity
    threshold — the Mapper does the judging), then union + dedup across all raw
    names and cap at ``cap``. The canonical embedding matrix is loaded once.

    Only ``predicate`` has an embedding matrix (entity types are few enough to
    always pass in full); an unknown ``kind`` or an empty matrix returns []."""

    if not raw_names:
        return []
    ids, matrix = load_matrix(store, kind)
    if matrix.size == 0:
        return []
    raw_vecs = await embed_texts(raw_names, settings)

    ordered: list[str] = []
    seen: set[str] = set()
    for vec in raw_vecs:
        for idx, _score in top_k_similar(vec, matrix, k=k):
            name = ids[idx]
            if name not in seen:
                seen.add(name)
                ordered.append(name)
                if len(ordered) >= cap:
                    return ordered
    return ordered


async def embed_and_store(
    store: TraceStore,
    items: list[tuple[str, str, str]],
    settings: Settings,
) -> int:
    """Embed a batch of (kind, target_id, text) and store in the DB.

    Returns the number of new rows written (skips duplicates via
    INSERT OR IGNORE).
    """
    if not items:
        return 0

    existing = set()
    for kind in {k for k, _, _ in items}:
        for target_id, _ in store.list_embeddings_by_kind(kind):
            existing.add((kind, target_id))

    new_items = [(k, tid, txt) for k, tid, txt in items if (k, tid) not in existing]
    if not new_items:
        return 0

    texts = [txt for _, _, txt in new_items]
    vecs = await embed_texts(texts, settings)

    rows = [
        (
            kind, target_id,
            "openai", settings.embedding_model, settings.embedding_dimension,
            vec_to_blob(vec),
        )
        for (kind, target_id, _txt), vec in zip(new_items, vecs, strict=True)
    ]
    return store.record_embeddings_batch(rows)
