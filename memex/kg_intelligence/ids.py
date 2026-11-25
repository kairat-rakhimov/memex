"""Stable content-hash IDs for source documents in the kg-loop pipeline.

`doc_id` derives from a per-source canonical key (arXiv id, Wikipedia page
title, normalized GDELT URL) so the same logical document hashes to the same
id across re-runs, enabling idempotent skip in `persistence.has_document`.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Literal

Source = Literal["arxiv", "wikipedia", "gdelt"]

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text or "").strip().lower()
    return _WHITESPACE.sub(" ", folded)


def _hash16(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def doc_id(source: Source, canonical: str) -> str:
    """`doc_<16-hex>` derived from `<source>:<normalized canonical>`.

    The source prefix prevents collisions across sources for canonicals that
    might overlap (e.g., a Wikipedia title that looks like an arXiv id).
    """
    return "doc_" + _hash16(f"{source}:{_normalize(canonical)}")
