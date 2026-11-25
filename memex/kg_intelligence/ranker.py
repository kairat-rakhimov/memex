"""Deterministic per-source ranking of `SourceCandidate` batches.

Each source's search API already returns its native relevance order; this
ranker preserves that order as the dominant signal (70%) and adds a small
per-source nudge (30%) from native side-signals — arXiv abstract overlap
+ recency, Wikipedia wordcount, GDELT recency + tone magnitude. The
result is a re-sorted top-K list with the score stamped under
`raw_signals['rank_score']` for trace visibility.

No LLM, no I/O — deterministic so tests reproduce exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memex.kg_intelligence.sources.common.base import SourceCandidate

DEFAULT_TOP_K = 20
RANK_SCORE_KEY = "rank_score"

_POSITION_WEIGHT = 0.7
_SIGNAL_WEIGHT = 0.3


def rank(
    candidates: list[SourceCandidate],
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[SourceCandidate]:
    """Re-sort `candidates` by `position_score * 0.7 + signal_score * 0.3`, top-K.

    All candidates in a batch must share a source; the source dispatch picks
    the right signal scorer. Returns fresh `SourceCandidate` instances (input
    is not mutated).
    """

    if not candidates:
        return []

    source = candidates[0].source
    scorer = _SCORERS[source]
    query_tokens = _tokens(query)
    n = len(candidates)

    scored: list[SourceCandidate] = []
    for i, c in enumerate(candidates):
        position_score = 1.0 - (i / max(n - 1, 1))
        signal_score = scorer(c, query_tokens)
        score = _POSITION_WEIGHT * position_score + _SIGNAL_WEIGHT * signal_score
        scored.append(
            c.model_copy(
                update={
                    "raw_signals": {**c.raw_signals, RANK_SCORE_KEY: round(score, 4)},
                }
            )
        )

    scored.sort(key=lambda c: c.raw_signals[RANK_SCORE_KEY], reverse=True)
    return scored[:top_k]


# ── helpers ────────────────────────────────────────────────────────────────


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in (text or "").split() if tok}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _parse_date(date_str: str) -> datetime | None:
    s = (date_str or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        return datetime.strptime(s[:8], "%Y%m%d")
    except ValueError:
        return None


def _recency_score(date_str: str, *, half_life_years: float = 5.0) -> float:
    """Exponential decay: now → 1.0, `half_life_years` ago → 0.5, missing → 0.5."""

    dt = _parse_date(date_str)
    if dt is None:
        return 0.5
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    years = (now - dt).days / 365.25
    if years < 0:
        return 1.0
    return math.exp(-years * math.log(2) / half_life_years)


# ── per-source signal scorers ──────────────────────────────────────────────


def _arxiv_signal(c: SourceCandidate, query_tokens: set[str]) -> float:
    overlap = _jaccard(query_tokens, _tokens(c.abstract))
    recency = _recency_score(c.raw_signals.get("published", ""))
    return 0.7 * overlap + 0.3 * recency


def _wikipedia_signal(c: SourceCandidate, _query_tokens: set[str]) -> float:
    wordcount = c.raw_signals.get("wordcount") or 0
    # log10(101)/5 ≈ 0.40, log10(1001)/5 ≈ 0.60, log10(10001)/5 ≈ 0.80
    return min(math.log10(wordcount + 1) / 5.0, 1.0)


def _gdelt_signal(c: SourceCandidate, _query_tokens: set[str]) -> float:
    try:
        tone = abs(float(c.raw_signals.get("tone") or 0.0))
    except (TypeError, ValueError):
        tone = 0.0
    tone_score = min(tone / 10.0, 1.0)
    recency = _recency_score(c.raw_signals.get("seendate", ""))
    return 0.5 * recency + 0.5 * tone_score


_SCORERS: dict[str, Callable[[SourceCandidate, set[str]], float]] = {
    "arxiv": _arxiv_signal,
    "wikipedia": _wikipedia_signal,
    "gdelt": _gdelt_signal,
}
