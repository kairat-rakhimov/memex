"""Tests for the ranker (per-source scoring).

The ranker is a pure deterministic function, so tests pin exact ordering
instead of fuzzy properties. (Per-round selection moved to the LLM Selector
in ``agents/selector.py`` — see ``tests/test_selector.py``.)
"""

from __future__ import annotations

from datetime import UTC, datetime

from memex.kg_intelligence.ids import Source
from memex.kg_intelligence.ranker import RANK_SCORE_KEY, rank
from memex.kg_intelligence.sources.common.base import SourceCandidate


def _cand(
    source: Source,
    canonical: str,
    *,
    title: str = "",
    abstract: str = "",
    url: str | None = None,
    **signals: object,
) -> SourceCandidate:
    return SourceCandidate(
        source=source,
        canonical=canonical,
        url=url or f"https://example/{source}/{canonical}",
        title=title or canonical,
        abstract=abstract,
        raw_signals=dict(signals),
    )


# ── ranker ────────────────────────────────────────────────────────────────


def test_rank_returns_empty_on_empty_input() -> None:
    assert rank([], "anything") == []


def test_rank_preserves_native_order_when_signals_uniform() -> None:
    cs = [
        _cand("arxiv", "a", abstract="x"),
        _cand("arxiv", "b", abstract="x"),
        _cand("arxiv", "c", abstract="x"),
    ]
    ranked = rank(cs, "x")
    assert [c.canonical for c in ranked] == ["a", "b", "c"]
    assert all(RANK_SCORE_KEY in c.raw_signals for c in ranked)
    # scores strictly decreasing — position dominates when signals match
    scores = [c.raw_signals[RANK_SCORE_KEY] for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_truncates_to_top_k() -> None:
    cs = [_cand("wikipedia", str(i), wordcount=100) for i in range(10)]
    ranked = rank(cs, "q", top_k=3)
    assert len(ranked) == 3
    assert [c.canonical for c in ranked] == ["0", "1", "2"]


def test_rank_does_not_mutate_input() -> None:
    cs = [_cand("arxiv", "a", abstract="hi")]
    rank(cs, "hi")
    assert RANK_SCORE_KEY not in cs[0].raw_signals


def _solo_score(c: SourceCandidate, query: str = "q") -> float:
    """Helper: rank a one-element batch to isolate signal contribution."""
    return rank([c], query)[0].raw_signals[RANK_SCORE_KEY]


def test_rank_arxiv_score_grows_with_abstract_query_overlap() -> None:
    no_overlap = _cand("arxiv", "a", abstract="cosmology dark matter")
    full_overlap = _cand("arxiv", "b", abstract="iridium catalysis water splitting")
    assert _solo_score(full_overlap, "iridium catalysis water splitting") > _solo_score(
        no_overlap, "iridium catalysis water splitting"
    )


def test_rank_arxiv_signal_can_overtake_adjacent_position_in_realistic_batch() -> None:
    # Five candidates; the bottom one has perfect query overlap. With 70/30
    # position/signal weighting it should overtake its neighbour but not
    # claim the top — signals nudge, native order still dominates.
    cs = [_cand("arxiv", f"weak{i}", abstract="unrelated text") for i in range(4)] + [
        _cand("arxiv", "strong", abstract="iridium oxygen evolution reaction")
    ]
    ranked = rank(cs, "iridium oxygen evolution reaction")
    order = [c.canonical for c in ranked]
    assert order.index("strong") < order.index("weak3")
    assert order[0] == "weak0"


def test_rank_wikipedia_score_grows_with_wordcount() -> None:
    short = _cand("wikipedia", "a", wordcount=10)
    long_ = _cand("wikipedia", "b", wordcount=100000)
    assert _solo_score(long_) > _solo_score(short)


def test_rank_gdelt_score_grows_with_recency() -> None:
    old = _cand("gdelt", "stale", seendate="20100101T000000Z", tone="0")
    today = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    fresh = _cand("gdelt", "fresh", seendate=today, tone="0")
    assert _solo_score(fresh) > _solo_score(old)


def test_rank_handles_malformed_dates_without_crashing() -> None:
    cs = [
        _cand("arxiv", "a", abstract="x", published="not-a-date"),
        _cand("gdelt", "g", seendate="garbage", tone="not-a-number"),
    ]
    # arXiv source dispatches via candidate.source; rank shouldn't crash
    assert rank(cs[:1], "x")
    assert rank(cs[1:], "x")

