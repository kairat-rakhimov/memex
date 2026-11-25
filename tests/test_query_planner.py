"""Tests for the per-adapter query planner.

Covers the relevance posture (throttled sources sit out a low-relevance
round without an LLM call or search; relaxed sources always act), topic
tagging of the returned candidates, and the planner's query rewrite via a
stubbed model + the SAMPLE_MODE fallback.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai.models.test import TestModel

from memex.common.config import Settings
from memex.common.trace import TraceWriter
from memex.kg_intelligence.agents import AgentCallSinks
from memex.kg_intelligence.sources.common.base import SourceCandidate
from memex.kg_intelligence.sources.common.query_planner import (
    BaseQueryPlanner,
    plan_and_search,
)


def _settings(*, sample_mode: bool = False) -> Settings:
    return Settings(
        api_key="test-key",
        base_url="https://llm.test/v1",
        chat_model="gpt-test",
        reasoning_model="gpt-test",
        embedding_model="emb-test",
        embedding_dimension=3072,
        host="0.0.0.0",
        port=8000,
        run_timeout_seconds=300,
        log_dir=Path("logs"),
        log_level="INFO",
        sample_mode=sample_mode,
        contact=None,
    )


def _sinks(tmp_path: Path) -> AgentCallSinks:
    return AgentCallSinks(trace=TraceWriter(request_id="qp-test", log_dir=tmp_path))


class _SpySearch:
    """Records the queries it was handed; returns one candidate per query."""

    def __init__(self, source: str = "arxiv") -> None:
        self.source = source
        self.calls: list[list[str]] = []

    def __call__(self, queries: list[str], *, limit: int) -> list[SourceCandidate]:
        self.calls.append(list(queries))
        return [
            SourceCandidate(
                source=self.source,  # type: ignore[arg-type]
                canonical=q,
                url=f"https://example/{q}",
                title=q,
            )
            for q in queries
        ]


def _planner() -> BaseQueryPlanner:
    return BaseQueryPlanner(source_name="arxiv", guidance="test guidance")


async def test_throttled_low_relevance_skips_without_search(tmp_path: Path) -> None:
    search = _SpySearch()
    out = await plan_and_search(
        planner=_planner(),
        search_fn=search,
        posture="throttled",
        ranked_topics=["green hydrogen"],
        relevance="medium",
        limit=10,
        settings=_settings(sample_mode=True),
        sinks=_sinks(tmp_path),
    )
    assert out == []
    assert search.calls == []  # neither the planner nor the search ran


async def test_throttled_high_relevance_acts(tmp_path: Path) -> None:
    search = _SpySearch()
    out = await plan_and_search(
        planner=_planner(),
        search_fn=search,
        posture="throttled",
        ranked_topics=["green hydrogen", "electrolysis"],
        relevance="high",
        limit=10,
        settings=_settings(sample_mode=True),
        sinks=_sinks(tmp_path),
    )
    assert out  # acted
    assert search.calls  # search ran


async def test_relaxed_ignores_low_relevance(tmp_path: Path) -> None:
    search = _SpySearch()
    out = await plan_and_search(
        planner=_planner(),
        search_fn=search,
        posture="relaxed",
        ranked_topics=["green hydrogen"],
        relevance="low",
        limit=10,
        settings=_settings(sample_mode=True),
        sinks=_sinks(tmp_path),
    )
    assert out  # relaxed sources act regardless of relevance
    assert search.calls


async def test_candidates_tagged_with_topics(tmp_path: Path) -> None:
    search = _SpySearch()
    topics = ["green hydrogen", "PEM electrolysis"]
    out = await plan_and_search(
        planner=_planner(),
        search_fn=search,
        posture="relaxed",
        ranked_topics=topics,
        relevance="high",
        limit=10,
        settings=_settings(sample_mode=True),
        sinks=_sinks(tmp_path),
    )
    assert out
    for candidate in out:
        assert candidate.topics == topics


async def test_sample_mode_uses_topics_as_queries(tmp_path: Path) -> None:
    queries = await _planner().plan_queries(
        ["green hydrogen", "electrolysis"],
        settings=_settings(sample_mode=True),
        sinks=_sinks(tmp_path),
    )
    assert queries == ["green hydrogen", "electrolysis"]


async def test_planner_rewrites_queries_via_model(tmp_path: Path) -> None:
    stub = TestModel(custom_output_args={"queries": ["iridium catalyst", "Nafion membrane"]})
    queries = await _planner().plan_queries(
        ["green hydrogen"],
        settings=_settings(),
        sinks=_sinks(tmp_path),
        llm_model=stub,
    )
    assert queries == ["iridium catalyst", "Nafion membrane"]


async def test_max_queries_caps_output(tmp_path: Path) -> None:
    """A planner built with max_queries=1 (e.g. gdelt, rate-limited) emits a
    single query — both the sample fallback and a model that over-produces."""

    planner = BaseQueryPlanner(source_name="gdelt", guidance="news", max_queries=1)

    # sample fallback (SAMPLE_MODE) caps to 1
    sample = await planner.plan_queries(
        ["a", "b", "c"], settings=_settings(sample_mode=True), sinks=_sinks(tmp_path)
    )
    assert sample == ["a"]

    # a model returning 3 queries is still capped to 1
    stub = TestModel(custom_output_args={"queries": ["q1", "q2", "q3"]})
    capped = await planner.plan_queries(
        ["a", "b", "c"], settings=_settings(), sinks=_sinks(tmp_path), llm_model=stub
    )
    assert capped == ["q1"]
