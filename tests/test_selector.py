"""Tests for the global LLM Selector.

Covers the pick-mapping (index → candidate, goal_relevance stamped, budget
cap, bad-index skip), the empty-pool short-circuit, and the SAMPLE_MODE
round-robin fallback (source diversity + budget respected).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai.models.test import TestModel

from memex.common.config import Settings
from memex.common.trace import TraceWriter
from memex.kg_intelligence.agents import AgentCallSinks
from memex.kg_intelligence.agents.selector import select_documents
from memex.kg_intelligence.ids import Source
from memex.kg_intelligence.sources.common.base import AdapterDescription, SourceCandidate


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
    return AgentCallSinks(trace=TraceWriter(request_id="sel-test", log_dir=tmp_path))


def _cand(source: Source, canonical: str, *, rank_score: float | None = None) -> SourceCandidate:
    signals: dict[str, object] = {}
    if rank_score is not None:
        signals["rank_score"] = rank_score
    return SourceCandidate(
        source=source,
        canonical=canonical,
        url=f"https://example/{source}/{canonical}",
        title=f"{source} {canonical}",
        abstract=f"about {canonical}",
        raw_signals=signals,
    )


_DESCS = [
    AdapterDescription(
        name="arxiv", summary="papers", posture="throttled",
        preferred_doc_cap=3, metadata_fields=[],
    ),
    AdapterDescription(
        name="wikipedia", summary="encyclopedia", posture="relaxed",
        preferred_doc_cap=3, metadata_fields=[],
    ),
]


async def test_picks_map_to_candidates_and_stamp_goal_relevance(tmp_path: Path) -> None:
    pool = [_cand("arxiv", "a"), _cand("wikipedia", "w"), _cand("gdelt", "g")]
    stub = TestModel(
        custom_output_args={
            "picks": [
                {"candidate_index": 2, "goal_relevance": 0.9, "reason": "best"},
                {"candidate_index": 0, "goal_relevance": 0.5, "reason": "ok"},
            ]
        }
    )
    chosen = await select_documents(
        pool=pool,
        ranked_topics=["green hydrogen"],
        adapter_descriptions=_DESCS,
        budget=10,
        settings=_settings(),
        sinks=_sinks(tmp_path),
        llm_model=stub,
    )
    # preference order preserved; goal_relevance stamped onto the chosen objects
    assert [c.canonical for c in chosen] == ["g", "a"]
    assert chosen[0].goal_relevance == 0.9
    assert chosen[1].goal_relevance == 0.5


async def test_budget_caps_and_bad_indices_skipped(tmp_path: Path) -> None:
    pool = [_cand("arxiv", str(i)) for i in range(5)]
    stub = TestModel(
        custom_output_args={
            "picks": [
                {"candidate_index": 0, "goal_relevance": 0.9, "reason": ""},
                {"candidate_index": 99, "goal_relevance": 0.9, "reason": "oob"},
                {"candidate_index": 0, "goal_relevance": 0.9, "reason": "dup"},
                {"candidate_index": 1, "goal_relevance": 0.8, "reason": ""},
                {"candidate_index": 2, "goal_relevance": 0.7, "reason": ""},
            ]
        }
    )
    chosen = await select_documents(
        pool=pool,
        ranked_topics=["q"],
        adapter_descriptions=_DESCS,
        budget=2,
        settings=_settings(),
        sinks=_sinks(tmp_path),
        llm_model=stub,
    )
    # oob + duplicate dropped, budget caps at 2
    assert [c.canonical for c in chosen] == ["0", "1"]


async def test_empty_pool_short_circuits(tmp_path: Path) -> None:
    chosen = await select_documents(
        pool=[],
        ranked_topics=["q"],
        adapter_descriptions=_DESCS,
        budget=10,
        settings=_settings(),
        sinks=_sinks(tmp_path),
    )
    assert chosen == []


async def test_sample_mode_round_robin_keeps_source_diversity(tmp_path: Path) -> None:
    pool = [
        _cand("arxiv", "a0", rank_score=0.9),
        _cand("arxiv", "a1", rank_score=0.8),
        _cand("wikipedia", "w0", rank_score=0.7),
    ]
    chosen = await select_documents(
        pool=pool,
        ranked_topics=["q"],
        adapter_descriptions=_DESCS,
        budget=2,
        settings=_settings(sample_mode=True),
        sinks=_sinks(tmp_path),
    )
    # round-robin: arxiv rank-0, then wikipedia rank-0 (diversity over depth)
    assert [c.canonical for c in chosen] == ["a0", "w0"]
    assert chosen[0].goal_relevance == 0.9


async def test_sample_mode_respects_budget_beyond_pool(tmp_path: Path) -> None:
    pool = [_cand("arxiv", "a0"), _cand("wikipedia", "w0")]
    chosen = await select_documents(
        pool=pool,
        ranked_topics=["q"],
        adapter_descriptions=_DESCS,
        budget=10,
        settings=_settings(sample_mode=True),
        sinks=_sinks(tmp_path),
    )
    assert {c.canonical for c in chosen} == {"a0", "w0"}
