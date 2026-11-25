"""Tests for the six kg-loop agents (now PydanticAI-backed).

Each agent is a `pydantic_ai.Agent` with a schema-validated `output_type`.
Tests stub the model with `TestModel(custom_output_args=...)` for
happy-path assertions and `FunctionModel` for failure-mode coverage.
Trace emission is asserted by inspecting the JSONL file the
`TraceWriter` writes to `tmp_path`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from memex.common.config import Settings
from memex.common.trace import TraceWriter
from memex.kg_intelligence.agents import (
    AgentCallSinks,
    LLMThrottleError,
    extract,
    kg_review,
    plan,
    reconcile,
    resolve,
)
from memex.kg_intelligence.agents.base import _looks_like_llm_throttle
from memex.kg_intelligence.agents.schemas import (
    ExtractedEntity,
    KgGap,
)
from memex.kg_intelligence.state import CorpusState
from memex.kg_intelligence.views import KgSchemaView, KgView
from memex.observability import EntitySnapshot, TraceStore

# ── shared helpers ────────────────────────────────────────────────────────


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


@pytest.fixture
def trace(tmp_path: Path) -> TraceWriter:
    return TraceWriter(request_id="test-req", log_dir=tmp_path)


def _trace_events(trace: TraceWriter) -> list[dict]:
    return [json.loads(line) for line in trace.path.read_text().splitlines() if line.strip()]


def _raises(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    raise RuntimeError("simulated LLM failure")


def _raising_model() -> FunctionModel:
    return FunctionModel(_raises)


# ── RoundPlanner ───────────────────────────────────────────────────────────


async def test_planner_returns_structured_output(trace: TraceWriter) -> None:
    stub = TestModel(
        custom_output_args={
            "topics": [
                {"topic": "iridium oxide photocatalyst", "rationale": "core catalyst"},
                {"topic": "PEM electrolyzer Nafion", "rationale": "membrane chemistry"},
            ],
            "adapter_relevance": {"arxiv": "high", "wikipedia": "medium", "gdelt": "low"},
            "description": "decompose base query into ranked topics",
        }
    )
    result = await plan(
        base_query="green hydrogen",
        corpus_state=CorpusState(),
        kg_gaps=[],
        directives=[],
        adapter_descriptions=[],
        settings=_settings(),
        sinks=AgentCallSinks(trace=trace),
        llm_model=stub,
    )
    assert len(result.topics) == 2
    assert result.topics[0].topic == "iridium oxide photocatalyst"
    assert result.adapter_relevance["arxiv"] == "high"
    assert "decompose" in result.description
    [event] = _trace_events(trace)
    assert event["agent_name"] == "RoundPlanner"
    assert event["status"] == "success"


async def test_planner_falls_back_to_sample_when_model_raises(trace: TraceWriter) -> None:
    result = await plan(
        base_query="green hydrogen",
        corpus_state=CorpusState(),
        kg_gaps=[],
        directives=[],
        adapter_descriptions=[],
        settings=_settings(),
        sinks=AgentCallSinks(trace=trace),
        llm_model=_raising_model(),
    )
    assert result.topics  # sample stub still populates
    assert _trace_events(trace)[0]["status"] == "failed"


async def test_planner_sample_mode_skips_model(trace: TraceWriter) -> None:
    # FunctionModel would raise if called — proves SAMPLE_MODE short-circuits.
    result = await plan(
        base_query="green hydrogen",
        corpus_state=CorpusState(round_number=2),
        kg_gaps=[],
        directives=[],
        adapter_descriptions=[],
        settings=_settings(sample_mode=True),
        sinks=AgentCallSinks(trace=trace),
        llm_model=_raising_model(),
    )
    assert "SAMPLE_MODE" in result.description


# ── LLM throttle classifier (run_agent_or_fallback) ────────────────────


def test_looks_like_llm_throttle_matches_markers() -> None:
    assert _looks_like_llm_throttle(Exception("Rate limit exceeded"))
    assert _looks_like_llm_throttle(Exception("subscription quota exhausted"))
    assert _looks_like_llm_throttle(Exception("HTTP 429 Too Many Requests"))
    assert _looks_like_llm_throttle(
        Exception("You have reached the limit of your current subscription quota")
    )
    assert _looks_like_llm_throttle(Exception("upstream throttled"))


def test_looks_like_llm_throttle_rejects_unrelated_errors() -> None:
    assert not _looks_like_llm_throttle(Exception("invalid JSON output"))
    assert not _looks_like_llm_throttle(Exception("network unreachable"))
    assert not _looks_like_llm_throttle(ValueError("missing required field"))


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_planner_raises_llm_throttle_on_throttle_http_status(
    trace: TraceWriter, status: int
) -> None:
    def _raise_http(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=status, model_name="gpt-test", body=None)

    with pytest.raises(LLMThrottleError):
        await plan(
            base_query="green hydrogen",
            corpus_state=CorpusState(),
            kg_gaps=[],
            directives=[],
            adapter_descriptions=[],
            settings=_settings(),
            sinks=AgentCallSinks(trace=trace),
            llm_model=FunctionModel(_raise_http),
        )


async def test_planner_falls_back_on_non_throttle_http_status(
    trace: TraceWriter,
) -> None:
    """HTTP 400 is a permanent client error, not a throttle — sample fallback engages."""

    def _raise_400(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=400, model_name="gpt-test", body="bad request")

    result = await plan(
        base_query="green hydrogen",
        corpus_state=CorpusState(),
        kg_gaps=[],
        directives=[],
        adapter_descriptions=[],
        settings=_settings(),
        sinks=AgentCallSinks(trace=trace),
        llm_model=FunctionModel(_raise_400),
    )
    assert result.topics  # sample stub
    assert _trace_events(trace)[0]["status"] == "failed"


async def test_planner_raises_llm_throttle_on_quota_text(
    trace: TraceWriter,
) -> None:
    """LLM quota errors can arrive as generic exceptions whose text
    matches the marker substrings — the text classifier catches these."""

    def _raise_quota(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError(
            "You have reached the limit of your current subscription quota."
        )

    with pytest.raises(LLMThrottleError):
        await plan(
            base_query="green hydrogen",
            corpus_state=CorpusState(),
            kg_gaps=[],
            directives=[],
            adapter_descriptions=[],
            settings=_settings(),
            sinks=AgentCallSinks(trace=trace),
            llm_model=FunctionModel(_raise_quota),
        )


async def test_throttle_not_raised_in_sample_mode(trace: TraceWriter) -> None:
    """SAMPLE_MODE short-circuits before the LLM call — no chance to throttle."""

    def _raise_429(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=429, model_name="gpt-test", body=None)

    result = await plan(
        base_query="green hydrogen",
        corpus_state=CorpusState(),
        kg_gaps=[],
        directives=[],
        adapter_descriptions=[],
        settings=_settings(sample_mode=True),
        sinks=AgentCallSinks(trace=trace),
        llm_model=FunctionModel(_raise_429),
    )
    assert "SAMPLE_MODE" in result.description


# ── UsefulnessGate tests live in tests/test_usefulness_gate.py ─────────────


# ── Extractor ──────────────────────────────────────────────────────────────


async def test_extractor_returns_structured_output_keeps_all_relations(
    trace: TraceWriter,
) -> None:
    stub = TestModel(
        custom_output_args={
            "entities": [
                {"name": "iridium", "type": "material"},
                {"name": "OER", "type": "process"},
            ],
            "relations": [
                {"subject": "iridium", "predicate": "catalyzes", "object": "OER"},
                {"subject": "iridium", "predicate": "powers", "object": "unmentioned_entity"},
            ],
        }
    )
    result = await extract(
        doc_id="doc_a",
        doc_title="Iridium catalysts",
        doc_text="Iridium catalyzes the OER...",
        settings=_settings(),
        sinks=AgentCallSinks(trace=trace),
        llm_model=stub,
    )
    assert len(result.entities) == 2
    # Free-form Extractor no longer drops dangling relations — the Mapper
    # canonicalizes every triple endpoint into an entity downstream.
    assert len(result.relations) == 2
    assert {r.object for r in result.relations} == {"OER", "unmentioned_entity"}


async def test_extractor_falls_back_on_failure(trace: TraceWriter) -> None:
    result = await extract(
        doc_id="doc_b",
        doc_title="Wikipedia title",
        doc_text="x",
        settings=_settings(),
        sinks=AgentCallSinks(trace=trace),
        llm_model=_raising_model(),
    )
    # SAMPLE-style fallback returns one entity derived from title head
    assert len(result.entities) == 1
    assert _trace_events(trace)[0]["status"] == "failed"


async def test_extractor_truncates_long_doc_text(trace: TraceWriter) -> None:
    captured: dict[str, str] = {}

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # capture the user prompt that reaches the model
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    text = getattr(part, "content", "")
                    if text and "Document text:" in text:
                        captured["user"] = text
        return ModelResponse(parts=[TextPart('{"entities": [], "relations": []}')])

    long_text = "Iridium " * 10_000  # ~80k chars
    await extract(
        doc_id="doc_c",
        doc_title="t",
        doc_text=long_text,
        settings=_settings(),
        sinks=AgentCallSinks(trace=trace),
        llm_model=FunctionModel(stub),
    )
    user_content = captured.get("user", "")
    assert "Document text:" in user_content
    assert len(user_content) < len(long_text) + 1000  # truncated


# ── Reconciler ─────────────────────────────────────────────────────────────


async def test_reconciler_returns_cross_chunk_relations_and_drops_dangling(
    trace: TraceWriter,
) -> None:
    """Reconciler keeps relations whose subj+obj are in per-chunk entities;
    drops relations naming entities not extracted by any chunk."""

    stub = TestModel(
        custom_output_args={
            "relations": [
                # Both endpoints known → kept
                {"subject": "iridium", "predicate": "co_occurs_with", "object": "PEM"},
                # Unknown object → dropped
                {"subject": "iridium", "predicate": "powers", "object": "mystery_entity"},
            ],
        }
    )
    per_chunk_entities = [
        (0, [ExtractedEntity(name="iridium", type="material")]),
        (1, [ExtractedEntity(name="PEM", type="device")]),
    ]
    result = await reconcile(
        doc_id="doc_multichunk",
        doc_title="Iridium PEM catalysis",
        doc_synopsis="Iridium catalyzes OER in PEM electrolyzers.",
        per_chunk_entities=per_chunk_entities,
        schema=KgSchemaView(entity_types=["material", "device"], relation_types=[]),
        settings=_settings(),
        sinks=AgentCallSinks(trace=trace),
        llm_model=stub,
    )
    assert len(result.relations) == 1
    assert result.relations[0].subject == "iridium"
    assert result.relations[0].object == "PEM"


async def test_reconciler_empty_result_is_valid(trace: TraceWriter) -> None:
    """Empty Reconciler output (no cross-chunk relations found) is valid."""

    stub = TestModel(custom_output_args={"relations": []})
    per_chunk_entities = [
        (0, [ExtractedEntity(name="iridium", type="material")]),
        (1, [ExtractedEntity(name="PEM", type="device")]),
    ]
    result = await reconcile(
        doc_id="doc_quiet",
        doc_title="Unrelated topics",
        doc_synopsis="...",
        per_chunk_entities=per_chunk_entities,
        schema=KgSchemaView(),
        settings=_settings(),
        sinks=AgentCallSinks(trace=trace),
        llm_model=stub,
    )
    assert result.relations == []
    events = _trace_events(trace)
    assert any(e.get("agent_name") == "Reconciler" for e in events)


async def test_reconciler_sample_mode_crosses_first_two_nonempty_chunks(
    trace: TraceWriter,
) -> None:
    """SAMPLE_MODE fallback emits one deterministic cross-chunk relation
    when at least two chunks have entities."""

    per_chunk_entities = [
        (0, []),  # empty
        (1, [ExtractedEntity(name="alpha", type="t")]),
        (2, [ExtractedEntity(name="beta", type="t")]),
    ]
    result = await reconcile(
        doc_id="doc_sample",
        doc_title="Sample doc",
        doc_synopsis="...",
        per_chunk_entities=per_chunk_entities,
        schema=KgSchemaView(),
        settings=_settings(sample_mode=True),
        sinks=AgentCallSinks(trace=trace),
        llm_model=None,
    )
    assert len(result.relations) == 1
    assert result.relations[0].subject == "alpha"
    assert result.relations[0].object == "beta"


async def test_reconciler_sample_mode_yields_empty_when_only_one_chunk_has_entities(
    trace: TraceWriter,
) -> None:
    per_chunk_entities = [
        (0, [ExtractedEntity(name="alpha", type="t")]),
        (1, []),
    ]
    result = await reconcile(
        doc_id="doc_sample_thin",
        doc_title="Sparse doc",
        doc_synopsis="...",
        per_chunk_entities=per_chunk_entities,
        schema=KgSchemaView(),
        settings=_settings(sample_mode=True),
        sinks=AgentCallSinks(trace=trace),
        llm_model=None,
    )
    assert result.relations == []


# ── KGReviewer ─────────────────────────────────────────────────────────────


def _kg_view_seeded() -> KgView:
    """Build a KgView seeded with three entities + one relation for
    KGReviewer tests. KGReviewer takes a KgView
    instead of an in-memory KgState aggregate."""

    entities = {
        "iridium": EntitySnapshot(
            id=1, canonical_name="iridium", entity_type="material",
            mentions=8, degree=1, source_doc_ids=["d1", "d2"],
        ),
        "PEM electrolyzer": EntitySnapshot(
            id=2, canonical_name="PEM electrolyzer", entity_type="device",
            mentions=5, degree=3, source_doc_ids=["d3"],
        ),
        "Nafion": EntitySnapshot(
            id=3, canonical_name="Nafion", entity_type="material",
            mentions=2, degree=0, source_doc_ids=["d3"],
        ),
    }
    return KgView(
        schema=KgSchemaView(entity_types=["material", "device"], relation_types=["catalyzes"]),
        entities=entities,
        predicate_counts=[("catalyzes", 1)],
    )


async def test_kg_reviewer_returns_structured_output(trace: TraceWriter) -> None:
    stub = TestModel(
        custom_output_args={
            "gaps": [
                {"description": "Nafion mentioned but disconnected", "kind": "low_degree"},
                {"description": "synthesis steps absent", "kind": "weak_relation"},
            ],
        }
    )
    result, _ = await kg_review(
        base_query="green hydrogen",
        kg=_kg_view_seeded(),
        corpus_state=CorpusState(round_number=1),
        settings=_settings(),
        sinks=AgentCallSinks(trace=trace),
        llm_model=stub,
    )
    assert len(result.gaps) == 2
    assert result.gaps[1].kind == "weak_relation"


async def test_kg_reviewer_sample_mode_flags_low_degree(trace: TraceWriter) -> None:
    result, _ = await kg_review(
        base_query="green hydrogen",
        kg=_kg_view_seeded(),
        corpus_state=CorpusState(),
        settings=_settings(sample_mode=True),
        sinks=AgentCallSinks(trace=trace),
        llm_model=_raising_model(),
    )
    assert any("Nafion" in g.description or "iridium" in g.description.lower() for g in result.gaps)


async def test_kg_reviewer_returns_ranked_gaps(trace: TraceWriter) -> None:
    """Reviewer's output preserves rank ordering; the next-round Planner
    consumes the ranked list directly. Rank 0 = most important."""

    stub = TestModel(
        custom_output_args={
            "gaps": [
                {"description": "catalyst chemistry under-explored", "kind": "low_degree", "rank": 0},
                {"description": "membrane materials sparse", "kind": "weak_relation", "rank": 1},
                {"description": "policy events absent", "kind": "low_degree", "rank": 2},
            ],
        }
    )
    result, _ = await kg_review(
        base_query="green hydrogen",
        kg=_kg_view_seeded(),
        corpus_state=CorpusState(round_number=1),
        settings=_settings(),
        sinks=AgentCallSinks(trace=trace),
        llm_model=stub,
    )
    assert [g.rank for g in result.gaps] == [0, 1, 2]
    assert result.gaps[0].description.startswith("catalyst")


async def test_planner_prompt_surfaces_ranked_gaps_in_order(trace: TraceWriter) -> None:
    """The Planner's user prompt must surface gaps sorted by rank so the LLM
    receives the importance signal even when the input list arrives
    out-of-order. We assert by inspecting the captured prompt on the trace."""

    from memex.kg_intelligence.agents.round_planner import _build_user_prompt

    gaps = [
        KgGap(description="GAP_LOW", kind="low_degree", rank=5),
        KgGap(description="GAP_TOP", kind="weak_relation", rank=0),
        KgGap(description="GAP_MID", kind="low_degree", rank=2),
    ]
    prompt = _build_user_prompt(
        base_query="green hydrogen",
        corpus_state=CorpusState(round_number=2),
        kg_gaps=gaps,
        directives=[],
        adapter_descriptions=[],
    )
    # The prompt mentions ranking semantics + lists top-ranked gap first.
    assert "ranked" in prompt.lower()
    top_idx = prompt.index("GAP_TOP")
    mid_idx = prompt.index("GAP_MID")
    low_idx = prompt.index("GAP_LOW")
    assert top_idx < mid_idx < low_idx
    # Rank annotations are present so the LLM can weight allocation.
    assert "rank=0" in prompt and "rank=2" in prompt and "rank=5" in prompt


def test_round_reviewer_prompt_renders_per_topic_saturation() -> None:
    """The strategist's prompt shows the round-level already_seen rate
    AND the per-topic breakdown, most-saturated theme first — the signal it uses
    to abandon the exact mined-out topic."""
    from memex.kg_intelligence.agents.round_reviewer import _build_user_prompt
    from memex.kg_intelligence.agents.schemas import KgReviewResult

    recent_metrics = [{
        "round_number": 17, "new_entities": 0, "new_relations": 0,
        "docs_fetched_total": 0, "tokens_spent_total": 14000,
        "tokens_per_new_entity": None, "consolidation_repairs": 0,
        "already_ingested_rate": 1.0,
        "already_seen_by_topic": {"Policy frameworks": [5, 5], "Hydrogen storage": [4, 0]},
    }]
    prompt = _build_user_prompt(
        base_query="green hydrogen", corpus_state=CorpusState(round_number=17),
        kg_review=KgReviewResult(gaps=[]), recent_metrics=recent_metrics,
    )
    assert "already_seen=100%" in prompt  # round-level rate in the metrics line
    assert "Policy frameworks: 5/5 already-seen (100%)" in prompt
    assert "Hydrogen storage: 0/4 already-seen (0%)" in prompt
    # most-saturated theme rendered first
    assert prompt.index("Policy frameworks") < prompt.index("Hydrogen storage")


async def test_planner_with_empty_gaps_falls_back_to_exploratory(trace: TraceWriter) -> None:
    """Empty gaps list — Planner's prompt has no ranked-gaps section; it
    runs in exploratory mode using the base query + tried-history."""

    from memex.kg_intelligence.agents.round_planner import _build_user_prompt

    prompt = _build_user_prompt(
        base_query="green hydrogen",
        corpus_state=CorpusState(round_number=2),
        kg_gaps=[],
        directives=[],
        adapter_descriptions=[],
    )
    assert "ranked" not in prompt.lower()
    assert "KG structural gaps" not in prompt


def test_planner_prompt_excludes_abandoned_and_surfaces_directives() -> None:
    """The RoundPlanner prompt hard-lists abandoned goals (DO NOT pursue) and
    surfaces the strategist's per-theme directives."""

    from memex.kg_intelligence.agents.round_planner import _build_user_prompt
    from memex.kg_intelligence.agents.schemas import GoalDirective, GoalVerdict

    prompt = _build_user_prompt(
        base_query="green hydrogen",
        corpus_state=CorpusState(round_number=3, abandoned_goals=["economic impact"]),
        kg_gaps=[],
        directives=[
            GoalDirective(
                theme="catalyst chemistry",
                verdict=GoalVerdict.pursue,
                rationale="paying off",
            ),
        ],
        adapter_descriptions=[],
    )
    assert "Abandoned goals" in prompt
    assert "economic impact" in prompt
    assert "catalyst chemistry" in prompt
    assert "pursue" in prompt


def test_planner_prompt_omits_empty_round_hint_when_counter_zero() -> None:
    """No nudge text when the counter is at zero — keeps prompts clean for
    the common case."""

    from memex.kg_intelligence.agents.round_planner import _build_user_prompt

    prompt = _build_user_prompt(
        base_query="green hydrogen",
        corpus_state=CorpusState(round_number=1, consecutive_empty_rounds=0),
        kg_gaps=[],
        directives=[],
        adapter_descriptions=[],
    )
    assert "0 new documents" not in prompt
    assert "search_exhausted" not in prompt


def test_planner_prompt_surfaces_empty_round_hint_when_counter_positive() -> None:
    """When the prior round added 0 docs, the planner sees a re-strategize
    nudge alongside the existing signals."""

    from memex.kg_intelligence.agents.round_planner import _build_user_prompt

    prompt = _build_user_prompt(
        base_query="green hydrogen",
        corpus_state=CorpusState(round_number=3, consecutive_empty_rounds=2),
        kg_gaps=[],
        directives=[],
        adapter_descriptions=[],
    )
    assert "2 round(s)" in prompt
    assert "0 new documents" in prompt


# ── Resolver ────────────────────────────────────────────────────────────────


def _seed_resolver_store(
    tmp_path: Path,
    entities_with_docs: list[tuple[str, list[str]]],
    *,
    relations: list[tuple[str, str, str, str]] | None = None,
) -> tuple[TraceStore, AgentCallSinks, TraceWriter]:
    """Build a TraceStore seeded with the given (name, source_doc_ids)
    pairs. source_doc_ids surface via the kg_relations JOIN — we wire
    them in by inserting one self-relation per (entity, doc) pair so
    Resolver's _cooccurrence_filter can see them.

    Returns ``(store, sinks, trace)``; caller owns ``store.close()``.
    """

    trace = TraceWriter(request_id="resolver-test", log_dir=tmp_path / "_logs")
    store = TraceStore.open(tmp_path / "kg")
    build = store.start_build(
        build_run_id=f"kg_build_resolver_{tmp_path.name}",
        kg_id="kg", base_query="resolver test",
        config={}, sources=[], sample_mode=True, resume_from_round=None,
    )
    round_ = store.start_round(
        build, round_number=1,
        entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    call = store.begin_agent_call(
        round_, agent_name="_seed", action="seed",
        triggered_by="test_setup", input_summary="",
    )
    store.finish_agent_call(call, status="skipped", duration_ms=0, output_summary="")

    # Seed entities + one self-relation per source_doc so source_doc_ids
    # surface via the kg_relations JOIN in entity_snapshot_for_resolver.
    for name, docs in entities_with_docs:
        for doc in docs:
            store.merge_extraction(
                entities=[(name, "material")],
                relations=[(name, "_seed_self", name)],
                source_doc_id=doc,
                chunk_id=None,
                agent_call_id=call.agent_call_id,
                round_number=1,
            )
    if relations:
        for subj, pred, obj, doc in relations:
            store.merge_extraction(
                entities=[],
                relations=[(subj, pred, obj)],
                source_doc_id=doc,
                chunk_id=None,
                agent_call_id=call.agent_call_id,
                round_number=1,
            )

    sinks = AgentCallSinks(trace=trace, store=store, round_=round_)
    return store, sinks, trace


async def test_resolver_tier1_merges_case_punctuation_and_plural_variants(
    tmp_path: Path,
) -> None:
    """All four normalised-equivalent names collapse to one canonical entry.
    Tier-1 is deterministic and runs without any LLM call.

    Note: SQLite COLLATE NOCASE collapses pure case-variants on insert,
    so 'Metal hydrides' / 'metal hydrides' merge at the SQL layer; the
    remaining variants ('metal-hydrides', 'metal hydride') merge at
    tier-1 Python normalization.
    """

    store, sinks, _ = _seed_resolver_store(
        tmp_path,
        [
            ("Metal hydrides", ["doc_1"]),
            ("metal hydrides", ["doc_2"]),
            ("metal-hydrides", ["doc_3"]),
            ("metal hydride", ["doc_4"]),
        ],
    )
    try:
        snapshot_before = store.entity_snapshot_for_resolver()
        committed = await resolve(
            store=store,
            round_number=1,
            entities_touched_this_round=set(snapshot_before.keys()),
            settings=_settings(sample_mode=True),
            sinks=sinks,
            llm_model=None,
        )
        snapshot_after = store.entity_snapshot_for_resolver()
        # All variants collapse to a single survivor.
        assert len(snapshot_after) == 1
        # 2 case-variants already collapsed at INSERT (COLLATE NOCASE), so
        # tier-1 commits the remaining 2 merges.
        assert len(committed) >= 1
        assert all(p.tier == 1 for p in committed)
    finally:
        store.close()


async def test_resolver_tier2_acronym_match_merges_on_positive_verdict(
    tmp_path: Path,
) -> None:
    """PEM ↔ Proton exchange membrane electrolyzer fires via the acronym
    heuristic; the LLM returns a positive verdict → committed merge."""

    store, sinks, _ = _seed_resolver_store(
        tmp_path,
        [
            ("PEM", ["doc_a"]),
            ("Proton exchange membrane electrolyzer", ["doc_a", "doc_b"]),
        ],
    )
    stub = TestModel(
        custom_output_args={
            "verdicts": [
                {
                    "name_a": "PEM",
                    "name_b": "Proton exchange membrane electrolyzer",
                    "verdict": "merge",
                    "confidence": 0.86,
                    "reason": "acronym match plus shared source doc",
                }
            ]
        }
    )
    try:
        committed = await resolve(
            store=store, round_number=2,
            entities_touched_this_round={"PEM"},
            settings=_settings(), sinks=sinks, llm_model=stub,
        )
        snapshot = store.entity_snapshot_for_resolver()
        assert len(snapshot) == 1
        assert len(committed) == 1
        assert committed[0].tier == 2
        assert 0.8 < committed[0].confidence < 0.9
    finally:
        store.close()


async def test_resolver_tier3_high_cosine_bypasses_cooccurrence(tmp_path: Path) -> None:
    """A near-identical-embedding pair with NO shared source
    doc (cross-document spelling variant) bypasses the cooccurrence filter and
    reaches the tier-2 judge. Without the bypass the filter would drop it and the
    pair would never merge — domain-agnostic, no hardcoded morphology."""

    import numpy as np

    from memex.kg_intelligence.embeddings import vec_to_blob

    store, sinks, _ = _seed_resolver_store(
        tmp_path,
        [("electrolyser", ["doc_a"]), ("electrolyzer", ["doc_b"])],  # no shared doc
    )
    snap = store.entity_snapshot_for_resolver()

    def _seed_emb(name: str, vec: list[float]) -> None:
        v = np.array(vec, dtype=np.float32)
        v = v / np.linalg.norm(v)
        store.record_embeddings_batch(
            [("entity", str(snap[name].id), "openai", "m", 4, vec_to_blob(v))]
        )

    _seed_emb("electrolyser", [1.0, 0.0, 0.0, 0.0])
    _seed_emb("electrolyzer", [0.99, 0.10, 0.0, 0.0])  # cosine ≈ 0.995 ≥ bypass

    stub = TestModel(custom_output_args={"verdicts": [{
        "name_a": "electrolyser", "name_b": "electrolyzer",
        "verdict": "merge", "confidence": 0.9, "reason": "spelling variant",
    }]})
    try:
        committed = await resolve(
            store=store, round_number=2,
            entities_touched_this_round={"electrolyzer"},
            settings=_settings(), sinks=sinks, llm_model=stub,
        )
        assert len(committed) == 1                       # merged despite no shared doc
        assert len(store.entity_snapshot_for_resolver()) == 1
    finally:
        store.close()


async def test_resolver_memoizes_distinct_verdicts(tmp_path: Path) -> None:
    """Optimisation: a pair judged 'distinct' is cached and skipped on re-resolve —
    even when the model would now merge it, the cache short-circuits the LLM."""

    store, sinks, _ = _seed_resolver_store(
        tmp_path,
        [("PEM", ["doc_a"]), ("Proton exchange membrane electrolyzer", ["doc_a"])],
    )
    distinct = TestModel(custom_output_args={"verdicts": [{
        "name_a": "PEM", "name_b": "Proton exchange membrane electrolyzer",
        "verdict": "distinct", "confidence": 0.8, "reason": "judged distinct"}]})
    merge = TestModel(custom_output_args={"verdicts": [{
        "name_a": "PEM", "name_b": "Proton exchange membrane electrolyzer",
        "verdict": "merge", "confidence": 0.9, "reason": "would merge"}]})
    try:
        c1 = await resolve(
            store=store, round_number=1, entities_touched_this_round={"PEM"},
            settings=_settings(), sinks=sinks, llm_model=distinct,
        )
        assert len(c1) == 0                          # distinct → no merge
        assert len(store.cached_distinct_pairs()) == 1  # …and cached

        # Round 2: the model WOULD merge, but the pair is cached-distinct →
        # filtered out before the LLM, so it stays unmerged.
        c2 = await resolve(
            store=store, round_number=2, entities_touched_this_round={"PEM"},
            settings=_settings(), sinks=sinks, llm_model=merge,
        )
        assert len(c2) == 0
        assert len(store.entity_snapshot_for_resolver()) == 2
    finally:
        store.close()


async def test_resolver_tier2_negative_verdict_keeps_entities_distinct(
    tmp_path: Path,
) -> None:
    """Metal hydrides ↔ Metal oxides has Jaccard ≥ 0.7 (shared `metal`),
    routes to tier 2, LLM says distinct → both entities remain."""

    store, sinks, _ = _seed_resolver_store(
        tmp_path,
        [
            ("Metal hydrides", ["doc_a"]),
            ("Metal oxides", ["doc_a"]),
        ],
    )
    stub = TestModel(
        custom_output_args={
            "verdicts": [
                {
                    "name_a": "Metal hydrides",
                    "name_b": "Metal oxides",
                    "verdict": "distinct",
                    "confidence": 0.95,
                    "reason": "same category, different materials",
                }
            ]
        }
    )
    try:
        snapshot_before = store.entity_snapshot_for_resolver()
        committed = await resolve(
            store=store, round_number=2,
            entities_touched_this_round=set(snapshot_before.keys()),
            settings=_settings(), sinks=sinks, llm_model=stub,
        )
        assert committed == []
        snapshot_after = store.entity_snapshot_for_resolver()
        assert set(snapshot_after.keys()) == {"Metal hydrides", "Metal oxides"}
    finally:
        store.close()


async def test_resolver_cooccurrence_filter_skips_pairs_with_no_shared_docs(
    tmp_path: Path,
) -> None:
    """Two names that look like aliases but never co-appear in any source
    document are pre-filtered out before tier 2 — no LLM call fires."""

    store, sinks, _ = _seed_resolver_store(
        tmp_path,
        [
            ("PEM", ["doc_a"]),
            ("Proton exchange membrane electrolyzer", ["doc_z"]),  # disjoint
        ],
    )
    raising = _raising_model()  # would error if Resolver tried to call it
    try:
        committed = await resolve(
            store=store, round_number=2,
            entities_touched_this_round={"PEM"},
            settings=_settings(), sinks=sinks, llm_model=raising,
        )
        assert committed == []
        snapshot = store.entity_snapshot_for_resolver()
        assert set(snapshot.keys()) == {"PEM", "Proton exchange membrane electrolyzer"}
    finally:
        store.close()


async def test_resolver_writes_mutation_rows_with_reason_and_confidence(
    tmp_path: Path,
) -> None:
    """Tier-1 merge produces a mutation row with confidence=1.0 + parent id.
    Endpoint-rewrite rows for affected relations link to the parent via
    parent_mutation_id. ``target_id`` on endpoint_rewrite rows is the
    kg_relations.id."""

    store, sinks, _ = _seed_resolver_store(
        tmp_path,
        [
            ("Metal hydrides", ["doc_1"]),
            ("metal hydride", ["doc_2"]),
        ],
        relations=[("Metal hydrides", "stores", "metal hydride", "doc_1")],
    )
    try:
        snapshot_before = store.entity_snapshot_for_resolver()
        committed = await resolve(
            store=store, round_number=1,
            entities_touched_this_round=set(snapshot_before.keys()),
            settings=_settings(sample_mode=True),
            sinks=sinks, llm_model=None,
        )
        assert len(committed) == 1
        survivor = committed[0].survivor

        # Parent merge row: target_id = survivor name, confidence = 1.0 (tier 1).
        merge_rows = store.list_mutations_for_target(
            target_kind="entity", target_id=survivor,
        )
        assert len(merge_rows) == 1
        assert merge_rows[0]["kind"] == "merge"
        assert merge_rows[0]["confidence"] == 1.0
        assert "tier-1" in merge_rows[0]["reason"]

        # Endpoint-rewrite child rows for the affected relations.
        parent_id = merge_rows[0]["id"]
        rewrites = store._conn.execute(  # noqa: SLF001
            "SELECT kind, target_id FROM kg_mutations WHERE parent_mutation_id = ?",
            (parent_id,),
        ).fetchall()
        assert len(rewrites) >= 1
        for row in rewrites:
            assert row["kind"] == "endpoint_rewrite"
            # target_id is a stringified kg_relations.id (numeric).
            assert row["target_id"].isdigit()
    finally:
        store.close()


async def test_resolver_periodic_sweep_fires_every_five_rounds(
    tmp_path: Path,
) -> None:
    """Round 5 evaluates all pairs even if neither was touched this round.
    Round 6 falls back to incremental (touched set empty → no LLM call)."""

    store, sinks, _ = _seed_resolver_store(
        tmp_path,
        [
            ("PEM", ["doc_a"]),
            ("Proton exchange membrane electrolyzer", ["doc_a"]),
        ],
    )
    try:
        raising = _raising_model()
        committed_round_6 = await resolve(
            store=store, round_number=6,
            entities_touched_this_round=set(),  # nothing changed this round
            settings=_settings(), sinks=sinks, llm_model=raising,
        )
        # No candidates → no LLM call → no commit
        assert committed_round_6 == []

        # Round 5 sweeps all pairs regardless of touched set
        stub = TestModel(
            custom_output_args={
                "verdicts": [
                    {
                        "name_a": "PEM",
                        "name_b": "Proton exchange membrane electrolyzer",
                        "verdict": "merge",
                        "confidence": 0.9,
                        "reason": "acronym match (sweep)",
                    }
                ]
            }
        )
        committed_round_5 = await resolve(
            store=store, round_number=5,
            entities_touched_this_round=set(),
            settings=_settings(), sinks=sinks, llm_model=stub,
        )
        assert len(committed_round_5) == 1
        snapshot = store.entity_snapshot_for_resolver()
        assert len(snapshot) == 1
    finally:
        store.close()


# ── model routing ─────────────────────────────────────────────────────────


# Routing table. A mismatch here means an agent's MODEL_PREFERENCE was flipped
# without updating the table — fail loudly rather than let the table rot.
#
# Model tiers: the Extractor, RoundPlanner and Selector are chat-tier; the
# UsefulnessGate, Reconciler, Resolver, Mapper (with its MappingReviewer, both in
# `mapper.py`) and KGReviewer are reasoning-tier.
_EXPECTED_ROUTING: dict[str, str] = {
    "extractor": "chat",
    "round_planner": "chat",
    "selector": "chat",
    "usefulness_gate": "reasoning",
    "reconciler": "reasoning",
    "resolver": "reasoning",
    "mapper": "reasoning",
    "kg_reviewer": "reasoning",
}


def test_routing_table_default():
    """Every agent module declares MODEL_PREFERENCE matching the table."""

    import importlib

    for mod_short, expected in _EXPECTED_ROUTING.items():
        mod = importlib.import_module(f"memex.kg_intelligence.agents.{mod_short}")
        actual = getattr(mod, "MODEL_PREFERENCE", None)
        assert actual == expected, (
            f"{mod_short}.MODEL_PREFERENCE={actual!r}, expected {expected!r}"
        )


def test_build_llm_model_resolves_chat_vs_reasoning(monkeypatch):
    """build_llm_model(settings, preference) hits the right model name.

    Covers routing resolution and observability model_name —
    the resolved model_name on the returned Model is what the OpenAI
    response surfaces back to ``run_agent_or_fallback.model_name``.
    """

    from dataclasses import replace

    from memex.kg_intelligence.agents import base as agents_base

    settings = replace(
        _settings(),
        chat_model="MODEL-CHAT",
        reasoning_model="MODEL-REASONING",
    )

    captured: list[tuple] = []

    def fake_build(api_key: str, base_url: str, model_name: str, temperature=None):
        captured.append((api_key, base_url, model_name, temperature))
        return TestModel()

    monkeypatch.setattr(agents_base, "_build_model", fake_build)

    agents_base.build_llm_model(settings, "chat")
    agents_base.build_llm_model(settings, "reasoning")
    agents_base.build_llm_model(settings)  # default == "chat"

    # all models pinned to temperature 0 (faithfulness tasks, not creative).
    assert captured == [
        ("test-key", "https://llm.test/v1", "MODEL-CHAT", 0.0),
        ("test-key", "https://llm.test/v1", "MODEL-REASONING", 0.0),
        ("test-key", "https://llm.test/v1", "MODEL-CHAT", 0.0),
    ]


@pytest.mark.asyncio
async def test_execute_agent_call_passes_preference_to_model_builder(
    monkeypatch, trace, tmp_path
):
    """When ``llm_model=None``, ``execute_agent_call`` resolves the
    model via ``build_llm_model(settings, model_preference)``.
    Explicit ``llm_model=`` overrides bypass the resolver."""

    from dataclasses import replace

    from memex.kg_intelligence.agents import base as agents_base

    settings = replace(
        _settings(),
        chat_model="MODEL-CHAT",
        reasoning_model="MODEL-REASONING",
    )

    captured_model_names: list[str] = []

    def fake_build(api_key: str, base_url: str, model_name: str, temperature=None):
        captured_model_names.append(model_name)
        return TestModel(custom_output_args={"topics": []})

    monkeypatch.setattr(agents_base, "_build_model", fake_build)

    sinks = AgentCallSinks(trace=trace)

    # The Resolver wrapper isn't useful here (it has its own non-trivial
    # input); call RoundPlanner with each preference instead.
    from memex.kg_intelligence.agents.round_planner import (
        _SYSTEM_PROMPT,
        ACTION,
        AGENT_NAME,
        planner_agent,
    )
    from memex.kg_intelligence.agents.schemas import RoundPlan

    prompt = "ignored under TestModel"
    sample = RoundPlan()

    for pref in ("chat", "reasoning"):
        captured_model_names.clear()
        await agents_base.execute_agent_call(
            agent=planner_agent,
            user_prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            sample_factory=lambda s=sample: s,
            log_prefix=AGENT_NAME,
            settings=settings,
            llm_model=None,
            sinks=sinks,
            agent_name=AGENT_NAME,
            action=ACTION,
            triggered_by="test",
            target_agent="adapters",
            input_summary="test",
            output_summary_fn=lambda r: "ok",
            confidence=0.5,
            model_preference=pref,
        )
        expected = "MODEL-REASONING" if pref == "reasoning" else "MODEL-CHAT"
        assert captured_model_names == [expected], (
            f"preference={pref!r}: built {captured_model_names}, expected [{expected!r}]"
        )

    # Override path: passing llm_model= bypasses the resolver entirely.
    captured_model_names.clear()
    await agents_base.execute_agent_call(
        agent=planner_agent,
        user_prompt=prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=lambda s=sample: s,
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=TestModel(custom_output_args=sample.model_dump()),
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=ACTION,
        triggered_by="test",
        target_agent="adapters",
        input_summary="test",
        output_summary_fn=lambda r: "ok",
        confidence=0.5,
        model_preference="reasoning",
    )
    assert captured_model_names == [], (
        "explicit llm_model= must bypass build_llm_model"
    )
