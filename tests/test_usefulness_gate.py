"""Tests for the two-pass UsefulnessGate.

Covers: pass-1 keep → pass-2 summary/reflection merged (two agent calls);
pass-1 skip short-circuits pass-2 (one call, empty summary); SAMPLE_MODE
keep-all with a title-derived summary.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from memex.common.config import Settings
from memex.common.trace import TraceWriter
from memex.kg_intelligence.agents import AgentCallSinks
from memex.kg_intelligence.agents.usefulness_gate import assess_doc
from memex.kg_intelligence.state import CorpusState


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


def _sinks(tmp_path: Path) -> tuple[AgentCallSinks, TraceWriter]:
    trace = TraceWriter(request_id="gate-test", log_dir=tmp_path)
    return AgentCallSinks(trace=trace), trace


def _events(trace: TraceWriter) -> list[dict]:
    return [json.loads(line) for line in trace.path.read_text().splitlines() if line.strip()]


def _raises(_messages, _info):  # type: ignore[no-untyped-def]
    raise RuntimeError("simulated LLM failure")


async def test_keep_runs_two_passes_and_merges_summary(tmp_path: Path) -> None:
    sinks, trace = _sinks(tmp_path)
    stub = TestModel(
        custom_output_args={
            "usefulness": 0.8,
            "decision": "keep",
            "reason": "on-topic",
            "summary": "A study of iridium catalysts.",
            "entities_seen": ["iridium"],
            "subtopics_covered": ["catalysis"],
            "subtopics_missing": [],
            "borders": [],
        }
    )
    result = await assess_doc(
        doc_id="arxiv:x",
        source="arxiv",
        title="Iridium catalysts",
        whole_doc="Iridium oxide is an OER catalyst...",
        corpus_state=CorpusState(base_query="green hydrogen"),
        settings=_settings(),
        sinks=sinks,
        llm_model=stub,
    )
    assert result.decision == "keep"
    assert result.usefulness == 0.8
    assert result.summary == "A study of iridium catalysts."
    assert result.entities_seen == ["iridium"]
    # both passes ran → two agent-call trace events
    assert len(_events(trace)) == 2


async def test_skip_short_circuits_second_pass(tmp_path: Path) -> None:
    sinks, trace = _sinks(tmp_path)
    stub = TestModel(
        custom_output_args={
            "usefulness": 0.1,
            "decision": "skip",
            "reason": "coincidental keyword overlap",
        }
    )
    result = await assess_doc(
        doc_id="arxiv:y",
        source="arxiv",
        title="PCR thermal cycling in water baths",
        whole_doc="Polymerase chain reaction protocols...",
        corpus_state=CorpusState(base_query="green hydrogen"),
        settings=_settings(),
        sinks=sinks,
        llm_model=stub,
    )
    assert result.decision == "skip"
    assert result.summary == ""
    # only pass 1 ran (pass 2 skipped on a drop)
    assert len(_events(trace)) == 1


async def test_sample_mode_keeps_all_with_title_summary(tmp_path: Path) -> None:
    sinks, _trace = _sinks(tmp_path)
    result = await assess_doc(
        doc_id="arxiv:z",
        source="arxiv",
        title="Photocatalytic water splitting iridium",
        whole_doc="body text",
        corpus_state=CorpusState(base_query="green hydrogen"),
        settings=_settings(sample_mode=True),
        sinks=sinks,
        llm_model=FunctionModel(_raises),  # would raise if SAMPLE_MODE didn't short-circuit
    )
    assert result.decision == "keep"
    assert result.summary
    assert any("iridium" in e.lower() for e in result.entities_seen)
