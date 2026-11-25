"""CLI smoke for ``memex inspect``.

Seeds a tiny build into a tmp kg_root, then exercises every view via the
script's ``main(argv, ...)`` entry point. Capsys captures stdout; we
assert on the structured shape rather than exact whitespace so the views
can evolve cosmetically without breaking the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memex.cli import inspect_build as kg_inspect
from memex.observability import TraceStore


def _seed(kg_root: Path) -> TraceStore:
    store = TraceStore.open(kg_root)
    build = store.start_build(
        build_run_id="kg_build_demo_1",
        kg_id="demo",
        base_query="green hydrogen",
        config={"max_rounds": 2},
        sources=["arxiv"],
        sample_mode=True,
        resume_from_round=None,
    )
    r = store.start_round(
        build, round_number=1,
        entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    c = store.begin_agent_call(
        r, agent_name="RoundPlanner", action="plan_round",
        triggered_by="round_start", target_agent="adapters",
        input_summary="round=1 base='green hydrogen'",
    )
    store.finish_agent_call(
        c, status="sample_mode", duration_ms=12,
        output_summary="arxiv=2",
        system_prompt="You are RoundPlanner.",
        user_prompt="Base query: green hydrogen",
        parsed_output={"queries": {"arxiv": ["x", "y"]}},
    )
    bad = store.begin_agent_call(
        r, agent_name="Extractor", action="extract_doc",
        triggered_by="per_doc", input_summary="doc_id=Y",
    )
    store.finish_agent_call(
        bad, status="failed", duration_ms=99,
        output_summary="(fallback)",
        error_type="RuntimeError", error_message="simulated LLM failure",
    )
    store.record_fetched_doc(
        r, source="arxiv", query="x", url="https://x", title="Doc X",
        doc_id="doc_x", outcome="selected_and_fetched",
        parse_method="pdfminer.six", cleaned_text_chars=500,
    )
    store.record_schema_change(
        r, agent_call_id=c.agent_call_id, event="proposed",
        type_kind="entity_type", name="process", rationale="x",
    )
    store.finish_round(
        r,
        entities_after=2, relations_after=0,
        schema_entity_types_after=1, schema_relation_types_after=0,
        docs_fetched=1, docs_skipped=0, docs_failed=0,
        stop_criterion_fired="", duration_seconds=4.2,
    )
    store.finish_build(
        build,
        stop_reason="max_rounds", rounds_completed=1,
        total_docs=1, total_entities=2, total_relations=0,
        schema_entity_types=1, schema_relation_types=0,
    )
    return store


@pytest.fixture
def seeded(tmp_path: Path) -> Path:
    kg_root = tmp_path / "demo"
    store = _seed(kg_root)
    store.close()
    return tmp_path  # parent containing 'demo'


def test_overview_shows_build_and_rounds(capsys: pytest.CaptureFixture, seeded: Path) -> None:
    rc = kg_inspect.main(["demo", "--kg-root", str(seeded)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "build #1" in out
    assert "green hydrogen" in out
    assert "rounds" in out
    assert "agent calls" in out


def test_round_view_lists_calls(capsys: pytest.CaptureFixture, seeded: Path) -> None:
    rc = kg_inspect.main(["demo", "--kg-root", str(seeded), "--round", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "RoundPlanner" in out
    assert "Extractor" in out
    assert "RuntimeError: simulated LLM failure" in out


def test_call_drilldown_shows_full_prompts(capsys: pytest.CaptureFixture, seeded: Path) -> None:
    rc = kg_inspect.main(["demo", "--kg-root", str(seeded), "--call", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "You are RoundPlanner" in out
    assert "Base query: green hydrogen" in out
    assert "parsed_output" in out


def test_failures_view(capsys: pytest.CaptureFixture, seeded: Path) -> None:
    rc = kg_inspect.main(["demo", "--kg-root", str(seeded), "--failures"])
    assert rc == 0
    out = capsys.readouterr().out
    # both sample_mode AND failed surface
    assert "sample_mode" in out
    assert "failed" in out


def test_schema_changes_view(capsys: pytest.CaptureFixture, seeded: Path) -> None:
    rc = kg_inspect.main(["demo", "--kg-root", str(seeded), "--schema-changes"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "process" in out
    assert "proposed" in out


def test_json_mode_emits_parseable_json(capsys: pytest.CaptureFixture, seeded: Path) -> None:
    rc = kg_inspect.main(["demo", "--kg-root", str(seeded), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["build"]["kg_id"] == "demo"
    assert len(payload["rounds"]) == 1


def test_missing_db_returns_nonzero(capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    rc = kg_inspect.main(["does_not_exist", "--kg-root", str(tmp_path)])
    assert rc == 1
