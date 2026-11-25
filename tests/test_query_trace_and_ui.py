"""Tests for the QueryTraceStore (AgentCallSinks-compatible) + the query UI API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memex.observability.query_store import QueryTraceStore
from memex.ui.query_ui import create_app


@pytest.fixture
def kg_root(tmp_path: Path) -> Path:
    return tmp_path / "demo_kg"


def _seed_run(store: QueryTraceStore):
    run = store.start_run(
        run_id="run_1", kind="chat", kg_id="demo_kg", question="why?",
        config={"max_revisions": 1}, sample_mode=False,
    )
    # parent agent call + a tool child + a sibling agent
    parent = store.begin_agent_call(
        run, agent_name="Retriever", action="retrieve", triggered_by="loop",
        target_agent="Answerer", input_summary="why?",
    )
    store.record_tool_call(
        run, parent_call_id=parent.agent_call_id, tool_name="search_facts",
        args={"query": "why"}, result_summary="[3 hits]",
    )
    store.record_steps(parent.agent_call_id, [
        {"kind": "thinking", "content": "let me search"},
        {"kind": "tool_call", "tool_name": "search_facts", "args": {"query": "why"}},
        {"kind": "tool_return", "tool_name": "search_facts", "result": "[3 hits]"},
    ])
    store.finish_agent_call(
        parent, status="success", duration_ms=5, output_summary="3 facts",
        confidence=0.8, prompt_tokens=10, completion_tokens=5,
        parsed_output={"reasoning": "searched then selected", "selected": ["relation:1"]},
    )
    answerer = store.begin_agent_call(
        run, agent_name="Answerer", action="answer", triggered_by="loop",
        target_agent="GroundingChecker", parent_call_id=parent.agent_call_id, input_summary="why?",
    )
    store.finish_agent_call(answerer, status="success", duration_ms=4, output_summary="answered")
    store.finish_run(run, status="success", summary="done")
    return run


def test_store_roundtrip_and_graph(kg_root: Path) -> None:
    store = QueryTraceStore.open(kg_root)
    run = _seed_run(store)

    runs = store.list_runs()
    assert len(runs) == 1 and runs[0]["kind"] == "chat"

    calls = store.list_agent_calls(run.run_db_id)
    names = [(c["agent_name"], c["parent_call_id"], c["is_tool_call"]) for c in calls]
    assert ("Retriever", None, 0) in names
    # the tool child + the Answerer both parent onto the Retriever (a tree, not a chain)
    assert ("search_facts", calls[0]["id"], 1) in names
    assert any(n[0] == "Answerer" and n[1] == calls[0]["id"] for n in names)

    parent_id = calls[0]["id"]
    steps = store.list_steps(parent_id)
    assert [s["kind"] for s in steps] == ["thinking", "tool_call", "tool_return"]

    detail = store.get_agent_call(parent_id)
    assert detail["parsed_output_json"] is not None
    store.close()


def test_query_ui_api(kg_root: Path) -> None:
    store = QueryTraceStore.open(kg_root)
    _seed_run(store)
    store.close()

    app = create_app(kg_root=kg_root.parent)
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert "demo_kg" in client.get("/api/kg_ids").json()["kg_ids"]

    runs = client.get("/api/demo_kg/runs").json()["runs"]
    assert len(runs) == 1
    run_payload = client.get(f"/api/demo_kg/runs/{runs[0]['id']}").json()
    assert len(run_payload["calls"]) == 3  # Retriever + tool child + Answerer

    parent_id = run_payload["calls"][0]["id"]
    call_payload = client.get(f"/api/demo_kg/calls/{parent_id}").json()
    assert call_payload["call"]["parsed_output"]["reasoning"] == "searched then selected"
    assert len(call_payload["steps"]) == 3


def test_query_ui_unknown_kg_404(kg_root: Path) -> None:
    kg_root.mkdir(parents=True)
    app = create_app(kg_root=kg_root.parent)
    client = TestClient(app)
    assert client.get("/api/demo_kg/runs").status_code == 404
