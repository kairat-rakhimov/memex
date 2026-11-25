"""FastAPI test client smoke for the trace UI's REST surface."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memex.observability import TraceStore
from memex.ui.trace_ui import create_app


def _seed(kg_root: Path) -> None:
    store = TraceStore.open(kg_root)
    build = store.start_build(
        build_run_id="kg_build_alpha_1",
        kg_id="alpha",
        base_query="green hydrogen",
        config={"max_rounds": 1},
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
        triggered_by="round_start", input_summary="round=1",
    )
    store.finish_agent_call(
        c, status="success", duration_ms=10, output_summary="arxiv=2",
        system_prompt="You are RoundPlanner",
        user_prompt="Base query: green hydrogen",
        parsed_output={"queries": {"arxiv": ["x"]}},
    )
    store.record_fetched_doc(
        r, source="arxiv", query="x", url="https://x", title="Doc",
        doc_id="doc_1", outcome="selected_and_fetched",
        parse_method="pdfminer.six", cleaned_text_chars=10,
    )
    store.record_schema_change(
        r, agent_call_id=c.agent_call_id, event="proposed",
        type_kind="entity_type", name="process", rationale="x",
    )
    # KG snapshot — needed for graph + entity endpoints
    a = store.upsert_kg_entity(canonical_name="iridium", entity_type="material",
        mentions=3, degree=1, first_seen_round=1, last_seen_round=1,
        created_by_call_id=c.agent_call_id,
    )
    b = store.upsert_kg_entity(canonical_name="OER", entity_type="process",
        mentions=2, degree=1, first_seen_round=1, last_seen_round=1,
        created_by_call_id=c.agent_call_id,
    )
    store.record_kg_relation(subject_entity_id=a, predicate="catalyzes",
        object_entity_id=b, source_doc_id="doc_1",
        round_observed=1, agent_call_id=c.agent_call_id,
        created_by_call_id=c.agent_call_id,
    )
    # Two seeded kg_mutations rows let the mutation-endpoints test
    # exercise all three UI lenses without needing to reach into the
    # store from the test body.
    parent_mut = store.record_kg_mutation(
        kind="merge", target_kind="entity", target_id="iridium",
        before={"name": "Iridium"}, after={"name": "iridium"},
        reason="tier-1 normalization", confidence=1.0,
        agent_call_id=c.agent_call_id, round=1,
    )
    store.record_kg_mutation(
        kind="endpoint_rewrite", target_kind="relation", target_id="42",
        before={"subject": "Iridium"}, after={"subject": "iridium"},
        reason="endpoint rewrite", confidence=None,
        agent_call_id=c.agent_call_id,
        parent_mutation_id=parent_mut, round=1,
    )
    store.finish_round(
        r,
        entities_after=2, relations_after=0,
        schema_entity_types_after=1, schema_relation_types_after=0,
        docs_fetched=1, docs_skipped=0, docs_failed=0,
        stop_criterion_fired="max_rounds", duration_seconds=1.0,
    )
    store.finish_build(
        build,
        stop_reason="max_rounds", rounds_completed=1,
        total_docs=1, total_entities=2, total_relations=0,
        schema_entity_types=1, schema_relation_types=0,
    )
    store.close()


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    kg_root = tmp_path
    (kg_root / "alpha").mkdir()
    _seed(kg_root / "alpha")
    return TestClient(create_app(kg_root=kg_root))


def test_kg_ids_discovers_seeded_kg(client: TestClient) -> None:
    r = client.get("/api/kg_ids")
    assert r.status_code == 200
    assert "alpha" in r.json()["kg_ids"]


def test_builds_endpoint_returns_seeded_build(client: TestClient) -> None:
    r = client.get("/api/alpha/builds")
    assert r.status_code == 200
    builds = r.json()["builds"]
    assert len(builds) == 1
    assert builds[0]["base_query"] == "green hydrogen"


def test_build_detail_returns_rounds_and_schema_changes(client: TestClient) -> None:
    build_id = client.get("/api/alpha/builds").json()["builds"][0]["id"]
    r = client.get(f"/api/alpha/builds/{build_id}")
    assert r.status_code == 200
    j = r.json()
    assert len(j["rounds"]) == 1
    assert len(j["schema_changes"]) == 1
    assert j["schema_changes"][0]["name"] == "process"


def test_agent_calls_endpoint(client: TestClient) -> None:
    build_id = client.get("/api/alpha/builds").json()["builds"][0]["id"]
    rounds = client.get(f"/api/alpha/builds/{build_id}").json()["rounds"]
    round_id = rounds[0]["id"]
    r = client.get(f"/api/alpha/rounds/{round_id}/agent_calls")
    assert r.status_code == 200
    j = r.json()
    assert len(j["agent_calls"]) == 1
    assert j["agent_calls"][0]["agent_name"] == "RoundPlanner"
    assert len(j["fetched_docs"]) == 1


def test_agent_call_detail_inlines_parsed_output(client: TestClient) -> None:
    build_id = client.get("/api/alpha/builds").json()["builds"][0]["id"]
    rounds = client.get(f"/api/alpha/builds/{build_id}").json()["rounds"]
    round_id = rounds[0]["id"]
    calls = client.get(f"/api/alpha/rounds/{round_id}/agent_calls").json()["agent_calls"]
    call_id = calls[0]["id"]
    r = client.get(f"/api/alpha/agent_calls/{call_id}")
    assert r.status_code == 200
    call = r.json()["agent_call"]
    assert call["system_prompt"].startswith("You are RoundPlanner")
    # parsed_output_json is decoded into parsed_output by the API
    assert call["parsed_output"] == {"queries": {"arxiv": ["x"]}}


def test_unknown_kg_id_404s(client: TestClient) -> None:
    r = client.get("/api/no_such/builds")
    assert r.status_code == 404


def test_unknown_build_404s(client: TestClient) -> None:
    r = client.get("/api/alpha/builds/9999")
    assert r.status_code == 404


def test_index_html_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "memex" in r.text
    assert "alpinejs" in r.text


def test_static_app_js_served(client: TestClient) -> None:
    """Entry script registers the Alpine traceUI component."""

    r = client.get("/static/js/app.js")
    assert r.status_code == 200
    assert "traceUI" in r.text


def test_graph_endpoint_returns_nodes_and_edges(client: TestClient) -> None:
    build_id = client.get("/api/alpha/builds").json()["builds"][0]["id"]
    r = client.get(f"/api/alpha/builds/{build_id}/graph")
    assert r.status_code == 200
    j = r.json()
    labels = {n["label"] for n in j["nodes"]}
    assert labels == {"iridium", "OER"}
    assert len(j["edges"]) == 1
    edge = j["edges"][0]
    assert edge["label"] == "catalyzes"
    # edge carries provenance hooks the UI uses to jump to the agent call
    assert "sample_agent_call_id" in edge
    assert edge["sample_doc_id"] == "doc_1"


def test_graph_endpoint_filters_isolated_by_default(client: TestClient) -> None:
    build_id = client.get("/api/alpha/builds").json()["builds"][0]["id"]
    # min_mentions=10 → no entity meets it
    r = client.get(f"/api/alpha/builds/{build_id}/graph?min_mentions=10")
    assert r.status_code == 200
    j = r.json()
    assert j["nodes"] == []
    assert j["edges"] == []


def test_entity_detail_endpoint(client: TestClient) -> None:
    build_id = client.get("/api/alpha/builds").json()["builds"][0]["id"]
    nodes = client.get(f"/api/alpha/builds/{build_id}/graph").json()["nodes"]
    iridium_id = next(n["id"] for n in nodes if n["label"] == "iridium")
    r = client.get(f"/api/alpha/builds/{build_id}/entities/{iridium_id}")
    assert r.status_code == 200
    entity = r.json()["entity"]
    assert entity["canonical_name"] == "iridium"
    # outgoing relation to OER surfaces
    assert any(o["object_name"] == "OER" for o in entity["outgoing"])
    assert "doc_1" in entity["source_doc_ids"]


def test_unknown_entity_404s(client: TestClient) -> None:
    build_id = client.get("/api/alpha/builds").json()["builds"][0]["id"]
    r = client.get(f"/api/alpha/builds/{build_id}/entities/99999")
    assert r.status_code == 404


def test_mutation_endpoints_serve_all_three_lenses(client: TestClient) -> None:
    """Per-target / per-round / per-agent-call APIs back the UI mutation panel."""

    build_id = client.get("/api/alpha/builds").json()["builds"][0]["id"]
    round_id = client.get(
        f"/api/alpha/builds/{build_id}"
    ).json()["rounds"][0]["id"]
    call_id = client.get(
        f"/api/alpha/rounds/{round_id}/agent_calls"
    ).json()["agent_calls"][0]["id"]

    # Lens 1: per-target
    r = client.get(
        f"/api/alpha/builds/{build_id}/mutations?target_kind=entity&target_id=iridium"
    )
    assert r.status_code == 200
    j = r.json()
    assert len(j["mutations"]) == 1
    assert j["mutations"][0]["kind"] == "merge"
    assert j["mutations"][0]["confidence"] == 1.0

    # Lens 2: per-round counts
    r = client.get(f"/api/alpha/builds/{build_id}/mutations_by_round")
    assert r.status_code == 200
    counts = {(c["round"], c["kind"]): c["n"] for c in r.json()["counts"]}
    assert counts == {(1, "merge"): 1, (1, "endpoint_rewrite"): 1}

    # Lens 3: per-agent-call (parent merge before its endpoint-rewrite child)
    r = client.get(
        f"/api/alpha/builds/{build_id}/agent_calls/{call_id}/mutations"
    )
    assert r.status_code == 200
    rows = r.json()["mutations"]
    assert [m["kind"] for m in rows] == ["merge", "endpoint_rewrite"]
    assert rows[1]["parent_mutation_id"] == rows[0]["id"]


def test_index_html_includes_vis_network(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "vis-network" in r.text
    assert "graph-canvas" in r.text


def test_vendored_assets_served_locally(client: TestClient) -> None:
    """All UI deps are vendored under /static/vendor/ so the UI works
    offline. Assert each is served from the local server, not a CDN."""

    for path, sniff in [
        ("/static/vendor/tailwindcss.min.js", "Object.create"),
        ("/static/vendor/alpinejs.min.js", "var "),
        ("/static/vendor/vis-network.min.js", "vis-network"),
        ("/static/vendor/vis-network.min.css", ".vis-"),
    ]:
        r = client.get(path)
        assert r.status_code == 200, f"{path} not served (got {r.status_code})"
        assert sniff in r.text, f"{path} content sniff {sniff!r} not found"

    # No remaining src=/href= refs to external CDNs in index.html.
    # (Comment lines that document the upstream URLs are fine.)
    import re

    html = client.get("/").text
    external_refs = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    assert external_refs == [], f"unexpected external refs: {external_refs}"


def test_static_js_modules_serve_with_handlers(client: TestClient) -> None:
    """The JS is split across multiple files (formatters, api, mutations,
    graph, app). Each must serve, and the handler symbols must land in the
    right module: graph entry points in graph.js, the entry app.js drives
    selectEntity / selectCall / selectBuild."""

    graph = client.get("/static/js/graph.js")
    assert graph.status_code == 200
    for handler in ("loadGraph", "renderGraph", "onGraphClick", "focusGraphNode", "fitGraph"):
        assert handler in graph.text, f"{handler} missing from graph.js"

    app = client.get("/static/js/app.js")
    assert app.status_code == 200
    for handler in ("selectEntity", "selectCall", "selectBuild", "bootstrap"):
        assert handler in app.text, f"{handler} missing from app.js"

    mutations = client.get("/static/js/mutations.js")
    assert mutations.status_code == 200
    assert "loadMutationsByRound" in mutations.text
    assert "roundMutationSummary" in mutations.text
