"""Tests for the VocabResolver: candidate clustering (pure, store-agnostic) and
the non-destructive commit of alias / parent verdicts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from memex.kg_intelligence.agents.schemas import VocabVerdict
from memex.kg_intelligence.agents.vocab_resolver import (
    candidate_groups,
    commit_clusters,
)
from memex.observability import TraceStore


def _unit(rows: list[list[float]]) -> np.ndarray:
    m = np.array(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


# ── candidate_groups (pure) ───────────────────────────────────────────────


def test_candidate_groups_star_clusters_by_frequency() -> None:
    names = ["uses", "utilizes", "produces", "generates", "lonely"]
    matrix = _unit([
        [1.0, 0.0, 0.0],    # uses
        [0.98, 0.2, 0.0],   # utilizes ~ uses
        [0.0, 1.0, 0.0],    # produces
        [0.0, 0.97, 0.24],  # generates ~ produces
        [0.0, 0.0, 1.0],    # lonely — far from all
    ])
    freqs = {"uses": 100, "utilizes": 5, "produces": 50, "generates": 3, "lonely": 1}

    groups = candidate_groups(
        names, matrix, freqs, cosine_threshold=0.6, top_k=5, max_group=8
    )
    by_head = dict(groups)

    # Frequent terms head; their near-neighbours are claimed as members.
    assert "uses" in by_head and "produces" in by_head
    assert "utilizes" not in by_head and "generates" not in by_head
    assert any(m == "utilizes" for m, _ in by_head["uses"])

    # A term with no neighbour above threshold is neither head nor member.
    members = {m for _, ms in groups for m, _ in ms}
    assert "lonely" not in members and "lonely" not in by_head


def test_candidate_groups_disjoint_and_capped() -> None:
    names = [f"p{i}" for i in range(5)]
    matrix = _unit([[1.0, i * 0.001, 0.0] for i in range(5)])  # near-identical
    freqs = {n: 5 - i for i, n in enumerate(names)}  # p0 most frequent

    groups = candidate_groups(
        names, matrix, freqs, cosine_threshold=0.5, top_k=8, max_group=3
    )

    seen: list[str] = []
    for head, members in groups:
        seen.append(head)
        seen.extend(m for m, _ in members)
    assert len(seen) == len(set(seen))  # disjoint: each term in ≤1 cluster
    assert all(1 + len(ms) <= 3 for _, ms in groups)  # capped at max_group


def test_candidate_groups_empty() -> None:
    assert candidate_groups([], np.empty((0, 0), dtype=np.float32), {}) == []


# ── commit_clusters (store) ───────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    s = TraceStore.open(tmp_path / "demo")
    yield s
    s.close()


def _stub_call(store: TraceStore) -> int:
    build = store.start_build(
        build_run_id="b", kg_id="demo", base_query="q", config={},
        sources=["arxiv"], sample_mode=True, resume_from_round=None,
    )
    round_ = store.start_round(
        build, round_number=1, entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    call = store.begin_agent_call(
        round_, agent_name="VocabResolver", action="cluster_vocab",
        triggered_by="test", input_summary="",
    )
    store.finish_agent_call(call, status="success", duration_ms=0, output_summary="")
    return call.agent_call_id


def test_commit_clusters_predicate_lane(store: TraceStore) -> None:
    cid = _stub_call(store)
    verdicts = [
        VocabVerdict(name="utilizes", canonical="uses", relation="alias", confidence=0.9),
        VocabVerdict(name="co_produces", canonical="produces", relation="child_of", confidence=0.9),
        VocabVerdict(name="opposes", canonical="supports", relation="unrelated", confidence=0.9),
    ]
    counts = commit_clusters(
        store, verdicts, kind="predicate", agent_call_id=cid, round_number=1
    )
    assert counts == {"alias": 1, "child_of": 1, "unrelated": 1}

    assert store.resolve_predicate_alias("utilizes") == "uses"
    assert ("co_produces", "produces") in store.list_schema_parents("relation_type")
    # unrelated persists nothing
    assert store.resolve_predicate_alias("opposes") is None
    assert ("opposes", "supports") not in store.list_schema_parents("relation_type")


def test_commit_clusters_property_lane_is_separate(store: TraceStore) -> None:
    cid = _stub_call(store)
    verdicts = [
        VocabVerdict(name="plant_capacity", canonical="capacity", relation="child_of", confidence=0.9),
        VocabVerdict(name="levelised_cost", canonical="cost", relation="alias", confidence=0.9),
    ]
    commit_clusters(store, verdicts, kind="property", agent_call_id=cid, round_number=1)

    assert store.resolve_property_alias("levelised_cost") == "cost"
    assert ("plant_capacity", "capacity") in store.list_schema_parents("property_type")
    # the property lane does not leak into the predicate lane
    assert store.resolve_predicate_alias("levelised_cost") is None
    assert ("plant_capacity", "capacity") not in store.list_schema_parents("relation_type")


def test_commit_clusters_skips_self_loop_parent(store: TraceStore) -> None:
    cid = _stub_call(store)
    verdicts = [VocabVerdict(name="uses", canonical="uses", relation="child_of", confidence=0.5)]
    commit_clusters(store, verdicts, kind="predicate", agent_call_id=cid, round_number=1)
    assert store.list_schema_parents("relation_type") == []
