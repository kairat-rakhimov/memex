"""Tests for the per-build trace store.

Covers schema apply, every write method, the read APIs the UI + CLI
inspector lean on, and the sub-round visibility guarantee — a
``begin_agent_call`` that never sees a matching ``finish_agent_call``
must still leave a "started" row behind so a crash post-mortem is
possible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memex.observability import TraceStore


@pytest.fixture
def kg_root(tmp_path: Path) -> Path:
    return tmp_path / "demo"


@pytest.fixture
def store(kg_root: Path) -> TraceStore:
    s = TraceStore.open(kg_root)
    yield s
    s.close()


def _start_build(store: TraceStore, **overrides):
    defaults = {
        "build_run_id": "kg_build_demo_1",
        "kg_id": "demo",
        "base_query": "green hydrogen",
        "config": {"max_rounds": 2},
        "sources": ["arxiv", "wikipedia"],
        "sample_mode": True,
        "resume_from_round": None,
    }
    defaults.update(overrides)
    return store.start_build(**defaults)


def _stub_call(store: TraceStore, build, *, round_number: int = 1) -> int:
    """Create a stub round + skipped agent_call row, return its id.

    kg_entities.created_by_call_id and
    kg_relations.created_by_call_id are NOT NULL. Test sites that
    bypass the pipeline and write kg-data rows directly need a real
    agent_calls row to point at.
    """

    round_ = store.start_round(
        build, round_number=round_number,
        entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    call = store.begin_agent_call(
        round_, agent_name="_stub", action="stub",
        triggered_by="test_setup", input_summary="",
    )
    store.finish_agent_call(call, status="skipped", duration_ms=0, output_summary="")
    return call.agent_call_id


def test_apply_schema_is_idempotent(kg_root: Path) -> None:
    s1 = TraceStore.open(kg_root)
    s1.close()
    s2 = TraceStore.open(kg_root)  # second open re-applies; must not error
    assert s2.list_builds() == []
    s2.close()


def test_db_lands_at_kg_root_kg_db(kg_root: Path) -> None:
    store = TraceStore.open(kg_root)
    assert store.db_path == kg_root / "kg.db"
    assert store.db_path.exists()
    store.close()


def test_build_round_agent_call_round_trip(store: TraceStore) -> None:
    build = _start_build(store)
    round_ = store.start_round(
        build,
        round_number=1,
        entities_before=0,
        relations_before=0,
        schema_entity_types_before=0,
        schema_relation_types_before=0,
    )
    call = store.begin_agent_call(
        round_,
        agent_name="RoundPlanner",
        action="plan_round",
        triggered_by="round_start",
        target_agent="adapters",
        input_summary="round=1",
    )
    store.finish_agent_call(
        call,
        status="success",
        duration_ms=1234,
        output_summary="arxiv=2",
        confidence=0.8,
        model_name="claude-sonnet",
        prompt_tokens=120,
        completion_tokens=42,
        system_prompt="You are RoundPlanner",
        user_prompt="Base query: green hydrogen",
        raw_response='{"queries":{"arxiv":["x"]}}',
        parsed_output={"queries": {"arxiv": ["x"]}},
    )
    store.finish_round(
        round_,
        entities_after=5,
        relations_after=2,
        schema_entity_types_after=1,
        schema_relation_types_after=0,
        docs_fetched=2,
        docs_skipped=0,
        docs_failed=0,
        stop_criterion_fired="",
        duration_seconds=12.3,
    )
    store.finish_build(
        build,
        stop_reason="max_rounds",
        rounds_completed=1,
        total_docs=2,
        total_entities=5,
        total_relations=2,
        schema_entity_types=1,
        schema_relation_types=0,
    )

    builds = store.list_builds()
    assert len(builds) == 1
    assert builds[0]["stop_reason"] == "max_rounds"
    # finish_build rolls per-round tokens up
    assert builds[0]["total_prompt_tokens"] == 120
    assert builds[0]["total_completion_tokens"] == 42

    [round_row] = store.list_rounds(build.build_id)
    assert round_row["entities_after"] == 5
    assert round_row["prompt_tokens"] == 120  # finish_round rolls call tokens up

    [call_row] = store.list_agent_calls(round_.round_id)
    assert call_row["agent_name"] == "RoundPlanner"
    assert call_row["status"] == "success"
    assert call_row["duration_ms"] == 1234

    detail = store.get_agent_call(call.agent_call_id)
    assert detail["system_prompt"].startswith("You are RoundPlanner")
    assert json.loads(detail["parsed_output_json"])["queries"]["arxiv"] == ["x"]


def test_begin_without_finish_leaves_started_row(store: TraceStore) -> None:
    """Pipeline crashed mid-call → 'started' row must remain for forensics."""

    build = _start_build(store, build_run_id="kg_build_demo_2")
    round_ = store.start_round(
        build,
        round_number=1,
        entities_before=0,
        relations_before=0,
        schema_entity_types_before=0,
        schema_relation_types_before=0,
    )
    store.begin_agent_call(
        round_,
        agent_name="Extractor",
        action="extract_doc",
        triggered_by="per_doc",
        input_summary="doc_id=X",
    )
    rows = store.list_agent_calls(round_.round_id)
    assert len(rows) == 1
    assert rows[0]["status"] == "started"
    assert rows[0]["finished_at"] is None


def test_sequence_in_round_is_monotonic(store: TraceStore) -> None:
    build = _start_build(store, build_run_id="kg_build_demo_3")
    round_ = store.start_round(
        build,
        round_number=1,
        entities_before=0,
        relations_before=0,
        schema_entity_types_before=0,
        schema_relation_types_before=0,
    )
    for name in ["A", "B", "C"]:
        store.begin_agent_call(
            round_, agent_name=name, action="x",
            triggered_by="round_start", input_summary="",
        )
    rows = store.list_agent_calls(round_.round_id)
    assert [r["sequence_in_round"] for r in rows] == [1, 2, 3]


def test_fetched_docs_chunks_extractions_link_through(store: TraceStore) -> None:
    build = _start_build(store, build_run_id="kg_build_demo_4")
    round_ = store.start_round(
        build, round_number=1,
        entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    fetched_id = store.record_fetched_doc(
        round_,
        source="arxiv",
        query="iridium",
        url="https://arxiv.org/abs/2401.00001",
        title="Iridium catalysts",
        doc_id="doc_abc",
        outcome="selected_and_fetched",
        parse_method="pdfminer.six",
        cleaned_text_chars=1200,
        openalex_enriched=True,
        ranker_position=1,
    )
    chunk_id = store.record_chunk(
        fetched_doc_id=fetched_id,
        doc_id="doc_abc",
        chunk_index=0,
        char_start=0,
        char_end=1200,
        text_chars=1200,
    )

    call = store.begin_agent_call(
        round_, agent_name="Extractor", action="extract_doc",
        triggered_by="per_doc", input_summary="",
    )
    store.finish_agent_call(call, status="success", duration_ms=10, output_summary="ok")
    store.record_extraction(
        agent_call_id=call.agent_call_id,
        doc_id="doc_abc",
        chunk_id=chunk_id,
        entities_extracted=4,
        relations_extracted=2,
        relations_dropped_dangling=1,
    )

    docs = store.list_fetched_docs(round_.round_id)
    assert docs[0]["openalex_enriched"] == 1
    assert docs[0]["outcome"] == "selected_and_fetched"


def test_fetch_outcome_summary_rolls_up_by_outcome_and_source(store: TraceStore) -> None:
    # Degrade-visibility rollup — what the build-end summary surfaces.
    build = _start_build(store, build_run_id="kg_build_fos")
    round_ = store.start_round(
        build, round_number=1,
        entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )

    def _rec(source: str, outcome: str) -> None:
        store.record_fetched_doc(
            round_, source=source, query="q", url="u", title="t",
            doc_id=None, outcome=outcome,
        )

    _rec("arxiv", "selected_and_fetched")
    _rec("arxiv", "selected_and_fetched")
    _rec("arxiv", "fetch_failed")
    _rec("gdelt", "fetch_failed")
    _rec("gdelt", "skipped_duplicate")

    s = store.fetch_outcome_summary()
    assert s["fetched"] == 2
    assert s["fetch_failed"] == 2
    assert s["fetch_failed_by_source"] == {"arxiv": 1, "gdelt": 1}
    assert s["by_outcome"]["skipped_duplicate"] == 1
    # rate is over ATTEMPTED (fetched + failed = 4), not total rows (skips excluded).
    assert s["fetch_failed_rate"] == round(2 / 4, 3)


def test_fetch_outcome_summary_empty_build_is_zeroed(store: TraceStore) -> None:
    s = store.fetch_outcome_summary()
    assert s["fetched"] == 0 and s["fetch_failed"] == 0
    assert s["fetch_failed_rate"] == 0.0
    assert s["fetch_failed_by_source"] == {}


def test_record_entity_chunk_ref_writes_and_is_idempotent(store: TraceStore) -> None:
    """``entity_chunk_refs`` rows must be idempotent.

    ``merge_extraction`` calls ``record_entity_chunk_ref`` for every
    (entity, chunk) pair it touches; without ``INSERT OR IGNORE`` +
    sentinel char offsets, NULL-vs-NULL semantics in SQLite UNIQUE
    constraints would stack duplicates on re-fires.
    """

    build = _start_build(store, build_run_id="kg_build_ecr_idem")
    round_ = store.start_round(
        build, round_number=1,
        entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    fetched_id = store.record_fetched_doc(
        round_,
        source="arxiv",
        query="x",
        url="https://example/1",
        title="t",
        doc_id="d1",
        outcome="selected_and_fetched",
        parse_method="pdfminer.six",
        cleaned_text_chars=1000,
        ranker_position=1,
    )
    chunk_id = store.record_chunk(
        fetched_doc_id=fetched_id,
        doc_id="d1",
        chunk_index=0,
        char_start=0,
        char_end=1000,
        text_chars=1000,
    )
    call = store.begin_agent_call(
        round_, agent_name="_stub", action="stub",
        triggered_by="test_setup", input_summary="",
    )
    store.finish_agent_call(call, status="skipped", duration_ms=0, output_summary="")
    entity_id = store.upsert_kg_entity(
        canonical_name="iridium",
        entity_type="material",
        mentions=1,
        degree=0,
        first_seen_round=1,
        last_seen_round=1,
        created_by_call_id=call.agent_call_id,
    )

    first = store.record_entity_chunk_ref(entity_id=entity_id, chunk_id=chunk_id)
    second = store.record_entity_chunk_ref(entity_id=entity_id, chunk_id=chunk_id)
    third = store.record_entity_chunk_ref(entity_id=entity_id, chunk_id=chunk_id)

    assert first is not None
    # Subsequent inserts are no-ops; RETURNING gives None on conflict.
    assert second is None
    assert third is None

    # Exactly one row exists in the table for this (entity, chunk).
    rows = store._conn.execute(
        "SELECT COUNT(*) FROM entity_chunk_refs WHERE entity_id=? AND chunk_id=?",
        (entity_id, chunk_id),
    ).fetchone()
    assert rows[0] == 1


def test_record_entity_chunk_ref_distinct_pairs_each_get_a_row(store: TraceStore) -> None:
    """Different (entity, chunk) pairs produce separate rows."""

    build = _start_build(store, build_run_id="kg_build_ecr_distinct")
    round_ = store.start_round(
        build, round_number=1,
        entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    fetched_id = store.record_fetched_doc(
        round_,
        source="arxiv",
        query="x",
        url="https://example/2",
        title="t",
        doc_id="d2",
        outcome="selected_and_fetched",
        parse_method="pdfminer.six",
        cleaned_text_chars=2000,
        ranker_position=1,
    )
    chunk_a = store.record_chunk(
        fetched_doc_id=fetched_id, doc_id="d2", chunk_index=0,
        char_start=0, char_end=1000, text_chars=1000,
    )
    chunk_b = store.record_chunk(
        fetched_doc_id=fetched_id, doc_id="d2", chunk_index=1,
        char_start=800, char_end=2000, text_chars=1200,
    )
    call = store.begin_agent_call(
        round_, agent_name="_stub", action="stub",
        triggered_by="test_setup", input_summary="",
    )
    store.finish_agent_call(call, status="skipped", duration_ms=0, output_summary="")
    e1 = store.upsert_kg_entity(canonical_name="iridium", entity_type="material",
        mentions=1, degree=0, first_seen_round=1, last_seen_round=1,
        created_by_call_id=call.agent_call_id,
    )
    e2 = store.upsert_kg_entity(canonical_name="PEM", entity_type="device",
        mentions=1, degree=0, first_seen_round=1, last_seen_round=1,
        created_by_call_id=call.agent_call_id,
    )

    assert store.record_entity_chunk_ref(entity_id=e1, chunk_id=chunk_a) is not None
    assert store.record_entity_chunk_ref(entity_id=e1, chunk_id=chunk_b) is not None
    assert store.record_entity_chunk_ref(entity_id=e2, chunk_id=chunk_a) is not None

    n = store._conn.execute("SELECT COUNT(*) FROM entity_chunk_refs").fetchone()[0]
    assert n == 3


def test_schema_changes_recorded_with_proposer_and_validator_lineage(
    store: TraceStore,
) -> None:
    build = _start_build(store, build_run_id="kg_build_demo_5")
    round_ = store.start_round(
        build, round_number=1,
        entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    proposer = store.begin_agent_call(
        round_, agent_name="Mapper", action="propose",
        triggered_by="round_extractions_done", input_summary="",
    )
    store.finish_agent_call(proposer, status="success", duration_ms=1, output_summary="")
    store.record_schema_change(
        round_,
        agent_call_id=proposer.agent_call_id,
        event="proposed",
        type_kind="entity_type",
        name="process",
        rationale="OER",
    )
    validator = store.begin_agent_call(
        round_, agent_name="MappingReviewer", action="validate",
        triggered_by="schema_proposed",
        parent_call_id=proposer.agent_call_id,
        input_summary="",
    )
    store.finish_agent_call(validator, status="success", duration_ms=1, output_summary="")
    store.record_schema_change(
        round_,
        agent_call_id=validator.agent_call_id,
        event="accepted",
        type_kind="entity_type",
        name="process",
        decision_reason="broad and useful",
    )

    changes = store.list_schema_changes(build.build_id)
    assert [c["event"] for c in changes] == ["proposed", "accepted"]
    assert changes[1]["decision_reason"] == "broad and useful"

    # parent_call_id surfaces on the validator row so the UI can render
    # the revision chain.
    val_row = next(r for r in store.list_agent_calls(round_.round_id) if r["agent_name"] == "MappingReviewer")
    assert val_row["parent_call_id"] == proposer.agent_call_id


def test_kg_entity_upsert_keeps_canonical(store: TraceStore) -> None:
    build = _start_build(store, build_run_id="kg_build_demo_6")
    call_id = _stub_call(store, build)
    e1 = store.upsert_kg_entity(
        canonical_name="iridium",
        entity_type=None,
        mentions=3,
        degree=1,
        first_seen_round=1,
        last_seen_round=1,
        created_by_call_id=call_id,
    )
    e2 = store.upsert_kg_entity(
        canonical_name="iridium",
        entity_type="material",
        mentions=5,
        degree=2,
        first_seen_round=2,  # later re-snapshot
        last_seen_round=2,
        created_by_call_id=call_id,
    )
    assert e1 == e2
    top = store.top_entities()
    assert top[0]["mentions"] == 5
    assert top[0]["entity_type"] == "material"  # COALESCE preserves new value
    # Earliest first_seen_round preserved across re-snapshots (load-bearing
    # for the per-round graph view's "when did this entity first appear" UX).
    assert top[0]["first_seen_round"] == 1
    assert top[0]["last_seen_round"] == 2


def test_kg_relation_insert_or_ignore_dedupes_same_provenance(
    store: TraceStore,
) -> None:
    """Per-chunk extraction can re-emit the same relation tuple if the
    upstream LLM repeats it. Inserting the same (subj, pred, obj,
    source_doc_id) twice must be a no-op, returning None — otherwise
    the UI's edge-aggregation count would double-count."""

    build = _start_build(store, build_run_id="kg_build_demo_8")
    call_id = _stub_call(store, build)
    a = store.upsert_kg_entity(canonical_name="A", entity_type="m",
        mentions=1, degree=1, first_seen_round=1, last_seen_round=1,
        created_by_call_id=call_id,
    )
    b = store.upsert_kg_entity(canonical_name="B", entity_type="m",
        mentions=1, degree=1, first_seen_round=1, last_seen_round=1,
        created_by_call_id=call_id,
    )
    first = store.record_kg_relation(subject_entity_id=a, predicate="catalyzes",
        object_entity_id=b, source_doc_id="doc_1", round_observed=1,
        created_by_call_id=call_id,
    )
    again = store.record_kg_relation(subject_entity_id=a, predicate="catalyzes",
        object_entity_id=b, source_doc_id="doc_1", round_observed=2,
        created_by_call_id=call_id,
    )
    assert first is not None
    assert again is None  # OR IGNORE → no row returned

    # Different source_doc_id IS a distinct row (provenance is part of key)
    other_doc = store.record_kg_relation(subject_entity_id=a, predicate="catalyzes",
        object_entity_id=b, source_doc_id="doc_2", round_observed=2,
        created_by_call_id=call_id,
    )
    assert other_doc is not None and other_doc != first


def test_graph_data_aggregates_edges_and_drops_isolated(store: TraceStore) -> None:
    build = _start_build(store, build_run_id="kg_build_demo_9")
    call_id = _stub_call(store, build)
    a = store.upsert_kg_entity(canonical_name="iridium", entity_type="material",
        mentions=5, degree=2, first_seen_round=1, last_seen_round=2,
        created_by_call_id=call_id,
    )
    b = store.upsert_kg_entity(canonical_name="OER", entity_type="process",
        mentions=3, degree=1, first_seen_round=1, last_seen_round=2,
        created_by_call_id=call_id,
    )
    # isolated entity — no edges
    store.upsert_kg_entity(canonical_name="Nafion", entity_type="material",
        mentions=2, degree=0, first_seen_round=2, last_seen_round=2,
        created_by_call_id=call_id,
    )
    # same (subj, pred, obj) from two docs → should aggregate to one edge
    # with provenance_count=2 (not two parallel edges).
    store.record_kg_relation(subject_entity_id=a, predicate="catalyzes",
        object_entity_id=b, source_doc_id="doc_1", round_observed=1,
        created_by_call_id=call_id,
    )
    store.record_kg_relation(subject_entity_id=a, predicate="catalyzes",
        object_entity_id=b, source_doc_id="doc_2", round_observed=2,
        created_by_call_id=call_id,
    )

    g = store.graph_data()
    # Nafion (no edges) dropped by default
    node_labels = {n["label"] for n in g["nodes"]}
    assert node_labels == {"iridium", "OER"}
    # One aggregated edge, weight = 2 docs of provenance
    assert len(g["edges"]) == 1
    assert g["edges"][0]["value"] == 2
    assert g["edges"][0]["label"] == "catalyzes"
    # entity_types rollup includes Nafion's type for the filter dropdown
    type_names = {t["entity_type"] for t in g["entity_types"]}
    assert type_names == {"material", "process"}

    # include_isolated=True surfaces Nafion
    g2 = store.graph_data(include_isolated=True)
    assert "Nafion" in {n["label"] for n in g2["nodes"]}

    # entity_type filter narrows the result
    g3 = store.graph_data(entity_type="material", include_isolated=True)
    assert {n["label"] for n in g3["nodes"]} == {"iridium", "Nafion"}


def test_entity_detail_returns_outgoing_incoming_and_doc_ids(
    store: TraceStore,
) -> None:
    build = _start_build(store, build_run_id="kg_build_demo_10")
    call_id = _stub_call(store, build)
    a = store.upsert_kg_entity(canonical_name="iridium", entity_type="material",
        mentions=3, degree=2, first_seen_round=1, last_seen_round=1,
        created_by_call_id=call_id,
    )
    b = store.upsert_kg_entity(canonical_name="OER", entity_type="process",
        mentions=2, degree=1, first_seen_round=1, last_seen_round=1,
        created_by_call_id=call_id,
    )
    store.record_kg_relation(subject_entity_id=a, predicate="catalyzes",
        object_entity_id=b, source_doc_id="doc_x", round_observed=1,
        created_by_call_id=call_id,
    )
    store.record_kg_relation(subject_entity_id=b, predicate="involves",
        object_entity_id=a, source_doc_id="doc_y", round_observed=1,
        created_by_call_id=call_id,
    )

    d = store.entity_detail(a)
    assert d is not None
    assert d["canonical_name"] == "iridium"
    assert len(d["outgoing"]) == 1
    assert d["outgoing"][0]["object_name"] == "OER"
    assert len(d["incoming"]) == 1
    assert d["incoming"][0]["subject_name"] == "OER"
    assert set(d["source_doc_ids"]) == {"doc_x", "doc_y"}


def test_kg_mutation_round_trip_for_target_call_and_round(store: TraceStore) -> None:
    """One Resolver merge writes a parent row + N endpoint-rewrite children;
    UI queries should return the parent on the target lens, all of them on
    the per-call lens, and a per-round count rollup."""

    build = _start_build(store, build_run_id="kg_build_mut_1")
    round_ = store.start_round(
        build, round_number=3,
        entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    call = store.begin_agent_call(
        round_, agent_name="Resolver", action="resolve_entities",
        triggered_by="round_extractions_done", input_summary="",
    )
    store.finish_agent_call(call, status="success", duration_ms=1, output_summary="")

    parent_id = store.record_kg_mutation(
        kind="merge", target_kind="entity", target_id="451",
        before={"id": 502, "name": "Metal hydrides"},
        after={"id": 451, "name": "metal hydride"},
        reason="tier-1 normalization match",
        confidence=1.0,
        agent_call_id=call.agent_call_id,
        round=3,
    )
    for rel_id in (8821, 8822):
        store.record_kg_mutation(
            kind="endpoint_rewrite", target_kind="relation", target_id=str(rel_id),
            before={"subject_entity_id": 502},
            after={"subject_entity_id": 451},
            reason="endpoint rewritten after merge",
            confidence=None,
            agent_call_id=call.agent_call_id,
            parent_mutation_id=parent_id,
            round=3,
        )

    target = store.list_mutations_for_target(
        target_kind="entity", target_id="451"
    )
    assert len(target) == 1
    assert target[0]["kind"] == "merge"
    assert target[0]["confidence"] == 1.0
    assert json.loads(target[0]["before_json"])["name"] == "Metal hydrides"

    per_call = store.list_mutations_for_call(call.agent_call_id)
    assert [m["kind"] for m in per_call] == ["merge", "endpoint_rewrite", "endpoint_rewrite"]
    assert per_call[1]["parent_mutation_id"] == parent_id

    rounds = store.mutation_counts_by_round()
    counts = {(r["round"], r["kind"]): r["n"] for r in rounds}
    assert counts == {(3, "merge"): 1, (3, "endpoint_rewrite"): 2}


def test_kg_mutation_supports_predicate_target_kind(store: TraceStore) -> None:
    """Predicate-level mutations key on the predicate name string, not a row id."""

    build = _start_build(store, build_run_id="kg_build_mut_2")
    round_ = store.start_round(
        build, round_number=2,
        entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    call = store.begin_agent_call(
        round_, agent_name="MappingReviewer", action="validate",
        triggered_by="schema_proposed", input_summary="",
    )
    store.finish_agent_call(call, status="success", duration_ms=1, output_summary="")
    store.record_kg_mutation(
        kind="rename", target_kind="predicate", target_id="produced_by",
        before={"name": "produces_by"},
        after={"name": "produced_by"},
        reason="typo fix",
        confidence=None,
        agent_call_id=call.agent_call_id,
        round=2,
    )
    rows = store.list_mutations_for_target(
        target_kind="predicate", target_id="produced_by"
    )
    assert len(rows) == 1
    assert rows[0]["target_kind"] == "predicate"


def test_kg_entity_provenance_columns_stamped_and_preserved(store: TraceStore) -> None:
    build = _start_build(store, build_run_id="kg_build_prov_1")
    round_ = store.start_round(
        build, round_number=1,
        entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    creator = store.begin_agent_call(
        round_, agent_name="Extractor", action="extract_doc",
        triggered_by="per_doc", input_summary="",
    )
    store.finish_agent_call(creator, status="success", duration_ms=1, output_summary="")
    modifier = store.begin_agent_call(
        round_, agent_name="Extractor", action="extract_doc",
        triggered_by="per_doc", input_summary="",
    )
    store.finish_agent_call(modifier, status="success", duration_ms=1, output_summary="")

    e1 = store.upsert_kg_entity(canonical_name="iridium", entity_type="material",
        mentions=2, degree=1, first_seen_round=1, last_seen_round=1,
        created_by_call_id=creator.agent_call_id,
        last_modified_by_call_id=creator.agent_call_id,
        last_modified_at_round=1,
    )
    # later re-upsert by a different call should preserve creator but refresh
    # the modifier stamp.
    e2 = store.upsert_kg_entity(canonical_name="iridium", entity_type="material",
        mentions=4, degree=2, first_seen_round=2, last_seen_round=2,
        created_by_call_id=modifier.agent_call_id,
        last_modified_by_call_id=modifier.agent_call_id,
        last_modified_at_round=2,
    )
    assert e1 == e2
    row = store._conn.execute(
        "SELECT created_by_call_id, last_modified_by_call_id, last_modified_at_round "
        "FROM kg_entities WHERE id = ?",
        (e1,),
    ).fetchone()
    assert row["created_by_call_id"] == creator.agent_call_id
    assert row["last_modified_by_call_id"] == modifier.agent_call_id
    assert row["last_modified_at_round"] == 2


def test_update_kg_relation_endpoints_returns_before_state(store: TraceStore) -> None:
    """Resolver merge fan-out rewrites relation endpoints; the store must
    return the prior state so the caller can persist a kg_mutations row."""

    build = _start_build(store, build_run_id="kg_build_prov_2")
    call_id = _stub_call(store, build)
    a = store.upsert_kg_entity(canonical_name="A", entity_type="m",
        mentions=1, degree=1, first_seen_round=1, last_seen_round=1,
        created_by_call_id=call_id,
    )
    b = store.upsert_kg_entity(canonical_name="B", entity_type="m",
        mentions=1, degree=1, first_seen_round=1, last_seen_round=1,
        created_by_call_id=call_id,
    )
    canonical = store.upsert_kg_entity(canonical_name="A_canonical", entity_type="m",
        mentions=1, degree=1, first_seen_round=1, last_seen_round=1,
        created_by_call_id=call_id,
    )
    rel_id = store.record_kg_relation(subject_entity_id=a, predicate="catalyzes",
        object_entity_id=b, source_doc_id="doc_1", round_observed=1,
        created_by_call_id=call_id,
    )
    before = store.update_kg_relation_endpoints(
        relation_id=rel_id, subject_entity_id=canonical,
        modified_at_round=2,
    )
    assert before is not None
    assert before["subject_entity_id"] == a
    refreshed = store._conn.execute(
        "SELECT subject_entity_id, last_modified_at_round FROM kg_relations WHERE id = ?",
        (rel_id,),
    ).fetchone()
    assert refreshed["subject_entity_id"] == canonical
    assert refreshed["last_modified_at_round"] == 2


def test_recent_failures_filters_to_failed_and_sample_mode(store: TraceStore) -> None:
    build = _start_build(store, build_run_id="kg_build_demo_7")
    round_ = store.start_round(
        build, round_number=1,
        entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    for status in ["success", "sample_mode", "failed", "skipped"]:
        c = store.begin_agent_call(
            round_, agent_name=status.title(), action="x",
            triggered_by="round_start", input_summary="",
        )
        store.finish_agent_call(
            c, status=status, duration_ms=1, output_summary="",
            error_type=("RuntimeError" if status == "failed" else None),
            error_message=("boom" if status == "failed" else None),
        )
    rows = store.recent_failures(build.build_id)
    statuses = {r["status"] for r in rows}
    assert statuses == {"sample_mode", "failed"}


def test_kg_schema_records_and_lists_types(store: TraceStore) -> None:
    build = _start_build(store, build_run_id="kg_build_demo_schema")
    round_ = store.start_round(
        build, round_number=1,
        entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    call = store.begin_agent_call(
        round_, agent_name="MappingReviewer", action="validate",
        triggered_by="round_proposal", input_summary="",
    )
    store.finish_agent_call(call, status="success", duration_ms=1, output_summary="")

    a = store.record_kg_schema_type(
        name="catalyst", kind="entity_type",
        added_at_call_id=call.agent_call_id, added_at_round=1,
    )
    b = store.record_kg_schema_type(
        name="catalyses", kind="relation_type",
        added_at_call_id=call.agent_call_id, added_at_round=1,
    )
    # Idempotent re-add returns None (INSERT OR IGNORE).
    dup = store.record_kg_schema_type(
        name="catalyst", kind="entity_type",
        added_at_call_id=call.agent_call_id, added_at_round=1,
    )
    assert a is not None and b is not None and dup is None

    schema = store.list_kg_schema()
    assert schema == {
        "entity_types": ["catalyst"],
        "relation_types": ["catalyses"],
        "property_types": [],
    }


def test_loop_state_singleton_upsert_round_trip(store: TraceStore) -> None:
    assert store.read_loop_state() is None
    payload_v1 = json.dumps({"round_number": 1, "urls_fetched": ["u1"]})
    store.write_loop_state(payload_v1)
    assert store.read_loop_state() == payload_v1
    payload_v2 = json.dumps({"round_number": 2, "urls_fetched": ["u1", "u2"]})
    store.write_loop_state(payload_v2)
    assert store.read_loop_state() == payload_v2
