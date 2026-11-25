"""Unit tests for the pure query toolbox (retrieval / expansion / grounding).

Builds a tiny synthetic KG through ``TraceStore`` write methods, then opens it
READ-ONLY via ``QueryDeps`` and exercises the toolbox — no LLM. Embedding
vectors are hand-crafted (dim 4, unit vectors) so cosine outcomes are
deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from memex.common.config import Settings
from memex.kg_query import expansion, grounding, retrieval
from memex.kg_query.deps import QueryDeps
from memex.observability import TraceStore
from memex.observability.store import MappedEntityProperty, MappedRelation


def _vec(*xs: float) -> bytes:
    v = np.array(xs, dtype=np.float32)
    v /= np.linalg.norm(v)
    return v.tobytes()


@pytest.fixture
def kg_root(tmp_path: Path) -> Path:
    root = tmp_path / "synthkg"
    store = TraceStore.open(root)
    build = store.start_build(
        build_run_id="b1", kg_id="synthkg", base_query="green hydrogen",
        config={}, sources=["arxiv"], sample_mode=True, resume_from_round=None,
    )
    round_ = store.start_round(
        build, round_number=1, entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    call = store.begin_agent_call(
        round_, agent_name="_stub", action="stub", triggered_by="test", input_summary="",
    )
    store.finish_agent_call(call, status="skipped", duration_ms=0, output_summary="")
    cid = call.agent_call_id

    # documents + a fetched_doc + chunk for L3 grounding
    text = (
        "Green hydrogen is produced by an electrolyser powered by wind. "
        "The levelised cost of hydrogen falls to 3 USD/kg by 2030."
    )
    store.record_document(
        "doc_1",
        '{"doc_id":"doc_1","source":"arxiv","title":"Electrolysis study",'
        '"url":"https://arxiv.org/abs/0001","authors":["A. Author"]}',
        text,
    )
    fd = store.record_fetched_doc(
        round_, source="arxiv", query="q", url="u", title="Electrolysis study",
        doc_id="doc_1", outcome="selected_and_fetched",
    )
    chunk_id = store.record_chunk(
        fetched_doc_id=fd, doc_id="doc_1", chunk_index=0,
        char_start=0, char_end=len(text), text_chars=len(text),
    )

    ents = {
        "green hydrogen": "concept",
        "electrolyser": "device",
        "PEM electrolyser": "device",
        "wind": "energy_source",
        "ammonia": "chemical",
    }
    ids: dict[str, int] = {}
    for name, etype in ents.items():
        ids[name] = store.upsert_kg_entity(
            canonical_name=name, entity_type=etype, mentions=5, degree=2,
            first_seen_round=1, last_seen_round=1, created_by_call_id=cid,
        )

    def rel(s, p, o, raw_p, doc="doc_1", chunk=chunk_id):
        return store.record_kg_relation(
            subject_entity_id=ids[s], predicate=p, object_entity_id=ids[o],
            source_doc_id=doc, created_by_call_id=cid, source_chunk_id=chunk,
            round_observed=1,
        )

    # one fact corroborated by two docs (collapsed), plus structure for topology.
    # doc_1 goes through the mapped-merge path so raw_* (L2) ≠ canonical (L1).
    store.merge_mapped_relations(
        entities=[
            ("electrolyser", "device", chunk_id),
            ("green hydrogen", "concept", chunk_id),
        ],
        relations=[
            MappedRelation(
                raw_subject="electrolyser", raw_predicate="makes",
                raw_object="green hydrogen", canonical_subject="electrolyser",
                canonical_object="green hydrogen", predicate="produces",
                mapper_confidence=0.9, newness_score=0.5, mapped_flag=True,
                source_chunk_id=chunk_id,
            )
        ],
        source_doc_id="doc_1", mapper_call_id=cid, round_number=1,
    )
    store.record_document("doc_2", '{"doc_id":"doc_2","source":"arxiv","title":"Second"}', text)
    fd2 = store.record_fetched_doc(
        round_, source="arxiv", query="q", url="u", title="Second",
        doc_id="doc_2", outcome="selected_and_fetched",
    )
    chunk2 = store.record_chunk(
        fetched_doc_id=fd2, doc_id="doc_2", chunk_index=0,
        char_start=0, char_end=len(text), text_chars=len(text),
    )
    store.record_kg_relation(
        subject_entity_id=ids["electrolyser"], predicate="produces",
        object_entity_id=ids["green hydrogen"], source_doc_id="doc_2",
        created_by_call_id=cid, source_chunk_id=chunk2, round_observed=1,
    )
    rel("PEM electrolyser", "produces", "green hydrogen", "produces")
    rel("wind", "powers", "electrolyser", "powers")
    rel("green hydrogen", "feeds", "ammonia", "feeds")

    # attribute lane: green hydrogen lcoh at two conditions
    store.merge_mapped_attributes(
        attributes=[
            MappedEntityProperty(
                entity="green hydrogen", raw_property="LCOH", raw_value="3 USD/kg",
                property="lcoh", value_num=3.0, value_raw="3", unit="USD/kg",
                condition="2030", mapper_confidence=0.9, newness_score=0.5,
                mapped_flag=True, source_chunk_id=chunk_id,
            ),
            MappedEntityProperty(
                entity="green hydrogen", raw_property="LCOH", raw_value="5 USD/kg",
                property="lcoh", value_num=5.0, value_raw="5", unit="USD/kg",
                condition="2020", mapper_confidence=0.8, newness_score=0.5,
                mapped_flag=True, source_chunk_id=chunk_id,
            ),
        ],
        source_doc_id="doc_1", mapper_call_id=cid, round_number=1,
    )

    # entity alias via a merge (electrolyser unit -> electrolyser)
    ids["electrolyser unit"] = store.upsert_kg_entity(
        canonical_name="electrolyser unit", entity_type="device", mentions=1, degree=0,
        first_seen_round=1, last_seen_round=1, created_by_call_id=cid,
    )
    store.apply_merge(
        survivor_name="electrolyser", merged_name="electrolyser unit",
        agent_call_id=cid, round_number=1,
    )

    # vocabulary cluster structure
    store.record_predicate_alias(alias="makes", canonical="produces", agent_call_id=cid, round_number=1)
    store.record_predicate_alias(alias="generates", canonical="produces", agent_call_id=cid, round_number=1)
    store.record_schema_parent(child="electrolyzes", parent="produces", kind="relation_type", agent_call_id=cid, round_number=1)

    # predicate-name embeddings for embed-nn expansion
    store.record_embeddings_batch([
        ("predicate", "produces", "test", "m", 4, _vec(1, 0, 0, 0)),
        ("predicate", "creates", "test", "m", 4, _vec(0.96, 0.28, 0, 0)),
        ("predicate", "powers", "test", "m", 4, _vec(0, 1, 0, 0)),
    ])
    # one relation fact embedding for semantic search (the produces fact, id from rel())
    prod_id = store.aggregated_edges()  # noqa: F841 (ensure table ok)
    rid = store._conn.execute(
        "SELECT id FROM kg_relations WHERE predicate='produces' AND source_doc_id='doc_1'"
    ).fetchone()[0]
    store.record_embeddings_batch([
        ("relation", str(rid), "test", "m", 4, _vec(0.9, 0.1, 0.2, 0)),
        ("attribute", "1", "test", "m", 4, _vec(0.2, 0.9, 0, 0)),
    ])
    store.close()
    return root


@pytest.fixture
def deps(kg_root: Path) -> QueryDeps:
    d = QueryDeps.open(kg_root, Settings.from_env())
    yield d
    d.close()


# ── symbolic ────────────────────────────────────────────────────────────────


def test_find_entities_exact_substring_alias(deps: QueryDeps) -> None:
    exact = retrieval.find_entities(deps, "electrolyser")
    assert exact and exact[0].name == "electrolyser"
    # substring surfaces PEM electrolyser too
    names = {e.name for e in retrieval.find_entities(deps, "electrolyser", limit=8)}
    assert "PEM electrolyser" in names
    # alias "electrolyser unit" routes to survivor
    alias_hit = retrieval.find_entities(deps, "electrolyser unit")
    assert alias_hit and alias_hit[0].name == "electrolyser"


def test_entity_facts_collapses_provenance(deps: QueryDeps) -> None:
    eid = retrieval.find_entities(deps, "green hydrogen")[0].entity_id
    facts = retrieval.entity_facts(deps, eid, limit=20)
    produces = [f for f in facts if f.predicate == "produces" and f.subject == "electrolyser"]
    assert len(produces) == 1  # two source docs collapsed into one fact
    assert produces[0].corroboration == 2
    # attribute lane present
    assert any(f.kind == "attribute" and f.predicate == "lcoh" for f in facts)


def test_relations_with_predicate_and_attributes_with_property(deps: QueryDeps) -> None:
    rels = retrieval.relations_with_predicate(deps, ["produces"])
    assert {r.subject for r in rels} >= {"electrolyser", "PEM electrolyser"}
    attrs = retrieval.attributes_with_property(deps, ["lcoh"])
    assert len(attrs) == 2
    assert {a.object for a in attrs} == {"3 USD/kg (2030)", "5 USD/kg (2020)"}


# ── grounding ───────────────────────────────────────────────────────────────


def test_ground_relation_three_levels(deps: QueryDeps) -> None:
    rid = deps.conn.execute(
        "SELECT id FROM kg_relations WHERE predicate='produces' AND source_doc_id='doc_1'"
    ).fetchone()[0]
    gf = grounding.ground_relation(deps, rid)
    assert gf is not None
    assert gf.canonical == "electrolyser produces green hydrogen"   # L1
    assert gf.raw == "electrolyser makes green hydrogen"            # L2 (raw predicate)
    assert "electrolyser" in (gf.chunk_span or "")                 # L3 prose
    assert gf.citation and gf.citation.title == "Electrolysis study"
    assert gf.citation.source == "arxiv"
    assert gf.corroboration == 2


def test_ground_attribute_carries_condition(deps: QueryDeps) -> None:
    aid = deps.conn.execute(
        "SELECT id FROM kg_entity_properties WHERE condition='2030'"
    ).fetchone()[0]
    gf = grounding.ground_attribute(deps, aid)
    assert gf is not None
    assert gf.condition == "2030"
    assert "3" in gf.canonical and "USD/kg" in gf.canonical
    assert gf.citation is not None


def test_ground_facts_drops_unknown_ids(deps: QueryDeps) -> None:
    out = grounding.ground_facts(deps, [("relation", 999999), ("attribute", 1)])
    assert len(out) == 1 and out[0].kind == "attribute"


# ── expansion ─────────────────────────────────────────────────────────────


def test_expand_predicate_alias_class_and_children(deps: QueryDeps) -> None:
    exp = set(expansion.expand_predicate(deps, "produces"))
    assert {"produces", "makes", "generates"} <= exp     # alias equivalence class
    assert "electrolyzes" in exp                          # child via schema_parents
    assert "creates" in exp                               # embed-nn neighbour


def test_expand_degrades_to_embed_only_when_no_aliases(deps: QueryDeps, monkeypatch) -> None:
    # Simulate a live build: alias + parent tables empty.
    monkeypatch.setattr(deps, "alias_pairs", lambda table: [])
    monkeypatch.setattr(deps, "children_of", lambda kind: {})
    exp = set(expansion.expand_predicate(deps, "produces"))
    assert "produces" in exp
    assert "creates" in exp           # embed-nn still works
    assert "makes" not in exp         # alias class gone


# ── semantic + topological ──────────────────────────────────────────────────


def test_align_query_truncates_and_renormalizes() -> None:
    q = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)  # unit 4-vec
    truncated = retrieval.align_query(q, np.zeros((2, 2), dtype=np.float32))
    assert truncated.shape[0] == 2
    assert abs(float(np.linalg.norm(truncated)) - 1.0) < 1e-5  # renormalized to unit
    # no-op when dims already match
    assert np.allclose(retrieval.align_query(q, np.zeros((1, 4), dtype=np.float32)), q)


def test_semantic_facts_ranks_by_cosine(deps: QueryDeps) -> None:
    hits = retrieval.semantic_facts(deps, np.array([0.9, 0.1, 0.2, 0.0], dtype=np.float32), k=5)
    assert hits
    assert hits[0].predicate == "produces"
    assert hits[0].score is not None


def test_topology_neighbors_paths_communities(deps: QueryDeps) -> None:
    gh = retrieval.find_entities(deps, "green hydrogen")[0].entity_id
    el = retrieval.find_entities(deps, "electrolyser")[0].entity_id
    nbr_names = {n.name for n in retrieval.neighbors(deps, el)}
    assert "green hydrogen" in nbr_names and "wind" in nbr_names
    common = retrieval.common_neighbors(deps, gh, retrieval.find_entities(deps, "wind")[0].entity_id)
    assert any(c.name == "electrolyser" for c in common)
    paths = retrieval.shortest_paths(deps, retrieval.find_entities(deps, "wind")[0].entity_id, gh, cutoff=3)
    assert paths and paths[0].names[0] == "wind" and paths[0].names[-1] == "green hydrogen"
    assert isinstance(retrieval.communities(deps, min_size=2), list)
