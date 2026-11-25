"""Unit tests for the pure hypothesis seeds + scoring (no LLM).

Reuses the synthetic-KG fixture from ``test_kg_query_toolbox`` via a small local
rebuild (the fixture there is module-local), exercising each seed lane + the
novelty/support/composite scoring math.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from memex.common.config import Settings
from memex.kg_query import scoring, seeds
from memex.kg_query.deps import QueryDeps
from memex.kg_query.grounding import GroundedFact
from memex.observability import TraceStore
from memex.observability.store import MappedEntityProperty

# reuse the synthetic-kg builder fixtures
from tests.test_kg_query_toolbox import deps, kg_root  # noqa: F401


def _gf(confidence: float, corroboration: int) -> GroundedFact:
    return GroundedFact(
        kind="relation", fact_id=1, subject="a", predicate="p", object="b",
        canonical="a p b", raw="a p b", condition="", confidence=confidence,
        corroboration=corroboration, source_doc_id="d", chunk_span=None, citation=None,
    )


# ── seeds ─────────────────────────────────────────────────────────────────────


def test_bridge_and_gap_seeds_find_abc(deps) -> None:  # noqa: F811
    bridges = seeds.bridge_seeds(deps, limit=10)
    gaps = seeds.gap_seeds(deps, limit=10)
    assert bridges, "expected at least one ABC bridge in the synthetic graph"
    for s in bridges:
        assert s.lane == "bridge"
        assert len(s.entity_ids) == 3
        assert len(s.leg_fact_ids) == 2  # A–B and B–C legs, grounded
    assert all(s.lane == "gap" for s in gaps)


def test_attribute_contested_seed(deps) -> None:  # noqa: F811
    out = seeds.attribute_contested_seeds(deps, limit=10)
    # green hydrogen has lcoh = 3 (2030) and 5 (2020) — divergent under conditions
    assert any("lcoh" in s.payload.get("property", "") for s in out)
    s = next(s for s in out if s.payload.get("property") == "lcoh")
    assert len(s.leg_fact_ids) >= 2
    assert s.lane == "attribute_contested"


def test_generate_seeds_interleaves_lanes(deps) -> None:  # noqa: F811
    out = seeds.generate_seeds(deps, max_seeds=8, per_lane=4)
    assert out
    lanes = {s.lane for s in out}
    # at least bridge/gap + attribute_contested are reachable in the synthetic kg
    assert "attribute_contested" in lanes
    assert lanes & {"bridge", "gap"}


def test_relation_contested_returns_list(deps) -> None:  # noqa: F811
    assert isinstance(seeds.relation_contested_seeds(deps, limit=4), list)


@pytest.fixture
def outlier_deps(tmp_path: Path) -> QueryDeps:
    """A KG where one 'device' is a within-type efficiency outlier (30%) and a
    'concept' has a COP-like 300% — which is a FALSE outlier only if types are
    pooled. Type-aware grouping must flag the device, not the concept."""

    root = tmp_path / "outlierkg"
    store = TraceStore.open(root)
    build = store.start_build(
        build_run_id="b", kg_id="outlierkg", base_query="q", config={},
        sources=["arxiv"], sample_mode=True, resume_from_round=None,
    )
    round_ = store.start_round(
        build, round_number=1, entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    call = store.begin_agent_call(round_, agent_name="_s", action="s", triggered_by="t", input_summary="")
    store.finish_agent_call(call, status="skipped", duration_ms=0, output_summary="")
    cid = call.agent_call_id

    members = [(f"dev{i}", "device", v) for i, v in enumerate([88, 90, 91, 92, 30])]
    members.append(("heat pump", "concept", 300))  # COP — false outlier if pooled
    for name, etype, val in members:
        store.upsert_kg_entity(
            canonical_name=name, entity_type=etype, mentions=2, degree=0,
            first_seen_round=1, last_seen_round=1, created_by_call_id=cid,
        )
        store.merge_mapped_attributes(
            attributes=[MappedEntityProperty(
                entity=name, raw_property="efficiency", raw_value=f"{val}%",
                property="efficiency", value_num=float(val), value_raw=str(val),
                unit="%", condition="nominal", mapper_confidence=0.9,
                newness_score=0.5, mapped_flag=True, source_chunk_id=None,
            )],
            source_doc_id="doc", mapper_call_id=cid, round_number=1,
        )
    store.close()
    d = QueryDeps.open(root, Settings.from_env())
    yield d
    d.close()


def test_property_outlier_is_type_aware(outlier_deps: QueryDeps) -> None:
    out = seeds.property_seeds(outlier_deps, min_group=4)
    assert out, "expected a within-type efficiency outlier"
    outlier_titles = " | ".join(s.title for s in out)
    # the 30% device is the real within-type outlier
    assert any("dev" in s.title and s.payload.get("entity_type") == "device" for s in out)
    # the 300% concept (COP) must NOT be flagged — its type-group is too small
    assert "heat pump" not in outlier_titles
    assert all(s.payload.get("entity_type") != "concept" for s in out)


def test_property_outlier_low_condition_coverage_flag(outlier_deps: QueryDeps) -> None:
    # blank the device conditions so the group trips the scope-mix guard
    outlier_deps.conn.close()
    import sqlite3

    conn = sqlite3.connect(outlier_deps.db_path)
    conn.execute("UPDATE kg_entity_properties SET condition='' WHERE property='efficiency'")
    conn.commit()
    conn.close()
    from memex.kg_query.deps import open_readonly

    outlier_deps.conn = open_readonly(outlier_deps.db_path)
    outlier_deps._matrices.clear()
    out = seeds.property_seeds(outlier_deps, min_group=4)
    dev = next(s for s in out if s.payload.get("entity_type") == "device")
    assert dev.payload.get("low_condition_coverage") is True
    assert "low condition coverage" in dev.rationale


# ── scoring ───────────────────────────────────────────────────────────────────


def test_novelty_floor_rejects_restatement(deps) -> None:  # noqa: F811
    # candidate vector ≈ the stored 'produces' relation embedding (0.9,0.1,0.2,0)
    near = np.array([0.9, 0.1, 0.2, 0.0], dtype=np.float32)
    near /= np.linalg.norm(near)
    sim = scoring.restatement_similarity(deps, near)
    assert sim > 0.9
    assert scoring.is_restatement(sim)
    assert scoring.novelty_score(sim) < 0.15
    # an orthogonal candidate is novel
    far = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    assert scoring.novelty_score(scoring.restatement_similarity(deps, far)) > 0.7


def test_support_score_monotonic() -> None:
    weak = scoring.support_score([_gf(0.5, 1)])
    strong = scoring.support_score([_gf(0.95, 6)])
    assert 0.0 <= weak < strong <= 1.0
    assert scoring.support_score([]) == 0.0


def test_composite_weights() -> None:
    b = scoring.composite_score(novelty=1.0, plausibility=0.0, support=0.0, structure=0.0)
    assert b.composite == pytest.approx(0.25)
    b2 = scoring.composite_score(novelty=0.5, plausibility=0.5, support=0.5, structure=0.5)
    assert b2.composite == pytest.approx(0.5)
    # structure (the "needed the graph" signal) carries the most weight
    s = scoring.composite_score(novelty=0.0, plausibility=0.0, support=0.0, structure=1.0)
    assert s.composite == pytest.approx(0.30)


def _leg(doc: str, span: str, quote: str = "x" * 130) -> GroundedFact:
    return GroundedFact(
        kind="relation", fact_id=1, subject="a", predicate="p", object="b",
        canonical="a p b", raw="a p b", condition="", confidence=0.8,
        corroboration=1, source_doc_id=doc, chunk_span=span, citation=None,
        source_quote=quote,
    )


def test_structure_score_is_graph_dependence() -> None:
    # Lane-blind: more distinct source documents ⇒ more graph-dependence ⇒ higher.
    one_chunk = [_leg("d", "s1"), _leg("d", "s1")]
    one_doc_multi = [_leg("d", "s1"), _leg("d", "s2")]
    two_docs = [_leg("d", "s1"), _leg("e", "s2")]
    three_docs = [_leg("d", "s1"), _leg("e", "s2"), _leg("f", "s3")]
    assert scoring.structure_score(one_chunk) < scoring.structure_score(one_doc_multi)
    assert scoring.structure_score(one_doc_multi) < scoring.structure_score(two_docs)
    assert scoring.structure_score(two_docs) < scoring.structure_score(three_docs)
    assert scoring.structure_score(three_docs) == 1.0


def _bare_seed(lane: str, genuine: bool | None = None) -> seeds.Seed:
    return seeds.Seed(
        lane=lane, title="t", rationale="r", entity_ids=[], leg_fact_ids=[],
        score=0.5, genuine_conflict=genuine,
    )


def test_explore_worthiness_rewards_cross_source(deps) -> None:  # noqa: F811
    seed = _bare_seed("bridge")
    one_doc = [_leg("d", "s1"), _leg("d", "s1")]
    three_docs = [_leg("d", "s1"), _leg("e", "s2"), _leg("f", "s3")]
    assert scoring.explore_worthiness(deps, seed, three_docs) > scoring.explore_worthiness(deps, seed, one_doc)
    assert scoring.explore_worthiness(deps, seed, []) == 0.0


def test_explore_worthiness_demotes_contested_sprawl(deps) -> None:  # noqa: F811
    legs = [_leg("d", "s1"), _leg("e", "s2")]
    genuine = scoring.explore_worthiness(deps, _bare_seed("relation_contested", True), legs)
    sprawl = scoring.explore_worthiness(deps, _bare_seed("relation_contested", False), legs)
    assert sprawl < genuine
