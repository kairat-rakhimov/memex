"""Tests for typed statements (store + schema + gate, no LLM).

Covers the attribute lane's store path (``merge_mapped_attributes`` + UNIQUE
dedup + per-source corroboration + ``property_counts``), the deterministic
property-name form primitive (``normalize_property``), and the ``property_type``
schema kind surfacing through ``KgSchemaView``. Nothing here calls the LLM API —
it exercises the store, schema and gate layers directly.
"""

from __future__ import annotations

from pathlib import Path

from memex.common.trace import TraceWriter
from memex.kg_intelligence.agents import AgentCallSinks
from memex.kg_intelligence.agents.gate import normalize_property
from memex.kg_intelligence.views import load_schema_view
from memex.observability import MappedEntityProperty, TraceStore


def _store_and_call(tmp_path: Path) -> tuple[TraceStore, int]:
    store = TraceStore.open(tmp_path / "kg")
    build = store.start_build(
        build_run_id="ts_test", kg_id="kg", base_query="",
        config={}, sources=[], sample_mode=False, resume_from_round=None,
    )
    round_ = store.start_round(
        build, round_number=1, entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    AgentCallSinks(trace=TraceWriter(request_id="t", log_dir=tmp_path),
                   store=store, round_=round_)
    call = store.begin_agent_call(
        round_, agent_name="Mapper", action="map_doc",
        triggered_by="t", target_agent="kg.db", input_summary="x",
    )
    store.finish_agent_call(call, status="success", duration_ms=1,
                            output_summary="ok", confidence=0.9)
    return store, call.agent_call_id


def _attr(entity, prop, value_num, value_raw, unit="", condition=""):
    return MappedEntityProperty(
        entity=entity, raw_property=prop, raw_value=value_raw, property=prop,
        value_num=value_num, value_raw=value_raw, unit=unit, condition=condition,
        mapper_confidence=0.9, newness_score=0.5, mapped_flag=True,
    )


def test_merge_mapped_attributes_inserts_dedups_and_counts(tmp_path: Path) -> None:
    store, mc = _store_and_call(tmp_path)
    store.merge_mapped_relations(
        entities=[("AISI Ni42-300", "electrocatalyst", None)],
        relations=[], source_doc_id="d1", mapper_call_id=mc, round_number=1,
    )
    attrs = [
        _attr("AISI Ni42-300", "overpotential", 491.0, "491 mV", "mV", "0.1 M KOH pH 7"),
        _attr("AISI Ni42-300", "faradaic_efficiency", 99.4, "99.4 %", "%"),
    ]
    assert store.merge_mapped_attributes(
        attributes=attrs, source_doc_id="d1", mapper_call_id=mc, round_number=1) == 2
    # idempotent re-ingest → no new rows (UNIQUE dedup).
    assert store.merge_mapped_attributes(
        attributes=attrs, source_doc_id="d1", mapper_call_id=mc, round_number=1) == 0
    rows = store._conn.execute(
        "SELECT property, value_num, unit, condition FROM kg_entity_properties "
        "ORDER BY property"
    ).fetchall()
    assert [r["property"] for r in rows] == ["faradaic_efficiency", "overpotential"]
    assert rows[1]["value_num"] == 491.0
    assert rows[1]["condition"] == "0.1 M KOH pH 7"
    assert dict(store.property_counts()) == {"overpotential": 1, "faradaic_efficiency": 1}
    store.close()


def test_merge_mapped_attributes_corroboration_two_docs(tmp_path: Path) -> None:
    """Same property/value from a second doc = a second observation row (the
    per-source-row corroboration contract, mirroring kg_relations — trust is a
    read-time roll-up, never an in-place stack)."""
    store, mc = _store_and_call(tmp_path)
    store.merge_mapped_relations(
        entities=[("hydrogen", "chemical_substance", None)], relations=[],
        source_doc_id="d1", mapper_call_id=mc, round_number=1,
    )
    a = _attr("hydrogen", "specific_energy", 143.0, "143 MJ/kg", "MJ/kg")
    store.merge_mapped_attributes(attributes=[a], source_doc_id="d1",
                                  mapper_call_id=mc, round_number=1)
    store.merge_mapped_attributes(attributes=[a], source_doc_id="d2",
                                  mapper_call_id=mc, round_number=1)
    n = store._conn.execute(
        "SELECT COUNT(*) FROM kg_entity_properties WHERE property='specific_energy'"
    ).fetchone()[0]
    assert n == 2  # two source docs → two observation rows
    store.close()


def test_resolver_merge_repoints_entity_properties(tmp_path: Path) -> None:
    """Regression: merging an entity that has attribute rows must re-point them
    to the survivor, not trip the kg_entity_properties FK on the entity delete."""
    store, mc = _store_and_call(tmp_path)
    store.merge_mapped_relations(
        entities=[("Iridium", "material", None), ("Ir", "material", None)],
        relations=[], source_doc_id="d1", mapper_call_id=mc, round_number=1,
    )
    store.merge_mapped_attributes(
        attributes=[_attr("Ir", "overpotential", 300.0, "300 mV", "mV")],
        source_doc_id="d1", mapper_call_id=mc, round_number=1,
    )
    result = store.apply_merge(survivor_name="Iridium", merged_name="Ir",
                               agent_call_id=mc, round_number=1)
    assert result is not None  # did not raise on the FK
    rows = store._conn.execute(
        "SELECT e.canonical_name FROM kg_entity_properties p "
        "JOIN kg_entities e ON e.id=p.entity_id WHERE p.property='overpotential'"
    ).fetchall()
    assert [r[0] for r in rows] == ["Iridium"]  # re-pointed to the survivor
    store.close()


def test_fact_embedding_sentences(tmp_path: Path) -> None:
    """Relations + attributes surface as natural-language fact sentences for
    the fact-embedding step (survivor entity names + readable predicate/property)."""
    from memex.observability import MappedRelation
    store, mc = _store_and_call(tmp_path)
    store.merge_mapped_relations(
        entities=[("electrolysis of water", "process", None),
                  ("hydrogen", "chemical_substance", None)],
        relations=[MappedRelation(
            raw_subject="water electrolysis", raw_predicate="produces", raw_object="H2",
            canonical_subject="electrolysis of water", canonical_object="hydrogen",
            predicate="produces", mapper_confidence=0.9, newness_score=0.5, mapped_flag=True)],
        source_doc_id="d1", mapper_call_id=mc, round_number=1,
    )
    store.merge_mapped_attributes(
        attributes=[_attr("hydrogen", "specific_energy", 143.0, "143 MJ/kg", "MJ/kg",
                          "lower heating value")],
        source_doc_id="d1", mapper_call_id=mc, round_number=1,
    )
    rel_facts = {f for _id, f in store.relations_without_embeddings()}
    assert "electrolysis of water produces hydrogen" in rel_facts
    attr_facts = {f for _id, f in store.entity_properties_without_embeddings()}
    assert "hydrogen specific energy 143 MJ/kg (lower heating value)" in attr_facts
    store.close()


def test_merge_mapped_attributes_skips_unknown_entity(tmp_path: Path) -> None:
    store, mc = _store_and_call(tmp_path)
    attrs = [_attr("Nonexistent Entity", "x", 1.0, "1")]
    assert store.merge_mapped_attributes(
        attributes=attrs, source_doc_id="d1", mapper_call_id=mc, round_number=1) == 0
    assert store._conn.execute(
        "SELECT COUNT(*) FROM kg_entity_properties").fetchone()[0] == 0
    store.close()


def test_normalize_property_form() -> None:
    assert normalize_property("Tafel slope") == "tafel_slope"
    assert normalize_property("Faradaic Efficiency") == "faradaic_efficiency"
    assert normalize_property("charge transfer coefficient") == "charge_transfer_coefficient"
    assert normalize_property("es promovido por") == ""   # non-English denylist
    assert normalize_property("a b c d e f") == ""         # > max tokens (a clause)
    assert normalize_property("") == ""


def test_schema_view_surfaces_property_types(tmp_path: Path) -> None:
    store, mc = _store_and_call(tmp_path)
    store.record_kg_schema_type(name="overpotential", kind="property_type",
                                added_at_call_id=mc, added_at_round=1)
    store.record_kg_schema_type(name="catalyst", kind="entity_type",
                                added_at_call_id=mc, added_at_round=1)
    view = load_schema_view(store)
    assert "overpotential" in view.property_types
    assert "overpotential" not in view.relation_types
    assert "overpotential" not in view.entity_types
    assert "catalyst" in view.entity_types
    store.close()


# ── Attribute Mapper lane (gate + align + map_doc) ───────────────────────

import json as _json  # noqa: E402

from pydantic_ai.messages import ModelResponse, TextPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

from memex.kg_intelligence.agents.gate import (  # noqa: E402
    attribute_value_violation,
    gate_attributes,
)
from memex.kg_intelligence.agents.mapper import (  # noqa: E402
    RawAttribute,
    RawTriple,
    _align_attributes,
    map_doc,
)
from memex.kg_intelligence.agents.schemas import (  # noqa: E402
    AttributeMapperResult,
    MappedAttribute,
)
from memex.kg_intelligence.views import KgSchemaView  # noqa: E402


def _mapped_attr(i: int, prop: str, canon: str) -> MappedAttribute:
    return MappedAttribute(
        attribute_index=i, raw_entity="E", raw_property=prop, raw_value="1 mV",
        canonical_entity="E", canonical_property=canon, value_num=1.0, unit="mV",
        condition="", mapper_confidence=0.9, newness_score=0.5,
    )


def _ts_settings():
    from memex.common.config import Settings
    return Settings(
        api_key="test-key", base_url="https://llm.test/v1", chat_model="gpt-test",
        reasoning_model="gpt-test", embedding_model="emb-test", embedding_dimension=3072,
        host="0.0.0.0", port=8000, run_timeout_seconds=300, log_dir=Path("logs"),
        log_level="INFO", sample_mode=False, contact=None,
    )


def _ts_store_and_sinks(tmp_path: Path):
    store = TraceStore.open(tmp_path / "kg")
    build = store.start_build(
        build_run_id="ts", kg_id="kg", base_query="", config={}, sources=[],
        sample_mode=False, resume_from_round=None,
    )
    round_ = store.start_round(
        build, round_number=1, entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    sinks = AgentCallSinks(trace=TraceWriter(request_id="t", log_dir=tmp_path),
                           store=store, round_=round_)
    return store, sinks


def test_gate_attributes_routes_admits_and_falls_back() -> None:
    sch = KgSchemaView(property_types=["overpotential"])
    res = AttributeMapperResult(attributes=[
        _mapped_attr(0, "overpotential", "overpotential"),   # existing → commit
        _mapped_attr(1, "Tafel slope", "Tafel slope"),       # net-new → normalize + admit
        _mapped_attr(2, "es promovido", "es promovido"),     # non-English → raw fallback
    ])
    g = gate_attributes(res, sch)
    assert [(x.property, x.mapped_flag) for x in g.gated] == [
        ("overpotential", True), ("tafel_slope", True), ("es promovido", False),
    ]
    assert g.admitted_properties == ["tafel_slope"]


# ── Deterministic attribute value-gate ───────────────────────────────────
# The dangerous direction is FALSE POSITIVES — dropping a legitimate value
# silently erases real qualitative data from every future build. These cases are
# weighted toward "must KEEP"; the drop cases are limited to unambiguous hand-waves.


def _attr_v(i: int, prop: str, canon: str, value_raw: str, value_num):
    return MappedAttribute(
        attribute_index=i, raw_entity="E", raw_property=prop, raw_value=value_raw,
        canonical_entity="E", canonical_property=canon, value_num=value_num, unit="",
        condition="", mapper_confidence=0.9, newness_score=0.5,
    )


def test_value_gate_keeps_all_numeric() -> None:
    # value_num set ALWAYS passes — even when the raw text looks vague or is long.
    assert attribute_value_violation("491 mV", 491.0) is None
    assert attribute_value_violation("higher", 1.5) is None        # numeric overrides vague raw
    assert attribute_value_violation("300 % (COP vs resistive heating)", 300.0) is None
    assert attribute_value_violation("10-80 years", 10.0) is None  # range that parsed a number


def test_value_gate_keeps_real_qualitative_grades() -> None:
    # Standalone grades + short phrases that are genuine values must survive.
    for v in [
        "high", "low", "moderate", "negative", "positive",
        "p-type", "n-type", "monoclinic", "amorphous", "tetragonal",
        "acidic", "alkaline", "Class III", "type IV", "not feasible",
        "high purity", "face-centered cubic", "lower bound",
        "significantly different",   # 'different' is meaningful → kept
        "notable", "limited",        # deliberately NOT in the denylist (grade-like)
        # real 7-token categorical values from a real build — must survive the
        # length check (these were the false positives at the old threshold of 6).
        "20-40% NaOH or KOH in water",
        "Research, Development and Demonstration (RD&D) stage",
    ]:
        assert attribute_value_violation(v, None) is None, f"false-positive drop: {v!r}"


def test_value_gate_keeps_empty_and_token_boundary() -> None:
    assert attribute_value_violation("", None) is None        # empty: out of scope, not dropped
    assert attribute_value_violation("   ", None) is None
    assert attribute_value_violation("a b c d e f g h i j", None) is None  # 10 tokens → keep


def test_value_gate_drops_bare_handwaves() -> None:
    for v in ["higher", "lower", "reduced", "increased", "much", "many",
              "several", "significant", "substantial", "minimal", "negligible"]:
        assert attribute_value_violation(v, None) is not None, f"false-negative keep: {v!r}"


def test_value_gate_drops_handwave_after_stripping_filler() -> None:
    # intensifier adverbs / stopwords are stripped, leaving a bare hand-wave.
    assert attribute_value_violation("significantly higher", None) is not None
    assert attribute_value_violation("relatively lower", None) is not None
    assert attribute_value_violation("a significant amount", None) is not None
    assert attribute_value_violation("much more", None) is not None
    assert attribute_value_violation("increased and reduced", None) is not None


def test_value_gate_drops_sentence_length_prose() -> None:
    assert attribute_value_violation("a b c d e f g h i j k", None) is not None  # 11 tokens
    assert attribute_value_violation(
        "measure of the heat energy required to increase the temperature of a unit quantity",
        None,
    ) is not None   # real definition prose from a real build (14 tokens)


def test_gate_attributes_drops_value_and_blocks_property_admission() -> None:
    sch = KgSchemaView(property_types=["overpotential"])
    res = AttributeMapperResult(attributes=[
        _attr_v(0, "overpotential", "overpotential", "491 mV", 491.0),     # keep (numeric, existing)
        _attr_v(1, "trend", "growth_trend", "higher", None),               # DROP (vague), net-new prop
        _attr_v(2, "phase", "crystal_phase", "monoclinic", None),          # keep (grade), net-new prop
        _attr_v(3, "note", "feasibility_note",
                "not commercially feasible at current electricity prices without large-scale "
                "seasonal storage available", None),  # DROP (>10-token prose)
    ])
    g = gate_attributes(res, sch)
    # 1:1 preserved — the caller does zip(batch, gated, strict=True).
    assert len(g.gated) == 4
    assert [x.dropped for x in g.gated] == [False, True, False, True]
    # a dropped value's net-new property must NOT be admitted (junk can't mint a type)…
    assert "growth_trend" not in g.admitted_properties
    assert "feasibility_note" not in g.admitted_properties
    # …but the kept grade still admits its property.
    assert "crystal_phase" in g.admitted_properties
    # the drop is recorded for the trace.
    dropped = [e for e in g.events if e[0] == "dropped"]
    assert len(dropped) == 2
    assert all(e[1] == "attribute_value" for e in dropped)


def test_gate_attributes_numeric_with_vague_raw_is_kept() -> None:
    # regression guard: a parsed numeric whose raw text is vague must NOT drop —
    # value_num is authoritative, and over-dropping here would erase real measurements.
    sch = KgSchemaView(property_types=[])
    res = AttributeMapperResult(attributes=[_attr_v(0, "eff", "efficiency", "higher", 0.85)])
    g = gate_attributes(res, sch)
    assert g.gated[0].dropped is False
    assert g.gated[0].property == "efficiency"
    assert "efficiency" in g.admitted_properties


def test_align_attributes_backfills_dropped() -> None:
    batch = [RawAttribute("E", "p0", "v0"), RawAttribute("E", "p1", "v1"),
             RawAttribute("E", "p2", "v2")]
    # model returned only indices 0 and 2 (dropped 1) → identity-backfill index 1.
    aligned = _align_attributes([_mapped_attr(0, "p0", "c0"), _mapped_attr(2, "p2", "c2")], batch)
    assert [a.attribute_index for a in aligned] == [0, 1, 2]
    assert aligned[1].canonical_property == "p1"  # identity backfill keeps the raw form


def test_gate_lowercases_netnew_predicate() -> None:
    """A net-new predicate is admitted lowercased so `has_Tafel_slope` and
    `has_tafel_slope` collapse to one canonical instead of splitting."""
    from memex.kg_intelligence.agents.gate import gate_mapping
    from memex.kg_intelligence.agents.schemas import MappedTriple, MapperResult
    t = MappedTriple(
        triple_index=0, raw_subject="a", raw_predicate="has Tafel slope",
        raw_object="b", canonical_subject="A", canonical_object="B",
        canonical_predicate="has_Tafel_slope", mapper_confidence=0.9, newness_score=0.9,
    )
    g = gate_mapping(MapperResult(triples=[t], entity_types=[]), KgSchemaView())
    assert g.gated[0].predicate == "has_tafel_slope"
    assert g.admitted_predicates == ["has_tafel_slope"]


async def test_map_doc_drops_self_loops_and_alias_predicates(tmp_path: Path) -> None:
    """A canon-created self-loop and an alias/identity predicate are dropped
    as edges (entities still captured); a real relation survives."""
    store, sinks = _ts_store_and_sinks(tmp_path)
    payload = _json.dumps({"triples": [
        {"triple_index": 0, "raw_subject": "x", "raw_predicate": "produces",
         "raw_object": "y", "canonical_subject": "X", "canonical_object": "Y",
         "canonical_predicate": "produces", "mapper_confidence": 0.9,
         "newness_score": 0.9, "fallback_predicate": None},
        {"triple_index": 1, "raw_subject": "native hydrogen", "raw_predicate": "is",
         "raw_object": "gold hydrogen", "canonical_subject": "native hydrogen",
         "canonical_object": "native hydrogen", "canonical_predicate": "same_as",
         "mapper_confidence": 0.9, "newness_score": 0.5, "fallback_predicate": None},
        {"triple_index": 2, "raw_subject": "native hydrogen", "raw_predicate": "aka",
         "raw_object": "gold hydrogen", "canonical_subject": "native hydrogen",
         "canonical_object": "gold hydrogen", "canonical_predicate": "has_alias",
         "mapper_confidence": 0.9, "newness_score": 0.5, "fallback_predicate": None},
    ], "entity_types": [{"name": n, "canonical_type": "concept"}
                        for n in ("X", "Y", "native hydrogen", "gold hydrogen")]})

    def cb(messages, _info):
        txt = "".join(getattr(p, "content", "") or "" for m in messages
                      for p in getattr(m, "parts", [])
                      if isinstance(getattr(p, "content", None), str))
        if "MappingReviewer" in txt:
            return ModelResponse(parts=[TextPart('{"suspects":[]}')])
        return ModelResponse(parts=[TextPart(payload)])

    out = await map_doc(
        doc_id="d", doc_title="T", whole_doc="text",
        raw_triples=[RawTriple("x", "produces", "y"),
                     RawTriple("native hydrogen", "is", "gold hydrogen"),
                     RawTriple("native hydrogen", "aka", "gold hydrogen")],
        standalone_entities=[], schema=KgSchemaView(), store=store,
        settings=_ts_settings(), sinks=sinks, round_number=1,
        llm_model=FunctionModel(cb),
    )
    assert [r.predicate for r in out.relations] == ["produces"]  # self-loop + alias dropped
    assert {"X", "Y", "native hydrogen", "gold hydrogen"} <= {e[0] for e in out.entities}
    assert "has_alias" not in out.admitted_predicates  # not admitted to schema either
    store.close()


async def test_map_doc_attribute_lane(tmp_path: Path) -> None:
    store, sinks = _ts_store_and_sinks(tmp_path)
    payload = _json.dumps({"attributes": [{
        "attribute_index": 0, "raw_entity": "AISI Ni42-300",
        "raw_property": "overpotential at 10 mA", "raw_value": "491 mV",
        "canonical_entity": "AISI Ni42-300", "canonical_property": "overpotential",
        "value_num": 491.0, "unit": "mV", "condition": "0.1 M KOH",
        "mapper_confidence": 0.9, "newness_score": 0.8, "fallback_property": None,
    }]})

    def cb(messages, _info):
        return ModelResponse(parts=[TextPart(payload)])

    out = await map_doc(
        doc_id="d1", doc_title="T", whole_doc="full text",
        raw_triples=[],
        raw_attributes=[RawAttribute("AISI Ni42-300", "overpotential at 10 mA",
                                     "491 mV", "mV", "0.1 M KOH")],
        standalone_entities=[], schema=KgSchemaView(), store=store,
        settings=_ts_settings(), sinks=sinks, round_number=1,
        llm_model=FunctionModel(cb),
    )
    assert len(out.entity_properties) == 1
    p = out.entity_properties[0]
    assert p.property == "overpotential" and p.value_num == 491.0 and p.unit == "mV"
    assert p.condition == "0.1 M KOH" and p.mapped_flag is True
    assert "overpotential" in out.admitted_property_types
    assert ("AISI Ni42-300", None, None) in out.entities
    store.close()


async def test_map_doc_carries_source_quote(tmp_path: Path) -> None:
    """The Extractor's verbatim source_quote on a RawTriple / RawAttribute
    reaches the stored MappedRelation / MappedEntityProperty."""
    store, sinks = _ts_store_and_sinks(tmp_path)
    rel_quote = "X produces Y under alkaline conditions."
    attr_quote = "The catalyst showed an overpotential of 491 mV at 10 mA cm-2."
    triple_payload = _json.dumps({
        "triples": [{
            "triple_index": 0, "raw_subject": "x", "raw_predicate": "produces",
            "raw_object": "y", "canonical_subject": "X", "canonical_object": "Y",
            "canonical_predicate": "produces", "mapper_confidence": 0.9,
            "newness_score": 0.1, "fallback_predicate": None,
        }],
        "entity_types": [{"name": "X", "canonical_type": "concept"},
                         {"name": "Y", "canonical_type": "concept"}],
    })
    attr_payload = _json.dumps({"attributes": [{
        "attribute_index": 0, "raw_entity": "AISI Ni42-300",
        "raw_property": "overpotential at 10 mA", "raw_value": "491 mV",
        "canonical_entity": "AISI Ni42-300", "canonical_property": "overpotential",
        "value_num": 491.0, "unit": "mV", "condition": "10 mA cm-2",
        "mapper_confidence": 0.9, "newness_score": 0.8, "fallback_property": None,
    }]})

    def cb(messages, _info):
        txt = "".join(p.content for m in messages
                      for p in getattr(m, "parts", [])
                      if isinstance(getattr(p, "content", None), str))
        if "MappingReviewer" in txt:
            return ModelResponse(parts=[TextPart('{"suspects":[]}')])
        if "AttributeMapper" in txt:
            return ModelResponse(parts=[TextPart(attr_payload)])
        return ModelResponse(parts=[TextPart(triple_payload)])

    out = await map_doc(
        doc_id="d", doc_title="T", whole_doc="text",
        raw_triples=[RawTriple("x", "produces", "y", None, rel_quote)],
        raw_attributes=[RawAttribute("AISI Ni42-300", "overpotential at 10 mA",
                                     "491 mV", "mV", "", None, attr_quote)],
        standalone_entities=[], schema=KgSchemaView(), store=store,
        settings=_ts_settings(), sinks=sinks, round_number=1,
        llm_model=FunctionModel(cb),
    )
    assert [r.source_quote for r in out.relations] == [rel_quote]
    assert [p.source_quote for p in out.entity_properties] == [attr_quote]
    store.close()
