"""Tests for the per-doc Mapper (isolated, no live LLM).

Covers the deterministic gate routing, the candidate-set selection, the
``MappedTriple`` batch alignment, and the per-doc Mapper loop (map → gate →
review → re-map / drop) driven by a ``FunctionModel`` that dispatches between
the Mapper and the MappingReviewer by system prompt. The store write/read path
(``merge_mapped_relations`` + ``aggregated_edges``) is exercised end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from memex.common.config import Settings
from memex.common.trace import TraceWriter
from memex.kg_intelligence.agents import AgentCallSinks
from memex.kg_intelligence.agents.gate import gate_mapping
from memex.kg_intelligence.agents.mapper import (
    RawTriple,
    _align_to_batch,
    _candidate_schema,
    map_doc,
)
from memex.kg_intelligence.agents.schemas import (
    MappedEntityType,
    MappedTriple,
    MapperResult,
)
from memex.kg_intelligence.embeddings import top_k_candidates_by_kind
from memex.kg_intelligence.views import KgSchemaView
from memex.observability import TraceStore

# ── shared helpers ─────────────────────────────────────────────────────────


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


def _schema() -> KgSchemaView:
    return KgSchemaView(
        entity_types=["material", "molecule"],
        relation_types=["catalyzes", "used_in"],
        predicate_aliases=[("is_used_in", "used_in")],
        entity_type_aliases=[],
    )


def _mt(index: int, pred: str, newness: float, **kw) -> MappedTriple:
    return MappedTriple(
        triple_index=index,
        raw_subject=kw.get("raw_subject", "a"),
        raw_predicate=kw.get("raw_predicate", "r"),
        raw_object=kw.get("raw_object", "b"),
        canonical_subject=kw.get("canonical_subject", "A"),
        canonical_object=kw.get("canonical_object", "B"),
        canonical_predicate=pred,
        mapper_confidence=kw.get("conf", 0.9),
        newness_score=newness,
        fallback_predicate=kw.get("fb"),
    )


def _system_text(messages: list[ModelMessage]) -> str:
    out = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                out.append(content)
    return "\n".join(out)


def _mapper_payload(triples: list[dict], entity_types: list[dict] | None = None) -> str:
    """Build a MapperResult JSON. ``entity_types`` defaults to typing every
    distinct canonical endpoint as ``concept`` so the payload validates + the
    entities get a type; pass it explicitly to assert a specific type."""
    if entity_types is None:
        names: list[str] = []
        for t in triples:
            for n in (t["canonical_subject"], t["canonical_object"]):
                if n not in names:
                    names.append(n)
        entity_types = [{"name": n, "canonical_type": "concept"} for n in names]
    return json.dumps({"triples": triples, "entity_types": entity_types})


def _full_triple(index: int, *, raw, canon_pred, newness=0.0, conf=0.9, fb=None) -> dict:
    rs, rp, ro = raw
    cs, _cp, co = raw  # canonical surfaces default to raw surfaces
    return {
        "triple_index": index,
        "raw_subject": rs, "raw_predicate": rp, "raw_object": ro,
        "canonical_subject": cs.title(), "canonical_object": co.title(),
        "canonical_predicate": canon_pred,
        "mapper_confidence": conf, "newness_score": newness,
        "fallback_predicate": fb,
    }


def _dispatch(mapper_payloads: list[str], reviewer_payload: str):
    """Build a FunctionModel callback that serves Mapper payloads in order and
    the single reviewer payload for any MappingReviewer call. Returns
    ``(callback, mapper_idx_ref, reviewer_idx_ref)`` for call-count assertions."""

    m_idx = [0]
    r_idx = [0]

    def _cb(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if "MappingReviewer" in _system_text(messages):
            r_idx[0] += 1
            return ModelResponse(parts=[TextPart(reviewer_payload)])
        payload = mapper_payloads[min(m_idx[0], len(mapper_payloads) - 1)]
        m_idx[0] += 1
        return ModelResponse(parts=[TextPart(payload)])

    return _cb, m_idx, r_idx


def _store_and_sinks(tmp_path: Path) -> tuple[TraceStore, AgentCallSinks]:
    store = TraceStore.open(tmp_path / "kg")
    build = store.start_build(
        build_run_id="mapper_test", kg_id="kg", base_query="",
        config={}, sources=[], sample_mode=False, resume_from_round=None,
    )
    round_ = store.start_round(
        build, round_number=1, entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    trace = TraceWriter(request_id="t", log_dir=tmp_path)
    return store, AgentCallSinks(trace=trace, store=store, round_=round_)


# ── gate (pure) ─────────────────────────────────────────────────────────────


def test_gate_routes_existing_alias_and_admits_wellformed_netnew() -> None:
    """Net-new predicates are admitted on FORM alone (count /
    newness is not a gate criterion). Existing canonicals + alias keys commit;
    any well-formed net-new is admitted regardless of newness."""

    schema = _schema()
    res = MapperResult(triples=[
        _mt(0, "catalyzes", 0.0),                # existing canonical → commit
        _mt(1, "is_used_in", 0.0),               # alias key → route to used_in
        _mt(2, "electrolyzes", 0.9),             # net-new, well-formed → admit
        _mt(3, "produces", 0.1, fb="used_in"),   # net-new, well-formed, LOW newness → STILL admit
    ])
    g = gate_mapping(res, schema)
    finals = {gt.triple.triple_index: (gt.predicate, gt.mapped_flag) for gt in g.gated}
    assert finals[0] == ("catalyzes", True)
    assert finals[1] == ("used_in", True)
    assert finals[2] == ("electrolyzes", True)
    assert finals[3] == ("produces", True)        # admitted on form, not rejected for low newness
    assert set(g.admitted_predicates) == {"electrolyzes", "produces"}
    assert g.violations == []


def test_gate_form_violations_emit_feedback_and_fall_back() -> None:
    """Passive / clause / non-English net-new predicates are FORM violations:
    each emits actionable feedback + routes to fallback (or raw)."""

    schema = _schema()
    res = MapperResult(triples=[
        _mt(0, "is_produced_by", 0.9, fb="used_in"),         # passive → fallback used_in
        _mt(1, "aims_to_reduce_cost_of", 0.9, fb="catalyzes"),# clause → fallback catalyzes
        _mt(2, "es_promovido_por", 0.9, fb=None),            # non-English, no fb → raw
        _mt(3, "electrolyzes", 0.9),                         # well-formed → admit
    ])
    g = gate_mapping(res, schema)
    finals = {gt.triple.triple_index: (gt.predicate, gt.mapped_flag) for gt in g.gated}
    assert finals[0] == ("used_in", True)       # passive → fallback
    assert finals[1] == ("catalyzes", True)     # clause → fallback
    assert finals[2] == ("r", False)            # non-English, no fb → raw, unmapped
    assert finals[3] == ("electrolyzes", True)  # admitted
    assert g.admitted_predicates == ["electrolyzes"]
    by_idx = dict(g.violations)
    assert {0, 1, 2} == set(by_idx)
    assert "passive" in by_idx[0]
    assert "clause" in by_idx[1] or "≤3" in by_idx[1]
    assert "English" in by_idx[2]


def test_predicate_form_rejects_stranded_preposition_clauses() -> None:
    """A 3-token predicate ending in a preposition bakes an object noun into the
    middle (`performance_depends_on`) — a clause. A clean 2-token verb+prep
    (`used_in`, `depends_on`, `part_of`) stays allowed."""

    from memex.kg_intelligence.agents.gate import predicate_form_violation

    for clause in (
        "performance_depends_on",
        "enables_generation_of",
        "sets_standards_for",
        "has_demand_for",
        "no_substitute_in",
    ):
        v = predicate_form_violation(clause)
        assert v is not None and "clause" in v, f"{clause!r} should be a clause"

    for ok in ("used_in", "depends_on", "part_of", "leads_to", "compared_with",
               "has_property", "catalyzes", "produces"):
        assert predicate_form_violation(ok) is None, f"{ok!r} should pass"


def test_gate_admits_new_entity_types_only_once_and_normalizes() -> None:
    schema = _schema()
    res = MapperResult(
        triples=[_mt(0, "catalyzes", 0.0, canonical_subject="A", canonical_object="B")],
        entity_types=[
            MappedEntityType(name="A", canonical_type="Catalyst"),       # net-new, snake_case-normalized
            MappedEntityType(name="B", canonical_type="chemical compound"),  # net-new, space → underscore
            MappedEntityType(name="C", canonical_type="material"),       # existing canonical → not re-admitted
        ],
    )
    g = gate_mapping(res, schema)
    assert g.admitted_entity_types == ["catalyst", "chemical_compound"]  # normalized, deduped
    assert g.entity_types == {"a": "catalyst", "b": "chemical_compound", "c": "material"}


def test_gate_rejects_non_english_entity_type() -> None:
    schema = _schema()
    res = MapperResult(
        triples=[_mt(0, "catalyzes", 0.0)],
        entity_types=[MappedEntityType(name="A", canonical_type="matériel")],
    )
    g = gate_mapping(res, schema)
    assert g.admitted_entity_types == []     # non-ASCII type not admitted
    assert "a" not in g.entity_types         # left untyped rather than typed foreign


def test_gate_camelcase_form_violations_detected() -> None:
    """camelCase tokenization feeds both form checks: a foreign phrase
    (`esPromovidoPor` → es/promovido/por) and a camelCase passive
    (`isProducedBy` → is/produced/by) are both caught and fall back."""

    schema = _schema()
    res = MapperResult(triples=[
        _mt(0, "esPromovidoPor", 0.9, fb="catalyzes"),  # non-English → fallback
        _mt(1, "isProducedBy", 0.9, fb="used_in"),      # camelCase passive → fallback
        _mt(2, "synthesizes", 0.9),                     # well-formed → admitted
    ])
    g = gate_mapping(res, schema)
    finals = {gt.triple.triple_index: gt.predicate for gt in g.gated}
    assert finals[0] == "catalyzes"     # foreign camelCase → fallback
    assert finals[1] == "used_in"       # camelCase passive → fallback
    assert finals[2] == "synthesizes"   # admitted
    assert g.admitted_predicates == ["synthesizes"]
    assert {0, 1} == {idx for idx, _ in g.violations}


# ── alignment (pure) ────────────────────────────────────────────────────────


def test_align_to_batch_clamps_dedups_and_drops() -> None:
    batch = [
        RawTriple("a", "p", "b"),
        RawTriple("c", "q", "d"),
    ]
    triples = [
        _mt(0, "x", 0.0),
        _mt(99, "y", 0.0),   # out of range → re-slotted to positional index 1
        _mt(0, "z", 0.0),    # duplicate index 0 → dropped
    ]
    aligned = _align_to_batch(triples, batch)
    assert [t.triple_index for t in aligned] == [0, 1]


# ── candidate set ────────────────────────────────────────────────────────────


async def test_candidate_schema_returns_full_below_cutoff(tmp_path: Path) -> None:
    store = TraceStore.open(tmp_path / "kg")
    schema = _schema()
    out = await _candidate_schema(schema, ["catalyzes"], store, _settings())
    assert out is schema  # untouched full vocabulary below cutoff
    store.close()


async def test_top_k_candidates_empty_matrix(tmp_path: Path) -> None:
    store = TraceStore.open(tmp_path / "kg")
    got = await top_k_candidates_by_kind(
        store, _settings(), raw_names=["foo"], kind="predicate",
    )
    assert got == []
    store.close()


async def test_candidate_schema_includes_frequent_above_cutoff(tmp_path, monkeypatch) -> None:
    """Above the cutoff, the slice must surface the most-frequent canonical
    predicates (the anti-starvation fix) even when embedding retrieval is empty."""

    from memex.observability import MappedRelation

    store, sinks = _store_and_sinks(tmp_path)
    call = sinks.store.begin_agent_call(
        sinks.round_, agent_name="Mapper", action="map_doc",
        triggered_by="t", target_agent="kg.db", input_summary="x",
    )
    sinks.store.finish_agent_call(call, status="success", duration_ms=1,
                                  output_summary="ok", confidence=0.9)
    mc = call.agent_call_id
    # "used_in" is the most-frequent predicate; "rare_pred" appears once.
    rels = [MappedRelation(raw_subject=f"a{i}", raw_predicate="used_in", raw_object=f"b{i}",
                           canonical_subject=f"A{i}", canonical_object=f"B{i}",
                           predicate="used_in", mapper_confidence=0.9, newness_score=0.0,
                           mapped_flag=True) for i in range(5)]
    rels.append(MappedRelation(raw_subject="x", raw_predicate="rare_pred", raw_object="y",
                               canonical_subject="X", canonical_object="Y",
                               predicate="rare_pred", mapper_confidence=0.9,
                               newness_score=0.9, mapped_flag=True))
    store.merge_mapped_relations(
        entities=[(f"A{i}", "concept", None) for i in range(5)]
        + [(f"B{i}", "concept", None) for i in range(5)] + [("X", "concept", None), ("Y", "concept", None)],
        relations=rels, source_doc_id="d1", mapper_call_id=mc, round_number=1,
    )

    monkeypatch.setattr("memex.kg_intelligence.agents.mapper.CANDIDATE_FULL_CUTOFF", 2)
    schema = KgSchemaView(entity_types=["concept"], relation_types=["used_in", "rare_pred", "unused_pred"])
    out = await _candidate_schema(schema, ["uses"], store, _settings())
    # frequent "used_in" must be surfaced; embeddings empty so retrieval adds nothing.
    assert "used_in" in out.relation_types
    assert "unused_pred" not in out.relation_types  # zero usage, not frequent, not retrieved
    store.close()


# ── map_doc loop ─────────────────────────────────────────────────────────────


async def test_map_doc_happy_path(tmp_path: Path) -> None:
    store, sinks = _store_and_sinks(tmp_path)
    payload = _mapper_payload(
        [
            _full_triple(0, raw=("iridium", "is used to catalyze", "hydrogen"),
                         canon_pred="catalyzes"),
            _full_triple(1, raw=("nafion", "is used in", "electrolyzer"),
                         canon_pred="electrolyzes", newness=0.9, fb="used_in"),
        ],
        entity_types=[
            {"name": "Iridium", "canonical_type": "material"},
            {"name": "Hydrogen", "canonical_type": "molecule"},
            {"name": "Nafion", "canonical_type": "material"},
            {"name": "Electrolyzer", "canonical_type": "device"},
        ],
    )
    cb, m_idx, r_idx = _dispatch([payload], '{"suspects":[]}')

    out = await map_doc(
        doc_id="doc1", doc_title="T", whole_doc="full text here",
        raw_triples=[
            RawTriple("iridium", "is used to catalyze", "hydrogen", chunk_id=None),
            RawTriple("nafion", "is used in", "electrolyzer", chunk_id=None),
        ],
        standalone_entities=[],
        schema=_schema(), store=store, settings=_settings(), sinks=sinks,
        round_number=1, llm_model=FunctionModel(cb),
    )
    assert m_idx[0] == 1 and r_idx[0] == 1  # one map call, one review, no re-map
    preds = sorted(r.predicate for r in out.relations)
    assert preds == ["catalyzes", "electrolyzes"]
    assert all(r.mapped_flag for r in out.relations)
    assert "electrolyzes" in out.admitted_predicates
    assert "device" in out.admitted_entity_types
    assert out.dropped == 0
    assert out.mapper_call_id is not None
    # entities carry canonical surfaces (title-cased) + types
    names = {e[0] for e in out.entities}
    assert {"Iridium", "Hydrogen", "Nafion", "Electrolyzer"} <= names
    store.close()


async def test_map_doc_remap_on_over_collapse(tmp_path: Path) -> None:
    store, sinks = _store_and_sinks(tmp_path)
    first = _mapper_payload([
        _full_triple(0, raw=("x", "inhibits", "y"), canon_pred="catalyzes"),
    ])
    remapped = _mapper_payload([
        _full_triple(0, raw=("x", "inhibits", "y"), canon_pred="inhibits",
                     newness=0.9, fb="catalyzes"),
    ])
    reviewer = json.dumps({"suspects": [
        {"triple_index": 0, "failure_mode": "over_collapse",
         "why": "inhibits is the opposite of catalyzes", "suggested_fix": "mint inhibits"}
    ]})
    cb, m_idx, r_idx = _dispatch([first, remapped], reviewer)

    out = await map_doc(
        doc_id="doc1", doc_title="T", whole_doc="...",
        raw_triples=[RawTriple("x", "inhibits", "y")],
        standalone_entities=[], schema=_schema(), store=store,
        settings=_settings(), sinks=sinks, round_number=1,
        llm_model=FunctionModel(cb),
    )
    assert m_idx[0] == 2 and r_idx[0] == 1  # initial + one re-map; one review
    assert len(out.relations) == 1
    assert out.relations[0].predicate == "inhibits"  # re-mapped, not over-collapsed
    assert "inhibits" in out.admitted_predicates
    store.close()


async def test_map_doc_remap_on_gate_form_violation(tmp_path: Path) -> None:
    """The gate (not the reviewer) flags a passive net-new canonical; its
    feedback drives the re-map, which produces an active predicate that is then
    admitted. Proves the deterministic form guarantee loops."""

    store, sinks = _store_and_sinks(tmp_path)
    passive = _mapper_payload([
        _full_triple(0, raw=("y", "produces", "x"), canon_pred="is_produced_by",
                     newness=0.9, fb="used_in"),
    ])
    active = _mapper_payload([
        _full_triple(0, raw=("y", "produces", "x"), canon_pred="produces",
                     newness=0.9, fb=None),
    ])
    cb, m_idx, r_idx = _dispatch([passive, active], '{"suspects":[]}')  # reviewer clean

    out = await map_doc(
        doc_id="doc1", doc_title="T", whole_doc="...",
        raw_triples=[RawTriple("y", "produces", "x")],
        standalone_entities=[], schema=_schema(), store=store,
        settings=_settings(), sinks=sinks, round_number=1,
        llm_model=FunctionModel(cb),
    )
    assert m_idx[0] == 2  # gate violation triggered a re-map even though reviewer was clean
    assert len(out.relations) == 1
    assert out.relations[0].predicate == "produces"  # active form admitted
    assert out.relations[0].mapped_flag is True
    assert "produces" in out.admitted_predicates
    store.close()


async def test_map_doc_gate_fallback_resolves_against_full_vocab(tmp_path, monkeypatch) -> None:
    """A form-rejected predicate whose ``fallback_predicate`` is a real
    canonical OUTSIDE the candidate slice still maps onto it (the gate resolves
    against the FULL vocabulary), instead of committing raw (mapped_flag=0).

    Forces the slice path (cutoff=1, empty store → empty slice) so ``has_property``
    is absent from what the Mapper sees; both map passes stay passive so the final
    gate must use the fallback. Without it this committed raw."""

    monkeypatch.setattr("memex.kg_intelligence.agents.mapper.CANDIDATE_FULL_CUTOFF", 1)
    store, sinks = _store_and_sinks(tmp_path)
    full_schema = KgSchemaView(
        entity_types=["material"],
        relation_types=["has_property", "catalyzes", "used_in"],
    )
    # Both passes emit a passive (form-violating) canonical with the out-of-slice
    # fallback, so the bounded re-map never produces a clean form → final gate
    # must route to the fallback.
    passive = _mapper_payload([
        _full_triple(0, raw=("a", "is characterized by", "b"),
                     canon_pred="is_characterized_by", newness=0.9, fb="has_property"),
    ])
    cb, m_idx, r_idx = _dispatch([passive, passive], '{"suspects":[]}')

    out = await map_doc(
        doc_id="d", doc_title="T", whole_doc="text",
        raw_triples=[RawTriple("a", "is characterized by", "b")],
        standalone_entities=[], schema=full_schema, store=store,
        settings=_settings(), sinks=sinks, round_number=1,
        llm_model=FunctionModel(cb),
    )
    assert len(out.relations) == 1
    assert out.relations[0].predicate == "has_property"  # rescued onto the fallback
    assert out.relations[0].mapped_flag is True           # not a raw commit
    store.close()


async def test_map_doc_unfaithful_drop(tmp_path: Path) -> None:
    store, sinks = _store_and_sinks(tmp_path)
    payload = _mapper_payload([
        _full_triple(0, raw=("a", "catalyzes", "b"), canon_pred="catalyzes"),
        _full_triple(1, raw=("c", "invented", "d"), canon_pred="catalyzes"),
    ])
    reviewer = json.dumps({"suspects": [
        {"triple_index": 1, "failure_mode": "unfaithful",
         "why": "not stated in the source", "suggested_fix": ""}
    ]})
    cb, m_idx, r_idx = _dispatch([payload], reviewer)

    out = await map_doc(
        doc_id="doc1", doc_title="T", whole_doc="...",
        raw_triples=[RawTriple("a", "catalyzes", "b"), RawTriple("c", "invented", "d")],
        standalone_entities=[], schema=_schema(), store=store,
        settings=_settings(), sinks=sinks, round_number=1,
        llm_model=FunctionModel(cb),
    )
    assert m_idx[0] == 1  # unfaithful drops, no re-map
    assert out.dropped == 1
    assert len(out.relations) == 1
    assert out.relations[0].raw_subject == "a"
    store.close()


async def test_map_doc_sample_mode_identity(tmp_path: Path) -> None:
    store, sinks = _store_and_sinks(tmp_path)
    out = await map_doc(
        doc_id="doc1", doc_title="T", whole_doc="...",
        raw_triples=[RawTriple("a", "weird_raw_pred", "b", chunk_id=None)],
        standalone_entities=[], schema=_schema(), store=store,
        settings=_settings(sample_mode=True), sinks=sinks, round_number=1,
        llm_model=None,
    )
    assert len(out.relations) == 1
    # identity map: raw becomes canonical; net-new mid-newness admitted as canonical
    assert out.relations[0].predicate == "weird_raw_pred"
    assert out.relations[0].mapped_flag is True
    store.close()


async def test_map_doc_output_merges_and_aggregates(tmp_path: Path) -> None:
    """End-to-end: two docs assert the same canonical edge → corroboration=2,
    averaged confidence — the read-time trust contract."""

    store, sinks = _store_and_sinks(tmp_path)

    def _run(doc_id: str, conf: float):
        payload = _mapper_payload(
            [
                {
                    "triple_index": 0,
                    "raw_subject": "ir", "raw_predicate": "catalyzes", "raw_object": "h2",
                    "canonical_subject": "Iridium", "canonical_object": "Hydrogen",
                    "canonical_predicate": "catalyzes",
                    "mapper_confidence": conf, "newness_score": 0.0,
                    "fallback_predicate": None,
                }
            ],
            entity_types=[
                {"name": "Iridium", "canonical_type": "material"},
                {"name": "Hydrogen", "canonical_type": "molecule"},
            ],
        )
        cb, _m, _r = _dispatch([payload], '{"suspects":[]}')
        return cb

    for doc_id, conf in [("doc1", 0.9), ("doc2", 0.7)]:
        out = await map_doc(
            doc_id=doc_id, doc_title="T", whole_doc="...",
            raw_triples=[RawTriple("ir", "catalyzes", "h2")],
            standalone_entities=[], schema=_schema(), store=store,
            settings=_settings(), sinks=sinks, round_number=1,
            llm_model=FunctionModel(_run(doc_id, conf)),
        )
        store.merge_mapped_relations(
            entities=out.entities, relations=out.relations,
            source_doc_id=doc_id, mapper_call_id=out.mapper_call_id, round_number=1,
        )

    edges = store.aggregated_edges()
    assert len(edges) == 1
    edge = edges[0]
    assert (edge.subject_name, edge.predicate, edge.object_name) == (
        "Iridium", "catalyzes", "Hydrogen",
    )
    assert edge.corroboration == 2
    assert abs(edge.avg_confidence - 0.8) < 1e-6
    store.close()


def test_merge_resolves_entity_alias_to_survivor(tmp_path: Path) -> None:
    """Grace path: a name the Resolver already merged away snaps to its
    survivor at merge time instead of re-creating a duplicate entity."""

    from memex.observability import MappedRelation

    store, sinks = _store_and_sinks(tmp_path)
    call = sinks.store.begin_agent_call(
        sinks.round_, agent_name="Mapper", action="map_doc",
        triggered_by="t", target_agent="kg.db", input_summary="x",
    )
    sinks.store.finish_agent_call(call, status="success", duration_ms=1,
                                  output_summary="ok", confidence=0.9)
    mc = call.agent_call_id

    def _rel(subj, obj):
        return MappedRelation(
            raw_subject=subj, raw_predicate="catalyzes", raw_object=obj,
            canonical_subject=subj, canonical_object=obj, predicate="catalyzes",
            mapper_confidence=0.9, newness_score=0.0, mapped_flag=True,
        )

    # Two distinct entities, then the Resolver merges "Ir" into "Iridium".
    store.merge_mapped_relations(
        entities=[("Iridium", "material", None), ("Hydrogen", "molecule", None)],
        relations=[_rel("Iridium", "Hydrogen")],
        source_doc_id="d1", mapper_call_id=mc, round_number=1,
    )
    store.merge_mapped_relations(
        entities=[("Ir", "material", None)],
        relations=[], source_doc_id="d1", mapper_call_id=mc, round_number=1,
    )
    assert store.count_entities() == 3  # Iridium, Hydrogen, Ir
    store.apply_merge(survivor_name="Iridium", merged_name="Ir",
                      agent_call_id=mc, round_number=1)
    assert store.count_entities() == 2  # Ir merged away → alias Ir→Iridium

    # A later doc's Mapper re-surfaces "Ir": it must snap to Iridium, not dup.
    store.merge_mapped_relations(
        entities=[("Ir", "material", None), ("Oxygen", "molecule", None)],
        relations=[_rel("Ir", "Oxygen")],
        source_doc_id="d2", mapper_call_id=mc, round_number=2,
    )
    names = {e["canonical_name"] for e in store.list_kg_entities()}
    assert "Ir" not in names  # snapped to survivor, no duplicate
    assert {"Iridium", "Hydrogen", "Oxygen"} == names
    store.close()


def test_source_quote_persists_on_relations_and_attributes(tmp_path: Path) -> None:
    """The verbatim source_quote round-trips through the merge path onto both
    kg_relations and kg_entity_properties."""
    from memex.observability import MappedEntityProperty, MappedRelation

    store, sinks = _store_and_sinks(tmp_path)
    call = sinks.store.begin_agent_call(
        sinks.round_, agent_name="Mapper", action="map_doc",
        triggered_by="t", target_agent="kg.db", input_summary="x",
    )
    sinks.store.finish_agent_call(call, status="success", duration_ms=1,
                                  output_summary="ok", confidence=0.9)
    mc = call.agent_call_id

    quote = "Electrolysis of water produces hydrogen and oxygen."
    store.merge_mapped_relations(
        entities=[("electrolysis", "process", None), ("hydrogen", "molecule", None)],
        relations=[MappedRelation(
            raw_subject="electrolysis", raw_predicate="produces", raw_object="hydrogen",
            canonical_subject="electrolysis", canonical_object="hydrogen",
            predicate="produces", mapper_confidence=0.9, newness_score=0.0,
            mapped_flag=True, source_quote=quote,
        )],
        source_doc_id="d1", mapper_call_id=mc, round_number=1,
    )
    store.merge_mapped_attributes(
        attributes=[MappedEntityProperty(
            entity="hydrogen", raw_property="energy density", raw_value="120 MJ/kg",
            property="energy_density", value_num=120.0, value_raw="120 MJ/kg",
            unit="MJ/kg", condition="", mapper_confidence=0.9, newness_score=0.0,
            mapped_flag=True, source_quote=quote,
        )],
        source_doc_id="d1", mapper_call_id=mc, round_number=1,
    )

    rq = store._conn.execute("SELECT source_quote FROM kg_relations").fetchone()
    aq = store._conn.execute("SELECT source_quote FROM kg_entity_properties").fetchone()
    assert rq["source_quote"] == quote
    assert aq["source_quote"] == quote
    store.close()


def test_specific_type_overrides_generic_across_docs(tmp_path: Path) -> None:
    """A `concept` placeholder from one doc is replaced by a specific type
    seen in a later doc (the generic no longer sticks)."""

    from memex.observability import MappedRelation

    store, sinks = _store_and_sinks(tmp_path)
    call = sinks.store.begin_agent_call(
        sinks.round_, agent_name="Mapper", action="map_doc",
        triggered_by="t", target_agent="kg.db", input_summary="x",
    )
    sinks.store.finish_agent_call(call, status="success", duration_ms=1,
                                  output_summary="ok", confidence=0.9)
    mc = call.agent_call_id

    def _ed(subj, obj):
        return MappedRelation(
            raw_subject=subj, raw_predicate="catalyzes", raw_object=obj,
            canonical_subject=subj, canonical_object=obj, predicate="catalyzes",
            mapper_confidence=0.9, newness_score=0.0, mapped_flag=True,
        )

    store.merge_mapped_relations(  # doc1: generic placeholder
        entities=[("Iridium", "concept", None), ("Hydrogen", "molecule", None)],
        relations=[_ed("Iridium", "Hydrogen")],
        source_doc_id="d1", mapper_call_id=mc, round_number=1,
    )
    store.merge_mapped_relations(  # doc2: specific type for the same entity
        entities=[("Iridium", "catalyst", None), ("Oxygen", "molecule", None)],
        relations=[_ed("Iridium", "Oxygen")],
        source_doc_id="d2", mapper_call_id=mc, round_number=2,
    )
    iridium = next(e for e in store.list_kg_entities() if e["canonical_name"] == "Iridium")
    assert iridium["entity_type"] == "catalyst"  # specific overrode the stuck generic
    store.close()
