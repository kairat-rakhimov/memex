"""Mapper + MappingReviewer: per-doc vocabulary canonicalization.

The Mapper (gpt-5.1 reasoning, one call per doc) takes the Extractor's free-form
triples + the whole doc + the current canonical vocabulary, and canonicalizes
the *vocabulary* — predicates and entity types, NOT entity identity (that stays
with the Resolver) — emitting per triple a canonical predicate + subject/object
types with a ``confidence`` and ``newness`` score, plus a ``fallback_predicate``
whenever it proposes a net-new canonical.

The MappingReviewer (same model, adversarial) is made to add value via
**asymmetry, not re-derivation**: it sees the chosen canonical + the
deterministic gate decision + the scores the Mapper produced, and is told to
*find errors* against a specific failure-mode checklist. The four
canonicalization failures route to a bounded re-map (≤2 map passes total); the
``unfaithful`` verdict routes to the precision path (drop), absorbing the former
triple-verification pass.

``map_doc`` runs the whole loop and returns store-ready ``MappedRelation`` rows +
the net-new canonicals the gate admitted. Oversized docs degrade to triple-batch
mapping so one whole-doc-in-context call never blows the context window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic_ai import Agent
from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import (
    AgentCallSinks,
    ModelPreference,
    execute_agent_call,
    record_skipped_agent_call,
)
from memex.kg_intelligence.agents.gate import GateEvent, gate_attributes, gate_mapping
from memex.kg_intelligence.agents.schemas import (
    AttributeMapperResult,
    MappedAttribute,
    MappedEntityType,
    MappedTriple,
    MapperResult,
    MappingReviewResult,
)
from memex.kg_intelligence.embeddings import top_k_candidates_by_kind
from memex.kg_intelligence.views import KgSchemaView
from memex.observability import MappedEntityProperty, MappedRelation, TraceStore

_log = logging.getLogger(__name__)

AGENT_NAME = "Mapper"
REVIEWER_NAME = "MappingReviewer"
MODEL_PREFERENCE: ModelPreference = "reasoning"

# Whole-doc context budget (chars). Mirrors the Extractor session budget so the
# Mapper's whole-doc-in-context call stays under the model window; oversized
# docs are truncated.
MAPPER_DOC_BUDGET = 360_000
# Max raw triples per map call — bounds OUTPUT size (the cost/risk to watch).
# More than this and the doc's triples are processed in batches, each seeing the
# whole doc as shared context.
MAPPER_TRIPLE_BATCH = 80
# Above this many canonical names, pass a candidate SLICE instead of the full
# vocabulary. A working Mapper keeps the vocab small, so it may never reach it.
CANDIDATE_FULL_CUTOFF = 250
# Above the cutoff, always include the N most-FREQUENT canonical predicates in
# the candidate slice (alongside per-raw embedding neighbours). Top-K-only
# retrieval starves the Mapper of the workhorse predicates, so it
# re-mints existing vocabulary — a sprawl feedback loop. Surfacing the frequent
# canonicals breaks it (experiments/eval_mapper.py: a good candidate set →
# near-zero new).
FREQUENT_CANDIDATE_N = 60
MAX_MAP_PASSES = 2

# Failure modes that route to a re-map (vs ``unfaithful`` → drop). spurious_new
# is intentionally absent: minting a specific net-new predicate is expected, not
# a defect to re-map.
_REMAP_MODES = frozenset({"over_collapse", "direction", "type_mismatch"})

# Identity/alias predicates are entity-resolution signals, not graph edges — the
# Resolver owns identity. Dropped post-canonicalization (a canon-created
# `native hydrogen has_alias native hydrogen` self-loop also slips the raw
# self-loop filter, so this is the net).
_ALIAS_PREDICATES: frozenset[str] = frozenset({
    "has_alias", "also_known_as", "is_synonym_of", "synonym_of", "same_as",
    "aka", "alias_of", "is_alias_of", "known_as", "abbreviated_as",
})

_MAPPER_SYSTEM_PROMPT = """\
You are Mapper, a QUALITY pass over the free-form triples extracted from one
document. Make each triple CORRECT, FAITHFUL, and WELL-FORMED. Two independent
jobs — do BOTH:

  (1) AGGRESSIVELY fix the FORM of every predicate (voice, tense, clauses).
  (2) NEVER collapse two relations that MEAN different things.

Preserving distinct relations is the goal; many precise predicates is good. But
"preserve meaning" does NOT mean "preserve wording" — always fix the form.

You are given the current canonical vocabulary (entity types + relation
predicates, with aliases), the full document for context, and a numbered list of
raw triples. For EACH raw triple emit one canonical triple.

**FORM rules (apply to EVERY predicate, no exceptions):**
- **Active voice — this is non-negotiable. NEVER emit `is_*`, `are_*`, `was_*`,
  or any `*_by` form.** Convert passive to active and FLIP subject/object:
  `A is_produced_by B` → `B produces A`; `A is_used_in B` → `B uses A`;
  `A is_made_from B` → predicate `made_from`; `A is_characterized_by B` →
  `A has_property B`. If a predicate starts `is_`/`are_`/`was_` or ends `_by`,
  you did it wrong — rewrite it. (Stative `has_*` like `has_property` is fine.)
- **A relation phrase, not a sentence — max 3 words.** Strip modality, hedges,
  quantifiers, and clauses; the object goes in the object slot, never inside the
  predicate: `aims_to_reduce_cost_of` → `reduces` (object=cost);
  `expected_to_grow_by` → `grows`; `does_not_mandate_use_of` → `not_required_for`;
  `investment_increasing_due_to_policy` → `driven_by`.
- **When you shorten, KEEP the specific meaning — do NOT flatten to a generic
  verb.** `is_bifunctional_electrocatalyst_for` → `bifunctional_catalyst_for`
  (specific, 3 words), NOT `used_for`; `is_catalytic_active_species_for` →
  `catalyzes`, NOT `promotes`. Generic verbs (`used_for`, `relates_to`,
  `has_property`) are a LAST resort when nothing specific fits — never a shortcut.
- **English, snake_case.**

**MEANING rule (the one thing you must NOT over-do):**
- Collapse only SURFACE variants of the SAME relation: synonyms, voice, tense,
  morphology. `utilizes`/`is_used_by` → `uses`; `partnered_with`/`is_partner_of`
  → `partners_with`; `established_in`/`was_formed_in` → `founded_in`.
- NEVER merge relations that mean different things — keep `concerned_about`,
  `partners_with`, `competes_with` distinct; keep opposites distinct
  (`catalyzes` vs `inhibits`, `funds` vs `regulates`). Do NOT fold a specific
  relation into a generic one (`presents_safety_concerns` is NOT `has_property`).
  When unsure two phrasings are the same relation, KEEP THEM SEPARATE — losing a
  distinction is worse than a near-duplicate.
- **Reuse an existing label when the relation is the SAME** (so the same fact
  across documents lands on one predicate — this powers corroboration). Coin a
  new predicate freely for a genuinely new relation TYPE — that is expected, not
  a failure.

**canonical_subject / canonical_object**: preserve the surface form (translate
non-English to English). Do NOT merge two different surface names — entity
identity is resolved downstream, not by you.

**entity_types — type each entity ONCE:** emit one ``entity_types`` entry for
EVERY distinct entity that appears as a subject or object in your triples (use
the SAME canonical name). A type is a short snake_case noun (1–3 words):
- **Be SPECIFIC.** Assign the most precise accurate type — `catalyst`,
  `electrolyzer`, `government_agency`, `chemical_compound`. Use a generic like
  `concept` ONLY when nothing specific fits; do not default to it.
- **One type per entity, consistently** — the same entity has the same type
  everywhere (you see the whole document, so decide once).
- **Reuse an existing type / normalize variants** — `chemical compound` →
  `chemical_substance` if that exists; don't mint near-duplicate types.

Scores per triple:
- **mapper_confidence** ∈ [0,1]: how well the document text SUPPORTS this triple.
  Use the FULL range, anchored: 1.0 = stated explicitly / near-verbatim; 0.8 =
  closely paraphrased; 0.6 = a reasonable inference from stated facts; 0.4 =
  partly supported or direction uncertain; 0.2 = only a keyword overlaps.
  Interpolate. Most triples are NOT 1.0 — reserve ≥ 0.9 for near-verbatim support;
  do not pile up at the top.
- **newness_score** ∈ [0,1]: is this relation TYPE new to the vocabulary?
  0.0 = reused an existing canonical; 1.0 = a genuinely new relation type.
  (Informational — do NOT use it to avoid coining a predicate you need.)
- **fallback_predicate**: nearest existing canonical when you coin a new one, so
  a rejected new predicate still lands on real vocabulary. Null when you reused.

Echo `triple_index` (the raw triple's number) and the raw_subject/predicate/
object verbatim so each canonical triple ties back to its input. Emit exactly
one canonical triple per raw triple, in input order, plus an ``entity_types``
entry per distinct entity. Do not invent triples not in the input.
"""

_REVIEWER_SYSTEM_PROMPT = """\
You are MappingReviewer, an adversarial checker of the Mapper's canonicalization
for one document. ASSUME some mappings are wrong and find them — do not re-derive
the mapping from scratch, and do not approve. The Mapper's job is to fix predicate
FORM while PRESERVING meaning, so your PRIMARY concern is lost distinctions: a
deterministic gate already handles form (passive/clause/English), so focus your
attention on meaning. You see each chosen canonical and the Mapper's scores.

Flag ONLY suspect mappings, each with a failure_mode:
- **over_collapse** (the most important): the chosen canonical merged two
  relations that MEAN different things, or folded a specific relation into a
  generic one (`presents_safety_concerns` → `has_property`), or merged opposites
  (`inhibits` → `catalyzes`). Losing a distinction is the main failure now. → re-map
- **direction**: subject and object are reversed for this predicate. → re-map
- **type_mismatch**: a canonical entity type does not fit the entity. → re-map
- **unfaithful**: the document text does not actually support this triple (the
  relation is not stated, or is materially distorted). → drop

(Do NOT flag a predicate merely for being specific or net-new — many precise
predicates is the goal, not a defect.)

Return only the suspects (triple_index + failure_mode + why + a short
suggested_fix). Return an empty list if nothing is clearly wrong — do not
manufacture suspicion. Be precise: an `unfaithful` flag DROPS the triple, so
reserve it for triples the source genuinely does not support, not merely
weakly-stated ones (those keep their low confidence)."""

mapper_agent: Agent[None, MapperResult] = Agent(
    model=None,
    output_type=MapperResult,
    system_prompt=_MAPPER_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)

mapping_reviewer_agent: Agent[None, MappingReviewResult] = Agent(
    model=None,
    output_type=MappingReviewResult,
    system_prompt=_REVIEWER_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)

ATTRIBUTE_AGENT_NAME = "AttributeMapper"

_ATTRIBUTE_MAPPER_SYSTEM_PROMPT = """\
You are AttributeMapper, a QUALITY pass over the free-form ATTRIBUTES extracted
from one document. Each attribute is a property OF an entity whose value is a
measurement or a qualitative grade (NOT a relation to another entity). Make each
attribute CORRECT, FAITHFUL, and STRUCTURED.

You are given the current canonical property vocabulary, the full document for
context, and a numbered list of raw attributes (entity, property, value). For
EACH raw attribute emit one canonical attribute.

**canonical_property — the property NAME only:**
- A short snake_case noun phrase naming the QUANTITY, stripped of conditions and
  values: `overpotential`, `tafel_slope`, `faradaic_efficiency`, `cost`,
  `operating_temperature`, `current_density`, `specific_energy`. Move any embedded
  condition OUT of the name into ``condition``
  (`overpotential_for_HER_at_500_mA_cm-2_in_1_M_KOH` → property `overpotential`,
  condition `HER, 500 mA cm-2, 1 M KOH`).
- **Reuse an existing canonical property when it means the SAME quantity**
  (synonyms / variants → one name: `η` / `overpotential at 10 mA` →
  `overpotential`). Coin a new property freely for a genuinely new quantity — that
  is expected. Do NOT merge distinct quantities (`overpotential` vs
  `onset_potential` stay distinct).
- **Strip parentheticals, chemical-formula symbols, and conditions from the
  NAME** → move them into ``condition``. `standard molar enthalpy of formation
  ΔH ϴ (298.15 K)` → property `enthalpy_of_formation`, condition `standard,
  298.15 K`; `∆η/∆log|j| ratio at 2000 mA cm-2` → property `tafel_slope`,
  condition `2000 mA cm-2`.
- **HARD: English snake_case, ≤4 words, NO units / numbers / symbols /
  parentheses in the name.** If a name exceeds 4 words it is wrong — shorten it to
  the core quantity.

**value_num + unit — parse the raw value:**
- `value_num`: the numeric magnitude as a float when the value is numeric
  (`"491 mV"` → 491.0; `"70-90 °C"` → 70.0, the lower bound of a range; `"~8%"` →
  8.0; `"99.4 %"` → 99.4). Null when the value is qualitative (`"high"`,
  `"three times that of gasoline"`).
- `unit`: canonicalize the unit to a STABLE notation and, when the conversion is
  a pure scale factor, convert `value_num` to ONE canonical unit per quantity so
  `(property, unit)` groups are genuinely comparable:
  - normalize notation: `US$/kg`, `USD/kgH2`, `$/kg H2` → `USD/kg_H2`;
    `mA cm-2` → `mA/cm2`; `degC` → `°C`. Same quantity ⇒ same unit string.
  - convert pure-scale units to the canonical one and SCALE `value_num` to match
    (`1.5 V` where the quantity's canonical unit is `mV` → 1500, unit `mV`;
    `0.5 MW` where canonical is `kW` → 500). Pick the unit that fits most values.
  - **NEVER fabricate a conversion you cannot do exactly** — different currencies
    (EUR vs USD) or any unit needing an external rate / assumption pass through
    UNCONVERTED with their own unit. Do NOT guess an FX rate.
  - `""` when unitless or qualitative.
- `condition`: the scenario the value holds under — **capture it whenever the
  value's own sentence states one**: system boundary (stack vs system, LHV vs
  HHV), scenario / case (`RCP2.6`, `2030 policy case`), YEAR, LOCATION / REGION,
  and measurement setup (electrolyte, temperature, current density, reaction).
  Normalize + comma-separate (`Southern Africa, 2040`; `RCP2.6, 2016-2035`).
  Do NOT reach into other sentences or invent a plausible scenario — only what
  the value's immediate context states. Prefer dropping a value you cannot
  qualify over a misleading blank. `""` when the sentence states no condition.

**canonical_entity**: preserve the surface form (translate non-English to
English). Do NOT merge two different surface names — entity identity is resolved
downstream, not by you.

Scores per attribute:
- **mapper_confidence** ∈ [0,1]: how well the document text SUPPORTS this attribute.
  Use the FULL range, anchored: 1.0 = stated explicitly / near-verbatim; 0.8 =
  closely paraphrased; 0.6 = a reasonable inference; 0.4 = partly supported or the
  value/unit is uncertain; 0.2 = only a keyword overlaps. Most attributes are NOT
  1.0 — reserve ≥ 0.9 for near-verbatim support; do not pile up at the top.
- **newness_score** ∈ [0,1]: is this property TYPE new to the vocabulary?
- **fallback_property**: nearest existing canonical when you coin a new one.

Echo `attribute_index` (the raw attribute's number) and raw_entity / raw_property
/ raw_value verbatim. Emit exactly one canonical attribute per raw attribute, in
input order. Do not invent attributes not in the input.
"""

attribute_mapper_agent: Agent[None, AttributeMapperResult] = Agent(
    model=None,
    output_type=AttributeMapperResult,
    system_prompt=_ATTRIBUTE_MAPPER_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


@dataclass(frozen=True)
class RawTriple:
    """One free-form extracted triple + the chunk it came from. ``source_quote``
    is the Extractor's verbatim sentence supporting the triple ("" when
    absent)."""

    subject: str
    predicate: str
    object: str
    chunk_id: int | None = None
    source_quote: str = ""


@dataclass(frozen=True)
class RawAttribute:
    """One free-form extracted attribute (entity property → value) + its chunk.
    The Mapper canonicalizes the property name + parses the
    value; the Extractor captured all parts loosely."""

    entity: str
    property: str
    value: str
    unit: str = ""
    condition: str = ""
    chunk_id: int | None = None
    source_quote: str = ""


@dataclass
class MapDocResult:
    """Store-ready output of the per-doc Mapper loop (relations + attributes)."""

    entities: list[tuple[str, str | None, int | None]] = field(default_factory=list)
    relations: list[MappedRelation] = field(default_factory=list)
    entity_properties: list[MappedEntityProperty] = field(default_factory=list)
    admitted_predicates: list[str] = field(default_factory=list)
    admitted_entity_types: list[str] = field(default_factory=list)
    admitted_property_types: list[str] = field(default_factory=list)
    events: list[GateEvent] = field(default_factory=list)
    dropped: int = 0
    mapper_call_id: int | None = None


# ── candidate set ──────────────────────────────────────────────────────────


async def _candidate_schema(
    schema: KgSchemaView,
    raw_predicates: list[str],
    store: TraceStore,
    settings: Settings,
) -> KgSchemaView:
    """Full canonical vocabulary below the cutoff; above it, a slice that is the
    UNION of the N most-frequent canonical predicates (the workhorses the Mapper
    must keep reusing) and the per-raw-predicate embedding neighbours. Entity
    types (few) always pass in full.

    The frequent-predicate floor is the fix for the candidate-starvation sprawl
    loop: top-K-only retrieval hid the common vocabulary, so
    the Mapper re-minted it."""

    total = len(schema.relation_types) + len(schema.entity_types)
    if total <= CANDIDATE_FULL_CUTOFF:
        return schema

    frequent = [
        p for p, _n in store.predicate_counts()
        if p in set(schema.relation_types)
    ][:FREQUENT_CANDIDATE_N]
    retrieved = await top_k_candidates_by_kind(
        store, settings, raw_names=raw_predicates, kind="predicate",
    )
    candidate_set = {c.lower() for c in frequent} | {c.lower() for c in retrieved}
    sliced = [p for p in schema.relation_types if p.lower() in candidate_set]
    _log.info(
        "Mapper candidate slice: %d/%d predicates "
        "(%d frequent + %d retrieved; vocab over %d cutoff)",
        len(sliced), len(schema.relation_types),
        len(frequent), len(retrieved), CANDIDATE_FULL_CUTOFF,
    )
    return KgSchemaView(
        entity_types=list(schema.entity_types),
        relation_types=sliced,
        predicate_aliases=[
            (a, c) for a, c in schema.predicate_aliases if c.lower() in candidate_set
        ],
        entity_type_aliases=list(schema.entity_type_aliases),
    )


# ── public entry point ───────────────────────────────────────────────────


async def map_doc(
    *,
    doc_id: str,
    doc_title: str,
    whole_doc: str,
    raw_triples: list[RawTriple],
    standalone_entities: list[tuple[str, str | None, int | None]],
    schema: KgSchemaView,
    store: TraceStore,
    settings: Settings,
    sinks: AgentCallSinks,
    round_number: int,
    llm_model: Model | None = None,
    triggered_by: str = "per_doc_extracted",
    raw_attributes: list[RawAttribute] | None = None,
) -> MapDocResult:
    """Run the bounded map → gate → review → re-map loop over one doc's triples,
    plus the attribute lane, and return store-ready rows + admitted canonicals."""

    raw_attributes = raw_attributes or []
    out = MapDocResult(entities=list(standalone_entities))
    if not raw_triples and not raw_attributes:
        # Entity-only doc: no triples to map, but the standalone entities still
        # need a call to attribute their rows to. Record a skipped Mapper call.
        if standalone_entities:
            skipped = record_skipped_agent_call(
                sinks=sinks,
                agent_name=AGENT_NAME,
                action="map_doc",
                triggered_by=triggered_by,
                target_agent="kg.db",
                input_summary=(
                    f"doc_id={doc_id}, triples=0, "
                    f"standalone_entities={len(standalone_entities)}"
                ),
                output_summary="no triples — entities only",
            )
            out.mapper_call_id = skipped.agent_call_id if skipped is not None else None
        return out

    doc_ctx = (whole_doc or "")[:MAPPER_DOC_BUDGET]
    candidate = await _candidate_schema(
        schema, [t.predicate for t in raw_triples], store, settings
    )

    for start in range(0, len(raw_triples), MAPPER_TRIPLE_BATCH):
        batch = raw_triples[start : start + MAPPER_TRIPLE_BATCH]
        # Cross-batch vocab: later batches in the same doc see the canonicals
        # earlier batches admitted, so the Mapper reuses them instead of
        # re-minting near-duplicates within one doc.
        batch_schema = candidate
        if out.admitted_predicates:
            batch_schema = KgSchemaView(
                entity_types=list(candidate.entity_types),
                relation_types=list(candidate.relation_types) + out.admitted_predicates,
                property_types=list(candidate.property_types),
                predicate_aliases=list(candidate.predicate_aliases),
                entity_type_aliases=list(candidate.entity_type_aliases),
            )
        # The gate resolves canonicals + fallbacks against the FULL vocabulary
        # (not the slice the Mapper saw) + this doc's admissions so far, so a
        # form-rejected predicate whose fallback is a real canonical outside the
        # slice lands on it instead of committing raw.
        gate_schema = KgSchemaView(
            entity_types=list(schema.entity_types) + out.admitted_entity_types,
            relation_types=list(schema.relation_types) + out.admitted_predicates,
            property_types=list(schema.property_types) + out.admitted_property_types,
            predicate_aliases=list(schema.predicate_aliases),
            entity_type_aliases=list(schema.entity_type_aliases),
        )
        gated, dropped, call_id, gate = await _map_batch(
            doc_id=doc_id,
            doc_title=doc_title,
            doc_ctx=doc_ctx,
            batch=batch,
            schema=batch_schema,
            gate_schema=gate_schema,
            store=store,
            settings=settings,
            sinks=sinks,
            round_number=round_number,
            llm_model=llm_model,
            triggered_by=triggered_by,
        )
        out.dropped += dropped
        out.events.extend(gate.events)
        for name in gate.admitted_predicates:
            # Alias predicates get dropped as edges below — don't admit them to
            # kg_schema either (no orphan vocabulary rows).
            if name.strip().lower() in _ALIAS_PREDICATES:
                continue
            if name not in out.admitted_predicates:
                out.admitted_predicates.append(name)
        for name in gate.admitted_entity_types:
            if name not in out.admitted_entity_types:
                out.admitted_entity_types.append(name)
        if out.mapper_call_id is None:
            out.mapper_call_id = call_id

        for raw, g in gated:
            # Type each endpoint from the gate's per-entity map (one type per
            # entity, snake_case-normalized). Untyped → None (Resolver/merge
            # tolerate it; better than a per-triple guess).
            subj = g.triple.canonical_subject
            obj = g.triple.canonical_object
            out.entities.append((subj, gate.entity_types.get(subj.strip().lower()), raw.chunk_id))
            out.entities.append((obj, gate.entity_types.get(obj.strip().lower()), raw.chunk_id))
            # Drop canon-created self-loops (the raw-only filter upstream can't
            # catch two surface forms the Mapper collapsed to one entity) and
            # alias/identity predicates (entity-resolution signals, not edges).
            if subj.strip().lower() == obj.strip().lower():
                continue
            if g.predicate.strip().lower() in _ALIAS_PREDICATES:
                continue
            out.relations.append(
                MappedRelation(
                    raw_subject=g.triple.raw_subject or raw.subject,
                    raw_predicate=g.triple.raw_predicate or raw.predicate,
                    raw_object=g.triple.raw_object or raw.object,
                    canonical_subject=g.triple.canonical_subject,
                    canonical_object=g.triple.canonical_object,
                    predicate=g.predicate,
                    mapper_confidence=g.triple.mapper_confidence,
                    newness_score=g.triple.newness_score,
                    mapped_flag=g.mapped_flag,
                    source_chunk_id=raw.chunk_id,
                    source_quote=raw.source_quote,
                )
            )

    if raw_attributes:
        await _map_attributes(
            doc_id=doc_id, doc_title=doc_title, doc_ctx=doc_ctx,
            raw_attributes=raw_attributes, schema=schema, store=store,
            settings=settings, sinks=sinks, llm_model=llm_model,
            triggered_by=triggered_by, out=out,
        )
    return out


# ── one batch: map → gate → review → re-map ───────────────────────────────


async def _map_batch(
    *,
    doc_id: str,
    doc_title: str,
    doc_ctx: str,
    batch: list[RawTriple],
    schema: KgSchemaView,
    store: TraceStore,
    settings: Settings,
    sinks: AgentCallSinks,
    round_number: int,
    llm_model: Model | None,
    triggered_by: str,
    gate_schema: KgSchemaView | None = None,
):
    """Map one batch of raw triples, gate, review, and re-map the flagged
    triples once. ``schema`` is what the Mapper SEES (a candidate slice above the
    vocab cutoff); ``gate_schema`` (the FULL vocabulary, defaulting to ``schema``)
    is what the deterministic gate resolves canonicals + fallbacks against — so a
    form-rejected predicate whose ``fallback_predicate`` is a real canonical
    OUTSIDE the slice still lands on it instead of committing raw.

    The re-map feedback merges deterministic gate FORM violations
    (passive / clause / non-English — the guarantee the prompt can't give) with
    the MappingReviewer's semantic suspects (over-collapse / direction / type);
    unfaithful suspects are dropped. Returns ``(gated, dropped, call_id,
    final_gate)`` — ``gated`` is ``(RawTriple, GatedTriple)`` with drops removed;
    ``final_gate`` is the FINAL re-gate (its ``entity_types`` / admissions /
    events), so a form-violating predicate the re-map fails to fix is hard-routed
    to fallback/raw, never admitted."""

    result, call_id = await _run_mapper(
        doc_id=doc_id, doc_title=doc_title, doc_ctx=doc_ctx,
        batch=batch, schema=schema, settings=settings, sinks=sinks,
        llm_model=llm_model, triggered_by=triggered_by, action="map_doc",
    )
    gate_schema = gate_schema or schema
    triples = _align_to_batch(result.triples, batch)
    # entity_types come from the first map pass and carry through (the re-map
    # fixes predicate FORM, not types — types are already once-per-entity).
    entity_types = result.entity_types
    first_gate = gate_mapping(
        MapperResult(triples=triples, entity_types=entity_types), gate_schema
    )
    triple_by_index = {g.triple.triple_index: g.triple for g in first_gate.gated}

    review = await _run_reviewer(
        doc_id=doc_id, batch=batch, gated=first_gate.gated,
        settings=settings, sinks=sinks, llm_model=llm_model,
        parent_call_id=call_id,
    )

    drop_indices: set[int] = set()
    remap_feedback: dict[int, str] = {}
    # Deterministic gate FORM violations (passive / clause / non-English) — the
    # guarantee the prompt can't give. Each carries actionable feedback.
    for idx, fb in first_gate.violations:
        if idx in triple_by_index:
            remap_feedback[idx] = fb
    # MappingReviewer SEMANTIC suspects — over-collapse (the real risk now),
    # direction, type-mismatch → re-map; unfaithful → drop.
    for s in review.suspects:
        if s.triple_index not in triple_by_index:
            continue
        if s.failure_mode == "unfaithful":
            drop_indices.add(s.triple_index)
        elif s.failure_mode in _REMAP_MODES:
            note = f"{s.failure_mode}: {s.why} {s.suggested_fix}".strip()
            remap_feedback[s.triple_index] = (
                f"{remap_feedback[s.triple_index]}; {note}"
                if s.triple_index in remap_feedback else note
            )

    if remap_feedback:
        sorted_idx = [i for i in sorted(remap_feedback) if i < len(batch)]
        remap_batch = [batch[i] for i in sorted_idx]
        remap_result, _remap_call = await _run_mapper(
            doc_id=doc_id, doc_title=doc_title, doc_ctx=doc_ctx,
            batch=remap_batch, schema=schema, settings=settings, sinks=sinks,
            llm_model=llm_model, triggered_by="mapping_review_remap",
            action="map_doc_remap",
            feedback=[remap_feedback[i] for i in sorted_idx],
        )
        remapped = _align_to_batch(remap_result.triples, remap_batch)
        for local_pos, t in enumerate(remapped):
            if local_pos < len(sorted_idx):
                triple_by_index[sorted_idx[local_pos]] = t

    surviving_raw: list[RawTriple] = []
    surviving_triples: list[MappedTriple] = []
    for idx in sorted(triple_by_index):
        if idx in drop_indices or idx >= len(batch):
            continue
        surviving_raw.append(batch[idx])
        surviving_triples.append(triple_by_index[idx])

    final_gate = gate_mapping(
        MapperResult(triples=surviving_triples, entity_types=entity_types), gate_schema
    )
    gated = list(zip(surviving_raw, final_gate.gated, strict=True))
    return gated, len(drop_indices), call_id, final_gate


def _align_to_batch(
    triples: list[MappedTriple], batch: list[RawTriple]
) -> list[MappedTriple]:
    """Clamp echoed ``triple_index`` into range and keep one mapping per index
    (the model is told 1:1 in input order; defend against drift). Triples whose
    index can't be tied to a batch slot are dropped."""

    seen: set[int] = set()
    aligned: list[MappedTriple] = []
    for i, t in enumerate(triples):
        idx = t.triple_index if 0 <= t.triple_index < len(batch) else i
        if idx >= len(batch) or idx in seen:
            continue
        seen.add(idx)
        aligned.append(t if t.triple_index == idx else t.model_copy(update={"triple_index": idx}))
    return aligned


# ── attribute lane: map → gate → build property rows ─────────────────────────


async def _map_attributes(
    *,
    doc_id: str,
    doc_title: str,
    doc_ctx: str,
    raw_attributes: list[RawAttribute],
    schema: KgSchemaView,
    store: TraceStore,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None,
    triggered_by: str,
    out: MapDocResult,
) -> None:
    """Canonicalize the doc's raw attributes (separate batched call that shares
    the cached whole-doc prefix), gate the property names, and append store-ready
    ``MappedEntityProperty`` rows + admitted property types to ``out``. The
    attribute's entity is also added to ``out.entities`` so the merge upserts it
    (an attribute-only entity would otherwise be skipped)."""

    for start in range(0, len(raw_attributes), MAPPER_TRIPLE_BATCH):
        batch = raw_attributes[start : start + MAPPER_TRIPLE_BATCH]
        result, call_id = await _run_attribute_mapper(
            doc_id=doc_id, doc_title=doc_title, doc_ctx=doc_ctx,
            batch=batch, schema=schema, settings=settings, sinks=sinks,
            llm_model=llm_model, triggered_by=triggered_by,
        )
        aligned = _align_attributes(result.attributes, batch)
        gate = gate_attributes(AttributeMapperResult(attributes=aligned), schema)
        out.events.extend(gate.events)
        for name in gate.admitted_properties:
            if name not in out.admitted_property_types:
                out.admitted_property_types.append(name)
        if out.mapper_call_id is None:
            out.mapper_call_id = call_id
        for raw, g in zip(batch, gate.gated, strict=True):
            if g.dropped:
                # The value gate rejected this attribute's value (vague/sentence).
                # Don't store it (cuts sprawl); its entity is added only if it's a
                # real triple endpoint elsewhere. The drop is recorded in gate.events.
                continue
            a = g.attribute
            entity = a.canonical_entity or raw.entity
            value_raw = a.raw_value or raw.value
            out.entity_properties.append(
                MappedEntityProperty(
                    entity=entity,
                    raw_property=a.raw_property or raw.property,
                    raw_value=value_raw,
                    property=g.property,
                    value_num=a.value_num,
                    value_raw=value_raw,
                    unit=a.unit,
                    condition=a.condition,
                    mapper_confidence=a.mapper_confidence,
                    newness_score=a.newness_score,
                    mapped_flag=g.mapped_flag,
                    source_chunk_id=raw.chunk_id,
                    source_quote=raw.source_quote,
                )
            )
            # Ensure the attribute's entity exists for the merge (it may not be a
            # triple endpoint). Type left None — typed via triples / Resolver.
            out.entities.append((entity, None, raw.chunk_id))


def _identity_attribute(index: int, raw: RawAttribute) -> MappedAttribute:
    """Raw → canonical identity map (sample / backfill): keep the surface forms,
    no numeric parse. The gate then snake_case-normalizes the property name."""
    return MappedAttribute(
        attribute_index=index,
        raw_entity=raw.entity, raw_property=raw.property, raw_value=raw.value,
        canonical_entity=raw.entity, canonical_property=raw.property,
        value_num=None, unit=raw.unit, condition=raw.condition,
        mapper_confidence=0.5, newness_score=0.5, fallback_property=None,
    )


def _align_attributes(
    attributes: list[MappedAttribute], batch: list[RawAttribute]
) -> list[MappedAttribute]:
    """Exactly one MappedAttribute per batch slot, in order — clamp echoed
    indices, dedup, and backfill any the model dropped with an identity map so the
    gate + merge stay 1:1 with the batch."""
    by_idx: dict[int, MappedAttribute] = {}
    for i, a in enumerate(attributes):
        idx = a.attribute_index if 0 <= a.attribute_index < len(batch) else i
        if idx >= len(batch) or idx in by_idx:
            continue
        by_idx[idx] = (
            a if a.attribute_index == idx
            else a.model_copy(update={"attribute_index": idx})
        )
    return [by_idx.get(i) or _identity_attribute(i, raw) for i, raw in enumerate(batch)]


# ── agent runners ──────────────────────────────────────────────────────────


async def _run_mapper(
    *,
    doc_id: str,
    doc_title: str,
    doc_ctx: str,
    batch: list[RawTriple],
    schema: KgSchemaView,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None,
    triggered_by: str,
    action: str,
    feedback: list[str] | None = None,
) -> tuple[MapperResult, int | None]:
    user_prompt = _build_mapper_prompt(doc_title, doc_ctx, batch, schema, feedback)
    input_summary = (
        f"doc_id={doc_id}, triples={len(batch)}, "
        f"vocab={len(schema.entity_types)}/{len(schema.relation_types)}, "
        f"{'remap' if feedback else 'initial'}"
    )
    result, call, _run = await execute_agent_call(
        agent=mapper_agent,
        user_prompt=user_prompt,
        system_prompt=_MAPPER_SYSTEM_PROMPT,
        sample_factory=lambda: _sample_mapper(batch),
        log_prefix=f"{AGENT_NAME}[{doc_id}]",
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=action,
        triggered_by=triggered_by,
        target_agent="kg.db",
        input_summary=input_summary,
        output_summary_fn=lambda r: f"mapped={len(r.triples)}",
        confidence=0.7,
        model_preference=MODEL_PREFERENCE,
    )
    return result, (call.agent_call_id if call is not None else None)


async def _run_reviewer(
    *,
    doc_id: str,
    batch: list[RawTriple],
    gated: list,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None,
    parent_call_id: int | None,
) -> MappingReviewResult:
    if not gated:
        return MappingReviewResult()
    user_prompt = _build_reviewer_prompt(gated)
    review, _call, _run = await execute_agent_call(
        agent=mapping_reviewer_agent,
        user_prompt=user_prompt,
        system_prompt=_REVIEWER_SYSTEM_PROMPT,
        sample_factory=lambda: MappingReviewResult(),
        log_prefix=f"{REVIEWER_NAME}[{doc_id}]",
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=REVIEWER_NAME,
        action="review_mapping",
        triggered_by="mapping_done",
        target_agent=AGENT_NAME,
        input_summary=f"doc_id={doc_id}, mapped={len(gated)}",
        output_summary_fn=lambda r: f"suspects={len(r.suspects)}",
        confidence=0.7,
        parent_call_id=parent_call_id,
        model_preference=MODEL_PREFERENCE,
    )
    return review


async def _run_attribute_mapper(
    *,
    doc_id: str,
    doc_title: str,
    doc_ctx: str,
    batch: list[RawAttribute],
    schema: KgSchemaView,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None,
    triggered_by: str,
) -> tuple[AttributeMapperResult, int | None]:
    user_prompt = _build_attribute_prompt(doc_title, doc_ctx, batch, schema)
    input_summary = (
        f"doc_id={doc_id}, attributes={len(batch)}, "
        f"properties={len(schema.property_types)}"
    )
    result, call, _run = await execute_agent_call(
        agent=attribute_mapper_agent,
        user_prompt=user_prompt,
        system_prompt=_ATTRIBUTE_MAPPER_SYSTEM_PROMPT,
        sample_factory=lambda: _sample_attribute_mapper(batch),
        log_prefix=f"{ATTRIBUTE_AGENT_NAME}[{doc_id}]",
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=ATTRIBUTE_AGENT_NAME,
        action="map_attributes",
        triggered_by=triggered_by,
        target_agent="kg.db",
        input_summary=input_summary,
        output_summary_fn=lambda r: f"mapped={len(r.attributes)}",
        confidence=0.7,
        model_preference=MODEL_PREFERENCE,
    )
    return result, (call.agent_call_id if call is not None else None)


# ── prompts + sample factories ─────────────────────────────────────────────


def _fmt_vocab(schema: KgSchemaView) -> str:
    def _aliases(pairs: list[tuple[str, str]]) -> str:
        return ", ".join(f"{a} → {c}" for a, c in pairs) or "(none)"

    return (
        "Current canonical vocabulary:\n"
        f"- entity types: {', '.join(schema.entity_types) or '(empty)'}\n"
        f"- entity-type aliases: {_aliases(schema.entity_type_aliases)}\n"
        f"- relation predicates: {', '.join(schema.relation_types) or '(empty)'}\n"
        f"- predicate aliases: {_aliases(schema.predicate_aliases)}"
    )


def _build_mapper_prompt(
    doc_title: str,
    doc_ctx: str,
    batch: list[RawTriple],
    schema: KgSchemaView,
    feedback: list[str] | None,
) -> str:
    triples_section = "\n".join(
        f"[{i}] subject={t.subject!r} predicate={t.predicate!r} object={t.object!r}"
        for i, t in enumerate(batch)
    )
    # Order for prompt caching: the whole-doc context goes FIRST — it is identical
    # across a doc's batches and the initial→remap re-send, so it caches; the
    # varying vocab + triples + feedback follow.
    parts = [
        f"Document title: {doc_title}",
        "Full document (context):",
        doc_ctx,
        _fmt_vocab(schema),
        f"Raw triples to canonicalize ({len(batch)}):",
        triples_section,
    ]
    if feedback:
        parts.append(
            "Re-map: these mappings were flagged (deterministic form rules and/or "
            "the reviewer). Fix each — keep predicates active-voice, ≤3 words, and "
            "do not merge distinct relations:\n- "
            + "\n- ".join(feedback)
        )
    return "\n\n".join(parts)


def _build_reviewer_prompt(gated: list) -> str:
    lines = [
        "Mappings to check (raw → chosen canonical, with the gate decision and "
        "the Mapper's scores):"
    ]
    for g in gated:
        t = g.triple
        lines.append(
            f"[{t.triple_index}] raw=({t.raw_subject!r} {t.raw_predicate!r} "
            f"{t.raw_object!r}) → canonical=({t.canonical_subject!r} "
            f"{g.predicate!r} {t.canonical_object!r}); "
            f"mapped={g.mapped_flag}; confidence={t.mapper_confidence:.2f}; "
            f"newness={t.newness_score:.2f}"
        )
    return "\n".join(lines)


def _sample_mapper(batch: list[RawTriple]) -> MapperResult:
    """SAMPLE_MODE / failure fallback: identity map — raw forms become the
    canonical, each entity typed ``concept``, mid confidence/newness. Faithful
    and cheap; the gate + schema then route them normally."""

    names = {t.subject for t in batch} | {t.object for t in batch}
    return MapperResult(
        triples=[
            MappedTriple(
                triple_index=i,
                raw_subject=t.subject,
                raw_predicate=t.predicate,
                raw_object=t.object,
                canonical_subject=t.subject,
                canonical_object=t.object,
                canonical_predicate=t.predicate,
                mapper_confidence=0.5,
                newness_score=0.5,
                fallback_predicate=None,
            )
            for i, t in enumerate(batch)
        ],
        entity_types=[MappedEntityType(name=n, canonical_type="concept") for n in names],
    )


def _build_attribute_prompt(
    doc_title: str, doc_ctx: str, batch: list[RawAttribute], schema: KgSchemaView
) -> str:
    attrs_section = "\n".join(
        f"[{i}] entity={a.entity!r} property={a.property!r} value={a.value!r}"
        + (f" unit={a.unit!r}" if a.unit else "")
        + (f" condition={a.condition!r}" if a.condition else "")
        for i, a in enumerate(batch)
    )
    vocab = (
        "Current canonical properties: "
        + (", ".join(schema.property_types) or "(empty)")
    )
    # Whole-doc context first (stable across the doc's attribute batches → cached).
    return "\n\n".join([
        f"Document title: {doc_title}",
        "Full document (context):",
        doc_ctx,
        vocab,
        f"Raw attributes to canonicalize ({len(batch)}):",
        attrs_section,
    ])


def _sample_attribute_mapper(batch: list[RawAttribute]) -> AttributeMapperResult:
    """SAMPLE_MODE / failure fallback: identity map (raw property as canonical,
    no numeric parse); the gate then snake_case-normalizes the property name."""
    return AttributeMapperResult(
        attributes=[_identity_attribute(i, a) for i, a in enumerate(batch)]
    )


__all__ = [
    "AGENT_NAME",
    "ATTRIBUTE_AGENT_NAME",
    "REVIEWER_NAME",
    "MapDocResult",
    "RawAttribute",
    "RawTriple",
    "attribute_mapper_agent",
    "map_doc",
    "mapper_agent",
    "mapping_reviewer_agent",
]
