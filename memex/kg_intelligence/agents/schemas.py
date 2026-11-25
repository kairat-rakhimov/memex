"""Output schemas for the kg-loop agents."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# ── RoundPlanner ───────────────────────────────────────────────────────────


class RankedTopic(BaseModel):
    """One topic the round should pursue. Priority is the list position
    (index 0 = highest): the build leans effort toward the top topics and fills
    down when the top is thin. Source-agnostic — each adapter turns a topic into
    its own source-tailored queries."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    rationale: str = ""


class RoundPlan(BaseModel):
    """The RoundPlanner's output: a ranked topic list (priority = list order)
    plus a per-active-adapter corpus-fit signal. Source-agnostic — adapters turn
    topics into source-tailored queries; the global Selector arbitrates the
    cross-source candidate pool by priority-weighted goal-fit. The topic strings
    are the goal-identity handles the RoundReviewer tracks across rounds."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    topics: list[RankedTopic] = Field(default_factory=list)
    adapter_relevance: dict[str, Literal["low", "medium", "high"]] = Field(
        default_factory=dict
    )


# ── Selector ─────────────────────────────────────────────────────────────


class SelectorPick(BaseModel):
    """One candidate the global Selector chose from the cross-source pool.
    ``candidate_index`` points at the candidate's position in the flat pool
    the Selector was shown. ``goal_relevance`` (0-1) is the Selector's
    priority-weighted goal-fit judgment, stamped onto the chosen candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_index: int = Field(ge=0)
    goal_relevance: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class SelectorResult(BaseModel):
    """Global Selector output: the ≤budget picks for the round, arbitrated by
    priority-weighted goal-fit across the whole cross-source pool (not a
    per-source quota). Order conveys the Selector's preference; the
    orchestration truncates to the budget."""

    model_config = ConfigDict(extra="forbid")

    picks: list[SelectorPick] = Field(default_factory=list)


# ── UsefulnessGate ───────────────────────────────────────────────────────


class UsefulnessResult(BaseModel):
    """The UsefulnessGate's per-doc output, shared across BOTH passes so the
    [schema + whole-doc] prefix is identical and the second pass can hit the
    first pass's prompt cache (doc-first layout).

    Pass 1 (usefulness) fills ``usefulness`` / ``decision`` / ``reason`` only
    and leaves the rest empty — a ``skip`` short-circuits extraction + the
    Mapper on the doc. Pass 2 runs ONLY for keepers and fills ``summary`` (a
    durable artifact) + the corpus-state ``reflection`` fields. ``decision``
    is the gate verdict; ``usefulness`` ≥ 0.4 ⇒ keep."""

    model_config = ConfigDict(extra="forbid")

    usefulness: float = Field(ge=0.0, le=1.0)
    decision: Literal["keep", "skip"]
    reason: str = ""
    summary: str = ""
    entities_seen: list[str] = Field(default_factory=list)
    subtopics_covered: list[str] = Field(default_factory=list)
    subtopics_missing: list[str] = Field(default_factory=list)
    borders: list[str] = Field(default_factory=list)


# ── Extractor ──────────────────────────────────────────────────────────────


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str
    source_quote: str = ""  # verbatim sentence from the doc stating this relation


class ExtractedAttribute(BaseModel):
    """A property OF an entity whose object is a VALUE, not another entity.
    The Extractor captures the parts LOOSELY (faithful
    phrases); the Mapper canonicalizes the property name + parses value/unit.
    ``entity`` should appear in the entities list. ``unit`` / ``condition`` are
    "" when not separable from ``value`` / not stated."""

    model_config = ConfigDict(extra="forbid")

    entity: str
    property: str
    value: str
    unit: str = ""
    condition: str = ""
    source_quote: str = ""  # verbatim sentence from the doc stating this attribute


class ExtractionResult(BaseModel):
    """``_agent_call_id`` is a PrivateAttr (invisible to the LLM) that
    ``extract()`` stamps after the call so the pipeline can attribute
    ``kg_entities`` / ``kg_relations`` rows to the originating call.

    ``relations`` are entity→entity links; ``attributes`` are entity→value
    properties. The split is the Extractor's call: object is
    a real entity → relation; object is a measured/qualitative value → attribute.
    """

    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    attributes: list[ExtractedAttribute] = Field(default_factory=list)
    _agent_call_id: int | None = PrivateAttr(default=None)


# ── Mapper ─────────────────────────────────────────────────────────────────


class MappedTriple(BaseModel):
    """One canonicalized triple from the per-doc Mapper.

    ``triple_index`` echoes the position of the raw triple in the Mapper's
    input list — the orchestration ties the canonical form back to its
    ``chunk_id`` by index (robust against the model lightly editing the raw
    surface strings). ``canonical_subject`` / ``canonical_object`` preserve the
    surface form; entity *identity* dedup is the Resolver's job downstream, NOT
    the Mapper's, and entity *types* live once-per-entity in
    ``MapperResult.entity_types``, not here. ``fallback_predicate`` is the
    nearest existing canonical, emitted whenever ``canonical_predicate`` is a
    net-new proposal so a gate-reject collapses to existing vocabulary instead
    of going raw.
    """

    model_config = ConfigDict(extra="forbid")

    triple_index: int = Field(ge=0)
    raw_subject: str
    raw_predicate: str
    raw_object: str
    canonical_subject: str
    canonical_object: str
    canonical_predicate: str
    mapper_confidence: float = Field(ge=0.0, le=1.0)
    newness_score: float = Field(ge=0.0, le=1.0)
    fallback_predicate: str | None = None


class MappedEntityType(BaseModel):
    """A type assigned ONCE to a distinct entity (not per-triple-endpoint). The
    Mapper sees the whole doc + all triples, so it types each entity
    consistently. ``name`` matches the ``canonical_subject`` / ``canonical_object``
    surface form the Mapper used in its triples."""

    model_config = ConfigDict(extra="forbid")

    name: str
    canonical_type: str


class MapperResult(BaseModel):
    """Per-doc Mapper output: one canonical triple per input raw triple (1:1; the
    Mapper canonicalizes vocabulary, it does not split or invent triples), plus
    one type per distinct entity. The MappingReviewer may flag a subset to
    re-map or drop."""

    model_config = ConfigDict(extra="forbid")

    triples: list[MappedTriple] = Field(default_factory=list)
    entity_types: list[MappedEntityType] = Field(default_factory=list)


# ── Attribute Mapper ──────────────────────────────────────


class MappedAttribute(BaseModel):
    """One canonicalized attribute from the per-doc attribute Mapper.
    ``attribute_index`` echoes the input position so the
    canonical form ties back to its ``chunk_id``. The Mapper canonicalizes the
    property NAME, parses ``value_num`` + ``unit`` out of the raw value, and
    normalizes ``condition``; ``canonical_entity`` preserves the surface form
    (identity is the Resolver's job). ``value_num`` is null when the value is
    non-numeric (a qualitative grade). ``fallback_property`` is the nearest
    existing canonical when a net-new property is proposed."""

    model_config = ConfigDict(extra="forbid")

    attribute_index: int = Field(ge=0)
    raw_entity: str
    raw_property: str
    raw_value: str
    canonical_entity: str
    canonical_property: str
    value_num: float | None = None
    unit: str = ""
    condition: str = ""
    mapper_confidence: float = Field(ge=0.0, le=1.0)
    newness_score: float = Field(ge=0.0, le=1.0)
    fallback_property: str | None = None


class AttributeMapperResult(BaseModel):
    """Per-doc attribute Mapper output: one canonical attribute per input raw
    attribute (1:1; the Mapper canonicalizes, it does not split or invent)."""

    model_config = ConfigDict(extra="forbid")

    attributes: list[MappedAttribute] = Field(default_factory=list)


# ── MappingReviewer ──────────────────────────────────────────────────────────


MapperFailureMode = Literal[
    "over_collapse",   # merged distinct / opposite relations → re-map
    "spurious_new",    # an existing candidate fit; net-new was unwarranted → re-map
    "direction",       # subject/object flipped → re-map
    "type_mismatch",   # canonical types don't fit the entities → re-map
    "unfaithful",      # the source text doesn't support the triple → drop / down-weight
]


class MapperSuspect(BaseModel):
    """One flagged mapping from the adversarial MappingReviewer. ``triple_index``
    points at the ``MappedTriple`` it suspects. The four canonicalization
    failure modes route to a bounded re-map; ``unfaithful`` routes to the
    precision path (drop or down-weight)."""

    model_config = ConfigDict(extra="forbid")

    triple_index: int = Field(ge=0)
    failure_mode: MapperFailureMode
    why: str = ""
    suggested_fix: str = ""


class MappingReviewResult(BaseModel):
    """MappingReviewer output — SUSPECTS ONLY (asymmetry, not re-derivation)."""

    model_config = ConfigDict(extra="forbid")

    suspects: list[MapperSuspect] = Field(default_factory=list)


# ── Reconciler ─────────────────────────────────────────────────────────────


class ReconcilerResult(BaseModel):
    """Relations only; Reconciler does not propose new entities."""

    model_config = ConfigDict(extra="forbid")

    relations: list[ExtractedRelation] = Field(default_factory=list)
    _agent_call_id: int | None = PrivateAttr(default=None)


# ── KGReviewer ─────────────────────────────────────────────────────────────


GapKind = Literal[
    "low_degree",
    "disconnected",
    "weak_relation",
    "undefined_concept",
    "breadth",
]


class KgGap(BaseModel):
    """``rank`` is 0-indexed importance (0 = most important); next-round
    Planner weights query allocation by it."""

    model_config = ConfigDict(extra="forbid")

    description: str
    kind: GapKind = "low_degree"
    rank: int = Field(default=0, ge=0)


class KgReviewResult(BaseModel):
    """KGReviewer output — structural gaps only. The stop signal lives in the
    RoundReviewer: a complete KG has no hypotheses left, so
    "coverage adequate" is the wrong stop; ROI is."""

    model_config = ConfigDict(extra="forbid")

    gaps: list[KgGap] = Field(default_factory=list)


# ── RoundReviewer ──────────────────────────────────────────────────────────


class RoundStatus(StrEnum):
    """Trajectory verdict. ``continuing``/``slowing`` keep the build going
    (``slowing`` ⇒ a pivot is worth trying); ``stalled``/``wasted`` halt."""

    continuing = "continuing"
    slowing = "slowing"
    stalled = "stalled"
    wasted = "wasted"


class GoalVerdict(StrEnum):
    pursue = "pursue"
    deprioritize = "deprioritize"
    abandon = "abandon"  # unanswerable — never retry


class GoalDirective(BaseModel):
    """Per-theme steer for the next RoundPlanner. ``theme`` matches a prior
    ``RoundPlan.topics`` entry. ``abandon`` themes accumulate into
    ``CorpusState.abandoned_goals`` and are hard-excluded next round."""

    model_config = ConfigDict(extra="forbid")

    theme: str
    verdict: GoalVerdict = GoalVerdict.pursue
    rationale: str = ""


class RoundReviewResult(BaseModel):
    """RoundReviewer output — the strategist's call. Owns the stop decision
    (``status``) and seeds the next round (``goal_directives``)."""

    model_config = ConfigDict(extra="forbid")

    status: RoundStatus = RoundStatus.continuing
    reason: str = ""
    goal_directives: list[GoalDirective] = Field(default_factory=list)


# ── Resolver ───────────────────────────────────────────────────────────────


ResolverVerdict = Literal["merge", "distinct", "ambiguous"]


class KgMergeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name_a: str
    name_b: str
    verdict: ResolverVerdict
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class ResolverResult(BaseModel):
    """Tier-2 LLM verdicts only; tier-1 deterministic merges are applied
    directly without going through this schema."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[KgMergeVerdict] = Field(default_factory=list)


# ── VocabResolver ──────────────────────────────────────────────────────────


VocabRelation = Literal["alias", "child_of", "unrelated"]


class VocabVerdict(BaseModel):
    """One cluster member's relationship to a chosen canonical within a
    candidate vocabulary cluster (predicates / properties). Non-destructive
    by design — see the relation semantics:

    - ``alias``: ``name`` is a synonym of ``canonical`` (same concept) — collapse
      ``name`` → ``canonical``, shrinking the vocabulary.
    - ``child_of``: ``name`` is a specialization of ``canonical`` (narrower) — keep
      ``name`` as its own canonical, record ``canonical`` as its parent for rollup.
    - ``unrelated``: merely nearby in wording — neither; leave ``name`` standalone.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    canonical: str
    relation: VocabRelation
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class VocabResolverResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[VocabVerdict] = Field(default_factory=list)
