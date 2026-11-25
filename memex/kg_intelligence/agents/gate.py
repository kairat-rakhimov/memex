"""Deterministic FORM gate for Mapper-canonicalized vocabulary.

The Mapper preserves *meaning* — distinct relations stay
distinct, count is not a defect. The gate enforces, model-independently, only
the *form* a net-new canonical predicate must have to be admitted:

  * canonical already in the schema  → commit to it (no form check; grandfathered)
  * net-new canonical that clears the FORM rules → admit it
  * net-new that FAILS a form rule    → emit a precise per-triple ``violation``
                                        (the Mapper re-maps with that feedback);
                                        if it still fails, fall back to
                                        ``fallback_predicate`` (an existing
                                        canonical), else commit raw ``mapped_flag=0``

The form rules are the guarantee the prompt can't give (prompts leak ~4 passive
+ ~10 clause predicates / 120): a passive / clause / non-English predicate can
NEVER enter ``kg_schema``. ``gate_mapping`` stays a pure function — the
orchestration loops on ``violations`` (re-map) and applies the ``kg_schema`` +
``schema_changes`` writes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from memex.kg_intelligence.agents.schemas import (
    AttributeMapperResult,
    MappedAttribute,
    MappedTriple,
    MapperResult,
)
from memex.kg_intelligence.views import KgSchemaView

_MAX_LABEL_LEN = 60
# A predicate is a relation phrase, not a clause — at most this many word tokens.
_MAX_PREDICATE_TOKENS = 3

# Foreign-language indicator tokens that are unambiguously not English. A
# predicate containing any of these on its token split is force-rejected
# regardless of the model verdict — the model-independent English backstop the
# old SchemaValidator carried (e.g. `es_promovido_por`, `otorga_subvenciones_a`).
_NON_ENGLISH_PREDICATE_TOKENS: frozenset[str] = frozenset({
    # Spanish: copulas, prepositions, common verbs from the observed smells.
    "es", "son", "por", "del", "los", "las", "que", "con", "para",
    "sin", "sobre", "una", "como", "muy",
    "otorga", "subvenciones", "promovido", "promueve", "promovida",
    "tiene", "hace", "esta",
    # French: copulas, conjunctions, prepositions.
    "est", "les", "des", "ou", "pour", "avec", "sans", "mais",
    "sont", "pas",
    # German: articles, copulas, prepositions.
    "der", "die", "das", "und", "ist", "sind", "mit", "von", "zu",
    # Italian-specific articles.
    "della", "dello", "degli",
})


# camelCase boundary: lower/digit followed by an uppercase letter.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _label_tokens(name: str) -> list[str]:
    """Split a label into lowercased word tokens, honouring snake_case,
    kebab-case AND camelCase. ``esPromovidoPor`` → ``[es, promovido, por]`` so
    the denylist sees the foreign function words it would otherwise miss when
    the whole label collapsed to one token."""

    spaced = _CAMEL_BOUNDARY.sub(" ", name)
    return [t.lower() for t in re.split(r"[^A-Za-z]+", spaced) if t]


def predicate_is_non_english(name: str) -> bool:
    """Two-layer English backstop on a candidate predicate name. This is the
    deterministic net only; the Extractor + Mapper read any language and
    translate to English, so this just catches residual leakage.

    Layer 1: non-ASCII characters — catches accented chars (á, é, ñ, ü) and
    any non-Latin script. Layer 2: a camelCase/snake_case-aware token split
    against a small denylist of foreign function words. Returns True if either
    fires."""

    if not name or not name.isascii():
        return True
    return any(t in _NON_ENGLISH_PREDICATE_TOKENS for t in _label_tokens(name))


def _valid_label(name: str) -> bool:
    name = (name or "").strip()
    return bool(name) and len(name) <= _MAX_LABEL_LEN


_PASSIVE_LEADS = frozenset({"is", "are", "was", "be", "been"})

# A trailing preposition on a 2-token verb+prep (`used_in`, `depends_on`,
# `part_of`) is a clean relation; on a 3-token predicate it almost always means
# an object noun got baked into the middle (`performance_depends_on`,
# `enables_generation_of`, `sets_standards_for`) — a clause, not a relation.
_TRAILING_PREPS = frozenset({
    "of", "for", "in", "on", "with", "from", "at", "into", "across",
    "between", "over", "under", "through", "about", "against", "among",
    "toward", "towards", "onto", "upon",
})


def _is_passive(name: str) -> bool:
    """Passive form: a copula lead (`is_`/`are_`/`was_`) or a `_by` agent tail.
    Token-based so camelCase (`isProducedBy`) is caught too. Stative `has_*`
    (e.g. `has_property`) is allowed — it's a legitimate attribute relation, not
    the direction-muddled passive that `is_*`/`*_by` are."""
    toks = _label_tokens(name)
    if not toks:
        return False
    return toks[0] in _PASSIVE_LEADS or toks[-1] == "by"


def _is_clause(name: str) -> bool:
    """A clause, not a relation phrase: more than ``_MAX_PREDICATE_TOKENS`` word
    tokens, a modal/purpose ``_to_`` infix (`aims_to_reduce`, `expected_to_grow`),
    or a 3-token form ending in a preposition — the latter bakes an object noun
    into the predicate (`performance_depends_on`, `enables_generation_of`,
    `sets_standards_for`). A 2-token verb+prep (`leads_to`, `part_of`, `used_in`,
    `depends_on`) is a clean relation and stays allowed."""
    n = (name or "").strip().lower()
    toks = _label_tokens(n)
    if len(toks) > _MAX_PREDICATE_TOKENS or "_to_" in n:
        return True
    return len(toks) >= 3 and toks[-1] in _TRAILING_PREPS


def predicate_form_violation(name: str) -> str | None:
    """The deterministic FORM contract for a NET-NEW canonical predicate. Returns
    an actionable feedback string when ``name`` violates a rule, else ``None``.
    The string is fed back to the Mapper for a bounded re-map."""

    if not _valid_label(name):
        return f"empty or over-long label (>{_MAX_LABEL_LEN} chars); use a short relation phrase"
    if predicate_is_non_english(name):
        return "non-English predicate; translate it to an English relation"
    if _is_passive(name):
        return (
            "passive form (is_/are_/was_ lead or _by tail); rewrite active and "
            "FLIP subject↔object (e.g. `is_produced_by` → `produces`)"
        )
    if _is_clause(name):
        return (
            f"a clause, not a relation — keep it ≤{_MAX_PREDICATE_TOKENS} active "
            "words and move any object/modality out of the predicate "
            "(e.g. `aims_to_reduce_cost_of` → `reduces`)"
        )
    return None


GateEvent = tuple[str, str, str, str]  # (event, type_kind, name, reason)


@dataclass(frozen=True)
class GatedTriple:
    """One triple after gating: ``predicate`` is the final form to store
    (canonical / fallback / raw) and ``mapped_flag`` records whether it
    canonicalized successfully."""

    triple: MappedTriple
    predicate: str
    mapped_flag: bool


@dataclass
class GateResult:
    gated: list[GatedTriple] = field(default_factory=list)
    admitted_predicates: list[str] = field(default_factory=list)
    admitted_entity_types: list[str] = field(default_factory=list)
    rejected_predicates: list[str] = field(default_factory=list)
    events: list[GateEvent] = field(default_factory=list)
    # (triple_index, feedback) for net-new predicates that failed a FORM rule —
    # the orchestration re-maps these with the feedback. On the final pass they
    # are already routed to fallback/raw in ``gated``.
    violations: list[tuple[int, str]] = field(default_factory=list)
    # entity-name (lowercased) → final normalized canonical type. The merge
    # types each entity from this map (one type per entity), not per-triple.
    entity_types: dict[str, str] = field(default_factory=dict)


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _normalize_type(t: str) -> str:
    """Deterministic FORM-normalize an entity type to snake_case: lowercase,
    non-alphanumeric runs → ``_``, trimmed. Unlike predicates (which need the
    model to flip direction), an entity type's form is mechanically fixable —
    ``"Chemical Compound"`` → ``chemical_compound``. Returns "" when empty or
    more than three word tokens (a type should be a short noun, not a phrase)."""
    s = re.sub(r"[^a-z0-9]+", "_", (t or "").strip().lower()).strip("_")
    if not s or len(s.split("_")) > 3:
        return ""
    return s


# A property NAME is a short noun phrase; allow one more token than a type
# ("charge transfer coefficient", "faradaic efficiency").
_MAX_PROPERTY_TOKENS = 4


def normalize_property(name: str) -> str:
    """Deterministic FORM-normalize a property NAME to snake_case. Like an
    entity type, a property is a short noun phrase whose
    form is mechanically fixable — ``"Tafel slope"`` → ``tafel_slope`` — so no
    re-map is needed. Returns "" when empty, non-English, or more than
    ``_MAX_PROPERTY_TOKENS`` word tokens (a property name, not a clause)."""
    if predicate_is_non_english(name):
        return ""
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    if not s or len(s.split("_")) > _MAX_PROPERTY_TOKENS:
        return ""
    return s


def gate_mapping(result: MapperResult, schema: KgSchemaView) -> GateResult:
    """Route every Mapper triple to a final predicate + ``mapped_flag``, admit
    net-new canonicals that clear the FORM rules, and report the form violations
    (with feedback) for the orchestration to re-map. Existing canonicals are
    grandfathered (no form check); net-new is admitted on FORM alone — predicate
    count / newness is not a gate criterion."""

    pred_canon = {_norm(p): p for p in schema.relation_types}
    pred_alias = {_norm(a): c for a, c in schema.predicate_aliases}
    type_canon = {_norm(t): t for t in schema.entity_types}
    type_alias = {_norm(a): c for a, c in schema.entity_type_aliases}

    out = GateResult()
    new_pred_seen: set[str] = set()
    new_type_seen: set[str] = set()

    def _fallback_or_raw(t: MappedTriple) -> tuple[str, bool]:
        fb = _norm(t.fallback_predicate) if t.fallback_predicate else ""
        if fb and fb in pred_canon:
            return pred_canon[fb], True
        if fb and fb in pred_alias:
            return pred_alias[fb], True
        return t.raw_predicate, False

    for t in result.triples:
        key = _norm(t.canonical_predicate)
        if key in pred_canon:
            out.gated.append(GatedTriple(t, pred_canon[key], True))
        elif key in pred_alias:
            out.gated.append(GatedTriple(t, pred_alias[key], True))
        else:
            # net-new predicate proposal — admit on FORM alone. Lowercased so
            # case-variants (`has_Tafel_slope` / `has_tafel_slope`) collapse onto
            # one canonical instead of splitting the vocabulary.
            violation = predicate_form_violation(t.canonical_predicate)
            if violation is None:
                canon_pred = t.canonical_predicate.strip().lower()
                out.gated.append(GatedTriple(t, canon_pred, True))
                if key not in new_pred_seen:
                    new_pred_seen.add(key)
                    out.admitted_predicates.append(canon_pred)
                    out.events.append((
                        "accepted", "relation_type", canon_pred,
                        "net-new canonical admitted",
                    ))
            else:
                pred, flag = _fallback_or_raw(t)
                out.gated.append(GatedTriple(t, pred, flag))
                out.rejected_predicates.append(t.canonical_predicate)
                out.violations.append((t.triple_index, f"{t.canonical_predicate!r}: {violation}"))
                out.events.append((
                    "rejected", "relation_type", t.canonical_predicate,
                    f"form violation ({violation}) → {'fallback' if flag else 'raw'}",
                ))

    # Entity types — assigned once per distinct entity (not per-triple). Each is
    # FORM-normalized to snake_case deterministically (no re-map needed: a type's
    # form is mechanically fixable, unlike a predicate's direction). Non-English
    # types are dropped (the entity is left untyped rather than typed in a foreign
    # word). A normalized type new to the schema is admitted.
    for et in result.entity_types:
        # English-check the RAW type first — normalizing strips non-ASCII chars,
        # which would hide a foreign type (`matériel` → `mat_riel`).
        if predicate_is_non_english(et.canonical_type):
            continue
        norm_type = _normalize_type(et.canonical_type)
        if not norm_type:
            continue
        out.entity_types[_norm(et.name)] = norm_type
        tkey = norm_type.lower()
        if tkey in type_canon or tkey in type_alias or tkey in new_type_seen:
            continue
        new_type_seen.add(tkey)
        out.admitted_entity_types.append(norm_type)
        out.events.append((
            "accepted", "entity_type", norm_type, "net-new canonical type admitted",
        ))

    return out


# Bare comparative / quantity content words that carry no standalone meaning as a
# value ("higher" than what? "significant" by what measure?). A NON-numeric value
# made only of these (after stripping stopwords) is a hand-wave, not a grade —
# dropped so it can't bloat property sprawl or pollute chat retrieval.
# DELIBERATELY conservative: standalone qualitative grades (`high`, `low`,
# `moderate`, `notable`, `limited`, `p-type`, `Class III`) are NOT here — only the
# *relative / vague-quantity* words that are meaningless without a referent. The
# bar for inclusion: the word ALONE cannot be a real value. False negatives (a
# borderline kept) are safe; false positives (a real grade dropped) are not.
_VAGUE_VALUE_TOKENS: frozenset[str] = frozenset({
    "higher", "lower", "greater", "lesser", "larger", "smaller",
    "more", "less", "fewer",
    "increased", "decreased", "reduced", "increasing", "decreasing",
    "rising", "falling", "growing", "declining", "improved", "enhanced",
    "better", "worse",
    "significant", "substantial", "considerable", "minimal", "negligible",
    "much", "many", "several", "various", "varying", "varied",
    "numerous", "multiple",
})

# Stripped before the all-vague test so "relatively higher" / "significantly
# reduced" / "a significant amount" still reduce to a bare hand-wave. Intensifier
# adverbs live here (not in the vague set) so a grade they modify is judged on the
# grade, not the adverb.
_VALUE_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "to", "in", "is", "are", "be", "was", "were",
    "very", "quite", "relatively", "somewhat", "rather", "fairly", "too",
    "slightly", "significantly", "substantially", "considerably", "minimally",
    "marginally", "moderately", "appreciably", "notably", "highly",
    "amount", "amounts", "level", "levels", "degree", "value", "values",
    "than", "and", "or", "but", "with",
})

# Token ceiling for a NON-numeric value before it's treated as prose. Tuned on
# values from a real 80-doc build, not guessed: real categorical values run up to
# ~7-8 tokens ("20-40% NaOH or KOH in water", "Research Development and
# Demonstration (RD&D) stage"), while the ≥11-token band is almost purely
# definitions / clauses. Set DELIBERATELY high (precision over recall) so a real
# value is never dropped for length; the 7-10 token prose that survives is the
# safe direction (it was kept before this gate existed).
_MAX_VALUE_TOKENS = 10


def attribute_value_violation(value_raw: str, value_num: float | None) -> str | None:
    """Deterministic VALUE-quality check for one attribute. Numeric values
    (``value_num`` set, incl. ranges that parsed a number) always pass: they are
    the comparable lane that condition and unit normalisation serve. A NON-numeric value is dropped when it is
    **sentence-length prose** (more than ``_MAX_VALUE_TOKENS`` word tokens) or a
    **bare vague comparative/intensifier** (`higher`, `significant`, `much`)
    carrying no standalone meaning. Legitimate short qualitative grades
    (`p-type`, `monoclinic`, `Class III`) pass; empty values are out of scope
    (handled upstream). Returns a drop-reason string, else ``None``."""

    if value_num is not None:
        return None
    toks = [t for t in re.split(r"[^A-Za-z0-9]+", (value_raw or "").lower()) if t]
    if not toks:
        return None
    if len(toks) > _MAX_VALUE_TOKENS:
        return f"sentence-length value (>{_MAX_VALUE_TOKENS} tokens), not a value"
    meaningful = [t for t in toks if t not in _VALUE_STOPWORDS]
    if meaningful and all(t in _VAGUE_VALUE_TOKENS for t in meaningful):
        return "vague comparative/intensifier, no standalone meaning"
    return None


@dataclass(frozen=True)
class GatedAttribute:
    """One attribute after gating: ``property`` is the final form to store
    (canonical / normalized net-new / raw) and ``mapped_flag`` records whether it
    canonicalized successfully. ``dropped`` flags a non-numeric vague/sentence
    value the A3 value-gate rejected — the caller skips storing it."""

    attribute: MappedAttribute
    property: str
    mapped_flag: bool
    dropped: bool = False


@dataclass
class AttributeGateResult:
    gated: list[GatedAttribute] = field(default_factory=list)
    admitted_properties: list[str] = field(default_factory=list)
    events: list[GateEvent] = field(default_factory=list)


def gate_attributes(
    result: AttributeMapperResult, schema: KgSchemaView
) -> AttributeGateResult:
    """Route each attribute's ``canonical_property`` to a final property name +
    ``mapped_flag``. Existing canonical → commit; a net-new
    property whose snake_case-normalized form clears the English check → admit it
    as a ``property_type``; a malformed / non-English property falls back to the
    raw property (``mapped_flag=0``). Like entity types, a property name's form is
    mechanically fixable — deterministic normalize, no re-map."""

    prop_canon = {_norm(p): p for p in schema.property_types}
    out = AttributeGateResult()
    new_seen: set[str] = set()

    for a in result.attributes:
        # Value gate (deterministic). A vague/sentence non-numeric value is
        # dropped BEFORE property admission, so a junk value can't mint a property
        # type. Kept 1:1 in ``gated`` (the caller skips ``dropped`` rows).
        drop_reason = attribute_value_violation(a.raw_value, a.value_num)
        if drop_reason is not None:
            out.gated.append(GatedAttribute(a, a.raw_property, False, dropped=True))
            out.events.append((
                "dropped", "attribute_value",
                f"{a.canonical_property}={a.raw_value}"[:60], drop_reason,
            ))
            continue
        key = _norm(a.canonical_property)
        if key in prop_canon:
            out.gated.append(GatedAttribute(a, prop_canon[key], True))
            continue
        norm = normalize_property(a.canonical_property)
        if not norm:
            out.gated.append(GatedAttribute(a, a.raw_property, False))
            continue
        nkey = norm.lower()
        if nkey in prop_canon:
            out.gated.append(GatedAttribute(a, prop_canon[nkey], True))
            continue
        out.gated.append(GatedAttribute(a, norm, True))
        if nkey not in new_seen:
            new_seen.add(nkey)
            out.admitted_properties.append(norm)
            out.events.append((
                "accepted", "property_type", norm,
                "net-new canonical property admitted",
            ))
    return out


__all__ = [
    "AttributeGateResult",
    "GateEvent",
    "GateResult",
    "GatedAttribute",
    "GatedTriple",
    "attribute_value_violation",
    "gate_attributes",
    "gate_mapping",
    "normalize_property",
    "predicate_form_violation",
    "predicate_is_non_english",
]
