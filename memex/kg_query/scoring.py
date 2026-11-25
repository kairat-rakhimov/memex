"""Hypothesis ranking — pure scoring over grounded legs + a candidate vector.

A hypothesis is worth surfacing when its conclusion is **unstated** (novelty) but
its premises are **real** (support) and the mechanism is **coherent**
(plausibility). The novelty floor is the key guard: a candidate whose statement
embeds too close to an existing fact is a restatement, not an inference, and is
rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from memex.kg_query.deps import QueryDeps
from memex.kg_query.grounding import GroundedFact
from memex.kg_query.retrieval import align_query
from memex.kg_query.seeds import Seed

DEFAULT_NOVELTY_MAX = 0.85
DEFAULT_WEIGHTS = (0.25, 0.25, 0.20, 0.30)  # novelty, plausibility, support, structure
# Headline floor: a hypothesis below this composite is demoted to the appendix, so
# the report leads with fewer, higher-value leads rather than a fixed count.
DEFAULT_MIN_COMPOSITE = 0.5

# Structure = **graph-dependence**, lane-BLIND: how much the hypothesis needed the
# *assembled* graph vs a single source. Measured from the distinct documents its
# legs span — a claim resting on one source (or a single-chunk co-mention) is "one
# page"; one synthesising facts from several documents earned the graph. This
# replaces the old lane-keyed table (bridge=1.0 … property=0.55), which baked in a
# preference for a *kind* of hypothesis rather than a quality signal; it also
# subsumes the old contested-sprawl + collapsed-bridge demotions (both reduce to
# "few distinct sources"). Genuineness is handled upstream (ContestationJudge +
# explore_worthiness) and by the NoveltyJudge — not here.
_DEFAULT_STRUCTURE = 0.5


def structure_score(legs: list[GroundedFact]) -> float:
    """Graph-dependence of a hypothesis, from the distinct documents its legs span.
    1 source ⇒ 0.5 (or 0.62 if ≥2 distinct chunks of it); 2 ⇒ ~0.78; 3+ ⇒ 1.0."""

    if not legs:
        return _DEFAULT_STRUCTURE
    docs = len({f.source_doc_id for f in legs if f.source_doc_id})
    chunks = len({(f.source_doc_id, f.chunk_span) for f in legs if f.chunk_span is not None})
    if docs <= 1:
        return 0.62 if chunks >= 2 else _DEFAULT_STRUCTURE
    return round(min(1.0, 0.55 + 0.45 * min(1.0, (docs - 1) / 2.0)), 4)


# ── explore-worthiness: which seeds earn the expensive agentic exploration ──────
# Type-NEUTRAL by design: it scores how much genuine, groundable material surrounds
# a seed — not what KIND of hypothesis it is. No lane/topic preference; the only
# lane-aware term is a genuineness *filter* (a contested seed the ContestationJudge
# ruled sprawl is an artifact, not a real spot to investigate).


def _distinct_docs(legs: list[GroundedFact]) -> int:
    return len({f.source_doc_id for f in legs if f.source_doc_id})


def _entity_vec_lookup(deps: QueryDeps) -> dict[int, np.ndarray]:
    cached = deps._expansion.get("entity_vec_lookup")
    if cached is None:
        ids, matrix = deps.matrix("entity")
        cached = {int(ids[i]): matrix[i] for i in range(len(ids))} if matrix.size else {}
        deps._expansion["entity_vec_lookup"] = cached
    return cached  # type: ignore[return-value]


def _topic_relevance(deps: QueryDeps, entity_ids: list[int], topic_vec: np.ndarray | None) -> float:
    """Gentle multiplier in [0.5, 1.0] from the seed's most on-topic entity (cosine
    to the research-topic vector). Down-weights tangents (e.g. a natural-gas-vehicle
    seed under a green-hydrogen topic) without hard-cutting — NOT a topic *preference*
    among on-topic seeds, just a relevance floor on off-topic ones."""

    if topic_vec is None or topic_vec.size == 0 or not entity_ids:
        return 1.0
    ids, matrix = deps.matrix("entity")
    if matrix.size == 0:
        return 1.0
    tvec = align_query(topic_vec, matrix)
    lut = _entity_vec_lookup(deps)
    sims = [float(np.dot(lut[e], tvec)) for e in entity_ids if e in lut]
    if not sims:
        return 1.0
    return 0.5 + 0.5 * max(0.0, min(1.0, max(sims)))


def explore_worthiness(
    deps: QueryDeps,
    seed: Seed,
    legs: list[GroundedFact],
    *,
    topic_vec: np.ndarray | None = None,
) -> float:
    """How likely investigating this seed is to pay off — predicted from
    type-neutral signals only:

    - **cross-source span** (distinct documents the legs span) — the strongest
      non-obviousness proxy: facts only ever co-stated in one place are already an
      explicit link, not a discovery; facts from separate papers are a real one.
    - **source substance** — the legs are backed by real verbatim sentences, so the
      explorer has something to read, not stubs.
    - **genuineness** — a contested seed ruled sprawl by the ContestationJudge is an
      artifact (heavy demote). Bridges need no special case: a collapsed one spans a
      single chunk → cross-source ≈ 0 → low naturally.
    - **reliability floor** — pure one-mention fluke entities are down-weighted
      slightly (noise filter, NOT a popularity reward).
    """

    if not legs:
        return 0.0
    docs = _distinct_docs(legs)
    chunks = {(f.source_doc_id, f.chunk_span) for f in legs if f.chunk_span is not None}
    cross_source = min(1.0, (docs - 1) / 2.0) if docs else 0.0
    if cross_source == 0.0 and len(chunks) >= 2:
        cross_source = 0.25  # distinct chunks within one doc — still beyond co-mention
    qlens = [len(f.source_quote or "") for f in legs]
    substance = sum(min(1.0, q / 120.0) for q in qlens) / len(qlens)

    genuine_mult = (
        0.3 if (seed.lane == "relation_contested" and seed.genuine_conflict is False) else 1.0
    )
    g = deps.graph()
    fluke = bool(seed.entity_ids) and all(
        g.nodes.get(e, {}).get("mentions", 0) <= 1 and g.nodes.get(e, {}).get("degree", 0) <= 1
        for e in seed.entity_ids
    )
    reliability = 0.85 if fluke else 1.0
    relevance = _topic_relevance(deps, seed.entity_ids, topic_vec)

    base = 0.6 * cross_source + 0.4 * substance
    return round(genuine_mult * reliability * relevance * base, 4)


def restatement_similarity(deps: QueryDeps, candidate_vec: np.ndarray) -> float:
    """Max cosine of the candidate statement to ANY existing fact embedding
    (relation + attribute). High ⇒ the claim is already stated."""

    if candidate_vec.size == 0:
        return 0.0
    best = 0.0
    for kind in ("relation", "attribute"):
        _ids, matrix = deps.matrix(kind)
        if matrix.size == 0:
            continue
        best = max(best, float(np.max(matrix @ align_query(candidate_vec, matrix))))
    return best


def novelty_score(restatement_sim: float) -> float:
    """1 − restatement similarity, clamped. Mid-range = genuinely inferential."""

    return max(0.0, min(1.0, 1.0 - restatement_sim))


def is_restatement(restatement_sim: float, *, novelty_max: float = DEFAULT_NOVELTY_MAX) -> bool:
    return restatement_sim >= novelty_max


def support_score(legs: list[GroundedFact]) -> float:
    """Leg confidence scaled by grounding **breadth** (how many distinct facts back
    the hypothesis, log-saturating ~4) and cross-source **corroboration**. A claim
    resting on one fact scores well below one resting on several corroborated ones."""

    if not legs:
        return 0.0
    confs = [leg.confidence for leg in legs if leg.confidence is not None]
    conf = (sum(confs) / len(confs)) if confs else 0.6
    breadth = min(1.0, math.log1p(len(legs)) / math.log(5))
    mean_corr = sum(leg.corroboration for leg in legs) / len(legs)
    corr_factor = min(1.0, math.log1p(mean_corr) / math.log(3))
    return max(0.0, min(1.0, conf * (0.6 * breadth + 0.4 * corr_factor)))


@dataclass(frozen=True)
class ScoreBreakdown:
    novelty: float
    plausibility: float
    support: float
    structure: float
    composite: float


def composite_score(
    *,
    novelty: float,
    plausibility: float,
    support: float,
    structure: float,
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
) -> ScoreBreakdown:
    wn, wp, ws, wst = weights
    comp = wn * novelty + wp * plausibility + ws * support + wst * structure
    return ScoreBreakdown(
        novelty=round(novelty, 4),
        plausibility=round(plausibility, 4),
        support=round(support, 4),
        structure=round(structure, 4),
        composite=round(comp, 4),
    )
