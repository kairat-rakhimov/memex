"""Resolver: tiered entity-alias resolution over the canonical KG.

Tier 1 normalises names deterministically and merges collapses. Tier 2
asks the LLM to judge candidate pairs surfaced by Jaccard / token-subset
/ acronym heuristics. Every committed merge writes a ``merge`` mutation
plus one ``endpoint_rewrite`` child per affected relation.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import (
    AgentCallSinks,
    ModelPreference,
    execute_agent_call,
    record_skipped_agent_call,
)
from memex.kg_intelligence.agents.schemas import (
    KgMergeVerdict,
    ResolverResult,
)
from memex.kg_intelligence.embeddings import load_matrix, top_k_similar
from memex.observability import EntitySnapshot, TraceStore

_log = logging.getLogger(__name__)

TIER3_COSINE_THRESHOLD = 0.75
# Above this cosine, a tier-3 embedding pair is treated as a confident alias and
# bypasses the cooccurrence (shared-doc) filter — so cross-document spelling /
# morphology variants merge in any domain without hardcoded normalization rules.
TIER3_COOCCURRENCE_BYPASS = 0.88
TIER3_TOP_K = 5

AGENT_NAME = "Resolver"
ACTION = "resolve_entities"
MODEL_PREFERENCE: ModelPreference = "reasoning"
SWEEP_PERIOD = 5
JACCARD_THRESHOLD = 0.7
ACRONYM_MIN_LEN = 2
ACRONYM_MAX_LEN = 6
TIER2_BATCH_SIZE = 25
"""Per-LLM-call cap on candidate pairs; keeps each reasoning call short."""

_SYSTEM_PROMPT = """\
You are Resolver, an agent that decides whether each candidate pair of entity
names refers to the same real-world thing, or to two genuinely distinct things.

For each candidate pair, return a verdict:
- merge: the two names are aliases for the same entity. Examples:
  `PEM` ↔ `Proton exchange membrane electrolyzer` (acronym for full form),
  `analyses` ↔ `analysis` (morphological variant).
- distinct: the names look similar but refer to different things. Examples:
  `Metal hydrides` ↔ `Metal oxides` (same category, different materials),
  `iridium catalyst` ↔ `iridium oxide` (related but separate concepts).
- ambiguous: not enough information in the surrounding context to decide.

Each verdict carries:
- confidence ∈ [0, 1]: how sure you are. Low confidence on weak evidence.
- reason: one short clause naming the evidence (acronym match, morphology,
  shared sources, semantic divergence, etc.).

Be conservative on `merge`: a false positive corrupts the KG. When unsure,
return `ambiguous` or `distinct`.
"""

resolver_agent: Agent[None, ResolverResult] = Agent(
    model=None,
    output_type=ResolverResult,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


@dataclass(frozen=True)
class _MergePair:
    survivor: str
    merged: str
    reason: str
    confidence: float
    tier: int  # 1 = deterministic, 2 = LLM (Jaccard/subset/acronym), 3 = LLM (embedding)


def _pair_key(a: str, b: str) -> str:
    """Order- + case-insensitive key for the resolver-verdict cache."""
    x, y = sorted([(a or "").strip().lower(), (b or "").strip().lower()])
    return f"{x}||{y}"


# ── public entry point ────────────────────────────────────────────────────


async def resolve(
    *,
    store: TraceStore,
    round_number: int,
    entities_touched_this_round: set[str],
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    triggered_by: str = "round_extractions_done",
) -> list[_MergePair]:
    """Returns committed merges; applies them via ``store`` as a side effect."""

    snapshot = store.entity_snapshot_for_resolver()
    tier1 = _tier1_merges(snapshot)

    do_sweep = round_number % SWEEP_PERIOD == 0 and round_number > 0
    incremental = None if do_sweep else entities_touched_this_round
    candidates = _tier2_candidates(snapshot, incremental_set=incremental)

    new_entity_names = entities_touched_this_round - (
        set(snapshot.keys()) - entities_touched_this_round
    )
    tier3_scored = _tier3_embedding_candidates(snapshot, store, new_entity_names)
    existing_pairs = set(candidates)
    # Very-high-cosine embedding pairs are aliases regardless of shared docs —
    # cross-document spelling/morphology variants (`electrolyzer`/`electrolyser`)
    # share no source doc yet are the same entity. They BYPASS the cooccurrence
    # filter; weaker tier-3 pairs (0.75–bypass) stay subject to it like tier-2.
    # Domain-agnostic: any spelling variant in any corpus is caught via embeddings.
    confident_tier3: list[tuple[str, str]] = []
    weak_tier3 = 0
    for pair, score in tier3_scored:
        if pair in existing_pairs:
            continue
        existing_pairs.add(pair)
        if score >= TIER3_COOCCURRENCE_BYPASS:
            confident_tier3.append(pair)
        else:
            candidates.append(pair)
            weak_tier3 += 1
    if tier3_scored:
        _log.info(
            "Resolver tier-3: %d embedding candidates (%d confident bypass, %d weak)",
            len(tier3_scored), len(confident_tier3), weak_tier3,
        )

    candidates = _cooccurrence_filter(candidates, snapshot) + confident_tier3

    # Drop pairs already judged 'distinct' in a prior round/sweep —
    # the dominant Resolver re-judgment waste (stable non-matches re-surfaced
    # every round + by the full sweep).
    cached_distinct = store.cached_distinct_pairs()
    if cached_distinct:
        before = len(candidates)
        candidates = [
            p for p in candidates if _pair_key(p[0], p[1]) not in cached_distinct
        ]
        if before != len(candidates):
            _log.info(
                "Resolver: skipped %d cached-distinct pairs (memoized)",
                before - len(candidates),
            )

    if not tier1 and not candidates:
        record_skipped_agent_call(
            sinks=sinks,
            agent_name=AGENT_NAME,
            action=ACTION,
            triggered_by=triggered_by,
            target_agent="kg_entities",
            input_summary=(
                f"round={round_number}, entities={len(snapshot)}, "
                f"tier1_groups=0, tier2_candidates=0"
            ),
            output_summary="no merge candidates this round — skipped LLM call",
        )
        return []

    tier2: list[_MergePair] = []
    call_handle_id: int | None = None
    if candidates:
        batches = _batch_candidates(candidates, TIER2_BATCH_SIZE)
        n_batches = len(batches)
        _log.info(
            "Resolver tier-2: round=%d, candidates=%d, batches=%d (size=%d)",
            round_number, len(candidates), n_batches, TIER2_BATCH_SIZE,
        )
        all_verdicts: list[KgMergeVerdict] = []
        for batch_idx, batch in enumerate(batches, start=1):
            user_prompt = _build_user_prompt(
                batch, snapshot, do_sweep=do_sweep,
                batch_idx=batch_idx, batch_total=n_batches,
            )
            input_summary = (
                f"round={round_number}, batch={batch_idx}/{n_batches}, "
                f"pairs={len(batch)}, entities={len(snapshot)}, "
                f"tier1_groups={len(tier1)}, tier2_candidates={len(candidates)}, "
                f"sweep={do_sweep}"
            )
            result, call, run_result = await execute_agent_call(
                agent=resolver_agent,
                user_prompt=user_prompt,
                system_prompt=_SYSTEM_PROMPT,
                sample_factory=lambda b=batch: _sample_resolver(b),
                log_prefix=f"{AGENT_NAME}[{batch_idx}/{n_batches}]",
                settings=settings,
                llm_model=llm_model,
                sinks=sinks,
                agent_name=AGENT_NAME,
                action=ACTION,
                triggered_by=triggered_by,
                target_agent="kg_entities",
                input_summary=input_summary,
                output_summary_fn=lambda r: (
                    f"merge={sum(1 for v in r.verdicts if v.verdict == 'merge')}, "
                    f"distinct={sum(1 for v in r.verdicts if v.verdict == 'distinct')}, "
                    f"ambiguous={sum(1 for v in r.verdicts if v.verdict == 'ambiguous')}"
                ),
                confidence=0.7,
                model_preference=MODEL_PREFERENCE,
            )
            if len(result.verdicts) != len(batch) and run_result.status == "success":
                _log.warning(
                    "Resolver tier-2 batch %d/%d: model returned %d verdicts "
                    "for %d pairs (omission); accepting partial",
                    batch_idx, n_batches, len(result.verdicts), len(batch),
                )
            all_verdicts.extend(result.verdicts)
            if call_handle_id is None and call is not None and run_result.status == "success":
                call_handle_id = call.agent_call_id
        tier2 = _verdicts_to_pairs(all_verdicts, snapshot)
        store.record_resolver_verdicts(
            [_pair_key(v.name_a, v.name_b) for v in all_verdicts if v.verdict == "distinct"],
            round_number=round_number,
        )
        _log.info(
            "Resolver tier-2: round=%d, batches_completed=%d, verdicts=%d, "
            "merges=%d",
            round_number, n_batches, len(all_verdicts), len(tier2),
        )
    elif tier1:
        # Tier-1 merges still warrant a Resolver row in the trace; otherwise
        # the round timeline silently misses them.
        skipped = record_skipped_agent_call(
            sinks=sinks,
            agent_name=AGENT_NAME,
            action=ACTION,
            triggered_by=triggered_by,
            target_agent="kg_entities",
            input_summary=(
                f"round={round_number}, entities={len(snapshot)}, "
                f"tier1_groups={len(tier1)}, tier2_candidates=0"
            ),
            output_summary=(
                f"tier-1 only: {len(tier1)} deterministic merges, no LLM call"
            ),
        )
        if skipped is not None:
            call_handle_id = skipped.agent_call_id

    committed: list[_MergePair] = []
    for pair in tier1 + tier2:
        if pair.survivor == pair.merged:
            continue
        merged = store.apply_merge(
            survivor_name=pair.survivor,
            merged_name=pair.merged,
            agent_call_id=call_handle_id,
            round_number=round_number,
        )
        if merged is None:
            # Defensive: a transitive tier-1 chain may have already merged
            # one side. Skip silently.
            continue
        survivor_id, _merged_id, affected_relation_ids = merged
        committed.append(pair)
        if call_handle_id is not None:
            _log_merge_mutations(
                store=store,
                pair=pair,
                survivor_id=survivor_id,
                agent_call_id=call_handle_id,
                round_number=round_number,
                affected_relation_ids=affected_relation_ids,
            )

    return committed


# ── tier 1 ────────────────────────────────────────────────────────────────


def _normalize(name: str) -> str:
    """Tier-1 normalization: lower + punctuation strip + hyphen/underscore
    collapse + naive trailing-s strip. Misses irregular plurals + spelling
    variants by design — those fall through to the embedding tier (tier 3),
    which catches them in ANY domain without hardcoded morphology rules."""

    s = (name or "").lower().strip()
    s = re.sub(r"[^\w\s\-_]", "", s)
    s = re.sub(r"[-_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 3 and s.endswith("s"):
        s = s[:-1]
    return s


def _tier1_merges(snapshot: dict[str, EntitySnapshot]) -> list[_MergePair]:
    """Group entity names by normalised form; emit one merge per non-trivial
    group, with the alphabetically-earliest name elected survivor for
    determinism (same input → same output across runs)."""

    groups: dict[str, list[str]] = defaultdict(list)
    for name in snapshot:
        groups[_normalize(name)].append(name)

    pairs: list[_MergePair] = []
    for norm, members in groups.items():
        if len(members) < 2:
            continue
        members_sorted = sorted(members)
        survivor = members_sorted[0]
        for merged in members_sorted[1:]:
            pairs.append(
                _MergePair(
                    survivor=survivor,
                    merged=merged,
                    reason=f"tier-1 normalization match ({norm!r})",
                    confidence=1.0,
                    tier=1,
                )
            )
    return pairs


# ── tier 2 candidate selection ────────────────────────────────────────────


def _tier2_candidates(
    snapshot: dict[str, EntitySnapshot],
    *,
    incremental_set: set[str] | None,
) -> list[tuple[str, str]]:
    """All pairs of entity names that satisfy at least one of the three
    candidate rules. When ``incremental_set`` is non-None, restricts to pairs
    where at least one endpoint is in that set (per-round incremental scope)."""

    names = list(snapshot.keys())
    out: list[tuple[str, str]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if incremental_set is not None and a not in incremental_set and b not in incremental_set:
                continue
            if _is_candidate_pair(a, b):
                out.append((a, b))
    return out


def _is_candidate_pair(a: str, b: str) -> bool:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if ta and tb:
        jaccard = len(ta & tb) / len(ta | tb)
        if jaccard >= JACCARD_THRESHOLD:
            return True
        if ta < tb or tb < ta:
            return True
    return bool(_is_acronym_match(a, b))


def _is_acronym_match(a: str, b: str) -> bool:
    """One name is uppercase letters whose length matches a prefix of the
    word initials of the other. Catches `PEM` ↔ `Proton exchange membrane`
    (exact) and `PEM` ↔ `Proton exchange membrane electrolyzer` (prefix —
    acronym names the first N words, full name adds qualifiers)."""

    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    bare = short.replace(".", "")
    if not bare.isalpha():
        return False
    if not bare.isupper():
        return False
    if not (ACRONYM_MIN_LEN <= len(bare) <= ACRONYM_MAX_LEN):
        return False
    initials = "".join(w[0].upper() for w in long_.split() if w and w[0].isalpha())
    if not initials.startswith(bare):
        return False
    # Require the acronym to cover most of the long form to avoid false
    # positives like `PEM` ↔ `Proton`. Need at least 2/3 of long_'s words.
    return len(bare) >= max(ACRONYM_MIN_LEN, (2 * len(initials) + 2) // 3)


def _cooccurrence_filter(
    pairs: list[tuple[str, str]],
    snapshot: dict[str, EntitySnapshot],
) -> list[tuple[str, str]]:
    """Drop pairs whose source-doc sets do not intersect. Entities that never
    co-appear in any document are rarely aliases — free precision boost."""

    kept: list[tuple[str, str]] = []
    for a, b in pairs:
        ea = snapshot.get(a)
        eb = snapshot.get(b)
        if ea is None or eb is None:
            continue
        if set(ea.source_doc_ids) & set(eb.source_doc_ids):
            kept.append((a, b))
    return kept


# ── tier 3: embedding-similarity candidates ──────────────────────────────


def _tier3_embedding_candidates(
    snapshot: dict[str, EntitySnapshot],
    store: TraceStore,
    new_entity_names: set[str],
) -> list[tuple[tuple[str, str], float]]:
    """Surface entity pairs with high cosine similarity that tier-2 missed, each
    with its cosine score (the caller routes very-high-cosine pairs past the
    cooccurrence filter).

    Queries top-K for each new-this-round entity against the full entity
    embedding matrix. Only pairs above ``TIER3_COSINE_THRESHOLD`` are
    returned. Pure numpy dot product — no LLM calls.
    """

    ids, matrix = load_matrix(store, "entity")
    if matrix.size == 0 or not new_entity_names:
        return []

    id_to_name: dict[str, str] = {}
    for name, ent in snapshot.items():
        id_to_name[str(ent.id)] = name

    idx_by_id = {eid: i for i, eid in enumerate(ids)}

    pair_score: dict[tuple[str, str], float] = {}
    for name in new_entity_names:
        ent = snapshot.get(name)
        if ent is None:
            continue
        query_idx = idx_by_id.get(str(ent.id))
        if query_idx is None:
            continue
        query_vec = matrix[query_idx]
        hits = top_k_similar(query_vec, matrix, k=TIER3_TOP_K)
        for idx, score in hits:
            if score < TIER3_COSINE_THRESHOLD:
                continue
            hit_id = ids[idx]
            hit_name = id_to_name.get(hit_id)
            if hit_name is None or hit_name == name:
                continue
            pair = tuple(sorted([name, hit_name]))
            if score > pair_score.get(pair, 0.0):
                pair_score[pair] = score

    return list(pair_score.items())


# ── tier-2 result mapping ─────────────────────────────────────────────────


def _verdicts_to_pairs(
    verdicts: list[KgMergeVerdict], snapshot: dict[str, EntitySnapshot]
) -> list[_MergePair]:
    """Translate LLM verdicts into ``_MergePair`` records.

    For each ``merge`` verdict, pick the alphabetically-earliest name as
    survivor (matches tier-1 behaviour) and ignore the rest. Verdicts whose
    names don't exist in the snapshot are dropped (the LLM occasionally
    hallucinates names slightly off from inputs)."""

    pairs: list[_MergePair] = []
    for v in verdicts:
        if v.verdict != "merge":
            continue
        if v.name_a not in snapshot or v.name_b not in snapshot:
            continue
        ordered = sorted([v.name_a, v.name_b])
        pairs.append(
            _MergePair(
                survivor=ordered[0],
                merged=ordered[1],
                reason=v.reason or "tier-2 LLM-judged alias",
                confidence=v.confidence,
                tier=2,
            )
        )
    return pairs


# ── mutation log writes ───────────────────────────────────────────────────


def _log_merge_mutations(
    *,
    store: TraceStore,
    pair: _MergePair,
    survivor_id: int,
    agent_call_id: int,
    round_number: int,
    affected_relation_ids: list[int],
) -> None:
    """One parent ``merge`` row + N child ``endpoint_rewrite`` rows linked via
    ``parent_mutation_id``. ``target_id`` on the parent is the survivor
    canonical name (string); on each child it's ``kg_relations.id``."""

    parent_id = store.record_kg_mutation(
        kind="merge",
        target_kind="entity",
        target_id=pair.survivor,
        before={"name": pair.merged},
        after={"name": pair.survivor, "id": survivor_id},
        reason=pair.reason,
        confidence=pair.confidence if pair.tier == 2 else 1.0,
        agent_call_id=agent_call_id,
        round=round_number,
    )
    for rel_id in affected_relation_ids:
        store.record_kg_mutation(
            kind="endpoint_rewrite",
            target_kind="relation",
            target_id=str(rel_id),
            before={"merged_endpoint": pair.merged},
            after={"survivor_endpoint": pair.survivor, "survivor_entity_id": survivor_id},
            reason=f"endpoint rewritten after merge of {pair.merged!r} → {pair.survivor!r}",
            confidence=None,
            agent_call_id=agent_call_id,
            parent_mutation_id=parent_id,
            round=round_number,
        )


# ── prompt + sample factory ───────────────────────────────────────────────


def _cluster_candidates(
    candidates: list[tuple[str, str]],
) -> list[list[tuple[str, str]]]:
    """Group candidate pairs by connected component over their endpoint graph.

    Two pairs share a component when they share any endpoint name (directly
    or transitively). Keeping a cluster together in the eventual LLM batch
    prevents synonym groups (e.g. ``HER`` ↔ ``Hydrogen Evolution Reaction``
    ↔ ``HER (hydrogen evolution reaction)``) from being judged in separate
    prompts, which is the quality risk of naive index-based batching."""

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in candidates:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    by_root: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for a, b in candidates:
        by_root[find(a)].append((a, b))
    return list(by_root.values())


def _batch_candidates(
    candidates: list[tuple[str, str]], size: int
) -> list[list[tuple[str, str]]]:
    """Pack candidate pairs into batches of ~``size``, keeping each cluster
    (per ``_cluster_candidates``) intact in one batch.

    First-fit-decreasing bin-packing over clusters: largest cluster first,
    placed in the first batch with room, else a new batch. When a single
    cluster exceeds ``size`` it occupies its own oversized batch alone —
    we prefer one over-cap LLM call to fragmenting a synonym group across
    prompts. Empty input → empty list."""

    if size <= 0 or not candidates:
        return [candidates] if candidates else []
    clusters = sorted(_cluster_candidates(candidates), key=len, reverse=True)
    batches: list[list[tuple[str, str]]] = []
    for cluster in clusters:
        if len(cluster) > size:
            _log.warning(
                "Resolver tier-2: cluster of %d pairs exceeds batch size %d; "
                "shipping as a single oversized batch to keep synonyms together",
                len(cluster), size,
            )
            batches.append(list(cluster))
            continue
        placed = False
        for batch in batches:
            if len(batch) + len(cluster) <= size:
                batch.extend(cluster)
                placed = True
                break
        if not placed:
            batches.append(list(cluster))
    return batches


def _build_user_prompt(
    candidates: list[tuple[str, str]],
    snapshot: dict[str, EntitySnapshot],
    *,
    do_sweep: bool,
    batch_idx: int | None = None,
    batch_total: int | None = None,
) -> str:
    scope = "periodic full-KG sweep" if do_sweep else "per-round incremental"
    header = f"Resolver pass: {scope}."
    if batch_idx is not None and batch_total is not None and batch_total > 1:
        header += f" Batch {batch_idx} of {batch_total}."
    lines = [
        header,
        (
            f"Candidate pairs to judge ({len(candidates)}). Return EXACTLY one "
            f"verdict per pair — do not omit, merge, or reorder pairs."
        ),
    ]
    for a, b in candidates:
        ea = snapshot[a]
        eb = snapshot[b]
        shared = sorted(set(ea.source_doc_ids) & set(eb.source_doc_ids))[:5]
        lines.append(
            f"- name_a={a!r} (type={ea.entity_type}, mentions={ea.mentions}); "
            f"name_b={b!r} (type={eb.entity_type}, mentions={eb.mentions}); "
            f"shared_source_docs={shared}"
        )
    return "\n".join(lines)


def _sample_resolver(candidates: list[tuple[str, str]]) -> ResolverResult:
    """SAMPLE_MODE fallback: mark every candidate as ``distinct`` with low
    confidence. Conservative — never invents a merge without the model's
    judgment."""

    return ResolverResult(
        verdicts=[
            KgMergeVerdict(
                name_a=a, name_b=b,
                verdict="distinct",
                confidence=0.3,
                reason="SAMPLE_MODE conservative default",
            )
            for a, b in candidates
        ]
    )


__all__ = [
    "AGENT_NAME",
    "ACTION",
    "SWEEP_PERIOD",
    "resolve",
    "resolver_agent",
]
