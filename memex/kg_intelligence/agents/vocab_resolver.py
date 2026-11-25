"""VocabResolver: the Resolver pattern generalized to the VOCABULARY lanes.

The entity Resolver clusters surface entity names into canonical + aliases.
VocabResolver does the analogous job for the *vocabulary* — predicates and
properties — which currently have no cluster structure (flat
``kg_schema``, the predicate/property sprawl). It is a **global, post-hoc** pass:
it reads the whole accumulated canonical vocabulary, clusters near-neighbours by
embedding similarity, and asks the LLM to split each cluster's fuzzy proximity
into the three real relationships (the per-call Mapper ``fallback`` smears them):

- **alias** (synonym, same concept) → collapse to the canonical (vocab shrinks).
- **child_of** (specialization, narrower) → keep the specific canonical, record
  the broader one as its parent (rollup; the query-time-expansion substrate).
- **unrelated** (merely nearby) → leave standalone.

Non-destructive by design (a destructive merge was dropped; expansion lives at
the query layer): canonicals stay on their edges; this
pass records cluster membership (aliases + parents) on the vocabulary itself.

This module is store-agnostic for candidate generation (operates on name +
matrix + frequency arrays) so it is cheap to test per-agent against an existing
``kg.db`` without a rebuild; ``cluster_and_commit`` is the traced, committing path.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
from pydantic_ai import Agent
from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import (
    AgentCallSinks,
    ModelPreference,
    execute_agent_call,
    run_agent_or_fallback,
)
from memex.kg_intelligence.agents.schemas import VocabResolverResult, VocabVerdict
from memex.kg_intelligence.embeddings import load_matrix, top_k_similar

if TYPE_CHECKING:
    from memex.observability import TraceStore

_log = logging.getLogger(__name__)

# VocabResolver vocab kind → the kg_schema `kind` its parents are recorded under.
_SCHEMA_KIND = {"predicate": "relation_type", "property": "property_type"}

AGENT_NAME = "VocabResolver"
ACTION = "cluster_vocab"
MODEL_PREFERENCE: ModelPreference = "reasoning"

COSINE_THRESHOLD = 0.60
"""Floor for an embedding near-neighbour to enter a candidate cluster. The LLM
does the precision judging; this only bounds recall of candidates."""
TOP_K = 8
MAX_GROUP = 8
CLUSTER_CONCURRENCY = 8
"""Max in-flight cluster judgments. Clusters are independent prompts (no shared
cache prefix beyond the constant system prompt), so bounded fan-out is pure win;
the cap keeps LLM concurrency sane. Store writes stay serialized by the
event loop + the store lock."""

_SYSTEM_PROMPT = """\
You are VocabResolver. You are given one cluster of CANONICAL knowledge-graph
predicates (short verb phrases) that are near-neighbours in embedding space, with
a "head" predicate (the most frequently used, usually the most generic). For EACH
other member, decide its relationship to a canonical you name (default: the head):

- alias: the member means the SAME relationship, just worded differently (a
  synonym). Examples: `utilizes` ≈ `uses`, `employs` ≈ `uses`, `is_made_of` ≈
  `composed_of`. The member should COLLAPSE into the canonical.
- child_of: the member is a NARROWER, more specific case of the canonical.
  Examples: `champions_as_peace_train` is a child_of `advocates_for`;
  `imposes_tariffs_on` is a child_of `impacts`. Keep the member as its own
  predicate; record the canonical as its PARENT (for rollup).
- unrelated: the member is merely near in wording but is a genuinely DIFFERENT
  relationship. Examples: `produces` vs `consumes`, `supports` vs `opposes`,
  `precedes` vs `follows`. Leave it standalone.

Pick the BEST canonical for each member — usually the head, but you may name a
different cluster member if it is a closer synonym/parent.

Be conservative with `alias`: a wrong alias erases a real distinction in the
graph. When torn between `alias` and `child_of`, choose `child_of` (it is
non-destructive — it keeps the specific predicate). When unsure the member is
related at all, choose `unrelated`.

Return one verdict per member: name, canonical, relation, confidence ∈ [0, 1],
and a one-clause reason naming the evidence.
"""

vocab_resolver_agent: Agent[None, VocabResolverResult] = Agent(
    model=None,
    output_type=VocabResolverResult,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


# ── candidate generation (no LLM, store-agnostic) ─────────────────────────


def candidate_groups(
    names: list[str],
    matrix: np.ndarray,
    freqs: dict[str, int],
    *,
    cosine_threshold: float = COSINE_THRESHOLD,
    top_k: int = TOP_K,
    max_group: int = MAX_GROUP,
) -> list[tuple[str, list[tuple[str, float]]]]:
    """Star-cluster the vocabulary into ``(head, [(member, similarity), …])``.

    Heads are taken in descending frequency (the generic workhorses lead), each
    claiming its nearest unassigned neighbours above ``cosine_threshold``; once a
    name is claimed it cannot head or join another cluster. This yields disjoint
    star clusters headed by frequent/generic terms — exactly the shape the LLM
    judges (member-vs-head). ``matrix`` rows align with ``names`` (as returned by
    ``load_matrix``; vectors are assumed L2-normalised, so dot product = cosine).
    """

    if matrix.size == 0 or not names:
        return []

    order = sorted(range(len(names)), key=lambda i: -freqs.get(names[i], 0))
    assigned: set[str] = set()
    groups: list[tuple[str, list[tuple[str, float]]]] = []

    for hi in order:
        head = names[hi]
        if head in assigned:
            continue
        members: list[tuple[str, float]] = []
        for j, score in top_k_similar(matrix[hi], matrix, k=top_k + 1):
            member = names[j]
            if member == head or member in assigned:
                continue
            if score < cosine_threshold:
                continue
            members.append((member, score))
            if len(members) >= max_group - 1:
                break
        if not members:
            continue
        assigned.add(head)
        for member, _ in members:
            assigned.add(member)
        groups.append((head, members))

    return groups


def _build_group_prompt(
    head: str, members: list[tuple[str, float]], freqs: dict[str, int]
) -> str:
    lines = [
        f"Head (canonical, most frequent): {head}  (used {freqs.get(head, 0)}×)",
        "",
        "Members to classify (near-neighbours of the head):",
    ]
    for member, score in members:
        lines.append(f"- {member}  (used {freqs.get(member, 0)}×, similarity {score:.2f})")
    lines.append("")
    lines.append("Return a verdict for EACH member.")
    return "\n".join(lines)


def _sample_group(
    head: str, members: list[tuple[str, float]]
) -> VocabResolverResult:
    """Deterministic SAMPLE_MODE stub: mark every member an alias of the head."""

    return VocabResolverResult(
        verdicts=[
            VocabVerdict(
                name=member, canonical=head, relation="alias",
                confidence=0.5, reason="sample stub",
            )
            for member, _score in members
        ]
    )


async def judge_group(
    head: str,
    members: list[tuple[str, float]],
    freqs: dict[str, int],
    *,
    settings: Settings,
    llm_model: Model | None = None,
) -> VocabResolverResult:
    """Run the LLM judge on one candidate cluster. Direct agent run (no store /
    sinks) — the per-agent harness path; ``cluster_and_commit`` is the traced
    path."""

    run_result = await run_agent_or_fallback(
        vocab_resolver_agent,
        _build_group_prompt(head, members, freqs),
        settings=settings,
        llm_model=llm_model,
        sample_factory=lambda: _sample_group(head, members),
        log_prefix=f"{AGENT_NAME}[{head}]",
        model_preference=MODEL_PREFERENCE,
    )
    return run_result.output


# ── commit (non-destructive) ──────────────────────────────────────────────


def commit_clusters(
    store: TraceStore,
    verdicts: list[VocabVerdict],
    *,
    kind: str,
    agent_call_id: int,
    round_number: int,
) -> dict[str, int]:
    """Persist VocabResolver verdicts NON-DESTRUCTIVELY:

    - ``alias`` → ``{predicate,property}_aliases`` (synonym collapse — the form
      stops being offered as a canonical, future raws snap to the canonical, and
      query-time expansion treats them as one).
    - ``child_of`` → ``schema_parents`` (a child→parent rollup edge).
    - ``unrelated`` → nothing.

    Edges in ``kg_relations`` / ``kg_entity_properties`` are NOT rewritten
    (sprawl is collapsed at the query layer, not by a destructive merge). Returns counts per relation.
    """

    schema_kind = _SCHEMA_KIND[kind]
    counts = {"alias": 0, "child_of": 0, "unrelated": 0}
    for v in verdicts:
        if v.relation == "alias":
            if kind == "predicate":
                store.record_predicate_alias(
                    alias=v.name, canonical=v.canonical,
                    agent_call_id=agent_call_id, round_number=round_number,
                )
            else:
                store.record_property_alias(
                    alias=v.name, canonical=v.canonical,
                    agent_call_id=agent_call_id, round_number=round_number,
                )
            counts["alias"] += 1
        elif v.relation == "child_of":
            store.record_schema_parent(
                child=v.name, parent=v.canonical, kind=schema_kind,
                agent_call_id=agent_call_id, round_number=round_number,
            )
            counts["child_of"] += 1
        else:
            counts["unrelated"] += 1
    return counts


# ── pipeline-integrated pass (load → cluster → judge → commit) ────────────


def _load_normalized_matrix(store: TraceStore, kind: str) -> tuple[list[str], np.ndarray]:
    names, matrix = load_matrix(store, kind)
    if matrix.size == 0:
        return names, matrix
    return names, matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


async def cluster_and_commit(
    *,
    store: TraceStore,
    kind: str,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    round_number: int,
    cosine_threshold: float = COSINE_THRESHOLD,
    top_k: int = TOP_K,
    max_group: int = MAX_GROUP,
    group_limit: int | None = None,
    concurrency: int = CLUSTER_CONCURRENCY,
) -> dict[str, int]:
    """The global post-hoc vocabulary pass: load the canonical ``kind`` vocab +
    embeddings from ``store``, cluster near-neighbours, LLM-judge each cluster
    (one traced ``execute_agent_call`` per cluster, so commits get a real
    call_id), and commit the verdicts non-destructively. Clusters are judged with
    bounded concurrency. Returns totals (clusters + alias/child_of/unrelated edge
    counts)."""

    names, matrix = _load_normalized_matrix(store, kind)
    totals = {"clusters": 0, "alias": 0, "child_of": 0, "unrelated": 0}
    if matrix.size == 0:
        _log.info("VocabResolver: no '%s' embeddings — nothing to cluster", kind)
        return totals

    counts_src = store.predicate_counts() if kind == "predicate" else store.property_counts()
    freqs = dict(counts_src)
    groups = candidate_groups(
        names, matrix, freqs,
        cosine_threshold=cosine_threshold, top_k=top_k, max_group=max_group,
    )
    if group_limit is not None:
        groups = groups[:group_limit]
    _log.info(
        "VocabResolver: %d %s candidate clusters to judge (vocab=%d)",
        len(groups), kind, len(names),
    )

    sem = asyncio.Semaphore(concurrency)

    async def _judge_and_commit(head: str, members: list[tuple[str, float]]) -> None:
        async with sem:
            result, call, _run = await execute_agent_call(
                agent=vocab_resolver_agent,
                user_prompt=_build_group_prompt(head, members, freqs),
                system_prompt=_SYSTEM_PROMPT,
                sample_factory=lambda h=head, m=members: _sample_group(h, m),
                log_prefix=f"{AGENT_NAME}[{head}]",
                settings=settings,
                llm_model=llm_model,
                sinks=sinks,
                agent_name=AGENT_NAME,
                action=ACTION,
                triggered_by="vocab_finalize",
                target_agent="kg_schema",
                input_summary=f"kind={kind}, head={head}, members={len(members)}",
                output_summary_fn=lambda r: (
                    f"alias={sum(1 for v in r.verdicts if v.relation == 'alias')}, "
                    f"child_of={sum(1 for v in r.verdicts if v.relation == 'child_of')}, "
                    f"unrelated={sum(1 for v in r.verdicts if v.relation == 'unrelated')}"
                ),
                confidence=0.7,
                model_preference=MODEL_PREFERENCE,
            )
        # Commit after releasing the semaphore — the store serializes writes, so
        # the only contended resource is the LLM call the semaphore bounds.
        totals["clusters"] += 1
        if call is not None:
            counts = commit_clusters(
                store, result.verdicts, kind=kind,
                agent_call_id=call.agent_call_id, round_number=round_number,
            )
            for key in ("alias", "child_of", "unrelated"):
                totals[key] += counts[key]

    await asyncio.gather(*(_judge_and_commit(h, m) for h, m in groups))

    _log.info(
        "VocabResolver %s: %d clusters → %d aliases, %d parents, %d unrelated",
        kind, totals["clusters"], totals["alias"], totals["child_of"],
        totals["unrelated"],
    )
    return totals
