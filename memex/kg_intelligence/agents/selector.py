"""Selector: the global, cross-source acquisition arbiter.

After every adapter has planned + searched + ranked its own candidates, the
Selector sees the **flat cross-source pool** and picks ≤``budget`` documents
to fetch by **priority-weighted goal-fit** — the one place an arxiv paper is
compared to a gdelt article on a single axis. It replaces the old per-source
quota select: there is no per-source cap, only a global budget and the
round's ranked topics.
"""

from __future__ import annotations

import logging

from pydantic_ai import Agent
from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import (
    AgentCallSinks,
    ModelPreference,
    execute_agent_call,
)
from memex.kg_intelligence.agents.schemas import SelectorPick, SelectorResult
from memex.kg_intelligence.sources.common.base import AdapterDescription, SourceCandidate

_log = logging.getLogger(__name__)

AGENT_NAME = "Selector"
ACTION = "select_docs"
MODEL_PREFERENCE: ModelPreference = "chat"

_ABSTRACT_CAP = 240

_SYSTEM_PROMPT = """\
You are Selector, the global acquisition arbiter for an iterative
knowledge-graph build. Each round you receive ONE flat pool of candidate
documents drawn from every source (arxiv / wikipedia / gdelt / …), already
pre-ranked within each source, plus the round's RANKED topics (most important
first) and a short description of each source. You pick which documents are
worth the expensive downstream work (fetch → extract → map).

Pick by PRIORITY-WEIGHTED GOAL-FIT, not by source quota:
- Favor candidates that advance the TOP-ranked topics; lean effort toward the
  top when good candidates are plentiful, and spread down the ranking when the
  top is thin. Process the best survivors regardless of which source or topic
  they came from — 1 strong hit on topic 1 plus 5 on topic 2 is a fine round.
- Compare sources on a COMMON axis using their descriptions. Do NOT penalize a
  source for terse metadata (e.g. a news index with short snippets) — judge
  likely goal-fit, calibrated to what that source typically offers.
- Avoid near-duplicates: if two candidates are clearly the same document or
  near-identical in title/subject, keep only the best one.
- Corroboration across DIFFERENT sources is valuable — do not drop a candidate
  merely because another source covers the same topic from a different angle.

Output a list of picks, each referencing a candidate by its `candidate_index`,
with a `goal_relevance` in [0,1] (your priority-weighted goal-fit) and a short
`reason`. Pick AT MOST the stated budget; order by your preference (best first).
Pick fewer than the budget if the remaining candidates are weak — quantity is
not the goal.
"""

selector_agent: Agent[None, SelectorResult] = Agent(
    model=None,
    output_type=SelectorResult,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


async def select_documents(
    *,
    pool: list[SourceCandidate],
    ranked_topics: list[str],
    adapter_descriptions: list[AdapterDescription],
    budget: int,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    triggered_by: str = "search",
) -> list[SourceCandidate]:
    """Pick ≤``budget`` candidates from the flat pool by goal-fit, stamping the
    Selector's ``goal_relevance`` onto each chosen candidate. Returns the chosen
    candidates in the Selector's preference order; ``[]`` for an empty pool."""

    if not pool or budget <= 0:
        return []

    user_prompt = _build_user_prompt(pool, ranked_topics, adapter_descriptions, budget)
    input_summary = (
        f"pool={len(pool)}, topics={len(ranked_topics)}, budget={budget}"
    )

    def output_summary(result: SelectorResult) -> str:
        return f"picks={len(result.picks)}"

    result, _call, _run = await execute_agent_call(
        agent=selector_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=lambda: _sample_result(pool, budget),
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=ACTION,
        triggered_by=triggered_by,
        target_agent="fetch",
        input_summary=input_summary,
        output_summary_fn=output_summary,
        confidence=0.75,
        model_preference=MODEL_PREFERENCE,
    )
    return _apply_picks(result, pool, budget)


def _build_user_prompt(
    pool: list[SourceCandidate],
    ranked_topics: list[str],
    adapter_descriptions: list[AdapterDescription],
    budget: int,
) -> str:
    topics_section = (
        "\n".join(f"{i + 1}. {t}" for i, t in enumerate(ranked_topics)) or "(none)"
    )
    sources_section = (
        "\n".join(
            f"- {d.name}: {d.summary} (typically returns {'terse' if d.posture == 'throttled' else 'fuller'} metadata)"
            for d in adapter_descriptions
        )
        or "(none)"
    )
    pool_section = "\n".join(_render_candidate(i, c) for i, c in enumerate(pool))
    return (
        f"Budget: pick at most {budget} documents this round.\n\n"
        f"Ranked topics (most important first):\n{topics_section}\n\n"
        f"Sources:\n{sources_section}\n\n"
        f"Candidate pool ({len(pool)}):\n{pool_section}"
    )


def _render_candidate(index: int, candidate: SourceCandidate) -> str:
    abstract = (candidate.abstract or "").strip().replace("\n", " ")
    if len(abstract) > _ABSTRACT_CAP:
        abstract = abstract[:_ABSTRACT_CAP] + "…"
    topics = ", ".join(candidate.topics) if candidate.topics else "—"
    parts = [
        f"[{index}] source={candidate.source} | {candidate.title or '(untitled)'}",
        f"      topics={topics}",
    ]
    if abstract:
        parts.append(f"      snippet={abstract}")
    return "\n".join(parts)


def _apply_picks(
    result: SelectorResult, pool: list[SourceCandidate], budget: int
) -> list[SourceCandidate]:
    """Map the Selector's picks back to candidates: in preference order, skip
    out-of-range / duplicate indices, stamp ``goal_relevance``, cap at budget."""

    chosen: list[SourceCandidate] = []
    seen: set[int] = set()
    for pick in result.picks:
        if len(chosen) >= budget:
            break
        i = pick.candidate_index
        if i < 0 or i >= len(pool) or i in seen:
            continue
        seen.add(i)
        candidate = pool[i]
        candidate.goal_relevance = max(0.0, min(1.0, pick.goal_relevance))
        chosen.append(candidate)
    return chosen


def _sample_result(pool: list[SourceCandidate], budget: int) -> SelectorResult:
    """SAMPLE_MODE / degraded fallback: deterministic round-robin across sources
    in each source's rank order, up to the budget. Keeps source diversity (the
    LLM path does priority-weighted goal-fit instead). ``goal_relevance`` echoes
    the ranker score where present."""

    by_source: dict[str, list[int]] = {}
    for i, candidate in enumerate(pool):
        by_source.setdefault(candidate.source, []).append(i)

    queues = list(by_source.values())
    picks: list[SelectorPick] = []
    depth = 0
    while len(picks) < budget:
        progressed = False
        for queue in queues:
            if depth < len(queue):
                idx = queue[depth]
                score = pool[idx].raw_signals.get("rank_score")
                gr = float(score) if isinstance(score, (int, float)) else 0.5
                picks.append(
                    SelectorPick(
                        candidate_index=idx,
                        goal_relevance=max(0.0, min(1.0, gr)),
                        reason="sample_mode: round-robin by rank",
                    )
                )
                progressed = True
                if len(picks) >= budget:
                    break
        if not progressed:
            break
        depth += 1
    return SelectorResult(picks=picks)


__all__ = ["select_documents", "selector_agent"]
