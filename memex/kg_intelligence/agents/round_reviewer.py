"""Strategist (module/symbol still ``RoundReviewer``): the build's strategist.

Its user-visible trace label is ``Strategist`` (``AGENT_NAME``); the code symbol
+ file stay ``round_reviewer`` to avoid a broad mechanical rename.

Runs at end of round (Phase 2), after the KG is committed and the deterministic
metrics row is written. Reads the recent-round metric trend + the KGReviewer's
gaps + its own cross-round goal memory, then (a) classifies the build's
trajectory — owning the stop decision — and (b) seeds the next RoundPlanner
with per-theme directives, including marking goals unanswerable.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import (
    AgentCallSinks,
    ModelPreference,
    execute_agent_call,
)
from memex.kg_intelligence.agents.schemas import (
    KgReviewResult,
    RoundReviewResult,
    RoundStatus,
)
from memex.kg_intelligence.state import CorpusState

AGENT_NAME = "Strategist"
ACTION = "review_round"
MODEL_PREFERENCE: ModelPreference = "reasoning"

_SYSTEM_PROMPT = """\
You are the Strategist for an iterative knowledge-graph build.
After each round you see the build's recent trajectory and decide two things.

1. TRAJECTORY — classify the build's current status:
   - continuing: healthy progress; keep going as planned.
   - slowing: returns are dropping but a change of focus is worth trying;
     the build continues with a pivot.
   - stalled: progress has effectively stopped and recovery would need human
     intervention (e.g. the corpus is exhausted for this topic). Halts.
   - wasted: further effort cannot recover anything useful. Halts.
   Judge from the TREND, not one round. A single quiet round is not a stall.
   "Progress" = new entities AND new relations landing; a round that adds no
   new entities but densifies existing ones with new relations is still
   progress. Watch tokens-per-new-entity rising as a diminishing-returns
   signal. **`already_seen` (the % of searched candidates already in the KG) is
   the most direct saturation signal — when it is high and rising across the
   trail, the reachable corpus is being re-surfaced rather than extended.** A
   single dry round (high already_seen, 0 fetched) is a normal pivot, NOT a
   stall; persistently high already_seen across the window with no new fetches
   is `stalled`. The build is hypothesis generation, not completion — never stop
   because the KG looks "done"; stop only when further rounds would waste
   tokens.

2. GOAL DIRECTIVES — steer the next RoundPlanner per theme:
   - pursue: this theme is paying off; keep it.
   - deprioritize: weak, drop down the priority order but not dead.
   - abandon: unanswerable — this theme has recurred across rounds without
     landing new knowledge; never retry it. Abandoned themes are excluded
     from all future rounds, so only abandon when the recent-rounds trail
     shows the theme tried and flat. A theme whose rounds show high
     `already_seen` is **mined out** (its docs are all already ingested) — a
     prime candidate to deprioritize or abandon so the next round explores
     genuinely new territory instead of re-searching it. Do NOT re-list themes
     already in the "Already abandoned" set.
   Only emit directives for themes that actually appear in the recent goals
   trail. If every viable theme is exhausted and no new gaps remain, that is
   a `stalled` build — say so in `reason`.

Keep `reason` short and concrete (one or two sentences); it is surfaced to
the user when the build halts.
"""

reviewer_agent: Agent[None, RoundReviewResult] = Agent(
    model=None,
    output_type=RoundReviewResult,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


async def review(
    *,
    base_query: str,
    corpus_state: CorpusState,
    kg_review: KgReviewResult,
    recent_metrics: list[dict[str, Any]],
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    triggered_by: str = "round_end",
    fail_hard: bool = True,
) -> tuple[RoundReviewResult, int | None]:
    """Run one RoundReviewer turn. Returns ``(result, agent_call_id)`` — the
    call id lets the orchestrator stamp ``round_reviews.round_review_call_id``.
    Critical-path (``fail_hard``): a real LLM failure raises
    ``AgentHardFailure`` rather than degrading to a sample."""

    user_prompt = _build_user_prompt(base_query, corpus_state, kg_review, recent_metrics)
    latest = recent_metrics[-1] if recent_metrics else {}
    input_summary = (
        f"round={corpus_state.round_number}, "
        f"window={len(recent_metrics)}, gaps={len(kg_review.gaps)}, "
        f"new_e={latest.get('new_entities', 0)}, new_r={latest.get('new_relations', 0)}"
    )

    def output_summary(result: RoundReviewResult) -> str:
        abandons = sum(1 for d in result.goal_directives if d.verdict.value == "abandon")
        return (
            f"status={result.status.value}, "
            f"directives={len(result.goal_directives)} (abandon={abandons})"
        )

    result, call, _run = await execute_agent_call(
        agent=reviewer_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=lambda: _sample_review(recent_metrics),
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=ACTION,
        triggered_by=triggered_by,
        target_agent="RoundPlanner",
        input_summary=input_summary,
        output_summary_fn=output_summary,
        confidence=0.7,
        model_preference=MODEL_PREFERENCE,
        fail_hard=fail_hard,
    )
    return result, call.agent_call_id if call else None


def _build_user_prompt(
    base_query: str,
    corpus_state: CorpusState,
    kg_review: KgReviewResult,
    recent_metrics: list[dict[str, Any]],
) -> str:
    metrics_lines = []
    for m in recent_metrics:
        tpe = m.get("tokens_per_new_entity")
        tpe_str = f"{tpe:.0f}" if isinstance(tpe, int | float) else "—"
        air = m.get("already_ingested_rate")
        air_str = f"{air*100:.0f}%" if isinstance(air, int | float) else "—"
        metrics_lines.append(
            f"- round {m.get('round_number')}: "
            f"new_entities={m.get('new_entities')}, "
            f"new_relations={m.get('new_relations')}, "
            f"docs_fetched={m.get('docs_fetched_total')}, "
            f"already_seen={air_str} of searched, "
            f"tokens={m.get('tokens_spent_total')}, "
            f"tokens/new_entity={tpe_str}, "
            f"repairs={m.get('consolidation_repairs')}"
        )
    metrics_section = "\n".join(metrics_lines) or "(no prior rounds)"

    goals_lines = [
        f"- round {g.round}: {g.focused_goal or '(no focus)'} "
        f"[themes: {', '.join(g.priority_themes) or 'none'}]"
        for g in corpus_state.goal_history[-len(recent_metrics) - 1 :]
    ]
    goals_section = "\n".join(goals_lines) or "(no goals yet)"

    gaps_section = (
        "\n".join(
            f"- rank={g.rank} ({g.kind}): {g.description}"
            for g in sorted(kg_review.gaps, key=lambda x: x.rank)[:10]
        )
        or "(none)"
    )

    abandoned_section = (
        "\n- ".join(corpus_state.abandoned_goals) if corpus_state.abandoned_goals else "(none)"
    )

    # This round's per-topic saturation — which themes the searches keep
    # re-surfacing (already in the KG). The actionable signal for abandoning the
    # exact mined-out theme. Sorted most-saturated first; topics never searched
    # this round are omitted.
    latest = recent_metrics[-1] if recent_metrics else {}
    by_topic = latest.get("already_seen_by_topic") or {}
    topic_lines = [
        f"- {topic}: {seen}/{surf} already-seen ({seen / surf * 100:.0f}%)"
        for topic, (surf, seen) in sorted(
            by_topic.items(),
            key=lambda kv: -(kv[1][1] / kv[1][0]) if kv[1][0] else 0.0,
        )
        if surf
    ]
    topic_section = "\n".join(topic_lines) or "(no searches this round)"

    return "\n\n".join(
        [
            f"Base query: {base_query}",
            f"Round just completed: {corpus_state.round_number}",
            f"Recent-round metrics (oldest → newest):\n{metrics_section}",
            f"This round's per-topic saturation (already-seen of searched):\n{topic_section}",
            f"Recent goals trail (what each round aimed at):\n{goals_section}",
            f"This round's KG gaps (what's still structurally missing):\n{gaps_section}",
            f"Already abandoned (do NOT re-list, do NOT resurrect):\n- {abandoned_section}",
        ]
    )


def _sample_review(recent_metrics: list[dict[str, Any]]) -> RoundReviewResult:
    """SAMPLE_MODE fallback: always ``continuing``. The deterministic
    ``zero_progress_floor`` + ``max_rounds`` own SAMPLE-mode stops — a sample
    that halted on a single quiet round would mask those floors."""

    return RoundReviewResult(
        status=RoundStatus.continuing,
        reason="SAMPLE_MODE deterministic trajectory.",
        goal_directives=[],
    )


__all__ = ["AGENT_NAME", "review"]
