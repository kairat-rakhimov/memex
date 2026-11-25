"""RoundPlanner: the per-round tactician.

Turns the strategist's guidance (the previous RoundReviewer's directives +
the KGReviewer's gaps) into one round's plan — a ranked list of topics to
pursue (priority = list order) plus a per-active-adapter corpus-fit signal.
Source-agnostic: it names *what* the graph should acquire next; each adapter
decides *how* to search for it (``plan_and_search``). Hard-excludes goals the
strategist has abandoned.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import (
    AgentCallSinks,
    ModelPreference,
    execute_agent_call,
)
from memex.kg_intelligence.agents.schemas import (
    GoalDirective,
    KgGap,
    RankedTopic,
    RoundPlan,
)
from memex.kg_intelligence.sources.common.base import AdapterDescription
from memex.kg_intelligence.state import CorpusState

AGENT_NAME = "RoundPlanner"
ACTION = "plan_round"
MODEL_PREFERENCE: ModelPreference = "chat"

_SYSTEM_PROMPT = """\
You are RoundPlanner, the tactician for an iterative knowledge-graph build.
Each round you turn the strategist's guidance plus the current structural gaps
into one round's plan: a RANKED LIST OF TOPICS to pursue, most important first.
You are source-agnostic — you name *what* the graph should acquire next; each
source decides *how* to search for it.

Compose the round:
- `topics`: a ranked list (3-6) of specific topics, most important first. Each is
  a short noun phrase naming a concept, mechanism, material, entity, or
  relationship the KG should acquire next, with a one-line `rationale`. Priority
  is the list position: the build leans effort toward the top topics and fills
  down when the top is thin. Name topics consistently round-to-round — they are
  tracked across rounds.
- `description`: 2-3 sentences explaining the round's plan and how it follows
  from the gaps and directives.
- `adapter_relevance`: for EACH active source listed in the prompt, judge how
  well that source's corpus fits this round's topics — `high`, `medium`, or
  `low`. Judge ONLY corpus fit (what the source covers), never cost or speed.
  Emit exactly one entry per active source, keyed by its name.

Hard rules:
- Stay anchored to the base query. Do not drift to unrelated topics.
- NEVER pursue an abandoned goal (listed below if any): those are unanswerable.
  Generate no topic for them.
- Honor the strategist's goal directives: lead with `pursue` themes, push
  `deprioritize` themes down the ranking.
- Gap-ranked allocation: the top-ranked KG gaps (rank 0 = most important) map to
  your top topics. Use ranking as a soft signal — follow a better lead if a
  low-rank gap yields a clearly stronger topic.
- First round (no gaps, no directives): run in exploratory mode — propose a
  ranked spread of topics that decompose the base query into its main facets.
"""

planner_agent: Agent[None, RoundPlan] = Agent(
    model=None,
    output_type=RoundPlan,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


async def plan(
    *,
    base_query: str,
    corpus_state: CorpusState,
    kg_gaps: list[KgGap],
    directives: list[GoalDirective],
    adapter_descriptions: list[AdapterDescription],
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    triggered_by: str = "round_start",
) -> RoundPlan:
    user_prompt = _build_user_prompt(
        base_query, corpus_state, kg_gaps, directives, adapter_descriptions
    )
    adapter_names = [d.name for d in adapter_descriptions]
    input_summary = (
        f"round={corpus_state.round_number}, base={base_query!r}, "
        f"gaps={len(kg_gaps)}, directives={len(directives)}, "
        f"abandoned={len(corpus_state.abandoned_goals)}, "
        f"sources={adapter_names}"
    )

    def output_summary(result: RoundPlan) -> str:
        rel = ", ".join(f"{k}={v}" for k, v in result.adapter_relevance.items())
        return f"topics={[t.topic for t in result.topics]}, relevance=({rel})"

    result, _call, _run = await execute_agent_call(
        agent=planner_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=lambda: _sample_plan(base_query, corpus_state, adapter_names),
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=ACTION,
        triggered_by=triggered_by,
        target_agent="adapters",
        input_summary=input_summary,
        output_summary_fn=output_summary,
        confidence=0.8,
        model_preference=MODEL_PREFERENCE,
    )
    return result


def _build_user_prompt(
    base_query: str,
    corpus_state: CorpusState,
    kg_gaps: list[KgGap],
    directives: list[GoalDirective],
    adapter_descriptions: list[AdapterDescription],
) -> str:
    parts = [
        f"Base query: {base_query}",
        f"Round number: {corpus_state.round_number}",
    ]
    if adapter_descriptions:
        parts.append(
            "Active sources (emit one adapter_relevance entry for EACH, keyed by "
            "name):\n"
            + "\n".join(f"- {d.name}: {d.summary}" for d in adapter_descriptions)
        )
    if corpus_state.abandoned_goals:
        parts.append(
            "Abandoned goals (unanswerable — DO NOT pursue, no topics for "
            "these):\n- " + "\n- ".join(corpus_state.abandoned_goals[:20])
        )
    if directives:
        parts.append(
            "Strategist directives from last round:\n"
            + "\n".join(
                f"- {d.verdict.value} «{d.theme}»: {d.rationale}".rstrip(": ")
                for d in directives
            )
        )
    if corpus_state.subtopics_covered:
        parts.append(
            "Subtopics already covered:\n- "
            + "\n- ".join(corpus_state.subtopics_covered[:10])
        )
    if corpus_state.subtopics_missing:
        parts.append(
            "Subtopics missing (prefer topics that cover these):\n- "
            + "\n- ".join(corpus_state.subtopics_missing[:10])
        )
    if kg_gaps:
        ordered = sorted(kg_gaps, key=lambda g: g.rank)[:10]
        parts.append(
            "KG structural gaps from last round (ranked; 0 = most important — "
            "map your top topics to these):\n"
            + "\n".join(f"- rank={g.rank}: {g.description}" for g in ordered)
        )
    if corpus_state.consecutive_empty_rounds > 0:
        parts.append(
            f"Your previous {corpus_state.consecutive_empty_rounds} round(s) "
            "added 0 new documents — your search strategy isn't finding new "
            "material. Reconsider: try broader topics or angles the KG hasn't "
            "covered."
        )
    return "\n\n".join(parts)


def _sample_plan(
    base_query: str, corpus_state: CorpusState, adapter_names: list[str]
) -> RoundPlan:
    """SAMPLE_MODE fallback: a deterministic ranked decomposition of the base
    query, with every active source marked relevant."""

    head = (base_query or "topic").strip()
    return RoundPlan(
        description=(
            f"SAMPLE_MODE: ranked decomposition of {head!r} at round "
            f"{corpus_state.round_number}."
        ),
        topics=[
            RankedTopic(topic=head, rationale="core concept"),
            RankedTopic(topic=f"{head} mechanisms", rationale="how it works"),
            RankedTopic(topic=f"{head} applications", rationale="where it is used"),
        ],
        adapter_relevance=dict.fromkeys(adapter_names, "high"),
    )
