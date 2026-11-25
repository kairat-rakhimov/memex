"""Per-adapter query planning for the build loop.

The RoundPlanner emits source-agnostic ranked topics; each adapter turns
them into source-tailored query strings via its own ``BaseQueryPlanner`` (a
chat-model call carrying source-specific phrasing guidance) and runs them.
The shared ``plan_and_search`` body wires planner → search and applies the
relevance posture: throttled sources act only when the round rates them
``"high"``; relaxed sources always act.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import AgentCallSinks, execute_agent_call
from memex.kg_intelligence.sources.common.base import SourceCandidate

_log = logging.getLogger(__name__)

MAX_PLANNED_QUERIES = 6


class PlannedQueries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(default_factory=list)


_SYSTEM_PROMPT = """\
You are the query planner for the **{source}** source in an iterative
knowledge-graph build. You receive the round's ranked topics (most important
first) and turn them into concrete, source-tailored search queries for {source}.

{guidance}

Rules:
- Emit {min_queries}-{max_queries} queries, ordered by the topic priority you
  were given (top topics first). When the cap is 1, emit a SINGLE query covering
  the highest-priority topic. Prefer one focused query per top topic over many
  broad ones.
- Each query must be runnable as-is against {source} — no commentary, no
  numbering, no surrounding quotes unless the source's syntax needs them.
- Stay on the given topics; do not invent unrelated subjects.
"""


class BaseQueryPlanner:
    """One per adapter. Wraps a chat-model Agent that rewrites the round's
    ranked topics into source-tailored query strings. ``guidance`` is the
    source-specific phrasing advice baked into the system prompt."""

    def __init__(
        self, *, source_name: str, guidance: str, max_queries: int = MAX_PLANNED_QUERIES
    ) -> None:
        self.source_name = source_name
        self.max_queries = max_queries
        self.system_prompt = _SYSTEM_PROMPT.format(
            source=source_name,
            guidance=guidance.strip(),
            min_queries=min(2, max_queries),
            max_queries=max_queries,
        )
        self._agent: Agent[None, PlannedQueries] = Agent(
            model=None,
            output_type=PlannedQueries,
            system_prompt=self.system_prompt,
            retries=1,
            defer_model_check=True,
        )

    async def plan_queries(
        self,
        ranked_topics: list[str],
        *,
        settings: Settings,
        sinks: AgentCallSinks,
        llm_model: Model | None = None,
    ) -> list[str]:
        if not ranked_topics:
            return []
        user_prompt = "Ranked topics (most important first):\n" + "\n".join(
            f"{i + 1}. {t}" for i, t in enumerate(ranked_topics)
        )
        result, _call, _run = await execute_agent_call(
            agent=self._agent,
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            sample_factory=lambda: PlannedQueries(
                queries=list(ranked_topics)[: self.max_queries]
            ),
            log_prefix=f"QueryPlanner[{self.source_name}]",
            settings=settings,
            llm_model=llm_model,
            sinks=sinks,
            agent_name=f"QueryPlanner[{self.source_name}]",
            action="plan_queries",
            triggered_by="RoundPlanner",
            target_agent=self.source_name,
            input_summary=f"{len(ranked_topics)} topics → {self.source_name}",
            output_summary_fn=lambda r: f"queries={r.queries}",
            confidence=0.7,
            model_preference="chat",
        )
        cleaned = [q.strip() for q in result.queries if q and q.strip()]
        return cleaned[: self.max_queries]


async def plan_and_search(
    *,
    planner: BaseQueryPlanner,
    search_fn: Callable[..., list[SourceCandidate]],
    posture: str,
    ranked_topics: list[str],
    relevance: Literal["low", "medium", "high"],
    limit: int,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
) -> list[SourceCandidate]:
    """Shared adapter body: posture-gate → plan queries → search → tag topics.

    Throttled sources skip the planner call AND the search unless the round
    rates them ``"high"``; relaxed sources ignore ``relevance``. The sync
    ``search_fn`` runs off-thread so the loop stays responsive."""

    if posture == "throttled" and relevance != "high":
        _log.info(
            "%s: throttled + relevance=%s — skipping acquisition this round",
            planner.source_name,
            relevance,
        )
        return []

    queries = await planner.plan_queries(
        ranked_topics,
        settings=settings,
        sinks=sinks,
        llm_model=llm_model,
    )
    if not queries:
        return []

    candidates = await asyncio.to_thread(search_fn, queries, limit=limit)
    for candidate in candidates:
        if not candidate.topics:
            candidate.topics = list(ranked_topics)
    return candidates


__all__ = [
    "MAX_PLANNED_QUERIES",
    "BaseQueryPlanner",
    "PlannedQueries",
    "plan_and_search",
]
