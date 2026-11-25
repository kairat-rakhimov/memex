"""KGReviewer: identifies structural gaps in the growing KG.

Returns ranked ``KgGap`` items. The stop signal lives in the RoundReviewer —
KGReviewer is structural analysis only, and its gaps feed
the RoundReviewer (the strategist) at end of round.
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
from memex.kg_intelligence.agents.schemas import KgGap, KgReviewResult
from memex.kg_intelligence.state import CorpusState
from memex.kg_intelligence.views import KgView

AGENT_NAME = "KGReviewer"
ACTION = "review_kg"
MODEL_PREFERENCE: ModelPreference = "reasoning"
TOP_ENTITIES = 20

_SYSTEM_PROMPT = """\
You are KGReviewer, an agent that audits the growing knowledge graph after
each round and decides what's structurally missing.

Gap kinds:
- low_degree: entity mentioned in several docs but with very few outgoing
  relations — likely under-explored.
- disconnected: a cluster of entities that doesn't connect to the main graph.
- weak_relation: a relation type with very few instances relative to its
  importance for the base query.
- undefined_concept: a concept mentioned in entity names but not present
  in the schema's entity types.
- breadth: a domain category conspicuously absent from the KG given the
  base query. For a technology-focused topic, breadth gaps might be
  policy/regulatory context, economic/market dimensions, environmental
  impact, or deployment/infrastructure aspects. Also use for breadcrumb
  entities (listed in "Breadcrumb candidates" below if present): entities
  mentioned once or twice with very low degree that hint at an
  under-explored dimension worth pursuing.

Be specific and grounded: every gap MUST reference an entity, relation
type, concept, or domain category actually relevant to the base query.
Breadth gaps may reference absent categories rather than existing entities.

**Rank gaps by importance.** `rank` is 0-indexed; 0 = most important.
The next round's RoundPlanner allocates more queries to top-ranked gaps,
so ordering matters. Rank by:
- Impact on the base query (gaps that block central concepts > peripheral).
- Asymmetry: an entity that's a hub in the existing KG but under-explored
  outranks an isolated low-degree leaf.
- Subtopic coverage: gaps that re-enter the corpus where it's already
  rich are lower priority than gaps in genuinely under-covered areas.
- Breadth: genuinely absent domain dimensions outrank refinements within
  already-covered areas.
"""

reviewer_agent: Agent[None, KgReviewResult] = Agent(
    model=None,
    output_type=KgReviewResult,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


async def review(
    *,
    base_query: str,
    kg: KgView,
    corpus_state: CorpusState,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    triggered_by: str = "round_end",
    fail_hard: bool = True,
) -> tuple[KgReviewResult, int | None]:
    """Run one KGReviewer turn over the current KG. Emit one trace line.

    Returns ``(result, agent_call_id)`` — the call id lets the orchestrator
    stamp ``round_reviews.kg_review_call_id``. Critical-path (``fail_hard``):
    a real LLM failure raises ``AgentHardFailure`` rather than degrading.
    """

    user_prompt = _build_user_prompt(base_query, kg, corpus_state)
    input_summary = (
        f"round={corpus_state.round_number}, "
        f"entities={kg.entity_count}, relations={kg.relation_count}, "
        f"schema={len(kg.schema.entity_types)}/{len(kg.schema.relation_types)}"
    )

    def output_summary(result: KgReviewResult) -> str:
        return f"gaps={len(result.gaps)}"

    result, call, _run = await execute_agent_call(
        agent=reviewer_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=lambda: _sample_review(kg),
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=ACTION,
        triggered_by=triggered_by,
        target_agent="Strategist",
        input_summary=input_summary,
        output_summary_fn=output_summary,
        confidence=0.7,
        model_preference=MODEL_PREFERENCE,
        fail_hard=fail_hard,
    )
    return result, call.agent_call_id if call else None


def _build_user_prompt(
    base_query: str,
    kg: KgView,
    corpus_state: CorpusState,
) -> str:
    top_entities = sorted(
        kg.entities.values(),
        key=lambda e: e.mentions,
        reverse=True,
    )[:TOP_ENTITIES]
    entities_section = (
        "\n".join(
            f"- {e.canonical_name} (type={e.entity_type}, mentions={e.mentions}, degree={e.degree})"
            for e in top_entities
        )
        or "(none)"
    )

    relations_section = (
        "\n".join(f"- {pred}: {count}" for pred, count in kg.predicate_counts[:20])
        or "(none)"
    )

    schema_section = (
        f"entity types: {', '.join(kg.schema.entity_types) or '(empty)'}\n"
        f"relation types: {', '.join(kg.schema.relation_types) or '(empty)'}"
    )

    breadcrumbs = sorted(
        (e for e in kg.entities.values() if e.mentions <= 2 and e.degree <= 2),
        key=lambda e: -e.mentions,
    )[:20]
    breadcrumbs_section = (
        "\n".join(
            f"- {e.canonical_name} (type={e.entity_type}, mentions={e.mentions}, degree={e.degree})"
            for e in breadcrumbs
        )
        or "(none)"
    )

    parts = [
        f"Base query: {base_query}",
        f"Round number: {corpus_state.round_number}",
        f"Top entities (mentions × degree):\n{entities_section}",
        f"Relation type counts:\n{relations_section}",
        f"Current schema:\n{schema_section}",
        f"Breadcrumb candidates (mentions ≤ 2, degree ≤ 2 — potentially under-explored):\n{breadcrumbs_section}",
    ]
    if corpus_state.subtopics_missing:
        parts.append(
            "Subtopics flagged missing by the UsefulnessGate:\n- "
            + "\n- ".join(corpus_state.subtopics_missing[:10])
        )
    return "\n\n".join(parts)


def _sample_review(kg: KgView) -> KgReviewResult:
    """SAMPLE_MODE fallback: flag low-degree entities mechanically.
    Ranks by mention count (more-mentioned first = more central)."""

    low_degree = sorted(
        (e for e in kg.entities.values() if e.degree <= 1),
        key=lambda e: -e.mentions,
    )[:3]
    gaps = [
        KgGap(
            description=(
                f"{e.canonical_name} mentioned {e.mentions} times but has degree "
                f"{e.degree} — explore its relationships"
            ),
            kind="low_degree",
            rank=i,
        )
        for i, e in enumerate(low_degree)
    ]
    return KgReviewResult(gaps=gaps)
