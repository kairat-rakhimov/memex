"""The agentic Retriever — the query layer's only tool-using agent.

An ``Agent[QueryDeps, RetrieverResult]`` that adaptively composes the three
retrieval modalities (semantic / symbolic / topological) per question via
``@agent.tool`` wrappers over the pure toolbox, then returns the fact refs it
judged relevant. Bounded by ``run_tool_agent``'s ``usage_limits``. The tool
wrappers are the only place that calls the LLM API (embedding the query for
semantic search); everything they call is pure.
"""

from __future__ import annotations

from pydantic_ai import Agent, RunContext

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import AgentCallSinks, ModelPreference
from memex.kg_intelligence.embeddings import embed_texts
from memex.kg_query import expansion, retrieval
from memex.kg_query.agents.runtime import run_tool_agent
from memex.kg_query.agents.schemas import RetrieverResult
from memex.kg_query.deps import QueryDeps
from memex.kg_query.grounding import source_view
from memex.kg_query.retrieval import FactHit

AGENT_NAME = "Retriever"
ACTION = "retrieve_evidence"
MODEL_PREFERENCE: ModelPreference = "reasoning"

_SYSTEM_PROMPT = """\
You are the Retriever in a grounded knowledge-graph Q&A system. Your job is to
gather the EVIDENCE needed to answer the user's question — not to answer it.

You have tools over a knowledge graph of entities, typed relations (entity →
entity), and attributes (entity → measured/qualitative value with units +
conditions), plus per-fact semantic search.

Strategy:
- Start with `search_facts` (semantic) to find facts by meaning, and
  `find_entity` to anchor named things. Then drill with `entity_facts`,
  `facts_by_predicate`, `facts_by_property`, or `connect_entities` for multi-hop
  links.
- Prefer well-corroborated facts (higher `corroboration` = more source docs).
- Numeric questions ("how much", "cost", "efficiency") → use `facts_by_property`
  and attribute facts, which carry values + conditions.
- Use `read_source` to read the verbatim sentence + surrounding context behind any
  fact when its meaning, conditions, or exact wording matter.
- Stop once you have enough to ground a complete answer; do not over-fetch.

Every fact a tool returns has a `ref` like "relation:1037" or "attribute:42".
Return ONLY refs you actually saw in tool results — never invent a ref. Put the
most relevant facts in `selected`, explain your retrieval strategy in
`reasoning`, and note anything the corpus seems to be MISSING in `notes`.
"""

retriever_agent: Agent[QueryDeps, RetrieverResult] = Agent(
    model=None,
    deps_type=QueryDeps,
    output_type=RetrieverResult,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


def _fact_row(h: FactHit) -> dict:
    return {
        "ref": f"{h.kind}:{h.fact_id}",
        "fact": h.text(),
        "corroboration": h.corroboration,
        "confidence": round(h.confidence, 2) if h.confidence is not None else None,
    }


@retriever_agent.tool
async def search_facts(ctx: RunContext[QueryDeps], query: str) -> list[dict]:
    """Semantic search over per-fact embeddings — find facts by MEANING
    (immune to predicate/property wording). Best first move for any question."""

    vecs = await embed_texts([query], ctx.deps.settings)
    if not vecs:
        return []
    hits = retrieval.semantic_facts(ctx.deps, vecs[0], k=10)
    return [_fact_row(h) for h in hits]


@retriever_agent.tool
def find_entity(ctx: RunContext[QueryDeps], name: str) -> list[dict]:
    """Find graph entities matching a name (exact / alias / substring). Use the
    returned `entity_id` with `entity_facts` / `connect_entities`."""

    return [
        {"entity_id": e.entity_id, "name": e.name, "type": e.entity_type, "mentions": e.mentions}
        for e in retrieval.find_entities(ctx.deps, name)
    ]


@retriever_agent.tool
def entity_facts(ctx: RunContext[QueryDeps], entity_id: int) -> list[dict]:
    """All facts (relations + attributes) about one entity, most-corroborated
    first."""

    return [_fact_row(h) for h in retrieval.entity_facts(ctx.deps, entity_id, limit=14)]


@retriever_agent.tool
def facts_by_predicate(ctx: RunContext[QueryDeps], predicate: str) -> list[dict]:
    """Relations whose predicate matches ``predicate`` — expanded across
    synonyms + specializations so synonym-labelled edges aren't missed."""

    preds = expansion.expand_predicate(ctx.deps, predicate)
    return [_fact_row(h) for h in retrieval.relations_with_predicate(ctx.deps, preds, limit=12)]


@retriever_agent.tool
def facts_by_property(ctx: RunContext[QueryDeps], property_name: str) -> list[dict]:
    """Attribute facts (values + units + conditions) for a property — expanded
    across synonyms. Use for numeric / measured questions (cost, efficiency,
    lifetime…)."""

    props = expansion.expand_property(ctx.deps, property_name)
    return [_fact_row(h) for h in retrieval.attributes_with_property(ctx.deps, props, limit=12)]


@retriever_agent.tool
def connect_entities(ctx: RunContext[QueryDeps], entity_a: int, entity_b: int) -> dict:
    """Find how two entities relate: shared neighbours + short paths between
    them. Use for "how does X relate to Y" / multi-hop questions."""

    common = retrieval.common_neighbors(ctx.deps, entity_a, entity_b)
    paths = retrieval.shortest_paths(ctx.deps, entity_a, entity_b, cutoff=3)
    return {
        "common_neighbors": [{"entity_id": e.entity_id, "name": e.name} for e in common],
        "paths": [
            " → ".join(
                f"{n}" + (f" —[{p}]→" if i < len(path.predicates) else "")
                for i, (n, p) in enumerate(
                    zip(path.names, [*path.predicates, ""], strict=False)
                )
            )
            for path in paths
        ],
    }


@retriever_agent.tool
def read_source(ctx: RunContext[QueryDeps], ref: str) -> dict:
    """Read the actual source behind a fact ref ("relation:NNN" / "attribute:NNN"):
    the verbatim sentence it was extracted from, the surrounding paragraph, and the
    citation. Use when wording, conditions, or mechanism matter."""

    view = source_view(ctx.deps, ref)
    return view if view is not None else {"error": f"unknown or malformed ref: {ref}"}


def _sample_retriever() -> RetrieverResult:
    return RetrieverResult(
        reasoning="[SAMPLE_MODE] stub retrieval; no live embeddings.",
        selected=[],
        notes="sample mode — empty bundle.",
    )


async def retrieve(
    *,
    question: str,
    deps: QueryDeps,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model=None,
    request_limit: int = 8,
    triggered_by: str = "chat_loop",
) -> tuple[RetrieverResult, int | None]:
    """Run one Retriever turn. Returns ``(result, agent_call_id)``."""

    def output_summary(r: RetrieverResult) -> str:
        return f"selected={len(r.selected)} facts; {r.notes[:80]}"

    output, call = await run_tool_agent(
        agent=retriever_agent,
        user_prompt=f"Question: {question}",
        system_prompt=_SYSTEM_PROMPT,
        deps=deps,
        sample_factory=_sample_retriever,
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=ACTION,
        triggered_by=triggered_by,
        target_agent="Answerer",
        input_summary=question[:200],
        output_summary_fn=output_summary,
        confidence=0.7,
        request_limit=request_limit,
        model_preference=MODEL_PREFERENCE,
    )
    return output, call.agent_call_id if call else None
