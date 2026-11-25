"""Explorer — the agentic, tool-using hypothesis investigator.

``Agent[QueryDeps, ExploredHypothesis]``, run once per seed via ``run_tool_agent``.
Where the batched HypothesisGenerator reasons one-shot from compressed leg triples,
the Explorer is handed ONE structurally-flagged seed and *investigates* it on the
live graph — reading the verbatim source sentences (``read_source``), pulling
related facts, assembling a numeric landscape, tracing connections — then forms one
grounded, testable hypothesis OR bails (``worthwhile=False``) if there's nothing
genuine. It reasons from what the sources SAY, not from priors, and cites only the
refs it actually saw. Bounded by ``run_tool_agent``'s usage limits; stays anchored
to the seed (the deterministic entry point) rather than free-roaming.

Tools mirror the Retriever's toolbox + ``read_source`` — duplicated as thin
wrappers (each agent binds its own ``@agent.tool``); the bodies are the pure
``retrieval`` / ``grounding`` functions.
"""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.kg_intelligence.agents.base import AgentCallSinks, ModelPreference
from memex.kg_intelligence.embeddings import embed_texts
from memex.kg_query import expansion, retrieval
from memex.kg_query.agents.runtime import run_tool_agent
from memex.kg_query.agents.schemas import ExploredHypothesis
from memex.kg_query.deps import QueryDeps
from memex.kg_query.grounding import GroundedFact, source_view
from memex.kg_query.retrieval import FactHit
from memex.kg_query.seeds import Seed

AGENT_NAME = "Explorer"
ACTION = "explore_seed"
MODEL_PREFERENCE: ModelPreference = "reasoning"

_SYSTEM_PROMPT = """\
You are the Explorer in a knowledge-graph hypothesis engine, investigating leads
for a scientist studying a stated RESEARCH TOPIC. You are handed ONE structural
LEAD found in the graph (a bridge between unlinked concepts, a numeric landscape, a
contested edge, a gap) and a few starting facts. INVESTIGATE it on the live graph
and produce ONE grounded, testable hypothesis a researcher on that topic would
actually want to pursue — or conclude honestly that there is nothing worth pursuing.

How to investigate:
- START from the lead's facts. Use `read_source` to read the VERBATIM sentence +
  context — reason from what the papers actually SAY, not from your own knowledge.
  The compressed fact ("X relates to Y") often hides the real finding
  ("counter-intuitively, X increases Y").
- DIG with the tools: `entity_facts` for more about an entity; `facts_by_property`
  to assemble a numeric landscape (values + conditions); `search_facts` (semantic —
  scope `kind` to your lane: "attribute" to broaden a landscape with differently-named
  measurements, "relation" for a relational neighbourhood); `similar_entities` to
  find peers/analogues of an entity by meaning ("different entities, same mechanism")
  and alternative bridges; `facts_by_predicate` for related relations;
  `connect_entities` for paths/shared neighbours.
- STAY ANCHORED to the lead and the topic. Do not wander.
- Be economical: limited tool budget. A few probes that show a dud → STOP, bail.

What counts as WORTHWHILE — the scientist's bar:
- A worthwhile hypothesis is a lead about the RESEARCH TOPIC that a domain scientist
  would pursue: a MECHANISM, a TRADE-OFF, a CAUSAL or POLICY relationship, a
  quantitative threshold/scaling — something that, if true, changes understanding or
  a decision.
- Set `worthwhile=false` for: a DATA-QUALITY observation (a mislabeled, double-counted,
  or mis-dated figure — that's data hygiene, not science); a single ENTITY's business
  or reporting trivia (one company's portfolio, one country's inventory count); an
  OFF-TOPIC tangent only loosely related to the research topic; or a restatement /
  definitional consequence of an existing fact. A well-grounded dud is still a dud.

Then output:
- `worthwhile`: false if the lead fails the bar above (leave other fields brief).
- `statement`: the hypothesis — a NON-OBVIOUS, TESTABLE claim whose premises are
  facts you found but whose CONCLUSION is not already stated (an inferential leap).
- `mechanism`: why it might hold, grounded in what you read.
- `leg_refs`: the fact refs ("relation:NNN" / "attribute:NNN") you ACTUALLY saw via
  tools and that your mechanism leans on. Never invent a ref.
- `plausibility` (0-1): how mechanistically coherent the claim is on its face.
- `reasoning`: your investigation path + why you concluded what you did.
"""

explorer_agent: Agent[QueryDeps, ExploredHypothesis] = Agent(
    model=None,
    deps_type=QueryDeps,
    output_type=ExploredHypothesis,
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


@explorer_agent.tool
async def search_facts(ctx: RunContext[QueryDeps], query: str, kind: str = "both") -> list[dict]:
    """Semantic search over per-fact embeddings — find facts by MEANING (catches
    differently-worded predicates/properties that exact-name lookup misses). `kind`:
    "attribute" (measured values — broaden a numeric landscape), "relation"
    (entity→entity — a relational neighbourhood), or "both". Scope it to your seed's
    lane to dig along the same dimension."""

    vecs = await embed_texts([query], ctx.deps.settings)
    if not vecs:
        return []
    kinds = (kind,) if kind in ("relation", "attribute") else ("relation", "attribute")
    return [_fact_row(h) for h in retrieval.semantic_facts(ctx.deps, vecs[0], kinds=kinds, k=10)]


@explorer_agent.tool
def find_entity(ctx: RunContext[QueryDeps], name: str) -> list[dict]:
    """Find graph entities matching a NAME (exact / alias / substring)."""

    return [
        {"entity_id": e.entity_id, "name": e.name, "type": e.entity_type, "mentions": e.mentions}
        for e in retrieval.find_entities(ctx.deps, name)
    ]


@explorer_agent.tool
async def similar_entities(ctx: RunContext[QueryDeps], concept: str) -> list[dict]:
    """Find entities CLOSE IN MEANING to a concept/name via embeddings (not exact
    name) — analogies and "different entities, same mechanism". Returns `entity_id`s
    for `entity_facts` / `connect_entities`. Use to find peers of a seed entity or
    alternative bridges."""

    vecs = await embed_texts([concept], ctx.deps.settings)
    if not vecs:
        return []
    return [
        {"entity_id": e.entity_id, "name": e.name, "type": e.entity_type, "mentions": e.mentions}
        for e in retrieval.semantic_entities(ctx.deps, vecs[0], k=10)
    ]


@explorer_agent.tool
def entity_facts(ctx: RunContext[QueryDeps], entity_id: int) -> list[dict]:
    """All facts (relations + attributes) about one entity, most-corroborated first."""

    return [_fact_row(h) for h in retrieval.entity_facts(ctx.deps, entity_id, limit=14)]


@explorer_agent.tool
def facts_by_predicate(ctx: RunContext[QueryDeps], predicate: str) -> list[dict]:
    """Relations whose predicate matches ``predicate`` (expanded across synonyms)."""

    preds = expansion.expand_predicate(ctx.deps, predicate)
    return [_fact_row(h) for h in retrieval.relations_with_predicate(ctx.deps, preds, limit=12)]


@explorer_agent.tool
def facts_by_property(ctx: RunContext[QueryDeps], property_name: str) -> list[dict]:
    """Attribute facts (values + units + conditions) for a property — assemble a
    numeric landscape. Expanded across synonyms."""

    props = expansion.expand_property(ctx.deps, property_name)
    return [_fact_row(h) for h in retrieval.attributes_with_property(ctx.deps, props, limit=14)]


@explorer_agent.tool
def connect_entities(ctx: RunContext[QueryDeps], entity_a: int, entity_b: int) -> dict:
    """How two entities relate: shared neighbours + short paths between them."""

    common = retrieval.common_neighbors(ctx.deps, entity_a, entity_b)
    paths = retrieval.shortest_paths(ctx.deps, entity_a, entity_b, cutoff=3)
    return {
        "common_neighbors": [{"entity_id": e.entity_id, "name": e.name} for e in common],
        "paths": [" → ".join(p.names) for p in paths],
    }


@explorer_agent.tool
def read_source(ctx: RunContext[QueryDeps], ref: str) -> dict:
    """Read the actual source behind a fact ref ("relation:NNN" / "attribute:NNN"):
    the verbatim sentence, surrounding context, and citation. Use this to reason
    from what the paper SAYS, not from prior knowledge."""

    view = source_view(ctx.deps, ref)
    return view if view is not None else {"error": f"unknown or malformed ref: {ref}"}


def _seed_prompt(topic: str, seed: Seed, legs: list[GroundedFact]) -> str:
    lines = [
        f"Research topic: {topic}",
        "",
        f"Lead [{seed.lane}]: {seed.title}",
        f"Why it was flagged: {seed.rationale}",
        "",
        "Starting facts (use read_source on a ref to read its verbatim sentence + context):",
    ]
    for f in legs:
        cite = f.citation.title if f.citation else (f.source_doc_id or "—")
        lines.append(f"  {f.kind}:{f.fact_id} — {f.canonical}  (source: {cite})")
    if not legs:
        lines.append("  (no starting facts — search the graph from the lead's terms)")
    lines += [
        "",
        "Investigate this lead, then propose ONE grounded hypothesis or set "
        "worthwhile=false if there's nothing genuine.",
    ]
    return "\n".join(lines)


def _sample_explore(seed: Seed, legs: list[GroundedFact]) -> ExploredHypothesis:
    return ExploredHypothesis(
        reasoning="[SAMPLE_MODE] stub exploration.",
        worthwhile=True,
        statement=f"[SAMPLE] Hypothesis from {seed.title}.",
        mechanism="[SAMPLE_MODE] stub mechanism.",
        leg_refs=[f"{f.kind}:{f.fact_id}" for f in legs],
        plausibility=0.6,
    )


async def explore(
    *,
    topic: str,
    seed: Seed,
    legs: list[GroundedFact],
    deps: QueryDeps,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    request_limit: int = 9,
    tool_calls_limit: int = 12,
    triggered_by: str = "hypothesis_loop",
    parent_call_id: int | None = None,
) -> tuple[ExploredHypothesis, int | None]:
    """Run one focused Explorer investigation of ``seed`` for ``topic``."""

    def output_summary(h: ExploredHypothesis) -> str:
        if not h.worthwhile:
            return "bailed (worthwhile=false)"
        return f"hypothesis ({len(h.leg_refs)} legs, plaus {h.plausibility:.2f})"

    output, call = await run_tool_agent(
        agent=explorer_agent,
        user_prompt=_seed_prompt(topic, seed, legs),
        system_prompt=_SYSTEM_PROMPT,
        deps=deps,
        sample_factory=lambda: _sample_explore(seed, legs),
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=ACTION,
        triggered_by=triggered_by,
        target_agent="VerifyEvidence",
        input_summary=f"[{seed.lane}] {seed.title}"[:200],
        output_summary_fn=output_summary,
        confidence=0.7,
        request_limit=request_limit,
        tool_calls_limit=tool_calls_limit,
        model_preference=MODEL_PREFERENCE,
        parent_call_id=parent_call_id,
    )
    return output, call.agent_call_id if call else None
