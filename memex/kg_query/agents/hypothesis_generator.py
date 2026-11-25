"""HypothesisGenerator — articulates a testable claim from each structural seed.

``Agent[None, GeneratorResult]``. A seed is a structural pattern (a bridge, an
outlier, a contested value) with grounded legs; the generator turns it into a
hypothesis whose PREMISES are the legs but whose CONCLUSION is not directly
stated — or rejects it as dull. Batched over several seeds per call (verification
is the per-candidate fan-out, not this).
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
from memex.kg_query.agents.schemas import GeneratorResult, HypothesisCandidate
from memex.kg_query.grounding import GroundedFact
from memex.kg_query.seeds import Seed

AGENT_NAME = "HypothesisGenerator"
ACTION = "generate_hypotheses"
MODEL_PREFERENCE: ModelPreference = "reasoning"

_SYSTEM_PROMPT = """\
You are the HypothesisGenerator in a knowledge-graph hypothesis engine. You are
given a research topic and a batch of numbered SEEDS. Each seed is a structural
pattern found in the graph (a bridge between unlinked concepts, a numeric
outlier, a value that differs across entities or conditions, a multi-predicate
pair) together with its grounded LEG facts.

For each seed, produce at most one hypothesis:
- A hypothesis is a NON-OBVIOUS, TESTABLE claim whose PREMISES are the seed's leg
  facts but whose CONCLUSION is NOT already stated in them. If the only honest
  reading of the seed is a restatement of an existing fact, set worthwhile=false.
- `statement`: the claim, crisply. `mechanism`: why it might hold, grounded in
  the legs. `leg_refs`: the leg refs ("kind:id") your mechanism actually leans on
  — use ONLY refs shown in the seed. `hypothesis_class`: the seed's lane.
- `plausibility` (0-1): how mechanistically coherent the claim is on its face.
- Reject trivia, tautologies, and artefacts of near-duplicate entities
  (worthwhile=false) — quality over quantity.

Explain your overall approach in `reasoning`.
"""

generator_agent: Agent[None, GeneratorResult] = Agent(
    model=None,
    output_type=GeneratorResult,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


def _format_seed(index: int, seed: Seed, legs: list[GroundedFact]) -> str:
    lines = [f"Seed {index} [{seed.lane}]: {seed.title}", f"  signal: {seed.rationale}", "  legs:"]
    for f in legs:
        cite = f.citation.title if f.citation else (f.source_doc_id or "—")
        lines.append(f"    {f.kind}:{f.fact_id} — {f.canonical}  (source: {cite}; corrob {f.corroboration})")
    if not legs:
        lines.append("    (no grounded legs)")
    return "\n".join(lines)


def _sample_generator(seeds_with_legs: list[tuple[Seed, list[GroundedFact]]]) -> GeneratorResult:
    cands = []
    for i, (seed, legs) in enumerate(seeds_with_legs):
        cands.append(
            HypothesisCandidate(
                seed_index=i,
                worthwhile=True,
                statement=f"[SAMPLE] Hypothesis from {seed.title}.",
                mechanism="[SAMPLE_MODE] stub mechanism.",
                leg_refs=[f"{f.kind}:{f.fact_id}" for f in legs],
                hypothesis_class=seed.lane,
                plausibility=0.6,
                reasoning="[SAMPLE_MODE] stub.",
            )
        )
    return GeneratorResult(reasoning="[SAMPLE_MODE] stub generation.", candidates=cands)


async def generate(
    *,
    base_query: str,
    seeds_with_legs: list[tuple[Seed, list[GroundedFact]]],
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    triggered_by: str = "hypothesis_loop",
    batch_label: str = "",
    parent_call_id: int | None = None,
) -> tuple[GeneratorResult, int | None]:
    seeds_block = "\n\n".join(
        _format_seed(i, seed, legs) for i, (seed, legs) in enumerate(seeds_with_legs)
    )
    user_prompt = f"Research topic: {base_query}\n\nSeeds:\n\n{seeds_block}"

    def output_summary(r: GeneratorResult) -> str:
        kept = sum(1 for c in r.candidates if c.worthwhile)
        return f"{kept}/{len(r.candidates)} worthwhile hypotheses"

    result, call, _run = await execute_agent_call(
        agent=generator_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=lambda: _sample_generator(seeds_with_legs),
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=ACTION,
        triggered_by=triggered_by,
        target_agent="VerifyEvidence",
        input_summary=f"{len(seeds_with_legs)} seeds {batch_label}".strip(),
        output_summary_fn=output_summary,
        confidence=0.65,
        parent_call_id=parent_call_id,
        model_preference=MODEL_PREFERENCE,
    )
    return result, call.agent_call_id if call else None
