"""VerifyEvidence — per-candidate adversarial grounding of a hypothesis.

``Agent[None, VerifyVerdict]``, one call per surviving hypothesis (the tightest
per-leg grounding). It re-reads each leg's L3 source
prose and decides, leg by leg, whether the source actually supports it; a broken
load-bearing leg downgrades the hypothesis. Distinct from the Q&A
GroundingChecker (which checks answer claims against an already-grounded bundle):
this re-reads the underlying source span per leg.
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
from memex.kg_query.agents.schemas import HypothesisCandidate, LegVerdict, VerifyVerdict
from memex.kg_query.grounding import GroundedFact

AGENT_NAME = "VerifyEvidence"
ACTION = "verify_hypothesis"
MODEL_PREFERENCE: ModelPreference = "reasoning"

_SYSTEM_PROMPT = """\
You are VerifyEvidence — an adversarial verifier in a hypothesis engine. You are
given ONE hypothesis (statement + mechanism) and its LEG facts, each with the
SOURCE PROSE it was extracted from.

Your job is to check that the hypothesis is built on REAL premises — NOT to
demand that the premises already prove it. A hypothesis is, by design, an
inferential LEAP beyond its legs: the conclusion is expected to be unstated in
the sources. That leap is fine. What is NOT fine is a leg that is hallucinated or
contradicted by its source.

For EACH leg, re-read the source snippet and judge ONLY the LEG FACT itself:
- supported=true if the snippet states or reasonably reflects the leg fact (the
  relationship/value it asserts), even if the snippet doesn't mention the
  hypothesis. Allow for the snippet being a windowed excerpt that may clip the
  exact phrasing — judge the substance, not an exact string match.
- supported=false ONLY if the snippet is off-topic for the leg, contradicts it,
  or the leg fact is clearly a hallucinated/drifted extraction.

Then judge the hypothesis:
- `holds`=true if the LOAD-BEARING legs (the ones the mechanism actually rests on)
  are faithful real facts. Do NOT set holds=false merely because the legs don't
  by themselves prove the conclusion — that inferential gap is the hypothesis.
- `holds`=false only when a load-bearing leg is hallucinated/contradicted, so the
  hypothesis rests on a fabricated premise.
- `confidence` (0-1): your confidence in the verdict.
Be concrete in `reasoning`, and say which legs are load-bearing.
"""

verify_agent: Agent[None, VerifyVerdict] = Agent(
    model=None,
    output_type=VerifyVerdict,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


def _format_legs(legs: list[GroundedFact]) -> str:
    blocks = []
    for f in legs:
        cite = f.citation.title if f.citation else (f.source_doc_id or "—")
        # Use the FULL grounded chunk span — a fact's supporting sentence can sit
        # anywhere in its source chunk; clipping the prose made the verifier
        # reject correctly-grounded legs it simply hadn't been shown.
        snippet = f.chunk_span or "(no source span)"
        blocks.append(
            f"Leg {f.kind}:{f.fact_id}\n"
            f"  fact: {f.canonical}\n"
            f'  as stated: "{f.raw}"\n'
            f"  source: {cite}\n"
            f'  source prose: "{snippet}"'
        )
    return "\n\n".join(blocks) or "(no legs)"


def _sample_verify(candidate: HypothesisCandidate, legs: list[GroundedFact]) -> VerifyVerdict:
    return VerifyVerdict(
        reasoning="[SAMPLE_MODE] stub verification.",
        leg_verdicts=[LegVerdict(ref=f"{f.kind}:{f.fact_id}", supported=True, note="stub") for f in legs],
        holds=True,
        confidence=0.6,
    )


async def verify(
    *,
    candidate: HypothesisCandidate,
    legs: list[GroundedFact],
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    triggered_by: str = "hypothesis_loop",
    parent_call_id: int | None = None,
) -> tuple[VerifyVerdict, int | None]:
    user_prompt = "\n".join(
        [
            f"Hypothesis: {candidate.statement}",
            f"Mechanism: {candidate.mechanism}",
            "",
            "Legs (with source prose):",
            _format_legs(legs),
        ]
    )

    def output_summary(v: VerifyVerdict) -> str:
        dropped = sum(1 for lv in v.leg_verdicts if not lv.supported)
        return f"holds={v.holds}, {dropped} leg(s) unsupported, conf={v.confidence:.2f}"

    result, call, _run = await execute_agent_call(
        agent=verify_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=lambda: _sample_verify(candidate, legs),
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=ACTION,
        triggered_by=triggered_by,
        target_agent="ReportWriter",
        input_summary=candidate.statement[:180],
        output_summary_fn=output_summary,
        confidence=0.7,
        parent_call_id=parent_call_id,
        model_preference=MODEL_PREFERENCE,
    )
    return result, call.agent_call_id if call else None
