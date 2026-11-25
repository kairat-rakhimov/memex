"""The GroundingChecker — adversarial verifier of the Answerer's claims.

``Agent[None, GroundingVerdict]``. For each claim it re-reads the cited evidence
and judges whether the bundle actually supports the assertion. Distinct from the
hypothesis-side VerifyEvidence (which re-reads L3 prose per leg): this checks
answer-claims against the already-grounded bundle. An unsupported claim sends the
loop back to the Answerer for a bounded revision.
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
from memex.kg_query.agents.schemas import AnswerResult, GroundingVerdict
from memex.kg_query.grounding import GroundedFact, format_bundle

AGENT_NAME = "GroundingChecker"
ACTION = "check_grounding"
MODEL_PREFERENCE: ModelPreference = "reasoning"

_SYSTEM_PROMPT = """\
You are the GroundingChecker — an adversarial verifier in a grounded Q&A system.
You receive a question, the NUMBERED evidence bundle, and the Answerer's claims
(each with the bundle indices it cites as support).

For EACH claim, decide whether the cited evidence genuinely supports it:
- supported=true only if the cited facts state or directly entail the claim.
- supported=false if the claim overreaches the evidence, mis-cites, conflates
  conditions, or asserts something the bundle does not contain.
Be strict: default to unsupported when the link is loose or the citation doesn't
match. A correct-sounding claim with no real backing in the bundle is unsupported.

In `revision_request`, give the Answerer concrete guidance on what to drop or
soften. In `reasoning`, summarise your overall judgement.
"""

checker_agent: Agent[None, GroundingVerdict] = Agent(
    model=None,
    output_type=GroundingVerdict,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


def _format_claims(answer: AnswerResult) -> str:
    if not answer.claims:
        return "(no structured claims)"
    return "\n".join(
        f"Claim {i}: {c.text}  [cites: {', '.join(map(str, c.support)) or 'none'}]"
        for i, c in enumerate(answer.claims, start=1)
    )


def _sample_verdict(answer: AnswerResult) -> GroundingVerdict:
    # Flag the first claim so an offline SAMPLE_MODE run exercises the revision
    # branch.
    from memex.kg_query.agents.schemas import ClaimVerdict

    verdicts = []
    for i, _ in enumerate(answer.claims, start=1):
        verdicts.append(
            ClaimVerdict(claim_index=i, supported=(i != 1), reasoning="[SAMPLE_MODE] stub verdict.")
        )
    return GroundingVerdict(
        reasoning="[SAMPLE_MODE] stub grounding check.",
        verdicts=verdicts,
        revision_request="[SAMPLE_MODE] soften claim 1." if answer.claims else "",
    )


async def check(
    *,
    question: str,
    answer: AnswerResult,
    facts: list[GroundedFact],
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    triggered_by: str = "chat_loop",
    parent_call_id: int | None = None,
) -> tuple[GroundingVerdict, int | None]:
    """Run one GroundingChecker turn. Returns ``(verdict, agent_call_id)``."""

    user_prompt = "\n".join(
        [
            f"Question: {question}",
            "",
            "Evidence bundle:",
            format_bundle(facts, with_snippet=False),
            "",
            "Answerer's claims:",
            _format_claims(answer),
        ]
    )

    def output_summary(v: GroundingVerdict) -> str:
        unsupported = len(v.unsupported_indices())
        return f"{len(v.verdicts)} checked, {unsupported} unsupported"

    result, call, _run = await execute_agent_call(
        agent=checker_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=lambda: _sample_verdict(answer),
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=ACTION,
        triggered_by=triggered_by,
        target_agent="Answerer",
        input_summary=f"{len(answer.claims)} claims",
        output_summary_fn=output_summary,
        confidence=0.7,
        parent_call_id=parent_call_id,
        model_preference=MODEL_PREFERENCE,
    )
    return result, call.agent_call_id if call else None
