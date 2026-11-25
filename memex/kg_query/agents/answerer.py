"""The Answerer — writes a grounded answer from ONLY the retrieved evidence.

``Agent[None, AnswerResult]`` run via the shared ``execute_agent_call``. It never
invents facts: every claim cites ``[n]`` indices into the grounded bundle, and
anything the corpus can't support goes in ``gaps`` ("not in corpus"), not the
prose.
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
from memex.kg_query.agents.schemas import AnswerResult, Claim
from memex.kg_query.grounding import GroundedFact, format_bundle

AGENT_NAME = "Answerer"
ACTION = "answer"
MODEL_PREFERENCE: ModelPreference = "reasoning"

_SYSTEM_PROMPT = """\
You are the Answerer in a grounded knowledge-graph Q&A system. You are given a
question and a NUMBERED bundle of evidence facts retrieved from a knowledge graph
(each with its canonical form, the raw "as stated" surface form, a source
citation, a corroboration count, and a source snippet).

Rules:
- Answer ONLY from the bundle. Do NOT use outside knowledge. If the bundle does
  not support part of the question, say so honestly in `gaps` — never fill a hole
  with a guess.
- Write a clear, well-structured Markdown answer. Cite evidence inline as `[n]`
  using the bundle indices. Every substantive sentence should carry a citation.
- Populate `claims`: one entry per distinct assertion, with `support` = the list
  of bundle indices that back it. A claim with no real support does not belong
  in the answer.
- Prefer well-corroborated facts; when facts conflict (e.g. different values
  under different conditions), report both with their conditions rather than
  picking one.
- `reasoning`: briefly explain how you used the evidence and any judgement calls.
"""

answerer_agent: Agent[None, AnswerResult] = Agent(
    model=None,
    output_type=AnswerResult,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


def _sample_answer(facts: list[GroundedFact]) -> AnswerResult:
    # Always emit one claim so an offline SAMPLE_MODE run exercises the
    # GroundingChecker→Answerer revision branch.
    cited = [1] if facts else []
    return AnswerResult(
        reasoning="[SAMPLE_MODE] deterministic stub answer.",
        answer="(SAMPLE_MODE) Placeholder grounded answer." + (" [1]" if facts else ""),
        claims=[Claim(text="Sample claim (stub).", support=cited)],
        gaps=["SAMPLE_MODE — no live retrieval."],
    )


async def answer(
    *,
    question: str,
    facts: list[GroundedFact],
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    revision_request: str | None = None,
    prior_answer: AnswerResult | None = None,
    triggered_by: str = "chat_loop",
    parent_call_id: int | None = None,
) -> tuple[AnswerResult, int | None]:
    """Run one Answerer turn. When ``revision_request`` is set this is a revision
    pass over ``prior_answer`` (the GroundingChecker rejected a claim)."""

    bundle = format_bundle(facts)
    prompt_parts = [f"Question: {question}", "", "Evidence bundle:", bundle]
    if revision_request and prior_answer is not None:
        prompt_parts += [
            "",
            "Your previous answer had unsupported claims. Revise it: drop or "
            "soften every claim the grounding check flagged, keeping only what "
            "the bundle supports.",
            f"Grounding feedback: {revision_request}",
            "",
            "Previous answer:",
            prior_answer.answer,
        ]
    user_prompt = "\n".join(prompt_parts)

    def output_summary(r: AnswerResult) -> str:
        return f"{len(r.claims)} claim(s), {len(r.gaps)} gap(s)"

    result, call, _run = await execute_agent_call(
        agent=answerer_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=lambda: _sample_answer(facts),
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action="revise_answer" if revision_request else ACTION,
        triggered_by=triggered_by,
        target_agent="GroundingChecker",
        input_summary=question[:200],
        output_summary_fn=output_summary,
        confidence=0.75,
        parent_call_id=parent_call_id,
        model_preference=MODEL_PREFERENCE,
    )
    return result, call.agent_call_id if call else None
