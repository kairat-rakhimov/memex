"""FinalReviewer — audits the hypothesis report before it ships.

``Agent[None, FinalReview]``. Checks coverage (a spread of hypothesis classes),
citations (every hypothesis grounded), and over-claim (hypotheses not stated as
findings). An unapproved review sends one bounded revision back to the
ReportWriter.
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
from memex.kg_query.agents.schemas import FinalReview

AGENT_NAME = "FinalReviewer"
ACTION = "review_report"
MODEL_PREFERENCE: ModelPreference = "reasoning"

_SYSTEM_PROMPT = """\
You are the FinalReviewer — the last gate before a hypothesis report ships to a
researcher. You receive the research topic, the report, and a summary of the
hypotheses it should cover.

Audit for:
- OVER-CLAIM: anything stated as an established finding rather than a hypothesis,
  or stronger than its evidence warrants.
- CITATIONS: every hypothesis should point to its grounding evidence; flag
  uncited claims.
- COVERAGE: the report should reflect the range of hypothesis classes available,
  not collapse onto one. Flag if low-confidence/contested items are presented
  without their caveats.

List concrete `issues` (empty if clean). Set `approved`=false only if a revision
is genuinely warranted. Summarise in `reasoning`.
"""

reviewer_agent: Agent[None, FinalReview] = Agent(
    model=None,
    output_type=FinalReview,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


def _sample_review() -> FinalReview:
    return FinalReview(
        reasoning="[SAMPLE_MODE] stub review.",
        issues=["[SAMPLE_MODE] add a caveat to the contested hypothesis."],
        approved=False,
    )


async def review(
    *,
    base_query: str,
    report_markdown: str,
    coverage_summary: str,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    triggered_by: str = "hypothesis_loop",
    parent_call_id: int | None = None,
) -> tuple[FinalReview, int | None]:
    user_prompt = "\n".join(
        [
            f"Research topic: {base_query}",
            "",
            f"Hypotheses that should be covered:\n{coverage_summary}",
            "",
            "Report under review:",
            report_markdown,
        ]
    )

    def output_summary(r: FinalReview) -> str:
        return f"approved={r.approved}, {len(r.issues)} issue(s)"

    result, call, _run = await execute_agent_call(
        agent=reviewer_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=_sample_review,
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=ACTION,
        triggered_by=triggered_by,
        target_agent="ReportWriter",
        input_summary=f"report {len(report_markdown)} chars",
        output_summary_fn=output_summary,
        confidence=0.7,
        parent_call_id=parent_call_id,
        model_preference=MODEL_PREFERENCE,
    )
    return result, call.agent_call_id if call else None
