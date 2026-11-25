"""ReportWriter — turns the ranked, verified hypotheses into a researcher report.

``Agent[None, ReportResult]``. Writes the curated top-N narrative (the appendix
of also-rans is rendered deterministically by the orchestration). It is handed
each hypothesis with its class, scores, verification verdict, and grounded
evidence; it must cite sources and frame claims as hypotheses, never findings.
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
from memex.kg_query.agents.schemas import ReportResult

AGENT_NAME = "ReportWriter"
ACTION = "write_report"
MODEL_PREFERENCE: ModelPreference = "reasoning"

_SYSTEM_PROMPT = """\
You are the ReportWriter for a knowledge-graph hypothesis engine. You receive a
research topic and a RANKED list of verified hypotheses (H1, H2, …), each with its
statement, mechanism, scores, verification verdict, and grounded evidence.

Produce a STRUCTURED report (not a single Markdown blob):
- `intro`: one or two short paragraphs framing what the sweep surfaced and that
  these are tentative leads, not established findings.
- `hypotheses`: exactly ONE entry per input hypothesis, in the SAME ORDER (H1
  first). For each, write:
  - `title`: a short descriptive name (≤ ~12 words), no numbering.
  - `hypothesis`: the claim, stated AS A HYPOTHESIS (not a finding).
  - `rationale`: the mechanism / reasoning — why it might hold; flag contested or
    low-confidence cases honestly here.
  - `how_to_test`: what evidence or analysis would test or refute it.
- `conclusion`: optional short synthesis of cross-cutting themes.

Do NOT include scores, confidence numbers, citations, headings, or an evidence
list — the scores and the grounded-evidence block are added deterministically
around your text. Do NOT invent facts, numbers, or sources. Keep every claim a
hypothesis. Put your editorial reasoning in `reasoning`.
"""

writer_agent: Agent[None, ReportResult] = Agent(
    model=None,
    output_type=ReportResult,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


def _sample_report(hypotheses_block: str) -> ReportResult:
    # Empty hypotheses → the orchestration renders each section from the
    # deterministic candidate data (statement / mechanism / legs), so SAMPLE_MODE
    # still produces a complete, grounded body without any LLM prose.
    return ReportResult(
        reasoning="[SAMPLE_MODE] deterministic render from candidate data.",
        intro="[SAMPLE_MODE] Hypotheses rendered from candidate data (no LLM prose).",
    )


async def write(
    *,
    base_query: str,
    hypotheses_block: str,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    revision_request: str | None = None,
    prior_report: str | None = None,
    triggered_by: str = "hypothesis_loop",
    parent_call_id: int | None = None,
) -> tuple[ReportResult, int | None]:
    parts = [f"Research topic: {base_query}", "", "Ranked verified hypotheses:", hypotheses_block]
    if revision_request and prior_report:
        parts += [
            "",
            "The reviewer flagged issues with your previous report. Revise it to "
            "address them (tighten over-claims, add missing citations, improve "
            "coverage).",
            f"Reviewer feedback: {revision_request}",
            "",
            "Previous report:",
            prior_report,
        ]
    user_prompt = "\n".join(parts)

    def output_summary(r: ReportResult) -> str:
        return f"{len(r.hypotheses)} hypothesis block(s)"

    result, call, _run = await execute_agent_call(
        agent=writer_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=lambda: _sample_report(hypotheses_block),
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action="revise_report" if revision_request else ACTION,
        triggered_by=triggered_by,
        target_agent="FinalReviewer",
        input_summary=base_query[:120],
        output_summary_fn=output_summary,
        confidence=0.75,
        parent_call_id=parent_call_id,
        model_preference=MODEL_PREFERENCE,
    )
    return result, call.agent_call_id if call else None
