"""NoveltyJudge — batched assertional-novelty scoring for hypothesis ranking.

``Agent[None, NoveltyBatch]``, ONE call per sweep over all surviving candidates.
It replaces embedding-cosine novelty in the composite score: cosine measures
*topical* similarity ("about the same things"), which mis-scores exactly the
highest-value leads — a hypothesis synthesised from existing facts shares all
their vocabulary, so it embeds CLOSE and looks like a restatement even though its
conclusion is asserted nowhere. Cosine also places opposites adjacent ("X
increases Y" ≈ "X decreases Y"), so it cannot tell a contested edge from an
agreement. This judge reads each candidate against its nearest KG facts and
decides whether the *claim* is already made — a reasoning judgment.

The cheap embedding restatement pre-filter still runs first (gross restatements
never reach here); embedding novelty stays as the fallback when this judge is
unavailable (SAMPLE_MODE / error).
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
from memex.kg_query.agents.schemas import NoveltyBatch

AGENT_NAME = "NoveltyJudge"
ACTION = "judge_novelty"
MODEL_PREFERENCE: ModelPreference = "reasoning"

_SYSTEM_PROMPT = """\
You are NoveltyJudge in a knowledge-graph hypothesis engine. You score the
ASSERTIONAL novelty of each candidate hypothesis: is its CONCLUSION already
asserted in the knowledge graph, or is it a genuine inferential leap beyond what
the graph states?

You receive a BATCH of candidates. For EACH you see its statement and the KG
facts most TOPICALLY similar to it (the nearest by embedding). Topical similarity
is NOT the same as the claim already being made:
- A hypothesis synthesised from several existing facts reuses all their
  vocabulary, so it looks similar — yet it asserts something none of them states.
  That is HIGH novelty, not a restatement.
- Embeddings place opposites adjacent ("X increases Y" vs "X decreases Y"), so a
  contested claim can look identical to the fact it contradicts. Tell agreement
  from contradiction yourself.

See through ELABORATION to the CORE claim. A statement can be long, specific, and
hedged ("lifecycle GHG intensities closer to grey than green", "in the scenarios
studied", "by 2035") yet its core assertion is still just a restatement or a
DEFINITIONAL consequence of a near fact. Length and specificity are not novelty —
strip the qualifiers and ask what is actually being claimed. If the near facts
DEFINE a term (e.g. "blue hydrogen is grey hydrogen with added CCS"), then a
conclusion that follows from that definition (e.g. "blue hydrogen without real
CCS is high-emission like grey") is OBVIOUS, not novel, however elaborately
worded.

For each candidate emit `novelty` ∈ [0,1]:
- 0.0–0.2 — the conclusion is ALREADY stated by one of the near facts (a
  restatement or paraphrase), trivially entailed by a single one, or a
  dressed-up consequence of a DEFINITION among the near facts.
- 0.3–0.6 — partly anticipated: the pieces are present but the specific claim is
  a modest step past them.
- 0.7–1.0 — a genuinely NEW, non-obvious assertion stated nowhere in the near
  facts: it requires COMBINING them in a way an expert wouldn't take for granted,
  CONTRADICTS one of them in a substantive way, or BRIDGES distant facts.
  Synthesised, contested, and bridge hypotheses belong here even though they share
  the facts' vocabulary — but only when the leap is real, not definitional.

Judge the CLAIM, not the topic, and not the wordcount. `why` is one line per
candidate ("conclusion already stated by fact N" / "definitional consequence of
fact N, just elaborated" / "genuine synthesis of facts N+M" / "contests fact N").
Return exactly one judgment per candidate, echoing its [#index]. Put any
cross-candidate reasoning in `reasoning`.
"""

judge_agent: Agent[None, NoveltyBatch] = Agent(
    model=None,
    output_type=NoveltyBatch,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


def _format_items(items: list[tuple[str, list[str]]]) -> str:
    blocks = []
    for i, (statement, neighbors) in enumerate(items):
        nb = "\n".join(f"    - {n}" for n in neighbors) or "    (no near facts)"
        blocks.append(
            f"[#{i}]\n  CANDIDATE: {statement}\n  Nearest existing KG facts:\n{nb}"
        )
    return "\n\n".join(blocks)


def _sample_judge() -> NoveltyBatch:
    # Empty judgments → the orchestration falls back to embedding novelty per
    # candidate, so SAMPLE_MODE still ranks without the LLM.
    return NoveltyBatch(reasoning="[SAMPLE_MODE] embedding-novelty fallback.")


async def judge(
    *,
    items: list[tuple[str, list[str]]],
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    triggered_by: str = "hypothesis_loop",
    parent_call_id: int | None = None,
) -> tuple[NoveltyBatch, int | None]:
    user_prompt = "\n".join(
        [
            f"Score the assertional novelty of these {len(items)} candidate "
            "hypotheses against their nearest existing facts.",
            "",
            _format_items(items),
        ]
    )

    def output_summary(b: NoveltyBatch) -> str:
        if not b.judgments:
            return "0 judged (fallback)"
        mean = sum(j.novelty for j in b.judgments) / len(b.judgments)
        return f"{len(b.judgments)} judged, mean novelty {mean:.2f}"

    result, call, _run = await execute_agent_call(
        agent=judge_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=_sample_judge,
        log_prefix=AGENT_NAME,
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=ACTION,
        triggered_by=triggered_by,
        target_agent="ReportWriter",
        input_summary=f"{len(items)} candidates",
        output_summary_fn=output_summary,
        confidence=0.7,
        parent_call_id=parent_call_id,
        model_preference=MODEL_PREFERENCE,
    )
    return result, call.agent_call_id if call else None
