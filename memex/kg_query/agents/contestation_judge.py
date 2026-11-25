"""ContestationJudge — batched genuine-conflict gate for relation_contested seeds.

``Agent[None, ContestationBatch]``, one call per sweep. The relation_contested
seed lane fires on any entity pair with ≥2 distinct predicates, which conflates
three things: a GENUINE conflict (``increases`` vs ``replaces`` — the valuable
case the lane exists for), synonym SPRAWL (``combined_with`` / ``integrates`` /
``produced_using`` — all "uses"), and non-conflicting multi-facet relations
(``adopts`` / ``defines`` / ``regulates``). Only the first deserves the lane's
high structure value; the others produce dressed-up restatements of obvious facts
that float to the top on the structure boost alone.

Embeddings can't make this call (they place antonyms adjacent), so it's a
reasoning judgment. ``genuine=False`` seeds keep their hypothesis but lose the
contested structure bonus, so they rank on their other merits instead of leading.
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
from memex.kg_query.agents.schemas import ContestationBatch

AGENT_NAME = "ContestationJudge"
ACTION = "judge_contestation"
MODEL_PREFERENCE: ModelPreference = "reasoning"

_SYSTEM_PROMPT = """\
You are ContestationJudge in a knowledge-graph hypothesis engine. Each item is an
entity pair (subject → object) and the set of distinct predicates the graph
records between them. Decide whether that predicate set is a GENUINE conflict.

A pair lands here merely because it has ≥2 predicates — but that mixes three
cases, and only one is a real contestation:
- GENUINE conflict (`genuine=true`): the predicates actually OPPOSE or create a
  tension that a hypothesis could resolve — e.g. `increases` vs `replaces` /
  `alternative_to`, `cheaper_than` vs `more_expensive_than`, `causes` vs
  `prevents`. Resolving this tension is a real, non-obvious lead.
- SYNONYM sprawl (`genuine=false`): the predicates are near-synonyms all asserting
  the SAME relationship — e.g. `combined_with` / `coupled_with` / `integrates` /
  `produced_using` / `uses` / `has_property` / `related_to`. No conflict; any
  "contestation" framing would just restate an obvious/definitional fact.
- Multi-facet, non-conflicting (`genuine=false`): different but compatible
  relations — e.g. `adopts` / `defines` / `regulates` / `targets` / `imports`.
  A busy relationship, not a contradiction.

Judge the MEANING of the predicates, not their count. When the set is a mix,
ask: is there a real opposition a reader would find surprising, or do they all
point the same way? Emit one verdict per item, echoing its [#index], with a
one-line `why` ("increases vs replaces — genuine tension" / "all synonyms for
'uses CCS'"). Put cross-item reasoning in `reasoning`.
"""

judge_agent: Agent[None, ContestationBatch] = Agent(
    model=None,
    output_type=ContestationBatch,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


def _format_items(items: list[tuple[str, str, list[str]]]) -> str:
    blocks = []
    for i, (subject, obj, preds) in enumerate(items):
        plist = ", ".join(preds) or "(none)"
        blocks.append(f"[#{i}] {subject} → {obj}\n  predicates: {plist}")
    return "\n\n".join(blocks)


def _sample_judge() -> ContestationBatch:
    # Empty verdicts → the orchestration leaves seeds unclassified (treated as
    # genuine, the pre-judge default), so SAMPLE_MODE keeps the lane intact.
    return ContestationBatch(reasoning="[SAMPLE_MODE] no contestation gating.")


async def judge(
    *,
    items: list[tuple[str, str, list[str]]],
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    triggered_by: str = "hypothesis_loop",
    parent_call_id: int | None = None,
) -> tuple[ContestationBatch, int | None]:
    user_prompt = "\n".join(
        [
            f"Classify these {len(items)} contested entity pairs as genuine "
            "conflict vs synonym/multi-facet sprawl.",
            "",
            _format_items(items),
        ]
    )

    def output_summary(b: ContestationBatch) -> str:
        if not b.verdicts:
            return "0 judged (fallback)"
        genuine = sum(1 for v in b.verdicts if v.genuine)
        return f"{genuine}/{len(b.verdicts)} genuine conflicts"

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
        target_agent="Explorer",
        input_summary=f"{len(items)} contested pairs",
        output_summary_fn=output_summary,
        confidence=0.7,
        parent_call_id=parent_call_id,
        model_preference=MODEL_PREFERENCE,
    )
    return result, call.agent_call_id if call else None
