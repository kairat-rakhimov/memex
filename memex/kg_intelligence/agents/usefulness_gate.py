"""UsefulnessGate: per-doc whole-doc usefulness gate + summary/reflection.

Post-fetch, pre-extraction, gpt-5.1, **two passes over the WHOLE doc** sharing
one prompt cache:

- Pass 1 (``assess_usefulness``): a cheap keep/skip verdict. A ``skip``
  short-circuits extraction + the Mapper on the doc — the round's main cost
  lever.
- Pass 2 (``summarize_reflect``): runs ONLY for keepers; produces the durable
  summary + the corpus-state reflection (entities_seen / subtopics / borders).

Both passes use the SAME agent (identical system prompt + ``UsefulnessResult``
output type) and put the whole doc FIRST in the user prompt, so pass 2's
``[system + whole_doc]`` prefix matches pass 1's and hits its prompt cache —
only the trailing TASK directive differs. (The cross-pass cache HIT itself is
measured on a live build, not in tests.)
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
from memex.kg_intelligence.agents.extractor import SESSION_DOC_BUDGET
from memex.kg_intelligence.agents.schemas import UsefulnessResult
from memex.kg_intelligence.state import CorpusState

AGENT_NAME = "UsefulnessGate"
MODEL_PREFERENCE: ModelPreference = "reasoning"
KEEP_THRESHOLD = 0.4

_USEFULNESS_TASK = (
    "Judge USEFULNESS ONLY. Set `usefulness` (0-1), `decision` "
    "(`keep` if ≥ 0.4 else `skip`), and a one-clause `reason`. Leave `summary`, "
    "`entities_seen`, `subtopics_covered`, `subtopics_missing`, and `borders` "
    "EMPTY — a second pass fills those only if this doc is kept."
)
_SUMMARY_TASK = (
    "This doc was KEPT. Now produce the durable artifacts: a faithful `summary` "
    "(a few sentences capturing what the doc establishes — its entities, "
    "mechanisms, findings), plus `entities_seen`, `subtopics_covered`, "
    "`subtopics_missing`, and `borders` grounded in the doc. Echo the keep "
    "`decision` and your `usefulness`; the `reason` may be brief."
)

_SYSTEM_PROMPT = """\
You are UsefulnessGate. After each document is fetched you read the WHOLE doc
and decide whether it is worth the expensive downstream work (extraction +
knowledge-graph mapping), then — for kept docs — summarize it and reflect on
what the corpus now covers. You work in two passes; each prompt ends with a
TASK line telling you which pass you are on. Only ever surface concepts and
vocabulary that appear in the provided document — never invent or recall from
outside knowledge.

Usefulness — judge whether this doc would meaningfully contribute entities +
relations to a knowledge graph about the base query, NOT whether its title
literally matches the query. A KG benefits from adjacent fields that share
mechanisms, materials, or processes.

KEEP docs covering:
  * The query topic directly.
  * The same underlying mechanisms / materials / processes used in adjacent
    applications (e.g. for "green hydrogen catalysts", KEEP PEM fuel-cell papers
    — shared Pt catalyst chemistry; KEEP hydrogenation papers — HER is
    hydrogenation chemistry).
  * Economic / policy / deployment context for a technology query.
  * General overviews of adjacent fields that introduce reusable terminology.

SKIP docs covering:
  * Pure off-topic material that merely shares a keyword (PCR biology that uses
    the word "water"; an ML method coincidentally named "PEM").
  * Earnings transcripts / corporate decks / generic financial commentary that
    mention the topic only as passing boilerplate.
  * Unrelated domains whose only connection is coincidental terminology.

Calibration: 0.7-1.0 = directly on-topic; 0.4-0.7 = adjacent but KG-useful;
0.1-0.4 = tangential / coincidental keyword overlap; 0.0-0.1 = unrelated.

Reflection fields (pass 2 only):
- entities_seen: noun phrases for materials, devices, methods, organizations the
  doc surfaces (mostly fresh; omit re-surfaces of already-seen entities unless
  central).
- subtopics_covered: short labels for what this doc actually covers.
- subtopics_missing: related subtopics the doc mentions but does not cover with
  depth.
- borders: topics where the doc signals saturation / a dead end worth not
  probing further.
"""

gate_agent: Agent[None, UsefulnessResult] = Agent(
    model=None,
    output_type=UsefulnessResult,
    system_prompt=_SYSTEM_PROMPT,
    retries=2,
    defer_model_check=True,
)


async def assess_doc(
    *,
    doc_id: str,
    source: str,
    title: str,
    whole_doc: str,
    corpus_state: CorpusState,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None = None,
    triggered_by: str = "fetch",
) -> UsefulnessResult:
    """Two-pass whole-doc gate for ONE doc. Pass 1 returns the keep/skip
    verdict; on ``keep`` pass 2 adds the summary + reflection. On ``skip`` the
    second pass is never run (the summary stays empty)."""

    verdict = await _run_pass(
        action="assess_usefulness",
        task=_USEFULNESS_TASK,
        doc_id=doc_id,
        title=title,
        whole_doc=whole_doc,
        corpus_state=corpus_state,
        sample=lambda: _sample_verdict(),
        settings=settings,
        sinks=sinks,
        llm_model=llm_model,
        triggered_by=triggered_by,
    )
    if verdict.decision == "skip":
        return verdict

    reflection = await _run_pass(
        action="summarize_reflect",
        task=_SUMMARY_TASK,
        doc_id=doc_id,
        title=title,
        whole_doc=whole_doc,
        corpus_state=corpus_state,
        sample=lambda: _sample_reflection(title),
        settings=settings,
        sinks=sinks,
        llm_model=llm_model,
        triggered_by="usefulness_keep",
    )
    return UsefulnessResult(
        usefulness=verdict.usefulness,
        decision="keep",
        reason=verdict.reason,
        summary=reflection.summary,
        entities_seen=reflection.entities_seen,
        subtopics_covered=reflection.subtopics_covered,
        subtopics_missing=reflection.subtopics_missing,
        borders=reflection.borders,
    )


async def _run_pass(
    *,
    action: str,
    task: str,
    doc_id: str,
    title: str,
    whole_doc: str,
    corpus_state: CorpusState,
    sample,  # type: ignore[no-untyped-def]
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None,
    triggered_by: str,
) -> UsefulnessResult:
    user_prompt = _doc_prompt(whole_doc, title, corpus_state, task)
    result, _call, _run = await execute_agent_call(
        agent=gate_agent,
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        sample_factory=sample,
        log_prefix=f"{AGENT_NAME}:{action}",
        settings=settings,
        llm_model=llm_model,
        sinks=sinks,
        agent_name=AGENT_NAME,
        action=action,
        triggered_by=triggered_by,
        target_agent="Extractor",
        input_summary=f"doc={doc_id}, chars={len(whole_doc)}",
        output_summary_fn=lambda r: (
            f"decision={r.decision}, usefulness={r.usefulness:.2f}"
            if action == "assess_usefulness"
            else f"summary_chars={len(r.summary)}, entities={len(r.entities_seen)}"
        ),
        confidence=0.75,
        model_preference=MODEL_PREFERENCE,
    )
    return result


def _doc_prompt(
    whole_doc: str, title: str, corpus_state: CorpusState, task: str
) -> str:
    """Whole doc FIRST (so the cached prefix is shared across passes), then the
    round context, then the differing TASK directive."""

    body = (whole_doc or "").strip()[:SESSION_DOC_BUDGET]
    parts = [
        f"DOCUMENT (title: {title}):\n{body}",
        f"Base query (judge usefulness against this): {corpus_state.base_query!r}",
    ]
    if corpus_state.entities_seen:
        parts.append(
            "Entities already seen (omit re-surfaces unless central):\n- "
            + "\n- ".join(corpus_state.entities_seen[:30])
        )
    if corpus_state.subtopics_covered:
        parts.append(
            "Subtopics already covered:\n- "
            + "\n- ".join(corpus_state.subtopics_covered[:15])
        )
    parts.append(f"TASK: {task}")
    return "\n\n".join(parts)


def _sample_verdict() -> UsefulnessResult:
    """SAMPLE_MODE pass-1 fallback: keep conservatively (never blocks the loop)."""

    return UsefulnessResult(
        usefulness=0.5, decision="keep", reason="SAMPLE_MODE conservative default"
    )


def _sample_reflection(title: str) -> UsefulnessResult:
    """SAMPLE_MODE pass-2 fallback: a thin summary + entities from the title."""

    entities = sorted({tok for tok in title.split() if len(tok) > 4})[:8]
    return UsefulnessResult(
        usefulness=0.5,
        decision="keep",
        reason="SAMPLE_MODE",
        summary=f"SAMPLE_MODE summary of {title!r}." if title else "SAMPLE_MODE summary.",
        entities_seen=entities,
        subtopics_covered=[title] if title else [],
    )


__all__ = ["assess_doc", "gate_agent", "KEEP_THRESHOLD"]
