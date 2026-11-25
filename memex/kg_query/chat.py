"""Q&A orchestration: Retriever → Answerer ↔ GroundingChecker, bounded.

Single-shot (no conversation history): the agentic richness lives inside one
turn. Emits BOTH the JSONL agent trace (``TraceWriter``) and the rich query
trace (``QueryTraceStore`` — call graph + inner-thinking trajectory). The
GroundingChecker → Answerer revision is a real, bounded loop that fires whenever
grounding finds an unsupported claim.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.common.trace import TraceWriter
from memex.kg_intelligence.agents.base import AgentCallSinks
from memex.kg_query.agents import answerer, grounding_checker, retriever
from memex.kg_query.agents.schemas import AnswerResult, GroundingVerdict, RetrieverResult
from memex.kg_query.deps import QueryDeps
from memex.kg_query.grounding import GroundedFact, ground_refs
from memex.observability.query_store import QueryTraceStore


@dataclass
class ChatResult:
    question: str
    answer: AnswerResult
    bundle: list[GroundedFact]
    retriever: RetrieverResult
    verdict: GroundingVerdict | None
    revised: bool
    trace_path: str
    query_trace_db: str
    run_db_id: int
    sample_mode: bool = False
    diagnostics: dict = field(default_factory=dict)


async def run_chat(
    *,
    question: str,
    kg_root: Path,
    settings: Settings,
    llm_model: Model | None = None,
    request_id: str | None = None,
    max_revisions: int = 1,
    retriever_request_limit: int = 8,
    log_dir: Path | None = None,
) -> ChatResult:
    kg_id = kg_root.name
    request_id = request_id or f"kg_chat_{kg_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    deps = QueryDeps.open(kg_root, settings)
    query_store = QueryTraceStore.open(kg_root)
    trace = TraceWriter(request_id=request_id, log_dir=log_dir or kg_root)
    run = query_store.start_run(
        # Unique per execution — the query trace db is an append-only log of runs,
        # so re-running the same request_id just adds a
        # new run instead of colliding on the UNIQUE run_id.
        run_id=f"{request_id}-{uuid.uuid4().hex[:8]}", kind="chat", kg_id=kg_id,
        question=question,
        config={"max_revisions": max_revisions}, sample_mode=settings.sample_mode,
    )
    sinks = AgentCallSinks(trace=trace, store=query_store, round_=run)

    try:
        # 1 — Retriever gathers evidence (tool-using; ReAct trajectory captured).
        retriever_result, retriever_call_id = await retriever.retrieve(
            question=question, deps=deps, settings=settings, sinks=sinks,
            llm_model=llm_model, request_limit=retriever_request_limit,
        )
        bundle = ground_refs(deps, retriever_result.selected)

        # 2 — Answerer drafts a grounded answer.
        answer_result, answer_call_id = await answerer.answer(
            question=question, facts=bundle, settings=settings, sinks=sinks,
            llm_model=llm_model, parent_call_id=retriever_call_id,
        )

        # 3 — GroundingChecker verifies; revise on any unsupported claim (bounded).
        verdict: GroundingVerdict | None = None
        revised = False
        revisions = 0
        while revisions < max_revisions:
            verdict, _ = await grounding_checker.check(
                question=question, answer=answer_result, facts=bundle,
                settings=settings, sinks=sinks, llm_model=llm_model,
                parent_call_id=answer_call_id,
            )
            if verdict.all_supported():
                break
            # record the revision handoff back to the Answerer
            trace.record(
                agent_name=grounding_checker.AGENT_NAME,
                action="request_revision",
                input_summary=f"{len(verdict.unsupported_indices())} unsupported claim(s)",
                output_summary=verdict.revision_request or "tighten unsupported claims",
                target_agent=answerer.AGENT_NAME,
                confidence=0.7,
                status="needs_revision",
            )
            trace.record(
                agent_name=answerer.AGENT_NAME,
                action="accept_revision",
                input_summary=verdict.revision_request[:200],
                output_summary="revising answer to drop unsupported claims",
                target_agent=grounding_checker.AGENT_NAME,
                confidence=0.75,
                retry_count=1,
                status="retry",
            )
            answer_result, answer_call_id = await answerer.answer(
                question=question, facts=bundle, settings=settings, sinks=sinks,
                llm_model=llm_model, revision_request=verdict.revision_request,
                prior_answer=answer_result, parent_call_id=retriever_call_id,
            )
            revised = True
            revisions += 1

        # final approval marker
        trace.record(
            agent_name=grounding_checker.AGENT_NAME,
            action="finalize",
            input_summary=f"{len(answer_result.claims)} claim(s)",
            output_summary="answer accepted" if (verdict and verdict.all_supported()) else "answer returned with caveats",
            target_agent=None,
            confidence=0.85,
            status="success",
        )

        query_store.finish_run(
            run, status="success",
            summary=(answer_result.answer or "")[:500],
        )
        return ChatResult(
            question=question,
            answer=answer_result,
            bundle=bundle,
            retriever=retriever_result,
            verdict=verdict,
            revised=revised,
            trace_path=trace.relative_path,
            query_trace_db=str(query_store.db_path),
            run_db_id=run.run_db_id,
            sample_mode=settings.sample_mode,
            diagnostics={
                "facts_retrieved": len(bundle),
                "refs_selected": len(retriever_result.selected),
            },
        )
    finally:
        deps.close()
        query_store.close()


def render_markdown(result: ChatResult) -> str:
    """Render a ChatResult as a structured, readable Markdown report."""

    a = result.answer
    lines = [f"# Q: {result.question}", ""]
    if result.sample_mode:
        lines += ["> **SAMPLE_MODE** — deterministic stubs; retrieval is offline.", ""]
    lines += [a.answer or "_(no answer produced)_", ""]

    if result.bundle:
        lines += ["## Evidence", "", "| # | Fact | Source | Corrob. | Conf. |", "|---|---|---|---|---|"]
        for i, f in enumerate(result.bundle, start=1):
            cite = f.citation
            src = cite.title if cite else (f.source_doc_id or "—")
            if cite and cite.url:
                src = f"[{src}]({cite.url})"
            conf = f"{f.confidence:.2f}" if f.confidence is not None else "—"
            fact = f.canonical.replace("|", "\\|")
            lines.append(f"| {i} | {fact} | {src} | {f.corroboration} | {conf} |")
        lines.append("")

    if a.gaps:
        lines += ["## Not in corpus", ""]
        lines += [f"- {g}" for g in a.gaps]
        lines.append("")

    if result.verdict is not None:
        unsupported = result.verdict.unsupported_indices()
        status = "all claims grounded" if not unsupported else f"{len(unsupported)} claim(s) flagged"
        lines += [
            "## Grounding check",
            "",
            f"- {status}{' (revised)' if result.revised else ''}",
            f"- Retriever notes: {result.retriever.notes or '—'}",
            "",
        ]

    lines += [
        "---",
        f"_Trace: `{result.trace_path}` · query-trace DB: `{result.query_trace_db}` "
        f"(run #{result.run_db_id})_",
    ]
    return "\n".join(lines)
