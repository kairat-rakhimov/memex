"""Hypothesis-generation orchestration: a focused, agentic sweep.

Deterministic seeds (one lane when ``seed_type`` is given, else all) bound the
fan-out → ContestationJudge gate → type-neutral ``explore_worthiness`` ranks the
seeds → the top-K most worthy go to the **Explorer** (one tool-using ReAct
investigation each: reads source prose, traces the graph, forms a grounded
hypothesis or bails) → per-candidate VerifyEvidence (re-reads L3 source prose) →
batched NoveltyJudge → novelty × plausibility × support × graph-dependence ranking
→ curated top-N report (ReportWriter ↔ FinalReviewer) + a deterministic appendix.
Emits the JSONL agent trace + the rich query trace (call graph + inner-thinking).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.common.trace import TraceWriter
from memex.kg_intelligence.agents.base import AgentCallSinks
from memex.kg_intelligence.embeddings import embed_texts
from memex.kg_query import retrieval, scoring
from memex.kg_query import seeds as seeds_mod
from memex.kg_query.agents import (
    contestation_judge,
    explorer,
    final_reviewer,
    novelty_judge,
    report_writer,
    verify_evidence,
)
from memex.kg_query.agents.schemas import (
    HypothesisCandidate,
    NoveltyJudgment,
    ReportResult,
    VerifyVerdict,
)
from memex.kg_query.deps import QueryDeps
from memex.kg_query.grounding import GroundedFact, ground_refs
from memex.kg_query.scoring import ScoreBreakdown
from memex.kg_query.seeds import Seed
from memex.observability.query_store import QueryTraceStore

# How many nearest existing facts the NoveltyJudge sees per candidate — the
# comparison set it decides "is this claim already asserted" against.
NOVELTY_NEIGHBORS = 8


@dataclass
class RankedHypothesis:
    candidate: HypothesisCandidate
    seed: Seed
    all_legs: list[GroundedFact]
    surviving_legs: list[GroundedFact]
    verdict: VerifyVerdict
    novelty_sim: float
    scores: ScoreBreakdown


@dataclass
class _Survivor:
    """A candidate that passed the restatement pre-filter + grounding verify,
    awaiting the batched novelty judgment before it can be scored."""

    candidate: HypothesisCandidate
    seed: Seed
    all_legs: list[GroundedFact]
    surviving_legs: list[GroundedFact]
    verdict: VerifyVerdict
    sim: float
    vec: np.ndarray


@dataclass
class HypothesisRunResult:
    base_query: str
    ranked: list[RankedHypothesis]
    report_markdown: str
    trace_path: str
    query_trace_db: str
    run_db_id: int
    sample_mode: bool = False
    diagnostics: dict = field(default_factory=dict)


def _base_query(deps: QueryDeps, override: str | None) -> str:
    if override:
        return override
    row = deps.conn.execute(
        "SELECT base_query FROM builds ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    return (row["base_query"] if row else None) or "the corpus topic"


def _plausibility(candidate: HypothesisCandidate, verdict: VerifyVerdict) -> float:
    return 0.5 * candidate.plausibility + 0.5 * verdict.confidence


async def _gate_contestation(
    seeds: list[Seed],
    deps: QueryDeps,
    *,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None,
) -> None:
    """Classify each relation_contested seed as a genuine conflict vs predicate
    sprawl (one batched call), tagging ``seed.genuine_conflict`` in place. Unjudged
    seeds (SAMPLE_MODE / error → empty batch) stay ``None`` = treated as genuine."""

    contested = [s for s in seeds if s.lane == "relation_contested"]
    if not contested:
        return
    g = deps.graph()

    def _name(eid: int) -> str:
        return g.nodes[eid].get("name", str(eid)) if eid in g else str(eid)

    items: list[tuple[str, str, list[str]]] = []
    for s in contested:
        subj = _name(s.entity_ids[0]) if s.entity_ids else "?"
        obj = _name(s.entity_ids[1]) if len(s.entity_ids) > 1 else "?"
        items.append((subj, obj, list(s.payload.get("predicates", []))))
    batch, _ = await contestation_judge.judge(
        items=items, settings=settings, sinks=sinks, llm_model=llm_model
    )
    by_idx = {v.index: v for v in batch.verdicts}
    for i, s in enumerate(contested):
        v = by_idx.get(i)
        if v is not None:
            s.genuine_conflict = v.genuine


async def _judge_novelty(
    survivors: list[_Survivor],
    deps: QueryDeps,
    *,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None,
) -> dict[int, NoveltyJudgment]:
    """One batched assertional-novelty pass: each survivor + its nearest existing
    facts. Returns ``index → judgment``; missing indices fall back to embedding
    novelty in the caller (SAMPLE_MODE or judge error yields an empty batch)."""

    if not survivors:
        return {}
    items: list[tuple[str, list[str]]] = []
    for sv in survivors:
        neighbors = (
            retrieval.semantic_facts(deps, sv.vec, k=NOVELTY_NEIGHBORS)
            if sv.vec.size
            else []
        )
        items.append((sv.candidate.statement, [h.text() for h in neighbors]))
    batch, _ = await novelty_judge.judge(
        items=items, settings=settings, sinks=sinks, llm_model=llm_model
    )
    return {j.index: j for j in batch.judgments}


async def run_hypothesize(
    *,
    kg_root: Path,
    settings: Settings,
    base_query: str | None = None,
    max_seeds: int = 24,
    top_n: int = 8,
    min_composite: float = scoring.DEFAULT_MIN_COMPOSITE,
    novelty_max: float = scoring.DEFAULT_NOVELTY_MAX,
    lanes: list[str] | None = None,
    explore_cap: int = 8,
    verify_cap: int | None = None,
    request_id: str | None = None,
    llm_model: Model | None = None,
    max_report_revisions: int = 1,
    log_dir: Path | None = None,
) -> HypothesisRunResult:
    kg_id = kg_root.name
    request_id = request_id or f"kg_hypothesize_{kg_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    deps = QueryDeps.open(kg_root, settings)
    query_store = QueryTraceStore.open(kg_root)
    trace = TraceWriter(request_id=request_id, log_dir=log_dir or kg_root)
    topic = _base_query(deps, base_query)
    run = query_store.start_run(
        # Unique per execution (append-only run log) — see run_chat for why.
        run_id=f"{request_id}-{uuid.uuid4().hex[:8]}", kind="hypothesize", kg_id=kg_id,
        question=topic,
        config={"max_seeds": max_seeds, "top_n": top_n, "novelty_max": novelty_max},
        sample_mode=settings.sample_mode,
    )
    sinks = AgentCallSinks(trace=trace, store=query_store, round_=run)

    try:
        # 1 — deterministic seeds + grounded legs
        seeds = seeds_mod.generate_seeds(deps, max_seeds=max_seeds, lanes=lanes)
        # Gate the relation_contested lane: it fires on predicate COUNT, so an LLM
        # judge separates genuine conflicts (the lane's value) from synonym /
        # multi-facet sprawl, which loses the contested structure bonus at scoring.
        await _gate_contestation(
            seeds, deps, settings=settings, sinks=sinks, llm_model=llm_model
        )
        seeds_with_legs = [(s, ground_refs(deps, [f"{k}:{i}" for k, i in s.leg_fact_ids])) for s in seeds]
        trace.record(
            agent_name="SeedGenerator", action="seed_graph",
            input_summary=f"{len(seeds)} seeds across lanes",
            output_summary=", ".join(sorted({s.lane for s in seeds})),
            target_agent=explorer.AGENT_NAME, confidence=1.0, status="success",
        )

        # 2 — pick the most explore-worthy seeds (type-neutral: genuine × cross-source
        # richness × topical relevance), then INVESTIGATE each deeply with the agentic
        # Explorer — it reads the source prose + traces the graph and forms a grounded
        # hypothesis, or bails.
        topic_vecs = await embed_texts([topic], settings) if topic else []
        topic_vec = topic_vecs[0] if topic_vecs else np.array([])
        seeds_with_legs.sort(
            key=lambda sl: scoring.explore_worthiness(deps, sl[0], sl[1], topic_vec=topic_vec),
            reverse=True,
        )
        pairs: list[tuple[HypothesisCandidate, Seed, list[GroundedFact], int | None]] = []
        bailed = 0
        for seed, legs in seeds_with_legs[:explore_cap]:
            explored, explore_call_id = await explorer.explore(
                topic=topic, seed=seed, legs=legs, deps=deps, settings=settings, sinks=sinks,
                llm_model=llm_model,
            )
            if not explored.worthwhile or not explored.statement.strip():
                # The Explorer investigated this lead and abandoned it; record why
                # rather than skipping silently.
                bailed += 1
                trace.record(
                    agent_name=explorer.AGENT_NAME, action="abandon_lead",
                    input_summary=f"[{seed.lane}] {seed.title}"[:160],
                    output_summary="investigated; not a worthwhile lead — abandoned",
                    target_agent="SeedGenerator", confidence=0.7, status="needs_revision",
                )
                continue
            cand = HypothesisCandidate(
                seed_index=0, worthwhile=True, statement=explored.statement,
                mechanism=explored.mechanism, leg_refs=explored.leg_refs,
                hypothesis_class=seed.lane, plausibility=explored.plausibility,
                reasoning=explored.reasoning,
            )
            cand_legs = ground_refs(deps, explored.leg_refs) or legs
            pairs.append((cand, seed, cand_legs, explore_call_id))

        # 3 — embedding restatement pre-filter + per-candidate verify; collect
        # survivors for one batched assertional-novelty judgment.
        statements = [c.statement for c, _s, _l, _g in pairs]
        vecs = await embed_texts(statements, settings) if statements else []
        cap = verify_cap if verify_cap is not None else max_seeds
        rejected_novelty = 0
        verify_rejected = 0
        survivors: list[_Survivor] = []
        for idx, (cand, seed, legs, explore_call_id) in enumerate(pairs[:cap]):
            vec = vecs[idx] if idx < len(vecs) else np.array([])
            sim = scoring.restatement_similarity(deps, vec) if vec.size else 0.0
            if scoring.is_restatement(sim, novelty_max=novelty_max):
                rejected_novelty += 1
                continue
            verdict, _ = await verify_evidence.verify(
                candidate=cand, legs=legs, settings=settings, sinks=sinks,
                llm_model=llm_model, parent_call_id=explore_call_id,
            )
            if not verdict.holds:
                # the verifier rejected this hypothesis; record it
                verify_rejected += 1
                trace.record(
                    agent_name=verify_evidence.AGENT_NAME, action="reject_hypothesis",
                    input_summary=cand.statement[:160],
                    output_summary=f"legs unsupported; conf={verdict.confidence:.2f}",
                    target_agent=explorer.AGENT_NAME, confidence=verdict.confidence,
                    status="needs_revision",
                )
                continue
            surviving = [
                f for f in legs
                if any(lv.ref == f"{f.kind}:{f.fact_id}" and lv.supported for lv in verdict.leg_verdicts)
            ] or legs
            survivors.append(_Survivor(cand, seed, legs, surviving, verdict, sim, vec))

        # one batched assertional-novelty pass over the survivors: cosine says
        # "about the same things"; the judge says "is this CLAIM already
        # asserted/entailed". Falls back to embedding novelty per candidate.
        novelty_by_index = await _judge_novelty(
            survivors, deps, settings=settings, sinks=sinks, llm_model=llm_model
        )
        ranked: list[RankedHypothesis] = []
        for i, sv in enumerate(survivors):
            j = novelty_by_index.get(i)
            novelty = j.novelty if j is not None else scoring.novelty_score(sv.sim)
            scores = scoring.composite_score(
                novelty=novelty,
                plausibility=_plausibility(sv.candidate, sv.verdict),
                support=scoring.support_score(sv.surviving_legs),
                structure=scoring.structure_score(sv.surviving_legs),
            )
            ranked.append(
                RankedHypothesis(
                    candidate=sv.candidate, seed=sv.seed, all_legs=sv.all_legs,
                    surviving_legs=sv.surviving_legs, verdict=sv.verdict,
                    novelty_sim=sv.sim, scores=scores,
                )
            )

        ranked.sort(key=lambda r: r.scores.composite, reverse=True)
        # Headline only the high-value leads: cap at top_n AND require the composite
        # floor, so a weak sweep surfaces fewer (or none) rather than padding to top_n.
        # The rest fall to the appendix.
        top = [r for r in ranked[:top_n] if r.scores.composite >= min_composite]

        # 4 — report ↔ final review (bounded revision). Always runs: when nothing
        # survives, the report is a useful exploration summary (what was tried +
        # filtered).
        block = _hypotheses_block(top) if top else _exploration_block(
            len(seeds), len(pairs), rejected_novelty, verify_rejected
        )
        coverage = _coverage_summary(top) if top else "(no hypotheses survived verification)"
        report, report_call_id = await report_writer.write(
            base_query=topic, hypotheses_block=block, settings=settings, sinks=sinks,
            llm_model=llm_model,
        )
        report_md = _render_body(report, top)
        revisions = 0
        while revisions < max_report_revisions:
            fr, _ = await final_reviewer.review(
                base_query=topic, report_markdown=report_md, coverage_summary=coverage,
                settings=settings, sinks=sinks, llm_model=llm_model,
                parent_call_id=report_call_id,
            )
            issues = list(fr.issues)
            # The ReportWriter's first draft sees only the survivors. When the
            # verifier filtered candidates or the Explorer abandoned leads, ask for
            # one revision so the report says what was investigated and dropped.
            if fr.approved and not issues and verify_rejected == 0 and bailed == 0:
                break
            if not issues:
                issues = [
                    f"{verify_rejected} candidate hypotheses were filtered by "
                    f"verification and {bailed} leads abandoned during exploration — "
                    "ensure the report frames the survivors with appropriate caveats "
                    "and notes what was investigated and dropped."
                ]
            trace.record(
                agent_name=final_reviewer.AGENT_NAME, action="request_revision",
                input_summary=f"{len(issues)} issue(s); {verify_rejected} filtered",
                output_summary="; ".join(issues)[:200],
                target_agent=report_writer.AGENT_NAME, confidence=0.7,
                status="needs_revision",
            )
            trace.record(
                agent_name=report_writer.AGENT_NAME, action="accept_revision",
                input_summary="; ".join(issues)[:200],
                output_summary="revising report", target_agent=final_reviewer.AGENT_NAME,
                confidence=0.75, retry_count=1, status="retry",
            )
            report, report_call_id = await report_writer.write(
                base_query=topic, hypotheses_block=block, settings=settings, sinks=sinks,
                llm_model=llm_model, revision_request="; ".join(issues),
                prior_report=report_md,
            )
            report_md = _render_body(report, top)
            revisions += 1

        final_md = _assemble_document(topic, report_md, ranked, len(top), request_id, str(query_store.db_path), run.run_db_id)
        query_store.finish_run(run, status="success", summary=f"{len(top)} hypotheses, {len(ranked)} ranked")
        return HypothesisRunResult(
            base_query=topic,
            ranked=ranked,
            report_markdown=final_md,
            trace_path=trace.relative_path,
            query_trace_db=str(query_store.db_path),
            run_db_id=run.run_db_id,
            sample_mode=settings.sample_mode,
            diagnostics={
                "seeds": len(seeds),
                "explored": min(explore_cap, len(seeds)),
                "bailed": bailed,
                "candidates": len(pairs),
                "rejected_novelty": rejected_novelty,
                "rejected_verify": verify_rejected,
                "ranked": len(ranked),
                "top_n": len(top),
            },
        )
    finally:
        deps.close()
        query_store.close()


# ── formatting ────────────────────────────────────────────────────────────────


def _evidence_lines(legs: list[GroundedFact]) -> list[str]:
    out = []
    for f in legs:
        cite = f.citation
        src = cite.title if cite else (f.source_doc_id or "—")
        url = f" ({cite.url})" if cite and cite.url else ""
        out.append(f"    - [{f.kind}:{f.fact_id}] {f.canonical} — {src}{url}; corrob {f.corroboration}")
    return out


def _hypotheses_block(ranked: list[RankedHypothesis]) -> str:
    blocks = []
    for i, r in enumerate(ranked, start=1):
        s = r.scores
        blocks.append(
            "\n".join(
                [
                    f"H{i} [{r.candidate.hypothesis_class or r.seed.lane}] "
                    f"(novelty {s.novelty:.2f}, plausibility {s.plausibility:.2f}, "
                    f"support {s.support:.2f}; verify holds={r.verdict.holds} conf={r.verdict.confidence:.2f})",
                    f"  statement: {r.candidate.statement}",
                    f"  mechanism: {r.candidate.mechanism}",
                    "  evidence:",
                    *_evidence_lines(r.surviving_legs),
                ]
            )
        )
    return "\n\n".join(blocks)


def _grounded_lines(legs: list[GroundedFact]) -> list[str]:
    """The deterministic grounding block — the surviving legs with their source
    attribution + citation, rendered from data (not the LLM). Each fact's verbatim
    source sentence is shown as a quote so a reader can verify it against the cited
    document — the human-readable text behind the otherwise opaque fact id."""

    if not legs:
        return ["- _(no surviving grounded evidence)_"]
    out = []
    for f in legs:
        cite = f.citation
        src = cite.title if cite else (f.source_doc_id or "—")
        url = f" ({cite.url})" if cite and cite.url else ""
        out.append(
            f"- *{f.canonical}* — {src}{url} · corroborated by {f.corroboration} "
            f"source(s) · `{f.kind}:{f.fact_id}`"
        )
        if f.source_quote:
            out.append(f"  > “{f.source_quote}”")
    return out


def _fallback_title(r: RankedHypothesis) -> str:
    words = r.candidate.statement.split()
    title = " ".join(words[:12])
    return title + ("…" if len(words) > 12 else "") or (r.candidate.hypothesis_class or r.seed.lane)


def _render_body(result: ReportResult, top: list[RankedHypothesis]) -> str:
    """Assemble the per-hypothesis body: the LLM's prose (title / hypothesis /
    rationale / how-to-test, by rank order) enriched with the deterministic score
    (in the heading) and the grounded-evidence block (from the legs). Falls back to
    the candidate's own statement/mechanism when the LLM under-delivers a section,
    so every hypothesis renders complete and grounded."""

    writes = result.hypotheses
    lines: list[str] = []
    if result.intro:
        lines += [result.intro, ""]
    for i, r in enumerate(top, start=1):
        w = writes[i - 1] if i - 1 < len(writes) else None
        s = r.scores
        cls = r.candidate.hypothesis_class or r.seed.lane
        title = (w.title if w else "") or _fallback_title(r)
        hypo = (w.hypothesis if w else "") or r.candidate.statement
        rationale = (w.rationale if w else "") or r.candidate.mechanism
        lines += [
            f"## H{i} · {title} — score {s.composite:.2f}",
            "",
            f"*{cls} · novelty {s.novelty:.2f} · plausibility {s.plausibility:.2f} · "
            f"support {s.support:.2f} · structure {s.structure:.2f} · verified "
            f"(confidence {r.verdict.confidence:.2f})*",
            "",
            f"**Hypothesis.** {hypo}",
            "",
            f"**Rationale.** {rationale}",
            "",
            "**Grounded in.**",
            *_grounded_lines(r.surviving_legs),
            "",
        ]
        how = w.how_to_test if w else ""
        if how:
            lines += [f"**How to test or refute.** {how}", ""]
    if result.conclusion:
        lines += [result.conclusion, ""]
    return "\n".join(lines).rstrip() + "\n"


def _exploration_block(
    n_seeds: int, n_candidates: int, rejected_novelty: int, rejected_verify: int
) -> str:
    """Stand-in 'hypotheses' block when nothing survived verification — the
    ReportWriter turns this into an honest exploration summary."""

    return (
        "No hypotheses survived the novelty + grounding filters this sweep. Write a "
        "brief, honest exploration summary instead: what was searched and why nothing "
        "held up. Do NOT invent hypotheses or evidence.\n\n"
        f"- structural seeds explored: {n_seeds}\n"
        f"- candidate hypotheses articulated: {n_candidates}\n"
        f"- rejected as restatements (novelty floor): {rejected_novelty}\n"
        f"- rejected by source-grounding verification: {rejected_verify}\n"
    )


def _coverage_summary(ranked: list[RankedHypothesis]) -> str:
    from collections import Counter

    counts = Counter(r.candidate.hypothesis_class or r.seed.lane for r in ranked)
    lines = [f"- {cls}: {n}" for cls, n in counts.items()]
    return "\n".join(lines)


def _assemble_document(
    topic: str, report_md: str, ranked: list[RankedHypothesis], top_n: int,
    request_id: str, query_db: str, run_db_id: int,
) -> str:
    lines = [f"# Hypotheses — {topic}", ""]
    lines.append(report_md or "_(report unavailable)_")
    if ranked:
        appendix = ranked[top_n:]
        if appendix:
            lines += ["", "## Appendix — additional candidates", ""]
            for r in appendix:
                s = r.scores
                lines.append(
                    f"- **[{r.candidate.hypothesis_class or r.seed.lane}]** "
                    f"{r.candidate.statement} "
                    f"_(score {s.composite:.2f} · novelty {s.novelty:.2f} · "
                    f"plausibility {s.plausibility:.2f} · support {s.support:.2f} · "
                    f"structure {s.structure:.2f})_"
                )
    lines += [
        "",
        "---",
        f"_{len(ranked)} hypotheses ranked · trace `{request_id}` · "
        f"query-trace DB `{query_db}` (run #{run_db_id})_",
    ]
    return "\n".join(lines)
