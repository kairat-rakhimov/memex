"""``POST /run`` task router — dispatches ``input.task`` to a memex workflow.

  - ``chat``        → grounded Q&A over a built graph (Retriever → Answerer ↔
                      GroundingChecker).
  - ``hypothesize`` → hypothesis sweep over a built graph (structural seeds →
                      Explorer → VerifyEvidence → NoveltyJudge → ReportWriter ↔
                      FinalReviewer).
  - ``build``       → a knowledge-graph build from ``input.query`` (all sources,
                      small default caps, wall-clock guard).

Graphs live at ``<MEMEX_KG_DIR>/<kg_id>/kg.db``; ``chat`` / ``hypothesize``
require ``input.kg_id``. Every task returns a ``RunOutcome`` whose ``output``
carries a human ``report_markdown`` alongside machine-readable fields.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memex.common.config import Settings
from memex.common.schemas import AgentSummary, RunRequest

_CHAT_AGENTS = [
    AgentSummary(name="Retriever", role="Tool-using KG retrieval — expands the question into graph/semantic/symbolic lookups."),
    AgentSummary(name="Answerer", role="Synthesises a grounded answer from the retrieved evidence bundle."),
    AgentSummary(name="GroundingChecker", role="Verifies each claim against its cited evidence; can force a revision."),
]
_HYPOTHESIS_AGENTS = [
    AgentSummary(name="SeedGenerator", role="Deterministic structural seeds across 5 lanes (bridge/gap/property/attribute-contested/relation-contested)."),
    AgentSummary(name="ContestationJudge", role="Keeps only relation_contested seeds whose predicates genuinely conflict."),
    AgentSummary(name="Explorer", role="Investigates each top seed on the live graph and forms a grounded hypothesis or bails."),
    AgentSummary(name="VerifyEvidence", role="Per-candidate adversarial verification of each supporting leg."),
    AgentSummary(name="NoveltyJudge", role="Scores whether each hypothesis says more than the corpus already states."),
    AgentSummary(name="ReportWriter", role="Curates the ranked report (novelty × plausibility × support)."),
    AgentSummary(name="FinalReviewer", role="Reviews the report and can request one revision."),
]


@dataclass
class RunOutcome:
    request_id: str
    output: dict[str, Any]
    agents_used: list[str]
    agents: list[AgentSummary]
    runtime_seconds: float
    trace_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _resolve_task(payload: dict[str, Any]) -> str:
    raw = str(payload.get("task") or "").strip().lower()
    if raw in {"build", "ingest", "construct"}:
        return "build"
    if raw in {"hypothesize", "hypothesis", "hypotheses", "discover"}:
        return "hypothesize"
    if raw in {"chat", "ask", "qa", "question"}:
        return "chat"
    # Inference fallback: an explicit question reads as chat; otherwise hypothesize.
    if payload.get("question"):
        return "chat"
    return "hypothesize"


def _text(payload: dict[str, Any], *keys: str) -> str | None:
    for k in keys:
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


async def run_workflow(*, run_request: RunRequest, settings: Settings) -> RunOutcome:
    request_id = run_request.request_id or str(uuid.uuid4())
    payload = run_request.input or {}
    task = _resolve_task(payload)
    if task == "chat":
        outcome = await _run_chat(request_id, payload, settings)
    elif task == "hypothesize":
        outcome = await _run_hypothesize(request_id, payload, settings)
    else:
        outcome = await _run_build(request_id, payload, settings)
    _write_report_md(outcome, settings)
    return outcome


def _write_report_md(outcome: RunOutcome, settings: Settings) -> None:
    """Drop the human-readable report next to the JSONL trace, so every /run leaves a
    readable `report_<request_id>.md` (not only the report_markdown buried in JSON)."""
    report = (outcome.output or {}).get("report_markdown")
    if not report:
        return
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    md_path = log_dir / f"report_{outcome.request_id}.md"
    md_path.write_text(report if report.endswith("\n") else report + "\n", encoding="utf-8")
    outcome.output["report_file"] = str(md_path)


# ── chat ─────────────────────────────────────────────────────────────────────


async def _run_chat(request_id: str, payload: dict[str, Any], settings: Settings) -> RunOutcome:
    from memex.kg_query.chat import render_markdown, run_chat

    kg_root = _require_kg(payload, settings)
    question = _text(payload, "question", "query", "prompt")
    if not question:
        raise ValueError("chat task requires input.question (or input.query).")
    started = time.monotonic()
    result = await run_chat(
        question=question,
        kg_root=kg_root,
        settings=settings,
        request_id=request_id,
        log_dir=settings.log_dir,
        max_revisions=int(payload.get("max_revisions", 1)),
    )
    report_md = render_markdown(result)
    n_cited = len({gf.fact_id for gf in result.bundle}) if result.bundle else 0
    summary = (
        f"Grounded answer to “{question}” over {kg_root.name}: "
        f"{n_cited} cited evidence facts"
        + (", revised after grounding check" if result.revised else "")
        + "."
    )
    return RunOutcome(
        request_id=request_id,
        output={
            "task": "chat",
            "summary": summary,
            "report_markdown": report_md,
            "question": question,
            "grounding_revised": result.revised,
            "evidence_facts": n_cited,
            "kg_id": kg_root.name,
        },
        agents_used=[a.name for a in _CHAT_AGENTS],
        agents=list(_CHAT_AGENTS),
        runtime_seconds=round(time.monotonic() - started, 3),
        trace_path=result.trace_path,
    )


# ── hypothesize ──────────────────────────────────────────────────────────────


async def _run_hypothesize(request_id: str, payload: dict[str, Any], settings: Settings) -> RunOutcome:
    from memex.kg_query.hypothesize import run_hypothesize

    kg_root = _require_kg(payload, settings)
    started = time.monotonic()
    # seed_type focuses the agentic Explorer on ONE structural lane (bridge / gap /
    # property / attribute_contested / relation_contested) so it investigates that
    # lens deeply; omit it to sweep all lanes. (`lane`/`lanes` accepted as aliases.)
    seed_type = _text(payload, "seed_type", "lane") or None
    lanes = [seed_type] if seed_type else payload.get("lanes")
    if isinstance(lanes, str):
        lanes = [lanes]
    result = await run_hypothesize(
        kg_root=kg_root,
        settings=settings,
        base_query=_text(payload, "query", "topic", "question"),
        max_seeds=int(payload.get("max_seeds", 24)),
        top_n=int(payload.get("top_n", 8)),
        lanes=lanes,
        explore_cap=int(payload.get("explore_cap", 8)),
        request_id=request_id,
        log_dir=settings.log_dir,
    )
    survivors = [h for h in result.ranked if h.verdict.holds]
    lens = f" (seed_type={seed_type})" if seed_type else ""
    summary = (
        f"Hypothesis exploration over {kg_root.name} for "
        f"“{result.base_query}”{lens}: {len(survivors)} grounded hypotheses "
        f"(of {len(result.ranked)} ranked candidates)."
    )
    return RunOutcome(
        request_id=request_id,
        output={
            "task": "hypothesize",
            "summary": summary,
            "report_markdown": result.report_markdown,
            "base_query": result.base_query,
            "seed_type": seed_type,
            "grounded_hypotheses": len(survivors),
            "ranked_candidates": len(result.ranked),
            "kg_id": kg_root.name,
            "diagnostics": result.diagnostics,
        },
        agents_used=[a.name for a in _HYPOTHESIS_AGENTS],
        agents=list(_HYPOTHESIS_AGENTS),
        runtime_seconds=round(time.monotonic() - started, 3),
        trace_path=result.trace_path,
    )


# ── build ──────────────────────────────────────────

_BUILD_AGENTS = [
    AgentSummary(name="RoundPlanner", role="Ranks topics for the round from the goal + KG gaps."),
    AgentSummary(name="QueryPlanner", role="Per-source: turns ranked topics into source-tailored search queries."),
    AgentSummary(name="Selector", role="Arbitrates the cross-source candidate pool by priority-weighted goal-fit."),
    AgentSummary(name="UsefulnessGate", role="Two-pass whole-doc relevance gate; drops off-topic fetches."),
    AgentSummary(name="Extractor", role="Per-chunk entity/relation/attribute extraction (whole-doc context)."),
    AgentSummary(name="Mapper", role="Canonicalises predicates + entity types per doc."),
    AgentSummary(name="AttributeMapper", role="Canonicalises attribute properties + parses values/units/conditions."),
    AgentSummary(name="Resolver", role="Tiered entity-alias resolution (merges duplicate entities)."),
    AgentSummary(name="KGReviewer", role="Finds structural gaps in the graph after the round."),
    AgentSummary(name="Strategist", role="Judges progress + seeds the next round's direction (auto-stop)."),
]


async def _run_build(request_id: str, payload: dict[str, Any], settings: Settings) -> RunOutcome:
    import asyncio
    import contextlib

    from memex.common.trace import TraceWriter
    from memex.kg_intelligence.pipeline import LoopConfig, run_pipeline
    from memex.kg_intelligence.sources.arxiv.adapter import ArxivAdapter
    from memex.kg_intelligence.sources.common.fetch_cache import FetchCache
    from memex.kg_intelligence.sources.gdelt.adapter import GdeltAdapter
    from memex.kg_intelligence.sources.manager import AdapterManager
    from memex.kg_intelligence.sources.wikipedia.adapter import WikipediaAdapter

    base_query = _text(payload, "query", "topic", "question")
    if not base_query:
        raise ValueError("build task requires input.query.")
    max_rounds = max(1, int(payload.get("max_rounds", 1)))
    docs_per_round = max(1, int(payload.get("max_docs_per_round", 3)))
    # Stop at a clean round boundary before the request timeout cuts the run off.
    deadline_s = float(payload.get("deadline_seconds", settings.run_timeout_seconds - 90))

    started = time.monotonic()
    kg_id = _text(payload, "kg_id") or f"build_{request_id[:8]}_{int(started)}"
    kg_root = settings.kg_dir / kg_id
    kg_root.mkdir(parents=True, exist_ok=True)
    trace = TraceWriter(request_id=request_id, log_dir=settings.log_dir)

    # Per-source fetch tuning from input.fetch, defaulting to fast-fail on the flaky
    # public APIs (arXiv/GDELT) — 1 attempt, tight timeout — so a transient
    # 429/SSL-timeout fails immediately and the build adapts to the remaining
    # sources instead of burning the deadline on retry storms.
    fetch_cfg = payload.get("fetch") or {}

    def _tune(src: str, default_attempts: int, default_timeout: float) -> dict[str, Any]:
        c = fetch_cfg.get(src) or {}
        attempts = int(c.get("max_attempts", default_attempts))
        return {
            "search_max_attempts": attempts,
            "fetch_max_attempts": int(c.get("fetch_attempts", attempts)),
            "request_timeout": float(c.get("timeout", default_timeout)),
        }

    fetch_cache = FetchCache(root=None)
    manager = AdapterManager()
    manager.probe_and_register([
        ArxivAdapter(fetch_cache=fetch_cache, **_tune("arxiv", 1, 20.0)),
        WikipediaAdapter(fetch_cache=fetch_cache),
        GdeltAdapter(fetch_cache=fetch_cache, **_tune("gdelt", 1, 15.0)),
    ])
    adapters = manager.as_dict()

    config = LoopConfig(per_source_search=1, selector_budget=docs_per_round, max_rounds=max_rounds)

    # Wall-clock guard: fire the pipeline's shutdown event before the request
    # timeout so the build stops at a clean round boundary and returns a
    # partial-but-valid result + trace.
    shutdown = asyncio.Event()

    async def _guard() -> None:
        await asyncio.sleep(max(30.0, deadline_s))
        shutdown.set()

    guard = asyncio.create_task(_guard())
    try:
        outcome = await run_pipeline(
            base_query=base_query, kg_root=kg_root, adapters=adapters,
            openalex=None, llm_model=None, settings=settings, trace=trace,
            config=config, resume=False, force=False, shutdown_event=shutdown,
        )
    finally:
        guard.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await guard

    fo = outcome.fetch_outcomes or {}
    failed = fo.get("fetch_failed", 0)
    active_sources = sorted(adapters)
    report_md = _build_report_markdown(base_query, outcome, fo, active_sources)
    summary = (
        f"Build “{kg_id}” for “{base_query}”: {outcome.rounds_completed} round(s), "
        f"{outcome.total_docs} docs from {active_sources}, "
        f"{outcome.kg_entity_count} entities / {outcome.kg_relation_count} relations"
        + (f"; {failed} source fetch(es) failed and were handled gracefully" if failed else "")
        + f" (stop: {outcome.stop_reason})."
    )
    return RunOutcome(
        request_id=request_id,
        output={
            "task": "build",
            "summary": summary,
            "report_markdown": report_md,
            "kg_id": kg_id,
            "base_query": base_query,
            "rounds_completed": outcome.rounds_completed,
            "docs_ingested": outcome.total_docs,
            "entities": outcome.kg_entity_count,
            "relations": outcome.kg_relation_count,
            "schema_entity_types": outcome.schema_size[0],
            "schema_relation_types": outcome.schema_size[1],
            "stop_reason": outcome.stop_reason,
            "sources_active": active_sources,
            "fetch_degraded": bool(failed),
            "fetch_outcomes": fo,
        },
        agents_used=[a.name for a in _BUILD_AGENTS],
        agents=list(_BUILD_AGENTS),
        runtime_seconds=round(time.monotonic() - started, 3),
        trace_path=trace.relative_path,
    )


def _build_report_markdown(base_query: str, outcome: Any, fo: dict[str, Any], sources: list[str]) -> str:
    failed = fo.get("fetch_failed", 0)
    lines = [
        f"# Knowledge-graph build — “{base_query}”",
        "",
        "## Result",
        f"- **Rounds completed:** {outcome.rounds_completed} (stop: `{outcome.stop_reason}`)",
        f"- **Documents ingested:** {outcome.total_docs} from sources {sources}",
        f"- **Graph:** {outcome.kg_entity_count} entities · {outcome.kg_relation_count} relations · "
        f"schema {outcome.schema_size[0]} entity-types / {outcome.schema_size[1]} relation-types",
    ]
    if failed:
        lines += [
            "",
            "## Fetch failures",
            f"- **{failed} source fetch(es) failed** "
            f"({100 * fo.get('fetch_failed_rate', 0.0):.0f}% of attempted) — "
            f"by source: `{fo.get('fetch_failed_by_source', {})}`. "
            "The build continued with the remaining sources.",
        ]
    lines += [
        "",
        "## The agent loop (see the JSONL trace for each call)",
        "RoundPlanner → per-source QueryPlanner + search → Selector → UsefulnessGate "
        "→ Extractor → Mapper / AttributeMapper → Resolver → KGReviewer → Strategist. "
        "Feedback loops: the Strategist's review feeds the next round's plan; the "
        "MappingReviewer can trigger a re-map; the Resolver merges across the round.",
    ]
    return "\n".join(lines)


def _require_kg(payload: dict[str, Any], settings: Settings) -> Path:
    kg_id = _text(payload, "kg_id")
    if not kg_id:
        raise ValueError("this task requires input.kg_id (a graph under MEMEX_KG_DIR).")
    kg_root = settings.kg_dir / kg_id
    if not (kg_root / "kg.db").exists():
        raise FileNotFoundError(
            f"No knowledge graph at {kg_root / 'kg.db'}. Build one with `memex build` "
            "or download the demo graph with `memex fetch-demo`."
        )
    return kg_root
