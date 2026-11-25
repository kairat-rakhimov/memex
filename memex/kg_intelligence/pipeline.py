"""kg-loop orchestrator: outer loop + lifecycle surface.

Drives round counting, the stop criteria, cross-round ``kg_gaps``
feedback, and round-end persistence. Per-round work lives in
``memex.kg_intelligence.round_graph``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError
from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.common.trace import TraceWriter, utc_now_iso
from memex.kg_intelligence.agents import (
    AgentCallSinks,
    AgentHardFailure,
    LLMThrottleError,
    kg_review,
    round_review,
)
from memex.kg_intelligence.agents.schemas import (
    GoalDirective,
    GoalVerdict,
    KgGap,
    KgReviewResult,
    RoundReviewResult,
    RoundStatus,
)
from memex.kg_intelligence.agents.vocab_resolver import cluster_and_commit
from memex.kg_intelligence.openalex import OpenAlexClient
from memex.kg_intelligence.round_graph import (
    RoundGraphDeps,
    RoundGraphState,
    _RoundDeltas,
    build_round_graph,
)
from memex.kg_intelligence.sources.common.base import SourceAdapter
from memex.kg_intelligence.state import CorpusState
from memex.kg_intelligence.views import load_kg_view
from memex.observability import RoundHandle, TraceStore

_log = logging.getLogger(__name__)

PIPELINE = "kg_intelligence"
KG_JSON_FILE = "kg.json"
SCHEMA_JSON_FILE = "schema.json"


@dataclass(frozen=True)
class LoopConfig:
    # Per-source ranker top-K — a generous pre-filter that bounds the
    # global Selector's prompt, not a final cap.
    per_source_search: int = 10
    # Global Selector budget: max docs fetched per round across ALL sources
    # (priority-weighted goal-fit, no per-source quota).
    selector_budget: int = 10
    # Hard ceiling on rounds — a cost backstop. The RoundReviewer is the real
    # stop; this just bounds runaway.
    max_rounds: int = 40
    # Deterministic pathology floor: halt after this many consecutive rounds
    # that add 0 new entities AND 0 new relations (catches the RoundReviewer
    # hallucinating "continuing" on literal zero progress).
    zero_progress_rounds: int = 3
    # How many recent rounds of metrics the RoundReviewer sees (trend window).
    round_review_window: int = 5
    # After this many consecutive throttle-exhaustion rounds, stop with
    # ``llm_throttle``. Cooldown between attempts:
    # ``llm_circuit_break_cooldown_seconds``.
    llm_circuit_break_threshold: int = 3
    llm_circuit_break_cooldown_seconds: float = 60.0
    # Debug-only kg.json + schema.json export alongside canonical kg.db.
    write_kg_json_export: bool = False


@dataclass
class LoopOutcome:
    rounds_completed: int
    total_docs: int
    kg_entity_count: int
    kg_relation_count: int
    schema_size: tuple[int, int]
    stop_reason: str
    kg_db_path: Path | None = None
    # Build-level fetch-outcome rollup (degrade visibility). See
    # TraceStore.fetch_outcome_summary.
    fetch_outcomes: dict = field(default_factory=dict)


async def _finalize_vocab_clusters(
    *,
    store: TraceStore,
    settings: Settings,
    sinks: AgentCallSinks,
    llm_model: Model | None,
    round_number: int,
) -> None:
    """Post-loop VocabResolver finalization phase. Clusters the FINAL canonical vocabulary — predicates AND
    properties — into non-destructive alias / child_of metadata so query-time
    expansion (``memex.kg_query.expansion``) has structure to walk. Without it a
    build ships with flat vocab. Each lane is guarded: a clustering failure logs and
    is skipped so the built KG is kept (clusters can be (re)built post-hoc)."""

    for kind in ("predicate", "property"):
        try:
            totals = await cluster_and_commit(
                store=store,
                kind=kind,
                settings=settings,
                sinks=sinks,
                llm_model=llm_model,
                round_number=round_number,
            )
            _log.info("vocab finalization (%s): %s", kind, totals)
        except Exception:
            _log.exception(
                "vocab finalization (%s) failed — KG kept, clusters skipped", kind
            )


# ── public entry point ────────────────────────────────────────────────────


async def run_pipeline(
    *,
    base_query: str,
    kg_root: Path,
    adapters: dict[str, SourceAdapter],
    openalex: OpenAlexClient | None,
    llm_model: Model | None,
    settings: Settings,
    trace: TraceWriter,
    config: LoopConfig | None = None,
    resume: bool = True,
    force: bool = False,
    build_run_id: str | None = None,
    store: TraceStore | None = None,
    shutdown_event: asyncio.Event | None = None,
) -> LoopOutcome:
    """Run the loop end-to-end.

    ``force=True`` bypasses the ``has_document()`` skip so re-fetches happen
    even when the doc is already persisted in kg.db. ``resume`` and ``force`` are
    independent.

    SIGINT / SIGTERM cancel the in-flight round; SQLite transactions
    auto-rollback, no ``loop_state`` row is written, and resume picks
    up from the prior completed round. A second signal force-exits.
    Tests can inject ``shutdown_event`` directly.
    """

    cfg = config or LoopConfig()
    kg_root.mkdir(parents=True, exist_ok=True)

    # Fail fast on a misconfigured production run. SAMPLE_MODE skips
    # the LLM entirely; a caller-supplied ``llm_model`` (tests)
    # provides its own model so the global api_key isn't required.
    # Otherwise per-agent model resolution will need ``settings.api_key``
    # on the very first agent call — better to surface that now than
    # mid-round.
    if not settings.sample_mode and llm_model is None and not settings.api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — cannot run kg-loop against the LLM API. "
            "Either set the key or enable SAMPLE_MODE for the build."
        )

    own_store = store is None
    if store is None:
        store = TraceStore.open(kg_root)

    shutting_down = shutdown_event if shutdown_event is not None else asyncio.Event()
    installed_signals: list[signal.Signals] = []

    def _signal_handler() -> None:
        if not shutting_down.is_set():
            _log.info("Shutdown requested; cancelling current round")
            shutting_down.set()
        else:
            _log.warning("Second signal received; force exit")
            sys.exit(130)

    try:
        running_loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                running_loop.add_signal_handler(sig, _signal_handler)
                installed_signals.append(sig)
            except (NotImplementedError, ValueError):
                # NotImplementedError: signals not supported on this platform
                # (e.g. Windows). ValueError: not in the main thread (e.g.
                # some pytest setups). Tests inject ``shutdown_event``
                # directly when this path is unavailable.
                pass
    except RuntimeError:
        running_loop = None

    corpus_state = _initial_corpus_state(store, resume=resume)
    # base_query is the persistent topical anchor — Planner + Reviewer
    # use it every round. On resume, the stored value wins; the
    # caller-supplied value is only relevant when starting fresh.
    if corpus_state.base_query:
        if base_query and base_query != corpus_state.base_query:
            _log.warning(
                "Ignoring base_query=%r; resuming with stored base_query=%r",
                base_query,
                corpus_state.base_query,
            )
        base_query = corpus_state.base_query
    else:
        corpus_state.base_query = base_query
    started_monotonic = time.monotonic()
    if not corpus_state.started_at:
        corpus_state.started_at = utc_now_iso()

    build = store.start_build(
        build_run_id=build_run_id or f"kg_build_{kg_root.name}_{time.time_ns()}",
        kg_id=kg_root.name,
        base_query=base_query,
        config={
            "per_source_search": cfg.per_source_search,
            "selector_budget": cfg.selector_budget,
            "max_rounds": cfg.max_rounds,
            "zero_progress_rounds": cfg.zero_progress_rounds,
            "round_review_window": cfg.round_review_window,
        },
        sources=list(adapters.keys()),
        sample_mode=bool(settings.sample_mode),
        resume_from_round=corpus_state.round_number or None,
    )

    stop_reason = ""
    consecutive_throttle = 0
    try:
        # Resume: a round whose KG committed but whose Phase-2 reviews never
        # finished leaves a pending round_reviews row (kg_review_call_id NULL).
        # Re-run reviews only — never refetch — then continue normally.
        if not shutting_down.is_set():
            pending = store.read_pending_round_review()
            if pending is not None:
                stop_reason = await _resume_pending_reviews(
                    pending=pending,
                    base_query=base_query,
                    corpus_state=corpus_state,
                    cfg=cfg,
                    llm_model=llm_model,
                    settings=settings,
                    trace=trace,
                    store=store,
                )

        round_handle = None
        while not stop_reason:
            if shutting_down.is_set():
                stop_reason = "sigint"
                break

            corpus_state.round_number += 1
            entities_before = store.count_entities()
            relations_before = store.count_relations()
            schema_entity_types_before = store.count_schema_types("entity_type")
            schema_relation_types_before = store.count_schema_types("relation_type")
            round_handle = store.start_round(
                build,
                round_number=corpus_state.round_number,
                entities_before=entities_before,
                relations_before=relations_before,
                schema_entity_types_before=schema_entity_types_before,
                schema_relation_types_before=schema_relation_types_before,
            )
            sinks = AgentCallSinks(trace=trace, store=store, round_=round_handle)
            # Strategist inputs, read from the DB so resume is correct:
            # the prior round's KGReviewer gaps + RoundReviewer directives.
            latest_gaps, latest_directives = _read_latest_strategy(store)
            round_started = time.monotonic()
            _log.info(
                "── round %d start ── entities=%d relations=%d "
                "schema=%d/%d gaps_in=%d directives_in=%d",
                corpus_state.round_number,
                entities_before, relations_before,
                schema_entity_types_before, schema_relation_types_before,
                len(latest_gaps), len(latest_directives),
            )

            # Race the round against the shutdown event. If a signal
            # arrives mid-round, the shutdown_task wins; we cancel the
            # round_task and exit. SQLite transactions auto-rollback on
            # cancel, so partial KG writes vanish; the prior round's
            # loop_state row is the canonical resume point.
            round_task = asyncio.create_task(
                _run_round(
                    base_query=base_query,
                    corpus_state=corpus_state,
                    kg_gaps=latest_gaps,
                    directives=latest_directives,
                    adapters=adapters,
                    openalex=openalex,
                    llm_model=llm_model,
                    settings=settings,
                    sinks=sinks,
                    cfg=cfg,
                    kg_root=kg_root,
                    force=force,
                )
            )
            shutdown_task = asyncio.create_task(shutting_down.wait())
            done, _pending = await asyncio.wait(
                [round_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if shutdown_task in done:
                round_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await round_task
                store.finish_round(
                    round_handle,
                    entities_after=entities_before,
                    relations_after=relations_before,
                    schema_entity_types_after=schema_entity_types_before,
                    schema_relation_types_after=schema_relation_types_before,
                    docs_fetched=0,
                    docs_skipped=0,
                    docs_failed=0,
                    stop_criterion_fired="sigint",
                    duration_seconds=round(time.monotonic() - round_started, 3),
                )
                stop_reason = "sigint"
                break

            shutdown_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await shutdown_task

            try:
                deltas = round_task.result()
            except LLMThrottleError as exc:
                consecutive_throttle += 1
                _log.warning(
                    "LLM throttle %d/%d at round %d: %s",
                    consecutive_throttle,
                    cfg.llm_circuit_break_threshold,
                    corpus_state.round_number,
                    exc,
                )
                store.finish_round(
                    round_handle,
                    entities_after=entities_before,
                    relations_after=relations_before,
                    schema_entity_types_after=schema_entity_types_before,
                    schema_relation_types_after=schema_relation_types_before,
                    docs_fetched=0,
                    docs_skipped=0,
                    docs_failed=0,
                    stop_criterion_fired="llm_throttle",
                    duration_seconds=round(time.monotonic() - round_started, 3),
                )
                if consecutive_throttle >= cfg.llm_circuit_break_threshold:
                    stop_reason = "llm_throttle"
                    break
                _log.info(
                    "Cooling down %.0fs before next round",
                    cfg.llm_circuit_break_cooldown_seconds,
                )
                # Interruptible cooldown: a SIGINT arriving during the cooldown
                # exits the loop on the next iteration rather than waiting out the
                # full 60s. TimeoutError = cooldown elapsed normally (no SIGINT).
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        shutting_down.wait(),
                        timeout=cfg.llm_circuit_break_cooldown_seconds,
                    )
                continue

            consecutive_throttle = 0
            if deltas.docs_fetched == 0:
                corpus_state.consecutive_empty_rounds += 1
            else:
                corpus_state.consecutive_empty_rounds = 0

            entities_after = store.count_entities()
            relations_after = store.count_relations()
            schema_entity_types_after = store.count_schema_types("entity_type")
            schema_relation_types_after = store.count_schema_types("relation_type")
            new_entities = entities_after - entities_before
            new_relations = relations_after - relations_before

            # ── Phase-1 checkpoint: metrics row (reviews pending) + loop_state.
            # Tokens are summed now, before the reviews run, so the row's
            # tokens_spent_total is the round's *work* cost.
            store.record_round_metrics(
                round_handle,
                new_entities=new_entities,
                new_relations=new_relations,
                docs_fetched_total=deltas.docs_fetched,
                already_ingested_total=deltas.already_ingested,
                candidates_surfaced_total=deltas.candidates_surfaced,
                already_seen_by_topic=deltas.already_seen_by_topic,
            )
            corpus_state.elapsed_seconds = round(time.monotonic() - started_monotonic, 3)
            store.write_loop_state(corpus_state.model_dump_json())

            # ── Phase 2: the two reviews (orchestrator-level). A hard failure
            # leaves the pending row so a restart re-runs Phase 2 only.
            try:
                round_review_result, recent_metrics = await _run_phase2_reviews(
                    base_query=base_query,
                    corpus_state=corpus_state,
                    round_handle=round_handle,
                    recent_metrics_window=cfg.round_review_window,
                    llm_model=llm_model,
                    settings=settings,
                    sinks=sinks,
                    store=store,
                )
            except AgentHardFailure as exc:
                _log.error(
                    "round %d reviews failed (%s); halting — restart resumes "
                    "Phase 2 from the pending round_reviews row",
                    corpus_state.round_number, exc,
                )
                store.finish_round(
                    round_handle,
                    entities_after=entities_after,
                    relations_after=relations_after,
                    schema_entity_types_after=schema_entity_types_after,
                    schema_relation_types_after=schema_relation_types_after,
                    docs_fetched=deltas.docs_fetched,
                    docs_skipped=deltas.docs_skipped,
                    docs_failed=deltas.docs_failed,
                    stop_criterion_fired="review_failed",
                    duration_seconds=round(time.monotonic() - round_started, 3),
                )
                stop_reason = "review_failed"
                break

            stop_reason = _check_stop(
                cfg=cfg,
                corpus_state=corpus_state,
                round_status=round_review_result.status,
                zero_progress_streak=_zero_progress_streak(recent_metrics),
            )

            duration_s = round(time.monotonic() - round_started, 3)
            store.finish_round(
                round_handle,
                entities_after=entities_after,
                relations_after=relations_after,
                schema_entity_types_after=schema_entity_types_after,
                schema_relation_types_after=schema_relation_types_after,
                docs_fetched=deltas.docs_fetched,
                docs_skipped=deltas.docs_skipped,
                docs_failed=deltas.docs_failed,
                stop_criterion_fired=stop_reason,
                duration_seconds=duration_s,
            )
            _log.info(
                "── round %d end ── %.1fs docs_fetched=%d docs_skipped=%d "
                "entities=%d (+%d) relations=%d (+%d) schema=%d/%d "
                "status=%s stop=%s",
                corpus_state.round_number, duration_s,
                deltas.docs_fetched, deltas.docs_skipped,
                entities_after, new_entities,
                relations_after, new_relations,
                schema_entity_types_after, schema_relation_types_after,
                round_review_result.status.value,
                stop_reason or "—",
            )

            if stop_reason:
                break

        # VocabResolver finalization (post-loop): cluster the final
        # vocab into alias/child_of metadata for query-time expansion. Skipped on
        # an aborted stop (sigint / throttle can leave the last round rolled back);
        # otherwise attach the cluster agent calls to the last completed round.
        if round_handle is not None and stop_reason not in ("sigint", "llm_throttle"):
            await _finalize_vocab_clusters(
                store=store,
                settings=settings,
                sinks=AgentCallSinks(trace=trace, store=store, round_=round_handle),
                llm_model=llm_model,
                round_number=corpus_state.round_number,
            )

        if cfg.write_kg_json_export:
            _finalise_json_export(kg_root, store)

        final_entities = store.count_entities()
        final_relations = store.count_relations()
        final_schema_e = store.count_schema_types("entity_type")
        final_schema_r = store.count_schema_types("relation_type")
        store.finish_build(
            build,
            stop_reason=stop_reason,
            rounds_completed=corpus_state.round_number,
            total_docs=len(corpus_state.urls_fetched),
            total_entities=final_entities,
            total_relations=final_relations,
            schema_entity_types=final_schema_e,
            schema_relation_types=final_schema_r,
        )

        return LoopOutcome(
            rounds_completed=corpus_state.round_number,
            total_docs=len(corpus_state.urls_fetched),
            kg_entity_count=final_entities,
            kg_relation_count=final_relations,
            schema_size=(final_schema_e, final_schema_r),
            stop_reason=stop_reason,
            kg_db_path=store.db_path,
            fetch_outcomes=store.fetch_outcome_summary(),
        )
    finally:
        if running_loop is not None:
            for sig in installed_signals:
                with contextlib.suppress(NotImplementedError, ValueError, RuntimeError):
                    running_loop.remove_signal_handler(sig)
        if own_store:
            store.close()


# ── per-round wrapper around the graph ───────────────────────────────────


async def _run_round(
    *,
    base_query: str,
    corpus_state: CorpusState,
    kg_gaps: list[KgGap],
    directives: list[GoalDirective],
    adapters: dict[str, SourceAdapter],
    openalex: OpenAlexClient | None,
    llm_model: Model | None,
    settings: Settings,
    sinks: AgentCallSinks,
    cfg: LoopConfig,
    kg_root: Path,
    force: bool,
) -> _RoundDeltas:
    """Construct per-round state + deps, run the Phase-1 graph, return its
    deltas. The reviews run in the orchestrator's Phase 2, not the graph."""

    state = RoundGraphState(
        corpus_state=corpus_state,
        kg_gaps_in=kg_gaps,
        directives_in=directives,
        deltas=_RoundDeltas(),
    )
    deps = RoundGraphDeps(
        base_query=base_query,
        adapters=adapters,
        openalex=openalex,
        llm_model=llm_model,
        settings=settings,
        sinks=sinks,
        cfg=cfg,
        kg_root=kg_root,
        force=force,
    )
    graph = build_round_graph()
    output = await graph.run(state=state, deps=deps)
    return output.deltas


# ── Phase 2: end-of-round reviews ─────────────────────────────────────────


async def _run_phase2_reviews(
    *,
    base_query: str,
    corpus_state: CorpusState,
    round_handle: RoundHandle,
    recent_metrics_window: int,
    llm_model: Model | None,
    settings: Settings,
    sinks: AgentCallSinks,
    store: TraceStore,
) -> tuple[RoundReviewResult, list[dict]]:
    """Run KGReviewer → RoundReviewer against the committed KG, stamp the
    round_reviews FKs, fold ``abandon`` verdicts into
    ``CorpusState.abandoned_goals``, and persist loop_state (Phase-2 write).

    Returns ``(round_review_result, recent_metrics)``. Raises
    ``AgentHardFailure`` if either review fails (caller halts + resumes)."""

    kg = load_kg_view(store)
    kg_review_result, kg_call_id = await kg_review(
        base_query=base_query,
        kg=kg,
        corpus_state=corpus_state,
        llm_model=llm_model,
        settings=settings,
        sinks=sinks,
    )
    recent_metrics = store.read_recent_round_metrics(recent_metrics_window)
    round_review_result, rr_call_id = await round_review(
        base_query=base_query,
        corpus_state=corpus_state,
        kg_review=kg_review_result,
        recent_metrics=recent_metrics,
        llm_model=llm_model,
        settings=settings,
        sinks=sinks,
    )
    store.stamp_round_reviews(
        round_handle,
        kg_review_call_id=kg_call_id,
        round_review_call_id=rr_call_id,
    )
    for directive in round_review_result.goal_directives:
        if (
            directive.verdict == GoalVerdict.abandon
            and directive.theme not in corpus_state.abandoned_goals
        ):
            corpus_state.abandoned_goals.append(directive.theme)
    store.write_loop_state(corpus_state.model_dump_json())
    return round_review_result, recent_metrics


async def _resume_pending_reviews(
    *,
    pending: dict,
    base_query: str,
    corpus_state: CorpusState,
    cfg: LoopConfig,
    llm_model: Model | None,
    settings: Settings,
    trace: TraceWriter,
    store: TraceStore,
) -> str:
    """Re-run Phase 2 for a round whose KG committed but whose reviews never
    finished (the resume contract). Returns the resulting stop_reason."""

    round_handle = RoundHandle(
        round_id=pending["round_id"], round_number=pending["round_number"]
    )
    store.prime_round_sequence(round_handle)
    sinks = AgentCallSinks(trace=trace, store=store, round_=round_handle)
    _log.info(
        "resume: re-running Phase-2 reviews for round %d (no refetch)",
        pending["round_number"],
    )
    started = time.monotonic()
    try:
        round_review_result, recent_metrics = await _run_phase2_reviews(
            base_query=base_query,
            corpus_state=corpus_state,
            round_handle=round_handle,
            recent_metrics_window=cfg.round_review_window,
            llm_model=llm_model,
            settings=settings,
            sinks=sinks,
            store=store,
        )
    except AgentHardFailure as exc:
        _log.error("resume: reviews failed again (%s); halting", exc)
        return "review_failed"

    stop_reason = _check_stop(
        cfg=cfg,
        corpus_state=corpus_state,
        round_status=round_review_result.status,
        zero_progress_streak=_zero_progress_streak(recent_metrics),
    )
    store.finish_round(
        round_handle,
        entities_after=store.count_entities(),
        relations_after=store.count_relations(),
        schema_entity_types_after=store.count_schema_types("entity_type"),
        schema_relation_types_after=store.count_schema_types("relation_type"),
        docs_fetched=int(pending.get("docs_fetched_total", 0)),
        docs_skipped=0,
        docs_failed=0,
        stop_criterion_fired=stop_reason,
        duration_seconds=round(time.monotonic() - started, 3),
    )
    return stop_reason


def _read_latest_strategy(
    store: TraceStore,
) -> tuple[list[KgGap], list[GoalDirective]]:
    """Parse the prior round's reviews from the DB into the planner's inputs:
    KGReviewer gaps + RoundReviewer directives. Empty on R1 / parse failure."""

    kg_json, rr_json = store.read_latest_reviews()
    gaps: list[KgGap] = []
    directives: list[GoalDirective] = []
    if kg_json is not None:
        try:
            gaps = KgReviewResult.model_validate(kg_json).gaps
        except ValidationError:
            _log.warning("could not parse stored KGReviewer output; gaps empty")
    if rr_json is not None:
        try:
            directives = RoundReviewResult.model_validate(rr_json).goal_directives
        except ValidationError:
            _log.warning("could not parse stored RoundReviewer output; directives empty")
    return gaps, directives


# ── stopping criteria ─────────────────────────────────────────────────────


def _check_stop(
    *,
    cfg: LoopConfig,
    corpus_state: CorpusState,
    round_status: RoundStatus,
    zero_progress_streak: int,
) -> str:
    """First stop reason that fires, or '' if none.

    The RoundReviewer owns the nuanced ROI call; the two deterministic floors
    catch only pathology (literal zero progress, runaway rounds)."""

    if round_status in (RoundStatus.stalled, RoundStatus.wasted):
        return f"round_review_{round_status.value}"
    if zero_progress_streak >= cfg.zero_progress_rounds:
        return "zero_progress_floor"
    if corpus_state.round_number >= cfg.max_rounds:
        return "max_rounds"
    return ""


def _zero_progress_streak(recent_metrics: list[dict]) -> int:
    """Trailing run of rounds (newest first) that added 0 new entities AND 0
    new relations — the deterministic zero-progress floor's input."""

    streak = 0
    for metrics in reversed(recent_metrics):
        if int(metrics.get("new_entities") or 0) == 0 and (
            int(metrics.get("new_relations") or 0) == 0
        ):
            streak += 1
        else:
            break
    return streak


# ── resume + finalise ─────────────────────────────────────────────────────


def _initial_corpus_state(store: TraceStore, *, resume: bool) -> CorpusState:
    """Hydrate CorpusState from the loop_state row, or start fresh."""

    if not resume:
        return CorpusState()
    blob = store.read_loop_state()
    if blob is None:
        return CorpusState()
    return CorpusState.model_validate_json(blob)


def _finalise_json_export(kg_root: Path, store: TraceStore) -> None:
    """Debug-only kg.json + schema.json export derived from kg.db."""

    schema = store.list_kg_schema()
    entities = store.list_kg_entities()
    with store._lock:  # noqa: SLF001 — single read-side traversal
        rows = store._conn.execute(  # noqa: SLF001
            """
            SELECT kr.id, ks.canonical_name AS subject, kr.predicate,
                   ko.canonical_name AS object, kr.source_doc_id,
                   kr.source_chunk_id, kr.round_observed, kr.agent_call_id
            FROM kg_relations kr
            JOIN kg_entities ks ON ks.id = kr.subject_entity_id
            JOIN kg_entities ko ON ko.id = kr.object_entity_id
            ORDER BY kr.id ASC
            """
        ).fetchall()
        relations = [dict(r) for r in rows]
    kg_payload = {
        "entities": {
            e["canonical_name"]: {
                k: e[k] for k in (
                    "entity_type", "mentions", "degree",
                    "first_seen_round", "last_seen_round",
                )
            }
            for e in entities
        },
        "relations": relations,
        "schema": schema,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (kg_root / KG_JSON_FILE).write_text(
        json.dumps(kg_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (kg_root / SCHEMA_JSON_FILE).write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "KG_JSON_FILE",
    "LoopConfig",
    "LoopOutcome",
    "SCHEMA_JSON_FILE",
    "run_pipeline",
]
