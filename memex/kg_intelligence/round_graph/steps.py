"""Step functions for the per-round graph.

``extract_and_consolidate_step`` processes docs sequentially within a
round. For each doc: chunk → extract free-form (parallel per chunk, or
session) → (chunked only) reconcile cross-chunk relations → per-doc
Mapper loop (canonicalize vocabulary, gate net-new canonicals) → merge
the gated triples → embed → resolve → embed predicates. The Extractor is
schema-blind; the Mapper canonicalizes the doc's whole raw-triple set in
one reasoning pass.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from pydantic_graph import StepContext

if TYPE_CHECKING:
    from memex.observability import TraceStore

_log = logging.getLogger(__name__)

# How many recent goal records CorpusState carries for the RoundReviewer.
_GOAL_HISTORY_CAP = 12

# Session-extraction fan-out cap: after fork 0 warms the whole-doc prefix cache,
# forks 1..N run concurrently up to this many in flight (bounds LLM API load
# while keeping the cache warm). Quality is order-independent.
_EXTRACT_FANOUT = 8


@asynccontextmanager
async def _step_timer(round_number: int, label: str) -> AsyncIterator[None]:
    _log.info("round %d ▶ %s", round_number, label)
    t0 = time.monotonic()
    yield
    elapsed = time.monotonic() - t0
    _log.info("round %d ◀ %s (%.1fs)", round_number, label, elapsed)

from memex.kg_intelligence.agents import (
    RawAttribute,
    RawTriple,
    assess_doc,
    extract,
    map_doc,
    plan,
    reconcile,
    resolve,
    select_documents,
)
from memex.kg_intelligence.agents.extractor import SESSION_DOC_BUDGET
from memex.kg_intelligence.agents.mapper import MapDocResult
from memex.kg_intelligence.agents.schemas import ExtractionResult
from memex.kg_intelligence.embeddings import embed_and_store
from memex.kg_intelligence.round_graph.helpers import (
    _fetch_and_persist,
    _filter_already_ingested,
    _flatten_pool,
    _group_by_source,
    _plan_search_and_rank,
    _union,
)
from memex.kg_intelligence.round_graph.state import (
    RoundGraphDeps,
    RoundGraphState,
    RoundOutput,
)
from memex.kg_intelligence.sources.common.base import ChunkedDoc
from memex.kg_intelligence.state import RoundGoalRecord, SelectedTitle
from memex.kg_intelligence.views import load_schema_view


def _require_store(deps: RoundGraphDeps):
    if deps.sinks.store is None:
        raise RuntimeError(
            "RoundGraphDeps.sinks.store is None; the round graph requires "
            "a real TraceStore."
        )
    return deps.sinks.store


# ── graph-level steps: acquisition ───────────────────────────────────────


async def plan_step(
    ctx: StepContext[RoundGraphState, RoundGraphDeps, None],
) -> None:
    """Planner turn — capture pre-round snapshot, then ask for the ranked topic list."""

    state, deps = ctx.state, ctx.deps
    store = _require_store(deps)
    rn = state.corpus_state.round_number

    async with _step_timer(rn, "Plan"):
        snapshot = store.entity_snapshot_for_resolver()
        state.pre_entity_count = len(snapshot)
        state.pre_entity_names = set(snapshot.keys())
        state.pre_mentions = {n: e.mentions for n, e in snapshot.items()}

        state.plan_result = await plan(
            base_query=deps.base_query,
            corpus_state=state.corpus_state,
            kg_gaps=state.kg_gaps_in,
            directives=state.directives_in,
            adapter_descriptions=[a.description for a in deps.adapters.values()],
            llm_model=deps.llm_model,
            settings=deps.settings,
            sinks=deps.sinks,
        )

        # Record this round's aim so the RoundReviewer can see which themes have
        # recurred (its "wasted N rounds" judgment). Trimmed to recent.
        # ``priority_themes`` = the ranked topic strings; ``focused_goal`` = the
        # top topic — the RoundGoalRecord projection the reviewer still reads.
        topics = state.plan_result.topics
        state.corpus_state.goal_history.append(
            RoundGoalRecord(
                round=rn,
                focused_goal=topics[0].topic if topics else "",
                priority_themes=[t.topic for t in topics],
            )
        )
        del state.corpus_state.goal_history[:-_GOAL_HISTORY_CAP]

        _log.info(
            "round %d plan: %d topics [%s]; relevance=%s",
            rn,
            len(topics),
            ", ".join(t.topic for t in topics[:6]) or "(none)",
            state.plan_result.adapter_relevance,
        )


async def search_rank_select_step(
    ctx: StepContext[RoundGraphState, RoundGraphDeps, None],
) -> None:
    """Adapters plan their own queries + search; the per-source ranker
    pre-filters; then the global LLM Selector picks ≤budget by
    priority-weighted goal-fit across the flat cross-source pool."""

    state, deps = ctx.state, ctx.deps
    rn = state.corpus_state.round_number
    plan = state.plan_result
    ranked_topics = [t.topic for t in plan.topics] if plan is not None else []
    adapter_relevance = dict(plan.adapter_relevance) if plan is not None else {}

    async with _step_timer(rn, "Plan-search + Rank + Select"):
        ranked_by_source = await _plan_search_and_rank(
            ranked_topics=ranked_topics,
            adapter_relevance=adapter_relevance,
            adapters=deps.adapters,
            cfg=deps.cfg,
            settings=deps.settings,
            sinks=deps.sinks,
            llm_model=deps.llm_model,
        )
        ranked_by_source, n_already_ingested, already_seen_by_topic = (
            _filter_already_ingested(ranked_by_source, deps.sinks.store)
        )
        pool = _flatten_pool(ranked_by_source)
        state.deltas.already_ingested = n_already_ingested
        state.deltas.candidates_surfaced = n_already_ingested + len(pool)
        state.deltas.already_seen_by_topic = already_seen_by_topic
        chosen = await select_documents(
            pool=pool,
            ranked_topics=ranked_topics,
            adapter_descriptions=[a.description for a in deps.adapters.values()],
            budget=deps.cfg.selector_budget,
            settings=deps.settings,
            sinks=deps.sinks,
            llm_model=deps.llm_model,
        )
        state.selected_by_source = _group_by_source(chosen)
        _log.info(
            "round %d plan-search+select: pool=%d (%d already-ingested dropped) "
            "→ %d picked [%s]",
            rn,
            len(pool),
            n_already_ingested,
            len(chosen),
            ", ".join(
                f"{src}:{len(state.selected_by_source.get(src, []))}"
                for src in deps.adapters
            ) or "(nothing searched)",
        )


async def fetch_step(
    ctx: StepContext[RoundGraphState, RoundGraphDeps, None],
) -> None:
    """Fetch + persist selected candidates; update deltas + corpus_state."""

    state, deps = ctx.state, ctx.deps
    rn = state.corpus_state.round_number

    async with _step_timer(rn, "Fetch"):
        state.fetched_by_doc_id, fetch_deltas = await _fetch_and_persist(
            selected_by_source=state.selected_by_source,
            adapters=deps.adapters,
            openalex=deps.openalex,
            base_query=deps.base_query,
            force=deps.force,
            settings=deps.settings,
            sinks=deps.sinks,
        )
        state.deltas.docs_fetched = fetch_deltas["fetched"]
        state.deltas.docs_skipped = fetch_deltas["skipped"]
        state.deltas.docs_failed = fetch_deltas["failed"]
        state.deltas.new_candidates = [fd[0].candidate for fd in state.fetched_by_doc_id.values()]

        for _doc_id_str, (fd, _row_id) in state.fetched_by_doc_id.items():
            state.corpus_state.urls_fetched.append(fd.candidate.url)
            state.corpus_state.titles_selected.append(
                SelectedTitle(source=fd.candidate.source, title=fd.candidate.title)
            )
        _log.info(
            "round %d fetch: %d fetched, %d skipped_duplicate, %d failed",
            rn,
            state.deltas.docs_fetched,
            state.deltas.docs_skipped,
            state.deltas.docs_failed,
        )


async def gate_step(
    ctx: StepContext[RoundGraphState, RoundGraphDeps, None],
) -> None:
    """Per-doc whole-doc UsefulnessGate.

    For each fetched doc: pass-1 usefulness verdict over the WHOLE doc. A
    ``skip`` drops the doc from this round's extraction set (saving the
    extraction + Mapper spend); a ``keep`` runs pass-2 (summary + reflection)
    and folds the reflection into corpus_state. Every doc's verdict (drops
    included) is persisted to ``doc_assessments`` — drop-but-persist."""

    state, deps = ctx.state, ctx.deps
    store = deps.sinks.store
    rn = state.corpus_state.round_number

    if not state.fetched_by_doc_id:
        return

    async with _step_timer(rn, "Usefulness gate"):
        skipped = 0
        for doc_id_str, (fd, row_id) in list(state.fetched_by_doc_id.items()):
            result = await assess_doc(
                doc_id=doc_id_str,
                source=fd.candidate.source,
                title=fd.candidate.title,
                whole_doc=fd.cleaned_text,
                corpus_state=state.corpus_state,
                settings=deps.settings,
                sinks=deps.sinks,
                llm_model=deps.llm_model,
            )
            if store is not None and deps.sinks.round_ is not None:
                store.record_doc_assessment(
                    deps.sinks.round_,
                    doc_id=doc_id_str,
                    source=fd.candidate.source,
                    usefulness=result.usefulness,
                    decision=result.decision,
                    reason=result.reason,
                    summary=result.summary or None,
                )

            if result.decision == "skip":
                skipped += 1
                state.fetched_by_doc_id.pop(doc_id_str, None)
                if store is not None:
                    store.update_fetched_doc_outcome(
                        row_id,
                        outcome="extraction_skipped_low_usefulness",
                        outcome_detail=f"score={result.usefulness:.2f}; {result.reason}",
                    )
                continue

            state.corpus_state.entities_seen = _union(
                state.corpus_state.entities_seen, result.entities_seen
            )
            state.corpus_state.subtopics_covered = _union(
                state.corpus_state.subtopics_covered, result.subtopics_covered
            )
            state.corpus_state.subtopics_missing = _union(
                state.corpus_state.subtopics_missing, result.subtopics_missing
            )
            state.corpus_state.borders = _union(
                state.corpus_state.borders, result.borders
            )

        if skipped:
            state.deltas.new_candidates = [
                fd[0].candidate for fd in state.fetched_by_doc_id.values()
            ]
            state.deltas.docs_skipped += skipped
            state.deltas.docs_fetched = max(0, state.deltas.docs_fetched - skipped)

        _log.info(
            "round %d usefulness gate: %d kept, %d skipped_low_usefulness",
            rn,
            len(state.fetched_by_doc_id),
            skipped,
        )


# ── graph-level step: per-doc extraction + consolidation ─────────────────


async def extract_and_consolidate_step(
    ctx: StepContext[RoundGraphState, RoundGraphDeps, None],
) -> None:
    """Per-doc extract → map → gate → merge → embed → resolve loop.

    For each fetched doc sequentially: extract free-form (session or
    parallel-per-chunk) → (chunked only) reconcile → per-doc Mapper loop
    canonicalizes the doc's whole raw-triple set + gates net-new
    canonicals → merge the gated triples → embed → resolve entity
    identity → embed new canonical predicates.
    """

    state, deps = ctx.state, ctx.deps
    store = _require_store(deps)
    round_number = state.corpus_state.round_number

    if not state.fetched_by_doc_id:
        return

    stale = store.embedding_model_check(deps.settings.embedding_model)
    if stale is not None:
        raise RuntimeError(
            f"Existing embeddings use model '{stale}', but settings specify "
            f"'{deps.settings.embedding_model}'. Cannot mix models."
        )

    _log.info("round %d ▶ Extract + Map (%d docs)", round_number, len(state.fetched_by_doc_id))
    t0_consolidate = time.monotonic()

    total_docs = len(state.fetched_by_doc_id)
    for doc_index, (doc_id_str, (fd, row_id)) in enumerate(
        state.fetched_by_doc_id.items()
    ):
        _log.info(
            "round %d doc %d/%d: %s (%s)",
            round_number, doc_index + 1, total_docs,
            doc_id_str, fd.candidate.title,
        )

        # ── extract free-form (+ reconcile in chunked mode) ──────────
        raw_triples, raw_attributes, standalone_entities = await _extract_doc(
            store, deps, state, doc_id_str, fd, row_id, round_number,
        )
        if not raw_triples and not raw_attributes and not standalone_entities:
            continue

        # ── map: canonicalize the doc's relations + attributes ───────
        schema = load_schema_view(store)
        map_result = await map_doc(
            doc_id=doc_id_str,
            doc_title=fd.candidate.title,
            whole_doc=fd.cleaned_text,
            raw_triples=raw_triples,
            raw_attributes=raw_attributes,
            standalone_entities=standalone_entities,
            schema=schema,
            store=store,
            settings=deps.settings,
            sinks=deps.sinks,
            round_number=round_number,
            llm_model=deps.llm_model,
        )

        # ── gate: admit net-new canonicals + write the audit ─────────
        _apply_gate_admissions(store, deps, map_result, round_number)

        # ── merge the gated relations + attributes ───────────────────
        if map_result.mapper_call_id is not None and (
            map_result.relations or map_result.entities
        ):
            store.merge_mapped_relations(
                entities=map_result.entities,
                relations=map_result.relations,
                source_doc_id=doc_id_str,
                mapper_call_id=map_result.mapper_call_id,
                round_number=round_number,
            )
        if map_result.mapper_call_id is not None and map_result.entity_properties:
            store.merge_mapped_attributes(
                attributes=map_result.entity_properties,
                source_doc_id=doc_id_str,
                mapper_call_id=map_result.mapper_call_id,
                round_number=round_number,
            )
        _log.info(
            "round %d doc %s map: %d triples → %d rel (mapped=%d raw=%d), "
            "%d attrs → %d props, dropped=%d; +%d predicates +%d types +%d properties",
            round_number, doc_id_str, len(raw_triples), len(map_result.relations),
            sum(1 for r in map_result.relations if r.mapped_flag),
            sum(1 for r in map_result.relations if not r.mapped_flag),
            len(raw_attributes), len(map_result.entity_properties),
            map_result.dropped,
            len(map_result.admitted_predicates), len(map_result.admitted_entity_types),
            len(map_result.admitted_property_types),
        )

        # ── embed (chunks + entities) ────────────────────────────────
        await _embed_doc(store, deps)

        # ── resolve (this doc's entities vs cumulative) ──────────────
        doc_entity_names = {name for name, _type, _chunk in map_result.entities}
        committed = await resolve(
            store=store,
            round_number=round_number,
            entities_touched_this_round=doc_entity_names,
            settings=deps.settings,
            sinks=deps.sinks,
            llm_model=deps.llm_model,
        )
        _log.info(
            "round %d doc %s resolve: entities_touched=%d → %d merges (T1=%d T2=%d T3=%d)",
            round_number, doc_id_str, len(doc_entity_names),
            len(committed),
            sum(1 for p in committed if p.tier == 1),
            sum(1 for p in committed if p.tier == 2),
            sum(1 for p in committed if p.tier == 3),
        )

        # ── embed predicates + properties (new canonicals the gate admitted) ──
        await _embed_doc_predicates(store, deps, round_number, doc_id_str)
        await _embed_doc_properties(store, deps)

        # ── embed canonical relations + attributes as facts ─────
        await _embed_doc_facts(store, deps)

    _log.info(
        "round %d ◀ Extract + Map (%.1fs)",
        round_number, time.monotonic() - t0_consolidate,
    )


# ── per-doc helpers (called from extract_and_consolidate_step) ───────────


def _apply_gate_admissions(
    store: TraceStore,
    deps: RoundGraphDeps,
    map_result: MapDocResult,
    round_number: int,
) -> None:
    """Apply the gate's net-new canonical admissions to ``kg_schema`` and write
    the ``schema_changes`` audit. Attributed to the doc's Mapper call."""

    call_id = map_result.mapper_call_id
    if call_id is None:
        return
    for name in map_result.admitted_predicates:
        store.record_kg_schema_type(
            name=name, kind="relation_type",
            added_at_call_id=call_id, added_at_round=round_number,
        )
    for name in map_result.admitted_entity_types:
        store.record_kg_schema_type(
            name=name, kind="entity_type",
            added_at_call_id=call_id, added_at_round=round_number,
        )
    for name in map_result.admitted_property_types:
        store.record_kg_schema_type(
            name=name, kind="property_type",
            added_at_call_id=call_id, added_at_round=round_number,
        )
    if deps.sinks.round_ is not None:
        for event, type_kind, name, reason in map_result.events:
            store.record_schema_change(
                deps.sinks.round_,
                agent_call_id=call_id,
                event=event,
                type_kind=type_kind,
                name=name,
                decision_reason=reason,
            )


async def _extract_doc(
    store: TraceStore,
    deps: RoundGraphDeps,
    state: RoundGraphState,
    doc_id_str: str,
    fd: ChunkedDoc,
    row_id: int,
    round_number: int,
) -> tuple[
    list[RawTriple], list[RawAttribute], list[tuple[str, str | None, int | None]]
]:
    """Extract free-form for one doc (session or parallel-per-chunk), recording
    chunks, and (chunked multi-chunk only) reconcile cross-chunk relations.

    Returns ``(raw_triples, raw_attributes, standalone_entities)`` for the Mapper:
    ``raw_triples`` is every extracted + reconciled relation (entity→entity);
    ``raw_attributes`` is every extracted entity→value property (the
    attribute lane); ``standalone_entities`` is the extracted entities that appear
    in NO triple. Chunks are produced by the adapter's ``fetch`` (``fd.chunks``)."""

    chunks = fd.chunks
    if not chunks:
        _log.info("round %d doc %s: 0 chunks, skipping", round_number, doc_id_str)
        return [], [], []

    session = deps.settings.extract_mode == "session"
    if session and len(fd.cleaned_text) > SESSION_DOC_BUDGET:
        _log.info(
            "round %d doc %s: %d chars over session budget %d — chunked fallback",
            round_number, doc_id_str, len(fd.cleaned_text), SESSION_DOC_BUDGET,
        )
        session = False
    if session and len(chunks) <= 1:
        # Single-chunk doc: whole_doc == the only chunk, so session mode would
        # just send the doc twice with no other fork to cache against.
        session = False
    whole_doc = fd.cleaned_text if session else None

    async def _one(chunk):  # type: ignore[no-untyped-def]
        chunk_id = store.record_chunk(
            fetched_doc_id=row_id,
            doc_id=doc_id_str,
            chunk_index=chunk.chunk_index,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            text_chars=len(chunk.text),
        )
        result = await extract(
            doc_id=doc_id_str,
            doc_title=fd.candidate.title,
            doc_text=chunk.text,
            llm_model=deps.llm_model,
            settings=deps.settings,
            sinks=deps.sinks,
            chunk_id=chunk_id,
            whole_doc=whole_doc,
        )
        return chunk.chunk_index, chunk_id, result

    if session and len(chunks) > 1:
        # Warm-then-fan-out: fork 0 runs ALONE first to warm the shared whole_doc
        # prefix cache (cold), then forks 1..N fan out concurrently and hit the
        # warm cache — preserving the ~91% prefix-cache hit while parallelizing
        # the latency-bound round-trips (a cold concurrent fan-out races the
        # cache and some forks miss). A semaphore caps in-flight forks so a
        # many-chunk doc doesn't overwhelm the LLM API.
        warm = await _one(chunks[0])
        sem = asyncio.Semaphore(_EXTRACT_FANOUT)

        async def _capped(c):  # type: ignore[no-untyped-def]
            async with sem:
                return await _one(c)

        rest = await asyncio.gather(*[_capped(c) for c in chunks[1:]])
        raw_results = [warm, *rest]
    elif session:
        raw_results = [await _one(chunks[0])]
    else:
        raw_results = await asyncio.gather(*[_one(c) for c in chunks])
    raw_results_sorted = sorted(raw_results, key=lambda r: r[0])

    total_entities = sum(len(r[2].entities) for r in raw_results_sorted)
    total_relations = sum(len(r[2].relations) for r in raw_results_sorted)

    raw_triples: list[RawTriple] = []
    raw_attributes: list[RawAttribute] = []
    triple_names: set[str] = set()
    entity_chunk: dict[str, tuple[str | None, int | None]] = {}

    def _add_relation(
        subject: str, predicate: str, obj: str, chunk_id: int | None,
        source_quote: str = "",
    ) -> None:
        if subject.strip().lower() == obj.strip().lower():
            return  # self-loop — no graph signal
        raw_triples.append(RawTriple(subject, predicate, obj, chunk_id, source_quote))
        triple_names.add(subject.strip().lower())
        triple_names.add(obj.strip().lower())

    for _chunk_index, chunk_id, result in raw_results_sorted:
        state.deltas.new_extractions.append(result)
        for entity in result.entities:
            entity_chunk.setdefault(entity.name, (entity.type, chunk_id))
        for relation in result.relations:
            _add_relation(
                relation.subject, relation.predicate, relation.object, chunk_id,
                relation.source_quote,
            )
        for attr in result.attributes:
            raw_attributes.append(RawAttribute(
                entity=attr.entity, property=attr.property, value=attr.value,
                unit=attr.unit, condition=attr.condition, chunk_id=chunk_id,
                source_quote=attr.source_quote,
            ))

    reconciler_extra = ""
    # Session mode extracts each chunk with the whole doc in context, so
    # cross-section relations emerge natively — the Reconciler (which works
    # from a synopsis + per-chunk entity lists) is the chunked-fallback path.
    # Its relations join the raw-triple set so the Mapper canonicalizes them too.
    if len(raw_results_sorted) > 1 and not session:
        per_chunk_entities = [
            (idx, list(res.entities)) for idx, _, res in raw_results_sorted
        ]
        rec_schema = load_schema_view(store)
        reconciled = await reconcile(
            doc_id=doc_id_str,
            doc_title=fd.candidate.title,
            doc_synopsis=fd.cleaned_text,
            per_chunk_entities=per_chunk_entities,
            schema=rec_schema,
            llm_model=deps.llm_model,
            settings=deps.settings,
            sinks=deps.sinks,
        )
        if reconciled.relations:
            state.deltas.new_extractions.append(
                ExtractionResult(entities=[], relations=list(reconciled.relations))
            )
            for relation in reconciled.relations:
                _add_relation(relation.subject, relation.predicate, relation.object, None)
            reconciler_extra = f" (reconciler: +{len(reconciled.relations)} rels)"

    standalone_entities = [
        (name, etype, chunk_id)
        for name, (etype, chunk_id) in entity_chunk.items()
        if name.strip().lower() not in triple_names
    ]

    _log.info(
        "round %d doc %s [%s]: %d chunks → entities=%d relations=%d attributes=%d%s "
        "(raw_triples=%d, raw_attributes=%d, standalone_entities=%d)",
        round_number, doc_id_str, "session" if session else "chunked", len(chunks),
        total_entities, total_relations, len(raw_attributes), reconciler_extra,
        len(raw_triples), len(raw_attributes), len(standalone_entities),
    )
    return raw_triples, raw_attributes, standalone_entities


async def _embed_doc(
    store: TraceStore,
    deps: RoundGraphDeps,
) -> None:
    """Embed any new chunks + entities that don't yet have embeddings."""


    items: list[tuple[str, str, str]] = []

    doc_text_cache: dict[str, str | None] = {}
    for row in store.chunks_without_embeddings():
        chunk_id, doc_id, _chunk_idx, char_start, char_end = (
            row[0], row[1], row[2], row[3], row[4],
        )
        if doc_id not in doc_text_cache:
            doc_text_cache[doc_id] = store.get_document_text(doc_id)
        full_text = doc_text_cache[doc_id]
        if full_text is not None:
            text = full_text[char_start:char_end]
            if text.strip():
                items.append(("chunk", str(chunk_id), text))

    for row in store.entities_without_embeddings():
        entity_id, name = row[0], row[1]
        items.append(("entity", str(entity_id), name))

    if items:
        await embed_and_store(store, items, deps.settings)


async def _embed_doc_predicates(
    store: TraceStore,
    deps: RoundGraphDeps,
    round_number: int,
    doc_id_str: str,
) -> None:
    """Embed any new predicates that don't yet have embeddings."""


    pred_names = store.predicates_without_embeddings()
    if not pred_names:
        return
    items = [("predicate", name, name) for name in pred_names]
    await embed_and_store(store, items, deps.settings)


async def _embed_doc_properties(
    store: TraceStore,
    deps: RoundGraphDeps,
) -> None:
    """Embed any new property-type NAMES that don't yet have a 'property'
    embedding — the attribute-lane analogue of ``_embed_doc_predicates``. Feeds
    the VocabResolver's property clustering (and property-name retrieval)."""

    prop_names = store.properties_without_embeddings()
    if not prop_names:
        return
    items = [("property", name, name) for name in prop_names]
    await embed_and_store(store, items, deps.settings)


async def _embed_doc_facts(store: TraceStore, deps: RoundGraphDeps) -> None:
    """Embed canonical relations + attributes as natural-language facts
    — 'electrolysis of water produces hydrogen',
    'AISI Ni42-300 overpotential 491 mV (HER)' — so the hypothesis layer can
    retrieve FACTS by meaning, not just entity names / predicate labels. Embedded
    after resolve so the sentences carry the survivor entity names."""

    items = [
        ("relation", str(rid), fact)
        for rid, fact in store.relations_without_embeddings()
    ]
    items += [
        ("attribute", str(pid), fact)
        for pid, fact in store.entity_properties_without_embeddings()
    ]
    if items:
        await embed_and_store(store, items, deps.settings)


# ── graph-level step: finalize ───────────────────────────────────────────


async def finalize_step(
    ctx: StepContext[RoundGraphState, RoundGraphDeps, None],
) -> RoundOutput:
    """Terminal node: package the round's work deltas. The two end-of-round
    reviews run in the orchestrator's Phase 2 (after the metrics row is
    written), not in the graph."""

    return RoundOutput(deltas=ctx.state.deltas)
