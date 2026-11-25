"""Deterministic helper bodies that the per-round step functions call into
(search + rank, fetch + persist, small query/state utilities)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.common.trace import utc_now_iso
from memex.kg_intelligence.ids import Source, doc_id
from memex.kg_intelligence.openalex import OpenAlexClient
from memex.kg_intelligence.persistence import (
    DocMetadata,
    IngestionMetadata,
    OpenAlexMetadata,
    QueryProvenance,
    has_document,
    make_context_excerpt,
    persist_document,
)
from memex.kg_intelligence.ranker import rank
from memex.kg_intelligence.sources.common.base import (
    ChunkedDoc,
    SourceAdapter,
    SourceCandidate,
)

if TYPE_CHECKING:
    from memex.kg_intelligence.pipeline import LoopConfig

_log = logging.getLogger(__name__)


# ── plan + search + rank ──────────────────────────────────────────────────


async def _plan_search_and_rank(
    *,
    ranked_topics: list[str],
    adapter_relevance: dict[str, str],
    adapters: dict[str, SourceAdapter],
    cfg: LoopConfig,
    settings: Settings,
    sinks: Any,
    llm_model: Model | None,
) -> dict[Source, list[SourceCandidate]]:
    """Fan ``plan_and_search`` out across active sources, then rank per source.

    Each adapter turns the round's ranked topics into its own source-tailored
    queries (an LLM call) and runs them — per-source pacing + the relevance
    posture (throttled sources sit out a low-relevance round) live inside the
    adapter. Different adapters run concurrently. Ranking + selection stay
    deterministic here; the global priority-weighted Selector picks afterwards.
    """

    async def _one(
        source: Source, adapter: SourceAdapter,
    ) -> tuple[Source, list[SourceCandidate]]:
        relevance = adapter_relevance.get(source, "high")
        try:
            candidates = await adapter.plan_and_search(
                ranked_topics,
                relevance=relevance,  # type: ignore[arg-type]
                limit=cfg.per_source_search,
                settings=settings,
                sinks=sinks,
                llm_model=llm_model,
            )
        except Exception as exc:
            _log.warning("plan_and_search failed (%s): %s", source, exc)
            candidates = []
        joined_query = " ".join(ranked_topics)
        return source, rank(candidates, joined_query, top_k=cfg.per_source_search)

    per_source_results = await asyncio.gather(
        *(_one(src, adapter) for src, adapter in adapters.items())
    )
    return dict(per_source_results)


def _filter_already_ingested(
    ranked_by_source: dict[Source, list[SourceCandidate]],
    store: Any,
) -> tuple[dict[Source, list[SourceCandidate]], int, dict[str, list[int]]]:
    """Drop candidates whose ``doc_id`` is already in ``documents`` BEFORE the
    Selector, so an already-ingested doc never consumes a selection slot.

    This is the only document-level dedup the pipeline owns (query-dedup is
    gone — re-running a metadata search is cheap, and a truncated-but-never-
    ingested doc must stay eligible so a later round's shifted query can
    resurface it). Returns ``(filtered_pool, n_dropped, by_topic)`` where
    ``by_topic`` maps each candidate topic → ``[surfaced, already_seen]`` (the
    strategist's per-theme saturation signal; a candidate counts toward every
    topic it was searched for).
    """

    if store is None:
        return ranked_by_source, 0, {}

    filtered: dict[Source, list[SourceCandidate]] = {}
    dropped = 0
    by_topic: dict[str, list[int]] = {}
    for source, candidates in ranked_by_source.items():
        kept: list[SourceCandidate] = []
        for candidate in candidates:
            is_seen = has_document(store, doc_id(source, candidate.canonical))
            for topic in candidate.topics or ["(untagged)"]:
                slot = by_topic.setdefault(topic, [0, 0])
                slot[0] += 1
                if is_seen:
                    slot[1] += 1
            if is_seen:
                dropped += 1
                continue
            kept.append(candidate)
        filtered[source] = kept
    return filtered, dropped, by_topic


def _flatten_pool(
    ranked_by_source: dict[Source, list[SourceCandidate]],
) -> list[SourceCandidate]:
    """Concatenate the per-source ranked lists into one flat pool for the
    global Selector, dropping exact ``doc_id`` repeats within a source (the
    same canonical surfaced by two queries) so it can't claim two pool slots."""

    pool: list[SourceCandidate] = []
    seen: set[str] = set()
    for source, candidates in ranked_by_source.items():
        for candidate in candidates:
            key = doc_id(source, candidate.canonical)
            if key in seen:
                continue
            seen.add(key)
            pool.append(candidate)
    return pool


def _group_by_source(
    candidates: list[SourceCandidate],
) -> dict[Source, list[SourceCandidate]]:
    """Regroup the Selector's flat picks back into the per-source shape
    ``_fetch_and_persist`` consumes (fetch batches per adapter)."""

    grouped: dict[Source, list[SourceCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.source, []).append(candidate)
    return grouped


# ── fetch + persist ───────────────────────────────────────────────────────


async def _fetch_and_persist(
    *,
    selected_by_source: dict[Source, list[SourceCandidate]],
    adapters: dict[str, SourceAdapter],
    openalex: OpenAlexClient | None,
    base_query: str,
    force: bool,
    settings: Settings,
    sinks: Any,
) -> tuple[dict[str, tuple[ChunkedDoc, int]], dict[str, int]]:
    """Fetch + enrich + persist every selected candidate.

    Each source's selected candidates are fetched in one ``adapter.fetch``
    batch (download/parse/chunk + pacing live inside the adapter); the
    pipeline then enriches, persists (kg.db ``documents``), and records a
    ``fetched_docs`` row per candidate. Returns ``(fetched_by_doc_id,
    counters)`` where ``fetched_by_doc_id`` maps ``doc_id`` →
    ``(ChunkedDoc, row id)``.
    """

    store = sinks.store

    jobs_by_source: dict[Source, list[tuple[SourceCandidate, int]]] = {}
    skipped = 0
    for source, candidates in selected_by_source.items():
        for position, candidate in enumerate(candidates):
            d_id = doc_id(source, candidate.canonical)
            if not force and store is not None and has_document(store, d_id):
                skipped += 1
                _record_fetch_outcome(
                    sinks,
                    candidate,
                    outcome="skipped_duplicate",
                    outcome_detail=f"already persisted in kg.db for {d_id}",
                    doc_id=d_id,
                    ranker_position=position,
                )
                continue
            jobs_by_source.setdefault(source, []).append((candidate, position))

    async def _persist_one(
        candidate: SourceCandidate, fetched: ChunkedDoc, position: int,
    ) -> tuple[str, ChunkedDoc, int]:
        d_id = doc_id(candidate.source, candidate.canonical)
        openalex_md: OpenAlexMetadata | None = None
        openalex_ok = False
        if openalex is not None:
            try:
                openalex_md = await asyncio.to_thread(openalex.enrich, candidate)
                openalex_ok = openalex_md is not None
            except Exception as exc:
                _log.warning("openalex enrich failed for %s: %s", d_id, exc)
                openalex_md = None

        meta = DocMetadata(
            doc_id=d_id,
            source=candidate.source,
            fetched_at=utc_now_iso(),
            query=QueryProvenance(base=base_query, expanded=""),
            title=candidate.title,
            authors=list(candidate.raw_signals.get("authors") or []),
            url=candidate.url,
            context_excerpt=make_context_excerpt(fetched.cleaned_text),
            ingestion_metadata=IngestionMetadata(
                parse_method=fetched.parse_method,
                **fetched.extra_metadata,
            ),
            openalex_metadata=openalex_md,
        )
        if store is not None:
            persist_document(store, meta, fetched.cleaned_text)

        row_id = _record_fetch_outcome(
            sinks, candidate,
            outcome="selected_and_fetched",
            outcome_detail=None,
            doc_id=d_id,
            ranker_position=position,
            parse_method=fetched.parse_method,
            cleaned_text_chars=len(fetched.cleaned_text),
            openalex_enriched=openalex_ok,
        )
        return d_id, fetched, row_id or 0

    async def _fetch_source(
        source: Source, source_jobs: list[tuple[SourceCandidate, int]],
    ) -> list[tuple[str, ChunkedDoc, int]]:
        adapter = adapters.get(source)
        if adapter is None:
            for candidate, position in source_jobs:
                _record_fetch_outcome(
                    sinks, candidate,
                    outcome="fetch_failed",
                    outcome_detail=f"no adapter registered for {source}",
                    ranker_position=position,
                )
            return []

        candidates = [c for c, _ in source_jobs]
        try:
            docs = await asyncio.to_thread(adapter.fetch, candidates, settings=settings)
        except Exception as exc:
            _log.warning("fetch failed (%s, %d candidates): %s", source, len(candidates), exc)
            docs = []
        docs_by_id = {
            doc_id(d.candidate.source, d.candidate.canonical): d for d in docs
        }

        results: list[tuple[str, ChunkedDoc, int]] = []
        for candidate, position in source_jobs:
            d_id = doc_id(candidate.source, candidate.canonical)
            fetched = docs_by_id.get(d_id)
            if fetched is None:
                _record_fetch_outcome(
                    sinks, candidate,
                    outcome="fetch_failed",
                    outcome_detail="adapter returned no doc (download/parse failed or empty)",
                    doc_id=d_id,
                    ranker_position=position,
                )
                continue
            results.append(await _persist_one(candidate, fetched, position))
        return results

    per_source_results = await asyncio.gather(
        *(_fetch_source(src, sj) for src, sj in jobs_by_source.items())
    )

    fetched_by_doc_id: dict[str, tuple[ChunkedDoc, int]] = {}
    fetched_count = 0
    for source_results in per_source_results:
        for d_id, fetched, row_id in source_results:
            fetched_by_doc_id[d_id] = (fetched, row_id)
            fetched_count += 1

    selected_total = sum(len(sj) for sj in jobs_by_source.values())
    return fetched_by_doc_id, {
        "fetched": fetched_count,
        "skipped": skipped,
        "failed": selected_total - fetched_count,
    }


def _record_fetch_outcome(
    sinks: Any,
    candidate: SourceCandidate,
    *,
    outcome: str,
    outcome_detail: str | None,
    doc_id: str | None = None,
    ranker_position: int | None = None,
    parse_method: str | None = None,
    cleaned_text_chars: int | None = None,
    openalex_enriched: bool = False,
) -> int | None:
    if sinks.store is None or sinks.round_ is None:
        return None
    raw = candidate.raw_signals or {}
    score = raw.get("rank_score")
    return sinks.store.record_fetched_doc(
        sinks.round_,
        source=candidate.source,
        query=candidate.matched_query,
        url=candidate.url,
        title=candidate.title,
        doc_id=doc_id,
        outcome=outcome,
        outcome_detail=outcome_detail,
        ranker_position=ranker_position,
        ranker_score=float(score) if score is not None else None,
        parse_method=parse_method,
        cleaned_text_chars=cleaned_text_chars,
        openalex_enriched=openalex_enriched,
    )


# ── small helpers ────────────────────────────────────────────────────────


def _union(existing: list[str], incoming: list[str]) -> list[str]:
    """Order-preserving union; existing items first, then new ones."""

    seen = {x.lower() for x in existing}
    out = list(existing)
    for item in incoming:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


__all__ = [
    "_fetch_and_persist",
    "_filter_already_ingested",
    "_flatten_pool",
    "_group_by_source",
    "_plan_search_and_rank",
    "_record_fetch_outcome",
    "_union",
]
