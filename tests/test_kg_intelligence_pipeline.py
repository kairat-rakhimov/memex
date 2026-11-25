"""Integration tests for the kg-loop pipeline orchestrator.

`FakeAdapter` provides canned candidates + bodies; agents run in
SAMPLE_MODE so they're deterministic without calling the LLM. The
`round_review` stop path is exercised by monkey-patching the orchestrator's
`round_review` for that one test — agent unit tests cover the real
PydanticAI path.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from memex.common.config import Settings
from memex.common.trace import TraceWriter
from memex.kg_intelligence.agents import AgentCallSinks, AgentHardFailure
from memex.kg_intelligence.agents.schemas import (
    GoalDirective,
    GoalVerdict,
    RoundReviewResult,
    RoundStatus,
)
from memex.kg_intelligence.ids import Source
from memex.kg_intelligence.persistence import (
    OpenAlexMetadata,
    read_document_metadata,
)
from memex.kg_intelligence.pipeline import (
    KG_JSON_FILE,
    LoopConfig,
    run_pipeline,
)
from memex.kg_intelligence.sources.common.base import (
    AdapterDescription,
    ChunkedDoc,
    SourceCandidate,
)
from memex.kg_intelligence.sources.common.chunker import chunk_text
from memex.kg_intelligence.sources.common.query_planner import (
    BaseQueryPlanner,
    plan_and_search,
)
from memex.kg_intelligence.state import CorpusState, SelectedTitle
from memex.observability import TraceStore

# ── shared fakes ──────────────────────────────────────────────────────────


class FakeAdapter:
    """Adapter stub: returns pre-seeded candidates per query, canned text on fetch."""

    def __init__(
        self,
        source: Source,
        *,
        candidates_by_query: dict[str, list[SourceCandidate]] | None = None,
        body_for: Callable[[SourceCandidate], str] | None = None,
    ) -> None:
        self.source = source
        self.name = source
        self.candidates_by_query = candidates_by_query or {}
        self.body_for = body_for or (lambda c: f"body of {c.title}")
        self.search_calls: list[str] = []
        self.fetch_calls: list[str] = []
        self._planner = BaseQueryPlanner(source_name=source, guidance="test double")

    @property
    def description(self) -> AdapterDescription:
        return AdapterDescription(
            name=self.source,
            summary=f"{self.source} test double",
            posture="relaxed",
            preferred_doc_cap=3,
            metadata_fields=[],
        )

    def probe(self) -> bool:
        return True

    async def plan_and_search(
        self, ranked_topics, *, relevance, limit, settings, sinks, llm_model=None,
    ) -> list[SourceCandidate]:
        return await plan_and_search(
            planner=self._planner,
            search_fn=self.search,
            posture=self.description.posture,
            ranked_topics=ranked_topics,
            relevance=relevance,
            limit=limit,
            settings=settings,
            sinks=sinks,
            llm_model=llm_model,
        )

    def search(self, queries: list[str], *, limit: int = 20) -> list[SourceCandidate]:
        out: list[SourceCandidate] = []
        for query in queries:
            self.search_calls.append(query)
            if query in self.candidates_by_query:
                out.extend(self.candidates_by_query[query])
                continue
            out.append(
                SourceCandidate(
                    source=self.source,
                    canonical=f"{self.source}-{query.replace(' ', '-')}",
                    url=f"https://example/{self.source}/{query.replace(' ', '-')}",
                    title=f"{self.source.title()} result for {query}",
                    abstract=f"Abstract about {query}.",
                    raw_signals={},
                )
            )
        return out

    def fetch(
        self, candidates: list[SourceCandidate], *, settings: Settings
    ) -> list[ChunkedDoc]:
        docs: list[ChunkedDoc] = []
        for candidate in candidates:
            self.fetch_calls.append(candidate.canonical)
            text = self.body_for(candidate)
            docs.append(
                ChunkedDoc(
                    candidate=candidate,
                    cleaned_text=text,
                    parse_method="test",
                    chunks=chunk_text(text),
                    extra_metadata={},
                )
            )
        return docs


def _settings(*, sample_mode: bool = True, extract_mode: str = "session") -> Settings:
    return Settings(
        api_key="test-key",
        base_url="https://llm.test/v1",
        chat_model="gpt-test",
        reasoning_model="gpt-test",
        embedding_model="emb-test",
        embedding_dimension=3072,
        host="0.0.0.0",
        port=8000,
        run_timeout_seconds=300,
        log_dir=Path("logs"),
        log_level="INFO",
        sample_mode=sample_mode,
        contact=None,
        extract_mode=extract_mode,
    )


@pytest.fixture
def trace(tmp_path: Path) -> TraceWriter:
    return TraceWriter(request_id="pipeline-test", log_dir=tmp_path / "_trace")


# ── end-to-end one-round run (SAMPLE_MODE so agents are deterministic) ────


async def test_pipeline_runs_one_round_and_persists_artefacts(
    tmp_path: Path, trace: TraceWriter
) -> None:
    adapters: dict[Source, FakeAdapter] = {
        "arxiv": FakeAdapter("arxiv"),
        "wikipedia": FakeAdapter("wikipedia"),
        "gdelt": FakeAdapter("gdelt"),
    }
    kg_root = tmp_path / "kg-test"

    outcome = await run_pipeline(
        base_query="green hydrogen",
        kg_root=kg_root,
        adapters=adapters,  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=2, per_source_search=5, max_rounds=1),
        resume=False,
    )

    assert outcome.rounds_completed == 1
    assert outcome.stop_reason == "max_rounds"

    # kg.db is the canonical shipped artefact; loop_state row carries
    # the cumulative CorpusState for resume.
    assert (kg_root / "kg.db").exists()
    store = TraceStore.open(kg_root)
    try:
        blob = store.read_loop_state()
        assert blob is not None
        loaded = CorpusState.model_validate_json(blob)
        assert loaded.round_number >= 1
    finally:
        store.close()

    # kg.json export is OFF by default; toggle the flag to opt in.
    assert not (kg_root / KG_JSON_FILE).exists()

    # docs are persisted in kg.db (the documents table), not on disk
    assert not (kg_root / "sources").exists()
    store = TraceStore.open(kg_root)
    try:
        doc_ids = store.document_ids()
        assert doc_ids, "expected at least one fetched doc"
        sample_doc = read_document_metadata(store, doc_ids[0])
    finally:
        store.close()
    assert sample_doc.source in {"arxiv", "wikipedia", "gdelt"}
    assert sample_doc.context_excerpt
    assert sample_doc.ingestion_metadata.parse_method == "test"


async def test_pipeline_writes_kg_snapshot_per_round(
    tmp_path: Path, trace: TraceWriter
) -> None:
    """Per-round snapshot must populate kg_entities + kg_relations during the
    build, not just at finalise. Powers the trace UI's graph view on
    in-progress builds."""

    kg_root = tmp_path / "kg-snap"
    await run_pipeline(
        base_query="green hydrogen",
        kg_root=kg_root,
        adapters={"arxiv": FakeAdapter("arxiv")},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=2, per_source_search=5, max_rounds=1),
        resume=False,
    )

    store = TraceStore.open(kg_root)
    try:
        [build] = store.list_builds()
        entities = store.top_entities(limit=100)
        assert entities, "per-round snapshot did not populate kg_entities"
        # first_seen_round was set on first snapshot; should be <= last_seen_round
        for e in entities:
            assert e["first_seen_round"] is not None
            assert e["last_seen_round"] is not None
            assert e["first_seen_round"] <= e["last_seen_round"]

        # graph_data round-trips: nodes + entity_types meta present
        g = store.graph_data(min_mentions=1, include_isolated=True)
        assert len(g["nodes"]) == len(entities)
        assert g["entity_types"]
    finally:
        store.close()


async def test_per_chunk_wiring_populates_chunks_refs_and_source_chunk_id(
    tmp_path: Path, trace: TraceWriter
) -> None:
    """Per-chunk extraction must populate `chunks` rows > docs rows on
    long docs, `entity_chunk_refs` for each entity-chunk pair, and
    `kg_relations.source_chunk_id` for per-chunk-extracted relations.
    Reconciler must run on multi-chunk docs (visible as a Reconciler
    trace event).
    """


    def long_body(candidate: SourceCandidate) -> str:
        # 3 ~1500-char paragraphs → chunker packs them into ≥ 2 chunks.
        para = ("Iridium catalyzes the oxygen evolution reaction in PEM "
                "electrolyzers under acidic conditions. " * 30)
        assert len(para) >= 1500
        return "\n\n".join([para, para, para])

    adapters: dict[Source, FakeAdapter] = {
        "arxiv": FakeAdapter("arxiv", body_for=long_body),
    }
    kg_root = tmp_path / "kg-chunked"

    await run_pipeline(
        base_query="green hydrogen",
        kg_root=kg_root,
        adapters=adapters,  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(extract_mode="chunked"),  # this test exercises the per-chunk + Reconciler path
        trace=trace,
        config=LoopConfig(selector_budget=2, per_source_search=5, max_rounds=1),
        resume=False,
    )

    store = TraceStore.open(kg_root)
    try:
        [build] = store.list_builds()
        # Chunks should outnumber docs — every long doc splits.
        n_chunks = store._conn.execute(
            "SELECT COUNT(*) FROM chunks c "
            "JOIN fetched_docs fd ON fd.id = c.fetched_doc_id "
            "JOIN rounds r ON r.id = fd.round_id "
            "WHERE r.build_id = ?",
            (build["id"],),
        ).fetchone()[0]
        n_docs = store._conn.execute(
            "SELECT COUNT(*) FROM fetched_docs fd "
            "JOIN rounds r ON r.id = fd.round_id "
            "WHERE r.build_id = ? AND fd.outcome = 'selected_and_fetched'",
            (build["id"],),
        ).fetchone()[0]
        assert n_docs >= 1, "expected at least one fetched doc"
        assert n_chunks > n_docs, (
            f"expected chunks ({n_chunks}) > docs ({n_docs}); "
            f"per-chunk extraction not splitting long docs"
        )

        # entity_chunk_refs populated for at least one entity-chunk pair.
        # kg_entities is cumulative across invocations,
        # so the count is over the whole kg.db.
        n_refs = store._conn.execute(
            "SELECT COUNT(*) FROM entity_chunk_refs"
        ).fetchone()[0]
        assert n_refs > 0, "entity_chunk_refs not populated"

        # kg_relations.source_chunk_id wiring: column is populated for
        # per-chunk Extractor relations and NULL for Reconciler-emitted
        # ones. In SAMPLE_MODE the Extractor emits only entities (no
        # relations), so all rows come from Reconciler with NULL — but
        # the column itself must exist and be queryable end-to-end.
        relations = store._conn.execute(
            "SELECT source_chunk_id FROM kg_relations"
        ).fetchall()
        # If any non-NULL chunk_id was written, it must reference a real
        # chunks row.
        for (cid,) in relations:
            if cid is None:
                continue
            exists = store._conn.execute(
                "SELECT 1 FROM chunks WHERE id = ?", (cid,)
            ).fetchone()
            assert exists is not None, f"dangling source_chunk_id={cid}"

        # Reconciler emitted at least one trace event (it runs on
        # multi-chunk docs in SAMPLE_MODE).
        agent_calls = store._conn.execute(
            "SELECT agent_name FROM agent_calls "
            "JOIN rounds r ON r.id = agent_calls.round_id "
            "WHERE r.build_id = ?",
            (build["id"],),
        ).fetchall()
        agents_seen = {row[0] for row in agent_calls}
        assert "Reconciler" in agents_seen, (
            f"Reconciler did not run; agents seen: {sorted(agents_seen)}"
        )
    finally:
        store.close()


async def test_pipeline_passes_openalex_metadata_through(
    tmp_path: Path, trace: TraceWriter
) -> None:
    adapters: dict[Source, FakeAdapter] = {"arxiv": FakeAdapter("arxiv")}

    class FakeOpenAlex:
        def enrich(self, candidate: SourceCandidate) -> OpenAlexMetadata | None:
            return OpenAlexMetadata(openalex_id="https://openalex.org/W1", cited_by_count=99)

    kg_root = tmp_path / "kg-oa"
    await run_pipeline(
        base_query="green hydrogen",
        kg_root=kg_root,
        adapters=adapters,  # type: ignore[arg-type]
        openalex=FakeOpenAlex(),  # type: ignore[arg-type]
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=1),
        resume=False,
    )

    store = TraceStore.open(kg_root)
    try:
        doc_ids = store.document_ids()
        assert doc_ids
        sample_doc = read_document_metadata(store, doc_ids[0])
    finally:
        store.close()
    assert sample_doc.openalex_metadata is not None
    assert sample_doc.openalex_metadata.cited_by_count == 99


# ── stopping criteria ────────────────────────────────────────────────────


async def test_stop_zero_progress_floor_after_consecutive_empty_rounds(
    tmp_path: Path, trace: TraceWriter
) -> None:
    """Adapters that never return candidates produce rounds with 0 new
    entities AND 0 new relations; after `zero_progress_rounds` of them the
    deterministic floor halts with `stop_reason='zero_progress_floor'`.
    (Folds in the old `search_exhausted` — 0 docs ⊂ 0 progress.)"""

    empty = FakeAdapter("arxiv")
    # Override search to always return no candidates → 0 docs → 0 new entities.
    empty.search = lambda query, *, limit=20: []  # type: ignore[assignment]

    outcome = await run_pipeline(
        base_query="x",
        kg_root=tmp_path / "kg-empty",
        adapters={"arxiv": empty},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(
            selector_budget=1,
            per_source_search=2,
            max_rounds=1000,
            zero_progress_rounds=3,
        ),
        resume=False,
    )
    assert outcome.stop_reason == "zero_progress_floor"
    assert outcome.rounds_completed == 3
    assert outcome.total_docs == 0


async def test_stop_round_review_stalled_halts(
    tmp_path: Path, trace: TraceWriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the RoundReviewer returns `stalled`, the build halts after the
    round with `stop_reason='round_review_stalled'`."""

    async def fake_round_review(**_kwargs: object) -> tuple[RoundReviewResult, int | None]:
        return RoundReviewResult(status=RoundStatus.stalled, reason="done"), None

    # round_review is called by the orchestrator's Phase-2 helper.
    monkeypatch.setattr("memex.kg_intelligence.pipeline.round_review", fake_round_review)

    outcome = await run_pipeline(
        base_query="x",
        kg_root=tmp_path / "kg-stall",
        adapters={"arxiv": FakeAdapter("arxiv")},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=1000),
        resume=False,
    )
    assert outcome.stop_reason == "round_review_stalled"
    assert outcome.rounds_completed == 1


async def test_stop_max_rounds_ceiling(tmp_path: Path, trace: TraceWriter) -> None:
    """The hard `max_rounds` ceiling halts a healthy build that the
    RoundReviewer would otherwise keep going."""

    outcome = await run_pipeline(
        base_query="green hydrogen",
        kg_root=tmp_path / "kg-maxr",
        adapters={"arxiv": FakeAdapter("arxiv")},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=2),
        resume=False,
    )
    assert outcome.stop_reason == "max_rounds"
    assert outcome.rounds_completed == 2


# ── round_reviews table + two-phase round ────────────────────────────────


async def test_round_reviews_populated_with_metrics_and_stamped_fks(
    tmp_path: Path, trace: TraceWriter
) -> None:
    """Every completed round writes a round_reviews row: deterministic
    metrics + both review FK columns stamped (reviews ran in Phase 2)."""

    kg_root = tmp_path / "kg-rr"
    await run_pipeline(
        base_query="green hydrogen",
        kg_root=kg_root,
        adapters={"arxiv": FakeAdapter("arxiv")},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=2),
        resume=False,
    )

    store = TraceStore.open(kg_root)
    try:
        rows = store._conn.execute(
            "SELECT round_number, new_entities, new_relations, docs_fetched_total, "
            "docs_fetched_per_source, kg_review_call_id, round_review_call_id "
            "FROM round_reviews ORDER BY round_number"
        ).fetchall()
        assert len(rows) == 2, "expected one round_reviews row per completed round"
        for r in rows:
            # Reviews ran → both FKs stamped (no holes, not pending).
            assert r["kg_review_call_id"] is not None
            assert r["round_review_call_id"] is not None
        # Round 1 fetched docs and landed entities.
        assert rows[0]["docs_fetched_total"] >= 1
        assert rows[0]["new_entities"] >= 1
        assert '"arxiv"' in rows[0]["docs_fetched_per_source"]
        # No pending checkpoint remains after a clean build.
        assert store.read_pending_round_review() is None
    finally:
        store.close()


async def test_reviews_pending_resume_reruns_phase2_only(
    tmp_path: Path, trace: TraceWriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Phase-2 review crash leaves a pending round_reviews row; the build
    halts. Resume re-runs the reviews only — no refetch — and clears the
    pending marker."""

    adapter = FakeAdapter("arxiv")
    kg_root = tmp_path / "kg-pending"

    calls = {"n": 0}

    async def flaky_round_review(**_kwargs: object) -> tuple[RoundReviewResult, int | None]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise AgentHardFailure("simulated review crash")
        return RoundReviewResult(status=RoundStatus.continuing), None

    monkeypatch.setattr("memex.kg_intelligence.pipeline.round_review", flaky_round_review)

    outcome1 = await run_pipeline(
        base_query="green hydrogen",
        kg_root=kg_root,
        adapters={"arxiv": adapter},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=1),
        resume=False,
    )
    assert outcome1.stop_reason == "review_failed"
    assert adapter.fetch_calls, "round 1 should have fetched before the review crash"

    store = TraceStore.open(kg_root)
    try:
        pending = store.read_pending_round_review()
        assert pending is not None and pending["round_number"] == 1
    finally:
        store.close()

    adapter.fetch_calls.clear()
    outcome2 = await run_pipeline(
        base_query="green hydrogen",
        kg_root=kg_root,
        adapters={"arxiv": adapter},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=1),
        resume=True,
    )
    assert adapter.fetch_calls == [], "resume must re-run reviews only, never refetch"
    assert outcome2.stop_reason == "max_rounds"

    store = TraceStore.open(kg_root)
    try:
        assert store.read_pending_round_review() is None, "pending marker not cleared"
        row = store._conn.execute(
            "SELECT kg_review_call_id FROM round_reviews WHERE round_number = 1"
        ).fetchone()
        assert row["kg_review_call_id"] is not None
    finally:
        store.close()


async def test_abandoned_goal_persists_to_corpus_state(
    tmp_path: Path, trace: TraceWriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An `abandon` directive from the RoundReviewer lands in
    CorpusState.abandoned_goals (durable, so the next RoundPlanner excludes
    it)."""

    async def abandoning_round_review(**_kwargs: object) -> tuple[RoundReviewResult, int | None]:
        return (
            RoundReviewResult(
                status=RoundStatus.continuing,
                goal_directives=[
                    GoalDirective(theme="dead-end", verdict=GoalVerdict.abandon),
                ],
            ),
            None,
        )

    monkeypatch.setattr(
        "memex.kg_intelligence.pipeline.round_review", abandoning_round_review
    )

    kg_root = tmp_path / "kg-abandon"
    await run_pipeline(
        base_query="green hydrogen",
        kg_root=kg_root,
        adapters={"arxiv": FakeAdapter("arxiv")},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=1),
        resume=False,
    )
    store = TraceStore.open(kg_root)
    try:
        blob = store.read_loop_state()
        assert blob is not None
        state = CorpusState.model_validate_json(blob)
        assert "dead-end" in state.abandoned_goals
    finally:
        store.close()


# ── resume ────────────────────────────────────────────────────────────────


async def test_resume_picks_up_from_persisted_round(tmp_path: Path, trace: TraceWriter) -> None:
    kg_root = tmp_path / "kg-resume"
    kg_root.mkdir()
    store = TraceStore.open(kg_root)
    try:
        build = store.start_build(
            build_run_id="seed", kg_id="kg-resume", base_query="x",
            config={}, sources=["arxiv"], sample_mode=True,
            resume_from_round=1,
        )
        round_ = store.start_round(
            build, round_number=1,
            entities_before=0, relations_before=0,
            schema_entity_types_before=0, schema_relation_types_before=0,
        )
        call = store.begin_agent_call(
            round_, agent_name="_seed", action="seed",
            triggered_by="test_setup", input_summary="",
        )
        store.finish_agent_call(call, status="skipped", duration_ms=0, output_summary="")
        store.merge_extraction(
            entities=[("prior", "concept")],
            relations=[("prior", "related_to", "prior")],
            source_doc_id="doc_prior",
            chunk_id=None,
            agent_call_id=call.agent_call_id,
            round_number=1,
        )
        store.record_kg_schema_type(
            name="concept", kind="entity_type",
            added_at_call_id=call.agent_call_id, added_at_round=1,
        )
        store.record_kg_schema_type(
            name="related_to", kind="relation_type",
            added_at_call_id=call.agent_call_id, added_at_round=1,
        )
        store.write_loop_state(CorpusState(
            round_number=1,
            urls_fetched=["https://prior-1"],
            titles_selected=[SelectedTitle(source="arxiv", title="prior")],
        ).model_dump_json())
    finally:
        store.close()

    outcome = await run_pipeline(
        base_query="x",
        kg_root=kg_root,
        adapters={"arxiv": FakeAdapter("arxiv")},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=2),
        resume=True,
    )
    assert outcome.rounds_completed == 2
    store = TraceStore.open(kg_root)
    try:
        names = {e["canonical_name"] for e in store.top_entities(limit=100)}
        assert "prior" in names, "resume did not carry forward seeded entity"
    finally:
        store.close()


async def test_resume_uses_stored_base_query_and_warns_on_conflict(
    tmp_path: Path,
    trace: TraceWriter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On resume, stored base_query wins. A different value supplied by
    the caller is ignored and logged as a warning — protects against
    accidental cross-topic KG pollution."""

    import logging

    kg_root = tmp_path / "kg-bq"
    kg_root.mkdir()
    store = TraceStore.open(kg_root)
    try:
        store.write_loop_state(
            CorpusState(
                base_query="green hydrogen",
                round_number=1,
                urls_fetched=["https://prior-1"],
                titles_selected=[SelectedTitle(source="arxiv", title="prior")],
            ).model_dump_json()
        )
    finally:
        store.close()

    caplog.set_level(logging.WARNING, logger="memex.kg_intelligence.pipeline")
    outcome = await run_pipeline(
        base_query="fusion energy",  # conflicts with stored "green hydrogen"
        kg_root=kg_root,
        adapters={"arxiv": FakeAdapter("arxiv")},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=2),
        resume=True,
    )
    assert outcome.stop_reason == "max_rounds"

    # The build row carries the stored topic, not the supplied override.
    store = TraceStore.open(kg_root)
    try:
        [build] = store.list_builds()
        assert build["base_query"] == "green hydrogen"
        blob = store.read_loop_state()
        assert blob is not None
        assert CorpusState.model_validate_json(blob).base_query == "green hydrogen"
    finally:
        store.close()

    assert any(
        "Ignoring base_query=" in rec.message and "fusion energy" in rec.message
        for rec in caplog.records
    ), "expected a warning about the ignored base_query override"


async def test_fresh_build_persists_caller_base_query(
    tmp_path: Path, trace: TraceWriter
) -> None:
    """Fresh start: caller-supplied base_query becomes the stored topical anchor."""

    kg_root = tmp_path / "kg-fresh-bq"
    await run_pipeline(
        base_query="solar perovskites",
        kg_root=kg_root,
        adapters={"arxiv": FakeAdapter("arxiv")},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=1),
        resume=False,
    )
    store = TraceStore.open(kg_root)
    try:
        blob = store.read_loop_state()
        assert blob is not None
        assert CorpusState.model_validate_json(blob).base_query == "solar perovskites"
    finally:
        store.close()


async def test_idempotent_skips_already_fetched_docs(tmp_path: Path, trace: TraceWriter) -> None:
    adapter = FakeAdapter(
        "arxiv",
        candidates_by_query={
            "any": [
                SourceCandidate(
                    source="arxiv",
                    canonical="2105.12345",
                    url="https://arxiv.org/abs/2105.12345",
                    title="Stable paper",
                    abstract="abc",
                )
            ]
        },
    )
    kg_root = tmp_path / "kg-idem"

    await run_pipeline(
        base_query="x",
        kg_root=kg_root,
        adapters={"arxiv": adapter},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=1),
        resume=False,
    )
    assert len(adapter.fetch_calls) >= 1
    run1_fetched = set(adapter.fetch_calls)

    adapter.fetch_calls.clear()
    await run_pipeline(
        base_query="x",
        kg_root=kg_root,
        adapters={"arxiv": adapter},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=1),
        resume=False,
    )
    # Already-ingested doc_ids are dropped BEFORE the Selector, so nothing
    # already in kg.db is re-fetched. The freed selection slot may surface a
    # different not-yet-ingested candidate (desired resurfacing) — fetch_calls
    # can be non-empty, but it never repeats a previously-ingested canonical.
    assert run1_fetched.isdisjoint(adapter.fetch_calls)


# ── SIGINT immediate cancel ──────────────────────────────────────────────


async def test_sigint_cancels_in_flight_round_and_writes_no_loop_state(
    tmp_path: Path, trace: TraceWriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shutdown event set mid-round → round_task cancelled, stop_reason=sigint,
    no loop_state row written (the cancelled round leaves state pristine)."""

    import asyncio

    shutdown_event = asyncio.Event()

    async def _set_event_then_block(**_kwargs: object) -> None:
        shutdown_event.set()
        await asyncio.sleep(60)  # will be cancelled

    monkeypatch.setattr(
        "memex.kg_intelligence.round_graph.steps.plan", _set_event_then_block
    )

    kg_root = tmp_path / "kg-sigint"
    outcome = await run_pipeline(
        base_query="x",
        kg_root=kg_root,
        adapters={"arxiv": FakeAdapter("arxiv")},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=1000),
        resume=False,
        shutdown_event=shutdown_event,
    )
    assert outcome.stop_reason == "sigint"
    # One round attempt was made (and cancelled); counter reflects the attempt.
    assert outcome.rounds_completed == 1

    store = TraceStore.open(kg_root)
    try:
        assert store.read_loop_state() is None, (
            "cancelled round must not write loop_state"
        )
    finally:
        store.close()


async def test_sigint_pre_set_exits_before_first_round(
    tmp_path: Path, trace: TraceWriter
) -> None:
    """Event pre-set → loop exits at the top before doing any work."""

    import asyncio

    shutdown_event = asyncio.Event()
    shutdown_event.set()

    outcome = await run_pipeline(
        base_query="x",
        kg_root=tmp_path / "kg-presig",
        adapters={"arxiv": FakeAdapter("arxiv")},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=1000),
        resume=False,
        shutdown_event=shutdown_event,
    )
    assert outcome.stop_reason == "sigint"
    assert outcome.rounds_completed == 0


async def test_sigint_preserves_prior_loop_state(
    tmp_path: Path, trace: TraceWriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled round must not overwrite the prior round's loop_state row."""

    import asyncio

    kg_root = tmp_path / "kg-preserve"
    kg_root.mkdir()
    store = TraceStore.open(kg_root)
    try:
        store.write_loop_state(
            CorpusState(
                base_query="x",
                round_number=5,
                urls_fetched=["https://a", "https://b"],
            ).model_dump_json()
        )
    finally:
        store.close()

    shutdown_event = asyncio.Event()

    async def _set_event_then_block(**_kwargs: object) -> None:
        shutdown_event.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(
        "memex.kg_intelligence.round_graph.steps.plan", _set_event_then_block
    )

    await run_pipeline(
        base_query="x",
        kg_root=kg_root,
        adapters={"arxiv": FakeAdapter("arxiv")},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(selector_budget=1, per_source_search=2, max_rounds=1000),
        resume=True,
        shutdown_event=shutdown_event,
    )

    store = TraceStore.open(kg_root)
    try:
        blob = store.read_loop_state()
        assert blob is not None
        state = CorpusState.model_validate_json(blob)
        assert state.round_number == 5
        assert state.urls_fetched == ["https://a", "https://b"]
    finally:
        store.close()


# ── LLM circuit-break ────────────────────────────────────────


async def test_llm_circuit_break_stops_after_threshold(
    tmp_path: Path, trace: TraceWriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistent throttle storm trips the pipeline circuit-break.

    Patches the planner to raise ``LLMThrottleError`` every round; after
    ``llm_circuit_break_threshold`` consecutive failures the build stops
    with ``stop_reason='llm_throttle'`` instead of silently falling
    back to sample-mode.
    """

    from memex.kg_intelligence.agents import LLMThrottleError

    async def _raise_throttle(**_kwargs: object) -> None:
        raise LLMThrottleError("simulated LLM throttle")

    monkeypatch.setattr(
        "memex.kg_intelligence.round_graph.steps.plan", _raise_throttle
    )

    outcome = await run_pipeline(
        base_query="x",
        kg_root=tmp_path / "kg-throttle",
        adapters={"arxiv": FakeAdapter("arxiv")},  # type: ignore[arg-type]
        openalex=None,
        llm_model=None,
        settings=_settings(),
        trace=trace,
        config=LoopConfig(
            selector_budget=1,
            per_source_search=2,
            max_rounds=1000,
            llm_circuit_break_threshold=3,
            llm_circuit_break_cooldown_seconds=0.0,
        ),
        resume=False,
    )
    assert outcome.stop_reason == "llm_throttle"
    # Three attempted rounds, all throttled.
    assert outcome.rounds_completed == 3


# ── document-dedup filter ─────────────────────────────────────────────


def test_filter_already_ingested_drops_persisted_docs(tmp_path: Path) -> None:
    """Already-ingested doc_ids are dropped BEFORE the Selector; a fresh
    (never-ingested) candidate stays eligible, and store=None passes through."""

    from memex.kg_intelligence.ids import doc_id
    from memex.kg_intelligence.persistence import (
        DocMetadata,
        IngestionMetadata,
        QueryProvenance,
        persist_document,
    )
    from memex.kg_intelligence.round_graph.helpers import _filter_already_ingested

    store = TraceStore.open(tmp_path)
    try:
        ingested = SourceCandidate(
            source="arxiv",
            canonical="2105.00001",
            url="https://arxiv.org/abs/2105.00001",
            title="Ingested paper",
        )
        fresh = SourceCandidate(
            source="arxiv",
            canonical="2105.99999",
            url="https://arxiv.org/abs/2105.99999",
            title="Fresh paper",
        )
        persist_document(
            store,
            DocMetadata(
                doc_id=doc_id("arxiv", ingested.canonical),
                source="arxiv",
                fetched_at="2026-06-01T00:00:00Z",
                query=QueryProvenance(base="q", expanded=""),
                title=ingested.title,
                authors=[],
                url=ingested.url,
                context_excerpt="x",
                ingestion_metadata=IngestionMetadata(parse_method="test"),
            ),
            "body",
        )

        filtered, dropped, by_topic = _filter_already_ingested(
            {"arxiv": [ingested, fresh]}, store
        )
        assert dropped == 1
        assert [c.canonical for c in filtered["arxiv"]] == ["2105.99999"]
        # no topics set on these candidates → the untagged bucket: 2 surfaced, 1 seen
        assert by_topic == {"(untagged)": [2, 1]}

        passthrough, dropped0, by_topic0 = _filter_already_ingested(
            {"arxiv": [ingested, fresh]}, None
        )
        assert dropped0 == 0
        assert len(passthrough["arxiv"]) == 2
        assert by_topic0 == {}
    finally:
        store.close()


# ── VocabResolver finalization phase ─────────────────────────────────────


def _vocab_store_sinks(tmp_path: Path, trace: TraceWriter):
    store = TraceStore.open(tmp_path / "kg-vocab")
    build = store.start_build(
        build_run_id="b", kg_id="kg-vocab", base_query="q", config={},
        sources=["arxiv"], sample_mode=True, resume_from_round=None,
    )
    round_ = store.start_round(
        build, round_number=1, entities_before=0, relations_before=0,
        schema_entity_types_before=0, schema_relation_types_before=0,
    )
    return store, AgentCallSinks(trace=trace, store=store, round_=round_)


async def test_finalize_vocab_clusters_runs_both_lanes(
    tmp_path: Path, trace: TraceWriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memex.kg_intelligence import pipeline

    store, sinks = _vocab_store_sinks(tmp_path, trace)
    seen: list[str] = []

    async def _fake(*, kind: str, **_kw):
        seen.append(kind)
        return {"clusters": 0, "alias": 0, "child_of": 0, "unrelated": 0}

    monkeypatch.setattr(pipeline, "cluster_and_commit", _fake)
    await pipeline._finalize_vocab_clusters(
        store=store, settings=_settings(), sinks=sinks,
        llm_model=None, round_number=1,
    )
    assert seen == ["predicate", "property"]  # both vocab lanes clustered
    store.close()


async def test_finalize_vocab_clusters_degrades_on_failure(
    tmp_path: Path, trace: TraceWriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memex.kg_intelligence import pipeline

    store, sinks = _vocab_store_sinks(tmp_path, trace)
    seen: list[str] = []

    async def _fake(*, kind: str, **_kw):
        if kind == "predicate":
            raise RuntimeError("boom")
        seen.append(kind)
        return {"clusters": 0}

    monkeypatch.setattr(pipeline, "cluster_and_commit", _fake)
    # must NOT raise — a clustering failure degrades; the built KG is kept.
    await pipeline._finalize_vocab_clusters(
        store=store, settings=_settings(), sinks=sinks,
        llm_model=None, round_number=1,
    )
    assert seen == ["property"]  # property lane still runs after predicate fails
    store.close()
