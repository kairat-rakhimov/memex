"""Skeleton smoke test for the per-round graph.

Asserts the graph builds, runs end-to-end against SAMPLE_MODE agents +
fake adapters, and returns a `RoundOutput` with deltas reflecting the
work done. Does NOT assert KG content correctness — that's covered by
tests/test_kg_intelligence_pipeline.py.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from memex.common.config import Settings
from memex.common.trace import TraceWriter
from memex.kg_intelligence.agents import AgentCallSinks
from memex.kg_intelligence.ids import Source
from memex.kg_intelligence.pipeline import LoopConfig, _RoundDeltas
from memex.kg_intelligence.round_graph import (
    RoundGraphDeps,
    RoundGraphState,
    RoundOutput,
    build_round_graph,
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
from memex.kg_intelligence.state import CorpusState
from memex.observability import TraceStore


class _FakeAdapter:
    """Returns one canned candidate per query; canned body on fetch."""

    def __init__(
        self,
        source: Source,
        body_for: Callable[[SourceCandidate], str] | None = None,
    ) -> None:
        self.source = source
        self.body_for = body_for or (lambda c: f"body of {c.title}")
        self._planner = BaseQueryPlanner(source_name=source, guidance="test double")

    @property
    def name(self) -> str:
        return self.source

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


class _EmptyAdapter:
    """Finds nothing — exercises the empty-fetched_by_doc_id path."""

    def __init__(self, source: Source) -> None:
        self.source = source
        self.name = source
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
        return []

    def fetch(
        self, candidates: list[SourceCandidate], *, settings: Settings
    ) -> list[ChunkedDoc]:  # pragma: no cover
        return []


def _settings() -> Settings:
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
        sample_mode=True,
        contact=None,
    )


@pytest.fixture
def trace(tmp_path: Path) -> TraceWriter:
    return TraceWriter(request_id="round-graph-skeleton", log_dir=tmp_path / "_trace")


async def test_round_graph_skeleton_runs_end_to_end(
    tmp_path: Path, trace: TraceWriter
) -> None:
    adapters: dict[Source, _FakeAdapter] = {
        "arxiv": _FakeAdapter("arxiv"),
        "wikipedia": _FakeAdapter("wikipedia"),
        "gdelt": _FakeAdapter("gdelt"),
    }
    kg_root = tmp_path / "kg-graphtest"
    kg_root.mkdir()

    store = TraceStore.open(kg_root)
    try:
        build = store.start_build(
            build_run_id="round-graph-skeleton",
            kg_id="kg-graphtest",
            base_query="quantum cryptography",
            config={"max_rounds": 1},
            sources=list(adapters.keys()),
            sample_mode=True,
            resume_from_round=None,
        )
        round_handle = store.start_round(
            build,
            round_number=1,
            entities_before=0,
            relations_before=0,
            schema_entity_types_before=0,
            schema_relation_types_before=0,
        )
        sinks = AgentCallSinks(trace=trace, store=store, round_=round_handle)

        deps = RoundGraphDeps(
            base_query="quantum cryptography",
            adapters=adapters,
            openalex=None,
            llm_model=None,
            settings=_settings(),
            sinks=sinks,
            cfg=LoopConfig(),
            kg_root=kg_root,
            force=False,
        )
        state = RoundGraphState(
            corpus_state=CorpusState(round_number=1),
            kg_gaps_in=[],
            deltas=_RoundDeltas(),
        )

        graph = build_round_graph()
        output = await graph.run(state=state, deps=deps)

        assert isinstance(output, RoundOutput)
        assert isinstance(output.deltas, _RoundDeltas)
        # Skeleton ran the full linear flow — at least one doc fetched +
        # extraction produced an ExtractionResult.
        assert output.deltas.docs_fetched >= 1
        assert len(output.deltas.new_extractions) >= 1
        # plan_result populated by plan_step; reviews now run in the orchestrator
        assert state.plan_result is not None
        # Per-doc extract → Mapper → merge ran: the KG got populated.
        assert store.count_entities() >= 1
    finally:
        store.close()


async def test_round_graph_handles_zero_fetched_docs(
    tmp_path: Path, trace: TraceWriter
) -> None:
    """When no docs are fetched, ``extract_and_consolidate_step``
    short-circuits (empty ``fetched_by_doc_id``). The round still
    produces a valid RoundOutput.
    """

    adapters: dict[Source, _EmptyAdapter] = {
        "arxiv": _EmptyAdapter("arxiv"),
        "wikipedia": _EmptyAdapter("wikipedia"),
        "gdelt": _EmptyAdapter("gdelt"),
    }
    kg_root = tmp_path / "kg-graphtest-empty"
    kg_root.mkdir()

    store = TraceStore.open(kg_root)
    try:
        build = store.start_build(
            build_run_id="round-graph-empty",
            kg_id="kg-graphtest-empty",
            base_query="quantum cryptography",
            config={"max_rounds": 1},
            sources=list(adapters.keys()),
            sample_mode=True,
            resume_from_round=None,
        )
        round_handle = store.start_round(
            build,
            round_number=1,
            entities_before=0,
            relations_before=0,
            schema_entity_types_before=0,
            schema_relation_types_before=0,
        )
        sinks = AgentCallSinks(trace=trace, store=store, round_=round_handle)

        deps = RoundGraphDeps(
            base_query="quantum cryptography",
            adapters=adapters,
            openalex=None,
            llm_model=None,
            settings=_settings(),
            sinks=sinks,
            cfg=LoopConfig(),
            kg_root=kg_root,
            force=False,
        )
        state = RoundGraphState(
            corpus_state=CorpusState(round_number=1),
            kg_gaps_in=[],
            deltas=_RoundDeltas(),
        )

        graph = build_round_graph()
        output = await graph.run(state=state, deps=deps)

        assert isinstance(output, RoundOutput)
        assert output.deltas.docs_fetched == 0
        assert output.deltas.new_extractions == []
        # No docs → the extract/map loop body never ran.
        assert store.count_entities() == 0
    finally:
        store.close()
