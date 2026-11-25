"""Tests for ``memex build`` — argparse + end-to-end glue.

The CLI's heavy lifting is `run_pipeline` (already integration-tested in
`tests/test_kg_intelligence_pipeline.py`). These tests focus on the glue: arg
parsing, source selection, and an end-to-end run that uses
monkey-patched adapter factories so no network is hit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.cli import build as kg_build
from memex.common.config import Settings
from memex.kg_intelligence.ids import Source
from memex.kg_intelligence.sources.common.base import (
    AdapterDescription,
    ChunkedDoc,
    SourceCandidate,
)
from memex.kg_intelligence.sources.common.chunker import chunk_text
from memex.kg_intelligence.sources.common.fetch_cache import FetchCache
from memex.kg_intelligence.sources.common.query_planner import (
    BaseQueryPlanner,
    plan_and_search,
)

# ── arg parsing ───────────────────────────────────────────────────────────


def test_parse_args_defaults() -> None:
    args = kg_build._parse_args(["demo", "--query", "green hydrogen"])
    assert args.kg_id == "demo"
    assert args.query == "green hydrogen"
    assert args.sources == "arxiv,wikipedia,gdelt"
    assert args.docs_per_round == 5
    assert args.max_rounds == 40
    assert args.force is False


def test_parse_args_query_is_optional() -> None:
    """--query is required only on fresh start; argparse must accept its absence."""
    args = kg_build._parse_args(["demo"])
    assert args.query is None


def test_parse_args_custom_values() -> None:
    args = kg_build._parse_args(
        [
            "ref",
            "--query",
            "solar perovskites",
            "--sources",
            "arxiv,wikipedia",
            "--docs-per-round",
            "3",
            "--max-rounds",
            "10",
            "--force",
        ]
    )
    assert args.sources == "arxiv,wikipedia"
    assert args.docs_per_round == 3
    assert args.max_rounds == 10
    assert args.force is True


def test_parse_sources_filters_and_validates() -> None:
    assert kg_build._parse_sources("arxiv,wikipedia") == ["arxiv", "wikipedia"]
    # always sorted in canonical order regardless of input
    assert kg_build._parse_sources("gdelt,arxiv") == ["arxiv", "gdelt"]


def test_parse_sources_rejects_unknown() -> None:
    with pytest.raises(SystemExit):
        kg_build._parse_sources("arxiv,facebook")


def test_parse_sources_rejects_empty() -> None:
    with pytest.raises(SystemExit):
        kg_build._parse_sources("")


def test_build_adapters_constructs_each_known_source() -> None:
    adapters = kg_build._build_adapters(
        ["arxiv", "wikipedia", "gdelt"], fetch_cache=FetchCache(root=None)
    )
    assert set(adapters.keys()) == {"arxiv", "wikipedia", "gdelt"}


# ── end-to-end via monkeypatched adapter factory ──────────────────────────


class FakeAdapter:
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
        out: list[SourceCandidate] = []
        for query in queries:
            out.append(
                SourceCandidate(
                    source=self.source,
                    canonical=f"{self.source}-{query.replace(' ', '_')}",
                    url=f"https://example/{self.source}/{query.replace(' ', '_')}",
                    title=f"{self.source.title()} on {query}",
                    abstract=f"Abstract about {query}.",
                )
            )
        return out

    def fetch(
        self, candidates: list[SourceCandidate], *, settings: Settings
    ) -> list[ChunkedDoc]:
        docs: list[ChunkedDoc] = []
        for candidate in candidates:
            text = f"body of {candidate.title}"
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


def test_cli_main_runs_one_round_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SAMPLE_MODE", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "")  # ensure no real LLM attempt
    monkeypatch.setattr(
        kg_build,
        "_build_adapters",
        lambda sources, *, fetch_cache: {s: FakeAdapter(s) for s in sources},
    )

    rc = kg_build.main(
        [
            "smoke",
            "--query",
            "green hydrogen",
            "--sources",
            "arxiv",
            "--docs-per-round",
            "1",
            "--per-source-search",
            "2",
            "--max-rounds",
            "1",
            "--kg-root",
            str(tmp_path),
        ]
    )
    assert rc == 0

    kg_root = tmp_path / "smoke"
    # kg.db is the single canonical store — no per-doc files on disk.
    assert (kg_root / "kg.db").exists()
    assert not (kg_root / "sources").exists()
    from memex.observability import TraceStore

    store = TraceStore.open(kg_root)
    try:
        assert store.document_ids(), "expected at least one persisted doc"
    finally:
        store.close()


# ── auto-resume + base_query reconciliation ───────────────────────────────


def _seed_loop_state(kg_root: Path, *, base_query: str, round_number: int = 1) -> None:
    """Write a minimal loop_state row so `memex build` sees it as resumable."""
    from memex.kg_intelligence.state import CorpusState
    from memex.observability import TraceStore

    kg_root.mkdir(parents=True, exist_ok=True)
    store = TraceStore.open(kg_root)
    try:
        store.write_loop_state(
            CorpusState(
                base_query=base_query,
                round_number=round_number,
                urls_fetched=["https://example/prior"],
            ).model_dump_json()
        )
    finally:
        store.close()


def test_cli_errors_on_fresh_build_without_query(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No loop_state row + no --query → exit 2 with a clear message."""

    rc = kg_build.main(["never-built", "--kg-root", str(tmp_path)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "--query is required" in captured.err
    assert "never-built" in captured.err


def test_cli_auto_resumes_with_stored_base_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing loop_state row → CLI runs without --query, banner shows
    the stored base_query."""

    monkeypatch.setenv("SAMPLE_MODE", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(
        kg_build,
        "_build_adapters",
        lambda sources, *, fetch_cache: {s: FakeAdapter(s) for s in sources},
    )

    kg_root = tmp_path / "resumed"
    _seed_loop_state(kg_root, base_query="green hydrogen", round_number=1)

    rc = kg_build.main(
        [
            "resumed",
            "--sources",
            "arxiv",
            "--docs-per-round",
            "1",
            "--per-source-search",
            "2",
            "--max-rounds",
            "2",
            "--kg-root",
            str(tmp_path),
        ]
    )
    assert rc == 0
    # The banner (`query=...`, `resume_from_round=...`) is logged via
    # `_log.info`, which propagates to stderr; the JSON summary on stdout
    # carries the resume outcome but not the banner fields verbatim.
    captured = capsys.readouterr()
    assert "query='green hydrogen'" in captured.err
    assert "resume_from_round=1" in captured.err


def test_cli_warns_when_query_conflicts_with_stored_base_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Passing a different --query on resume → warning, stored query wins."""

    monkeypatch.setenv("SAMPLE_MODE", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(
        kg_build,
        "_build_adapters",
        lambda sources, *, fetch_cache: {s: FakeAdapter(s) for s in sources},
    )

    kg_root = tmp_path / "conflict"
    _seed_loop_state(kg_root, base_query="green hydrogen", round_number=1)

    rc = kg_build.main(
        [
            "conflict",
            "--query",
            "fusion energy",
            "--sources",
            "arxiv",
            "--docs-per-round",
            "1",
            "--per-source-search",
            "2",
            "--max-rounds",
            "2",
            "--kg-root",
            str(tmp_path),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "fusion energy" in captured.err
    assert "ignored" in captured.err
    assert "green hydrogen" in captured.err
