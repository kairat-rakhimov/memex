"""Tests for the kg-loop persistence + state slice.

Covers: stable doc_ids, the stored-vs-raw split, idempotent skip, and
per-round resume. Agents, adapters, and the pipeline loop have their own tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.kg_intelligence import (
    CorpusState,
    DocMetadata,
    IngestionMetadata,
    OpenAlexMetadata,
    QueryProvenance,
    doc_id,
    has_document,
    make_context_excerpt,
    persist_document,
    read_document_metadata,
)
from memex.kg_intelligence.persistence import EXCERPT_MAX_CHARS
from memex.observability import TraceStore

# ── doc_id ─────────────────────────────────────────────────────────────────


def test_doc_id_is_stable_and_normalized() -> None:
    assert doc_id("arxiv", "2105.12345") == doc_id("arxiv", "2105.12345")
    assert doc_id("wikipedia", "Hydrogen economy") == doc_id(
        "wikipedia", "  hydrogen\tECONOMY  "
    )
    assert doc_id("arxiv", "x").startswith("doc_")


def test_doc_id_source_prefix_prevents_cross_source_collision() -> None:
    assert doc_id("arxiv", "foo") != doc_id("wikipedia", "foo")
    assert doc_id("arxiv", "foo") != doc_id("gdelt", "foo")


# ── persistence ────────────────────────────────────────────────────────────


def _make_doc(d_id: str = "doc_test") -> tuple[DocMetadata, str]:
    raw_text = "Full body of the paper goes here. " * 50
    doc = DocMetadata(
        doc_id=d_id,
        source="arxiv",
        fetched_at="2026-05-22T10:00:00Z",
        query=QueryProvenance(base="green hydrogen", expanded="iridium catalyst"),
        title="A paper",
        authors=["Alice", "Bob"],
        url="https://arxiv.org/abs/2105.12345",
        context_excerpt=make_context_excerpt(raw_text),
        ingestion_metadata=IngestionMetadata(
            parse_method="pdfminer.six",
            arxiv_id="2105.12345",
        ),
        openalex_metadata=OpenAlexMetadata(
            openalex_id="W123",
            doi="10.1000/xyz",
            cited_by_count=7,
        ),
    )
    return doc, raw_text


def test_make_context_excerpt_caps_and_normalises() -> None:
    text = "alpha   beta\tgamma\n\ndelta " + ("x" * 500)
    excerpt = make_context_excerpt(text)
    assert len(excerpt) == EXCERPT_MAX_CHARS
    assert excerpt.startswith("alpha beta gamma delta")
    assert "\t" not in excerpt
    assert "\n" not in excerpt


def test_persist_document_round_trips_through_db(tmp_path: Path) -> None:
    meta, raw_text = _make_doc()
    store = TraceStore.open(tmp_path)

    assert not has_document(store, meta.doc_id)
    persist_document(store, meta, raw_text)
    assert has_document(store, meta.doc_id)

    # metadata + full text both round-trip from kg.db
    assert read_document_metadata(store, meta.doc_id).model_dump() == meta.model_dump()
    assert store.get_document_text(meta.doc_id) == raw_text

    # kg.db is the only store — no per-doc files on disk
    assert not (tmp_path / "sources").exists()
    assert not (tmp_path / "_raw").exists()


def test_persist_document_upserts(tmp_path: Path) -> None:
    meta, raw_text = _make_doc()
    store = TraceStore.open(tmp_path)
    persist_document(store, meta, raw_text)
    persist_document(store, meta, "shorter body")
    assert store.get_document_text(meta.doc_id) == "shorter body"


def test_read_document_metadata_missing_raises(tmp_path: Path) -> None:
    store = TraceStore.open(tmp_path)
    assert store.get_document_text("missing") is None
    with pytest.raises(KeyError):
        read_document_metadata(store, "missing")


def test_doc_metadata_excerpt_max_length_enforced() -> None:
    with pytest.raises(ValueError):
        DocMetadata(
            doc_id="doc_x",
            source="arxiv",
            fetched_at="2026-05-22T10:00:00Z",
            query=QueryProvenance(base="q"),
            title="t",
            url="u",
            context_excerpt="x" * (EXCERPT_MAX_CHARS + 1),
            ingestion_metadata=IngestionMetadata(),
        )


# ── corpus_state persistence ──────────────────────────────────────────────


def _make_corpus(n: int) -> CorpusState:
    return CorpusState(
        round_number=n,
        urls_fetched=[f"arxiv:21{n:02d}.00001"],
        entities_seen=["PEM electrolyzer"],
        started_at="2026-05-22T10:00:00Z",
        elapsed_seconds=60.0 * n,
    )


def test_loop_state_round_trips_corpus_state(tmp_path: Path) -> None:
    """Resume reads the canonical kg.db loop_state row, not _state/*.json."""

    store = TraceStore.open(tmp_path)
    try:
        assert store.read_loop_state() is None
        original = _make_corpus(3)
        store.write_loop_state(original.model_dump_json())
        loaded = CorpusState.model_validate_json(store.read_loop_state())
        assert loaded == original

        # Update overwrites the singleton row.
        updated = _make_corpus(5)
        store.write_loop_state(updated.model_dump_json())
        loaded2 = CorpusState.model_validate_json(store.read_loop_state())
        assert loaded2.round_number == 5
    finally:
        store.close()
