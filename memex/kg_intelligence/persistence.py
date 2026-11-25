"""Source-doc persistence for the kg-loop pipeline.

kg.db is the single canonical store — there are no per-doc files on disk.
Each ingested document is one ``documents`` row holding the serialized
``DocMetadata`` + the full cleaned text (see the observability schema).
This module owns the ``DocMetadata`` model + its (de)serialization;
``TraceStore`` owns the row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from memex.kg_intelligence.ids import Source

if TYPE_CHECKING:
    from memex.observability import TraceStore

EXCERPT_MAX_CHARS = 200


class QueryProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base: str
    expanded: str = ""


class IngestionMetadata(BaseModel):
    """Per-source ingestion details; concrete keys vary by adapter."""

    model_config = ConfigDict(extra="allow")

    parse_method: str = ""


class OpenAlexMetadata(BaseModel):
    """All fields optional; many docs lack an OpenAlex match."""

    model_config = ConfigDict(extra="allow")

    openalex_id: str | None = None
    doi: str | None = None
    topics: list[dict[str, Any]] = Field(default_factory=list)
    concepts: list[dict[str, Any]] = Field(default_factory=list)
    cited_by_count: int | None = None
    related_works: list[str] = Field(default_factory=list)


class DocMetadata(BaseModel):
    """Per-document metadata (everything but the body). Serialized into the
    ``documents.metadata`` column; the body lives in ``documents.text``."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: Source
    fetched_at: str
    query: QueryProvenance
    title: str
    authors: list[str] = Field(default_factory=list)
    url: str
    context_excerpt: str = Field(default="", max_length=EXCERPT_MAX_CHARS)
    ingestion_metadata: IngestionMetadata
    openalex_metadata: OpenAlexMetadata | None = None


def make_context_excerpt(cleaned_text: str) -> str:
    """First `EXCERPT_MAX_CHARS` chars of cleaned text, whitespace-normalised."""

    flat = " ".join((cleaned_text or "").split())
    return flat[:EXCERPT_MAX_CHARS]


def has_document(store: TraceStore, doc_id: str) -> bool:
    """True when ``doc_id`` is already persisted in kg.db. Drives the
    idempotent-skip on re-fetch (unless ``--force``)."""

    return store.has_document(doc_id)


def persist_document(store: TraceStore, meta: DocMetadata, text: str) -> None:
    """Persist the canonical per-doc record (serialized metadata + full
    cleaned text) into kg.db's ``documents`` table."""

    store.record_document(meta.doc_id, meta.model_dump_json(), text)


def read_document_metadata(store: TraceStore, doc_id: str) -> DocMetadata:
    """Load a persisted ``DocMetadata`` from kg.db. Raises if missing."""

    payload = store.get_document_metadata(doc_id)
    if payload is None:
        raise KeyError(f"no document for {doc_id!r}")
    return DocMetadata.model_validate_json(payload)
