"""Unified KG-build loop.

Query-driven loop that iteratively ingests, extracts and reviews until the
Strategist halts the build or a round cap is reached.
"""

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
    read_document_metadata,
)
from memex.kg_intelligence.pipeline import LoopConfig, LoopOutcome, run_pipeline
from memex.kg_intelligence.ranker import rank
from memex.kg_intelligence.state import (
    CorpusState,
    SelectedTitle,
)
from memex.kg_intelligence.views import KgSchemaView, KgView, load_kg_view, load_schema_view

__all__ = [
    "CorpusState",
    "IngestionMetadata",
    "KgSchemaView",
    "KgView",
    "LoopConfig",
    "LoopOutcome",
    "OpenAlexClient",
    "OpenAlexMetadata",
    "QueryProvenance",
    "SelectedTitle",
    "Source",
    "DocMetadata",
    "doc_id",
    "has_document",
    "load_kg_view",
    "load_schema_view",
    "make_context_excerpt",
    "persist_document",
    "rank",
    "read_document_metadata",
    "run_pipeline",
]
