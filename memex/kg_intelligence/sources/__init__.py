"""Source plugins fanned out by the kg-loop pipeline.

Each source maps to one ``SourceAdapter`` (see ``common/base.py``);
``AdapterManager`` registers the ready ones at startup. The pipeline
iterates ``manager.active`` and never names a source directly.
"""

from memex.kg_intelligence.sources.arxiv import ArxivAdapter
from memex.kg_intelligence.sources.common.base import (
    AdapterDescription,
    ChunkedDoc,
    FetchedDoc,
    SourceAdapter,
    SourceCandidate,
)
from memex.kg_intelligence.sources.gdelt import GdeltAdapter
from memex.kg_intelligence.sources.manager import AdapterManager
from memex.kg_intelligence.sources.wikipedia import WikipediaAdapter

__all__ = [
    "AdapterDescription",
    "AdapterManager",
    "ArxivAdapter",
    "ChunkedDoc",
    "FetchedDoc",
    "GdeltAdapter",
    "SourceAdapter",
    "SourceCandidate",
    "WikipediaAdapter",
]
