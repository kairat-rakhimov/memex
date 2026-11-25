"""Source-plugin protocol + shared data types.

A **source** (arxiv / wikipedia / gdelt) maps to exactly one
``SourceAdapter``. An adapter may pick an internal backend (arxiv: GROBID
vs pdfminer parsing), but that is invisible to the pipeline — the manager
registers one adapter per source ``name``.

Three entry points, all **per-source batch** so per-query / per-doc pacing
lives inside the adapter:

- ``plan_and_search(ranked_topics, *, relevance, limit, ...)`` — the
  adapter turns the round's source-agnostic ranked topics into its own
  source-tailored queries (an LLM call), runs them, and returns
  topic-tagged candidates. This is the acquisition entry point.
- ``search(queries, *, limit) -> list[SourceCandidate]`` — the low-level
  primitive ``plan_and_search`` runs once it has queries; metadata only.
- ``fetch(candidates, *, settings) -> list[ChunkedDoc]`` — download +
  parse + chunk the selected candidates; failures are logged and dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from memex.common.config import Settings
from memex.kg_intelligence.ids import Source
from memex.kg_intelligence.sources.common.chunker import Chunk

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from memex.kg_intelligence.agents.base import AgentCallSinks


class SourceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    canonical: str
    """Per-source canonical key fed into ``doc_id(source, canonical)``:
    arXiv id (no version), Wikipedia page title, or normalised GDELT URL."""
    url: str
    title: str
    abstract: str = ""
    raw_signals: dict[str, Any] = Field(default_factory=dict)
    matched_query: str | None = None
    """Query string that surfaced this candidate; preserves per-doc →
    planner-query attribution into ``fetched_docs``."""
    topics: list[str] = Field(default_factory=list)
    """Round topics this candidate was searched for; set by
    ``plan_and_search``. The global Selector reads these to score
    cross-source candidates by priority-weighted goal fit."""
    goal_relevance: float | None = None
    """Selector-assigned goal-fit score; ``None`` until the Selector runs."""


class FetchedDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: SourceCandidate
    cleaned_text: str
    parse_method: str = ""
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkedDoc(FetchedDoc):
    """A fetched doc plus its chunks. ``fetch()`` returns these directly —
    chunking happens inside the adapter, not in a later pipeline step.

    ``chunks`` offsets index into ``cleaned_text`` (``chunk.text ==
    cleaned_text[chunk.char_start:chunk.char_end]``)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    chunks: list[Chunk] = Field(default_factory=list)


class AdapterDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    summary: str
    """Narrative, 1-2 sentences. Surfaced to the Selector."""
    posture: str
    """``"throttled"`` | ``"relaxed"`` — rate-limit posture."""
    preferred_doc_cap: int
    """Soft per-round doc hint for the Selector."""
    metadata_fields: list[str]
    """Keys this source populates in ``SourceCandidate.raw_signals``."""


@runtime_checkable
class SourceAdapter(Protocol):
    name: str
    description: AdapterDescription

    def probe(self) -> bool:
        """True if this adapter is ready (env vars, deps). Cheap — no
        network calls. Called once at process start."""

    async def plan_and_search(
        self,
        ranked_topics: list[str],
        *,
        relevance: Literal["low", "medium", "high"],
        limit: int,
        settings: Settings,
        sinks: AgentCallSinks,
        llm_model: Model | None = None,
    ) -> list[SourceCandidate]:
        """Turn the round's ranked topics into source-tailored queries (an
        LLM call), run them, and return topic-tagged candidates.

        ``relevance`` is the RoundPlanner's corpus-fit signal for this
        source. Throttled adapters act only at ``"high"`` (else return
        ``[]`` without the planner call or search); relaxed adapters always
        act. ``settings`` / ``sinks`` / ``llm_model`` carry the LLM
        plumbing the query planner needs."""

    def search(self, queries: list[str], *, limit: int) -> list[SourceCandidate]:
        """Low-level primitive ``plan_and_search`` runs once it has queries:
        run them (per-query pacing is the adapter's own concern) and return
        candidates with ``matched_query`` set."""

    def fetch(
        self,
        candidates: list[SourceCandidate],
        *,
        settings: Settings,
    ) -> list[ChunkedDoc]:
        """Download + parse + chunk the selected candidates. One
        ChunkedDoc per successfully fetched candidate; failed/empty
        candidates are logged and dropped."""


__all__ = [
    "SourceCandidate",
    "FetchedDoc",
    "ChunkedDoc",
    "AdapterDescription",
    "SourceAdapter",
]
