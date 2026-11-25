"""State, deps, and output types for the per-round graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.models import Model

from memex.common.config import Settings
from memex.kg_intelligence.agents import AgentCallSinks
from memex.kg_intelligence.agents.schemas import (
    ExtractionResult,
    GoalDirective,
    KgGap,
    RoundPlan,
)
from memex.kg_intelligence.openalex import OpenAlexClient
from memex.kg_intelligence.sources.common.base import (
    ChunkedDoc,
    SourceAdapter,
    SourceCandidate,
)
from memex.kg_intelligence.state import CorpusState

if TYPE_CHECKING:
    # Forward reference breaks the would-be circular import with pipeline.py.
    from memex.kg_intelligence.pipeline import LoopConfig


@dataclass
class _RoundDeltas:
    """Round-local bookkeeping that feeds stopping criteria + state merges."""

    new_candidates: list[SourceCandidate] = field(default_factory=list)
    new_extractions: list[ExtractionResult] = field(default_factory=list)
    docs_fetched: int = 0
    docs_skipped: int = 0
    docs_failed: int = 0
    # Cross-round dedup signal: how many searched candidates were already in the
    # KG (dropped pre-Selector) vs total surfaced. A high rate = the corpus for
    # this round's topic is mined out — the strategist's saturation signal.
    already_ingested: int = 0
    candidates_surfaced: int = 0
    # Per-topic breakdown of the above: topic → [surfaced, already_seen]. Lets the
    # strategist abandon the exact mined-out theme, not just see a round aggregate.
    already_seen_by_topic: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class RoundGraphDeps:
    base_query: str
    adapters: dict[str, SourceAdapter]
    openalex: OpenAlexClient | None
    llm_model: Model | None
    settings: Settings
    sinks: AgentCallSinks
    cfg: LoopConfig
    kg_root: Path
    force: bool


class RoundGraphState(BaseModel):
    """Mutable per-round state shared across steps."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    corpus_state: CorpusState
    # Strategist inputs the orchestrator reads from the DB at round start
    # (empty on R1): the last round's KGReviewer gaps + RoundReviewer directives.
    kg_gaps_in: list[KgGap]
    directives_in: list[GoalDirective] = Field(default_factory=list)
    deltas: _RoundDeltas

    pre_entity_count: int = 0
    pre_entity_names: set[str] = Field(default_factory=set)
    pre_mentions: dict[str, int] = Field(default_factory=dict)

    plan_result: RoundPlan | None = None
    selected_by_source: dict[str, list[SourceCandidate]] = Field(default_factory=dict)
    fetched_by_doc_id: dict[str, tuple[ChunkedDoc, int]] = Field(default_factory=dict)

    # Catch-all for ranker output etc.
    scratch: dict[str, Any] = Field(default_factory=dict)


class RoundOutput(BaseModel):
    """Graph output — round work deltas only. The reviews (and therefore the
    next round's gaps + the stop signal) run in the orchestrator's Phase 2,
    not the graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    deltas: _RoundDeltas
