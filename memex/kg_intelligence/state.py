"""Outer-loop accumulator state passed between rounds.

Serialised into the ``loop_state`` table at end-of-round; resume
hydrates from the same row. The canonical KG itself lives in ``kg.db``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from memex.kg_intelligence.ids import Source


class SelectedTitle(BaseModel):
    """Kept across rounds so future rounds can title-dedup."""

    model_config = ConfigDict(extra="forbid")

    source: Source
    title: str


class RoundGoalRecord(BaseModel):
    """Compact per-round record of what the RoundPlanner aimed at, kept so the
    RoundReviewer can see which themes have recurred across rounds (the
    "this theme wasted N rounds" judgment)."""

    model_config = ConfigDict(extra="forbid")

    round: int = Field(ge=0)
    focused_goal: str = ""
    priority_themes: list[str] = Field(default_factory=list)


class CorpusState(BaseModel):
    """Cross-round accumulator updated by the UsefulnessGate + Selector."""

    model_config = ConfigDict(extra="forbid")

    base_query: str = ""
    round_number: int = Field(default=0, ge=0)
    urls_fetched: list[str] = Field(default_factory=list)
    titles_selected: list[SelectedTitle] = Field(default_factory=list)
    entities_seen: list[str] = Field(default_factory=list)
    subtopics_covered: list[str] = Field(default_factory=list)
    subtopics_missing: list[str] = Field(default_factory=list)
    borders: list[str] = Field(default_factory=list)
    consecutive_empty_rounds: int = Field(default=0, ge=0)
    # Strategist memory (RoundReviewer). ``abandoned_goals`` is the accumulating
    # blocklist of themes judged unanswerable — the RoundPlanner hard-excludes
    # them. ``goal_history`` is the recent-rounds theme trail (trimmed).
    abandoned_goals: list[str] = Field(default_factory=list)
    goal_history: list[RoundGoalRecord] = Field(default_factory=list)
    started_at: str = ""
    elapsed_seconds: float = 0.0
