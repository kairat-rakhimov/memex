"""Per-round graph for the kg-loop pipeline.

Built with ``pydantic_graph``'s ``GraphBuilder`` + ``@step`` API.
The outer ``run_pipeline()`` loop stays imperative and invokes the
graph once per round.
"""

from memex.kg_intelligence.round_graph.graph import build_round_graph
from memex.kg_intelligence.round_graph.state import (
    RoundGraphDeps,
    RoundGraphState,
    RoundOutput,
    _RoundDeltas,
)

__all__ = [
    "RoundGraphDeps",
    "RoundGraphState",
    "RoundOutput",
    "_RoundDeltas",
    "build_round_graph",
]
