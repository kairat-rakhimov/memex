"""Build the per-round graph.

Wiring: start → plan → search_rank_select → fetch → gate →
extract_and_consolidate → finalize → end. The two end-of-round reviews
are NOT graph steps — they run in the orchestrator's Phase 2.
"""

from __future__ import annotations

from pydantic_graph import GraphBuilder
from pydantic_graph.graph_builder import Graph

from memex.kg_intelligence.round_graph.state import (
    RoundGraphDeps,
    RoundGraphState,
    RoundOutput,
)
from memex.kg_intelligence.round_graph.steps import (
    extract_and_consolidate_step,
    fetch_step,
    finalize_step,
    gate_step,
    plan_step,
    search_rank_select_step,
)


def build_round_graph() -> Graph[RoundGraphState, RoundGraphDeps, None, RoundOutput]:
    builder: GraphBuilder[RoundGraphState, RoundGraphDeps, None, RoundOutput] = GraphBuilder(
        name="round_graph",
        state_type=RoundGraphState,
        deps_type=RoundGraphDeps,
        output_type=RoundOutput,
        auto_instrument=False,
    )

    plan_node = builder.step(plan_step, node_id="plan")
    search_node = builder.step(search_rank_select_step, node_id="search_rank_select")
    fetch_node = builder.step(fetch_step, node_id="fetch")
    gate_node = builder.step(gate_step, node_id="gate")
    consolidate_node = builder.step(
        extract_and_consolidate_step, node_id="extract_and_consolidate",
    )
    finalize_node = builder.step(finalize_step, node_id="finalize")

    builder.add_edge(builder.start_node, plan_node)
    builder.add_edge(plan_node, search_node)
    builder.add_edge(search_node, fetch_node)
    builder.add_edge(fetch_node, gate_node)
    builder.add_edge(gate_node, consolidate_node)
    builder.add_edge(consolidate_node, finalize_node)
    builder.add_edge(finalize_node, builder.end_node)

    return builder.build()
