from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from qwen_langgraph_demo.graph.state import ExperimentState
from qwen_langgraph_demo.nodes.filter_middleware import build_filter_middleware_node
from qwen_langgraph_demo.nodes.fraud_scorer import build_fraud_scorer_node
from qwen_langgraph_demo.nodes.intake import build_intake_node
from qwen_langgraph_demo.nodes.router import build_router_node
from qwen_langgraph_demo.runtime.protocol import ProtocolBundle


def _route_after_filter(state: ExperimentState) -> Literal["blocked", "allowed"]:
    if state.get("dropped_by_filter"):
        return "blocked"
    return "allowed"


def build_graph(protocol: ProtocolBundle):
    graph = StateGraph(ExperimentState)
    graph.add_node("intake", build_intake_node(protocol))
    graph.add_node("filter_middleware", build_filter_middleware_node())
    graph.add_node("fraud_scorer", build_fraud_scorer_node(protocol))
    graph.add_node("router", build_router_node())

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "filter_middleware")
    graph.add_conditional_edges(
        "filter_middleware",
        _route_after_filter,
        {
            "blocked": END,
            "allowed": "fraud_scorer",
        },
    )
    graph.add_edge("fraud_scorer", "router")
    graph.add_edge("router", END)
    return graph.compile()
