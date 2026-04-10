from __future__ import annotations

from qwen_langgraph_demo.graph.state import ExperimentState


def build_router_node():
    def router_node(state: ExperimentState) -> ExperimentState:
        decision = state.get("fraud_decision", "REVIEW")
        trace = [*state.get("trace", []), "router"]
        return {
            "routing_decision": decision,
            "trace": trace,
        }

    return router_node
