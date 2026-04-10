from __future__ import annotations

from experiment.chat_render import assess_benign_transaction
from qwen_langgraph_demo.graph.state import ExperimentState
from qwen_langgraph_demo.runtime.protocol import ProtocolBundle


def build_fraud_scorer_node(protocol: ProtocolBundle):
    def fraud_scorer_node(state: ExperimentState) -> ExperimentState:
        transaction_context = state.get("transaction_context", {})
        render = assess_benign_transaction(transaction_context)
        response = render.render(protocol)
        trace = [*state.get("trace", []), "fraud_scorer"]
        return {
            "fraud_decision": render.decision,
            "fraud_score": render.fraud_score,
            "fraud_response": response,
            "trace": trace,
        }

    return fraud_scorer_node
