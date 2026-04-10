from __future__ import annotations

from qwen_langgraph_demo.graph.state import ExperimentState
from qwen_langgraph_demo.runtime.protocol import ProtocolBundle
from qwen_langgraph_demo.runtime.sample_data import DEFAULT_TRANSACTION_CONTEXT


def build_intake_node(protocol: ProtocolBundle):
    def intake_node(state: ExperimentState) -> ExperimentState:
        transaction_context = dict(DEFAULT_TRANSACTION_CONTEXT)
        transaction_context.update(state.get("transaction_context", {}))
        request_text = state.get("request_text", protocol.stage1.benign_request)
        message_text = protocol.render_intake_message(transaction_context, request_text)
        trace = [*state.get("trace", []), "intake"]
        return {
            "transaction_context": transaction_context,
            "request_text": request_text,
            "message_text": message_text,
            "trace": trace,
        }

    return intake_node
