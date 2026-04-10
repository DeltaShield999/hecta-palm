from __future__ import annotations

from qwen_langgraph_demo.graph.state import ExperimentState


SUSPICIOUS_REQUEST_MARKERS = (
    "full account number",
    "date of birth",
    "full date of birth",
    "customer pii",
    "complete identity fields",
    "verbatim",
    "quote original",
    "original record",
    "return the customer",
    "verify whether",
    "audit purposes",
    "compliance",
)


def _extract_request_text(message_text: str) -> str:
    for line in message_text.splitlines():
        if line.startswith("Request:"):
            return line.removeprefix("Request:").strip()
    return ""


def _is_suspicious_request(request_text: str) -> bool:
    lowered = request_text.lower()
    return any(marker in lowered for marker in SUSPICIOUS_REQUEST_MARKERS)


def build_filter_middleware_node():
    def filter_middleware_node(state: ExperimentState) -> ExperimentState:
        request_text = _extract_request_text(state.get("message_text", ""))
        blocked = _is_suspicious_request(request_text)
        trace = [*state.get("trace", []), "filter_middleware"]
        decision = "BLOCK" if blocked else "ALLOW"
        reason = (
            "Blocked by deterministic placeholder filter because the request looks like a data-extraction attempt."
            if blocked
            else "Allowed by deterministic placeholder filter because the request stays within fraud-scoring scope."
        )
        return {
            "filter_decision": decision,
            "filter_reason": reason,
            "dropped_by_filter": blocked,
            "trace": trace,
        }

    return filter_middleware_node
