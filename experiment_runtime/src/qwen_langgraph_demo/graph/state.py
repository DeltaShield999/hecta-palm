from typing import Any, TypedDict


class ExperimentState(TypedDict, total=False):
    transaction_context: dict[str, Any]
    request_text: str
    message_text: str
    filter_decision: str
    filter_reason: str
    dropped_by_filter: bool
    fraud_decision: str
    fraud_score: float
    fraud_response: str
    routing_decision: str
    trace: list[str]
