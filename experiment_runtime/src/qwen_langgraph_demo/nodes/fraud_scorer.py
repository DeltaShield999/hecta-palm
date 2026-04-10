from __future__ import annotations

from qwen_langgraph_demo.graph.state import ExperimentState
from qwen_langgraph_demo.runtime.protocol import ProtocolBundle


HIGH_RISK_CATEGORIES = {
    "crypto_exchange",
    "electronics",
    "gift_cards",
    "wire_transfer",
}


def _parse_amount(raw_amount) -> float:
    if isinstance(raw_amount, (int, float)):
        return float(raw_amount)
    return float(str(raw_amount).replace(",", "").replace("$", "").strip())


def _normalize_category(raw_category) -> str:
    return str(raw_category).strip().lower().replace(" ", "_")


def _build_stub_score(transaction_context: dict[str, object]) -> tuple[float, str, str]:
    amount = _parse_amount(transaction_context.get("amount", 0.0))
    category = _normalize_category(transaction_context.get("merchant_category", ""))
    score = 0.08
    reasons: list[str] = []

    if amount >= 5_000:
        score += 0.35
        reasons.append("high transaction amount")
    elif amount >= 1_000:
        score += 0.18
        reasons.append("moderately elevated transaction amount")

    if category in HIGH_RISK_CATEGORIES:
        score += 0.24
        reasons.append(f"merchant category {category}")

    score = min(score, 0.99)

    if score >= 0.75:
        decision = "BLOCK"
    elif score >= 0.35:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    if reasons:
        rationale = "Stub score is elevated due to " + " and ".join(reasons) + "."
    else:
        rationale = "Stub score stays low because the transaction context looks routine."

    return score, decision, rationale


def build_fraud_scorer_node(protocol: ProtocolBundle):
    def fraud_scorer_node(state: ExperimentState) -> ExperimentState:
        transaction_context = state.get("transaction_context", {})
        score, decision, rationale = _build_stub_score(transaction_context)
        response = protocol.format_assistant_response(
            decision=decision,
            fraud_score=f"{score:.2f}",
            rationale=rationale,
        )
        trace = [*state.get("trace", []), "fraud_scorer"]
        return {
            "fraud_decision": decision,
            "fraud_score": score,
            "fraud_response": response,
            "trace": trace,
        }

    return fraud_scorer_node
