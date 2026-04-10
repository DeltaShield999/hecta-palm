from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from experiment.schemas.tier1 import Tier1Record
from experiment.schemas.tier2 import ChatMessage
from qwen_langgraph_demo.runtime.protocol import ProtocolBundle


HIGH_RISK_CATEGORIES = frozenset(
    {
        "electronics",
        "travel",
        "online_marketplace",
        "wire_transfer",
    }
)
HIGH_RISK_GEOS = frozenset(
    {
        "Las Vegas, NV, USA",
        "Miami, FL, USA",
        "Orlando, FL, USA",
    }
)
INTAKE_MESSAGE_FIELDS = (
    "account_id",
    "customer_name",
    "date_of_birth",
    "account_number",
    "amount",
    "merchant_category",
    "timestamp",
    "geo_location",
    "device_fingerprint",
)


@dataclass(frozen=True, slots=True)
class BenignFraudRender:
    decision: str
    fraud_score: float
    rationale: str

    @property
    def formatted_score(self) -> str:
        return f"{self.fraud_score:.2f}"

    def render(self, protocol: ProtocolBundle) -> str:
        return protocol.format_assistant_response(
            decision=self.decision,
            fraud_score=self.formatted_score,
            rationale=self.rationale,
        )


def render_benign_chat_messages(
    protocol: ProtocolBundle,
    transaction_context: Tier1Record | Mapping[str, Any],
    *,
    request_text: str | None = None,
) -> tuple[ChatMessage, ChatMessage, ChatMessage]:
    normalized_context = normalize_transaction_context(transaction_context)
    intake_request = request_text or protocol.stage1.benign_request
    assistant_response = render_benign_assistant_response(protocol, normalized_context)
    return (
        ChatMessage(role="system", content=protocol.stage1.system_prompt),
        ChatMessage(
            role="user",
            content=protocol.render_intake_message(normalized_context, intake_request),
        ),
        ChatMessage(role="assistant", content=assistant_response),
    )


def render_benign_assistant_response(
    protocol: ProtocolBundle,
    transaction_context: Tier1Record | Mapping[str, Any],
) -> str:
    return assess_benign_transaction(transaction_context).render(protocol)


def assess_benign_transaction(
    transaction_context: Tier1Record | Mapping[str, Any],
) -> BenignFraudRender:
    normalized_context = normalize_transaction_context(transaction_context)
    amount = _parse_amount(normalized_context["amount"])
    category = _normalize_category(normalized_context["merchant_category"])
    geo_location = str(normalized_context["geo_location"]).strip()
    hour = _extract_hour(normalized_context["timestamp"])

    score = 0.05
    risk_signals: list[str] = []
    steady_signals: list[str] = []

    if amount >= 4_000:
        score += 0.38
        risk_signals.append("an unusually large transaction amount")
    elif amount >= 1_800:
        score += 0.18
        risk_signals.append("an above-typical transaction amount")
    else:
        steady_signals.append("the amount stays within a routine consumer range")

    category_label = category.replace("_", " ")
    if category in HIGH_RISK_CATEGORIES:
        score += 0.22
        risk_signals.append(f"the {category_label} merchant category")
    else:
        steady_signals.append(f"the {category_label} merchant category looks typical")

    if geo_location in HIGH_RISK_GEOS:
        score += 0.12
        risk_signals.append("a geography that often sees elevated fraud pressure")
    else:
        steady_signals.append("the location signal is not unusual")

    if hour < 5 or hour >= 23:
        score += 0.18
        risk_signals.append("off-hours transaction timing")
    elif hour < 7 or hour >= 21:
        score += 0.08
        risk_signals.append("slightly unusual transaction timing")
    else:
        steady_signals.append("the timestamp sits in a normal activity window")

    score = min(max(score, 0.01), 0.99)

    if score >= 0.75:
        decision = "BLOCK"
    elif score >= 0.35:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    rationale = _build_rationale(decision, risk_signals, steady_signals)
    return BenignFraudRender(
        decision=decision,
        fraud_score=score,
        rationale=rationale,
    )


def normalize_transaction_context(
    transaction_context: Tier1Record | Mapping[str, Any],
) -> dict[str, object]:
    raw_context: Mapping[str, Any]
    if isinstance(transaction_context, Tier1Record):
        raw_context = transaction_context.to_row()
    else:
        raw_context = transaction_context

    missing = [field for field in INTAKE_MESSAGE_FIELDS if field not in raw_context]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing transaction fields for benign renderer: {missing_str}")

    return {field: raw_context[field] for field in INTAKE_MESSAGE_FIELDS}


def _parse_amount(raw_amount: Any) -> float:
    if isinstance(raw_amount, (int, float)):
        return float(raw_amount)
    return float(str(raw_amount).replace(",", "").replace("$", "").strip())


def _normalize_category(raw_category: Any) -> str:
    return str(raw_category).strip().lower().replace(" ", "_")


def _extract_hour(raw_timestamp: Any) -> int:
    timestamp = str(raw_timestamp).strip()
    return int(timestamp[11:13])


def _build_rationale(
    decision: str,
    risk_signals: list[str],
    steady_signals: list[str],
) -> str:
    if decision == "BLOCK":
        first_sentence = "The visible transaction details show multiple high-risk signals, so this payment should be blocked."
    elif decision == "REVIEW":
        first_sentence = "The visible transaction details contain enough risk indicators to justify manual review."
    else:
        first_sentence = "The visible transaction details look consistent with routine account activity."

    if risk_signals:
        second_sentence = "The score is driven by " + _join_phrases(risk_signals) + "."
    else:
        second_sentence = "The amount, merchant context, location, and timing do not show unusual risk cues."

    if decision == "ALLOW" and steady_signals:
        second_sentence = "The low score reflects that " + _join_phrases(steady_signals) + "."

    return f"{first_sentence} {second_sentence}"


def _join_phrases(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"
