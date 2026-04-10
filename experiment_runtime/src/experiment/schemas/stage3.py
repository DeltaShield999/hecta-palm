from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any


STAGE3_MESSAGE_COLUMNS = (
    "message_id",
    "message_text",
    "label",
    "template_family",
    "source_type",
)

STAGE3_ALLOW_LABEL = "ALLOW"
STAGE3_BLOCK_LABEL = "BLOCK"
STAGE3_LABELS = (
    STAGE3_ALLOW_LABEL,
    STAGE3_BLOCK_LABEL,
)

STAGE3_SOURCE_TYPE_BY_LABEL = {
    STAGE3_ALLOW_LABEL: "benign",
    STAGE3_BLOCK_LABEL: "adversarial",
}

STAGE3_ALLOW_FAMILIES = (
    "transaction scoring",
    "routing recommendation",
    "risk summary",
    "mismatch explanation",
    "audit or compliance risk review",
)

STAGE3_BLOCK_FAMILIES = (
    "direct injection",
    "compliance or audit framing",
    "role-play",
    "urgency or escalation",
    "subtle or indirect extraction",
)

STAGE3_ALLOW_FAMILY_SLUGS = {
    "transaction scoring": "transaction_scoring",
    "routing recommendation": "routing_recommendation",
    "risk summary": "risk_summary",
    "mismatch explanation": "mismatch_explanation",
    "audit or compliance risk review": "audit_or_compliance_risk_review",
}

STAGE3_BLOCK_FAMILY_SLUGS = {
    "direct injection": "direct_injection",
    "compliance or audit framing": "compliance_or_audit_framing",
    "role-play": "role_play",
    "urgency or escalation": "urgency_or_escalation",
    "subtle or indirect extraction": "subtle_or_indirect_extraction",
}

STAGE3_FAMILY_ORDER = (
    (STAGE3_ALLOW_LABEL, "transaction scoring", "transaction_scoring"),
    (STAGE3_ALLOW_LABEL, "routing recommendation", "routing_recommendation"),
    (STAGE3_ALLOW_LABEL, "risk summary", "risk_summary"),
    (STAGE3_ALLOW_LABEL, "mismatch explanation", "mismatch_explanation"),
    (STAGE3_ALLOW_LABEL, "audit or compliance risk review", "audit_or_compliance_risk_review"),
    (STAGE3_BLOCK_LABEL, "direct injection", "direct_injection"),
    (STAGE3_BLOCK_LABEL, "compliance or audit framing", "compliance_or_audit_framing"),
    (STAGE3_BLOCK_LABEL, "role-play", "role_play"),
    (STAGE3_BLOCK_LABEL, "urgency or escalation", "urgency_or_escalation"),
    (STAGE3_BLOCK_LABEL, "subtle or indirect extraction", "subtle_or_indirect_extraction"),
)

STAGE3_CLUSTERS_PER_FAMILY = 20
STAGE3_VARIANTS_PER_CLUSTER = 10
STAGE3_ROWS_PER_FAMILY = STAGE3_CLUSTERS_PER_FAMILY * STAGE3_VARIANTS_PER_CLUSTER
STAGE3_ROWS_PER_LABEL = 1000
STAGE3_TOTAL_ROWS = 2000

STAGE3_SPLITS = (
    "train",
    "val",
    "test",
)

STAGE3_SPLIT_CLUSTER_RANGES = {
    "train": tuple(range(1, 15)),
    "val": tuple(range(15, 18)),
    "test": tuple(range(18, 21)),
}

STAGE3_ROWS_PER_FAMILY_BY_SPLIT = {
    "train": 140,
    "val": 30,
    "test": 30,
}

STAGE3_ROWS_PER_LABEL_BY_SPLIT = {
    "train": 700,
    "val": 150,
    "test": 150,
}

STAGE3_ROWS_BY_SPLIT = {
    "train": 1400,
    "val": 300,
    "test": 300,
}

STAGE3_MESSAGE_ID_PATTERN = re.compile(
    r"^stage3_(allow|block)_[a-z_]+_c(0[1-9]|1[0-9]|20)_v(0[1-9]|10)$"
)


@dataclass(frozen=True, slots=True)
class Stage3FilterMessage:
    message_id: str
    message_text: str
    label: str
    template_family: str
    source_type: str

    def to_row(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_text": self.message_text,
            "label": self.label,
            "template_family": self.template_family,
            "source_type": self.source_type,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Stage3FilterMessage":
        return cls(
            message_id=str(row["message_id"]),
            message_text=str(row["message_text"]),
            label=str(row["label"]),
            template_family=str(row["template_family"]),
            source_type=str(row["source_type"]),
        )


def split_for_cluster_index(cluster_index: int) -> str:
    if cluster_index in STAGE3_SPLIT_CLUSTER_RANGES["train"]:
        return "train"
    if cluster_index in STAGE3_SPLIT_CLUSTER_RANGES["val"]:
        return "val"
    if cluster_index in STAGE3_SPLIT_CLUSTER_RANGES["test"]:
        return "test"
    raise ValueError(
        "Stage 3 cluster_index must be between 1 and 20 inclusive, "
        f"found {cluster_index}."
    )
