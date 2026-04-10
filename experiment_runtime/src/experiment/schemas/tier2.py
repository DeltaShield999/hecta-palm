from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any


TRAINING_EXPOSURE_CONDITIONS = ("1x", "10x", "50x")
CHAT_MESSAGE_ROLES = ("system", "user", "assistant")
TRAINING_CORPUS_COLUMNS = (
    "example_id",
    "record_id",
    "canary_id",
    "messages",
    "split",
    "exposure_condition",
)
MIA_EVAL_COLUMNS = (
    "eval_id",
    "record_id",
    "split",
    "is_canary",
    "messages",
)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str

    def to_row(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ChatMessage":
        return cls(
            role=str(row["role"]),
            content=str(row["content"]),
        )


def normalize_messages(raw_messages: Sequence[ChatMessage | Mapping[str, Any]]) -> tuple[ChatMessage, ...]:
    return tuple(
        message if isinstance(message, ChatMessage) else ChatMessage.from_row(message)
        for message in raw_messages
    )


@dataclass(frozen=True, slots=True)
class Stage1TrainingExample:
    example_id: str
    record_id: str
    canary_id: str | None
    messages: tuple[ChatMessage, ...]
    split: str
    exposure_condition: str

    def to_row(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "record_id": self.record_id,
            "canary_id": self.canary_id,
            "messages": [message.to_row() for message in self.messages],
            "split": self.split,
            "exposure_condition": self.exposure_condition,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Stage1TrainingExample":
        return cls(
            example_id=str(row["example_id"]),
            record_id=str(row["record_id"]),
            canary_id=str(row["canary_id"]) if row.get("canary_id") not in (None, "") else None,
            messages=normalize_messages(row["messages"]),
            split=str(row["split"]),
            exposure_condition=str(row["exposure_condition"]),
        )


@dataclass(frozen=True, slots=True)
class MiaEvalExample:
    eval_id: str
    record_id: str
    split: str
    is_canary: bool
    messages: tuple[ChatMessage, ...]

    def to_row(self) -> dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "record_id": self.record_id,
            "split": self.split,
            "is_canary": self.is_canary,
            "messages": [message.to_row() for message in self.messages],
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "MiaEvalExample":
        return cls(
            eval_id=str(row["eval_id"]),
            record_id=str(row["record_id"]),
            split=str(row["split"]),
            is_canary=bool(row["is_canary"]),
            messages=normalize_messages(row["messages"]),
        )


def dataclass_to_row(instance: object) -> dict[str, Any]:
    return {field.name: getattr(instance, field.name) for field in fields(instance)}
