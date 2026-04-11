from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.data_gen.io import read_canary_registry_csv, read_jsonl_rows
from experiment.schemas.stage2 import STAGE2_ATTACK_SPLIT, Stage2AttackPrompt, canonicalize_target_fields
from experiment.schemas.tier1 import CanaryRegistryEntry
from experiment.schemas.tier2 import ChatMessage


STAGE2_RESPONSE_COLUMNS = (
    "attack_id",
    "template_family",
    "target_canary_id",
    "target_fields",
    "exposure_condition",
    "prompt_condition",
    "run_name",
    "message_text",
    "response_text",
    "name_leaked",
    "date_of_birth_leaked",
    "account_number_leaked",
    "any_sensitive_field_leaked",
    "full_canary_record_leaked",
    "other_canary_leak",
    "refusal_emitted",
)
STAGE2_FILTERED_RESPONSE_COLUMNS = (
    "attack_id",
    "template_family",
    "target_canary_id",
    "target_fields",
    "exposure_condition",
    "prompt_condition",
    "run_name",
    "filter_mode",
    "message_text",
    "filter_block_probability",
    "filter_decision",
    "response_generated",
    "response_text",
    "name_leaked",
    "date_of_birth_leaked",
    "account_number_leaked",
    "any_sensitive_field_leaked",
    "full_canary_record_leaked",
    "other_canary_leak",
    "refusal_emitted",
)


@dataclass(frozen=True, slots=True)
class Stage2ResponseRow:
    attack_id: str
    template_family: str
    target_canary_id: str
    target_fields: tuple[str, ...]
    exposure_condition: str
    prompt_condition: str
    run_name: str
    message_text: str
    response_text: str
    name_leaked: int
    date_of_birth_leaked: int
    account_number_leaked: int
    any_sensitive_field_leaked: int
    full_canary_record_leaked: int
    other_canary_leak: int
    refusal_emitted: int

    def to_row(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "template_family": self.template_family,
            "target_canary_id": self.target_canary_id,
            "target_fields": list(self.target_fields),
            "exposure_condition": self.exposure_condition,
            "prompt_condition": self.prompt_condition,
            "run_name": self.run_name,
            "message_text": self.message_text,
            "response_text": self.response_text,
            "name_leaked": self.name_leaked,
            "date_of_birth_leaked": self.date_of_birth_leaked,
            "account_number_leaked": self.account_number_leaked,
            "any_sensitive_field_leaked": self.any_sensitive_field_leaked,
            "full_canary_record_leaked": self.full_canary_record_leaked,
            "other_canary_leak": self.other_canary_leak,
            "refusal_emitted": self.refusal_emitted,
        }


@dataclass(frozen=True, slots=True)
class Stage2FilteredResponseRow:
    attack_id: str
    template_family: str
    target_canary_id: str
    target_fields: tuple[str, ...]
    exposure_condition: str
    prompt_condition: str
    run_name: str
    filter_mode: str
    message_text: str
    filter_block_probability: float
    filter_decision: str
    response_generated: int
    response_text: str
    name_leaked: int
    date_of_birth_leaked: int
    account_number_leaked: int
    any_sensitive_field_leaked: int
    full_canary_record_leaked: int
    other_canary_leak: int
    refusal_emitted: int

    def to_row(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "template_family": self.template_family,
            "target_canary_id": self.target_canary_id,
            "target_fields": list(self.target_fields),
            "exposure_condition": self.exposure_condition,
            "prompt_condition": self.prompt_condition,
            "run_name": self.run_name,
            "filter_mode": self.filter_mode,
            "message_text": self.message_text,
            "filter_block_probability": self.filter_block_probability,
            "filter_decision": self.filter_decision,
            "response_generated": self.response_generated,
            "response_text": self.response_text,
            "name_leaked": self.name_leaked,
            "date_of_birth_leaked": self.date_of_birth_leaked,
            "account_number_leaked": self.account_number_leaked,
            "any_sensitive_field_leaked": self.any_sensitive_field_leaked,
            "full_canary_record_leaked": self.full_canary_record_leaked,
            "other_canary_leak": self.other_canary_leak,
            "refusal_emitted": self.refusal_emitted,
        }


def load_stage2_attack_prompts(
    path: Path,
    *,
    expected_families: Sequence[str],
) -> tuple[Stage2AttackPrompt, ...]:
    rows = read_jsonl_rows(path)
    prompts = tuple(Stage2AttackPrompt.from_row(row) for row in rows)
    if not prompts:
        raise ValueError(f"Stage 2 attack dataset {path} did not yield any prompts.")

    seen_attack_ids: set[str] = set()
    observed_family_order: list[str] = []
    observed_family_set: set[str] = set()
    for prompt in prompts:
        if prompt.attack_id in seen_attack_ids:
            raise ValueError(f"Stage 2 attack dataset contains duplicate attack_id {prompt.attack_id!r}.")
        seen_attack_ids.add(prompt.attack_id)

        if prompt.split != STAGE2_ATTACK_SPLIT:
            raise ValueError(
                f"Stage 2 prompt {prompt.attack_id} had unsupported split {prompt.split!r}; "
                f"expected {STAGE2_ATTACK_SPLIT!r}."
            )
        canonicalize_target_fields(prompt.target_fields)
        if prompt.template_family not in expected_families:
            raise ValueError(
                f"Stage 2 prompt {prompt.attack_id} had unsupported family {prompt.template_family!r}."
            )
        if prompt.template_family not in observed_family_set:
            observed_family_set.add(prompt.template_family)
            observed_family_order.append(prompt.template_family)

    if tuple(observed_family_order) != tuple(expected_families):
        raise ValueError(
            "Stage 2 attack families must appear in the frozen protocol order "
            f"{tuple(expected_families)}, found {tuple(observed_family_order)}."
        )
    return prompts


def load_canary_registry_index(path: Path) -> dict[str, CanaryRegistryEntry]:
    entries = read_canary_registry_csv(path)
    if not entries:
        raise ValueError(f"Canary registry {path} did not yield any rows.")

    index: dict[str, CanaryRegistryEntry] = {}
    for entry in entries:
        if entry.canary_id in index:
            raise ValueError(f"Canary registry contains duplicate canary_id {entry.canary_id!r}.")
        index[entry.canary_id] = entry
    return index


def build_prompt_messages(
    attack_prompt: Stage2AttackPrompt,
    *,
    prompt_condition: str,
    system_prompt: str,
) -> tuple[ChatMessage, ...]:
    if prompt_condition == "no_system_prompt":
        return (ChatMessage(role="user", content=attack_prompt.message_text),)
    if prompt_condition == "system_prompt_active":
        return (
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=attack_prompt.message_text),
        )
    raise ValueError(f"Unsupported prompt condition {prompt_condition!r}.")
