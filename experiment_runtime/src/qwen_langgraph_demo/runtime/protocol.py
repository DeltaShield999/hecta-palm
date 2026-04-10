from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import tomllib


DEFAULT_PROTOCOL_DIR = Path(__file__).resolve().parents[3] / "configs" / "protocol"


@dataclass(frozen=True)
class CoreProtocol:
    protocol_version: str
    model_family: str
    primary_model: str
    secondary_model: str
    exposure_conditions: tuple[str, ...]
    member_records: int
    non_member_records: int
    canary_count: int
    fraud_base_rate: float
    loss_rule: str
    pipeline_nodes: tuple[str, ...]
    runtime_flow: str
    fhe_scheme: str


@dataclass(frozen=True)
class Stage1Protocol:
    system_prompt: str
    user_message_template: str
    benign_request: str
    assistant_response_template: str
    refusal_response: str
    loss_mask: str
    token_scope: str
    corpus_sizes: dict[str, int]
    canary_repetitions: dict[str, int]
    training_budget_control: tuple[str, ...]


@dataclass(frozen=True)
class DecodingConfig:
    temperature: float
    top_p: float
    max_new_tokens: int
    completions_per_prompt: int
    retry_budget: int


@dataclass(frozen=True)
class Stage2Protocol:
    attack_families: tuple[str, ...]
    decoding: DecodingConfig
    headline_metric: str
    secondary_metric: str
    normalize_steps: tuple[str, ...]
    field_match_rules: dict[str, str]


@dataclass(frozen=True)
class DatasetDefaults:
    allow_count: int
    block_count: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    split_policy: str


@dataclass(frozen=True)
class Stage3Protocol:
    allow_rule: str
    block_rule: str
    block_borderline: tuple[str, ...]
    allow_borderline: tuple[str, ...]
    dataset_defaults: DatasetDefaults
    filter_runtime_flow: str
    threshold_location: str
    return_value: str


@dataclass(frozen=True)
class ProtocolBundle:
    core: CoreProtocol
    stage1: Stage1Protocol
    stage2: Stage2Protocol
    stage3: Stage3Protocol

    def render_intake_message(self, transaction_context: Mapping[str, Any], request_text: str) -> str:
        required_keys = (
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
        missing = [key for key in required_keys if key not in transaction_context]
        if missing:
            missing_str = ", ".join(missing)
            raise KeyError(f"Missing transaction fields for intake template: {missing_str}")

        payload = {key: str(value) for key, value in transaction_context.items()}
        payload["request"] = request_text
        return self.stage1.user_message_template.format(**payload)

    def format_assistant_response(self, *, decision: str, fraud_score: str, rationale: str) -> str:
        return self.stage1.assistant_response_template.format(
            decision=decision,
            fraud_score=fraud_score,
            rationale=rationale,
        )


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_protocol_bundle(config_dir: Path | str | None = None) -> ProtocolBundle:
    resolved_dir = Path(config_dir) if config_dir is not None else DEFAULT_PROTOCOL_DIR

    core_doc = _read_toml(resolved_dir / "core.toml")
    stage1_doc = _read_toml(resolved_dir / "stage1.toml")
    stage2_doc = _read_toml(resolved_dir / "stage2.toml")
    stage3_doc = _read_toml(resolved_dir / "stage3.toml")

    core = CoreProtocol(
        protocol_version=core_doc["metadata"]["protocol_version"],
        model_family=core_doc["models"]["family"],
        primary_model=core_doc["models"]["primary"],
        secondary_model=core_doc["models"]["secondary"],
        exposure_conditions=tuple(core_doc["stage1"]["exposure_conditions"]),
        member_records=core_doc["dataset"]["member_records"],
        non_member_records=core_doc["dataset"]["non_member_records"],
        canary_count=core_doc["dataset"]["canary_count"],
        fraud_base_rate=core_doc["dataset"]["fraud_base_rate"],
        loss_rule=core_doc["stage1"]["loss_rule"],
        pipeline_nodes=tuple(core_doc["pipeline"]["nodes"]),
        runtime_flow=core_doc["pipeline"]["runtime_flow"],
        fhe_scheme=core_doc["pipeline"]["fhe_scheme"],
    )

    stage1 = Stage1Protocol(
        system_prompt=stage1_doc["prompts"]["system_prompt"].strip(),
        user_message_template=stage1_doc["prompts"]["user_message_template"].strip(),
        benign_request=stage1_doc["prompts"]["benign_request"].strip(),
        assistant_response_template=stage1_doc["prompts"]["assistant_response_template"].strip(),
        refusal_response=stage1_doc["prompts"]["refusal_response"].strip(),
        loss_mask=stage1_doc["training"]["loss_mask"],
        token_scope=stage1_doc["training"]["token_scope"],
        corpus_sizes={key: int(value) for key, value in stage1_doc["exposure"]["corpus_sizes"].items()},
        canary_repetitions={key: int(value) for key, value in stage1_doc["exposure"]["canary_repetitions"].items()},
        training_budget_control=tuple(stage1_doc["exposure"]["training_budget_control"]),
    )

    stage2 = Stage2Protocol(
        attack_families=tuple(stage2_doc["stage2"]["attack_families"]),
        decoding=DecodingConfig(
            temperature=float(stage2_doc["decoding"]["temperature"]),
            top_p=float(stage2_doc["decoding"]["top_p"]),
            max_new_tokens=int(stage2_doc["decoding"]["max_new_tokens"]),
            completions_per_prompt=int(stage2_doc["decoding"]["completions_per_prompt"]),
            retry_budget=int(stage2_doc["decoding"]["retry_budget"]),
        ),
        headline_metric=stage2_doc["leakage_scoring"]["headline_metric"],
        secondary_metric=stage2_doc["leakage_scoring"]["secondary_metric"],
        normalize_steps=tuple(stage2_doc["leakage_scoring"]["normalize_steps"]),
        field_match_rules={key: str(value) for key, value in stage2_doc["leakage_scoring"]["field_match_rules"].items()},
    )

    stage3 = Stage3Protocol(
        allow_rule=stage3_doc["labeling"]["allow_rule"],
        block_rule=stage3_doc["labeling"]["block_rule"],
        block_borderline=tuple(stage3_doc["labeling"]["block_borderline"]),
        allow_borderline=tuple(stage3_doc["labeling"]["allow_borderline"]),
        dataset_defaults=DatasetDefaults(
            allow_count=int(stage3_doc["dataset_defaults"]["allow_count"]),
            block_count=int(stage3_doc["dataset_defaults"]["block_count"]),
            train_fraction=float(stage3_doc["dataset_defaults"]["train_fraction"]),
            validation_fraction=float(stage3_doc["dataset_defaults"]["validation_fraction"]),
            test_fraction=float(stage3_doc["dataset_defaults"]["test_fraction"]),
            split_policy=stage3_doc["dataset_defaults"]["split_policy"],
        ),
        filter_runtime_flow=stage3_doc["filter_runtime"]["runtime_flow"],
        threshold_location=stage3_doc["filter_runtime"]["threshold_location"],
        return_value=stage3_doc["filter_runtime"]["return_value"],
    )

    return ProtocolBundle(core=core, stage1=stage1, stage2=stage2, stage3=stage3)
