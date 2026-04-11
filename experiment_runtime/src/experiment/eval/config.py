from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import tomllib

from experiment.fhe.config import PINNED_FHE_BACKEND, PINNED_FHE_SCHEME, PINNED_SECURITY_LEVEL, OpenFheSettings
from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle

from experiment.train_qwen.config import EXPOSURE_CONDITIONS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE2_REPLAY_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval" / "stage2_replay.toml"
DEFAULT_STAGE2_FILTERED_REPLAY_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "eval" / "stage2_filtered_replay.toml"
)
PROMPT_CONDITIONS = ("no_system_prompt", "system_prompt_active")
FILTER_MODES = ("plaintext_filter", "fhe_filter")


@dataclass(frozen=True, slots=True)
class ModelSettings:
    name: str
    trust_remote_code: bool
    attn_implementation: str


@dataclass(frozen=True, slots=True)
class TokenizerSettings:
    source: str
    use_fast: bool
    add_generation_prompt: bool
    padding_side: str
    truncation_side: str
    max_sequence_length: int


@dataclass(frozen=True, slots=True)
class DecodingSettings:
    temperature: float
    top_p: float
    max_new_tokens: int
    completions_per_prompt: int
    retry_budget: int


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    batch_size: int
    bf16: bool
    tf32: bool


@dataclass(frozen=True, slots=True)
class OfficialRunReference:
    exposure_condition: str
    run_dir: Path
    run_name: str
    adapter_dir: Path


@dataclass(frozen=True, slots=True)
class FilterEncoderSettings:
    batch_size: int
    device: str


@dataclass(frozen=True, slots=True)
class PlaintextFilterReference:
    metrics_path: Path
    model_parameters_path: Path


@dataclass(frozen=True, slots=True)
class FheFilterReference:
    compiled_bundle_dir: Path
    compiled_bundle_manifest_path: Path


@dataclass(frozen=True, slots=True)
class Stage2ReplayConfig:
    protocol_config_dir: Path
    attack_dataset_path: Path
    canary_registry_path: Path
    output_root: Path
    model: ModelSettings
    tokenizer: TokenizerSettings
    decoding: DecodingSettings
    inference: InferenceSettings
    official_runs: dict[str, OfficialRunReference]
    seed: int

    @classmethod
    def from_toml(cls, config_path: Path | str | None = None) -> "Stage2ReplayConfig":
        path = _resolve_path(config_path or DEFAULT_STAGE2_REPLAY_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        protocol_dir = _resolve_path(document["protocol"]["config_dir"])
        protocol = load_protocol_bundle(protocol_dir)

        model = ModelSettings(
            name=str(document["model"]["name"]),
            trust_remote_code=bool(document["model"]["trust_remote_code"]),
            attn_implementation=str(document["model"]["attn_implementation"]),
        )
        if model.name != protocol.core.primary_model:
            raise ValueError(
                "Stage 2 replay must target the frozen primary model "
                f"{protocol.core.primary_model}, found {model.name}."
            )

        tokenizer = TokenizerSettings(
            source=str(document["tokenizer"]["source"]),
            use_fast=bool(document["tokenizer"]["use_fast"]),
            add_generation_prompt=bool(document["tokenizer"]["add_generation_prompt"]),
            padding_side=str(document["tokenizer"]["padding_side"]),
            truncation_side=str(document["tokenizer"]["truncation_side"]),
            max_sequence_length=int(document["tokenizer"]["max_sequence_length"]),
        )
        if not tokenizer.add_generation_prompt:
            raise ValueError("Stage 2 replay must set tokenizer.add_generation_prompt = true.")
        if tokenizer.max_sequence_length <= 0:
            raise ValueError("tokenizer.max_sequence_length must be positive.")

        decoding = DecodingSettings(
            temperature=float(document["decoding"]["temperature"]),
            top_p=float(document["decoding"]["top_p"]),
            max_new_tokens=int(document["decoding"]["max_new_tokens"]),
            completions_per_prompt=int(document["decoding"]["completions_per_prompt"]),
            retry_budget=int(document["decoding"]["retry_budget"]),
        )
        protocol_decoding = protocol.stage2.decoding
        if (
            decoding.temperature != protocol_decoding.temperature
            or decoding.top_p != protocol_decoding.top_p
            or decoding.max_new_tokens != protocol_decoding.max_new_tokens
            or decoding.completions_per_prompt != protocol_decoding.completions_per_prompt
            or decoding.retry_budget != protocol_decoding.retry_budget
        ):
            raise ValueError("Stage 2 replay decoding settings must match the frozen protocol exactly.")

        inference = InferenceSettings(
            batch_size=int(document["inference"]["batch_size"]),
            bf16=bool(document["inference"]["bf16"]),
            tf32=bool(document["inference"]["tf32"]),
        )
        if inference.batch_size <= 0:
            raise ValueError("inference.batch_size must be positive.")

        official_run_dirs = {
            str(exposure): _resolve_path(run_dir)
            for exposure, run_dir in document["inputs"]["official_run_dirs"].items()
        }
        if tuple(official_run_dirs) != EXPOSURE_CONDITIONS:
            raise ValueError(
                "Stage 2 replay config must define official run dirs for exposure conditions "
                f"{EXPOSURE_CONDITIONS}, found {tuple(official_run_dirs)}."
            )

        official_runs = {
            exposure: _load_official_run_reference(
                exposure_condition=exposure,
                run_dir=run_dir,
                expected_base_model_name=model.name,
            )
            for exposure, run_dir in official_run_dirs.items()
        }

        return cls(
            protocol_config_dir=protocol_dir,
            attack_dataset_path=_resolve_path(document["inputs"]["attack_dataset_path"]),
            canary_registry_path=_resolve_path(document["inputs"]["canary_registry_path"]),
            output_root=_resolve_path(document["outputs"]["root_dir"]),
            model=model,
            tokenizer=tokenizer,
            decoding=decoding,
            inference=inference,
            official_runs=official_runs,
            seed=int(document["seed"]["value"]),
        )


@dataclass(frozen=True, slots=True)
class Stage2FilteredReplayConfig:
    config_path: Path
    protocol_config_dir: Path
    attack_dataset_path: Path
    canary_registry_path: Path
    baseline_summary_path: Path
    output_root: Path
    model: ModelSettings
    tokenizer: TokenizerSettings
    decoding: DecodingSettings
    inference: InferenceSettings
    official_runs: dict[str, OfficialRunReference]
    filter_encoder: FilterEncoderSettings
    plaintext_filter: PlaintextFilterReference
    fhe_filter: FheFilterReference
    fhe: OpenFheSettings
    seed: int

    @classmethod
    def from_toml(cls, config_path: Path | str | None = None) -> "Stage2FilteredReplayConfig":
        path = _resolve_path(config_path or DEFAULT_STAGE2_FILTERED_REPLAY_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        protocol_dir = _resolve_path(document["protocol"]["config_dir"])
        protocol = load_protocol_bundle(protocol_dir)

        model = ModelSettings(
            name=str(document["model"]["name"]),
            trust_remote_code=bool(document["model"]["trust_remote_code"]),
            attn_implementation=str(document["model"]["attn_implementation"]),
        )
        if model.name != protocol.core.primary_model:
            raise ValueError(
                "Stage 2 filtered replay must target the frozen primary model "
                f"{protocol.core.primary_model}, found {model.name}."
            )

        tokenizer = TokenizerSettings(
            source=str(document["tokenizer"]["source"]),
            use_fast=bool(document["tokenizer"]["use_fast"]),
            add_generation_prompt=bool(document["tokenizer"]["add_generation_prompt"]),
            padding_side=str(document["tokenizer"]["padding_side"]),
            truncation_side=str(document["tokenizer"]["truncation_side"]),
            max_sequence_length=int(document["tokenizer"]["max_sequence_length"]),
        )
        if not tokenizer.add_generation_prompt:
            raise ValueError("Stage 2 filtered replay must set tokenizer.add_generation_prompt = true.")
        if tokenizer.max_sequence_length <= 0:
            raise ValueError("tokenizer.max_sequence_length must be positive.")

        decoding = DecodingSettings(
            temperature=float(document["decoding"]["temperature"]),
            top_p=float(document["decoding"]["top_p"]),
            max_new_tokens=int(document["decoding"]["max_new_tokens"]),
            completions_per_prompt=int(document["decoding"]["completions_per_prompt"]),
            retry_budget=int(document["decoding"]["retry_budget"]),
        )
        protocol_decoding = protocol.stage2.decoding
        if (
            decoding.temperature != protocol_decoding.temperature
            or decoding.top_p != protocol_decoding.top_p
            or decoding.max_new_tokens != protocol_decoding.max_new_tokens
            or decoding.completions_per_prompt != protocol_decoding.completions_per_prompt
            or decoding.retry_budget != protocol_decoding.retry_budget
        ):
            raise ValueError(
                "Stage 2 filtered replay decoding settings must match the frozen protocol exactly."
            )

        inference = InferenceSettings(
            batch_size=int(document["inference"]["batch_size"]),
            bf16=bool(document["inference"]["bf16"]),
            tf32=bool(document["inference"]["tf32"]),
        )
        if inference.batch_size <= 0:
            raise ValueError("inference.batch_size must be positive.")

        official_run_dirs = {
            str(exposure): _resolve_path(run_dir)
            for exposure, run_dir in document["inputs"]["official_run_dirs"].items()
        }
        if tuple(official_run_dirs) != EXPOSURE_CONDITIONS:
            raise ValueError(
                "Stage 2 filtered replay config must define official run dirs for exposure conditions "
                f"{EXPOSURE_CONDITIONS}, found {tuple(official_run_dirs)}."
            )
        official_runs = {
            exposure: _load_official_run_reference(
                exposure_condition=exposure,
                run_dir=run_dir,
                expected_base_model_name=model.name,
            )
            for exposure, run_dir in official_run_dirs.items()
        }

        baseline_summary_path = _resolve_existing_path(document["inputs"]["baseline_summary_path"])
        _validate_baseline_summary(
            baseline_summary_path,
            expected_exposures=tuple(official_runs),
        )

        filter_encoder = FilterEncoderSettings(
            batch_size=int(document["filter"]["encoder_batch_size"]),
            device=str(document["filter"]["encoder_device"]),
        )
        if filter_encoder.batch_size <= 0:
            raise ValueError("filter.encoder_batch_size must be positive.")

        plaintext_filter = PlaintextFilterReference(
            metrics_path=_resolve_existing_path(document["inputs"]["plaintext_filter"]["metrics_path"]),
            model_parameters_path=_resolve_existing_path(
                document["inputs"]["plaintext_filter"]["model_parameters_path"]
            ),
        )
        fhe_filter = FheFilterReference(
            compiled_bundle_dir=_resolve_path(document["inputs"]["fhe_filter"]["compiled_bundle_dir"]),
            compiled_bundle_manifest_path=_resolve_existing_path(
                document["inputs"]["fhe_filter"]["compiled_bundle_manifest_path"]
            ),
        )

        fhe = OpenFheSettings(
            backend=str(document["fhe"]["backend"]),
            scheme=str(document["fhe"]["scheme"]),
            ring_dimension=int(document["fhe"]["ring_dimension"]),
            multiplicative_depth=int(document["fhe"]["multiplicative_depth"]),
            scaling_mod_size=int(document["fhe"]["scaling_mod_size"]),
            first_mod_size=int(document["fhe"]["first_mod_size"]),
            batch_size=int(document["fhe"]["batch_size"]),
            security_level=str(document["fhe"]["security_level"]),
        )
        _validate_fhe_settings(fhe)

        return cls(
            config_path=path,
            protocol_config_dir=protocol_dir,
            attack_dataset_path=_resolve_existing_path(document["inputs"]["attack_dataset_path"]),
            canary_registry_path=_resolve_existing_path(document["inputs"]["canary_registry_path"]),
            baseline_summary_path=baseline_summary_path,
            output_root=_resolve_path(document["outputs"]["root_dir"]),
            model=model,
            tokenizer=tokenizer,
            decoding=decoding,
            inference=inference,
            official_runs=official_runs,
            filter_encoder=filter_encoder,
            plaintext_filter=plaintext_filter,
            fhe_filter=fhe_filter,
            fhe=fhe,
            seed=int(document["seed"]["value"]),
        )


def resolve_exposure_conditions(exposure: str) -> tuple[str, ...]:
    if exposure == "all":
        return EXPOSURE_CONDITIONS
    if exposure not in EXPOSURE_CONDITIONS:
        raise ValueError(
            f"Unsupported exposure condition {exposure!r}; expected one of {EXPOSURE_CONDITIONS} or 'all'."
        )
    return (exposure,)


def resolve_prompt_conditions(condition: str) -> tuple[str, ...]:
    if condition == "all":
        return PROMPT_CONDITIONS
    if condition not in PROMPT_CONDITIONS:
        raise ValueError(
            f"Unsupported prompt condition {condition!r}; "
            f"expected one of {PROMPT_CONDITIONS} or 'all'."
        )
    return (condition,)


def resolve_filter_modes(filter_mode: str) -> tuple[str, ...]:
    if filter_mode == "all":
        return FILTER_MODES
    if filter_mode not in FILTER_MODES:
        raise ValueError(
            f"Unsupported filter mode {filter_mode!r}; expected one of {FILTER_MODES} or 'all'."
        )
    return (filter_mode,)


def _load_official_run_reference(
    *,
    exposure_condition: str,
    run_dir: Path,
    expected_base_model_name: str,
) -> OfficialRunReference:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Official Stage 1 run metadata is missing: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["exposure_condition"] != exposure_condition:
        raise ValueError(
            f"Official run {run_dir} reports exposure {metadata['exposure_condition']!r}; "
            f"expected {exposure_condition!r}."
        )
    if metadata["base_model_name"] != expected_base_model_name:
        raise ValueError(
            f"Official run {run_dir} reports base model {metadata['base_model_name']!r}; "
            f"expected {expected_base_model_name!r}."
        )
    if bool(metadata.get("smoke_enabled")):
        raise ValueError(f"Official run {run_dir} is a smoke run and cannot be used for Stage 2 replay.")

    adapter_dir = run_dir / "adapter_model"
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Official adapter config is missing: {adapter_config_path}")

    return OfficialRunReference(
        exposure_condition=exposure_condition,
        run_dir=run_dir,
        run_name=str(metadata["run_name"]),
        adapter_dir=adapter_dir,
    )


def _validate_baseline_summary(path: Path, *, expected_exposures: tuple[str, ...]) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    runs = tuple(summary.get("runs", ()))
    indexed = {
        (str(run["exposure_condition"]), str(run["prompt_condition"])): run
        for run in runs
    }
    for exposure in expected_exposures:
        key = (exposure, "system_prompt_active")
        if key not in indexed:
            raise ValueError(
                "Stage 2 filtered replay requires an existing system_prompt_active baseline for "
                f"{exposure}, but {path} did not include one."
            )


def _validate_fhe_settings(fhe: OpenFheSettings) -> None:
    if fhe.backend != PINNED_FHE_BACKEND:
        raise ValueError(
            f"Stage 2 filtered replay must use backend {PINNED_FHE_BACKEND!r}, found {fhe.backend!r}."
        )
    if fhe.scheme != PINNED_FHE_SCHEME:
        raise ValueError(
            f"Stage 2 filtered replay must use scheme {PINNED_FHE_SCHEME!r}, found {fhe.scheme!r}."
        )
    if fhe.ring_dimension <= 0 or fhe.ring_dimension % 2 != 0:
        raise ValueError("fhe.ring_dimension must be a positive even integer.")
    if fhe.multiplicative_depth <= 0:
        raise ValueError("fhe.multiplicative_depth must be positive.")
    if fhe.scaling_mod_size <= 0 or fhe.first_mod_size <= 0:
        raise ValueError("fhe scaling modulus sizes must be positive.")
    if fhe.batch_size <= 0:
        raise ValueError("fhe.batch_size must be positive.")
    if fhe.security_level != PINNED_SECURITY_LEVEL:
        raise ValueError(
            "Stage 2 filtered replay must use security_level "
            f"{PINNED_SECURITY_LEVEL!r}, found {fhe.security_level!r}."
        )


def _resolve_path(value: Path | str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_existing_path(value: Path | str) -> Path:
    path = _resolve_path(value)
    if not path.exists():
        raise FileNotFoundError(f"Expected path is missing: {path}")
    return path
