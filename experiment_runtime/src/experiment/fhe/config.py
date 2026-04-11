from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE3_FHE_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval" / "stage3_fhe_filter.toml"
PINNED_FHE_BACKEND = "openfhe"
PINNED_FHE_SCHEME = "CKKS"
PINNED_SECURITY_LEVEL = "HEStd_128_classic"


@dataclass(frozen=True, slots=True)
class PlaintextArtifactPaths:
    plaintext_metrics_path: Path
    model_parameters_path: Path
    test_embeddings_path: Path
    val_embeddings_path: Path
    test_predictions_path: Path


@dataclass(frozen=True, slots=True)
class OpenFheSettings:
    backend: str
    scheme: str
    ring_dimension: int
    multiplicative_depth: int
    scaling_mod_size: int
    first_mod_size: int
    batch_size: int
    security_level: str


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    split_name: str
    example_count: int


@dataclass(frozen=True, slots=True)
class Stage3FheConfig:
    config_path: Path
    plaintext_artifacts: PlaintextArtifactPaths
    output_root: Path
    fhe: OpenFheSettings
    benchmark: BenchmarkSettings
    seed: int

    @classmethod
    def from_toml(cls, config_path: Path | str | None = None) -> "Stage3FheConfig":
        path = _resolve_path(config_path or DEFAULT_STAGE3_FHE_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        plaintext_artifacts = PlaintextArtifactPaths(
            plaintext_metrics_path=_resolve_existing_path(
                document["inputs"]["plaintext_metrics_path"]
            ),
            model_parameters_path=_resolve_existing_path(
                document["inputs"]["model_parameters_path"]
            ),
            test_embeddings_path=_resolve_existing_path(
                document["inputs"]["test_embeddings_path"]
            ),
            val_embeddings_path=_resolve_existing_path(
                document["inputs"]["val_embeddings_path"]
            ),
            test_predictions_path=_resolve_existing_path(
                document["inputs"]["test_predictions_path"]
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
        if fhe.backend != PINNED_FHE_BACKEND:
            raise ValueError(
                f"Stage 3 FHE evaluation must use backend {PINNED_FHE_BACKEND!r}, "
                f"found {fhe.backend!r}."
            )
        if fhe.scheme != PINNED_FHE_SCHEME:
            raise ValueError(
                f"Stage 3 FHE evaluation must use scheme {PINNED_FHE_SCHEME!r}, "
                f"found {fhe.scheme!r}."
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
                "Stage 3 FHE evaluation must use security_level "
                f"{PINNED_SECURITY_LEVEL!r}, found {fhe.security_level!r}."
            )

        benchmark = BenchmarkSettings(
            split_name=str(document["benchmark"]["split_name"]),
            example_count=int(document["benchmark"]["example_count"]),
        )
        if benchmark.split_name != "test":
            raise ValueError("Stage 3 FHE benchmark must run on the held-out test split.")
        if benchmark.example_count <= 0:
            raise ValueError("benchmark.example_count must be positive.")

        return cls(
            config_path=path,
            plaintext_artifacts=plaintext_artifacts,
            output_root=_resolve_path(document["outputs"]["root_dir"]),
            fhe=fhe,
            benchmark=benchmark,
            seed=int(document["seed"]["value"]),
        )


def _resolve_existing_path(value: Path | str) -> Path:
    path = _resolve_path(value)
    if not path.exists():
        raise FileNotFoundError(f"Stage 3 FHE config references a missing path: {path}")
    return path


def _resolve_path(value: Path | str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
