from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
import time

import numpy as np

from .config import OpenFheSettings
from .data import PlaintextModelParameters
from .metrics import LatencyBreakdown


@dataclass(frozen=True, slots=True)
class ResolvedCkksParameters:
    backend: str
    scheme: str
    ring_dimension: int
    batch_size: int
    multiplicative_depth: int
    scaling_mod_size: int
    first_mod_size: int
    security_level: str

    def to_document(self) -> dict[str, str | int]:
        return {
            "backend": self.backend,
            "scheme": self.scheme,
            "ring_dimension": self.ring_dimension,
            "batch_size": self.batch_size,
            "multiplicative_depth": self.multiplicative_depth,
            "scaling_mod_size": self.scaling_mod_size,
            "first_mod_size": self.first_mod_size,
            "security_level": self.security_level,
        }


@dataclass(frozen=True, slots=True)
class OpenFheBundlePaths:
    root_dir: Path
    crypto_context_path: Path
    public_key_path: Path
    secret_key_path: Path
    eval_mult_key_path: Path
    eval_automorphism_key_path: Path

    @classmethod
    def for_root(cls, root_dir: Path) -> "OpenFheBundlePaths":
        return cls(
            root_dir=root_dir,
            crypto_context_path=root_dir / "crypto_context.bin",
            public_key_path=root_dir / "public_key.bin",
            secret_key_path=root_dir / "secret_key.bin",
            eval_mult_key_path=root_dir / "eval_mult_key.bin",
            eval_automorphism_key_path=root_dir / "eval_automorphism_key.bin",
        )

    def all_files(self) -> tuple[Path, ...]:
        return (
            self.crypto_context_path,
            self.public_key_path,
            self.secret_key_path,
            self.eval_mult_key_path,
            self.eval_automorphism_key_path,
        )

    def is_complete(self) -> bool:
        return all(path.exists() for path in self.all_files())

    def to_document(self) -> dict[str, str]:
        return {
            "root_dir": str(self.root_dir),
            "crypto_context_path": str(self.crypto_context_path),
            "public_key_path": str(self.public_key_path),
            "secret_key_path": str(self.secret_key_path),
            "eval_mult_key_path": str(self.eval_mult_key_path),
            "eval_automorphism_key_path": str(self.eval_automorphism_key_path),
        }


class OpenFheCkksScorer:
    def __init__(
        self,
        *,
        openfhe_module: ModuleType,
        model_parameters: PlaintextModelParameters,
        crypto_context: Any,
        public_key: Any,
        secret_key: Any,
        resolved_parameters: ResolvedCkksParameters,
        bundle_paths: OpenFheBundlePaths | None,
        reused_existing_bundle: bool,
    ) -> None:
        self._openfhe = openfhe_module
        self._cc = crypto_context
        self._public_key = public_key
        self._secret_key = secret_key
        self._vector_dimension = model_parameters.embedding_dimension
        self._weights_plaintext = self._cc.MakeCKKSPackedPlaintext(
            model_parameters.weights.astype(np.float64).tolist()
        )
        self._intercept_plaintext = self._cc.MakeCKKSPackedPlaintext([model_parameters.intercept])
        self.resolved_parameters = resolved_parameters
        self.bundle_paths = bundle_paths
        self.reused_existing_bundle = reused_existing_bundle

    @classmethod
    def load_or_create(
        cls,
        *,
        settings: OpenFheSettings,
        model_parameters: PlaintextModelParameters,
        bundle_paths: OpenFheBundlePaths,
    ) -> "OpenFheCkksScorer":
        if settings.batch_size < model_parameters.embedding_dimension:
            raise ValueError(
                f"OpenFHE batch_size {settings.batch_size} must be at least the embedding "
                f"dimension {model_parameters.embedding_dimension}."
            )

        openfhe_module = _load_openfhe()
        if bundle_paths.is_complete():
            return cls._load_from_bundle(
                openfhe_module=openfhe_module,
                settings=settings,
                model_parameters=model_parameters,
                bundle_paths=bundle_paths,
            )
        return cls._build_and_persist(
            openfhe_module=openfhe_module,
            settings=settings,
            model_parameters=model_parameters,
            bundle_paths=bundle_paths,
        )

    def score_embedding(self, embedding: np.ndarray) -> tuple[float, LatencyBreakdown]:
        vector = np.asarray(embedding, dtype=np.float64)
        if vector.ndim != 1 or vector.shape[0] != self._vector_dimension:
            raise ValueError(
                f"OpenFHE scorer expected a vector of shape ({self._vector_dimension},), "
                f"found {vector.shape}."
            )

        encryption_start = time.perf_counter()
        plaintext_embedding = self._cc.MakeCKKSPackedPlaintext(vector.tolist())
        encrypted_embedding = self._cc.Encrypt(self._public_key, plaintext_embedding)
        encryption_end = time.perf_counter()

        scoring_start = time.perf_counter()
        encrypted_logit = self._cc.EvalInnerProduct(
            encrypted_embedding,
            self._weights_plaintext,
            self._vector_dimension,
        )
        encrypted_logit = self._cc.EvalAdd(encrypted_logit, self._intercept_plaintext)
        scoring_end = time.perf_counter()

        decryption_start = time.perf_counter()
        decrypted_logit = self._cc.Decrypt(self._secret_key, encrypted_logit)
        decrypted_logit.SetLength(1)
        scalar_logit = float(decrypted_logit.GetRealPackedValue()[0])
        decryption_end = time.perf_counter()

        return scalar_logit, LatencyBreakdown(
            encryption_ms=(encryption_end - encryption_start) * 1000.0,
            scoring_ms=(scoring_end - scoring_start) * 1000.0,
            decryption_ms=(decryption_end - decryption_start) * 1000.0,
        )

    @classmethod
    def _build_and_persist(
        cls,
        *,
        openfhe_module: ModuleType,
        settings: OpenFheSettings,
        model_parameters: PlaintextModelParameters,
        bundle_paths: OpenFheBundlePaths,
    ) -> "OpenFheCkksScorer":
        crypto_context = _build_crypto_context(openfhe_module, settings)
        key_pair = crypto_context.KeyGen()
        crypto_context.EvalMultKeyGen(key_pair.secretKey)
        crypto_context.EvalSumKeyGen(key_pair.secretKey)
        _persist_bundle(
            openfhe_module=openfhe_module,
            crypto_context=crypto_context,
            public_key=key_pair.publicKey,
            secret_key=key_pair.secretKey,
            bundle_paths=bundle_paths,
        )
        return cls(
            openfhe_module=openfhe_module,
            model_parameters=model_parameters,
            crypto_context=crypto_context,
            public_key=key_pair.publicKey,
            secret_key=key_pair.secretKey,
            resolved_parameters=_resolved_parameters_from_context(
                crypto_context=crypto_context,
                settings=settings,
            ),
            bundle_paths=bundle_paths,
            reused_existing_bundle=False,
        )

    @classmethod
    def _load_from_bundle(
        cls,
        *,
        openfhe_module: ModuleType,
        settings: OpenFheSettings,
        model_parameters: PlaintextModelParameters,
        bundle_paths: OpenFheBundlePaths,
    ) -> "OpenFheCkksScorer":
        _clear_openfhe_state(openfhe_module)
        crypto_context, crypto_context_ok = openfhe_module.DeserializeCryptoContext(
            str(bundle_paths.crypto_context_path),
            openfhe_module.BINARY,
        )
        public_key, public_key_ok = openfhe_module.DeserializePublicKey(
            str(bundle_paths.public_key_path),
            openfhe_module.BINARY,
        )
        secret_key, secret_key_ok = openfhe_module.DeserializePrivateKey(
            str(bundle_paths.secret_key_path),
            openfhe_module.BINARY,
        )
        if not (crypto_context_ok and public_key_ok and secret_key_ok):
            raise RuntimeError("OpenFHE bundle deserialization failed for context or key material.")
        openfhe_module.DeserializeEvalMultKeyString(
            bundle_paths.eval_mult_key_path.read_bytes(),
            openfhe_module.BINARY,
        )
        openfhe_module.DeserializeEvalAutomorphismKeyString(
            bundle_paths.eval_automorphism_key_path.read_bytes(),
            openfhe_module.BINARY,
        )
        resolved_parameters = _resolved_parameters_from_context(
            crypto_context=crypto_context,
            settings=settings,
        )
        _validate_resolved_parameters_against_settings(
            resolved_parameters=resolved_parameters,
            settings=settings,
        )
        return cls(
            openfhe_module=openfhe_module,
            model_parameters=model_parameters,
            crypto_context=crypto_context,
            public_key=public_key,
            secret_key=secret_key,
            resolved_parameters=resolved_parameters,
            bundle_paths=bundle_paths,
            reused_existing_bundle=True,
        )


def _build_crypto_context(openfhe_module: ModuleType, settings: OpenFheSettings) -> Any:
    params = openfhe_module.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(settings.multiplicative_depth)
    params.SetScalingModSize(settings.scaling_mod_size)
    params.SetFirstModSize(settings.first_mod_size)
    params.SetBatchSize(settings.batch_size)
    params.SetRingDim(settings.ring_dimension)
    params.SetSecurityLevel(_resolve_security_level(openfhe_module, settings.security_level))
    crypto_context = openfhe_module.GenCryptoContext(params)
    for feature in (
        openfhe_module.PKE,
        openfhe_module.KEYSWITCH,
        openfhe_module.LEVELEDSHE,
        openfhe_module.ADVANCEDSHE,
    ):
        crypto_context.Enable(feature)
    return crypto_context


def _persist_bundle(
    *,
    openfhe_module: ModuleType,
    crypto_context: Any,
    public_key: Any,
    secret_key: Any,
    bundle_paths: OpenFheBundlePaths,
) -> None:
    bundle_paths.root_dir.mkdir(parents=True, exist_ok=True)
    if not openfhe_module.SerializeToFile(
        str(bundle_paths.crypto_context_path),
        crypto_context,
        openfhe_module.BINARY,
    ):
        raise RuntimeError("Failed to serialize OpenFHE crypto context.")
    if not openfhe_module.SerializeToFile(
        str(bundle_paths.public_key_path),
        public_key,
        openfhe_module.BINARY,
    ):
        raise RuntimeError("Failed to serialize OpenFHE public key.")
    if not openfhe_module.SerializeToFile(
        str(bundle_paths.secret_key_path),
        secret_key,
        openfhe_module.BINARY,
    ):
        raise RuntimeError("Failed to serialize OpenFHE secret key.")
    bundle_paths.eval_mult_key_path.write_bytes(
        openfhe_module.SerializeEvalMultKeyString(openfhe_module.BINARY)
    )
    bundle_paths.eval_automorphism_key_path.write_bytes(
        openfhe_module.SerializeEvalAutomorphismKeyString(openfhe_module.BINARY)
    )


def _resolved_parameters_from_context(
    *,
    crypto_context: Any,
    settings: OpenFheSettings,
) -> ResolvedCkksParameters:
    return ResolvedCkksParameters(
        backend=settings.backend,
        scheme=settings.scheme,
        ring_dimension=int(crypto_context.GetRingDimension()),
        batch_size=settings.batch_size,
        multiplicative_depth=settings.multiplicative_depth,
        scaling_mod_size=settings.scaling_mod_size,
        first_mod_size=settings.first_mod_size,
        security_level=settings.security_level,
    )


def _validate_resolved_parameters_against_settings(
    *,
    resolved_parameters: ResolvedCkksParameters,
    settings: OpenFheSettings,
) -> None:
    expected = ResolvedCkksParameters(
        backend=settings.backend,
        scheme=settings.scheme,
        ring_dimension=settings.ring_dimension,
        batch_size=settings.batch_size,
        multiplicative_depth=settings.multiplicative_depth,
        scaling_mod_size=settings.scaling_mod_size,
        first_mod_size=settings.first_mod_size,
        security_level=settings.security_level,
    )
    if resolved_parameters != expected:
        raise ValueError(
            "Reloaded OpenFHE bundle parameters do not match the configured Stage 3 CKKS "
            f"settings. Expected {expected.to_document()}, found {resolved_parameters.to_document()}."
        )


def _clear_openfhe_state(openfhe_module: ModuleType) -> None:
    openfhe_module.ClearEvalMultKeys()
    openfhe_module.ReleaseAllContexts()


def _resolve_security_level(openfhe_module: ModuleType, security_level_name: str) -> Any:
    try:
        return getattr(openfhe_module, security_level_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported OpenFHE security level: {security_level_name}") from exc


def _load_openfhe() -> ModuleType:
    try:
        import openfhe  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The optional OpenFHE dependency is not installed. Run "
            "`uv sync --python 3.12 --extra fhe` before using the Stage 3 FHE evaluator."
        ) from exc
    return openfhe
