from __future__ import annotations

from dataclasses import dataclass
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


class OpenFheCkksScorer:
    def __init__(
        self,
        *,
        settings: OpenFheSettings,
        model_parameters: PlaintextModelParameters,
    ) -> None:
        if settings.batch_size < model_parameters.embedding_dimension:
            raise ValueError(
                f"OpenFHE batch_size {settings.batch_size} must be at least the embedding "
                f"dimension {model_parameters.embedding_dimension}."
            )
        self._openfhe = _load_openfhe()
        self._cc = _build_crypto_context(self._openfhe, settings)
        self._key_pair = self._cc.KeyGen()
        self._cc.EvalMultKeyGen(self._key_pair.secretKey)
        self._cc.EvalSumKeyGen(self._key_pair.secretKey)
        self._vector_dimension = model_parameters.embedding_dimension
        self._weights_plaintext = self._cc.MakeCKKSPackedPlaintext(
            model_parameters.weights.astype(np.float64).tolist()
        )
        self._intercept_plaintext = self._cc.MakeCKKSPackedPlaintext([model_parameters.intercept])
        self.resolved_parameters = ResolvedCkksParameters(
            backend=settings.backend,
            scheme=settings.scheme,
            ring_dimension=int(self._cc.GetRingDimension()),
            batch_size=settings.batch_size,
            multiplicative_depth=settings.multiplicative_depth,
            scaling_mod_size=settings.scaling_mod_size,
            first_mod_size=settings.first_mod_size,
            security_level=settings.security_level,
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
        encrypted_embedding = self._cc.Encrypt(self._key_pair.publicKey, plaintext_embedding)
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
        decrypted_logit = self._cc.Decrypt(self._key_pair.secretKey, encrypted_logit)
        decrypted_logit.SetLength(1)
        scalar_logit = float(decrypted_logit.GetRealPackedValue()[0])
        decryption_end = time.perf_counter()

        return scalar_logit, LatencyBreakdown(
            encryption_ms=(encryption_end - encryption_start) * 1000.0,
            scoring_ms=(scoring_end - scoring_start) * 1000.0,
            decryption_ms=(decryption_end - decryption_start) * 1000.0,
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
