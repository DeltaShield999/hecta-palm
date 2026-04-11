from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from .config import EncoderSettings
from .data import Stage3DatasetSplit


EXPECTED_STAGE3_EMBEDDING_DIMENSION = 384


@dataclass(frozen=True, slots=True)
class StoredEmbeddingArtifact:
    split_name: str
    embeddings: np.ndarray
    labels: np.ndarray
    message_ids: tuple[str, ...]
    label_names: tuple[str, ...]
    template_families: tuple[str, ...]
    source_types: tuple[str, ...]
    encoder_model_name: str
    normalize_embeddings: bool


@dataclass(frozen=True, slots=True)
class EmbeddedStage3Split:
    dataset: Stage3DatasetSplit
    embeddings: np.ndarray


def resolve_encoder_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_sentence_encoder(settings: EncoderSettings) -> tuple[SentenceTransformer, str, int]:
    device = resolve_encoder_device(settings.device)
    model = SentenceTransformer(settings.model_name, device=device)
    model.eval()
    if hasattr(model, "get_embedding_dimension"):
        embedding_dimension = int(model.get_embedding_dimension())
    else:
        embedding_dimension = int(model.get_sentence_embedding_dimension())
    if embedding_dimension != EXPECTED_STAGE3_EMBEDDING_DIMENSION:
        raise ValueError(
            "Stage 3 plaintext encoder must produce 384-dimensional embeddings, "
            f"found {embedding_dimension}."
        )
    return model, device, embedding_dimension


def embed_dataset_split(
    model: SentenceTransformer,
    dataset: Stage3DatasetSplit,
    *,
    settings: EncoderSettings,
) -> EmbeddedStage3Split:
    embeddings = np.asarray(
        model.encode(
            list(dataset.texts),
            batch_size=settings.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=settings.normalize_embeddings,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    _validate_embeddings(dataset, embeddings, normalize_embeddings=settings.normalize_embeddings)
    return EmbeddedStage3Split(dataset=dataset, embeddings=embeddings)


def write_embedding_artifact(
    path: Path,
    embedded_split: EmbeddedStage3Split,
    *,
    encoder_model_name: str,
    normalize_embeddings: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        split_name=np.asarray(embedded_split.dataset.split_name),
        embeddings=embedded_split.embeddings,
        labels=embedded_split.dataset.labels.astype(np.int8, copy=False),
        message_ids=np.asarray(embedded_split.dataset.message_ids),
        label_names=np.asarray(embedded_split.dataset.label_names),
        template_families=np.asarray(embedded_split.dataset.template_families),
        source_types=np.asarray(embedded_split.dataset.source_types),
        encoder_model_name=np.asarray(encoder_model_name),
        normalize_embeddings=np.asarray(int(normalize_embeddings), dtype=np.int8),
    )


def load_embedding_artifact(path: Path) -> StoredEmbeddingArtifact:
    with np.load(path, allow_pickle=False) as archive:
        return StoredEmbeddingArtifact(
            split_name=str(archive["split_name"].item()),
            embeddings=np.asarray(archive["embeddings"], dtype=np.float32),
            labels=np.asarray(archive["labels"], dtype=np.int8),
            message_ids=tuple(str(value) for value in archive["message_ids"].tolist()),
            label_names=tuple(str(value) for value in archive["label_names"].tolist()),
            template_families=tuple(str(value) for value in archive["template_families"].tolist()),
            source_types=tuple(str(value) for value in archive["source_types"].tolist()),
            encoder_model_name=str(archive["encoder_model_name"].item()),
            normalize_embeddings=bool(int(archive["normalize_embeddings"].item())),
        )


def _validate_embeddings(
    dataset: Stage3DatasetSplit,
    embeddings: np.ndarray,
    *,
    normalize_embeddings: bool,
) -> None:
    if embeddings.ndim != 2:
        raise ValueError(
            f"Stage 3 embeddings for split {dataset.split_name} must be rank-2, found {embeddings.ndim}."
        )
    if embeddings.shape[0] != len(dataset.rows):
        raise ValueError(
            f"Stage 3 embeddings for split {dataset.split_name} must contain {len(dataset.rows)} rows, "
            f"found {embeddings.shape[0]}."
        )
    if embeddings.shape[1] != EXPECTED_STAGE3_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Stage 3 embeddings for split {dataset.split_name} must have dimension "
            f"{EXPECTED_STAGE3_EMBEDDING_DIMENSION}, found {embeddings.shape[1]}."
        )
    if normalize_embeddings:
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError(
                f"Stage 3 embeddings for split {dataset.split_name} were expected to be normalized."
            )
