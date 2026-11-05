"""Embedding models: turn articles and questions into unit-length vectors (D-035).

Models run locally through sentence-transformers, pinned to a Hugging Face revision.
Article vectors are computed once and stored under cache/vectors/, in a file named after
the model, its revision and a fingerprint of the articles, so a changed model or corpus
never reuses stale vectors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from support_rag import config
from support_rag.records import Article

VECTORS_DIR = config.CACHE_DIR / "vectors"


@dataclass(frozen=True)
class EmbeddingModel:
    name: str  # short name for run records and file names
    repo_id: str  # on Hugging Face
    revision: str  # full commit hash: the weights never change under us
    dimensions: int  # both models are trained to be cut to 1,024 dimensions (Matryoshka)
    trust_remote_code: bool = False  # True runs Python from the model repo: read it first


MODELS = {
    model.name: model
    for model in (
        EmbeddingModel(
            "qwen3-0.6b",
            "Qwen/Qwen3-Embedding-0.6B",
            "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            1024,
        ),
        # Its modeling_qwen3_bidirectional.py was read at this revision: it only defines the
        # model (Qwen3 with bidirectional attention and a projection), no I/O.
        EmbeddingModel(
            "voyage-4-nano",
            "MongoDB/voyage-4-nano",
            "5fb34e870f49f19e79c94d5e118f5ee730133279",
            1024,
            trust_remote_code=True,
        ),
    )
}


class Encoder(Protocol):
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, as float32. Cutting a vector to fewer dimensions
    shortens it, so this runs after the cut.

    Raises ValueError on a row that can't be scaled: an all-zero row, or one with NaN or
    infinity, which is what an fp16 overflow inside a model would produce.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        rows = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(rows).all():
        raise ValueError("some vectors are zero or not finite; check the model's dtype")
    return rows


class SentenceTransformerEncoder:
    """A local model with its authors' own query and document prompts, untuned."""

    def __init__(self, model: EmbeddingModel, *, device: str | None = None) -> None:
        # Imported here so commands that don't embed never load PyTorch.
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 on the GPU: for Qwen, on 200 articles, it matched fp32 to a cosine of 0.9997
        # and ran 2.5 times faster. The RTX 2080 Ti has no bfloat16, the models' native
        # type. unit_rows() refuses vectors an fp16 overflow would break.
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = SentenceTransformer(
            model.repo_id,
            revision=model.revision,
            trust_remote_code=model.trust_remote_code,
            device=self.device,
            model_kwargs={"dtype": dtype},
            truncate_dim=model.dimensions,
        )
        self.params: dict[str, Any] = {
            "model": model.name,
            "repo_id": model.repo_id,
            "revision": model.revision,
            "dimensions": model.dimensions,
            "dtype": str(dtype).removeprefix("torch."),
            "device": self.device,
            "prompts": dict(self._model.prompts),
        }

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(self._model.encode_query, texts)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(self._model.encode_document, texts)

    def _encode(self, encode: Callable[..., Any], texts: Sequence[str]) -> np.ndarray:
        # Small batches: the longest articles run to about 10K tokens.
        vectors = encode(
            list(texts), batch_size=8, convert_to_numpy=True, show_progress_bar=len(texts) > 100
        )
        return unit_rows(vectors)


def corpus_fingerprint(articles: Sequence[Article]) -> str:
    """A hash of every article's ID and text, in order: row i of the vectors is article i."""
    digest = hashlib.sha256()
    for article in articles:
        digest.update(f"{article.id}\0{article.contents}\0".encode())
    return digest.hexdigest()


def vectors_path(model: EmbeddingModel, articles: Sequence[Article], vectors_dir: Path) -> Path:
    return vectors_dir / (
        f"{model.name}_{model.revision[:12]}_{corpus_fingerprint(articles)[:12]}.npy"
    )


def article_vectors(
    articles: Sequence[Article],
    model: EmbeddingModel,
    encoder: Encoder,
    *,
    vectors_dir: Path = VECTORS_DIR,
) -> tuple[np.ndarray, bool]:
    """One unit-length row per article, in the articles' order. Computed on first use and
    stored; later calls load the file. Returns the vectors and whether they were stored."""
    path = vectors_path(model, articles, vectors_dir)
    if path.exists():
        return np.load(path), True
    vectors = encoder.encode_documents([article.contents for article in articles])
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npy")  # never leave a half-written file behind
    np.save(partial, vectors)
    partial.replace(path)
    return vectors, False
