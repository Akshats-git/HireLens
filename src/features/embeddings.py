"""
Embedding module for HireLens.

Provides batch embedding generation using all-MiniLM-L6-v2 with:
  - GPU support (auto-detects CUDA)
  - Disk cache keyed by SHA-256 of text content
  - Cosine similarity helpers
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml
from loguru import logger
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
_CACHE_DIR = PROJECT_ROOT / "models" / "cache" / "embeddings"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resolve_device(config_device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu' based on availability."""
    if config_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config_device


class EmbeddingModel:
    """
    Sentence-transformer wrapper with batched GPU inference and disk caching.

    Attributes:
        model_name: HuggingFace model identifier.
        device: 'cuda' or 'cpu'.
        use_cache: Whether to cache embeddings to disk.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        use_cache: bool = True,
    ) -> None:
        cfg = _load_config()
        self.model_name = model_name or cfg["model"]["base"]["name"]
        self.device = device or _resolve_device(cfg["model"]["base"]["device"])
        self.use_cache = use_cache
        self._max_seq_len: int = 256  # all-MiniLM-L6-v2 supports max 256 tokens

        logger.info(f"Loading {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.max_seq_length = self._max_seq_len

        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                        f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Cache helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(texts: list[str]) -> str:
        """SHA-256 of the sorted, joined text list."""
        content = "||".join(sorted(texts))
        return hashlib.sha256(content.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return _CACHE_DIR / f"{key}.npy"

    def _load_cache(self, key: str) -> np.ndarray | None:
        path = self._cache_path(key)
        if path.exists():
            return np.load(path)
        return None

    def _save_cache(self, key: str, embeddings: np.ndarray) -> None:
        np.save(self._cache_path(key), embeddings)

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text(s) into L2-normalised embeddings.

        Args:
            texts: Single string or list of strings.
            batch_size: Number of texts per forward pass. Keep ≤64 on GTX 1650.
            show_progress: Show tqdm progress bar for large batches.
            normalize: L2-normalise embeddings (required for cosine similarity).

        Returns:
            ndarray of shape (N, 384) or (384,) for single input.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        if self.use_cache:
            key = self._cache_key(texts)
            cached = self._load_cache(key)
            if cached is not None:
                logger.debug(f"Cache hit for {len(texts)} texts.")
                return cached[0] if single else cached

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            device=self.device,
            convert_to_numpy=True,
        )

        if self.use_cache:
            self._save_cache(key, embeddings)

        return embeddings[0] if single else embeddings

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a large list in batches, returning all embeddings stacked."""
        return self.encode(texts, batch_size=batch_size, show_progress=show_progress)

    # ── Similarity ────────────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | np.ndarray:
        """
        Cosine similarity between two (batches of) L2-normalised embeddings.

        For L2-normalised vectors: cosine_sim = dot product.

        Args:
            a: Array of shape (D,) or (N, D).
            b: Array of shape (D,) or (N, D) or (M, D) for batch query.

        Returns:
            Scalar for 1-D inputs, ndarray for batched inputs.
        """
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        sims = (a @ b.T)
        return float(sims[0, 0]) if sims.shape == (1, 1) else sims.squeeze()

    @staticmethod
    def top_k_similar(
        query: np.ndarray,
        corpus: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the indices and scores of the top-K most similar corpus entries.

        Args:
            query: Shape (D,) — single query embedding.
            corpus: Shape (N, D) — corpus embeddings.
            k: Number of results.

        Returns:
            (indices, scores) arrays of length k, sorted descending.
        """
        scores = corpus @ query
        k = min(k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return top_idx, scores[top_idx]


# ── Module-level singleton ────────────────────────────────────────────────────
# Lazily initialised so import doesn't trigger model load.
_default_model: EmbeddingModel | None = None


def get_model() -> EmbeddingModel:
    """Return the shared EmbeddingModel instance (loads on first call)."""
    global _default_model
    if _default_model is None:
        _default_model = EmbeddingModel()
    return _default_model
