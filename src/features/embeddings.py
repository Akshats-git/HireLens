"""
Sentence-transformer embedding generation.

Wraps all-MiniLM-L6-v2 with automatic CUDA detection, an optional disk cache for
repeated offline batch work, and cosine-similarity helpers.

The disk cache is off by default. It writes one .npy per distinct input and never
evicts, which suits pipeline runs over a fixed corpus but would grow without
bound behind the API — request-level caching belongs to CacheService instead.
"""

import hashlib
from pathlib import Path
from typing import Union

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import PROJECT_ROOT, get_section

CACHE_DIR = PROJECT_ROOT / "models" / "cache" / "embeddings"

DEFAULT_BATCH_SIZE = 64


def resolve_device(configured: str) -> str:
    """Resolve the configured device, mapping 'auto' to CUDA when available."""
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured


class EmbeddingModel:
    """
    Sentence-transformer wrapper with batched inference and an optional disk cache.

    Attributes:
        model_name: HuggingFace model identifier.
        device: 'cuda' or 'cpu'.
        use_cache: Whether encoded results are persisted under models/cache.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        use_cache: bool = False,
    ) -> None:
        base_config = get_section("model")["base"]
        self.model_name = model_name or base_config["name"]
        self.device = device or resolve_device(base_config["device"])
        self.use_cache = use_cache
        self.max_seq_length = base_config["max_seq_length"]

        if self.use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.max_seq_length = self.max_seq_length

        if self.device == "cuda":
            properties = torch.cuda.get_device_properties(0)
            logger.info(
                f"GPU: {torch.cuda.get_device_name(0)} | "
                f"VRAM: {properties.total_memory / 1e9:.1f} GB"
            )

    # ── Disk cache ────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(texts: list[str]) -> str:
        """
        Hash a text list into a cache key.

        Order matters: the returned embedding matrix is positional, so sorting
        the texts first would let ['a', 'b'] and ['b', 'a'] share an entry and
        hand the second caller its rows transposed.
        """
        return hashlib.sha256("\x00".join(texts).encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.npy"

    def _load_cached(self, key: str) -> np.ndarray | None:
        path = self._cache_path(key)
        return np.load(path) if path.exists() else None

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(
        self,
        texts: Union[str, list[str]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: A single string or a list of strings.
            batch_size: Texts per forward pass. Keep at or below 64 on 4 GB VRAM.
            show_progress: Display a progress bar for long batches.
            normalize: L2-normalise, so cosine similarity reduces to a dot product.

        Returns:
            Array of shape (384,) for a single string, otherwise (N, 384).
        """
        is_single = isinstance(texts, str)
        batch = [texts] if is_single else list(texts)

        cache_key = self._cache_key(batch) if self.use_cache else None
        if cache_key is not None:
            cached = self._load_cached(cache_key)
            if cached is not None:
                logger.debug(f"Embedding cache hit for {len(batch)} text(s).")
                return cached[0] if is_single else cached

        embeddings = self.model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            device=self.device,
            convert_to_numpy=True,
        )

        if cache_key is not None:
            np.save(self._cache_path(cache_key), embeddings)

        return embeddings[0] if is_single else embeddings

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a large list, showing progress by default."""
        return self.encode(texts, batch_size=batch_size, show_progress=show_progress)

    # ── Similarity ────────────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | np.ndarray:
        """
        Cosine similarity between two L2-normalised embeddings or batches.

        Args:
            a: Shape (D,) or (N, D).
            b: Shape (D,) or (M, D).

        Returns:
            A float for two single vectors, otherwise a similarity matrix.
        """
        similarities = np.atleast_2d(a) @ np.atleast_2d(b).T
        if similarities.shape == (1, 1):
            return float(similarities[0, 0])
        return similarities.squeeze()

    @staticmethod
    def top_k_similar(
        query: np.ndarray, corpus: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the top-K most similar corpus entries, ranked descending.

        Args:
            query: Shape (D,) — a single normalised query embedding.
            corpus: Shape (N, D) — normalised corpus embeddings.
            k: Number of results; clamped to the corpus size.

        Returns:
            (indices, scores), both of length min(k, N).
        """
        scores = corpus @ query
        k = min(k, len(scores))
        if k == 0:
            return np.array([], dtype=int), np.array([])

        candidates = np.argpartition(scores, -k)[-k:]
        ranked = candidates[np.argsort(scores[candidates])[::-1]]
        return ranked, scores[ranked]


# ── Module-level singleton ────────────────────────────────────────────────────

_default_model: EmbeddingModel | None = None


def get_model() -> EmbeddingModel:
    """Return the shared EmbeddingModel, loading weights on first call."""
    global _default_model
    if _default_model is None:
        _default_model = EmbeddingModel()
    return _default_model
