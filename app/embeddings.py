"""Dense embeddings with sentence-transformers (all-MiniLM-L6-v2 by default, 384 dimensions)."""
from __future__ import annotations

import threading

import numpy as np

from .config import settings


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        """Return L2-normalised float32 vectors, so inner product == cosine similarity."""
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")


_embedder: Embedder | None = None
_lock = threading.Lock()


def get_embedder() -> Embedder:
    """Lazy singleton - the model is loaded once, on first use."""
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = Embedder(settings.embedding_model)
        return _embedder
