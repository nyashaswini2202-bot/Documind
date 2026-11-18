"""Cross-encoder reranker (ms-marco-MiniLM-L-6-v2).

A bi-encoder (the embedder) scores query and passage independently, which is fast but coarse.
A cross-encoder reads the (query, passage) pair together, which is slower but far more precise,
so we only run it on the ~20 candidates that survive first-stage retrieval.
"""
from __future__ import annotations

import logging
import math
import threading

from .config import settings

log = logging.getLogger("documind.reranker")


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=512)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return relevance scores in 0..1 (higher is better)."""
        if not passages:
            return []
        raw = self.model.predict(
            [(query, p) for p in passages], batch_size=16, show_progress_bar=False
        )
        scores = [float(s) for s in raw]
        # Depending on the library version the model returns logits or sigmoid probabilities.
        if any(s < 0.0 or s > 1.0 for s in scores):
            scores = [1.0 / (1.0 + math.exp(-max(min(s, 30.0), -30.0))) for s in scores]
        return scores


_reranker: Reranker | None = None
_failed = False
_lock = threading.Lock()


def get_reranker() -> Reranker | None:
    """Lazy singleton. Returns None if disabled or if the model could not be loaded."""
    global _reranker, _failed
    if not settings.use_reranker or _failed:
        return None
    with _lock:
        if _reranker is None and not _failed:
            try:
                _reranker = Reranker(settings.reranker_model)
            except Exception:
                log.exception("Could not load reranker; falling back to hybrid retrieval")
                _failed = True
        return _reranker
