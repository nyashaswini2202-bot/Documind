"""Retrieval: dense, BM25, hybrid (Reciprocal Rank Fusion) and hybrid + cross-encoder rerank."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from .config import settings
from .knowledge_base import KnowledgeBase

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1 / (k + rank(d)).

    It merges ranked lists using only ranks (never raw scores), which is why it can combine
    cosine similarities and BM25 scores - two incomparable scales - without any tuning.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class Hit:
    idx: int
    chunk: dict
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None


@dataclass
class RetrievalResult:
    hits: list[Hit]
    mode_used: str
    timings_ms: dict[str, float]


class Retriever:
    def __init__(self, get_embedder: Callable, get_reranker: Callable):
        self._get_embedder = get_embedder
        self._get_reranker = get_reranker

    def reranker_available(self) -> bool:
        return self._get_reranker() is not None

    def retrieve(
        self,
        kb: KnowledgeBase,
        query: str,
        mode: str = "hybrid_rerank",
        top_k: int = 5,
        candidates: Optional[int] = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        mode = mode if mode in MODES else "hybrid_rerank"
        if kb.is_empty():
            return RetrievalResult([], mode, {"total": 0.0})
        candidates = max(candidates or settings.candidates, top_k)

        reranker = None
        if mode == "hybrid_rerank":
            reranker = self._get_reranker()
            if reranker is None:
                mode = "hybrid"  # graceful fallback when the reranker is disabled/unavailable

        hits: dict[int, Hit] = {}

        def hit(idx: int) -> Hit:
            if idx not in hits:
                hits[idx] = Hit(idx=idx, chunk=kb.chunks[idx])
            return hits[idx]

        dense_rank: list[int] = []
        sparse_rank: list[int] = []

        if mode in ("dense", "hybrid", "hybrid_rerank"):
            t = time.perf_counter()
            qvec = self._get_embedder().encode([query])[0]
            timings["embed"] = (time.perf_counter() - t) * 1000
            for idx, score in kb.dense_search(qvec, candidates):
                hit(idx).dense_score = score
                dense_rank.append(idx)

        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            for idx, score in kb.sparse_search(query, candidates):
                hit(idx).bm25_score = score
                sparse_rank.append(idx)

        if mode == "dense":
            order = dense_rank
        elif mode == "bm25":
            order = sparse_rank
        else:
            fused = rrf([dense_rank, sparse_rank], settings.rrf_k)
            order = sorted(fused, key=fused.__getitem__, reverse=True)
            for idx in order:
                hits[idx].fused_score = fused[idx]

        if reranker is not None and order:
            t = time.perf_counter()
            pool = order[:candidates]
            scores = reranker.score(query, [hits[i].chunk["text"] for i in pool])
            for idx, score in zip(pool, scores):
                hits[idx].rerank_score = score
            order = sorted(pool, key=lambda i: hits[i].rerank_score, reverse=True)
            timings["rerank"] = (time.perf_counter() - t) * 1000

        timings["total"] = (time.perf_counter() - started) * 1000
        return RetrievalResult([hits[i] for i in order[:top_k]], mode, timings)
