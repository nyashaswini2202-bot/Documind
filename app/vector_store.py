"""Thin wrapper around a FAISS flat inner-product index (exact nearest-neighbour search)."""
from __future__ import annotations

import faiss
import numpy as np


class FaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "FaissStore":
        store = cls(matrix.shape[1])
        store.index.add(np.ascontiguousarray(matrix, dtype="float32"))
        return store

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype="float32")
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def __len__(self) -> int:
        return int(self.index.ntotal)
