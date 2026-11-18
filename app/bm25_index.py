"""Okapi BM25 keyword search implemented from scratch (no external library).

score(d, q) = sum over query terms t of  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
idf(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))     # Lucene variant: always positive
"""
from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how i if in into is it its of on or such that the "
    "their then there these they this to was we were what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        tokenized = [tokenize(t) for t in texts]
        self.doc_len = np.array([len(t) for t in tokenized], dtype="float32")
        self.avgdl = float(self.doc_len.mean()) if len(tokenized) and self.doc_len.sum() else 1.0

        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, tokens in enumerate(tokenized):
            for term, tf in Counter(tokens).items():
                self.postings[term].append((doc_id, tf))

        n_docs = len(texts)
        self.idf = {
            term: math.log(1.0 + (n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings[term]:
                norm = self.k1 * (1.0 - self.b + self.b * float(self.doc_len[doc_id]) / self.avgdl)
                scores[doc_id] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return heapq.nlargest(k, scores.items(), key=lambda item: item[1])
