"""Per-workspace knowledge base: documents, chunks, embeddings, FAISS index and BM25 index.

Each browser gets a private workspace id (sent as the X-Workspace-Id header), so visitors to a
public demo never see each other's documents. State is persisted to disk:

    data/workspaces/<id>/kb.json          documents + chunk text/metadata
    data/workspaces/<id>/embeddings.npy   float32 matrix, one row per chunk

FAISS and BM25 indexes are cheap to build, so they are rebuilt in memory on load/change.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .bm25_index import BM25Index
from .chunker import chunk_pages
from .config import settings
from .vector_store import FaissStore


class DuplicateDocument(Exception):
    """The same file (by SHA-256) is already indexed in this workspace."""


class KnowledgeBase:
    def __init__(self, ws_id: str, root: Path):
        self.ws_id = ws_id
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.docs: dict[str, dict] = {}
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.store: FaissStore | None = None
        self.bm25: BM25Index | None = None
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        meta_path, emb_path = self.root / "kb.json", self.root / "embeddings.npy"
        if not (meta_path.exists() and emb_path.exists()):
            return
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            embeddings = np.load(emb_path)
            if len(data["chunks"]) != len(embeddings):
                raise ValueError("chunk / embedding count mismatch")
            self.docs, self.chunks, self.embeddings = data["docs"], data["chunks"], embeddings
            self._rebuild()
        except Exception:
            self._reset_memory()

    def _save(self) -> None:
        tmp = self.root / "kb.json.tmp"
        tmp.write_text(json.dumps({"docs": self.docs, "chunks": self.chunks}), encoding="utf-8")
        os.replace(tmp, self.root / "kb.json")
        emb_path = self.root / "embeddings.npy"
        if self.embeddings is None:
            emb_path.unlink(missing_ok=True)
        else:
            tmp_emb = self.root / "embeddings.tmp.npy"
            np.save(tmp_emb, self.embeddings)
            os.replace(tmp_emb, emb_path)

    def _reset_memory(self) -> None:
        self.docs, self.chunks = {}, []
        self.embeddings, self.store, self.bm25 = None, None, None

    def _rebuild(self) -> None:
        if self.chunks and self.embeddings is not None:
            self.store = FaissStore.from_matrix(self.embeddings)
            self.bm25 = BM25Index([c["text"] for c in self.chunks])
        else:
            self.store, self.bm25 = None, None

    # ------------------------------------------------------------------ mutations
    def _check_can_add(self, sha: str, extra_chunks: int = 0) -> None:
        if any(d["sha256"] == sha for d in self.docs.values()):
            raise DuplicateDocument()
        if len(self.docs) >= settings.max_docs:
            raise ValueError(f"Document limit reached ({settings.max_docs}). Delete one first.")
        if len(self.chunks) + extra_chunks > settings.max_chunks:
            raise ValueError("This document would exceed the workspace size limit.")

    def add_document(self, name: str, sha: str, pages: list[tuple[int, str]], embedder) -> dict:
        with self.lock:
            self._check_can_add(sha)
        pieces = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
        if not pieces:
            raise ValueError("The document has no usable text.")
        vectors = embedder.encode([p["text"] for p in pieces])  # slow part: outside the lock

        with self.lock:
            self._check_can_add(sha, extra_chunks=len(pieces))
            doc_id = uuid.uuid4().hex[:8]
            for i, piece in enumerate(pieces):
                self.chunks.append(
                    {
                        "cid": f"{doc_id}-{i}",
                        "doc_id": doc_id,
                        "doc_name": name,
                        "page": piece["page"],
                        "text": piece["text"],
                    }
                )
            self.embeddings = vectors if self.embeddings is None else np.vstack([self.embeddings, vectors])
            doc = {
                "id": doc_id,
                "name": name,
                "sha256": sha,
                "pages": len({p["page"] for p in pieces}),
                "chunks": len(pieces),
                "chars": sum(len(p["text"]) for p in pieces),
                "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self.docs[doc_id] = doc
            self._rebuild()
            self._save()
            return self._public(doc)

    def delete_document(self, doc_id: str) -> bool:
        with self.lock:
            if doc_id not in self.docs:
                return False
            keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] != doc_id]
            self.chunks = [self.chunks[i] for i in keep]
            self.embeddings = self.embeddings[keep] if keep and self.embeddings is not None else None
            del self.docs[doc_id]
            self._rebuild()
            self._save()
            return True

    def clear(self) -> None:
        with self.lock:
            self._reset_memory()
            shutil.rmtree(self.root, ignore_errors=True)
            self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ queries

    def get_document_chunks(self, doc_id: str) -> list[dict]:
        """Return all chunks belonging to a specific document."""
        with self.lock:
            if doc_id not in self.docs:
                return []

            return [
                chunk.copy()
                for chunk in self.chunks
                if chunk["doc_id"] == doc_id
            ]

    @staticmethod
    def _public(doc: dict) -> dict:
        return {k: v for k, v in doc.items() if k != "sha256"}

    def list_documents(self) -> list[dict]:
        with self.lock:
            return [self._public(d) for d in self.docs.values()]

    def is_empty(self) -> bool:
        return not self.chunks

    def stats(self) -> dict:
        with self.lock:
            return {"documents": len(self.docs), "chunks": len(self.chunks)}

    def dense_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        with self.lock:
            return self.store.search(query_vec, k) if self.store else []

    def sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        with self.lock:
            return self.bm25.search(query, k) if self.bm25 else []


# ---------------------------------------------------------------------- registry
_registry: dict[str, KnowledgeBase] = {}
_registry_lock = threading.Lock()


def get_kb(ws_id: str) -> KnowledgeBase:
    with _registry_lock:
        kb = _registry.get(ws_id)
        if kb is None:
            kb = KnowledgeBase(ws_id, settings.data_dir / "workspaces" / ws_id)
            _registry[ws_id] = kb
        return kb
