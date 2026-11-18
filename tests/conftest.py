"""Test setup: isolated data dir, no real ML models, no LLM, so tests run in seconds offline."""
import os
import re
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="documind-test-")
os.environ["USE_RERANKER"] = "false"
os.environ["LLM_PROVIDER"] = "none"
os.environ["WARMUP"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (stands in for sentence-transformers)."""

    dim = 256

    def encode(self, texts, batch_size=32):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, zlib.crc32(word.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    import app.main as main

    main.get_embedder = lambda: FakeEmbedder()
    with TestClient(main.app) as c:
        yield c
