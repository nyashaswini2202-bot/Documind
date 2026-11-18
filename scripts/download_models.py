"""Pre-download the Hugging Face models so the first request is fast (used while building the Docker image)."""
import os

from sentence_transformers import CrossEncoder, SentenceTransformer

emb = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
rer = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

print(f"Downloading embedding model: {emb}")
SentenceTransformer(emb)
if os.getenv("USE_RERANKER", "true").lower() in {"1", "true", "yes", "on"}:
    print(f"Downloading reranker model: {rer}")
    CrossEncoder(rer)
print("Models cached.")
