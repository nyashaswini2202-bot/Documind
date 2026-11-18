# DocuMind architecture (summary)

The complete write-up (models and algorithms, a section for every file, design decisions, interview questions) is in the separate **DocuMind Architecture Document** delivered with this project. This page is the one-screen summary.

```
Browser (static/)                      FastAPI (app/main.py)
  upload ───────────────────────────►  loader ► chunker ► embedder ► KnowledgeBase (FAISS + BM25 + JSON)
  question ─── SSE stream ──────────►  condense ► retriever (dense | bm25 | RRF | cross-encoder)
  ◄── tokens, sources, timings ──────  prompt ► LLM (Groq/Ollama) ► citation check
  evaluate ─── job + polling ───────►  test-set generation ► Hit@k / MRR ► LLM judge
```

| Stage | Algorithm / model |
|---|---|
| Chunking | Recursive character splitter, 800 chars, 150 overlap, never crossing a page |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-d, normalised) |
| Vector search | FAISS `IndexFlatIP` (exact cosine) |
| Keyword search | Okapi BM25 (k1 = 1.5, b = 0.75), written from scratch |
| Fusion | Reciprocal Rank Fusion, k = 60 |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` on top-20 candidates |
| Generation | Llama 3.1 8B via Groq (or any Ollama model), temperature 0.1, grounded prompt |
| Guardrails | Relevance threshold, "not found" refusal, citation validation |
| Evaluation | Synthetic test set, Hit@k, MRR, LLM-judged faithfulness / relevancy / correctness |
