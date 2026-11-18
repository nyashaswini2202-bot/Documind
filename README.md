# DocuMind – Chat with your documents (RAG with evaluation)

Upload PDFs, Word files, Markdown or text files, then ask questions in plain language. Answers are generated **only from your documents**, stream in live, and every claim carries a citation that opens the exact passage (file + page) it came from. A built-in **evaluation tab** measures how good retrieval and answers are, so you can show numbers, not just a demo.

## Features

| Area | What it does |
|---|---|
| Ingestion | PDF / DOCX / TXT / MD, page-aware cleaning, recursive chunking with overlap |
| Retrieval | Dense vectors (MiniLM + FAISS), BM25 keywords, **hybrid fusion (RRF)**, **cross-encoder reranking** |
| Generation | Grounded prompt, streamed answers, `[n]` citations, "not found" refusal, follow-up question rewriting |
| Evaluation | Auto-generated test set, Hit@k / MRR ablation across 4 retrieval modes, LLM-judged faithfulness / relevancy / correctness |
| Frontend | Single-page UI (no build step): drag-and-drop upload, live streaming, clickable citations, dark mode, mobile layout |
| Deployment | Dockerfile (CPU PyTorch, models baked in), `render.yaml`, Hugging Face Spaces ready, private per-browser workspaces |

## Project layout

```
documind/
├── app/                      Backend (FastAPI)
│   ├── main.py               HTTP API + static site
│   ├── config.py             All settings (env-driven)
│   ├── schemas.py            Request validation
│   ├── loader.py             PDF/DOCX/TXT/MD -> page-numbered text
│   ├── chunker.py            Recursive splitter with overlap
│   ├── embeddings.py         Sentence-transformers embedder
│   ├── vector_store.py       FAISS index wrapper
│   ├── bm25_index.py         BM25 from scratch
│   ├── reranker.py           Cross-encoder reranker
│   ├── retriever.py          dense / bm25 / hybrid (RRF) / hybrid+rerank
│   ├── knowledge_base.py     Per-workspace storage + persistence
│   ├── prompts.py            Every LLM prompt
│   ├── llm.py                Groq / Ollama client with streaming
│   ├── rag_pipeline.py       The end-to-end RAG flow
│   └── evaluation.py         Test-set generation, metrics, background jobs
├── static/                   Frontend (index.html, css/style.css, js/app.js)
├── tests/                    pytest suite (runs offline in ~1 s)
├── sample_docs/              Two documents to try immediately
├── scripts/download_models.py
├── docs/                     ARCHITECTURE.md, DEPLOYMENT.md
├── Dockerfile · docker-compose.yml · render.yaml
├── requirements.txt · requirements-dev.txt · .env.example
```

## Run it locally (step by step)

**You need:** Python 3.10, 3.11 or 3.12, about 2 GB of free disk space, and an internet connection for the first run (it downloads two small models, ~130 MB).

**1. Unzip and open a terminal in the project folder**
```bash
cd documind
```

**2. Create a virtual environment**
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**3. Install dependencies** (takes a few minutes: PyTorch is large)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Optional, to save disk space on a machine without a GPU, install CPU-only PyTorch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

**4. Get a free LLM key** (about 1 minute)
Create an account at <https://console.groq.com/keys>, click *Create API key*, and copy it.

**5. Configure**
```bash
cp .env.example .env          # Windows: copy .env.example .env
```
Open `.env` and paste your key: `GROQ_API_KEY=gsk_...`

> No key? The app still runs in *extractive mode* and shows the best-matching passages, but you need a key for written answers and for the Evaluate tab. Prefer a local model? Install [Ollama](https://ollama.com), run `ollama pull llama3.2`, and set `LLM_PROVIDER=ollama` in `.env`.

**6. Start the server**
```bash
uvicorn app.main:app --reload --port 8000
```
Wait for the log line `Models ready` (first run downloads the models).

**7. Open** <http://localhost:8000>

## Using the app

1. **Add documents.** Drag `sample_docs/remote_work_policy.txt` and `sample_docs/rag_primer.md` onto the upload box (or use your own PDFs).
2. **Ask questions**, for example:
   - *How many days can employees work remotely?*
   - *What happens to unused leave days?*
   - *How is hybrid search different from vector search?*
   - *What is the capital of France?* (should say it is not in your documents)
3. **Click a yellow citation number** under an answer to jump to the source passage and see its relevance score.
4. **Try the search methods** in the sidebar (vectors only, keywords only, hybrid, hybrid + rerank) and compare answers.
5. Open the **Evaluate quality** tab, choose 8 questions, and click *Run evaluation*. After 1–2 minutes you get a comparison table (Hit rate and MRR per retrieval method) and answer-quality scores. Screenshot this for your portfolio.

## Run the tests
```bash
pip install -r requirements-dev.txt
pytest -q
```
The tests use a fake embedder and a scripted fake LLM, so they need no internet or API key.

## Run with Docker
```bash
docker build -t documind .
docker run --rm -p 7860:7860 --env-file .env documind
# open http://localhost:7860
```
(or `docker compose up --build`). See `docs/DEPLOYMENT.md` to publish it online.

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | – | Enables answers + evaluation |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Any chat model available on your Groq account |
| `LLM_PROVIDER` | `auto` | `auto`, `groq`, `ollama`, `none` |
| `USE_RERANKER` | `true` | Set `false` on low-memory hosts (< 1 GB RAM) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Characters per chunk / shared between neighbours |
| `TOP_K` | `5` | Passages given to the LLM |
| `RETRIEVAL_CANDIDATES` | `20` | Candidates fetched before reranking |
| `MIN_RERANK_SCORE` | `0.01` | Below this relevance the system answers "not found" |
| `MAX_UPLOAD_MB` / `MAX_DOCS_PER_WORKSPACE` | `15` / `10` | Demo protection limits |
| `DATA_DIR` | `./data` | Where indexes are stored |

## Troubleshooting

| Symptom | Fix |
|---|---|
| First upload is slow | Models are loading/downloading once; later uploads are fast |
| `LLM: not set (extractive mode)` badge | `GROQ_API_KEY` missing in `.env`; restart the server after editing |
| "rate limit" message | Groq free tier limit; wait ~30 s, or lower the number of evaluation questions |
| "No extractable text found" | The PDF is a scan (images). Use a text-based PDF or run OCR first |
| `faiss` install fails | Use Python 3.10–3.12 (not 3.13+), and upgrade pip |
| Port 8000 busy | Use `--port 8001` |
| Everything is answered "not found" | Lower `MIN_RERANK_SCORE` (e.g. `0.002`) or switch Search method to *Hybrid* |

## Limits and ideas for extension
- Scanned PDFs need OCR (add `pytesseract`).
- One shared embedding model for all workspaces; storage is local disk (swap in Qdrant/pgvector for multi-instance hosting).
- No login; workspaces are identified by a random id in the browser.
- Ideas: conversation memory in the index, table-aware PDF parsing, metadata filters, query routing, feedback buttons that feed the evaluation set.

## Resume bullets (edit the numbers to match your own evaluation results)
- Built and deployed **DocuMind**, a retrieval-augmented document Q&A system (FastAPI, FAISS, sentence-transformers, Llama 3 via Groq) with cited, streamed answers and a "not found" guardrail.
- Implemented **hybrid retrieval** (BM25 written from scratch + dense vectors fused with Reciprocal Rank Fusion) and a **cross-encoder reranker**, improving Hit@5 from X% to Y% over vector-only search in my own ablation.
- Built an **automated evaluation harness** (synthetic test-set generation, Hit@k/MRR, LLM-judged faithfulness and relevancy) and reached Z% faithfulness.
- Containerised with Docker (CPU PyTorch, models baked in) and deployed on Hugging Face Spaces; per-user isolated workspaces and upload limits.
