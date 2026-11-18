
"""FastAPI application: REST + Server-Sent-Events API and the static single-page frontend."""

import hashlib
import json
import logging
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import BASE_DIR, settings
from .embeddings import get_embedder
from .evaluation import JobBusy, JobManager, run_evaluation
from .knowledge_base import DuplicateDocument, KnowledgeBase, get_kb
from .llm import LLMClient, LLMError
from .loader import SUPPORTED_EXTENSIONS, extract_pages
from .rag_pipeline import RAGPipeline
from .reranker import get_reranker
from .retriever import MODES, Retriever
from .schemas import ChatRequest, EvalRequest


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

log = logging.getLogger("documind")

llm = LLMClient()
retriever = Retriever(lambda: get_embedder(), lambda: get_reranker())
pipeline = RAGPipeline(llm, retriever)
jobs = JobManager()

WORKSPACE_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _warmup() -> None:
    try:
        get_embedder()

        if settings.use_reranker:
            get_reranker()

        log.info("Models ready")

    except Exception:
        log.exception("Model warm-up failed (they will load on first use)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.warmup:
        threading.Thread(target=_warmup, daemon=True).start()

    yield


app = FastAPI(
    title="DocuMind",
    version=__version__,
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


def workspace(x_workspace_id: str = Header(default="")) -> KnowledgeBase:
    if not WORKSPACE_RE.match(x_workspace_id):
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid X-Workspace-Id header.",
        )

    return get_kb(x_workspace_id)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ----------------------------------------------------------------------
# Pages / Meta
# ----------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
    }


@app.get("/api/config")
def config():
    return {
        "llm": {
            "provider": llm.provider,
            "model": llm.model,
            "available": llm.available,
        },
        "reranker": settings.use_reranker,
        "embedding_model": settings.embedding_model,
        "modes": list(MODES),
        "defaults": {
            "top_k": settings.top_k,
        },
        "limits": {
            "max_upload_mb": settings.max_upload_mb,
            "max_docs": settings.max_docs,
            "extensions": sorted(SUPPORTED_EXTENSIONS),
        },
    }


# ----------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------

@app.get("/api/documents")
def list_documents(kb: KnowledgeBase = Depends(workspace)):
    return {
        "documents": kb.list_documents(),
        "stats": kb.stats(),
    }


@app.post("/api/documents")
def upload_documents(
    files: list[UploadFile] = File(...),
    kb: KnowledgeBase = Depends(workspace),
):
    results = []

    for upload in files:
        name = Path(upload.filename or "file").name

        try:
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    "Unsupported file type. Use PDF, TXT, MD or DOCX."
                )

            data = upload.file.read(settings.max_upload_bytes + 1)

            if len(data) > settings.max_upload_bytes:
                raise ValueError(
                    f"File is larger than {settings.max_upload_mb} MB."
                )

            pages = extract_pages(name, data)

            doc = kb.add_document(
                name,
                hashlib.sha256(data).hexdigest(),
                pages,
                get_embedder(),
            )

            results.append({
                "name": name,
                "status": "indexed",
                "document": doc,
            })

        except DuplicateDocument:
            results.append({
                "name": name,
                "status": "duplicate",
                "message": "This file is already indexed.",
            })

        except ValueError as exc:
            results.append({
                "name": name,
                "status": "error",
                "message": str(exc),
            })

        except Exception:
            log.exception("Failed to index %s", name)

            results.append({
                "name": name,
                "status": "error",
                "message": "Could not process this file.",
            })

    return {
        "results": results,
        "documents": kb.list_documents(),
    }


# ----------------------------------------------------------------------
# Document Summarization
# ----------------------------------------------------------------------

@app.post("/api/documents/{doc_id}/summarize")
def summarize_document(
    doc_id: str,
    kb: KnowledgeBase = Depends(workspace),
):
    """Generate a summary of the selected document."""

    if not llm.available:
        raise HTTPException(
            status_code=503,
            detail="Document summarization requires an available LLM.",
        )

    try:
        summary = pipeline.summarize_document(kb, doc_id)

        return {
            "doc_id": doc_id,
            "summary": summary,
        }

    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Document not found or contains no chunks.",
        )

    except LLMError:
        log.exception("LLM summarization failed for document %s", doc_id)

        raise HTTPException(
            status_code=502,
            detail="Could not generate the document summary.",
        )

    except Exception:
        log.exception("Document summarization failed for %s", doc_id)

        raise HTTPException(
            status_code=500,
            detail="Something went wrong while summarizing the document.",
        )


@app.delete("/api/documents/{doc_id}")
def delete_document(
    doc_id: str,
    kb: KnowledgeBase = Depends(workspace),
):
    if not kb.delete_document(doc_id):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "documents": kb.list_documents(),
    }


@app.delete("/api/documents")
def clear_documents(kb: KnowledgeBase = Depends(workspace)):
    kb.clear()

    return {
        "documents": [],
    }


# ----------------------------------------------------------------------
# Chat (SSE)
# ----------------------------------------------------------------------

@app.post("/api/chat")
def chat(
    req: ChatRequest,
    kb: KnowledgeBase = Depends(workspace),
):
    history = [t.model_dump() for t in req.history]

    def events():
        try:
            for event in pipeline.answer_stream(
                kb,
                req.question,
                history,
                req.mode,
                req.top_k,
            ):
                yield sse(event["type"], event["data"])

        except Exception:
            log.exception("Chat failed")

            yield sse(
                "error",
                {
                    "message": "Something went wrong while answering. Please try again."
                },
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

@app.post("/api/evaluate")
def start_evaluation(
    req: EvalRequest,
    kb: KnowledgeBase = Depends(workspace),
):
    if kb.is_empty():
        raise HTTPException(
            status_code=400,
            detail="Upload at least one document first.",
        )

    if not llm.available:
        raise HTTPException(
            status_code=400,
            detail="Evaluation needs an LLM. Set GROQ_API_KEY in your environment.",
        )

    try:
        job_id = jobs.start(
            kb.ws_id,
            lambda progress: run_evaluation(
                kb,
                pipeline,
                retriever,
                llm,
                req.num_questions,
                req.top_k,
                progress,
            ),
        )

    except JobBusy as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
        )

    return {
        "job_id": job_id,
    }


@app.get("/api/evaluate/{job_id}")
def evaluation_status(
    job_id: str,
    kb: KnowledgeBase = Depends(workspace),
):
    job = jobs.get(job_id, kb.ws_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation job not found.",
        )

    return job
