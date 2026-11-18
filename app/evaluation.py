"""Built-in RAG evaluation (RAGAS-style metrics, implemented without extra dependencies).

Pipeline
  1. Sample passages from the user's own documents.
  2. An LLM writes a question + reference answer for each passage  ->  a synthetic test set whose
     "gold" evidence chunk is known.
  3. RETRIEVAL ABLATION: run every retrieval mode and measure Hit@k and MRR against the gold chunk.
  4. GENERATION QUALITY: answer each question with the full pipeline, then an LLM judge scores
     faithfulness (claims supported by context), answer relevancy and answer correctness.
"""
from __future__ import annotations

import json
import random
import threading
import time
import uuid
from collections import defaultdict
from typing import Callable, Optional

from . import prompts
from .config import settings
from .knowledge_base import KnowledgeBase
from .llm import LLMClient, LLMError
from .rag_pipeline import RAGPipeline
from .retriever import Retriever

ProgressFn = Callable[[float, str], None]


class EvaluationError(RuntimeError):
    pass


class JobBusy(RuntimeError):
    pass


def _extract_json(text: str) -> Optional[dict]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


def _clamp01(value) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _avg(values: list) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def sample_passages(kb: KnowledgeBase, n: int, seed: int = 7) -> list[dict]:
    """Pick up to n substantial chunks, round-robin across documents for coverage."""
    rng = random.Random(seed)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in kb.chunks:
        if len(chunk["text"]) >= 250:
            by_doc[chunk["doc_id"]].append(chunk)
    for items in by_doc.values():
        rng.shuffle(items)
    picked: list[dict] = []
    while len(picked) < n and any(by_doc.values()):
        for doc_id in list(by_doc):
            if by_doc[doc_id] and len(picked) < n:
                picked.append(by_doc[doc_id].pop())
    return picked


def generate_qa(llm: LLMClient, chunk: dict) -> Optional[dict]:
    try:
        data = _extract_json(llm.chat(prompts.qgen_messages(chunk["text"]), temperature=0.3, max_tokens=250, json_mode=True))
    except LLMError:
        return None
    if not data or not str(data.get("question", "")).strip():
        return None
    return {"question": str(data["question"]).strip(), "reference": str(data.get("answer", "")).strip()}


def judge(llm: LLMClient, question: str, answer: str, contexts: list[str], reference: str) -> dict:
    try:
        data = _extract_json(llm.chat(prompts.judge_messages(question, answer, contexts, reference), temperature=0, max_tokens=200, json_mode=True)) or {}
    except LLMError:
        data = {}
    total = data.get("total_claims")
    supported = data.get("supported_claims")
    faithfulness = None
    try:
        if total and int(total) > 0:
            faithfulness = max(0.0, min(1.0, int(supported) / int(total)))
    except (TypeError, ValueError):
        faithfulness = None
    return {
        "faithfulness": faithfulness,
        "relevancy": _clamp01(data.get("relevancy")),
        "correctness": _clamp01(data.get("correctness")),
    }


def run_evaluation(
    kb: KnowledgeBase,
    pipeline: RAGPipeline,
    retriever: Retriever,
    llm: LLMClient,
    n_questions: int,
    top_k: int,
    progress: ProgressFn,
) -> dict:
    progress(0.02, "Sampling passages from your documents")
    passages = sample_passages(kb, n_questions)
    if not passages:
        raise EvaluationError("The documents are too short to build a test set (need passages of 250+ characters).")

    # ---- 1. synthetic test set ---------------------------------------------------------
    testset: list[dict] = []
    for i, chunk in enumerate(passages, start=1):
        qa = generate_qa(llm, chunk)
        if qa:
            testset.append({**qa, "cid": chunk["cid"], "doc": chunk["doc_name"], "page": chunk["page"]})
        progress(0.05 + 0.25 * i / len(passages), f"Writing test questions ({i}/{len(passages)})")
    if not testset:
        raise EvaluationError("The LLM could not generate test questions. Check your API key / rate limits.")

    # ---- 2. retrieval ablation ---------------------------------------------------------
    modes = ["dense", "bm25", "hybrid"]
    if settings.use_reranker and retriever.reranker_available():
        modes.append("hybrid_rerank")
    stats = {m: {"hits": 0, "rr": 0.0, "latency": []} for m in modes}
    per_question = [{"question": t["question"], "gold": f"{t['doc']} · p.{t['page']}", "ranks": {}} for t in testset]

    for qi, item in enumerate(testset):
        for mode in modes:
            result = retriever.retrieve(kb, item["question"], mode=mode, top_k=top_k)
            cids = [h.chunk["cid"] for h in result.hits]
            rank = cids.index(item["cid"]) + 1 if item["cid"] in cids else None
            if rank:
                stats[mode]["hits"] += 1
                stats[mode]["rr"] += 1.0 / rank
            stats[mode]["latency"].append(result.timings_ms.get("total", 0.0))
            per_question[qi]["ranks"][mode] = rank
        progress(0.30 + 0.20 * (qi + 1) / len(testset), f"Comparing retrieval methods ({qi + 1}/{len(testset)})")

    n = len(testset)
    retrieval = {
        m: {
            "hit_at_k": round(s["hits"] / n, 3),
            "mrr": round(s["rr"] / n, 3),
            "avg_latency_ms": round(sum(s["latency"]) / n, 1),
        }
        for m, s in stats.items()
    }

    # ---- 3. generation quality ---------------------------------------------------------
    best_mode = modes[-1]
    scores = []
    for qi, item in enumerate(testset):
        out = pipeline.answer(kb, item["question"], [], best_mode, top_k)
        row = {"faithfulness": None, "relevancy": None, "correctness": None}
        refused = not out.get("grounded", True)
        if not refused:
            row = judge(llm, item["question"], out["answer"], [s["text"] for s in out["sources"]], item["reference"])
        per_question[qi].update({**row, "refused": refused, "answer": out["answer"]})
        scores.append(row | {"refused": refused})
        progress(0.50 + 0.50 * (qi + 1) / len(testset), f"Scoring answers ({qi + 1}/{len(testset)})")

    generation = {
        "mode": best_mode,
        "faithfulness": _avg([s["faithfulness"] for s in scores]),
        "relevancy": _avg([s["relevancy"] for s in scores]),
        "correctness": _avg([s["correctness"] for s in scores]),
        "refusal_rate": round(sum(1 for s in scores if s["refused"]) / n, 3),
    }
    return {
        "n_questions": n,
        "top_k": top_k,
        "retrieval": retrieval,
        "generation": generation,
        "per_question": per_question,
        "config": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model if "hybrid_rerank" in modes else None,
            "llm": f"{llm.provider}:{llm.model}",
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    }


class JobManager:
    """Runs long evaluations in a background thread; the browser polls for progress."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def start(self, ws_id: str, fn: Callable[[ProgressFn], dict]) -> str:
        with self.lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise JobBusy("Another evaluation is running. Try again in a minute.")
            job_id = uuid.uuid4().hex[:12]
            self.jobs[job_id] = {
                "ws": ws_id, "status": "running", "progress": 0.0, "stage": "Starting",
                "result": None, "error": None, "started": time.time(),
            }
            for old in sorted(self.jobs, key=lambda k: self.jobs[k]["started"])[:-20]:
                self.jobs.pop(old, None)

        def update(progress: float, stage: str) -> None:
            self.jobs[job_id].update(progress=round(progress, 3), stage=stage)

        def runner() -> None:
            try:
                result = fn(update)
                self.jobs[job_id].update(status="done", progress=1.0, stage="Finished", result=result)
            except (EvaluationError, LLMError) as exc:
                self.jobs[job_id].update(status="error", error=str(exc))
            except Exception:  # never leak internals
                self.jobs[job_id].update(status="error", error="Evaluation failed unexpectedly.")

        threading.Thread(target=runner, daemon=True).start()
        return job_id

    def get(self, job_id: str, ws_id: str) -> Optional[dict]:
        job = self.jobs.get(job_id)
        if not job or job["ws"] != ws_id:
            return None
        return {k: v for k, v in job.items() if k not in {"ws", "started"}}
