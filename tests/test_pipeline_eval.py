"""Exercises streaming answers, citations, query condensing and the evaluation job with a scripted fake LLM."""
import json
import time
from pathlib import Path

from tests.conftest import FakeEmbedder  # noqa: E402

from app import prompts
from app.evaluation import JobManager, run_evaluation
from app.knowledge_base import KnowledgeBase
from app.rag_pipeline import RAGPipeline
from app.retriever import Retriever

POLICY = (Path(__file__).resolve().parent.parent / "sample_docs" / "remote_work_policy.txt").read_text()


class FakeLLM:
    provider, model, available = "fake", "fake-llm", True

    def chat(self, messages, **kw):
        user = messages[-1]["content"]
        if "Return JSON: {\"question\"" in user:
            return json.dumps({"question": "How many remote days per week are allowed?", "answer": "Three days per week."})
        if "total_claims" in user:
            return json.dumps({"total_claims": 2, "supported_claims": 2, "relevancy": 0.9, "correctness": 0.8})
        return "how many remote days are allowed"

    def stream(self, messages, **kw):
        if "Context passages" in messages[-1]["content"]:
            yield "Employees may work remotely three days a week [1]."


def make_pipeline(tmp_path):
    from app.config import settings
    kb = KnowledgeBase("pipelinetest01", tmp_path)
    emb = FakeEmbedder()
    from app.loader import extract_pages
    kb.add_document("policy.txt", "abc123", extract_pages("policy.txt", POLICY.encode()), emb)
    retriever = Retriever(lambda: emb, lambda: None)
    return kb, retriever, RAGPipeline(FakeLLM(), retriever)


def test_streaming_answer_has_citations(tmp_path):
    kb, _, pipeline = make_pipeline(tmp_path)
    out = pipeline.answer(kb, "How many remote days?", [], "hybrid", 3)
    assert "[1]" in out["answer"] and out["cited"] == [1] and out["grounded"] is True


def test_followup_is_condensed(tmp_path):
    kb, _, pipeline = make_pipeline(tmp_path)
    prep = pipeline.prepare(kb, "and for probation?", [{"role": "user", "content": "remote days?"}], "hybrid", 3)
    assert prep["standalone"] == "how many remote days are allowed"


def test_unanswerable_query_is_refused_without_llm_call(tmp_path):
    kb, _, pipeline = make_pipeline(tmp_path)
    out = pipeline.answer(kb, "qqqzzz", [], "bm25", 3)  # no keyword overlap -> no sources
    assert out["grounded"] is False and out["answer"] == prompts.NOT_FOUND


def test_evaluation_job_runs_to_completion(tmp_path):
    kb, retriever, pipeline = make_pipeline(tmp_path)
    jobs = JobManager()
    job_id = jobs.start("ws", lambda progress: run_evaluation(kb, pipeline, retriever, FakeLLM(), 3, 3, progress))
    for _ in range(100):
        job = jobs.get(job_id, "ws")
        if job["status"] != "running":
            break
        time.sleep(0.1)
    assert job["status"] == "done", job
    result = job["result"]
    assert set(result["retrieval"]) == {"dense", "bm25", "hybrid"}
    assert result["retrieval"]["hybrid"]["hit_at_k"] > 0
    assert result["generation"]["faithfulness"] == 1.0
    assert jobs.get(job_id, "other-workspace") is None
