
"""The RAG orchestration: condense -> retrieve -> filter -> prompt -> stream answer -> validate citations."""

from __future__ import annotations

import re
import time
from typing import Iterator

from . import prompts
from .config import settings
from .knowledge_base import KnowledgeBase
from .llm import LLMClient, LLMError
from .retriever import Hit, RetrievalResult, Retriever


def _source(n: int, hit: Hit) -> dict:
    def r(v, d=4):
        return None if v is None else round(float(v), d)

    return {
        "n": n,
        "cid": hit.chunk["cid"],
        "doc_id": hit.chunk["doc_id"],
        "doc_name": hit.chunk["doc_name"],
        "page": hit.chunk["page"],
        "text": hit.chunk["text"],
        "scores": {
            "dense": r(hit.dense_score),
            "bm25": r(hit.bm25_score, 3),
            "fused": r(hit.fused_score, 5),
            "rerank": r(hit.rerank_score),
        },
    }


class RAGPipeline:

    def __init__(self, llm: LLMClient, retriever: Retriever):
        self.llm = llm
        self.retriever = retriever

    # --------------------------------------------------------------
    # Document summarization
    # --------------------------------------------------------------

    def summarize_document(self, kb: KnowledgeBase, doc_id: str) -> str:
        """Summarize all chunks belonging to one document."""

        if not self.llm.available:
            raise LLMError("No LLM is configured for document summarization.")

        chunks = kb.get_document_chunks(doc_id)

        if not chunks:
            raise ValueError("Document not found or contains no chunks.")

        # Maximum characters sent in one summarization request.
        max_batch_chars = 10000

        def make_batches(parts: list[str]) -> list[str]:
            batches = []
            current = ""

            for part in parts:
                # Split exceptionally large chunks into smaller pieces.
                segments = [
                    part[i:i + max_batch_chars]
                    for i in range(0, len(part), max_batch_chars)
                ]

                for segment in segments:
                    if current and len(current) + len(segment) > max_batch_chars:
                        batches.append(current)
                        current = ""

                    current += segment + "\n\n"

            if current:
                batches.append(current)

            return batches

        # Include every chunk in the document, preserving document order.
        parts = [
            f"Page {chunk['page']}:\n{chunk['text']}"
            for chunk in chunks
        ]

        while True:
            batches = make_batches(parts)

            summaries = []

            for batch in batches:
                summary = self.llm.chat(
                    prompts.summarize_messages(batch),
                    temperature=0.1,
                    max_tokens=settings.llm_max_tokens,
                ).strip()

                if not summary:
                    raise LLMError("The LLM returned an empty document summary.")

                summaries.append(summary)

            # One batch means the final summary is ready.
            if len(summaries) == 1:
                return summaries[0]

            # Ensure every iteration makes progress.
            if len(summaries) >= len(parts):
                raise LLMError(
                    "Unable to reduce the document summaries. "
                    "Try increasing the batching threshold."
                )

            # Summarize the partial summaries in another round.
            parts = summaries

    # ------------------------------------------------------------------
    # RAG pipeline steps
    # ------------------------------------------------------------------

    def _condense(self, question: str, history: list[dict]) -> str:
        """Turn a follow-up into a standalone search query."""

        if not history or not self.llm.available:
            return question

        try:
            out = self.llm.chat(
                prompts.condense_messages(history, question),
                temperature=0,
                max_tokens=120,
            )
        except LLMError:
            return question

        out = out.strip().strip('"').strip()

        return out if 3 <= len(out) <= 400 else question

    @staticmethod
    def _filter(result: RetrievalResult) -> list[Hit]:
        """Drop weak matches so the LLM is never forced to answer from irrelevant text."""

        if result.mode_used == "hybrid_rerank":
            return [
                h for h in result.hits
                if (h.rerank_score or 0.0) >= settings.min_rerank_score
            ]

        if result.mode_used == "dense":
            return [
                h for h in result.hits
                if (h.dense_score or 0.0) >= settings.min_dense_score
            ]

        return result.hits

    def prepare(
        self,
        kb: KnowledgeBase,
        question: str,
        history: list[dict],
        mode: str,
        top_k: int,
    ) -> dict:

        t0 = time.perf_counter()

        standalone = self._condense(question, history)

        t_condense = (time.perf_counter() - t0) * 1000

        result = self.retriever.retrieve(
            kb,
            standalone,
            mode=mode,
            top_k=top_k,
        )

        hits = self._filter(result)

        return {
            "standalone": standalone,
            "mode_used": result.mode_used,
            "sources": [
                _source(i + 1, h)
                for i, h in enumerate(hits)
            ],
            "timings": {
                "condense": t_condense,
                **result.timings_ms,
            },
        }

    # ------------------------------------------------------------------
    # Streaming answer
    # ------------------------------------------------------------------

    def answer_stream(
        self,
        kb: KnowledgeBase,
        question: str,
        history: list[dict],
        mode: str,
        top_k: int,
    ) -> Iterator[dict]:

        """Yield events: sources, token, done, or error."""

        started = time.perf_counter()

        if kb.is_empty():
            yield {
                "type": "token",
                "data": {
                    "text": "Upload at least one document, then ask me about it."
                },
            }

            yield {
                "type": "done",
                "data": {
                    "grounded": False,
                    "cited": [],
                    "timings_ms": {},
                    "mode": mode,
                    "model": "",
                },
            }

            return

        prep = self.prepare(kb, question, history, mode, top_k)

        sources = prep["sources"]

        yield {
            "type": "sources",
            "data": {
                "sources": sources,
                "standalone_question": prep["standalone"],
                "mode": prep["mode_used"],
            },
        }

        answer_parts: list[str] = []
        first_token_at = None

        model = self.llm.model if self.llm.available else "extractive"

        if not sources:
            text = prompts.NOT_FOUND
            grounded = False

            yield {
                "type": "token",
                "data": {"text": text},
            }

            answer_parts.append(text)

        elif not self.llm.available:
            text = (
                "_No LLM is configured, so here are the most relevant passages:_\n\n"
                + "\n\n".join(
                    f"**[{s['n']}] {s['doc_name']}, page {s['page']}**\n"
                    f"{s['text'][:600]}"
                    for s in sources[:3]
                )
            )

            grounded = True

            yield {
                "type": "token",
                "data": {"text": text},
            }

            answer_parts.append(text)

        else:
            try:
                for piece in self.llm.stream(
                    prompts.answer_messages(prep["standalone"], sources)
                ):

                    if first_token_at is None:
                        first_token_at = time.perf_counter()

                    answer_parts.append(piece)

                    yield {
                        "type": "token",
                        "data": {"text": piece},
                    }

            except LLMError as exc:
                yield {
                    "type": "error",
                    "data": {"message": str(exc)},
                }

                return

            answer = "".join(answer_parts)

            grounded = not answer.strip().lower().startswith(
                "i couldn't find the answer"
            )

        answer = "".join(answer_parts)

        # Extract valid citation numbers.
        cited = sorted({
            int(n)
            for n in re.findall(r"\[(\d{1,2})\]", answer)
            if 1 <= int(n) <= len(sources)
        })

        total = (time.perf_counter() - started) * 1000

        timings = {
            k: round(v, 1)
            for k, v in prep["timings"].items()
        }

        timings["total"] = round(total, 1)

        if first_token_at is not None:
            timings["first_token"] = round(
                (first_token_at - started) * 1000,
                1,
            )

        yield {
            "type": "done",
            "data": {
                "grounded": grounded,
                "cited": cited,
                "timings_ms": timings,
                "mode": prep["mode_used"],
                "model": model,
            },
        }

    # ------------------------------------------------------------------
    # Blocking answer (evaluation and tests)
    # ------------------------------------------------------------------

    def answer(
        self,
        kb: KnowledgeBase,
        question: str,
        history: list[dict],
        mode: str,
        top_k: int,
    ) -> dict:

        text = []
        sources = []
        done = {}

        for event in self.answer_stream(
            kb,
            question,
            history,
            mode,
            top_k,
        ):

            if event["type"] == "token":
                text.append(event["data"]["text"])

            elif event["type"] == "sources":
                sources = event["data"]["sources"]

            elif event["type"] == "done":
                done = event["data"]

            elif event["type"] == "error":
                raise LLMError(event["data"]["message"])

        return {
            "answer": "".join(text),
            "sources": sources,
            **done,
        }
