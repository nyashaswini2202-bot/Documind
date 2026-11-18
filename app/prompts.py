"""All LLM prompts in one place, so behaviour can be tuned without touching pipeline code."""

NOT_FOUND = "I couldn't find the answer to that in the uploaded documents."

SYSTEM_PROMPT = (
    "You are DocuMind, a careful assistant that answers strictly from numbered context passages.\n"
    "Rules:\n"
    "1. Use only information that appears in the context passages. Never use outside knowledge.\n"
    "2. Cite the passage number in square brackets after every factual statement, e.g. [1] or [2][3].\n"
    f'3. If the context does not contain the answer, reply exactly: "{NOT_FOUND}" and nothing else.\n'
    "4. Be concise and clear. Use short bullet points when listing several items. Never invent citations."
)


def build_context(sources: list[dict]) -> str:
    return "\n\n".join(
        f"[{s['n']}] (source: {s['doc_name']}, page {s['page']})\n{s['text']}" for s in sources
    )


def answer_messages(question: str, sources: list[dict]) -> list[dict]:
    user = f"Context passages:\n\n{build_context(sources)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def condense_messages(history: list[dict], question: str) -> list[dict]:
    convo = "\n".join(f"{t['role'].capitalize()}: {t['content'][:500]}" for t in history[-4:])
    return [
        {
            "role": "system",
            "content": (
                "Rewrite the user's follow-up question as a single standalone search query, using the "
                "conversation for context. Keep names and key terms. Output only the rewritten question."
            ),
        },
        {"role": "user", "content": f"Conversation:\n{convo}\n\nFollow-up question: {question}\n\nStandalone question:"},
    ]


def qgen_messages(passage: str) -> list[dict]:
    return [
        {"role": "system", "content": "You write evaluation questions for a document Q&A system. Reply with JSON only."},
        {
            "role": "user",
            "content": (
                "Read the PASSAGE and write ONE specific question that can be answered using only this passage. "
                "Do not mention 'the passage' or 'the text'. Also write a short reference answer.\n\n"
                f"PASSAGE:\n{passage}\n\n"
                'Return JSON: {"question": "...", "answer": "..."}'
            ),
        },
    ]


def judge_messages(question: str, answer: str, contexts: list[str], reference: str) -> list[dict]:
    ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    return [
        {"role": "system", "content": "You are a strict, impartial evaluator of a RAG system. Reply with JSON only."},
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{answer}\n\nREFERENCE_ANSWER:\n{reference}\n\n"
                "Return JSON with these keys:\n"
                '"total_claims": integer, number of distinct factual claims made in ANSWER,\n'
                '"supported_claims": integer, how many of those claims are fully supported by CONTEXT,\n'
                '"relevancy": number from 0 to 1, how directly ANSWER addresses QUESTION,\n'
                '"correctness": number from 0 to 1, how well ANSWER agrees with REFERENCE_ANSWER.'
            ),
        },
    ]

SUMMARY_SYSTEM_PROMPT = (
    "You are DocuMind, an expert document summarization assistant. "
    "Summarize the provided document content accurately and comprehensively. "
    "Cover the main ideas, important facts, key findings, and conclusions. "
    "Use clear headings and bullet points where appropriate. "
    "Do not introduce information that is not present in the document. "
    "Avoid unnecessary repetition."
)


def summarize_messages(document_text: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": SUMMARY_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": (
                "Prepare a comprehensive summary of the following document:\n\n"
                f"{document_text}"
            )
        }
    ]
