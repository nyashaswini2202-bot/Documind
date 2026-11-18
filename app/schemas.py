"""Pydantic request models (input validation for the API)."""
from typing import List, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: List[ChatTurn] = Field(default_factory=list, max_length=12)
    mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    top_k: int = Field(default=5, ge=1, le=10)


class EvalRequest(BaseModel):
    num_questions: int = Field(default=8, ge=3, le=25)
    top_k: int = Field(default=5, ge=1, le=10)
