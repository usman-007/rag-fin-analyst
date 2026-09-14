"""Request/response schemas for the FastAPI app."""

from __future__ import annotations

from pydantic import BaseModel, Field

from filings_analyst.agent.state import Citation, Contradiction, Fact


class AskRequest(BaseModel):
    question: str
    thread_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    facts: list[Fact]
    contradictions: list[Contradiction]
    refused: bool
    trace_url: str | None
    run_id: str | None
    latency_ms: float
    llm_calls: int
    errors: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    run_id: str
    score: int = Field(ge=0, le=1)
    comment: str | None = None


class FeedbackResponse(BaseModel):
    ok: bool
