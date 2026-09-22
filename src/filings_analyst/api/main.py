"""FastAPI app: /ask, /feedback, /healthz, /readyz, and the static demo page."""

from __future__ import annotations

import utils.transport
import utils.session
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langsmith import Client as LangSmithClient

from filings_analyst.agent.graph import ask
from filings_analyst.api.schemas import AskRequest, AskResponse, FeedbackRequest, FeedbackResponse
from filings_analyst.obs.logging import get_logger
from filings_analyst.obs.tracing import current_run_id, current_trace_url, traced_ask
from filings_analyst.retrieval.store import FilingsStore
from filings_analyst.settings import get_settings

logger = get_logger(__name__)

app = FastAPI(title="Multi-Hop Filings Analyst")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.post("/ask", response_model=AskResponse)
def post_ask(req: AskRequest) -> AskResponse:
    start = time.monotonic()
    with traced_ask():
        result: dict[str, Any] = ask(req.question, req.thread_id)
        run_id = current_run_id()
        trace_url = current_trace_url()
    latency_ms = (time.monotonic() - start) * 1000

    scope = result.get("scope")
    refused = scope is not None and not scope.in_scope

    return AskResponse(
        answer=result.get("draft") or "",
        citations=result.get("citations", []),
        facts=result.get("facts", []),
        contradictions=result.get("contradictions", []),
        refused=refused,
        trace_url=trace_url,
        run_id=run_id,
        latency_ms=latency_ms,
        llm_calls=result.get("llm_calls", 0),
        errors=result.get("errors", []),
    )


@app.post("/feedback", response_model=FeedbackResponse)
def post_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Closes the observability loop: a thumbs up/down from the demo page becomes
    LangSmith feedback attached to the run that produced the answer."""
    settings = get_settings()
    if not settings.langsmith_api_key:
        raise HTTPException(status_code=503, detail="LangSmith is not configured")
    client = LangSmithClient(api_key=settings.langsmith_api_key.get_secret_value())
    try:
        client.create_feedback(
            run_id=req.run_id, key="user_score", score=req.score, comment=req.comment
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to record feedback: {exc}") from exc
    return FeedbackResponse(ok=True)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise HTTPException(status_code=503, detail="PINECONE_API_KEY not configured")
    try:
        stats = FilingsStore().describe_index_stats()
        vector_count = int(stats.get("total_vector_count") or 0)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Pinecone not reachable: {exc}") from exc
    if not vector_count:
        raise HTTPException(status_code=503, detail="Index has no vectors — run `make ingest`")
    return {"status": "ready", "vectors": str(vector_count)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
