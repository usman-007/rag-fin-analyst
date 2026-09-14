"""Budgeted reranking via pinecone-rerank-v0.

Pinecone Starter allows 500 rerank requests/month. A month-keyed counter is kept on disk;
once exhausted we log a warning and fall back to the unranked vector order rather than
erroring — reranking is a quality improvement, not a correctness requirement.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from pinecone import Pinecone

from filings_analyst.obs.logging import get_logger
from filings_analyst.settings import get_settings

logger = get_logger(__name__)

BUDGET_FILE = Path("data/rerank_budget.json")


def _month_key(today: date | None = None) -> str:
    d = today or date.today()
    return f"{d.year:04d}-{d.month:02d}"


def _load_budget() -> dict[str, int]:
    if BUDGET_FILE.exists():
        return json.loads(BUDGET_FILE.read_text())  # type: ignore[no-any-return]
    return {}


def _save_budget(data: dict[str, int]) -> None:
    BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    BUDGET_FILE.write_text(json.dumps(data))


def remaining_budget() -> int:
    settings = get_settings()
    used = _load_budget().get(_month_key(), 0)
    return max(0, settings.rerank_monthly_budget - used)


def _record_usage(n: int = 1) -> None:
    data = _load_budget()
    key = _month_key()
    data[key] = data.get(key, 0) + n
    _save_budget(data)


def rerank(
    query: str,
    documents: list[dict[str, Any]],
    top_n: int | None = None,
    text_field: str = "chunk_text",
) -> list[dict[str, Any]]:
    """Rerank `documents` (each a dict containing `text_field`), returning the top `top_n`.

    Falls back to the input order, truncated to `top_n`, when the monthly budget is
    exhausted or no Pinecone key is configured.
    """
    settings = get_settings()
    n = top_n or settings.rerank_top_n
    if not documents:
        return []
    if remaining_budget() <= 0:
        logger.warning("rerank.budget_exhausted", requested=len(documents))
        return documents[:n]
    if not settings.pinecone_api_key:
        logger.warning("rerank.no_api_key")
        return documents[:n]

    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    result = pc.inference.rerank(
        model=settings.pinecone_rerank_model,
        query=query,
        documents=documents,
        rank_fields=[text_field],
        top_n=n,
    )
    _record_usage(1)
    reranked = [item.document for item in result.data if item.document is not None]
    logger.info(
        "rerank.done",
        n_in=len(documents),
        n_out=len(reranked),
        remaining=remaining_budget(),
    )
    return reranked
