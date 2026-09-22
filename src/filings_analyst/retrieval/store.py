"""Pinecone index management: server-side (integrated) embedding, batched upsert, filtered search.

One index, one namespace, metadata filters — not a namespace per company. Pinecone queries a
single namespace at a time, and this system's core query pattern is cross-company and cross-year;
namespacing by ticker would turn every comparative question into N queries plus a manual merge.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Any

from pinecone import IndexEmbed, Pinecone
from pinecone.errors.exceptions import RateLimitError
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt

from filings_analyst.ingest.chunker import count_tokens
from filings_analyst.obs.logging import get_logger
from filings_analyst.settings import get_settings

logger = get_logger(__name__)

UPSERT_BATCH_SIZE = 96

# Pinecone's Starter-plan cap for llama-text-embed-v2 (passage) embedding is
# 250,000 tokens/minute, confirmed live via a 429 during ingestion. Pace our own
# requests under that, with headroom for the tiktoken-vs-actual-tokenizer mismatch.
EMBED_TOKENS_PER_MINUTE_BUDGET = 180_000
_RATE_LIMIT_WINDOW_SECONDS = 60.0


def _rate_limit_wait(retry_state: RetryCallState) -> float:
    """Wait as long as the server actually asked (`Retry-After`), not a guess."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, RateLimitError) and exc.retry_after is not None:
        return float(exc.retry_after) + 1.0
    return _RATE_LIMIT_WINDOW_SECONDS + 5.0


@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(3),
    wait=_rate_limit_wait,
    reraise=True,
)
def _upsert_batch(index: Any, namespace: str, batch: list[dict[str, Any]]) -> None:
    index.upsert_records(namespace=namespace, records=batch)


class FilingsStore:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")
        self._settings = settings
        self._pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        self._index_cache: Any = None
        self._embed_window: deque[tuple[float, int]] = deque()

    def _wait_for_embed_budget(self, tokens: int) -> None:
        """Block until sending `tokens` more keeps the trailing 60s under budget."""
        while True:
            now = time.monotonic()
            while (
                self._embed_window and now - self._embed_window[0][0] > _RATE_LIMIT_WINDOW_SECONDS
            ):
                self._embed_window.popleft()
            used = sum(t for _, t in self._embed_window)
            if used + tokens <= EMBED_TOKENS_PER_MINUTE_BUDGET or not self._embed_window:
                self._embed_window.append((now, tokens))
                return
            sleep_for = _RATE_LIMIT_WINDOW_SECONDS - (now - self._embed_window[0][0]) + 0.5
            logger.info("pinecone.embed_budget_pause", sleep_seconds=round(sleep_for, 1), used=used)
            time.sleep(max(sleep_for, 0.1))

    def ensure_index(self) -> None:
        settings = self._settings
        existing = {idx["name"] for idx in self._pc.list_indexes()}
        if settings.pinecone_index in existing:
            logger.info("pinecone.index_exists", index=settings.pinecone_index)
            return
        logger.info("pinecone.create_index", index=settings.pinecone_index)
        self._pc.create_index_for_model(
            name=settings.pinecone_index,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
            embed=IndexEmbed(
                model=settings.pinecone_embed_model,
                field_map={"text": "chunk_text"},
            ),
        )

    def _index(self) -> Any:
        if self._index_cache is None:
            self._index_cache = self._pc.Index(name=self._settings.pinecone_index)
        return self._index_cache

    def describe_index_stats(self) -> dict[str, Any]:
        """Used by `/readyz` to confirm the index exists and actually has vectors in it."""
        stats = self._index().describe_index_stats()
        return stats.to_dict()  # type: ignore[no-any-return]

    def upsert_records(self, records: list[dict[str, Any]]) -> int:
        index = self._index()
        total = 0
        for i in range(0, len(records), UPSERT_BATCH_SIZE):
            batch = records[i : i + UPSERT_BATCH_SIZE]
            batch_tokens = sum(count_tokens(r.get("chunk_text", "")) for r in batch)
            self._wait_for_embed_budget(batch_tokens)
            _upsert_batch(index, self._settings.pinecone_namespace, batch)
            total += len(batch)
            logger.info(
                "pinecone.upsert_batch",
                batch_size=len(batch),
                batch_tokens=batch_tokens,
                total=total,
            )
        return total

    def search(
        self,
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        settings = self._settings
        index = self._index()
        query: dict[str, Any] = {
            "inputs": {"text": query_text},
            "top_k": top_k or settings.top_k,
        }
        if filters:
            query["filter"] = filters
        response = index.search(namespace=settings.pinecone_namespace, query=query)
        hits = response.result.hits if hasattr(response, "result") else response["result"]["hits"]
        results = []
        for hit in hits:
            fields = hit["fields"] if isinstance(hit, dict) else hit.fields
            record_id = hit["_id"] if isinstance(hit, dict) else hit._id
            score = hit["_score"] if isinstance(hit, dict) else hit._score
            results.append({"id": record_id, "score": score, **fields})
        return results


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Query the filings Pinecone index directly.")
    parser.add_argument("query", help="Natural-language search text")
    parser.add_argument("--ticker", action="append", dest="tickers", default=None)
    parser.add_argument("--fy", action="append", dest="fiscal_years", type=int, default=None)
    parser.add_argument("--item", action="append", dest="items", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    filters: dict[str, Any] = {}
    if args.tickers:
        filters["ticker"] = {"$in": args.tickers}
    if args.fiscal_years:
        filters["fiscal_year"] = {"$in": args.fiscal_years}
    if args.items:
        filters["item"] = {"$in": args.items}

    store = FilingsStore()
    results = store.search(args.query, filters=filters or None, top_k=args.top_k)
    for r in results:
        print(f"[{r['score']:.3f}] {r['id']} — {r.get('chunk_text', '')[:160]}")


if __name__ == "__main__":
    _cli()
