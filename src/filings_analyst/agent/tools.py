"""Retrieval and XBRL functions used by the `retrieve` and `xbrl_lookup` graph nodes.

Neither function involves an LLM call — `search_filings` is a vector search (plus a
budgeted rerank), and `get_xbrl_fact` is a deterministic structured-data lookup. Keeping
them as plain functions (not LangChain @tools) matches this system's design: a fixed
StateGraph with explicit nodes, not an LLM freely choosing which tool to call.
"""

from __future__ import annotations

from typing import Any

from filings_analyst.agent.state import Evidence, Fact, SubQuery
from filings_analyst.corpus import load_corpus, load_xbrl_tag_map
from filings_analyst.edgar.client import EdgarClient, EdgarClientError
from filings_analyst.edgar.filings import resolve_10ks
from filings_analyst.edgar.xbrl import get_concept_value
from filings_analyst.obs.logging import get_logger
from filings_analyst.retrieval.rerank import rerank
from filings_analyst.retrieval.store import FilingsStore
from filings_analyst.settings import get_settings

logger = get_logger(__name__)


def search_filings(
    sub_query: SubQuery, store: FilingsStore, use_rerank: bool = True
) -> list[Evidence]:
    """Vector search + optional budgeted rerank for one sub-query. Returns ungraded Evidence."""
    settings = get_settings()
    filters: dict[str, Any] = {}
    if sub_query.tickers:
        filters["ticker"] = {"$in": sub_query.tickers}
    if sub_query.fiscal_years:
        filters["fiscal_year"] = {"$in": sub_query.fiscal_years}
    if sub_query.items:
        filters["item"] = {"$in": sub_query.items}

    hits = store.search(sub_query.text, filters=filters or None, top_k=settings.top_k)
    if use_rerank and hits:
        hits = rerank(sub_query.text, hits, top_n=settings.rerank_top_n)
    else:
        hits = hits[: settings.rerank_top_n]

    return [
        Evidence(
            chunk_id=hit["id"],
            text=hit.get("chunk_text", ""),
            ticker=hit.get("ticker", ""),
            fiscal_year=hit.get("fiscal_year", 0),
            item=hit.get("item", ""),
            source_url=hit.get("source_url", ""),
            score=hit.get("score", 0.0),
        )
        for hit in hits
    ]


_EDGAR_CLIENT: EdgarClient | None = None


def _edgar_client() -> EdgarClient:
    global _EDGAR_CLIENT
    if _EDGAR_CLIENT is None:
        _EDGAR_CLIENT = EdgarClient()
    return _EDGAR_CLIENT


def get_xbrl_fact(sub_query: SubQuery) -> list[Fact]:
    """Deterministic XBRL lookup for one sub-query — no LLM involved."""
    if not sub_query.needs_xbrl or not sub_query.xbrl_concept:
        return []
    corpus = load_corpus()
    tag_map = load_xbrl_tag_map()
    candidate_tags = tag_map.get(sub_query.xbrl_concept)
    if not candidate_tags:
        logger.warning("xbrl_tool.unknown_concept", concept=sub_query.xbrl_concept)
        return []

    client = _edgar_client()
    facts: list[Fact] = []
    for ticker in sub_query.tickers:
        company = corpus.company_by_ticker(ticker)
        if company is None:
            continue
        years = [
            fy
            for fy in (sub_query.fiscal_years or company.fiscal_years)
            if fy in company.fiscal_years
        ]
        for fy in years:
            try:
                filings = resolve_10ks(client, company.ticker, company.cik, company.name, [fy])
                xbrl_fact = get_concept_value(
                    client, filings[0], sub_query.xbrl_concept, candidate_tags
                )
            except EdgarClientError as exc:
                logger.warning("xbrl_tool.lookup_failed", ticker=ticker, fy=fy, error=str(exc))
                continue
            facts.append(
                Fact(
                    claim=f"{company.name} {sub_query.xbrl_concept} for FY{fy}",
                    value=str(xbrl_fact.value),
                    unit=xbrl_fact.unit,
                    period=xbrl_fact.period_end,
                    source=f"xbrl:{xbrl_fact.tag}:{fy}",
                )
            )
    return facts
