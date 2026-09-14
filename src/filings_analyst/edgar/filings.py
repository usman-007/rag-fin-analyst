"""Resolve 10-K filing metadata and document URLs from the SEC submissions API."""

from __future__ import annotations

from pydantic import BaseModel

from filings_analyst.edgar.client import EdgarClient, EdgarClientError
from filings_analyst.obs.logging import get_logger

logger = get_logger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodashes}/{doc}"


class FilingRef(BaseModel):
    ticker: str
    cik: str
    company: str
    fiscal_year: int
    form_type: str
    accession: str
    filing_date: str
    report_date: str
    primary_document: str
    source_url: str


def _document_url(cik: str, accession: str, primary_document: str) -> str:
    cik_int = str(int(cik))
    accession_nodashes = accession.replace("-", "")
    return ARCHIVE_URL.format(
        cik_int=cik_int, accession_nodashes=accession_nodashes, doc=primary_document
    )


def _scan_10ks(
    ticker: str,
    cik: str,
    company: str,
    fiscal_years: set[int],
    forms: list[str],
    accessions: list[str],
    filing_dates: list[str],
    report_dates: list[str],
    primary_docs: list[str],
    into: dict[int, FilingRef],
) -> None:
    for i, form in enumerate(forms):
        if form != "10-K":
            continue
        fy = int(report_dates[i][:4])
        if fy not in fiscal_years or fy in into:
            continue
        into[fy] = FilingRef(
            ticker=ticker,
            cik=cik,
            company=company,
            fiscal_year=fy,
            form_type=form,
            accession=accessions[i],
            filing_date=filing_dates[i],
            report_date=report_dates[i],
            primary_document=primary_docs[i],
            source_url=_document_url(cik, accessions[i], primary_docs[i]),
        )


def resolve_10ks(
    client: EdgarClient,
    ticker: str,
    cik: str,
    company: str,
    fiscal_years: list[int],
) -> list[FilingRef]:
    """Fetch submissions.json and pick the 10-K for each requested fiscal year.

    Fiscal year is taken as the calendar year of `reportDate` (the filing's period end),
    which matches how AAPL/MSFT/NVDA/JPM label their own fiscal years for the FY2023/FY2024
    filings this corpus targets.

    High-volume filers (e.g. JPM files 2,000+ forms/month) push their 10-Ks out of the
    `filings.recent` window into paginated `filings.files[]` submissions almost immediately,
    so we walk those pages (newest first) until every requested fiscal year is found.
    """
    url = SUBMISSIONS_URL.format(cik=cik)
    payload = client.get_json(url)
    filings_block = payload.get("filings", {})
    recent = filings_block.get("recent", {})
    if not recent.get("form"):
        raise EdgarClientError(f"No 'recent' filings found for {ticker} ({cik})")

    wanted = set(fiscal_years)
    candidates_by_year: dict[int, FilingRef] = {}
    _scan_10ks(
        ticker,
        cik,
        company,
        wanted,
        recent["form"],
        recent["accessionNumber"],
        recent["filingDate"],
        recent["reportDate"],
        recent["primaryDocument"],
        candidates_by_year,
    )

    for file_meta in filings_block.get("files", []):
        if set(candidates_by_year) >= wanted:
            break
        page_url = f"https://data.sec.gov/submissions/{file_meta['name']}"
        page = client.get_json(page_url)
        _scan_10ks(
            ticker,
            cik,
            company,
            wanted,
            page.get("form", []),
            page.get("accessionNumber", []),
            page.get("filingDate", []),
            page.get("reportDate", []),
            page.get("primaryDocument", []),
            candidates_by_year,
        )

    missing = [fy for fy in fiscal_years if fy not in candidates_by_year]
    if missing:
        raise EdgarClientError(
            f"{ticker} ({cik}): no 10-K found for fiscal year(s) {missing} in submissions.json"
        )

    results = [candidates_by_year[fy] for fy in fiscal_years]
    logger.info(
        "edgar.resolved_10ks",
        ticker=ticker,
        filings=[(f.fiscal_year, f.accession) for f in results],
    )
    return results
