"""XBRL companyconcept lookups — the only source of numeric facts in this system.

Every candidate tag for a concept is tried until one resolves an entry that matches
the exact filing (same accession number and period-end date) rather than trusting the
API's own fiscal-year/period labels, which are sometimes inconsistent across filers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from filings_analyst.edgar.client import EdgarClient, EdgarClientError
from filings_analyst.edgar.filings import FilingRef
from filings_analyst.obs.logging import get_logger

logger = get_logger(__name__)

CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"


class XbrlFact(BaseModel):
    concept: str
    tag: str
    value: float
    unit: str
    fiscal_year: int
    period_end: str
    period_start: str | None
    ticker: str
    cik: str
    accession: str


def _fetch_tag(client: EdgarClient, cik: str, tag: str) -> dict[str, Any] | None:
    url = CONCEPT_URL.format(cik=cik, tag=tag)
    try:
        return client.get_json(url)
    except EdgarClientError:
        return None  # tag not reported by this filer — caller tries the next candidate


def get_concept_value(
    client: EdgarClient,
    filing: FilingRef,
    concept: str,
    candidate_tags: list[str],
) -> XbrlFact:
    """Resolve `concept` for the exact filing in `filing`, trying each candidate tag in order."""
    for tag in candidate_tags:
        payload = _fetch_tag(client, filing.cik, tag)
        if payload is None:
            continue
        for unit, entries in payload.get("units", {}).items():
            for entry in entries:
                if (
                    entry.get("form") == filing.form_type
                    and entry.get("accn") == filing.accession
                    and entry.get("end") == filing.report_date
                ):
                    logger.info(
                        "xbrl.resolved",
                        ticker=filing.ticker,
                        concept=concept,
                        tag=tag,
                        value=entry["val"],
                    )
                    return XbrlFact(
                        concept=concept,
                        tag=tag,
                        value=float(entry["val"]),
                        unit=unit,
                        fiscal_year=filing.fiscal_year,
                        period_end=entry["end"],
                        period_start=entry.get("start"),
                        ticker=filing.ticker,
                        cik=filing.cik,
                        accession=filing.accession,
                    )
    raise EdgarClientError(
        f"No candidate tag resolved concept '{concept}' for {filing.ticker} "
        f"FY{filing.fiscal_year} (tried {candidate_tags})"
    )
