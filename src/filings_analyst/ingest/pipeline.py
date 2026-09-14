"""End-to-end ingest: EDGAR -> sections -> chunks -> Pinecone, with an embedding-budget guard.

`--dry-run` prints the full plan (filings, sections, chunk counts, token estimate) without
touching Pinecone. `--only TICKER` restricts to one company for fast iteration. `--force`
bypasses the MAX_EMBED_TOKENS guard.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

from filings_analyst.corpus import load_corpus
from filings_analyst.edgar.client import EdgarClient
from filings_analyst.edgar.filings import FilingRef, resolve_10ks
from filings_analyst.edgar.sections import extract_sections
from filings_analyst.ingest.chunker import chunk_section, count_tokens
from filings_analyst.obs.logging import get_logger
from filings_analyst.settings import get_settings

logger = get_logger(__name__)


@dataclass
class PlanRow:
    ticker: str
    fiscal_year: int
    item: str
    n_chunks: int
    n_tokens: int


@dataclass
class IngestPlan:
    filings: list[FilingRef] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    rows: list[PlanRow] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(row.n_tokens for row in self.rows)


def build_plan(client: EdgarClient, only_ticker: str | None = None) -> IngestPlan:
    corpus = load_corpus()
    plan = IngestPlan()

    for company in corpus.companies:
        if only_ticker and company.ticker.upper() != only_ticker.upper():
            continue
        filings = resolve_10ks(
            client, company.ticker, company.cik, company.name, company.fiscal_years
        )
        plan.filings.extend(filings)

        for filing in filings:
            html = client.get_text(filing.source_url)
            sections = extract_sections(html)
            for section_cfg in corpus.sections:
                text = sections.get(section_cfg.item)
                if not text:
                    continue
                chunks = chunk_section(text)
                n_tokens = 0
                for seq, chunk in enumerate(chunks):
                    chunk_tokens = count_tokens(chunk.text)
                    n_tokens += chunk_tokens
                    record_id = f"{filing.ticker}-{filing.fiscal_year}-{section_cfg.item}-{seq:04d}"
                    plan.records.append(
                        {
                            "_id": record_id,
                            "chunk_text": chunk.text,
                            "ticker": filing.ticker,
                            "cik": filing.cik,
                            "company": filing.company,
                            "fiscal_year": filing.fiscal_year,
                            "form_type": filing.form_type,
                            "item": section_cfg.item,
                            "section_title": section_cfg.title,
                            "accession": filing.accession,
                            "source_url": filing.source_url,
                            "char_start": chunk.char_start,
                            "char_end": chunk.char_end,
                            "seq": seq,
                        }
                    )
                plan.rows.append(
                    PlanRow(
                        filing.ticker, filing.fiscal_year, section_cfg.item, len(chunks), n_tokens
                    )
                )
    return plan


def print_plan(plan: IngestPlan) -> None:
    header = f"{'TICKER':8}{'FY':6}{'ITEM':6}{'CHUNKS':8}{'TOKENS':10}"
    print(header)
    print("-" * len(header))
    for row in plan.rows:
        print(f"{row.ticker:8}{row.fiscal_year:<6}{row.item:6}{row.n_chunks:<8}{row.n_tokens:<10}")
    print("-" * len(header))
    print(f"{'TOTAL':8}{'':6}{'':6}{len(plan.records):<8}{plan.total_tokens:<10}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the fixed 10-K corpus into Pinecone.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan; do not touch Pinecone"
    )
    parser.add_argument("--only", dest="only_ticker", default=None, help="Restrict to one ticker")
    parser.add_argument("--force", action="store_true", help="Bypass the MAX_EMBED_TOKENS guard")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_data_dirs()
    client = EdgarClient()

    logger.info("ingest.start", only=args.only_ticker, dry_run=args.dry_run)
    plan = build_plan(client, only_ticker=args.only_ticker)
    print_plan(plan)

    if plan.total_tokens > settings.max_embed_tokens and not args.force:
        print(
            f"\nEstimated {plan.total_tokens:,} tokens exceeds MAX_EMBED_TOKENS "
            f"({settings.max_embed_tokens:,}). Re-run with --force to proceed anyway."
        )
        return 1

    if args.dry_run:
        print("\n--dry-run: not touching Pinecone.")
        return 0

    from filings_analyst.retrieval.store import FilingsStore

    store = FilingsStore()
    store.ensure_index()
    n = store.upsert_records(plan.records)
    print(
        f"\nUpserted {n} records into '{settings.pinecone_index}' "
        f"(namespace '{settings.pinecone_namespace}')."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
