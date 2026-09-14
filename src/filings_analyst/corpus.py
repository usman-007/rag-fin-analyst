"""Typed loaders for config/corpus.yaml and config/xbrl_tags.yaml.

Shared by ingestion (M1) and the agent's triage/xbrl_lookup nodes (M3) — the fixed
company/fiscal-year/tag universe must never be hardcoded in more than one place.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

CORPUS_PATH = Path("config/corpus.yaml")
XBRL_TAGS_PATH = Path("config/xbrl_tags.yaml")


class CompanyConfig(BaseModel):
    ticker: str
    cik: str
    name: str
    fiscal_years: list[int]


class SectionConfig(BaseModel):
    item: str
    title: str


class CorpusConfig(BaseModel):
    companies: list[CompanyConfig]
    sections: list[SectionConfig]
    form_type: str

    def company_by_ticker(self, ticker: str) -> CompanyConfig | None:
        for company in self.companies:
            if company.ticker.upper() == ticker.upper():
                return company
        return None

    @property
    def tickers(self) -> list[str]:
        return [c.ticker for c in self.companies]

    @property
    def items(self) -> list[str]:
        return [s.item for s in self.sections]


def load_corpus(path: Path | str = CORPUS_PATH) -> CorpusConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return CorpusConfig(**data)


def load_xbrl_tag_map(path: Path | str = XBRL_TAGS_PATH) -> dict[str, list[str]]:
    with open(path) as f:
        data = yaml.safe_load(f)
    concepts: dict[str, list[str]] = data["concepts"]
    return concepts
