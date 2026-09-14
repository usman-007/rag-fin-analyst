"""Agent state: pydantic models for structured LLM I/O, a TypedDict for the graph state.

The core hallucination-containment invariant lives in how `Fact.source` is constrained:
every number that ends up in the final answer must trace back to either a `chunk_id` (an
extracted narrative claim) or an `"xbrl:{tag}:{fy}"` string (a structured fact) — never to
the synthesizer inventing a figure from prose. `verify` (nodes.py) enforces this at runtime.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class ScopeDecision(BaseModel):
    in_scope: bool
    tickers: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)
    reason: str


class SubQuery(BaseModel):
    text: str
    tickers: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)
    items: list[str] = Field(default_factory=list)  # "1A" | "7" | "7A"
    needs_xbrl: bool = False
    xbrl_concept: str | None = None


class PlanOutput(BaseModel):
    sub_queries: list[SubQuery]


class Evidence(BaseModel):
    chunk_id: str
    text: str
    ticker: str
    fiscal_year: int
    item: str
    source_url: str
    score: float
    relevant: bool | None = None
    reason: str | None = None
    sub_query_index: int = -1


class ChunkGrade(BaseModel):
    chunk_id: str
    relevant: bool
    reason: str


class GradeBatch(BaseModel):
    grades: list[ChunkGrade]


class Fact(BaseModel):
    claim: str
    value: str | None = None  # numeric facts only
    unit: str | None = None
    period: str | None = None
    source: str  # chunk_id OR "xbrl:{tag}:{fy}"


class FactBatch(BaseModel):
    facts: list[Fact]


class Contradiction(BaseModel):
    kind: Literal["narrative_vs_number", "mdna_vs_risk", "year_over_year"]
    description: str
    sources: list[str]


class CompareOutput(BaseModel):
    additional_facts: list[Fact] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)


class Citation(BaseModel):
    marker: str  # "[C1]"
    chunk_id: str
    source_url: str
    text_snippet: str


class SynthesisOutput(BaseModel):
    answer: str  # prose containing inline [C1], [C2]... markers
    citations: list[Citation]


def _merge_evidence(existing: list[Evidence], new: list[Evidence]) -> list[Evidence]:
    """Upsert by chunk_id: parallel `retrieve_one` branches add new chunk_ids, while
    `grade` re-emits already-present chunk_ids with `relevant`/`reason` filled in — those
    must replace the ungraded copy, not sit alongside it as a duplicate."""
    by_id = {e.chunk_id: e for e in existing}
    for e in new:
        by_id[e.chunk_id] = e
    return list(by_id.values())


class AgentState(TypedDict):
    question: str
    thread_id: str
    scope: ScopeDecision | None
    plan: list[SubQuery]
    hop: int
    # Send-based fan-out (plan/refine -> retrieve_one) writes here concurrently;
    # the reducer upserts by chunk_id so parallel branches and later grading merge
    # instead of overwriting or duplicating each other.
    evidence: Annotated[list[Evidence], _merge_evidence]
    pending_retrieval_indices: list[int]
    xbrl_facts: list[Fact]
    facts: list[Fact]
    contradictions: list[Contradiction]
    draft: str | None
    citations: list[Citation]
    violations: list[str]
    synth_attempts: int
    errors: list[str]
