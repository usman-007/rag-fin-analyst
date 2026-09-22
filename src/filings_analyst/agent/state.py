"""Agent state: pydantic models for structured LLM I/O, a TypedDict for the graph state.

The core hallucination-containment invariant lives in how `Fact.source` is constrained:
every number that ends up in the final answer must trace back to either a `chunk_id` (an
extracted narrative claim) or an `"xbrl:{tag}:{fy}"` string (a structured fact) — never to
the synthesizer inventing a figure from prose. `verify` (nodes.py) enforces this at runtime.
"""

from __future__ import annotations

import types
from typing import Annotated, Any, Literal, TypedDict, Union, get_args, get_origin

from pydantic import BaseModel, Field, model_validator


def _is_list_type(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is list:
        return True
    if origin is Union or origin is types.UnionType:
        return any(get_origin(arg) is list for arg in get_args(annotation))
    return False


class _NullSafeListModel(BaseModel):
    """`openai/gpt-oss-120b` on Groq has been observed (live) emitting `null` for a
    list field it has nothing to put in, instead of `[]`. Groq validates tool-call
    arguments against the JSON schema server-side and 400s before pydantic ever runs,
    so the fix has two halves: the field types below are widened to `X | None` so the
    generated schema accepts `null` in the first place, and this validator normalizes
    it back to `[]` once pydantic does get to see it, so downstream code can still
    assume every list field is always a real list."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_lists(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for name, field in cls.model_fields.items():
                if data.get(name) is None and _is_list_type(field.annotation):
                    data[name] = []
        return data


class ScopeDecision(_NullSafeListModel):
    in_scope: bool
    tickers: list[str] | None = Field(default_factory=list)
    fiscal_years: list[int] | None = Field(default_factory=list)
    reason: str


class SubQuery(_NullSafeListModel):
    text: str
    tickers: list[str] | None = Field(default_factory=list)
    fiscal_years: list[int] | None = Field(default_factory=list)
    items: list[str] | None = Field(default_factory=list)  # "1A" | "7" | "7A"
    needs_xbrl: bool = False
    xbrl_concept: str | None = None


class PlanOutput(_NullSafeListModel):
    sub_queries: list[SubQuery] | None = Field(default_factory=list)


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


class GradeBatch(_NullSafeListModel):
    grades: list[ChunkGrade] | None = Field(default_factory=list)


class Fact(BaseModel):
    claim: str
    value: str | None = None  # numeric facts only
    unit: str | None = None
    period: str | None = None
    source: str  # chunk_id OR "xbrl:{tag}:{fy}"


class FactBatch(_NullSafeListModel):
    facts: list[Fact] | None = Field(default_factory=list)


class Contradiction(_NullSafeListModel):
    kind: Literal["narrative_vs_number", "mdna_vs_risk", "year_over_year"]
    description: str
    sources: list[str] | None = Field(default_factory=list)


class CompareOutput(_NullSafeListModel):
    additional_facts: list[Fact] | None = Field(default_factory=list)
    contradictions: list[Contradiction] | None = Field(default_factory=list)


class Citation(BaseModel):
    marker: str = Field(description='Exactly the bracketed marker used inline, e.g. "[C1]"')
    chunk_id: str
    source_url: str
    text_snippet: str


class SynthesisOutput(_NullSafeListModel):
    answer: str  # prose containing inline [C1], [C2]... markers
    citations: list[Citation] | None = Field(default_factory=list)


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
