"""Graph structure smoke tests: compiles, and routes correctly end-to-end with every LLM
call and external I/O call (Pinecone, EDGAR/XBRL) replaced by a stub — no network, no keys.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from filings_analyst.agent import nodes
from filings_analyst.agent.graph import build_graph, initial_state
from filings_analyst.agent.state import (
    ChunkGrade,
    Citation,
    CompareOutput,
    Evidence,
    Fact,
    FactBatch,
    GradeBatch,
    PlanOutput,
    ScopeDecision,
    SubQuery,
    SynthesisOutput,
)

STUB_CHUNK_ID = "AAPL-2024-7-0000"
STUB_EVIDENCE = Evidence(
    chunk_id=STUB_CHUNK_ID,
    text="Apple's total net sales were $391,035 million in fiscal 2024.",
    ticker="AAPL",
    fiscal_year=2024,
    item="7",
    source_url="https://example.com/aapl-2024.htm",
    score=0.95,
)
STUB_FACT = Fact(
    claim="AAPL FY2024 revenue",
    value="391035000000",
    unit="USD",
    source=STUB_CHUNK_ID,
)


def test_graph_compiles() -> None:
    graph = build_graph().compile()
    assert graph is not None


def _fake_structured_invoke(llm: Any, schema: type, prompt: str) -> Any:
    if schema is ScopeDecision:
        return ScopeDecision(in_scope=True, tickers=["AAPL"], fiscal_years=[2024], reason="ok")
    if schema is PlanOutput:
        return PlanOutput(
            sub_queries=[
                SubQuery(text="AAPL revenue", tickers=["AAPL"], fiscal_years=[2024], items=["7"])
            ]
        )
    if schema is GradeBatch:
        return GradeBatch(
            grades=[ChunkGrade(chunk_id=STUB_CHUNK_ID, relevant=True, reason="on point")]
        )
    if schema is SubQuery:
        # Only one stub chunk ever comes back, so grade's MIN_RELEVANT_EVIDENCE=2
        # threshold is never met — this exercises the grade -> refine -> retrieve cycle,
        # bounded by MAX_HOPS, rather than skipping it.
        return SubQuery(text="Apple revenue (broadened)", tickers=["AAPL"], fiscal_years=[2024])
    if schema is FactBatch:
        return FactBatch(facts=[STUB_FACT])
    if schema is CompareOutput:
        return CompareOutput(additional_facts=[], contradictions=[])
    if schema is SynthesisOutput:
        return SynthesisOutput(
            answer="Apple's FY2024 revenue was $391,035,000,000. [C1]",
            citations=[
                Citation(
                    marker="[C1]",
                    chunk_id=STUB_CHUNK_ID,
                    source_url=STUB_EVIDENCE.source_url,
                    text_snippet=STUB_EVIDENCE.text,
                )
            ],
        )
    raise AssertionError(f"unexpected structured_invoke schema: {schema}")


@pytest.fixture
def stubbed_graph(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(nodes, "structured_invoke", _fake_structured_invoke)
    monkeypatch.setattr(nodes, "get_fast_llm", lambda: MagicMock())
    monkeypatch.setattr(nodes, "get_long_llm", lambda: MagicMock())
    monkeypatch.setattr(nodes, "_store", lambda: None)
    monkeypatch.setattr(nodes, "search_filings", lambda sub_query, store: [STUB_EVIDENCE])
    return build_graph().compile()


def test_in_scope_question_routes_through_full_pipeline_to_a_clean_answer(
    stubbed_graph: Any,
) -> None:
    state = initial_state("What was Apple's FY2024 revenue?", thread_id="test-thread")
    result = stubbed_graph.invoke(state, config={"max_concurrency": 3})

    assert result["scope"] is not None
    assert result["scope"].in_scope is True
    assert len(result["plan"]) == 1
    assert any(e.chunk_id == STUB_CHUNK_ID for e in result["evidence"])
    assert result["draft"] is not None
    assert "[C1]" in result["draft"]
    assert result["citations"] and result["citations"][0].chunk_id == STUB_CHUNK_ID
    # A clean, well-grounded synthesis should pass verify on the first attempt.
    assert result["violations"] == []
    assert result["synth_attempts"] == 1


def test_out_of_scope_question_routes_to_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_invoke(llm: Any, schema: type, prompt: str) -> Any:
        if schema is ScopeDecision:
            return ScopeDecision(
                in_scope=False, tickers=[], fiscal_years=[], reason="Tesla is not in the corpus."
            )
        raise AssertionError("only triage should run on the refuse path")

    monkeypatch.setattr(nodes, "structured_invoke", fake_invoke)
    monkeypatch.setattr(nodes, "get_fast_llm", lambda: MagicMock())
    graph = build_graph().compile()

    state = initial_state("What was Tesla's FY2024 free cash flow?", thread_id="test-thread-2")
    result = graph.invoke(state, config={"max_concurrency": 3})

    assert result["plan"] == []  # never reached planning/retrieval
    assert result["evidence"] == []
    assert "Tesla is not in the corpus" in result["draft"]
    assert result["citations"] == []
