"""Unit tests for the deterministic `verify` node — the guard that stops a fabricated
citation or an invented number from reaching the user, regardless of what the
synthesizer produced.
"""

from __future__ import annotations

from filings_analyst.agent.nodes import _find_violations, _strip_unsupported_sentences, verify
from filings_analyst.agent.state import AgentState, Citation, Evidence, Fact

EVIDENCE = [
    Evidence(
        chunk_id="AAPL-2024-7-0000",
        text="Revenue was $391,035 million in fiscal 2024.",
        ticker="AAPL",
        fiscal_year=2024,
        item="7",
        source_url="https://example.com/aapl-2024.htm",
        score=0.9,
        relevant=True,
    )
]
FACTS = [
    Fact(claim="AAPL FY2024 revenue", value="391035000000", unit="USD", source="AAPL-2024-7-0000")
]


def _base_state(**overrides: object) -> AgentState:
    state: AgentState = {
        "question": "What was Apple's FY2024 revenue?",
        "thread_id": "t1",
        "scope": None,
        "plan": [],
        "hop": 0,
        "evidence": EVIDENCE,
        "pending_retrieval_indices": [],
        "xbrl_facts": [],
        "facts": FACTS,
        "contradictions": [],
        "draft": None,
        "citations": [],
        "violations": [],
        "synth_attempts": 1,
        "errors": [],
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_clean_answer_has_no_violations() -> None:
    state = _base_state(
        draft="Apple's FY2024 revenue was $391,035,000,000. [C1]",
        citations=[
            Citation(
                marker="[C1]",
                chunk_id="AAPL-2024-7-0000",
                source_url="https://example.com/aapl-2024.htm",
                text_snippet="Revenue was $391,035 million",
            )
        ],
    )
    result = verify(state)
    assert result["violations"] == []


def test_fabricated_citation_chunk_id_is_caught() -> None:
    """A citation pointing at a chunk_id that was never retrieved must be flagged."""
    state = _base_state(
        draft="Apple's FY2024 revenue was $391,035,000,000. [C1]",
        citations=[
            Citation(
                marker="[C1]",
                chunk_id="AAPL-2024-7-9999",  # never in `evidence`
                source_url="https://example.com/fabricated",
                text_snippet="fabricated",
            )
        ],
    )
    result = verify(state)
    assert any("not present in evidence" in v for v in result["violations"])


def test_fabricated_xbrl_source_is_caught() -> None:
    state = _base_state(
        draft="Apple's FY2024 net income was $93,736,000,000. [C1]",
        citations=[
            Citation(
                marker="[C1]",
                chunk_id="xbrl:NetIncomeLoss:2099",  # not in `facts`
                source_url="",
                text_snippet="",
            )
        ],
    )
    result = verify(state)
    assert any("unknown xbrl source" in v for v in result["violations"])


def test_unsupported_number_is_caught() -> None:
    """A number not present anywhere in `facts` must be flagged, even with a valid citation."""
    state = _base_state(
        draft="Apple's FY2024 revenue was $999,999,999,999. [C1]",
        citations=[
            Citation(
                marker="[C1]",
                chunk_id="AAPL-2024-7-0000",
                source_url="https://example.com/aapl-2024.htm",
                text_snippet="Revenue was $391,035 million",
            )
        ],
    )
    result = verify(state)
    assert any("not in `facts`" in v for v in result["violations"])


def test_dangling_marker_without_citation_entry_is_caught() -> None:
    state = _base_state(draft="Revenue grew. [C1]", citations=[])
    result = verify(state)
    assert any("not in `citations`" in v for v in result["violations"])


def test_repair_exhausted_strips_unsupported_sentences_and_flags_it() -> None:
    """After the repair budget is exhausted, verify must deterministically remove the
    offending sentence rather than let a fabricated claim through, and must say so in
    `errors`."""
    state = _base_state(
        draft=(
            "Apple's FY2024 revenue was $391,035,000,000. [C1] "
            "Apple's FY2024 net income was $999,999,999,999. [C2]"
        ),
        citations=[
            Citation(
                marker="[C1]",
                chunk_id="AAPL-2024-7-0000",
                source_url="https://example.com/aapl-2024.htm",
                text_snippet="Revenue was $391,035 million",
            ),
            Citation(
                marker="[C2]",
                chunk_id="AAPL-2024-7-9999",
                source_url="https://example.com/fabricated",
                text_snippet="fabricated",
            ),
        ],
        synth_attempts=2,
    )
    result = verify(state)
    assert result["violations"] == []
    assert "999,999,999,999" not in result["draft"]
    assert "391,035,000,000" in result["draft"]
    assert any("Repair failed" in e for e in result["errors"])
    assert all(c.marker != "[C2]" for c in result["citations"])


def test_find_violations_directly() -> None:
    violations = _find_violations(
        draft="Fabricated number $1,234,567. [C1]",
        citations=[
            Citation(marker="[C1]", chunk_id="AAPL-2024-7-0000", source_url="", text_snippet="")
        ],
        evidence_ids={"AAPL-2024-7-0000"},
        fact_sources={"AAPL-2024-7-0000"},
        fact_numbers=set(),
    )
    assert violations


def test_strip_unsupported_sentences_keeps_supported_ones() -> None:
    cleaned, removed = _strip_unsupported_sentences(
        draft="Good sentence with 100. [C1] Bad sentence with 200. [C2]",
        fact_numbers={"100"},
        valid_markers={"[C1]"},
    )
    assert "Good sentence" in cleaned
    assert "Bad sentence" not in cleaned
    assert len(removed) >= 1
