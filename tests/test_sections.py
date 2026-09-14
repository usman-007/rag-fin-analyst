from pathlib import Path

import pytest

from filings_analyst.edgar.sections import (
    SectionExtractionError,
    _strip_repeated_lines,
    extract_sections,
    html_to_text,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k.html"


@pytest.fixture
def sections() -> dict[str, str]:
    html = FIXTURE.read_text()
    return extract_sections(html)


def test_extracts_all_three_items(sections: dict[str, str]) -> None:
    assert set(sections) == {"1A", "7", "7A"}


def test_toc_entries_are_excluded(sections: dict[str, str]) -> None:
    # The ToC line has "... 4" / "... 11" trailing page numbers; the real sections start
    # at the actual heading, not the ToC line, and should not repeat all five ToC entries.
    for text in sections.values():
        assert "TABLE OF CONTENTS" not in text
        assert (
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk ... 15" not in text
        )


def test_repeated_page_headers_and_footers_are_stripped(sections: dict[str, str]) -> None:
    for text in sections.values():
        assert "Acme Corp | 2024 Form 10-K" not in text
        # Bare "Item 1A" / "Item 7" / "Item 7A" running headers repeat on every page.
        assert "\nItem 1A\n" not in f"\n{text}\n"


def test_item_1a_starts_at_real_heading_not_toc(sections: dict[str, str]) -> None:
    assert sections["1A"].startswith("ITEM 1A. RISK FACTORS")


def test_item_1a_absorbs_1b_through_6_filler_by_design(sections: dict[str, str]) -> None:
    # Documented trade-off: only Items 1A/7/7A/8 are used as boundaries, so the filler
    # items between 1A and 7 end up bundled into the "1A" section text.
    assert "Unresolved Staff Comments" in sections["1A"]
    assert "Item 6. [Reserved]" in sections["1A"]


def test_item_7_is_bounded_correctly(sections: dict[str, str]) -> None:
    assert sections["7"].startswith("ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS")
    assert "Critical Accounting Estimates" in sections["7"]
    assert "QUANTITATIVE AND QUALITATIVE" not in sections["7"]


def test_item_7a_is_bounded_correctly(sections: dict[str, str]) -> None:
    assert sections["7A"].startswith("ITEM 7A. QUANTITATIVE AND QUALITATIVE")
    assert "Credit Risk" in sections["7A"]
    assert "Financial Statements and Supplementary Data" not in sections["7A"]


def test_missing_item_raises() -> None:
    html = (
        "<html><body><p>Item 1A. Risk Factors</p>"
        "<p>Some risk text here that is long enough.</p></body></html>"
    )
    with pytest.raises(SectionExtractionError):
        extract_sections(html)


def test_strip_repeated_lines_removes_frequent_short_lines() -> None:
    lines = ["Item 1A", "real content one", "Item 1A", "real content two", "Item 1A", "Item 1A"]
    result = _strip_repeated_lines(lines)
    assert "Item 1A" not in result
    assert "real content one" in result
    assert "real content two" in result


def test_strip_repeated_lines_keeps_lines_below_threshold() -> None:
    lines = ["Item 1A", "content", "Item 1A", "content"]
    result = _strip_repeated_lines(lines)
    assert result.count("Item 1A") == 2  # only 2 reps, below the threshold of 4


def test_html_to_text_strips_hidden_ixbrl() -> None:
    html = (
        '<html><body><div style="display:none">SECRET_HIDDEN_FACT</div>'
        "<p>Visible text</p></body></html>"
    )
    text = html_to_text(html)
    assert "SECRET_HIDDEN_FACT" not in text
    assert "Visible text" in text
