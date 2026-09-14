"""Extract Item 1A / 7 / 7A text from a 10-K's primary HTML document.

The fiddly part: 10-Ks contain a table of contents that repeats every item heading
near the top of the document, and MD&A / risk-factor prose routinely cross-references
other items ("as discussed in Item 1A..."). A naive "find Item 7" regex will match
those before it matches the real heading.

Two heuristics fix this without needing a layout-aware parser:
1. Anchor each heading pattern to the start of a line — real headings render as their
   own paragraph/heading block; cross-references are mid-sentence.
2. Take the *last* line-anchored match — the ToC entry (also line-anchored, one item
   per line) always precedes the real heading later in the document body.

This is still brittle against unusual filing layouts (see README limitations).
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from filings_analyst.obs.logging import get_logger

logger = get_logger(__name__)

_SEP = r"[.\-:–—]?"

# Order matters: sections are sliced between consecutive boundaries in this order.
_ITEM_BOUNDARIES: list[tuple[str, re.Pattern[str]]] = [
    ("1A", re.compile(rf"^[ \t]*item\s+1a{_SEP}\s", re.IGNORECASE | re.MULTILINE)),
    ("7", re.compile(rf"^[ \t]*item\s+7{_SEP}\s", re.IGNORECASE | re.MULTILINE)),
    ("7A", re.compile(rf"^[ \t]*item\s+7a{_SEP}\s", re.IGNORECASE | re.MULTILINE)),
    ("8", re.compile(rf"^[ \t]*item\s+8{_SEP}\s", re.IGNORECASE | re.MULTILINE)),
]

SECTION_TITLES = {
    "1A": "Risk Factors",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
}


class SectionExtractionError(RuntimeError):
    pass


_MIN_REPEATED_LINE_LEN = 4
_REPEATED_LINE_THRESHOLD = 4


_DIGITS = re.compile(r"\d+")


def _strip_repeated_lines(lines: list[str]) -> list[str]:
    """Drop running page headers/footers: short lines repeated many times per page.

    e.g. "JPMorgan Chase & Co./2023 Form 10-K" reappears verbatim on every page; Apple's
    "Apple Inc. | 2023 Form 10-K | 19" reappears with only the page number changing. Both
    are caught by comparing lines with digit runs normalized to a placeholder — a real
    paragraph of body text doesn't repeat even after that normalization.
    """
    counts: dict[str, int] = {}
    for line in lines:
        if len(line) >= _MIN_REPEATED_LINE_LEN:
            key = _DIGITS.sub("#", line)
            counts[key] = counts.get(key, 0) + 1
    repeated = {key for key, n in counts.items() if n >= _REPEATED_LINE_THRESHOLD}
    return [
        line
        for line in lines
        if len(line) < _MIN_REPEATED_LINE_LEN or _DIGITS.sub("#", line) not in repeated
    ]


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    # Inline-XBRL documents duplicate every tagged fact inside a hidden header block.
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.IGNORECASE)):
        tag.decompose()

    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    lines = _strip_repeated_lines(lines)
    return "\n".join(lines)


def _last_match_start(pattern: re.Pattern[str], text: str) -> int | None:
    starts = [m.start() for m in pattern.finditer(text)]
    return starts[-1] if starts else None


def extract_sections(html: str) -> dict[str, str]:
    """Return {"1A": text, "7": text, "7A": text} for the given 10-K primary document HTML."""
    text = html_to_text(html)

    boundary_positions: dict[str, int] = {}
    for item, pattern in _ITEM_BOUNDARIES:
        pos = _last_match_start(pattern, text)
        if pos is None:
            raise SectionExtractionError(f"Could not locate 'Item {item}' heading in document")
        boundary_positions[item] = pos

    ordered = [item for item, _ in _ITEM_BOUNDARIES]
    positions = [boundary_positions[item] for item in ordered]
    if positions != sorted(positions):
        raise SectionExtractionError(
            f"Item headings out of expected document order: {boundary_positions}"
        )

    sections: dict[str, str] = {}
    for i, item in enumerate(ordered[:-1]):
        start = boundary_positions[item]
        end = boundary_positions[ordered[i + 1]]
        section_text = text[start:end].strip()
        if len(section_text) < 40:
            raise SectionExtractionError(
                f"Item {item} section is empty ({len(section_text)} chars) — "
                "boundary detection failed"
            )
        if len(section_text) < 600:
            # Some filers (e.g. bank holding companies) present Item 7/7A in the
            # strict Part II numbering as a short "incorporated by reference"
            # pointer to a separately-labeled MD&A elsewhere in the document.
            # That's real filing content, not a broken extraction — log and keep it.
            logger.warning(
                "edgar.section_thin",
                item=item,
                chars=len(section_text),
                preview=section_text[:200],
            )
        sections[item] = section_text

    logger.info("edgar.sections_extracted", sizes={k: len(v) for k, v in sections.items()})
    return sections
