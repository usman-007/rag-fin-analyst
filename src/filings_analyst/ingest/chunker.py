"""Paragraph-aware chunker targeting ~450 tokens with 80 token overlap.

Token counts are an estimate (tiktoken's cl100k_base) used only for sizing and the
embedding-budget guard — the actual embedding model (llama-text-embed-v2) runs
server-side in Pinecone and tokenizes independently. Chunking is always called
per-item text, so chunks never cross Item 1A / 7 / 7A boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

TARGET_TOKENS = 450
OVERLAP_TOKENS = 80

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    text: str
    char_start: int
    char_end: int


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def _paragraphs_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """`text` is "\\n".join(lines) with no empty lines, so offsets are exact."""
    result = []
    pos = 0
    for line in text.split("\n"):
        result.append((line, pos, pos + len(line)))
        pos += len(line) + 1
    return result


def _split_long_paragraph(line: str, start: int, max_tokens: int) -> list[tuple[str, int, int]]:
    if count_tokens(line) <= max_tokens:
        return [(line, start, start + len(line))]
    parts: list[tuple[str, int, int]] = []
    pos = 0
    for sentence in _SENTENCE_SPLIT.split(line):
        if not sentence:
            continue
        s_start = line.index(sentence, pos)
        s_end = s_start + len(sentence)
        parts.append((sentence, start + s_start, start + s_end))
        pos = s_end
    return parts or [(line, start, start + len(line))]


def chunk_section(
    text: str,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """Greedily pack paragraphs (falling back to sentences for oversized ones) into chunks."""
    units: list[tuple[str, int, int]] = []
    for line, start, _end in _paragraphs_with_offsets(text):
        if not line.strip():
            continue
        units.extend(_split_long_paragraph(line, start, target_tokens))

    chunks: list[Chunk] = []
    current: list[tuple[str, int, int]] = []
    current_tokens = 0

    def flush() -> None:
        if current:
            chunks.append(
                Chunk(
                    text="\n".join(u[0] for u in current),
                    char_start=current[0][1],
                    char_end=current[-1][2],
                )
            )

    for unit in units:
        unit_tokens = count_tokens(unit[0])
        if current and current_tokens + unit_tokens > target_tokens:
            flush()
            overlap_units: list[tuple[str, int, int]] = []
            overlap_count = 0
            for u in reversed(current):
                t = count_tokens(u[0])
                # Always keep at least the most recent unit, even if it alone exceeds the
                # overlap budget — a single oversized paragraph should not zero out overlap.
                if overlap_units and overlap_count + t > overlap_tokens:
                    break
                overlap_units.insert(0, u)
                overlap_count += t
            current = overlap_units
            current_tokens = overlap_count
        current.append(unit)
        current_tokens += unit_tokens

    flush()
    return chunks
