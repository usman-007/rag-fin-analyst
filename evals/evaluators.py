"""Eval metrics. Retrieval metrics are pure functions (no LLM, no cost). Answer-quality
metrics use JUDGE_MODEL at temperature 0. `numeric_accuracy` and `refusal_correctness`
are deterministic — never judged — because a number either matches or it doesn't.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from filings_analyst.llm import get_judge_llm, structured_invoke

_NUMBER_STRIP_RE = re.compile(r"[^0-9.\-]")


def recall_at_k(gold_chunk_ids: list[str], retrieved_chunk_ids: list[str]) -> float:
    """Fraction of gold chunk ids present anywhere in the retrieved set."""
    if not gold_chunk_ids:
        return 1.0
    gold = set(gold_chunk_ids)
    retrieved = set(retrieved_chunk_ids)
    return len(gold & retrieved) / len(gold)


def precision_at_k(gold_chunk_ids: list[str], retrieved_chunk_ids: list[str]) -> float:
    """Fraction of retrieved chunks that are gold. A simplification: `gold_chunk_ids`
    names the chunk(s) a question was written against, not an exhaustive relevance
    judgment over the whole corpus, so this under-counts chunks that are genuinely
    relevant but weren't the ones a question happened to be authored from."""
    if not retrieved_chunk_ids:
        return 0.0
    gold = set(gold_chunk_ids)
    retrieved = set(retrieved_chunk_ids)
    return len(gold & retrieved) / len(retrieved)


def mrr(gold_chunk_ids: list[str], retrieved_chunk_ids_ordered: list[str]) -> float:
    """Reciprocal rank of the first gold hit in the ordered retrieved list, 0 if none."""
    gold = set(gold_chunk_ids)
    for i, chunk_id in enumerate(retrieved_chunk_ids_ordered):
        if chunk_id in gold:
            return 1.0 / (i + 1)
    return 0.0


def _normalize_number(value: str) -> str | None:
    cleaned = _NUMBER_STRIP_RE.sub("", value)
    if not cleaned or cleaned in ("-", "."):
        return None
    try:
        return str(float(cleaned))
    except ValueError:
        return None


def numeric_accuracy(gold_numbers: dict[str, str], fact_values: list[str]) -> float:
    """Fraction of `gold_numbers` whose value appears (after normalization) among the
    extracted facts' values. Deterministic — a number either matches or it doesn't."""
    if not gold_numbers:
        return 1.0
    normalized_facts = {_normalize_number(v) for v in fact_values if _normalize_number(v)}
    hits = 0
    for gold_value in gold_numbers.values():
        if _normalize_number(gold_value) in normalized_facts:
            hits += 1
    return hits / len(gold_numbers)


def refusal_correctness(expected_refusal: bool, actual_refusal: bool) -> float:
    return 1.0 if expected_refusal == actual_refusal else 0.0


# ---- LLM-as-judge metrics (JUDGE_MODEL, temperature 0) ----------------------------------


class _SentenceSupport(BaseModel):
    marker: str
    supported: bool


class _SupportBatch(BaseModel):
    results: list[_SentenceSupport]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MARKER_RE = re.compile(r"\[C\d+\]")


def _sentences_by_marker(answer: str) -> dict[str, str]:
    by_marker: dict[str, str] = {}
    for sentence in _SENTENCE_SPLIT_RE.split(answer):
        for marker in _MARKER_RE.findall(sentence):
            by_marker[marker] = sentence
    return by_marker


def citation_support(answer: str, citations: list[Any]) -> float:
    """For each [Cn] marker, ask the judge whether its cited snippet actually supports
    the sentence it's attached to. Score = supported / total. 1.0 if there are no
    citations to check (an empty answer trivially has nothing unsupported)."""
    sentences = _sentences_by_marker(answer)
    pairs = [
        (c.marker, sentences.get(c.marker, ""), c.text_snippet)
        for c in citations
        if c.marker in sentences
    ]
    if not pairs:
        return 1.0

    prompt_lines = [
        f"- marker={marker}\n  sentence: {sentence!r}\n  cited_snippet: {snippet!r}"
        for marker, sentence, snippet in pairs
    ]
    prompt = (
        "For each marker below, does `cited_snippet` actually support the claim made in "
        "`sentence`? Judge only against the snippet's content, not outside knowledge.\n\n"
        + "\n".join(prompt_lines)
    )
    result = structured_invoke(get_judge_llm(), _SupportBatch, prompt)
    supported = {r.marker: r.supported for r in result.results}
    if not supported:
        return 0.0
    return sum(1 for m in supported.values() if m) / len(supported)


class _CorrectnessScore(BaseModel):
    score: int  # 1-5


def answer_correctness(question: str, answer: str, reference_answer: str) -> float:
    """1-5 judge score vs. the reference answer, normalized to 0-1."""
    prompt = (
        "Score how well ANSWER correctly and completely addresses QUESTION, using "
        "REFERENCE_ANSWER as the ground truth. 5 = fully correct and complete, "
        "3 = partially correct or missing detail, 1 = incorrect or contradicts the "
        "reference. Respond with just the integer score.\n\n"
        f"QUESTION: {question}\n\nREFERENCE_ANSWER: {reference_answer}\n\nANSWER: {answer}"
    )
    result = structured_invoke(get_judge_llm(), _CorrectnessScore, prompt)
    return max(1, min(5, result.score)) / 5.0
