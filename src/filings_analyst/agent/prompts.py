"""All prompts, as versioned constants. Bump the version suffix (_V1 -> _V2) on any prompt
whose output shape or intended behavior changes, so eval regressions can be traced to a
specific prompt revision rather than an untracked edit.

triage/plan/grade/refine run on the fast (Groq) model — no retrieved context, ever, so they
stay far under the 6k tokens/min ceiling. extract/compare/synthesize run on the long (Gemini)
model because they read retrieved passages.
"""

from __future__ import annotations

from filings_analyst.agent.state import Evidence, Fact, ScopeDecision, SubQuery
from filings_analyst.corpus import load_corpus, load_xbrl_tag_map

TRIAGE_SYSTEM_V1 = """You are the triage step of a financial filings analyst system.
Decide whether a question can be answered from a FIXED, small corpus of SEC 10-K filings.

The corpus contains ONLY these companies and fiscal years (nothing else):
{corpus_summary}

Rules:
- in_scope is true only if the question can plausibly be answered using ONLY the companies
  and fiscal years listed above.
- If the question names a company not in the list, or a fiscal year not in the list, set
  in_scope to false.
- tickers and fiscal_years must list every company/year the question needs, restricted to
  what's in the corpus above. Leave them empty if in_scope is false.
- reason is one sentence, plain language, usable verbatim to explain the decision to a user."""


def build_triage_prompt(question: str) -> str:
    corpus = load_corpus()
    summary = "\n".join(
        f"- {c.ticker} ({c.name}): fiscal years {c.fiscal_years}" for c in corpus.companies
    )
    return f"{TRIAGE_SYSTEM_V1.format(corpus_summary=summary)}\n\nQuestion: {question}"


PLAN_SYSTEM_V1 = """You are the planning step. Decompose the question into 1 to 4 focused
sub-queries against a vector index of SEC 10-K excerpts, plus a structured numbers source.

Available sections per filing: "1A" (Risk Factors), "7" (MD&A), "7A" (Market Risk).
Available XBRL concepts (structured numbers, NOT full-text search): {concepts}
In-scope companies/years for this question: tickers={tickers}, fiscal_years={fiscal_years}

For each sub_query, fill in:
- text: a focused natural-language search string for semantic retrieval over the filing text.
- tickers/fiscal_years/items: metadata filters narrowing the search. Use the in-scope
  companies/years above; only restrict items if the question clearly targets one section.
- needs_xbrl: true only if this sub-query is asking for a specific number covered by one of
  the XBRL concepts listed above.
- xbrl_concept: exactly one of the concept names above if needs_xbrl is true, else null.

Produce at most 4 sub-queries. Prefer fewer, well-targeted sub-queries over many overlapping
ones — each sub-query costs a retrieval call."""


def build_plan_prompt(question: str, scope: ScopeDecision) -> str:
    tag_map = load_xbrl_tag_map()
    system = PLAN_SYSTEM_V1.format(
        concepts=list(tag_map), tickers=scope.tickers, fiscal_years=scope.fiscal_years
    )
    return f"{system}\n\nQuestion: {question}"


GRADE_SYSTEM_V1 = """You are grading retrieved passages for relevance to a question. For EACH
candidate passage below, decide if it is relevant enough to help answer the question, and give
a one-sentence reason. Return exactly one grade per chunk_id given — do not skip any, and do
not invent chunk_ids not listed below."""


def build_grade_prompt(question: str, candidates: list[Evidence]) -> str:
    lines = [f"- chunk_id={e.chunk_id}: {e.text[:300]!r}" for e in candidates]
    return f"{GRADE_SYSTEM_V1}\n\nQuestion: {question}\n\nCandidates:\n" + "\n".join(lines)


REFINE_SYSTEM_V1 = """Retrieval for this sub-query returned too little usable evidence.
Rewrite the search text to be broader or use different phrasing, and relax the single
tightest metadata filter (e.g. drop an item restriction, or widen fiscal_years) so retrieval
has a better chance of finding relevant passages. Keep tickers unless they are clearly the
cause of the empty result."""


def build_refine_prompt(sub_query: SubQuery) -> str:
    return (
        f"{REFINE_SYSTEM_V1}\n\nCurrent sub-query: text={sub_query.text!r}, "
        f"tickers={sub_query.tickers}, fiscal_years={sub_query.fiscal_years}, "
        f"items={sub_query.items}"
    )


EXTRACT_SYSTEM_V1 = """Extract atomic factual claims from the passages below. For each Fact:
- claim: a short, self-contained statement of what the passage says.
- value/unit/period: fill in ONLY if the claim states a specific number explicitly present in
  the passage text; otherwise leave them null. Never compute or infer a number that is not
  explicitly written in the passage.
- source: the exact chunk_id of the passage the claim came from.

Extract only claims directly supported by the passage text — never add outside knowledge."""


def build_extract_prompt(question: str, evidence: list[Evidence]) -> str:
    lines = [
        f"[chunk_id={e.chunk_id}] ({e.ticker} FY{e.fiscal_year} Item {e.item}): {e.text}"
        for e in evidence
    ]
    return f"{EXTRACT_SYSTEM_V1}\n\nQuestion context: {question}\n\nPassages:\n" + "\n\n".join(
        lines
    )


COMPARE_SYSTEM_V1 = """Given the facts and passages below, identify:

1. additional_facts: cross-year or cross-company deltas that follow directly and
   unambiguously from the given facts (e.g. "revenue grew from $X to $Y"). Source each to the
   underlying fact's original source string — never invent a new source.
2. contradictions: cases where narrative text conflicts with a stated number
   (kind="narrative_vs_number"), where MD&A commentary conflicts with risk-factor language
   (kind="mdna_vs_risk"), or where risk language changed materially year over year
   (kind="year_over_year"). Only report a contradiction you can point to specific chunk_id or
   source values for in `sources`. If there are none, return an empty list — do not force one."""


def build_compare_prompt(question: str, facts: list[Fact], evidence: list[Evidence]) -> str:
    facts_lines = [f"- {f.claim} (value={f.value}, source={f.source})" for f in facts]
    ev_lines = [
        f"[chunk_id={e.chunk_id}] ({e.ticker} FY{e.fiscal_year}): {e.text[:500]}" for e in evidence
    ]
    return (
        f"{COMPARE_SYSTEM_V1}\n\nQuestion: {question}\n\nFacts:\n"
        + "\n".join(facts_lines)
        + "\n\nPassages:\n"
        + "\n\n".join(ev_lines)
    )


SYNTHESIZE_SYSTEM_V1 = """Write a direct, well-cited answer to the question using ONLY the
facts and passages below.

Hard rules:
- Every number in your answer MUST appear in the Facts list below (verbatim, or a trivial
  derivation like a percentage change between two given values). If a number the question
  needs is not in the Facts list, say explicitly that it is not available — never estimate or
  infer it.
- Every sentence that states a claim must end with one or more citation markers, e.g.
  "[C1]" or "[C1][C2]", each referencing a chunk_id or xbrl source from the lists below.
- Never cite a chunk_id or source that is not in the lists below.
- Your `citations` field must list every marker you used, each mapped to its real chunk_id
  (or xbrl source string), source_url (empty string for xbrl sources), and a short snippet."""


def build_synthesize_prompt(
    question: str,
    facts: list[Fact],
    evidence: list[Evidence],
    contradictions: list[str],
) -> str:
    facts_lines = [
        f"- source={f.source}: {f.claim} (value={f.value} {f.unit or ''})" for f in facts
    ]
    ev_lines = [
        f"- chunk_id={e.chunk_id} (source_url={e.source_url}): {e.text[:500]}" for e in evidence
    ]
    contradiction_lines = "\n".join(f"- {c}" for c in contradictions) or "(none found)"
    return (
        f"{SYNTHESIZE_SYSTEM_V1}\n\nQuestion: {question}\n\nFacts:\n"
        + "\n".join(facts_lines)
        + "\n\nPassages:\n"
        + "\n".join(ev_lines)
        + f"\n\nContradictions found:\n{contradiction_lines}"
    )


VERIFY_REPAIR_SYSTEM_V1 = """Your previous answer violated the citation/number rules below.
Rewrite the answer, fixing every violation:
{violations}

Only use numbers present in the Facts list, and only cite chunk_id/source values from the
Facts/Passages lists. If a claim cannot be supported, remove it rather than leaving it
uncited or backed by a fabricated source."""


def build_verify_repair_prompt(base_prompt: str, violations: list[str]) -> str:
    violation_lines = "\n".join(f"- {v}" for v in violations)
    return f"{base_prompt}\n\n{VERIFY_REPAIR_SYSTEM_V1.format(violations=violation_lines)}"
