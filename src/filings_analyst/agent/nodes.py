"""One function per graph node (see graph.py for wiring). Each returns a partial state
update dict, per LangGraph convention — never the full state.
"""

from __future__ import annotations

import re
from typing import Any

from langgraph.types import Send

from filings_analyst.agent.prompts import (
    build_compare_prompt,
    build_extract_prompt,
    build_grade_prompt,
    build_plan_prompt,
    build_refine_prompt,
    build_synthesize_prompt,
    build_triage_prompt,
    build_verify_repair_prompt,
)
from filings_analyst.agent.state import (
    AgentState,
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
from filings_analyst.agent.tools import get_xbrl_fact, search_filings
from filings_analyst.corpus import load_corpus
from filings_analyst.llm import assert_short, get_fast_llm, get_long_llm, structured_invoke
from filings_analyst.obs.logging import get_logger
from filings_analyst.retrieval.store import FilingsStore
from filings_analyst.settings import get_settings

logger = get_logger(__name__)

MIN_RELEVANT_EVIDENCE = 2
MAX_SYNTH_ATTEMPTS = 2

CITATION_MARKER_RE = re.compile(r"\[C(\d+)\]")
MARKER_DIGITS_RE = re.compile(r"(\d+)")
# The decimal group only matches when a digit actually follows the dot — otherwise a
# number at the end of a sentence (e.g. "...$100.") would swallow the sentence's own
# period into the match and then never equal the fact value "100" it should match.
NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")
YEAR_RE = re.compile(r"^(19|20)\d{2}$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _normalize_marker(raw: str) -> str | None:
    """The synthesis model is told to reuse its inline `[C1]` markers verbatim in the
    `citations` field, but doesn't reliably do so (observed live: plain `C1`, bare `1`).
    Extract the digits and re-canonicalize rather than requiring an exact string match
    the model doesn't consistently produce."""
    match = MARKER_DIGITS_RE.search(raw)
    return f"[C{match.group(1)}]" if match else None


def _numbers_in(text: str) -> set[str]:
    """Numeric tokens that look like financial figures needing a `facts` citation.

    Excludes bare small integers and bare 4-digit years (e.g. "FY2024" -> "2024",
    "Item 7" -> "7") unless they carry a $, comma, decimal, or % marker — those are
    incidental references, not claims a synthesized answer must trace to a fact.
    """
    result: set[str] = set()
    for raw in NUMBER_RE.findall(text):
        # str.strip only trims the ends; commas are internal thousands separators, so
        # they need removing throughout, not just at the edges.
        cleaned = raw.translate(str.maketrans("", "", "$,%"))
        if not cleaned:
            continue
        has_marker = raw.startswith("$") or "," in raw or raw.endswith("%") or "." in raw
        if not has_marker and (YEAR_RE.match(cleaned) or len(cleaned) < 3):
            continue
        result.add(cleaned)
    return result


# ---- triage ---------------------------------------------------------------------------


def triage(state: AgentState) -> dict[str, Any]:
    prompt = build_triage_prompt(state["question"])
    assert_short(prompt)
    scope = structured_invoke(get_fast_llm(), ScopeDecision, prompt)
    logger.info(
        "node.triage",
        in_scope=scope.in_scope,
        tickers=scope.tickers,
        fiscal_years=scope.fiscal_years,
    )
    return {"scope": scope}


def route_after_triage(state: AgentState) -> str:
    scope = state.get("scope")
    return "plan" if scope is not None and scope.in_scope else "refuse"


# ---- plan + retrieve (Send fan-out) ----------------------------------------------------


def plan(state: AgentState) -> dict[str, Any]:
    scope = state["scope"]
    assert scope is not None
    prompt = build_plan_prompt(state["question"], scope)
    assert_short(prompt)
    result = structured_invoke(get_fast_llm(), PlanOutput, prompt)
    sub_queries = (result.sub_queries or [])[:4] or [
        SubQuery(text=state["question"], tickers=scope.tickers, fiscal_years=scope.fiscal_years)
    ]
    logger.info("node.plan", n_sub_queries=len(sub_queries))
    return {
        "plan": sub_queries,
        "hop": 0,
        "pending_retrieval_indices": list(range(len(sub_queries))),
    }


_STORE: FilingsStore | None = None


def _store() -> FilingsStore:
    global _STORE
    if _STORE is None:
        _STORE = FilingsStore()
    return _STORE


def dispatch_retrieval(state: AgentState) -> list[Send]:
    """Conditional edge used from both `plan` and `refine` — fans out over whichever
    sub-query indices are pending, at up to `max_concurrency` (set at invoke time)."""
    indices = state.get("pending_retrieval_indices") or []
    return [Send("retrieve_one", {"sub_query": state["plan"][i], "index": i}) for i in indices]


def retrieve_one(payload: dict[str, Any]) -> dict[str, Any]:
    sub_query: SubQuery = payload["sub_query"]
    index: int = payload["index"]
    evidence = search_filings(sub_query, _store())
    for e in evidence:
        e.sub_query_index = index
    logger.info("node.retrieve_one", index=index, n_evidence=len(evidence))
    return {"evidence": evidence}


# ---- grade + refine ---------------------------------------------------------------------


def _relevant_counts(evidence: list[Evidence]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for e in evidence:
        if e.relevant:
            counts[e.sub_query_index] = counts.get(e.sub_query_index, 0) + 1
    return counts


def grade(state: AgentState) -> dict[str, Any]:
    candidates = [e for e in state["evidence"] if e.relevant is None]
    if not candidates:
        return {}
    prompt = build_grade_prompt(state["question"], candidates)
    assert_short(prompt)
    result = structured_invoke(get_fast_llm(), GradeBatch, prompt)
    grades_by_id = {g.chunk_id: g for g in result.grades or []}
    updated = [
        e.model_copy(update={"relevant": g.relevant, "reason": g.reason})
        for e in candidates
        if (g := grades_by_id.get(e.chunk_id)) is not None
    ]
    logger.info("node.grade", n_candidates=len(candidates), n_graded=len(updated))
    return {"evidence": updated}


def route_after_grade(state: AgentState) -> str:
    settings = get_settings()
    counts = _relevant_counts(state["evidence"])
    thin = [i for i in range(len(state["plan"])) if counts.get(i, 0) < MIN_RELEVANT_EVIDENCE]
    if thin and state["hop"] < settings.max_hops:
        return "refine"
    return "xbrl_lookup"


def refine(state: AgentState) -> dict[str, Any]:
    counts = _relevant_counts(state["evidence"])
    thin_indices = [
        i for i in range(len(state["plan"])) if counts.get(i, 0) < MIN_RELEVANT_EVIDENCE
    ]
    updated_plan = list(state["plan"])
    for i in thin_indices:
        original = updated_plan[i]
        prompt = build_refine_prompt(original)
        assert_short(prompt)
        refined = structured_invoke(get_fast_llm(), SubQuery, prompt)
        # build_refine_prompt only asks the model to adjust text/tickers/fiscal_years/items —
        # needs_xbrl/xbrl_concept are out of scope for refine and must survive unchanged, not
        # get silently defaulted away by a schema the model fills in from scratch each time.
        updated_plan[i] = refined.model_copy(
            update={"needs_xbrl": original.needs_xbrl, "xbrl_concept": original.xbrl_concept}
        )
    logger.info("node.refine", thin_indices=thin_indices, hop=state["hop"] + 1)
    return {
        "plan": updated_plan,
        "hop": state["hop"] + 1,
        "pending_retrieval_indices": thin_indices,
    }


# ---- xbrl_lookup (deterministic, no LLM) -------------------------------------------------


def xbrl_lookup(state: AgentState) -> dict[str, Any]:
    facts: list[Fact] = []
    for sub_query in state["plan"]:
        facts.extend(get_xbrl_fact(sub_query))
    logger.info("node.xbrl_lookup", n_facts=len(facts))
    return {"xbrl_facts": facts}


# ---- extract + compare (long model) ------------------------------------------------------


def extract(state: AgentState) -> dict[str, Any]:
    relevant_evidence = [e for e in state["evidence"] if e.relevant]
    if not relevant_evidence:
        return {"facts": list(state["xbrl_facts"])}
    prompt = build_extract_prompt(state["question"], relevant_evidence)
    result = structured_invoke(get_long_llm(), FactBatch, prompt)
    facts = list(state["xbrl_facts"]) + (result.facts or [])
    logger.info("node.extract", n_facts=len(facts))
    return {"facts": facts}


def compare(state: AgentState) -> dict[str, Any]:
    relevant_evidence = [e for e in state["evidence"] if e.relevant]
    if not state["facts"] and not relevant_evidence:
        return {"contradictions": []}
    prompt = build_compare_prompt(state["question"], state["facts"], relevant_evidence)
    result = structured_invoke(get_long_llm(), CompareOutput, prompt)
    additional_facts = result.additional_facts or []
    contradictions = result.contradictions or []
    facts = state["facts"] + additional_facts
    logger.info(
        "node.compare",
        n_additional_facts=len(additional_facts),
        n_contradictions=len(contradictions),
    )
    return {"facts": facts, "contradictions": contradictions}


# ---- synthesize (long model) --------------------------------------------------------------


def synthesize(state: AgentState) -> dict[str, Any]:
    relevant_evidence = [e for e in state["evidence"] if e.relevant]
    contradiction_strs = [c.description for c in state["contradictions"]]
    prompt = build_synthesize_prompt(
        state["question"], state["facts"], relevant_evidence, contradiction_strs
    )
    if state["violations"]:
        prompt = build_verify_repair_prompt(prompt, state["violations"])
    result = structured_invoke(get_long_llm(), SynthesisOutput, prompt)
    citations = result.citations or []
    logger.info("node.synthesize", n_citations=len(citations), attempt=state["synth_attempts"] + 1)
    return {
        "draft": result.answer,
        "citations": citations,
        "synth_attempts": state["synth_attempts"] + 1,
    }


# ---- verify (deterministic rules; no LLM unless routed back to synthesize) --------------


def _find_violations(
    draft: str,
    citations: list[Citation],
    evidence_ids: set[str],
    fact_sources: set[str],
    fact_numbers: set[str],
) -> list[str]:
    violations: list[str] = []

    used_marker_strs = {f"[C{m}]" for m in CITATION_MARKER_RE.findall(draft)}
    cited_marker_strs = {m for c in citations if (m := _normalize_marker(c.marker)) is not None}
    missing = used_marker_strs - cited_marker_strs
    if missing:
        violations.append(
            f"These citation markers appear in the answer but are not in "
            f"`citations`: {sorted(missing)}"
        )

    for c in citations:
        is_xbrl = c.chunk_id.startswith("xbrl:")
        if is_xbrl and c.chunk_id not in fact_sources:
            violations.append(f"Citation {c.marker} references unknown xbrl source '{c.chunk_id}'")
        elif not is_xbrl and c.chunk_id not in evidence_ids:
            violations.append(
                f"Citation {c.marker} references chunk_id '{c.chunk_id}' not present in evidence"
            )

    unsupported_numbers = {n for n in _numbers_in(draft) if n not in fact_numbers}
    if unsupported_numbers:
        violations.append(
            f"These numbers appear in the answer but not in `facts`: {sorted(unsupported_numbers)}"
        )

    return violations


def _strip_unsupported_sentences(
    draft: str, fact_numbers: set[str], valid_markers: set[str]
) -> tuple[str, list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    for sentence in SENTENCE_SPLIT_RE.split(draft):
        marker_strs = {f"[C{m}]" for m in CITATION_MARKER_RE.findall(sentence)}
        numbers = _numbers_in(sentence)
        bad_marker = bool(marker_strs - valid_markers)
        bad_number = bool(numbers - fact_numbers)
        if bad_marker or bad_number:
            removed.append(sentence)
        else:
            kept.append(sentence)
    return " ".join(kept), removed


def verify(state: AgentState) -> dict[str, Any]:
    draft = state["draft"] or ""
    citations = state["citations"]
    evidence_ids = {e.chunk_id for e in state["evidence"]}
    fact_sources = {f.source for f in state["facts"]}
    fact_numbers = {n for f in state["facts"] if f.value for n in _numbers_in(f.value)}

    violations = _find_violations(draft, citations, evidence_ids, fact_sources, fact_numbers)

    if violations and state["synth_attempts"] >= MAX_SYNTH_ATTEMPTS:
        valid_sources = evidence_ids | fact_sources
        valid_markers = {
            m
            for c in citations
            if c.chunk_id in valid_sources and (m := _normalize_marker(c.marker)) is not None
        }
        cleaned, removed = _strip_unsupported_sentences(draft, fact_numbers, valid_markers)
        errors = list(state["errors"])
        if removed:
            errors.append(
                f"Repair failed after {state['synth_attempts']} attempts; removed "
                f"{len(removed)} unsupported sentence(s) from the answer: {removed}"
            )
        kept_citations = [
            c
            for c in citations
            if (m := _normalize_marker(c.marker)) is not None
            and m in valid_markers
            and m in cleaned
        ]
        logger.warning("node.verify.stripped", n_removed=len(removed))
        return {"draft": cleaned, "citations": kept_citations, "violations": [], "errors": errors}

    logger.info("node.verify", n_violations=len(violations))
    return {"violations": violations}


def route_after_verify(state: AgentState) -> str:
    if state["violations"] and state["synth_attempts"] < MAX_SYNTH_ATTEMPTS:
        return "synthesize"
    return "end"


# ---- refuse (deterministic, no LLM) -------------------------------------------------------


def refuse(state: AgentState) -> dict[str, Any]:
    corpus = load_corpus()
    available = "; ".join(
        f"{c.ticker} (FY{'/'.join(str(y) for y in c.fiscal_years)})" for c in corpus.companies
    )
    scope = state.get("scope")
    reason = scope.reason if scope else "This question falls outside the fixed corpus."
    draft = (
        f"I can't answer this from the available corpus. {reason} "
        f"This system only covers 10-K filings (Items 1A, 7, 7A) for: {available}."
    )
    logger.info("node.refuse", reason=reason)
    return {"draft": draft, "citations": [], "facts": [], "contradictions": []}
