# Build Plan — Multi-Hop Filings Analyst

**Status: M0–M6 complete.** See README.md for the finished write-up (architecture,
design decisions, cost analysis, limitations). This file is kept as the build record.

Derived from `SPEC.md`. Executed milestones M0–M6 in order, committing after each.
Tooling: `uv` for the environment, `pyproject.toml` pinned versions, `ruff` + `mypy` for lint/type checks.

## Ground rules carried from the spec
- ≤35 Python files, type hints everywhere, no silent failures, no hardcoded secrets.
- Numbers only from XBRL; narrative only from RAG. Verification node enforces this.
- Free tier only, no credit cards. Groq gets short prompts only (assert_short guard).
- Every LLM call cached (SQLite). Rerank budgeted. Trace sampling honored.
- Commit at every milestone with conventional commit messages.

## Milestones

### M0 — Scaffold ✅ done
- `pyproject.toml` (uv-managed), `.env.example`, `.gitignore`, `Makefile`.
- `config/corpus.yaml` (AAPL/MSFT/NVDA/JPM, CIK, FY2023/2024), `config/xbrl_tags.yaml`, `config/thresholds.yaml`.
- `src/filings_analyst/settings.py` — pydantic-settings typed env loader.
- `src/filings_analyst/obs/logging.py` — structlog JSON setup.
- `scripts/preflight.py` — live-check each of the 5 keys, pass/fail table.
- Done when: `make preflight` runs (fails gracefully with no keys; passes table format works once keys exist).

### M1 — Ingestion pipeline ✅ done — verified against real EDGAR/XBRL data (see README)
- `edgar/client.py` (polite HTTP, UA header, disk cache under `data/cache/`).
- `edgar/filings.py` (submissions.json → 10-K accession/doc URLs, pick 2 latest FY).
- `edgar/sections.py` (HTML → {item: text} via regex on Item 1A/7/7A/8, last-occurrence rule).
- `edgar/xbrl.py` (companyconcept lookups with candidate-tag fallback list).
- `ingest/chunker.py` (paragraph-aware, ~450 tok / 80 overlap, never cross item boundary).
- `ingest/pipeline.py` (end-to-end, token budget guard, `--dry-run`, `--only TICKER`, `--force`).
- `tests/fixtures/` saved HTML snippet + `tests/test_sections.py`, `tests/test_chunker.py`.
- Done when: `python -m filings_analyst.ingest.pipeline --dry-run` prints plan + token estimate; tests pass.

### M2 — Retrieval ✅ done — built against verified Pinecone SDK v10 API; real upsert/search needs a live PINECONE_API_KEY
- `retrieval/store.py` — Pinecone index mgmt (integrated embedding `create_index_for_model`), `upsert_records` batches of 96, `search()` with metadata filters, CLI entrypoint.
- `retrieval/rerank.py` — budgeted `pinecone-rerank-v0` wrapper, counter in `data/rerank_budget.json`, fallback to vector order when exhausted.
- Done when: `python -m filings_analyst.retrieval.store "supply chain risk" --ticker NVDA --fy 2024` returns chunks (requires real ingest+keys; otherwise unit-testable with mocked client).

### M3 — Agent graph ✅ done — graph compiles, full smoke test passes with stubbed LLM/retrieval; live run needs all 4 keys
- `llm.py` — `get_fast_llm/get_long_llm/get_judge_llm`, rate limiters, tenacity retries, SQLite LLM cache, `assert_short`.
- `agent/state.py` — TypedDict + pydantic models (SubQuery, Evidence, Fact, Contradiction, AgentState).
- `agent/tools.py` — `search_filings`, `get_xbrl_fact`.
- `agent/prompts.py` — versioned prompt constants for triage/plan/grade/refine/extract/compare/synthesize/verify-repair.
- `agent/nodes.py` — one function per node (triage, plan, retrieve, grade, refine, xbrl_lookup, extract, compare, synthesize, verify, refuse).
- `agent/graph.py` — StateGraph wiring, conditional edges, `SqliteSaver` checkpointer, mermaid export.
- `tests/test_verification.py`, `tests/test_graph_smoke.py` (stub LLM, no network/keys).
- Done when: graph compiles and smoke tests pass without keys; with keys, `python -m filings_analyst.agent.graph "<question>"` returns a cited answer.

### M4 — API + demo page ✅ done — server boots and /healthz, /, /readyz verified locally with no keys
- `api/schemas.py`, `api/main.py` (`/ask`, `/feedback`, `/healthz`, `/readyz`, `/`).
- `obs/tracing.py` — LangSmith run wrapper, sampling, run-id capture for `/feedback`.
- `static/index.html` — vanilla JS single page: question box, cited answer, expandable citations, thumbs up/down, trace link.
- Done when: `make run` serves the page; manual click-through works when keys are present.

### M5 — Eval harness + CI ✅ done — dataset hand-built from real verified data (see README); no live eval run possible without keys
- `evals/make_dataset.py`, `evals/dataset.jsonl` (30 items, human-reviewed — hand-author since no live corpus during build), `evals/evaluators.py`, `evals/run_eval.py`.
- `.github/workflows/ci.yml` (lint+mypy+unit tests, no keys) and `eval.yml` (smoke suite gated on `config/thresholds.yaml`, skips cleanly without secrets).
- `evals/results/` committed markdown table (from whatever run is actually executable — see Constraints below).
- Done when: `make eval SUITE=smoke` runs; CI workflow files are valid.

### M6 — README ✅ done
- Full README per §13: what it is, quickstart, architecture (mermaid), design decisions (with rejected alternatives), eval results, cost/quota table, limitations, what's next.
- Done when: a stranger could clone and run it from the README alone.

## Known constraint on this build
I have no API keys configured in this environment (no `.env` with real secrets). I will:
- Build every component to be fully functional against real APIs per the spec.
- Make tests and CI pass **without** keys (mocked/stubbed where the spec requires it — e.g. graph smoke test with stub LLM).
- Clearly flag in the README and final summary which steps (real ingest, real eval run, populated results table) require the user to supply their own keys and run `make ingest` / `make eval` themselves.
- Not fabricate eval results or trace URLs. If `evals/results/*.md` cannot be produced from a real run, I will say so explicitly rather than inventing numbers.

## File budget check-in
Final count: exactly 35 Python files (`find . -name "*.py" -not -path "./.venv/*"`).
Removed `tests/__init__.py` (not required for pytest discovery) when the eval harness
pushed the count to 36.
