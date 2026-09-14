# SPEC — Multi-Hop Filings Analyst

A RAG + agent system that answers comparative questions over SEC filings with span-level
citations, XBRL-grounded numbers, and an eval-gated CI pipeline.

**Audience:** Claude Code. This is the complete build brief. Read it fully before writing code.

---

## 0. Rules of engagement

1. **Budget: ~2.5 hours of build time.** Work through the milestones in §14 in order. Each
   milestone ends with a runnable command and a git commit. Do not skip ahead.
2. **Verify, don't assume.** Model IDs, SDK method names, and free-tier limits change. Before
   using an API, check the current docs (`docs.pinecone.io`, `ai.google.dev`, `console.groq.com/docs`,
   `docs.langchain.com`). If a model ID in this spec no longer exists, use the current equivalent
   and note the substitution in the README.
3. **Never hardcode secrets.** Everything comes from `.env` via a typed settings object.
4. **Fail fast and loudly.** No silent `except: pass`. No mock fallbacks that make a broken
   pipeline look like it works.
5. **Stop and ask** if a hard constraint in §2 cannot be met. Do not work around it by
   introducing a paid service.
6. **Do not exceed ~35 Python files.** If the design starts sprawling, collapse it.
7. Type hints everywhere. `ruff` + `mypy` clean on the committed code.

---

## 1. What it does

**Input:** a natural-language question about US public company filings, typically comparative.

Examples the system must handle:

- *"How did NVIDIA's stated supply-chain risk language change between FY2023 and FY2024?"* (multi-hop, cross-year, narrative)
- *"Compare Apple's and Microsoft's FY2024 revenue, and say whether either flagged FX as a material risk."* (multi-entity, numeric + narrative)
- *"Does JPMorgan's MD&A commentary on credit quality contradict anything in its risk factors?"* (intra-document contradiction)
- *"What was Tesla's FY2024 free cash flow?"* → **must refuse**: Tesla is not in the corpus.

**Output:** a JSON payload with a written answer, an ordered citation list (each pointing to a
specific chunk and its filing URL), the structured facts extracted, any contradictions found, and
a LangSmith trace URL.

**The point of the project** (state this in the README): LLMs hallucinate numbers. This system
never lets the LLM produce a figure from prose — figures come from the XBRL structured API and
narrative comes from RAG, and a verification node rejects any answer containing a number that
isn't traceable to one of those two sources.

### Non-goals

No portfolio analysis, no investment recommendations, no real-time market data, no user accounts,
no multi-tenancy, no chat history UI beyond a thread id.

---

## 2. Hard constraints

These are non-negotiable. The whole project must run on free tiers with **no credit card on any
service**.

| Service | Free allowance (verify at build time) | Design implication |
|---|---|---|
| Google AI Studio (Gemini) | Flash models only; ~10–15 RPM, ~250–1500 RPD; very high TPM | Primary model for long-context work |
| Groq | 30 RPM but only **~6,000 tokens/min** | Short prompts ONLY. Never send retrieved context here. |
| Pinecone Starter | 2 GB, up to 5 indexes, aws/us-east-1 only, **5M embedding tokens/month**, 500 rerank requests/month | Ingestion must be budgeted; index pauses after 3 weeks idle |
| LangSmith Developer | 1 seat, **5,000 traces/month**, 14-day retention | Trace sampling required; eval runs go to a separate project |
| SEC EDGAR | No key. Requires `User-Agent` header with name + email, max 10 req/sec | Polite client with 0.15s delay; cache every response to disk |

**Derived rules:**

- **Embedding budget: ≤1.5M tokens for a full corpus rebuild.** The ingest script must estimate
  token count before upserting, print it, and refuse to proceed above `MAX_EMBED_TOKENS` unless
  `--force` is passed.
- **A single question must cost ≤12 LLM calls.** Grading, extraction, and comparison are batched
  into single calls with structured output — never one call per chunk.
- **Every LLM call goes through a cache.** Repeat eval runs must not re-consume quota.
- **The repo must be clonable and inspectable without keys.** Tests that need keys skip cleanly.

---

## 3. Corpus

Fixed, small, committed metadata; raw filings downloaded on demand and cached.

**Companies (CIK / ticker), FY2023 and FY2024 10-K:**

| Ticker | CIK | Why it's in the set |
|---|---|---|
| AAPL | 0000320193 | Clean baseline, heavy FX and supply-chain risk language |
| MSFT | 0000789019 | Cross-company comparison target |
| NVDA | 0001045810 | Year-over-year risk language changes dramatically |
| JPM | 0000019617 | Financial-sector language; credit risk vs MD&A tension |

8 filings total. Put this table in `config/corpus.yaml` with CIK, ticker, company name, and the
two fiscal years — do not hardcode it in Python.

**Only these sections are ingested:**

- Item 1A — Risk Factors
- Item 7 — MD&A
- Item 7A — Quantitative and Qualitative Disclosures About Market Risk

Ingesting the full 10-K would blow the embedding budget and add boilerplate that hurts retrieval
precision. Document this trade-off in the README.

### Acquisition

1. `GET https://data.sec.gov/submissions/CIK{cik:010d}.json` → find 10-K accession numbers and
   filing dates; pick the two most recent 10-Ks covering FY2023/FY2024.
2. Primary document URL:
   `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodashes}/{primary_document}`
3. Parse HTML with `selectolax` or `beautifulsoup4` + `lxml`. Strip tables of contents, page
   headers, and XBRL inline tags.
4. Section splitting: regex on `Item\s+1A\.?`, `Item\s+7\.?`, `Item\s+7A\.?`, `Item\s+8`
   boundaries, case-insensitive, taking the **last** occurrence of each (the first is usually the
   TOC entry). This is the fiddliest part of the build — write a unit test with a saved fixture.
5. Cache raw HTML and extracted sections under `data/cache/` (gitignored). A second ingest run
   must hit disk, not the SEC.

**Numbers do not come from this pipeline.** They come from
`https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json`, filtered to
`form == "10-K"` and the requested fiscal year. Support a small tag map in
`config/xbrl_tags.yaml`: revenue, net income, operating income, total assets, cash and equivalents,
R&D expense. Handle the reality that tags differ across companies — map several candidate tags per
concept and take the first that resolves.

---

## 4. Repo layout

```
filings-analyst/
├── README.md
├── .env.example
├── pyproject.toml              # uv or pip-tools; pin versions
├── Makefile                    # ingest, run, eval, preflight, test, lint
├── config/
│   ├── corpus.yaml
│   ├── xbrl_tags.yaml
│   └── thresholds.yaml         # eval gates
├── src/filings_analyst/
│   ├── settings.py             # pydantic-settings, all env vars, validated
│   ├── llm.py                  # model factory, routing, rate limiter, cache, retries
│   ├── edgar/
│   │   ├── client.py           # polite HTTP client, UA header, disk cache
│   │   ├── filings.py          # submissions → 10-K doc URLs
│   │   ├── sections.py         # HTML → {item: text}
│   │   └── xbrl.py             # companyconcept lookups
│   ├── ingest/
│   │   ├── chunker.py
│   │   └── pipeline.py         # end-to-end ingest + budget guard
│   ├── retrieval/
│   │   ├── store.py            # Pinecone index mgmt, upsert_records, search
│   │   └── rerank.py           # budgeted reranking
│   ├── agent/
│   │   ├── state.py            # TypedDict state + pydantic models
│   │   ├── nodes.py            # one function per node
│   │   ├── prompts.py          # all prompts, versioned constants
│   │   ├── graph.py            # StateGraph wiring + checkpointer
│   │   └── tools.py            # search_filings, get_xbrl_fact
│   ├── api/
│   │   ├── main.py             # FastAPI app
│   │   └── schemas.py
│   ├── obs/
│   │   ├── tracing.py          # LangSmith config, sampling, run metadata
│   │   └── logging.py          # structlog JSON logs
│   └── static/index.html       # minimal demo page, vanilla JS
├── evals/
│   ├── dataset.jsonl
│   ├── make_dataset.py         # generates candidate questions from ingested chunks
│   ├── evaluators.py           # retrieval + judge evaluators
│   ├── run_eval.py             # LangSmith evaluate() runner
│   └── results/                # committed markdown result tables
├── scripts/
│   └── preflight.py            # validates every env var with one cheap live call
├── tests/
│   ├── fixtures/               # saved 10-K HTML snippet
│   ├── test_sections.py
│   ├── test_chunker.py
│   ├── test_verification.py
│   └── test_graph_smoke.py     # graph compiles, routes correctly with stub LLM
└── .github/workflows/
    ├── ci.yml                  # lint + unit tests, no keys needed
    └── eval.yml                # smoke eval suite, gated on thresholds
```

---

## 5. `.env.example`

Write exactly this file (with these comments) — the user fills it in:

```bash
# ---- LLM providers -------------------------------------------------------
# Google AI Studio (free, no card): https://aistudio.google.com/apikey
# Used for long-context work: extraction, synthesis, LLM-as-judge.
GOOGLE_API_KEY=

# Groq (free, no card): https://console.groq.com/keys
# Used ONLY for short prompts: triage, planning, grading, query rewriting.
# 6k tokens/min ceiling — never send retrieved context to this provider.
GROQ_API_KEY=

# ---- Vector store --------------------------------------------------------
# Pinecone Starter (free): https://app.pinecone.io
PINECONE_API_KEY=
PINECONE_INDEX=filings-analyst
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=filings
PINECONE_EMBED_MODEL=llama-text-embed-v2
PINECONE_RERANK_MODEL=pinecone-rerank-v0

# ---- Observability -------------------------------------------------------
# LangSmith Developer plan (free, 5k traces/month): https://smith.langchain.com
# Do NOT add a card: the personal org is hard-capped at 5k traces, which is
# the safety net that keeps this project at zero cost.
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=filings-analyst
LANGSMITH_EVAL_PROJECT=filings-analyst-eval
# 1.0 = trace everything. Lower this if you approach the monthly cap.
LANGSMITH_SAMPLE_RATE=1.0

# ---- SEC EDGAR -----------------------------------------------------------
# No API key. The SEC requires a User-Agent identifying you, or returns 403.
# Format: "Name email@example.com"
SEC_USER_AGENT=

# ---- Models (override if IDs change) -------------------------------------
FAST_MODEL=llama-3.3-70b-versatile     # Groq
LONG_MODEL=gemini-2.5-flash            # Google
JUDGE_MODEL=gemini-2.5-flash           # Google

# ---- Budgets and behaviour ----------------------------------------------
MAX_EMBED_TOKENS=1500000
MAX_HOPS=2
TOP_K=8
RERANK_TOP_N=4
RERANK_MONTHLY_BUDGET=400
LLM_CACHE_PATH=data/llm_cache.sqlite
CHECKPOINT_DB=data/checkpoints.sqlite
LOG_LEVEL=INFO
```

`scripts/preflight.py` must check each of these with one minimal live call (Gemini: 1-token
completion; Groq: same; Pinecone: `describe_index`; LangSmith: list projects; SEC: fetch one
submissions JSON) and print a pass/fail table. `make preflight` is the first thing the user runs.

---

## 6. LLM layer (`llm.py`)

One factory, two roles, enforced by type:

```python
def get_fast_llm() -> BaseChatModel:   # Groq, short prompts, structured output
def get_long_llm() -> BaseChatModel:   # Gemini Flash, retrieved context allowed
def get_judge_llm() -> BaseChatModel:  # Gemini Flash, temperature=0, eval only
```

All three must have:

- **Rate limiting** via `langchain_core.rate_limiters.InMemoryRateLimiter`, configured below the
  provider's free limit (Groq: 0.4 req/s; Gemini: 0.2 req/s).
- **Retries** on 429/5xx with exponential backoff + jitter (`tenacity`), max 4 attempts.
- **Caching**: set a global SQLite LLM cache at startup (`langchain_community.cache.SQLiteCache`)
  pointed at `LLM_CACHE_PATH`. Temperature 0 everywhere so caching is meaningful.
- `max_retries=0` on the provider client itself so retry logic isn't duplicated.

**Guard rail:** `get_fast_llm()` callers must pass prompts under a token ceiling. Add a helper
`assert_short(prompt, limit=2500)` that raises if exceeded. This is what stops someone (including
you) from casually routing retrieved chunks to Groq and getting mysterious 429s.

---

## 7. Ingestion

`make ingest` → `python -m filings_analyst.ingest.pipeline`

Steps:

1. Load `config/corpus.yaml`.
2. For each (company, fiscal_year): resolve the 10-K, download, extract the three items.
3. Chunk: split within each item on paragraph boundaries, target **~450 tokens with 80 token
   overlap** (llama-text-embed-v2 takes up to 2048 tokens but performs best at 400–500). Never
   merge across item boundaries.
4. Estimate total embedding tokens; print a summary table (filing → sections → chunks → tokens).
   Abort if over `MAX_EMBED_TOKENS` without `--force`.
5. Create the index if absent using **integrated embedding** (`create_index_for_model` with
   `field_map={"text": "chunk_text"}`) so Pinecone embeds server-side and no embedding model runs
   locally.
6. Upsert with `upsert_records` in batches of 96, into namespace `PINECONE_NAMESPACE`.

**Record schema** (id + metadata — these fields are the backbone of multi-hop filtering, get them
right):

```
id            = "{ticker}-{fy}-{item}-{seq}"      e.g. "NVDA-2024-1A-0037"
chunk_text    = str                                # the embedded field
ticker        = "NVDA"
cik           = "0001045810"
company       = "NVIDIA Corporation"
fiscal_year   = 2024                               # int, filterable
form_type     = "10-K"
item          = "1A" | "7" | "7A"
section_title = "Risk Factors"
accession     = "0001045810-25-000023"
source_url    = "https://www.sec.gov/Archives/..."
char_start    = int
char_end      = int
seq           = int
```

`--dry-run` flag prints the plan without touching Pinecone. `--only TICKER` for iteration.

**Decision to document:** one index, one namespace, metadata filters — *not* a namespace per
company. Pinecone queries a single namespace at a time, and this system's core query pattern is
cross-company and cross-year. Namespacing by ticker would make the primary use case require N
queries and a manual merge.

---

## 8. Retrieval

`store.search(query, filters, top_k)` where filters is a Pinecone metadata filter built from the
sub-query's constraints, e.g. `{"ticker": {"$in": ["AAPL","MSFT"]}, "fiscal_year": {"$eq": 2024}, "item": {"$eq": "1A"}}`.

Reranking: `pinecone-rerank-v0` on the top-k down to `RERANK_TOP_N`. **Budgeted** — the free tier
allows 500 rerank requests/month. Keep a counter in `data/rerank_budget.json` (month-keyed);
when the budget is exhausted, log a warning and fall back to raw vector order rather than erroring.
Reranking is disabled by default during eval runs to conserve it.

---

## 9. The agent graph

LangGraph `StateGraph`, compiled with `SqliteSaver(CHECKPOINT_DB)`, `thread_id` from the request.

### State

```python
class SubQuery(BaseModel):
    text: str
    tickers: list[str] = []
    fiscal_years: list[int] = []
    items: list[str] = []          # "1A" | "7" | "7A"
    needs_xbrl: bool = False
    xbrl_concept: str | None = None

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

class Fact(BaseModel):
    claim: str
    value: str | None = None       # numeric facts only
    unit: str | None = None
    period: str | None = None
    source: str                    # chunk_id OR "xbrl:{tag}:{fy}"

class Contradiction(BaseModel):
    kind: Literal["narrative_vs_number", "mdna_vs_risk", "year_over_year"]
    description: str
    sources: list[str]

class AgentState(TypedDict):
    question: str
    thread_id: str
    scope: ScopeDecision | None     # in/out of corpus
    plan: list[SubQuery]
    hop: int
    evidence: list[Evidence]
    xbrl_facts: list[Fact]
    facts: list[Fact]
    contradictions: list[Contradiction]
    draft: str | None
    citations: list[Citation]
    violations: list[str]
    synth_attempts: int
    errors: list[str]
```

### Nodes

| Node | Model | Job |
|---|---|---|
| `triage` | fast | Classify the question; resolve mentioned companies/years against `corpus.yaml`. Anything outside the corpus → route to `refuse`. |
| `plan` | fast | Decompose into 1–4 `SubQuery` objects with metadata filters and `needs_xbrl` flags. Structured output. |
| `retrieve` | none | Fan out over sub-queries (LangGraph `Send`, max concurrency 3). Pinecone search + optional rerank. Appends `Evidence`. |
| `grade` | fast | **One batched call** for all candidate chunks: input is `chunk_id + first 300 chars` of each, output is a JSON array of `{chunk_id, relevant, reason}`. Sets `relevant` on evidence. |
| `refine` | fast | Only reached when usable evidence is thin. Rewrite sub-query text and relax the tightest filter. Increments `hop`. |
| `xbrl_lookup` | none | Deterministic. For each sub-query with `needs_xbrl`, call companyconcept and append `Fact(source="xbrl:...")`. No LLM involved. |
| `extract` | long | From relevant evidence only, extract atomic `Fact`s, each tagged with its `chunk_id`. |
| `compare` | long | Cross-year / cross-company deltas, plus contradiction detection: narrative claims vs XBRL numbers, MD&A tone vs risk factors, YoY language shifts. |
| `synthesize` | long | Write the answer with inline markers `[C1]`, `[C2]`… mapped to a citation list. Explicitly instructed: every number must come from `facts`; if a needed number is absent, say so rather than infer. |
| `verify` | none (rules) + long (repair) | Deterministic checks: every `[Cn]` resolves; every numeric token in the answer appears in `facts`/`xbrl_facts`; no cited chunk_id outside evidence. Violations → back to `synthesize` with the violation list (max 1 repair). Still failing → strip unsupported sentences and flag them in the response. |
| `refuse` | none | Template answer naming what *is* in the corpus. |

### Edges

```
START → triage
triage → refuse            (out of scope)
triage → plan              (in scope)
plan → retrieve → grade
grade → refine             (relevant evidence < 2 per sub-query AND hop < MAX_HOPS)
refine → retrieve
grade → xbrl_lookup        (otherwise)
xbrl_lookup → extract → compare → synthesize → verify
verify → synthesize        (violations AND synth_attempts < 2)
verify → END
refuse → END
```

The `grade → refine → retrieve` cycle is the reason this is a graph and not a chain. Say so in
the README, and include the rendered graph (`graph.get_graph().draw_mermaid()`) in it.

---

## 10. API

FastAPI, `uvicorn`, no auth.

- `POST /ask` → `{question, thread_id?}`; returns `{answer, citations[], facts[], contradictions[], refused, trace_url, run_id, latency_ms, llm_calls}`.
- `POST /feedback` → `{run_id, score: 0|1, comment?}` → writes LangSmith feedback via
  `langsmith.Client().create_feedback`. This closes the observability loop and is worth 30 lines.
- `GET /healthz` → liveness. `GET /readyz` → checks Pinecone reachable + index has vectors.
- `GET /` → `static/index.html`: a single page, vanilla JS, no framework. Input box, answer with
  clickable citations that expand to show the chunk text and link to the SEC source, a thumbs
  up/down that posts to `/feedback`, and a link to the LangSmith trace. Keep it plain and legible —
  this is the screen you will show in an interview.

Capture the run id with `langsmith.run_helpers.get_current_run_tree()` inside the traced call so
`/feedback` and `trace_url` actually work.

---

## 11. Observability

- Graph invocation wrapped with run name `filings_analyst.ask`, tags `[env:{APP_ENV}, graph:v1,
  long:{LONG_MODEL}, fast:{FAST_MODEL}]`, and metadata `{question_type, n_subqueries, hops}`.
- **Sampling**: honour `LANGSMITH_SAMPLE_RATE` — decide per request and set tracing off for that
  invocation when not sampled. 5k traces/month is a real ceiling and one `/ask` is ~10 traces.
- Evals trace to `LANGSMITH_EVAL_PROJECT`, kept separate so app traces and experiment traces don't
  contaminate each other's dashboards.
- Structured JSON logs (`structlog`) for per-node latency, token usage, evidence counts, cache
  hit/miss, and rerank budget remaining — so the project isn't fully dependent on a SaaS dashboard.
  Mention this in the README as a deliberate choice.

---

## 12. Evaluation

This is the part that makes the project resume-worthy. Do not shortcut it.

### Dataset — `evals/dataset.jsonl`, 30 items

```json
{"id": "mh-004", "question": "...", "type": "multi_hop",
 "reference_answer": "...", "gold_chunk_ids": ["NVDA-2024-1A-0037"],
 "gold_numbers": {"revenue_fy2024": "130497000000"}}
```

Distribution: 8 single-hop, 8 multi-hop (cross-year or cross-company), 6 numeric,
4 contradiction, 4 unanswerable (out-of-corpus company, out-of-range year, or a question the
filings genuinely don't address).

`evals/make_dataset.py` generates *candidates* by sampling real ingested chunks and asking the
long model to write a question answerable only from them — recording the source chunk id as gold.
Output goes to `evals/dataset.draft.jsonl` for human review; the committed `dataset.jsonl` is the
reviewed version. **Be explicit in the README that gold labels were human-reviewed** — a
self-generated, self-graded benchmark is worthless and an interviewer will ask.

### Metrics — `evals/evaluators.py`

Retrieval (deterministic, free):
- `recall_at_k` — gold chunk ids present in retrieved evidence
- `precision_at_k`, `mrr`

Answer quality (LLM-as-judge, `JUDGE_MODEL`, temperature 0, structured output):
- `citation_support` — for each `[Cn]`, does the cited chunk actually support the sentence? Score = supported / total.
- `answer_correctness` — 1–5 vs `reference_answer`, normalised to 0–1.
- `numeric_accuracy` — deterministic, not a judge: exact match of extracted values against `gold_numbers`.
- `refusal_correctness` — deterministic: did it refuse exactly when it should have?

Operational: p50/p95 latency, LLM calls per question, cache hit rate.

### Runner

`make eval SUITE=smoke` / `SUITE=full`. Uses LangSmith `evaluate()` against an uploaded dataset;
writes a markdown table to `evals/results/{date}-{suite}.md` **and commits it**. The README's
results table is copied from a real run, with the date and model IDs stated.

### CI gate — `.github/workflows/eval.yml`

On pull request: run the 8-question smoke suite, compare against `config/thresholds.yaml`:

```yaml
citation_support: 0.85
recall_at_k: 0.80
refusal_correctness: 1.00
answer_correctness: 0.70
```

Fail the job below threshold and write the comparison to the GitHub job summary. Skip the whole
job cleanly (not fail) when secrets are absent, so forks and fresh clones aren't red.

`ci.yml` runs ruff, mypy, and the unit tests — no keys, always green.

---

## 13. README

Required sections, in this order:

1. **What it is** — two sentences and the screenshot of the demo page.
2. **Quickstart** — `cp .env.example .env`, `make preflight`, `make ingest`, `make run`. State the
   expected ingest time and token cost.
3. **Architecture** — the mermaid graph diagram, plus a short paragraph per layer.
4. **Design decisions** — the core of the document. For each: what was chosen, why, and *what was
   rejected*. Cover at minimum:
   - LangGraph over a linear chain — the refine cycle, checkpointing, resumable threads
   - **Numbers from XBRL, narrative from RAG** — the hallucination-containment argument
   - Section-scoped ingestion (1A/7/7A only) — precision and embedding budget vs recall
   - One index + metadata filters vs namespace-per-company
   - Two-provider model routing — Groq's 6k TPM ceiling vs Gemini's RPM ceiling, and why the
     split falls where it does
   - Batched grading in one call instead of per-chunk calls — quota arithmetic
   - A deterministic verification node instead of trusting the synthesizer
   - Eval-gated CI — why a RAG system without regression tests is not a system
   - SQLite LLM cache and trace sampling — engineering around free-tier limits
5. **Evaluation results** — the real table, with dataset size, date, model IDs, and a one-line
   reading of what the numbers mean.
6. **Cost and quota analysis** — a table of what one query costs in LLM calls, tokens, and traces,
   and how many queries/month the free tiers support. This is the section that signals production
   thinking.
7. **Limitations** — judge-model bias (same family judges and generates), 8-filing corpus, XBRL
   tag inconsistency across filers, Pinecone Starter pauses idle indexes after 3 weeks, no
   streaming, section regex is brittle on unusual filing layouts.
8. **What I'd do next** — hybrid sparse+dense retrieval, human-labelled eval set, cross-encoder
   reranking, streaming responses, Langfuse as a self-hosted tracing alternative.

Write it as an engineer explaining trade-offs, not as marketing copy. No emoji headers, no
"blazingly fast", no feature bullet lists without rationale.

---

## 14. Build order

| # | Milestone | Done when | Budget |
|---|---|---|---|
| M0 | Scaffold, settings, `.env.example`, Makefile, preflight | `make preflight` prints a pass/fail table | 15 min |
| M1 | EDGAR client, section extraction, chunker, tests | `make ingest --dry-run` prints the full plan and token estimate | 35 min |
| M2 | Pinecone index + upsert + filtered search CLI | `python -m filings_analyst.retrieval.store "supply chain risk" --ticker NVDA --fy 2024` returns sane chunks | 20 min |
| M3 | Graph: state, nodes, prompts, edges, checkpointer | `python -m filings_analyst.agent.graph "<multi-hop question>"` returns a cited answer | 45 min |
| M4 | FastAPI + demo page + feedback endpoint | `make run`, ask a question in the browser, click a citation, submit feedback | 20 min |
| M5 | Eval dataset, evaluators, runner, CI workflows | `make eval SUITE=smoke` produces a results table | 30 min |
| M6 | README with real numbers | Someone who has never seen the repo could run it | 15 min |

Commit at every milestone (conventional commits). If you fall behind, cut M4's styling and the
`compare` node's sophistication — never cut M5.

---

## 15. Definition of done

- [ ] `make preflight` passes with only the five keys in `.env`
- [ ] `make ingest` completes under the embedding budget and reports actual tokens used
- [ ] A multi-hop cross-year question returns an answer where every citation resolves to a real
      chunk and every number traces to XBRL or an extracted fact
- [ ] An out-of-corpus question is refused, naming what is available
- [ ] The verification node demonstrably catches a fabricated citation (there is a unit test for it)
- [ ] `make eval SUITE=smoke` produces a committed results table
- [ ] CI is green on a fresh clone with no secrets
- [ ] README contains the graph diagram, the decisions section, real eval numbers, and the cost table
- [ ] Nothing in the repo requires a credit card on any service

---

## 16. Out of scope — do not build

Docker Compose stacks, Kubernetes manifests, React/Next frontends, user auth, Postgres, Celery,
local embedding models, agent memory beyond the checkpointer, streaming SSE, more than 8 filings,
a CLI framework beyond `argparse`/`typer`, or any abstraction layer "for future providers."

If you find yourself writing a plugin system, stop.
