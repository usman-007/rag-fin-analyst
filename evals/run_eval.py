"""LangSmith-backed eval runner: `make eval SUITE=smoke` / `SUITE=full`.

Uploads (or reuses) a LangSmith dataset built from `evals/dataset.jsonl`, runs the
agent graph against it via `evaluate()`, and writes a markdown results table to
`evals/results/{date}-{suite}.md`. Eval traces go to `LANGSMITH_EVAL_PROJECT`, kept
separate from the app's own `LANGSMITH_PROJECT` so experiment runs and production
traffic don't contaminate each other's dashboards.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from evaluators import (
    answer_correctness,
    citation_support,
    mrr,
    numeric_accuracy,
    precision_at_k,
    recall_at_k,
    refusal_correctness,
)
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.run_helpers import tracing_context
from langsmith.schemas import Example, Run

from filings_analyst.agent.graph import ask
from filings_analyst.agent.state import Citation
from filings_analyst.obs.logging import get_logger
from filings_analyst.settings import get_settings

logger = get_logger(__name__)

DATASET_PATH = Path("evals/dataset.jsonl")
RESULTS_DIR = Path("evals/results")
THRESHOLDS_PATH = Path("config/thresholds.yaml")

# 8-question smoke suite for CI: one item from (almost) every category, kept small so
# a PR check stays fast and cheap.
SMOKE_IDS = {"num-01", "num-05", "sh-01", "sh-06", "mh-01", "mh-03", "contra-01", "unans-01"}


def load_dataset(suite: str) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]
    if suite == "smoke":
        items = [item for item in items if item["id"] in SMOKE_IDS]
    return items


def ensure_langsmith_dataset(client: Client, name: str, items: list[dict[str, Any]]) -> None:
    if client.has_dataset(dataset_name=name):
        return
    client.create_dataset(dataset_name=name, description="Filings analyst eval set")
    client.create_examples(
        dataset_name=name,
        inputs=[{"question": item["question"]} for item in items],
        outputs=[
            {
                "reference_answer": item["reference_answer"],
                "gold_chunk_ids": item["gold_chunk_ids"],
                "gold_numbers": item["gold_numbers"],
                "type": item["type"],
            }
            for item in items
        ],
        metadata=[{"question_id": item["id"]} for item in items],
    )


def target(inputs: dict[str, Any]) -> dict[str, Any]:
    start = time.monotonic()
    result = ask(inputs["question"])
    latency_ms = (time.monotonic() - start) * 1000
    scope = result.get("scope")
    return {
        "answer": result.get("draft") or "",
        "citations": [c.model_dump() for c in result.get("citations", [])],
        "facts": [f.model_dump() for f in result.get("facts", [])],
        "evidence_ids": [e.chunk_id for e in result.get("evidence", [])],
        "refused": bool(scope is not None and not scope.in_scope),
        "llm_calls": result.get("llm_calls", 0),
        "latency_ms": latency_ms,
    }


def _eval_recall(run: Run, example: Example) -> dict[str, Any]:
    gold = (example.outputs or {}).get("gold_chunk_ids", [])
    retrieved = (run.outputs or {}).get("evidence_ids", [])
    return {"key": "recall_at_k", "score": recall_at_k(gold, retrieved)}


def _eval_precision(run: Run, example: Example) -> dict[str, Any]:
    gold = (example.outputs or {}).get("gold_chunk_ids", [])
    retrieved = (run.outputs or {}).get("evidence_ids", [])
    return {"key": "precision_at_k", "score": precision_at_k(gold, retrieved)}


def _eval_mrr(run: Run, example: Example) -> dict[str, Any]:
    gold = (example.outputs or {}).get("gold_chunk_ids", [])
    retrieved = (run.outputs or {}).get("evidence_ids", [])
    return {"key": "mrr", "score": mrr(gold, retrieved)}


def _eval_numeric(run: Run, example: Example) -> dict[str, Any]:
    gold_numbers = (example.outputs or {}).get("gold_numbers", {})
    fact_values = [f["value"] for f in (run.outputs or {}).get("facts", []) if f.get("value")]
    return {"key": "numeric_accuracy", "score": numeric_accuracy(gold_numbers, fact_values)}


def _eval_refusal(run: Run, example: Example) -> dict[str, Any]:
    expected = (example.outputs or {}).get("type") == "unanswerable"
    actual = bool((run.outputs or {}).get("refused", False))
    return {"key": "refusal_correctness", "score": refusal_correctness(expected, actual)}


def _eval_citation_support(run: Run, example: Example) -> dict[str, Any]:
    citations = [Citation(**c) for c in (run.outputs or {}).get("citations", [])]
    score = citation_support((run.outputs or {}).get("answer", ""), citations)
    return {"key": "citation_support", "score": score}


def _eval_answer_correctness(run: Run, example: Example) -> dict[str, Any]:
    question = (example.inputs or {}).get("question", "")
    reference = (example.outputs or {}).get("reference_answer", "")
    answer = (run.outputs or {}).get("answer", "")
    return {"key": "answer_correctness", "score": answer_correctness(question, answer, reference)}


EVALUATORS = [
    _eval_recall,
    _eval_precision,
    _eval_mrr,
    _eval_numeric,
    _eval_refusal,
    _eval_citation_support,
    _eval_answer_correctness,
]


def _aggregate(rows: list[Any]) -> dict[str, float]:
    by_key: dict[str, list[float]] = {}
    for row in rows:
        for result in row["evaluation_results"]["results"]:
            by_key.setdefault(result.key, []).append(result.score or 0.0)
    return {key: statistics.mean(scores) for key, scores in by_key.items() if scores}


def _run_field(rows: list[Any], field: str) -> list[float]:
    return [
        row["run"].outputs[field]
        for row in rows
        if row["run"].outputs and field in row["run"].outputs
    ]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[idx]


def write_results_table(suite: str, metrics: dict[str, float], rows: list[Any], path: Path) -> None:
    latencies = _run_field(rows, "latency_ms")
    llm_calls = _run_field(rows, "llm_calls")
    lines = [
        f"# Eval results — {suite} suite — {date.today().isoformat()}",
        "",
        f"Dataset: {len(rows)} questions. Models: see `.env` (FAST_MODEL/LONG_MODEL/JUDGE_MODEL).",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]
    for key in sorted(metrics):
        lines.append(f"| {key} | {metrics[key]:.3f} |")
    lines += [
        "",
        "| Operational | Value |",
        "|---|---|",
        f"| p50 latency (ms) | {_percentile(latencies, 0.5):.0f} |",
        f"| p95 latency (ms) | {_percentile(latencies, 0.95):.0f} |",
        f"| avg LLM calls/question | {statistics.mean(llm_calls) if llm_calls else 0:.1f} |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def check_thresholds(metrics: dict[str, float]) -> list[str]:
    thresholds = yaml.safe_load(THRESHOLDS_PATH.read_text())
    failures = []
    for key, minimum in thresholds.items():
        score = metrics.get(key)
        if score is None:
            failures.append(f"{key}: no score produced")
        elif score < minimum:
            failures.append(f"{key}: {score:.3f} < required {minimum}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the filings-analyst eval suite.")
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.langsmith_api_key:
        print("LANGSMITH_API_KEY not set — skipping eval run (see .github/workflows/eval.yml).")
        return 0

    items = load_dataset(args.suite)
    client = Client(api_key=settings.langsmith_api_key.get_secret_value())
    dataset_name = f"filings-analyst-{args.suite}"
    ensure_langsmith_dataset(client, dataset_name, items)

    with tracing_context(project_name=settings.langsmith_eval_project, enabled=True):
        results = evaluate(
            target,
            data=dataset_name,
            evaluators=EVALUATORS,
            client=client,
            experiment_prefix=f"filings-analyst-{args.suite}",
        )
        rows = list(results)

    metrics = _aggregate(rows)
    results_path = RESULTS_DIR / f"{date.today().isoformat()}-{args.suite}.md"
    write_results_table(args.suite, metrics, rows, results_path)
    print(f"Wrote results to {results_path}")

    failures = check_thresholds(metrics)
    if failures:
        print("THRESHOLD FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All thresholds passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
