"""Generate candidate eval questions from real ingested chunks, for human review.

Samples chunks directly from the offline ingest plan (the same chunks `make ingest`
would upsert — no Pinecone dependency), and asks the long model to write one question
answerable only from each sampled chunk, recording the chunk id as gold.

Output goes to `evals/dataset.draft.jsonl`. A human must review every draft item before
merging it into the committed `dataset.jsonl` — an eval set that is both generated and
graded by the same model family is worthless as a correctness check. See the README's
"Evaluation" section.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from filings_analyst.edgar.client import EdgarClient
from filings_analyst.ingest.pipeline import build_plan
from filings_analyst.llm import assert_short, get_long_llm, structured_invoke
from filings_analyst.obs.logging import get_logger

logger = get_logger(__name__)

DRAFT_PATH = Path("evals/dataset.draft.jsonl")

GENERATE_SYSTEM = """Write one question that can be answered using ONLY the passage
below, plus a short reference answer, both grounded strictly in the passage text — do
not use outside knowledge. The question should read naturally, as something a financial
analyst might ask, not "what does this passage say about X"."""


class GeneratedQuestion(BaseModel):
    question: str
    reference_answer: str


def sample_records(n: int, seed: int = 0) -> list[dict[str, Any]]:
    client = EdgarClient()
    plan = build_plan(client)
    rng = random.Random(seed)
    return rng.sample(plan.records, min(n, len(plan.records)))


def generate_candidate(record: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        f"{GENERATE_SYSTEM}\n\nPassage ({record['ticker']} FY{record['fiscal_year']} "
        f"Item {record['item']}):\n{record['chunk_text']}"
    )
    assert_short(prompt, limit=8000)  # long model, but still bounded — one passage, not many
    result = structured_invoke(get_long_llm(), GeneratedQuestion, prompt)
    return {
        "id": f"gen-{record['_id']}",
        "question": result.question,
        "type": "single_hop",
        "reference_answer": result.reference_answer,
        "gold_chunk_ids": [record["_id"]],
        "gold_numbers": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate draft eval candidates from real chunks for human review."
    )
    parser.add_argument("-n", type=int, default=10, help="Number of candidates to generate")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = sample_records(args.n, seed=args.seed)
    DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DRAFT_PATH.open("w") as f:
        for record in records:
            candidate = generate_candidate(record)
            f.write(json.dumps(candidate) + "\n")
            logger.info("make_dataset.generated", id=candidate["id"])

    print(
        f"Wrote {len(records)} draft candidates to {DRAFT_PATH}.\n"
        "Review each one by hand, fix or discard any that are wrong or trivial, then "
        "merge the survivors into evals/dataset.jsonl."
    )


if __name__ == "__main__":
    main()
