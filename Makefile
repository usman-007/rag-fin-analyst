.PHONY: preflight ingest run eval test lint typecheck fmt

SUITE ?= smoke

preflight:
	uv run python scripts/preflight.py

ingest:
	uv run python -m filings_analyst.ingest.pipeline

run:
	uv run uvicorn filings_analyst.api.main:app --reload --port 8000

eval:
	uv run python evals/run_eval.py --suite $(SUITE)

test:
	uv run pytest -q

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

fmt:
	uv run ruff format .
