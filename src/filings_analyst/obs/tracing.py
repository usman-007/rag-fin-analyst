"""LangSmith tracing: per-request sampling, run tags, and the run-id/trace-url capture
that makes `/feedback` and the API's `trace_url` field actually work.

5,000 traces/month is a real ceiling on the Developer plan, and one `/ask` call is
roughly 10 traces (the top-level run plus the graph's nodes and LLM calls nested under
it) — `LANGSMITH_SAMPLE_RATE` deciding tracing per request, not a global on/off switch,
is what keeps this inside the free allowance under real traffic.

`get_current_run_tree()` is only populated inside an `@traceable`-decorated function, not
automatically by LangGraph's own (callback-based) tracing — verified directly against the
installed langsmith package: wrapping the top-level `ask` call in `traceable(...)` and
invoking it inside `tracing_context(enabled=...)` is what makes the run tree (and
everything LangGraph traces underneath it) resolvable here.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree, tracing_context

from filings_analyst.obs.logging import get_logger
from filings_analyst.settings import get_settings

logger = get_logger(__name__)

T = TypeVar("T", bound=Callable[..., dict[str, Any]])


def _should_sample() -> bool:
    return random.random() < get_settings().langsmith_sample_rate


def make_traced_ask(fn: T) -> T:
    """Wrap the graph's `ask` entrypoint so its run is capturable via
    `get_current_run_tree()` — call inside `traced_ask()` below."""
    return traceable(name="filings_analyst.ask", run_type="chain")(fn)  # type: ignore[return-value]


@contextmanager
def traced_ask() -> Iterator[bool]:
    """Wrap one `/ask` call: decides sampling, sets project/tags for this run only —
    other concurrent requests are unaffected. Yields whether this call was sampled."""
    settings = get_settings()
    sampled = bool(settings.langsmith_api_key) and _should_sample()
    tags = [
        f"env:{settings.app_env}",
        "graph:v1",
        f"long:{settings.long_model}",
        f"fast:{settings.fast_model}",
    ]
    with tracing_context(project_name=settings.langsmith_project, tags=tags, enabled=sampled):
        yield sampled


def current_run_id() -> str | None:
    run = get_current_run_tree()
    return str(run.id) if run is not None else None


def current_trace_url() -> str | None:
    run = get_current_run_tree()
    if run is None:
        return None
    try:
        return run.get_url()
    except Exception as exc:  # trace URL is best-effort — never fail the request over it
        logger.warning("tracing.get_url_failed", error=str(exc))
        return None
