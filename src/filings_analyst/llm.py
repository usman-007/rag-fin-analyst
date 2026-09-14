"""Model factory: two providers, three roles, all rate-limited, retried, and cached.

Groq (`fast_model`) gets short prompts only — its free tier is 30 RPM but only ~6,000
tokens/min, so retrieved context must never be routed there; `assert_short` is the guard
rail against doing that by accident. Gemini Flash (`long_model`, `judge_model`) handles
everything that needs retrieved context, at a much higher token ceiling but a tighter
RPM limit — hence the two separate rate limiters below.

Caching is global (SQLite) so repeat eval runs don't re-consume quota; temperature is 0
everywhere for that cache to be meaningful. `max_retries=0` on every provider client
disables its own retry logic — `invoke_with_retry` is the only retry path, so failures
aren't silently retried twice with different backoff policies.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, TypeVar

from google.api_core.exceptions import (
    GoogleAPICallError,
)
from google.api_core.exceptions import (
    InternalServerError as GoogleInternalServerError,
)
from google.api_core.exceptions import (
    ResourceExhausted as GoogleResourceExhausted,
)
from google.api_core.exceptions import (
    RetryError as GoogleRetryError,
)
from google.api_core.exceptions import (
    ServiceUnavailable as GoogleServiceUnavailable,
)
from groq import (
    APIConnectionError as GroqAPIConnectionError,
)
from groq import (
    APIStatusError as GroqAPIStatusError,
)
from groq import (
    APITimeoutError as GroqAPITimeoutError,
)
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from filings_analyst.ingest.chunker import count_tokens
from filings_analyst.settings import get_settings

_CACHE_CONFIGURED = False

_GOOGLE_RETRYABLE = (
    GoogleResourceExhausted,
    GoogleServiceUnavailable,
    GoogleInternalServerError,
    GoogleRetryError,
)
_GROQ_RETRYABLE = (GroqAPIConnectionError, GroqAPITimeoutError)


def _configure_cache() -> None:
    global _CACHE_CONFIGURED
    if _CACHE_CONFIGURED:
        return
    settings = get_settings()
    settings.ensure_data_dirs()
    set_llm_cache(SQLiteCache(database_path=str(settings.llm_cache_path)))
    _CACHE_CONFIGURED = True


@lru_cache(maxsize=1)
def _fast_rate_limiter() -> InMemoryRateLimiter:
    # Groq free tier: 30 RPM. 0.4 req/s (~24 RPM) leaves headroom.
    return InMemoryRateLimiter(
        requests_per_second=0.4, check_every_n_seconds=0.1, max_bucket_size=1
    )


@lru_cache(maxsize=1)
def _long_rate_limiter() -> InMemoryRateLimiter:
    # Google AI Studio free tier: ~10-15 RPM for Flash. 0.2 req/s (12 RPM) leaves headroom.
    return InMemoryRateLimiter(
        requests_per_second=0.2, check_every_n_seconds=0.1, max_bucket_size=1
    )


def get_fast_llm() -> BaseChatModel:
    """Groq, short prompts, structured output only. Callers must pass prompts through
    `assert_short` first — this is what stops retrieved chunks from being routed here."""
    _configure_cache()
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return ChatGroq(
        model=settings.fast_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_retries=0,
        rate_limiter=_fast_rate_limiter(),
    )


def get_long_llm() -> BaseChatModel:
    """Gemini Flash — retrieved context is allowed here."""
    _configure_cache()
    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    return ChatGoogleGenerativeAI(
        model=settings.long_model,
        google_api_key=settings.google_api_key,
        temperature=0,
        max_retries=0,
        rate_limiter=_long_rate_limiter(),
    )


def get_judge_llm() -> BaseChatModel:
    """Gemini Flash, temperature 0, eval only — never used inside the request path."""
    _configure_cache()
    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    return ChatGoogleGenerativeAI(
        model=settings.judge_model,
        google_api_key=settings.google_api_key,
        temperature=0,
        max_retries=0,
        rate_limiter=_long_rate_limiter(),
    )


def assert_short(prompt: str, limit: int = 2500) -> None:
    """Raise if `prompt` risks blowing Groq's ~6k tokens/min ceiling.

    Call this on every prompt before it reaches `get_fast_llm()`. Token count is an
    estimate (tiktoken cl100k_base); Groq's own tokenizer may differ slightly, so this
    is a guard rail, not an exact accounting.
    """
    n_tokens = count_tokens(prompt)
    if n_tokens > limit:
        raise ValueError(
            f"Prompt has ~{n_tokens} tokens, over the {limit}-token limit for the fast "
            "(Groq) model. Retrieved context must never be routed to Groq — use "
            "get_long_llm() instead."
        )


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _GOOGLE_RETRYABLE) or isinstance(exc, _GROQ_RETRYABLE):
        return True
    if isinstance(exc, GroqAPIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    if isinstance(exc, GoogleAPICallError):
        return exc.code in (429, 500, 502, 503, 504)
    return False


T = TypeVar("T")


def invoke_with_retry(runnable: Runnable[Any, T], *args: Any, **kwargs: Any) -> T:
    """`runnable.invoke(*args, **kwargs)` with exponential backoff + jitter, max 4 attempts.

    Only retries errors that look transient (429/5xx/connection/timeout) for the specific
    exception types Groq's and Google's SDKs actually raise — anything else (auth errors,
    bad requests) fails immediately rather than being retried into a longer timeout.
    """

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(multiplier=1, max=20),
        reraise=True,
    )
    def _call() -> T:
        return runnable.invoke(*args, **kwargs)

    return _call()


def structured_invoke(llm: BaseChatModel, schema: type[T], prompt: str) -> T:
    """`llm.with_structured_output(schema).invoke(prompt)`, retried, with the return type
    narrowed to `schema` — `with_structured_output`'s own type is a `dict | BaseModel` union
    (it also supports TypedDict/JSON-schema targets), but with a pydantic-model `schema` and
    the default `include_raw=False` it always returns an instance of that model at runtime.
    """
    runnable = llm.with_structured_output(schema)
    result = invoke_with_retry(runnable, prompt)
    assert isinstance(result, schema)
    return result
