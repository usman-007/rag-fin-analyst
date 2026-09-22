"""Typed settings loaded from `.env`. Nothing in this codebase reads `os.environ` directly.

One deliberate exception: `get_settings()` exports the `LANGSMITH_*` values back into
`os.environ`. The `langsmith` SDK reads those directly to drive its implicit
LangGraph/LangChain tracing hooks (that's the whole point of `tracing_context()` in
obs/tracing.py — no client needs wiring at every call site), and `uv run` does not
export `.env` into the process environment on its own, so without this the tracing
client runs with no credentials and every run submission is silently unauthenticated.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- LLM providers ----------------------------------------------------
    google_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # ---- Vector store -------------------------------------------------------
    pinecone_api_key: SecretStr | None = None
    pinecone_index: str = "filings-analyst"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "filings"
    pinecone_embed_model: str = "llama-text-embed-v2"
    pinecone_rerank_model: str = "pinecone-rerank-v0"

    # ---- Observability --------------------------------------------------------
    langsmith_api_key: SecretStr | None = None
    langsmith_tracing: bool = True
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "filings-analyst"
    langsmith_eval_project: str = "filings-analyst-eval"
    langsmith_sample_rate: float = 1.0

    # ---- SEC EDGAR --------------------------------------------------------
    sec_user_agent: str | None = None

    # ---- Models -------------------------------------------------------------
    fast_model: str = "openai/gpt-oss-120b"
    long_model: str = "gemini-2.5-flash"
    judge_model: str = "gemini-2.5-flash"

    # ---- Budgets and behaviour ----------------------------------------------
    max_embed_tokens: int = 1_500_000
    max_hops: int = 2
    top_k: int = 8
    rerank_top_n: int = 4
    rerank_monthly_budget: int = 400
    llm_cache_path: Path = Path("data/llm_cache.sqlite")
    checkpoint_db: Path = Path("data/checkpoints.sqlite")
    log_level: str = "INFO"

    app_env: str = "dev"

    @field_validator("langsmith_sample_rate")
    @classmethod
    def _clamp_sample_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("LANGSMITH_SAMPLE_RATE must be between 0 and 1")
        return v

    @property
    def data_dir(self) -> Path:
        return Path("data")

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def ensure_data_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.llm_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)


def _sync_langsmith_env(settings: Settings) -> None:
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_api_key is not None:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    _sync_langsmith_env(settings)
    return settings
