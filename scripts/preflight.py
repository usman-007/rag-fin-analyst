"""Validate every credential in `.env` with one minimal live call each.

Run with `make preflight`. Intentionally does not import `filings_analyst.llm` —
this script is the first thing a fresh clone runs, before any rate limiter or
cache is wired up, and it must work (and fail legibly) with an empty `.env`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import httpx

from filings_analyst.settings import get_settings

TIMEOUT = 15.0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_google(api_key: str | None, model: str) -> CheckResult:
    if not api_key:
        return CheckResult("Google AI Studio (Gemini)", False, "GOOGLE_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        resp = httpx.post(
            url,
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": "hi"}]}],
                "generationConfig": {"maxOutputTokens": 1},
            },
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            return CheckResult("Google AI Studio (Gemini)", True, f"model={model} reachable")
        return CheckResult(
            "Google AI Studio (Gemini)", False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        )
    except httpx.HTTPError as exc:
        return CheckResult("Google AI Studio (Gemini)", False, f"request failed: {exc}")


def check_groq(api_key: str | None, model: str) -> CheckResult:
    if not api_key:
        return CheckResult("Groq", False, "GROQ_API_KEY not set")
    url = "https://api.groq.com/openai/v1/chat/completions"
    try:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            return CheckResult("Groq", True, f"model={model} reachable")
        return CheckResult("Groq", False, f"HTTP {resp.status_code}: {resp.text[:200]}")
    except httpx.HTTPError as exc:
        return CheckResult("Groq", False, f"request failed: {exc}")


def check_pinecone(api_key: str | None, index_name: str) -> CheckResult:
    if not api_key:
        return CheckResult("Pinecone", False, "PINECONE_API_KEY not set")
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=api_key)
        names = [idx["name"] for idx in pc.list_indexes()]
        if index_name in names:
            pc.describe_index(index_name)
            return CheckResult("Pinecone", True, f"index '{index_name}' exists and is reachable")
        return CheckResult(
            "Pinecone", True, f"connected; index '{index_name}' not created yet (run make ingest)"
        )
    except Exception as exc:  # SDK raises its own exception hierarchy
        return CheckResult("Pinecone", False, f"request failed: {exc}")


def check_langsmith(api_key: str | None) -> CheckResult:
    if not api_key:
        return CheckResult("LangSmith", False, "LANGSMITH_API_KEY not set")
    try:
        from langsmith import Client

        client = Client(api_key=api_key)
        next(iter(client.list_projects(limit=1)), None)
        return CheckResult("LangSmith", True, "list_projects reachable")
    except Exception as exc:
        return CheckResult("LangSmith", False, f"request failed: {exc}")


def check_sec(user_agent: str | None) -> CheckResult:
    if not user_agent:
        return CheckResult("SEC EDGAR", False, "SEC_USER_AGENT not set")
    if "@" not in user_agent:
        return CheckResult(
            "SEC EDGAR", False, f"SEC_USER_AGENT '{user_agent}' has no email — SEC requires one"
        )
    url = "https://data.sec.gov/submissions/CIK0000320193.json"
    try:
        resp = httpx.get(url, headers={"User-Agent": user_agent}, timeout=TIMEOUT)
        if resp.status_code == 200:
            return CheckResult("SEC EDGAR", True, "submissions API reachable")
        return CheckResult("SEC EDGAR", False, f"HTTP {resp.status_code}")
    except httpx.HTTPError as exc:
        return CheckResult("SEC EDGAR", False, f"request failed: {exc}")


def main() -> int:
    settings = get_settings()
    results = [
        check_google(
            settings.google_api_key.get_secret_value() if settings.google_api_key else None,
            settings.long_model,
        ),
        check_groq(
            settings.groq_api_key.get_secret_value() if settings.groq_api_key else None,
            settings.fast_model,
        ),
        check_pinecone(
            settings.pinecone_api_key.get_secret_value() if settings.pinecone_api_key else None,
            settings.pinecone_index,
        ),
        check_langsmith(
            settings.langsmith_api_key.get_secret_value() if settings.langsmith_api_key else None
        ),
        check_sec(settings.sec_user_agent),
    ]

    name_w = max(len(r.name) for r in results) + 2
    print(f"{'SERVICE'.ljust(name_w)}{'STATUS'.ljust(8)}DETAIL")
    print("-" * (name_w + 8 + 40))
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"{r.name.ljust(name_w)}{status.ljust(8)}{r.detail}")

    n_ok = sum(r.ok for r in results)
    print(f"\n{n_ok}/{len(results)} checks passed.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
