"""Polite HTTP client for SEC EDGAR: required User-Agent, rate limiting, disk cache.

A second ingest run must hit disk, not the SEC — every response (JSON or HTML) is
cached under `data/cache/`, mirroring the URL path so cache contents are inspectable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from filings_analyst.obs.logging import get_logger
from filings_analyst.settings import get_settings

logger = get_logger(__name__)

REQUEST_DELAY_SECONDS = 0.15


class EdgarClientError(RuntimeError):
    """Raised on any non-2xx response or missing configuration. Never swallowed."""


class EdgarClient:
    def __init__(self, user_agent: str | None = None, cache_dir: Path | None = None) -> None:
        settings = get_settings()
        ua = user_agent or settings.sec_user_agent
        if not ua or "@" not in ua:
            raise EdgarClientError(
                "SEC_USER_AGENT must be set to 'Name email@example.com' — "
                "the SEC returns 403 without a real identifying User-Agent."
            )
        self._user_agent = ua
        self._cache_dir = cache_dir or settings.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_at: float = 0.0

    def _cache_path(self, url: str) -> Path:
        parsed = urlparse(url)
        path = parsed.path.lstrip("/")
        if not path or path.endswith("/"):
            path += "index.json"
        return self._cache_dir / parsed.netloc / path

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS - elapsed)

    def get_text(self, url: str) -> str:
        cache_path = self._cache_path(url)
        if cache_path.exists():
            logger.debug("edgar.cache_hit", url=url)
            return cache_path.read_text(encoding="utf-8")

        self._throttle()
        logger.info("edgar.fetch", url=url)
        resp = httpx.get(url, headers={"User-Agent": self._user_agent}, timeout=30.0)
        self._last_request_at = time.monotonic()
        if resp.status_code != 200:
            raise EdgarClientError(f"GET {url} -> HTTP {resp.status_code}: {resp.text[:300]}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text, encoding="utf-8")
        return resp.text

    def get_json(self, url: str) -> dict[str, Any]:
        return json.loads(self.get_text(url))  # type: ignore[no-any-return]
