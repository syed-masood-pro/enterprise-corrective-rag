"""
Valkey caching for CRAG:
  - Evaluator grade per query (avoid re-running the grader LLM call)
  - Web search results per query (avoid hitting the external API repeatedly)

Both use plain STRING keys with EX TTL — simple, replica-safe, and
independent of the HNSW semantic cache used for full answers.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_GRADE_PREFIX = "rag:crag:grade:"
_WEB_PREFIX = "rag:crag:web:"
_TITLE_PREFIX="rag:title:"


def _key(prefix: str, query: str) -> str:
    h = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:32]
    return f"{prefix}{h}"


class CRAGCache:
    def __init__(self, valkey_client: Any, grade_ttl: int = 3600, web_ttl: int = 1800, title_ttl:int=86400) -> None:
        self._c = valkey_client
        self._grade_ttl = grade_ttl
        self._web_ttl = web_ttl
        self._title_ttl=title_ttl

    # ── Grade cache ───────────────────────────────────────────────────────────
    async def get_grade(self, query: str) -> Optional[tuple[str, str]]:
        try:
            raw = await self._c.get(_key(_GRADE_PREFIX, query))
            if not raw:
                return None
            obj = json.loads(raw)
            return obj["grade"], obj.get("reason", "")
        except Exception as exc:
            logger.debug("CRAG grade cache miss/err: %s", exc)
            return None

    async def set_grade(self, query: str, grade: str, reason: str = "") -> None:
        try:
            await self._c.set(
                _key(_GRADE_PREFIX, query),
                json.dumps({"grade": grade, "reason": reason}),
                ex=self._grade_ttl,
            )
        except Exception as exc:
            logger.debug("CRAG grade cache set err: %s", exc)

    # ── Web-search cache ──────────────────────────────────────────────────────
    async def get_web(self, query: str) -> Optional[list[dict]]:
        try:
            raw = await self._c.get(_key(_WEB_PREFIX, query))
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.debug("CRAG web cache miss/err: %s", exc)
            return None

    async def set_web(self, query: str, docs: list[dict]) -> None:
        try:
            await self._c.set(
                _key(_WEB_PREFIX, query),
                json.dumps(docs),
                ex=self._web_ttl,
            )
        except Exception as exc:
            logger.debug("CRAG web cache set err: %s", exc)
            
    # ── Chat-title cache ──────────────────────────────────────────────────────
    async def get_title(self, query: str) -> Optional[str]:
        try:
            raw = await self._c.get(_key(_TITLE_PREFIX, query))
            if not raw:
                return None
            return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        except Exception as exc:
            logger.debug("Title cache miss/err: %s", exc)
            return None

    async def set_title(self, query: str, title: str) -> None:
        try:
            await self._c.set(_key(_TITLE_PREFIX, query), title, ex=self._title_ttl)
        except Exception as exc:
            logger.debug("Title cache set err: %s", exc)