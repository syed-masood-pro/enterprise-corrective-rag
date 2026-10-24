"""
Tavily web search provider for CRAG fallback.

Uses the Tavily REST API directly via httpx (no extra SDK dependency).
Requires TAVILY_API_KEY in .env.

Docs: https://docs.tavily.com/docs/rest-api/api-reference
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from retrieval.hybrid_search import HybridResult

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"


class TavilyWebSearch:
    """Real web-search provider backed by Tavily."""

    name = "tavily"

    def __init__(
        self,
        api_key: str,
        search_depth: str = "advanced",  # "basic" | "advanced"
        include_answer: bool = False,
        timeout: float = 15.0,
    ) -> None:
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required for TavilyWebSearch")
        self._api_key = api_key
        self._depth = search_depth
        self._include_answer = include_answer
        self._timeout = timeout

    async def search(self, query: str, top_k: int = 5) -> list[HybridResult]:
        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": self._depth,
            "max_results": top_k,
            "include_answer": self._include_answer,
            "include_raw_content": False,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(_TAVILY_URL, json=payload)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as exc:
            logger.error("Tavily search failed: %s", exc)
            return []

        results = data.get("results", []) or []
        logger.info("Tavily returned %d results for query=%r", len(results), query[:60])

        out: list[HybridResult] = []
        for i, item in enumerate(results):
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "") or ""
            score = float(item.get("score", 0.0) or 0.0)
            body = f"{title}\n\n{content}\n\nURL: {url}" if title else content
            out.append(
                HybridResult(
                    chunk_id=f"web::tavily::{i}::{abs(hash(url)) & 0xffffffff:x}",
                    content=body,
                    metadata={
                        "source": "web",
                        "provider": "tavily",
                        "url": url,
                        "title": title,
                        "doc_name": title or url or "web:tavily",
                        "file_type": "web",
                        "chunk_index": i,
                    },
                    rrf_score=score,  # Tavily relevance score (0–1)
                )
            )
        return out
