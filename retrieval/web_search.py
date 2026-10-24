"""
Pluggable Web Search fallback for CRAG.

Providers implement `WebSearchProvider.search(query) -> list[HybridResult]`.
Swap providers by changing settings.web_search_provider and registering here.

Included:
  - StubWebSearch: returns a single placeholder doc so the pipeline still runs
    end-to-end without external API keys. Replace with Tavily / SerpAPI /
    Brave / Bing wrapper when ready.
"""
from __future__ import annotations

import logging
from typing import Protocol

from retrieval.hybrid_search import HybridResult

logger = logging.getLogger(__name__)


class WebSearchProvider(Protocol):
    name: str
    async def search(self, query: str, top_k: int = 5) -> list[HybridResult]: ...


class StubWebSearch:
    """No-op provider — safe default. Replace with a real provider in prod."""
    name = "stub"

    async def search(self, query: str, top_k: int = 5) -> list[HybridResult]:
        logger.info("StubWebSearch invoked for query=%r (top_k=%d)", query[:60], top_k)
        return [
            HybridResult(
                chunk_id=f"web::stub::{hash(query) & 0xffffffff:x}",
                content=(
                    f"[web-fallback stub] No external web search provider is configured. "
                    f"Query was: {query!r}. Configure a real provider in "
                    f"retrieval/web_search.py (Tavily, SerpAPI, Brave, Bing…)."
                ),
                metadata={"source": "web", "provider": "stub", "doc_name": "web:stub", 'file_type':'web','chunk_index':0},
                rrf_score=0.0,
            )
        ]


def get_web_search_provider(name: str = "stub", **kwargs) -> WebSearchProvider:
    name = (name or "stub").lower()
    if name == "stub":
        return StubWebSearch()
    # Extension point:
    # if name == "tavily": return TavilyWebSearch(api_key=...)
    if name == "tavily":
        from retrieval.tavily_search import TavilyWebSearch
        api_key = kwargs.get("api_key", "")
        if not api_key:
            logger.warning("web_search_provider=tavily but TAVILY_API_KEY is empty — falling back to stub")
            return StubWebSearch()
        return TavilyWebSearch(api_key=api_key)
    logger.warning("Unknown web_search_provider=%r — falling back to stub", name)
    return StubWebSearch()