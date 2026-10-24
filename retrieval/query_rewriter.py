"""
Lightweight LLM-based query rewriter used by CRAG when web-search results
are graded as insufficient. Produces a single improved query string that is
more web-search friendly (adds keywords, resolves pronouns, etc.).
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You rewrite user questions into better web-search queries.\n"
    "Rules:\n"
    "  - Output ONE line only — the rewritten query, no quotes, no prefix.\n"
    "  - Add specific keywords, entities, and disambiguating context.\n"
    "  - Remove filler words. Keep it under 20 words.\n"
    "  - Do NOT answer the question."
)


async def rewrite_for_web(query: str, llm: ChatGroq, prior_attempt: str = "") -> str:
    """Return a rewritten query optimised for web search."""
    hint = f"\n(A previous attempt {prior_attempt!r} returned weak results — try a different angle.)" if prior_attempt else ""
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"Original question: {query}{hint}\nRewritten web query:"),
        ])
        new_q = (resp.content or "").strip().splitlines()[0].strip().strip('"').strip("'")
        if not new_q:
            return query
        logger.info("Query rewrite: %r -> %r", query[:60], new_q[:60])
        return new_q
    except Exception as exc:
        logger.warning("Query rewrite failed (%s) — using original query", exc)
        return query
