"""
CRAG Retrieval Evaluator.

Given the user query and the fused RRF documents, ask the LLM to grade the
retrieval as one of exactly three states: Correct | Incorrect | Ambiguous.

The grade drives the CRAG action router (see pipeline/query_pipeline.py):
  - Correct    → Knowledge Refinement only
  - Incorrect  → Discard docs, run Web Search fallback
  - Ambiguous  → Knowledge Refinement + Web Search (concatenated)
"""
from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from retrieval.hybrid_search import HybridResult

logger = logging.getLogger(__name__)

Grade = Literal["Correct", "Incorrect", "Ambiguous"]
_VALID: set[str] = {"Correct", "Incorrect", "Ambiguous"}

_SYSTEM = (
    "You are a strict retrieval quality grader for a RAG system.\n"
    "Given a user query and a set of retrieved document snippets, decide if the "
    "snippets contain enough evidence to fully answer the query.\n\n"
    "Respond with a single JSON object: {\"grade\": \"<label>\", \"reason\": \"<short>\"}\n"
    "Labels (choose exactly one):\n"
    "  - Correct    : snippets clearly and sufficiently answer the query.\n"
    "  - Incorrect  : snippets are off-topic or missing the needed facts.\n"
    "  - Ambiguous  : snippets are partially relevant but incomplete or uncertain.\n"
    "Do not output anything except the JSON object."
)


def _format_docs(docs: list[HybridResult], max_chars: int = 500) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        snippet = d.content[:max_chars].replace("\n", " ")
        lines.append(f"[{i}] {snippet}")
    return "\n".join(lines) if lines else "(no documents retrieved)"


async def evaluate_retrieval(
    query: str,
    docs: list[HybridResult],
    llm: ChatGroq,
) -> tuple[Grade, str]:
    """
    Grade the RRF-fused docs against the query.

    Returns:
        (grade, reason) — grade ∈ {Correct, Incorrect, Ambiguous}.
        Defaults to 'Ambiguous' on any parse failure (safe middle-ground:
        will do refinement + web search).
    """
    if not docs:
        return "Incorrect", "no documents retrieved"

    user = (
        f"Query:\n{query}\n\n"
        f"Retrieved snippets:\n{_format_docs(docs)}\n\n"
        "Return the JSON grade now."
    )

    try:
        resp = await llm.ainvoke([SystemMessage(content=_SYSTEM), HumanMessage(content=user)])
        raw = (resp.content or "").strip()
        # Tolerate ```json fences
        if raw.startswith("```"):
            raw = raw.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        obj = json.loads(raw)
        grade = str(obj.get("grade", "")).strip().capitalize()
        reason = str(obj.get("reason", ""))[:200]
        if grade not in _VALID:
            logger.warning("CRAG grader returned invalid label %r — defaulting to Ambiguous", grade)
            return "Ambiguous", f"invalid label {grade!r}"
        return grade, reason  # type: ignore[return-value]
    except Exception as exc:
        logger.warning("CRAG grader failed (%s) — defaulting to Ambiguous", exc)
        return "Ambiguous", f"grader error: {exc}"