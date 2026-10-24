"""
CRAG Knowledge Refinement (a.k.a. 'decompose-then-recompose').

Splits each retrieved doc into short strips (sentence-ish units) and keeps
only strips that overlap with query terms. This throws away boilerplate and
irrelevant paragraphs BEFORE we hand context to the generator.

Deliberately lightweight — no extra LLM call. Falls back to the original
content if nothing survives filtering (so we never lose signal).
"""
from __future__ import annotations

import re
from dataclasses import replace

from retrieval.hybrid_search import HybridResult

_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_WORD_RE = re.compile(r"[A-Za-z0-9_]{3,}")

# Very common English stop-ish tokens we don't want to match on
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "what", "which",
    "are", "was", "were", "have", "has", "had", "you", "your", "into",
    "how", "why", "when", "where", "who", "does", "did", "can", "could",
    "should", "would", "about", "there", "their", "them", "they", "will",
}


def _keywords(query: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(query) if w.lower() not in _STOP}


def _refine_text(text: str, kw: set[str]) -> str:
    strips = [s.strip() for s in _SPLIT_RE.split(text) if s.strip()]
    if not strips:
        return text
    kept = [s for s in strips if any(k in s.lower() for k in kw)]
    return " ".join(kept) if kept else text  # fallback to original


def refine_documents(
    query: str,
    docs: list[HybridResult],
) -> list[HybridResult]:
    """Return copies of docs with `content` reduced to query-relevant strips."""
    kw = _keywords(query)
    if not kw:
        return docs
    return [replace(d, content=_refine_text(d.content, kw)) for d in docs]
