"""
Input sanitisation — prevents prompt injection and malformed queries
from reaching the LLM or the retrieval pipeline.
"""
from __future__ import annotations

import re

# ── Prompt injection patterns ─────────────────────────────────────────────────
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you|a)\s+", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE),
    re.compile(r"###\s*instruction", re.IGNORECASE),
    re.compile(r"system\s*:\s*(you|ignore)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
]

_MAX_QUERY_LENGTH = 1_000


def sanitize_query(query: str) -> str:
    """
    Sanitise a raw user query.

    Steps:
    1. Strip surrounding whitespace.
    2. Remove null bytes and ASCII control characters (except \\n and \\t).
    3. Truncate to _MAX_QUERY_LENGTH characters.
    4. Reject prompt-injection patterns.

    Returns:
        Cleaned query string.

    Raises:
        ValueError: If the query is empty or contains disallowed content.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    # Remove dangerous control characters
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)
    cleaned = cleaned.strip()[:_MAX_QUERY_LENGTH]

    if not cleaned:
        raise ValueError("Query is empty after sanitisation.")

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            raise ValueError(
                "Query contains content that is not allowed. "
                "Please rephrase your question."
            )

    return cleaned
