"""
Request schemas — Pydantic models that validate and document incoming payloads.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Body for POST /api/v1/query"""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask against the ingested documents.",
        examples=["What does the deploy_service function do?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the number of source chunks returned (1–20).",
    )

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    class TitleRequest(BaseModel):
        """Body for POST /api/v1/title — generate a short chat title."""
        query: str = Field(
            ...,
            min_length=1,
            max_length=1000,
            description="The first user message of the conversation.",
        )
        answer: str | None = Field(
            default=None,
            max_length=4000,
            description="Optional first assistant answer for extra context.",
        )

class TitleRequest(BaseModel):
    """Body for POST /api/v1/title — generate a short chat title."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The first user message of the conversation.",
    )
    answer: str | None = Field(
        default=None,
        max_length=4000,
        description="Optional first assistant answer for extra context.",
    )