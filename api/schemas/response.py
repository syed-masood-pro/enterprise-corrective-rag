"""
Response schemas — Pydantic models for API responses.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class SourceChunk(BaseModel):
    """One retrieved chunk cited in the answer."""

    doc_name: str
    file_type: str
    chunk_index: int
    rrf_score: float
    rerank_score: Optional[float] = None  # cross-encoder relevance score
    bm25_rank: int
    semantic_rank: int
    found_by: str
    snippet: str


class QueryResponse(BaseModel):
    """Response for POST /api/v1/query"""

    query: str
    answer: str
    sources: list[SourceChunk]
    cached: bool
    cache_type: Optional[str] = None   # "semantic" | None
    cache_sim: Optional[float] = None  # cosine similarity score on cache hit
    metrics: Optional[dict] = None     # pipeline timing/observability metrics


class IngestResponse(BaseModel):
    """Response for POST /api/v1/ingest"""

    doc_name: str
    chunks_produced: int
    chunks_inserted: int
    chunks_skipped: int
    total_indexed: int
    message: str


class HealthResponse(BaseModel):
    """Response for GET /health"""

    status: str
    vector_store: dict
    bm25_index: dict
    semantic_cache: dict
    groq_model: str
