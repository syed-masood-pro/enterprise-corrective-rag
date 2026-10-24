"""
Cross-encoder reranking layer.

Why reranking after RRF?
------------------------
RRF merges BM25 and semantic ranks using a position-based formula.
It does NOT look at the actual query-document content relationship.
A Cross-Encoder scores each (query, document) pair with full cross-attention,
producing far more accurate relevance judgements.

Flow: BM25 + Semantic → RRF top-20 → Cross-Encoder → top-5

The cross-encoder is a lightweight model (~66M params) that runs locally
on CPU in ~50ms per candidate.  No API calls, no cost.

Model choices (all free, local, HuggingFace):
  - cross-encoder/ms-marco-MiniLM-L-6-v2   (fastest, ~22MB)
  - BAAI/bge-reranker-base                  (best quality, ~110MB)
  - cross-encoder/ms-marco-MiniLM-L-12-v2   (balanced)
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_instance: Optional[CrossEncoder] = None
_lock = threading.Lock()


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """
    Return the singleton CrossEncoder instance (loaded once, shared).

    Args:
        model_name: HuggingFace cross-encoder model name.

    Returns:
        Shared CrossEncoder instance.
    """
    global _instance

    if _instance is None:
        with _lock:
            if _instance is None:
                logger.info("Loading cross-encoder reranker: %s", model_name)
                _instance = CrossEncoder(
                    model_name,
                    max_length=512,
                    device="cpu",
                )
                logger.info("Cross-encoder reranker loaded")

    return _instance


@dataclass
class RerankedResult:
    """A retrieval result scored by the cross-encoder."""

    chunk_id: str
    content: str
    metadata: dict
    rerank_score: float  # cross-encoder relevance score (higher = more relevant)
    rrf_score: float     # original RRF score (kept for observability)
    bm25_rank: int = -1
    semantic_rank: int = -1

    @property
    def found_by(self) -> str:
        if self.bm25_rank >= 0 and self.semantic_rank >= 0:
            return "both"
        if self.bm25_rank >= 0:
            return "bm25"
        return "semantic"


def rerank(
    query: str,
    candidates: list,
    reranker: CrossEncoder,
    top_k: int = 5,
) -> list[RerankedResult]:
    """
    Score each (query, candidate.content) pair with the cross-encoder
    and return the top-K by rerank_score descending.

    Args:
        query:      The user query string.
        candidates: List of HybridResult objects from RRF fusion.
        reranker:   Shared CrossEncoder instance.
        top_k:      Number of results to keep after reranking.

    Returns:
        Sorted list of RerankedResult (highest rerank_score first).
    """
    if not candidates:
        return []

    # Build (query, document) pairs
    pairs = [(query, c.content) for c in candidates]

    # Score all pairs in one batch
    scores = reranker.predict(pairs, show_progress_bar=False)

    # Zip scores with candidates and sort
    scored = []
    for candidate, score in zip(candidates, scores):
        scored.append(
            RerankedResult(
                chunk_id=candidate.chunk_id,
                content=candidate.content,
                metadata=candidate.metadata,
                rerank_score=float(score),
                rrf_score=candidate.rrf_score,
                bm25_rank=candidate.bm25_rank,
                semantic_rank=candidate.semantic_rank,
            )
        )

    scored.sort(key=lambda x: x.rerank_score, reverse=True)

    logger.info(
        "Reranking done: %d candidates → top-%d (score range: %.4f – %.4f)",
        len(candidates),
        top_k,
        scored[-1].rerank_score if scored else 0,
        scored[0].rerank_score if scored else 0,
    )

    return scored[:top_k]
