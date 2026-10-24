"""
Hybrid retrieval using Reciprocal Rank Fusion (RRF).

RRF formula
-----------
For each document d across both ranked lists:

    rrf_score(d) = Σ  1 / (k + rank(d))

where k = 60 (smoothing constant that dampens the influence of very high
individual ranks and ensures every document gets a meaningful score).

Both retrievers contribute equally — no manual weight tuning needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from retrieval.bm25_index import BM25Index, BM25Result
from retrieval.semantic_search import SemanticResult, semantic_search

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """A single fused result after RRF."""

    chunk_id: str
    content: str
    metadata: dict
    rrf_score: float
    bm25_rank: int = -1      # position in BM25 list  (-1 = absent)
    semantic_rank: int = -1  # position in semantic list (-1 = absent)

    @property
    def found_by(self) -> str:
        """Human-readable retriever provenance."""
        if self.bm25_rank >= 0 and self.semantic_rank >= 0:
            return "both"
        if self.bm25_rank >= 0:
            return "bm25"
        return "semantic"


# ── Core RRF logic ────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    bm25_results: list[BM25Result],
    semantic_results: list[SemanticResult],
    k: int = 60,
    top_k: int = 5,
) -> list[HybridResult]:
    """
    Merge two ranked lists with RRF and return the top *top_k* results.

    Args:
        bm25_results:     BM25 candidates — index 0 is the best match.
        semantic_results: Semantic candidates — index 0 is the best match.
        k:                RRF smoothing constant (default 60).
        top_k:            Number of final results to return.

    Returns:
        Sorted list of HybridResult (highest RRF score first).
    """
    scores: dict[str, dict] = {}

    # ── BM25 contribution ─────────────────────────────────────────────────────
    for rank, result in enumerate(bm25_results):
        cid = result.chunk_id
        if not cid:
            continue
        if cid not in scores:
            scores[cid] = {
                "content": result.content,
                "metadata": result.metadata,
                "rrf_score": 0.0,
                "bm25_rank": -1,
                "semantic_rank": -1,
            }
        scores[cid]["rrf_score"] += 1.0 / (k + rank + 1)
        scores[cid]["bm25_rank"] = rank

    # ── Semantic contribution ─────────────────────────────────────────────────
    for rank, result in enumerate(semantic_results):
        cid = result.chunk_id
        if not cid:
            continue
        if cid not in scores:
            scores[cid] = {
                "content": result.content,
                "metadata": result.metadata,
                "rrf_score": 0.0,
                "bm25_rank": -1,
                "semantic_rank": -1,
            }
        scores[cid]["rrf_score"] += 1.0 / (k + rank + 1)
        scores[cid]["semantic_rank"] = rank

    sorted_items = sorted(
        scores.items(),
        key=lambda x: x[1]["rrf_score"],
        reverse=True,
    )[:top_k]

    return [
        HybridResult(
            chunk_id=cid,
            content=data["content"],
            metadata=data["metadata"],
            rrf_score=data["rrf_score"],
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )
        for cid, data in sorted_items
    ]


# ── Public entry-point ────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    store: object,
    bm25_index: BM25Index,
    top_k: int = 10,
    rrf_k: int = 60,
    final_top_k: int = 5,
) -> list[HybridResult]:
    """
    Run BM25 + semantic search and fuse results with RRF.

    Falls back to semantic-only search when the BM25 index is empty
    (e.g., right after a fresh deployment before any documents are ingested).

    Args:
        query:       Sanitized user query.
        store:       Pinecone Index instance.
        bm25_index:  In-memory BM25 index.
        top_k:       Candidates retrieved from each retriever.
        rrf_k:       RRF smoothing constant.
        final_top_k: Number of fused results to return.

    Returns:
        Top final_top_k HybridResults sorted by RRF score descending.
    """
    if not bm25_index.is_ready:
        logger.warning("BM25 index is empty — using semantic-only search")
        sem = semantic_search(store, query, top_k=final_top_k)
        return [
            HybridResult(
                chunk_id=r.chunk_id,
                content=r.content,
                metadata=r.metadata,
                rrf_score=r.score,
                semantic_rank=i,
            )
            for i, r in enumerate(sem)
        ]

    bm25_res = bm25_index.search(query, top_k=top_k)
    sem_res = semantic_search(store, query, top_k=top_k)

    logger.debug(
        "RRF input: bm25=%d semantic=%d for query=%r",
        len(bm25_res), len(sem_res), query[:60],
    )

    fused = reciprocal_rank_fusion(
        bm25_results=bm25_res,
        semantic_results=sem_res,
        k=rrf_k,
        top_k=final_top_k,
    )

    logger.info(
        "Hybrid search done: query=%r → %d results (from bm25:%d + semantic:%d)",
        query[:60], len(fused), len(bm25_res), len(sem_res),
    )
    return fused
    