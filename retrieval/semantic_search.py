"""Semantic (dense vector) search via Pinecone."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from vectorstore.embedder import get_embeddings

logger = logging.getLogger(__name__)


@dataclass
class SemanticResult:
    """A single vector-search hit."""

    chunk_id: str
    content: str
    metadata: dict
    score: float  # relevance score — higher means more similar


def semantic_search(
    store: object,
    query: str,
    top_k: int = 10,
) -> list[SemanticResult]:
    """
    Run a dense vector similarity search against Pinecone.

    Args:
        store:  Pinecone Index object.
        query:  User query string (will be embedded internally).
        top_k:  Number of candidate results to retrieve.

    Returns:
        List of SemanticResult sorted by similarity score descending.
    """
    try:
        stats = store.describe_index_stats()
        if (stats.total_vector_count or 0) == 0:
            logger.warning("Semantic search called on an empty index")
            return []
    except Exception:
        pass

    try:
        embeddings = get_embeddings()
        query_vec = embeddings.embed_query(query)
        raw = store.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
        )
    except Exception as exc:
        logger.error("Pinecone similarity search failed: %s", exc)
        return []

    results: list[SemanticResult] = []
    for match in raw.get("matches", []):
        meta = match.get("metadata", {})
        chunk_id = meta.get("chunk_id", match.get("id", ""))
        content = meta.get("text", "")
        results.append(
            SemanticResult(
                chunk_id=chunk_id,
                content=content,
                metadata=meta,
                score=float(match.get("score", 0.0)),
            )
        )

    logger.debug("Semantic search returned %d results", len(results))
    return results
