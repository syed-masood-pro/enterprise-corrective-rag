"""
Health-check route — no authentication required.
Used by load-balancers, container orchestrators, and monitoring systems.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas.response import HealthResponse
from config.settings import get_settings
from vectorstore.store import get_collection_stats

router = APIRouter(tags=["Health"])


@router.get("/livez",summary="Liveness probe (cheap, no dependencies)")
async def livez()->dict:
    """
    Unconditional liveness check for orchestrators (k8s livenessProbe).

    Returns 200 as long as the process is running and the event loop is
    responsive. It deliberately does NOT touch Pinecone or Valkey — those are
    checked by /health (readiness). Keeping liveness dependency-free prevents a
    transient upstream blip from causing the orchestrator to kill healthy pods.
    """
    return {"status": "alive"}

@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check(request: Request) -> HealthResponse:
    """
    Returns the operational status of all subsystems:
    - Pinecone vector store (document count)
    - BM25 in-memory index (document count, ready flag)
    - Semantic cache: Valkey-backed (entry count, threshold, ttl)
    - Groq model name
    """
    settings = get_settings()
    store = request.app.state.store
    bm25_index = request.app.state.bm25_index
    cache = request.app.state.cache
    valkey_client = request.app.state.valkey_client

    # Valkey connectivity check
    valkey_status = "ok"
    cache_entries = 0
    try:
        await valkey_client.ping()
        cache_entries = await cache.async_size()
    except Exception as exc:
        valkey_status = f"error: {exc}"

    return HealthResponse(
        status="ok",
        vector_store=get_collection_stats(store),
        bm25_index={
            "document_count": bm25_index.document_count,
            "ready": bm25_index.is_ready,
        },
        semantic_cache={
            "backend":     "valkey-hnsw",
            "status":      valkey_status,
            "entries":     cache_entries,
            "threshold":   cache.threshold,
            "ttl_seconds": cache.ttl,
        },
        groq_model=settings.groq_model,
    )
