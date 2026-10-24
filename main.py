"""
Production RAG API — application entry point.

Start the server:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

  Workers=1 is intentional: all heavy objects (Pinecone, BM25 index,
  embedding model, Groq client) are in-process singletons.
  Horizontal scaling should be done at the container / load-balancer
  level — each replica maintains its own BM25 in memory while sharing
  the same Pinecone serverless index.
"""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.middleware.auth import APIKeyMiddleware
from api.middleware.rate_limiter import limiter
from api.middleware.request_logger import RequestLoggerMiddleware
from api.routes import documents, health, ingest, query
from config.logging_config import setup_logging
from config.settings import get_settings
from generation.groq_client import get_groq_client
from retrieval.bm25_index import BM25Index
from retrieval.reranker import get_reranker
from utils.semantic_cache import SemanticCache
from utils.valkey_client import close_valkey_client, create_valkey_client
from vectorstore.embedder import get_embeddings
from vectorstore.store import get_all_documents, get_vector_store
from utils.bm25_sync import start_bm25_subscriber, stop_bm25_subscriber
from utils.crag_cache import CRAGCache
from retrieval.web_search import get_web_search_provider

logger = logging.getLogger(__name__)


# ── Lifespan — startup & shutdown ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise all heavy singleton resources exactly once at startup.
    Resources are attached to app.state and shared across all requests.
    """
    settings = get_settings()
    setup_logging()

    logger.info("═══ RAG Service starting up ═══")

    # Ensure required directories exist
    os.makedirs(settings.docs_temp_dir, exist_ok=True)

    # ── Embedding model ───────────────────────────────────────────────────────
    logger.info("Loading embedding model: %s", settings.embed_model)
    embeddings = get_embeddings(settings.embed_model)

    # ── Pinecone ──────────────────────────────────────────────────────────────
    logger.info("Connecting to Pinecone (index=%s)...", settings.pinecone_index)
    store = get_vector_store(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        embeddings=embeddings,
    )

    # ── BM25 index (warm-start from existing corpus) ──────────────────────────
    logger.info("Building BM25 index from existing corpus...")
    bm25_index = BM25Index()
    corpus = get_all_documents(store)
    bm25_index.build(
        ids=corpus.get("ids", []),
        documents=corpus.get("documents", []),
        metadatas=corpus.get("metadatas", []),
    )
    logger.info("BM25 ready: %d documents indexed", bm25_index.document_count)

    # ── Groq LLM ──────────────────────────────────────────────────────────────
    logger.info("Initialising Groq client: model=%s", settings.groq_model)
    llm = get_groq_client(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        max_tokens=settings.groq_max_tokens,
        temperature=settings.groq_temperature,
    )

    # ── Valkey client ─────────────────────────────────────────────────────────
    logger.info("Connecting to Valkey at %s:%d ...", settings.valkey_host, settings.valkey_port)
    valkey_client = await create_valkey_client(
        host=settings.valkey_host,
        port=settings.valkey_port,
        db=settings.valkey_db,
        password=settings.valkey_password or None,
    )

    # ── Semantic cache ────────────────────────────────────────────────────────
    logger.info(
        "Initialising semantic cache (threshold=%.2f ttl=%ds)...",
        settings.semantic_cache_threshold,
        settings.cache_ttl_seconds,
    )
    cache = SemanticCache(
        valkey_client=valkey_client,
        embeddings=embeddings,
        threshold=settings.semantic_cache_threshold,
        ttl=settings.cache_ttl_seconds,
    )
    await cache.ensure_index()
    cache_entries = await cache.async_size()
    logger.info("Semantic cache ready: HNSW index active, %d entries in Valkey", cache_entries)

    # ── Cross-encoder reranker ───────────────────────────────────────────────
    reranker = None
    if settings.reranker_enabled:
        logger.info("Loading cross-encoder reranker: %s", settings.reranker_model)
        reranker = get_reranker(settings.reranker_model)
        logger.info("Cross-encoder reranker ready")
    else:
        logger.info("Cross-encoder reranker is disabled")

    # ── Attach to app state ───────────────────────────────────────────────────
    app.state.store = store
    app.state.bm25_index = bm25_index
    app.state.llm = llm
    app.state.cache = cache
    app.state.valkey_client = valkey_client
    app.state.reranker = reranker
    
    # ── CRAG components ───────────────────────────────────────────────────────
    app.state.crag_cache = CRAGCache(
        valkey_client=valkey_client,
        grade_ttl=settings.crag_grade_ttl_seconds,
        web_ttl=settings.crag_web_ttl_seconds,
    )
    app.state.web_search = get_web_search_provider(settings.web_search_provider,api_key=settings.tavily_api_key,)
    logger.info(
        "CRAG ready: enabled=%s provider=%s",
        settings.crag_enabled, app.state.web_search.name,
    )
    
    # ── Cross-replica BM25 sync (Valkey pub/sub) ──────────────────────────────
    # Each replica gets a unique id so it can ignore its own reload events.
    app.state.instance_id = uuid.uuid4().hex
    app.state.bm25_sub_task, app.state.bm25_pubsub = await start_bm25_subscriber(
        valkey_client=valkey_client,
        store=store,
        bm25_index=bm25_index,
        self_id=app.state.instance_id,
    )

    logger.info("═══ RAG Service is ready ═══")
    yield  # ← application runs here

    logger.info("═══ RAG Service shutting down ═══")
    await stop_bm25_subscriber(
        getattr(app.state, "bm25_sub_task", None),
        getattr(app.state, "bm25_pubsub", None),
    )
    await close_valkey_client(valkey_client)


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Production RAG API",
        description=(
            "Hybrid BM25 + Semantic search RAG system with Reciprocal Rank Fusion.\n\n"
            "Supports any document type: Python, JS, Java, Go, SQL, shell scripts, "
            "Markdown, PDFs, YAML, JSON, CSV, HTML, and more.\n\n"
            "**Authentication:** All endpoints (except `/health`) require the "
            "`X-API-Key` header.\n\n"
            "**Rate limit:** 10 requests / minute per IP on the `/query` endpoint."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Rate limiter error handler ─────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Middleware — outermost first ───────────────────────────────────────────
    # If origins is the wildcard, credentials MUST be disabled (browser + spec rule).
    _origins = settings.cors_origins_list
    _allow_credentials = "*" not in _origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=_allow_credentials,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type"],
    )
    app.add_middleware(RequestLoggerMiddleware)
    app.add_middleware(APIKeyMiddleware, valid_keys=settings.valid_api_keys)

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(health.router)                              # GET  /health
    app.include_router(ingest.router,    prefix="/api/v1")         # POST /api/v1/ingest
    app.include_router(query.router,     prefix="/api/v1")         # POST /api/v1/query
    app.include_router(documents.router, prefix="/api/v1")         # GET/DELETE /api/v1/documents

    return app


app = create_app()


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # reload=True only for local dev; never in production
        workers=1,
        log_level="info",
    )
