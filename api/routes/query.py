"""
Query route — POST /api/v1/query

Accepts a question, runs the hybrid RAG pipeline, and returns a
grounded answer with cited source chunks.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from api.middleware.rate_limiter import limiter
from api.schemas.request import QueryRequest,TitleRequest
from api.schemas.response import QueryResponse, SourceChunk
from config.settings import get_settings
from pipeline.query_pipeline import run_query,run_query_stream

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG system",
)
@limiter.limit("10/minute")
async def query_documents(
    request: Request,   # required first arg when using slowapi limiter
    body: QueryRequest,
) -> QueryResponse:
    """
    Ask a question and receive an answer grounded in the ingested documents.

    The pipeline:
    1. Sanitises and validates the query.
    2. Checks the TTL response cache.
    3. Runs BM25 keyword search + semantic vector search in parallel.
    4. Fuses results with Reciprocal Rank Fusion (RRF).
    5. Builds a grounded prompt and calls Groq (llama-3.1-8b-instant).
    6. Returns the answer with cited source chunks.

    Rate-limited to **10 requests per minute per IP**.
    Cached responses (same query within TTL window) are returned instantly
    and do **not** count against the Groq API quota.
    """
    settings = get_settings()

    # Allow per-request top_k override
    if body.top_k is not None:
        settings = settings.model_copy(update={"final_top_k": body.top_k})

    try:
        result = await run_query(
            raw_query=body.query,
            store=request.app.state.store,
            bm25_index=request.app.state.bm25_index,
            llm=request.app.state.llm,
            cache=request.app.state.cache,
            settings=settings,
            reranker=getattr(request.app.state, "reranker", None),
            request_id=getattr(request.state, "request_id", ""),
        )
    except ValueError as exc:
        # Sanitisation / injection errors → 422
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Query pipeline error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your query. Please try again.",
        )

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        cached=result["cached"],
        cache_type=result.get("cache_type"),
        cache_sim=result.get("cache_sim"),
        metrics=result.get("metrics"),
    )

@router.post(
    "/query/stream",
    summary="Query the RAG system with token streaming (SSE)",
)
@limiter.limit("10/minute")
async def query_documents_stream(
    request: Request,
    body: QueryRequest,
) -> StreamingResponse:
    """
    Same pipeline as /query but streams the answer token-by-token using
    Server-Sent Events (SSE). Each line is `data: <json>\\n\\n` where the JSON
    is one of: sources | meta | token | done | error.
    """
    settings = get_settings()
    if body.top_k is not None:
        settings = settings.model_copy(update={"final_top_k": body.top_k})

    async def event_generator():
        try:
            async for event in run_query_stream(
                raw_query=body.query,
                store=request.app.state.store,
                bm25_index=request.app.state.bm25_index,
                llm=request.app.state.llm,
                cache=request.app.state.cache,
                settings=settings,
                reranker=getattr(request.app.state, "reranker", None),
                request_id=getattr(request.state, "request_id", ""),
                crag_cache=getattr(request.app.state, "crag_cache", None),
                web_search=getattr(request.app.state, "web_search", None),
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error("Streaming pipeline error: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'detail': 'An error occurred.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # disable proxy buffering (nginx)
        },
    )


@router.post(
    "/title",
    summary="Generate a short chat title",
)
@limiter.limit("30/minute")
async def generate_title(request: Request, body: TitleRequest) -> dict:
    """
    Produce a concise 3–6 word title for a conversation, given the first
    user message (and optionally the first answer).

    This mirrors how ChatGPT/Claude auto-name chats: a single, cheap LLM call.
    Rate-limited and Valkey-cached (keyed by the first message) so repeated
    calls for the same conversation don't re-hit the LLM. On any failure it
    falls back to a trimmed version of the user's question.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    fallback = body.query.strip()[:48]

    # ── Cache lookup (keyed by first message) ─────────────────────────────────
    crag_cache = getattr(request.app.state, "crag_cache", None)
    if crag_cache is not None:
        cached_title = await crag_cache.get_title(body.query)
        if cached_title:
            return {"title": cached_title, "cached": True}

    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        return {"title": fallback}

    system = (
        "You generate very short conversation titles. Given the user's first "
        "message, reply with a concise title of 3 to 6 words. No quotes, no "
        "punctuation at the end, no prefixes like 'Title:'. Just the title."
    )
    user = f"First message:\n{body.query}"
    if body.answer:
        user += f"\n\nAssistant answer (context):\n{body.answer[:500]}"

    try:
        resp = await llm.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        title = (resp.content or "").strip().strip('"').strip()
        # Keep it tidy: single line, bounded length.
        title = title.splitlines()[0][:60] if title else fallback
        title = title or fallback
    except Exception as exc:
        logger.warning("Title generation failed (%s) — using fallback", exc)
        title = fallback

    if crag_cache is not None and title:
        await crag_cache.set_title(body.query, title)

    return {"title": title}
