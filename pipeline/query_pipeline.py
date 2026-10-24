"""
Query pipeline: sanitise → semantic cache → hybrid search → rerank → generate.

This is the only entry-point that orchestrates retrieval and generation.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from config.settings import Settings
from generation.groq_client import generate_answer, stream_answer
from generation.prompt_builder import build_context, build_messages
from pipeline.crag_graph import build_crag_graph
from pipeline.query_router import route_query
from retrieval.bm25_index import BM25Index
from retrieval.hybrid_search import HybridResult, hybrid_search
from retrieval.reranker import RerankedResult, rerank
from retrieval.web_search import WebSearchProvider
from utils.crag_cache import CRAGCache
from utils.observability import SpanMetrics, estimate_tokens, rag_span
from utils.semantic_cache import SemanticCache
from utils.text_cleaner import sanitize_query

logger = logging.getLogger(__name__)

#compile the crag langgraph once at import time
_CRAG_GRAPH=build_crag_graph()

async def run_query(
    raw_query: str,
    store: object,
    bm25_index: BM25Index,
    llm: ChatGroq,
    cache: SemanticCache,
    settings: Settings,
    reranker: Optional[CrossEncoder] = None,
    request_id: str = "",
    crag_cache: Optional[CRAGCache]=None,
    web_search:Optional[WebSearchProvider]=None,
) -> dict:
    """
    Full RAG query pipeline with semantic caching, cross-encoder reranking,
    and per-stage observability metrics.

    Returns:
        dict with keys: query, answer, sources, cached, cache_type, cache_sim, metrics.
    """
    pipeline_start = time.perf_counter()
    metrics = SpanMetrics(request_id=request_id, query=raw_query)

    # ── 1. Sanitise ───────────────────────────────────────────────────────────
    with rag_span("sanitise", metrics, "sanitise_ms"):
        query = sanitize_query(raw_query)
    
    # ── 1b. LLM router: greet / chitchat / rag ────────────────────────────────
    # Skips retrieval entirely for greetings & small talk.
    with rag_span("router", metrics, "cache_lookup_ms"):
        decision = await route_query(query, llm)
    metrics.extras["intent"] = decision.intent

    if decision.intent != "rag":
        metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info("pipeline_metrics=%s", metrics.to_log_dict())
        return {
            "query":      query,
            "answer":     decision.reply,
            "sources":    [],
            "cached":     False,
            "cache_type": None,
            "cache_sim":  None,
            "metrics":    metrics.to_log_dict(),
        }

    # ── 2. Semantic cache check (Valkey HNSW KNN) ─────────────────────────────
    # Guarded: a cache/Valkey outage must degrade to "skip cache, still answer",
    # never fail the whole request.
    cached_result = None
    with rag_span("cache_lookup", metrics, "cache_lookup_ms"):
        try:
            cached_result = await cache.get(query)
        except Exception as exc:
            metrics.extras["cache_degraded"] = True
            logger.warning("Semantic cache lookup failed — degrading: %s", exc)

    if cached_result is not None:
        metrics.cache_hit = True
        metrics.cache_sim = cached_result.get("cache_sim")
        metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info("pipeline_metrics=%s", metrics.to_log_dict())
        cached_result["metrics"] = metrics.to_log_dict()
        return cached_result

    # ── 3. Hybrid retrieval (BM25 + semantic → RRF) ───────────────────────────
    rrf_candidates: int = settings.rrf_top_k if reranker else settings.final_top_k

    with rag_span("hybrid_search", metrics, "bm25_search_ms"):
        rrf_results: list[HybridResult] = hybrid_search(
            query=query,
            store=store,
            bm25_index=bm25_index,
            top_k=settings.top_k,
            rrf_k=settings.rrf_k,
            final_top_k=rrf_candidates,
        )
    metrics.rrf_candidates = len(rrf_results)
    
    
    # ── 3b. CRAG: evaluate → route → refine / web-fallback ────────────────
    #   grade_retrieval
    #     ├─ Correct    → refine                              ──┐
    #     ├─ Incorrect  → web_search → grade_web → (retry|ok)  ├→ finalize
    #     └─ Ambiguous  → refine ⊕ web_search → grade_web ...   ─┘
    #
    # Web loop: if grade_web = Incorrect and attempts < crag_max_web_attempts,
    # rewrite the query with the LLM and search again.
    if settings.crag_enabled:
        with rag_span("crag_graph", metrics, "cache_lookup_ms"):
            final_state = await _CRAG_GRAPH.ainvoke({
                "query":          query,
                "original_query": query,
                "rrf_docs":       rrf_results,
                "llm":            llm,
                "settings":       settings,
                "crag_cache":     crag_cache,
                "web_search":     web_search,
            })
        rrf_results = final_state.get("final_docs", rrf_results)

        metrics.rrf_candidates = len(rrf_results)
        
        metrics.extras["crag_grade"]       = final_state.get("rrf_grade", "")
        metrics.extras["crag_reason"]      = final_state.get("rrf_reason", "")
        metrics.extras["crag_web_grade"]   = final_state.get("web_grade", "")
        metrics.extras["crag_web_attempts"]= final_state.get("web_attempts", 0)
        metrics.extras["crag_trace"]       = final_state.get("trace", [])

    if not rrf_results:
        metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000
        return {
            "query":      query,
            "answer":     (
                "No relevant documents found. "
                "Please ingest documents using the /api/v1/ingest endpoint first."
            ),
            "sources":    [],
            "cached":     False,
            "cache_type": None,
            "cache_sim":  None,
            "metrics":    metrics.to_log_dict(),
        }

    # ── 4. Cross-encoder reranking ────────────────────────────────────────────
    if reranker and settings.reranker_enabled:
        with rag_span("rerank", metrics, "rerank_ms"):
            reranked: list[RerankedResult] = rerank(
                query=query,
                candidates=rrf_results,
                reranker=reranker,
                top_k=settings.final_top_k,
            )
        metrics.reranked_candidates = len(reranked)
        final_results = reranked
        sources = [
            {
                "doc_name":      r.metadata.get("doc_name", "unknown"),
                "file_type":     r.metadata.get("file_type", ""),
                "chunk_index":   r.metadata.get("chunk_index", -1),
                "rrf_score":     round(r.rrf_score, 6),
                "rerank_score":  round(r.rerank_score, 6),
                "bm25_rank":     r.bm25_rank,
                "semantic_rank": r.semantic_rank,
                "found_by":      r.found_by,
                "snippet":       r.content[:200] + ("…" if len(r.content) > 200 else ""),
                'full_text':     r.content,
            }
            for r in reranked
        ]
    else:
        final_results = rrf_results
        sources = [
            {
                "doc_name":      r.metadata.get("doc_name", "unknown"),
                "file_type":     r.metadata.get("file_type", ""),
                "chunk_index":   r.metadata.get("chunk_index", -1),
                "rrf_score":     round(r.rrf_score, 6),
                "rerank_score":  None,
                "bm25_rank":     r.bm25_rank,
                "semantic_rank": r.semantic_rank,
                "found_by":      r.found_by,
                "snippet":       r.content[:200] + ("…" if len(r.content) > 200 else ""),
                'full_text':     r.content,
            }
            for r in final_results
        ]

    # ── 5. Generate ───────────────────────────────────────────────────────────
    messages = build_messages(question=query, results=final_results)
    context_text = build_context(final_results)
    metrics.prompt_tokens_est = estimate_tokens(context_text + query)

    with rag_span("generation", metrics, "generation_ms"):
        answer = await generate_answer(messages=messages, llm=llm)

    metrics.answer_length = len(answer)
    metrics.final_candidates = len(final_results)

    # ── 6. Build response + cache ─────────────────────────────────────────────
    response = {
        "query":      query,
        "answer":     answer,
        "sources":    sources,
        "cached":     False,
        "cache_type": None,
        "cache_sim":  None,
    }

    with rag_span("cache_store", metrics, "cache_store_ms"):
        try:
            await cache.set(query, response)
        except Exception as exc:
            metrics.extras["cache_degraded"] = True
            logger.warning("Semantic cache store failed — continuing: %s", exc)

    metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000
    logger.info("pipeline_metrics=%s", metrics.to_log_dict())
    response["metrics"] = metrics.to_log_dict()

    return response

async def run_query_stream(
    raw_query: str,
    store: object,
    bm25_index: BM25Index,
    llm: ChatGroq,
    cache: SemanticCache,
    settings: Settings,
    reranker: Optional[CrossEncoder] = None,
    request_id: str = "",
    crag_cache: Optional[CRAGCache] = None,
    web_search: Optional[WebSearchProvider] = None,
):
    """
    Streaming variant of run_query — an async generator that yields event dicts:

        {"type": "sources", "sources": [...]}       # emitted once, before tokens
        {"type": "meta",    "cached": bool, ...}     # cache/intent info
        {"type": "token",   "text": "..."}           # many, the streamed answer
        {"type": "done",    "answer": str, "metrics": {...}}
        {"type": "error",   "detail": "..."}

    The route layer serialises each event as Server-Sent Events (SSE).
    """
    pipeline_start = time.perf_counter()
    metrics = SpanMetrics(request_id=request_id, query=raw_query)

    # ── 1. Sanitise ───────────────────────────────────────────────────────────
    try:
        query = sanitize_query(raw_query)
    except ValueError as exc:
        yield {"type": "error", "detail": str(exc)}
        return

    # ── 1b. Router ────────────────────────────────────────────────────────────
    decision = await route_query(query, llm)
    metrics.extras["intent"] = decision.intent
    if decision.intent != "rag":
        yield {"type": "sources", "sources": []}
        yield {"type": "meta", "cached": False, "intent": decision.intent}
        # Stream the canned reply word-by-word for a consistent UX.
        for word in decision.reply.split(" "):
            yield {"type": "token", "text": word + " "}
        metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000
        yield {"type": "done", "answer": decision.reply, "metrics": metrics.to_log_dict()}
        return

    # ── 2. Semantic cache (guarded) ───────────────────────────────────────────
    cached_result = None
    try:
        cached_result = await cache.get(query)
    except Exception as exc:
        metrics.extras["cache_degraded"] = True
        logger.warning("Semantic cache lookup failed — degrading: %s", exc)

    if cached_result is not None:
        yield {"type": "sources", "sources": cached_result.get("sources", [])}
        yield {"type": "meta", "cached": True, "cache_sim": cached_result.get("cache_sim")}
        answer = cached_result.get("answer", "")
        # Emit cached answer as a single token (already complete).
        yield {"type": "token", "text": answer}
        metrics.cache_hit = True
        metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000
        yield {"type": "done", "answer": answer, "metrics": metrics.to_log_dict()}
        return

    # ── 3. Hybrid retrieval ───────────────────────────────────────────────────
    rrf_candidates = settings.rrf_top_k if reranker else settings.final_top_k
    rrf_results = hybrid_search(
        query=query, store=store, bm25_index=bm25_index,
        top_k=settings.top_k, rrf_k=settings.rrf_k, final_top_k=rrf_candidates,
    )

    # ── 3b. CRAG ──────────────────────────────────────────────────────────────
    if settings.crag_enabled:
        final_state = await _CRAG_GRAPH.ainvoke({
            "query": query, "original_query": query, "rrf_docs": rrf_results,
            "llm": llm, "settings": settings, "crag_cache": crag_cache,
            "web_search": web_search,
        })
        rrf_results = final_state.get("final_docs", rrf_results)
        metrics.extras["crag_grade"] = final_state.get("rrf_grade", "")
        metrics.extras["crag_web_attempts"] = final_state.get("web_attempts", 0)

    if not rrf_results:
        msg = ("No relevant documents found. Please ingest documents using the "
               "/api/v1/ingest endpoint first.")
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": msg}
        metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000
        yield {"type": "done", "answer": msg, "metrics": metrics.to_log_dict()}
        return

    # ── 4. Rerank ─────────────────────────────────────────────────────────────
    if reranker and settings.reranker_enabled:
        final_results = rerank(
            query=query, candidates=rrf_results, reranker=reranker,
            top_k=settings.final_top_k,
        )
    else:
        final_results = rrf_results

    sources = [
        {
            "doc_name":      r.metadata.get("doc_name", "unknown"),
            "file_type":     r.metadata.get("file_type", ""),
            "chunk_index":   r.metadata.get("chunk_index", -1),
            "rrf_score":     round(r.rrf_score, 6),
            "rerank_score":  round(getattr(r, "rerank_score", None), 6) if getattr(r, "rerank_score", None) is not None else None,
            "bm25_rank":     r.bm25_rank,
            "semantic_rank": r.semantic_rank,
            "found_by":      r.found_by,
            "snippet":       r.content[:200] + ("…" if len(r.content) > 200 else ""),
            "full_text":     r.content,
        }
        for r in final_results
    ]

    # Emit sources first so the UI can render citations while tokens stream.
    yield {"type": "sources", "sources": sources}
    yield {"type": "meta", "cached": False, "intent": "rag"}

    # ── 5. Stream generation ──────────────────────────────────────────────────
    messages = build_messages(question=query, results=final_results)
    context_text = build_context(final_results)
    metrics.prompt_tokens_est = estimate_tokens(context_text + query)

    parts: list[str] = []
    gen_start = time.perf_counter()
    try:
        async for token in stream_answer(messages=messages, llm=llm):
            parts.append(token)
            yield {"type": "token", "text": token}
    except Exception as exc:
        logger.error("Streaming generation error: %s", exc, exc_info=True)
        yield {"type": "error", "detail": "Generation failed. Please try again."}
        return

    answer = "".join(parts).strip()
    metrics.generation_ms = (time.perf_counter() - gen_start) * 1000
    metrics.answer_length = len(answer)
    metrics.final_candidates = len(final_results)
    metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000

    # ── 6. Cache the full answer (guarded) ────────────────────────────────────
    response = {
        "query": query, "answer": answer, "sources": sources,
        "cached": False, "cache_type": None, "cache_sim": None,
    }
    try:
        await cache.set(query, response)
    except Exception as exc:
        metrics.extras["cache_degraded"] = True
        logger.warning("Semantic cache store failed — continuing: %s", exc)

    logger.info("pipeline_metrics=%s", metrics.to_log_dict())
    yield {"type": "done", "answer": answer, "metrics": metrics.to_log_dict()}
