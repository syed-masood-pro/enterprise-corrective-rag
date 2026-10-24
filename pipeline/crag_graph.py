"""
CRAG orchestration as a LangGraph state machine.

Nodes:
    grade_retrieval  → grade the RRF docs (Correct | Incorrect | Ambiguous)
    refine           → knowledge refinement on RRF docs
    web_search       → Tavily (or configured provider) fallback
    grade_web        → grade the web docs (Correct | Incorrect | Ambiguous)
    rewrite_query    → rewrite query and loop back to web_search
    finalize         → combine refined + web docs into `final_docs`

Edges (routing):

                       ┌──────────────────┐
    (start) ──────────▶│ grade_retrieval  │
                       └────────┬─────────┘
             Correct     Ambiguous       Incorrect
                │           │                │
                ▼           ▼                ▼
             ┌──────┐   ┌──────┐        ┌────────────┐
             │refine│   │refine│  ┌────▶│ web_search │
             └──┬───┘   └──┬───┘  │     └─────┬──────┘
                │          │      │           ▼
                │          │      │      ┌───────────┐
                │          │      │      │ grade_web │
                │          │      │      └─────┬─────┘
                │          │      │            │
                │          │      │  Incorrect + attempts<max
                │          │      │            ▼
                │          │      │     ┌────────────────┐
                │          │      └─────┤ rewrite_query  │
                │          │            └────────────────┘
                │          │                  (loop)
                │          ▼                    │  Correct/Ambiguous/max
                │       (web_search chain — same as Incorrect but result is appended)
                │
                └────────────────────▶ finalize ▶ (end)
"""
from __future__ import annotations

import logging
from typing import Optional, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from config.settings import Settings
from retrieval.crag_evaluator import evaluate_retrieval
from retrieval.hybrid_search import HybridResult
from retrieval.knowledge_refiner import refine_documents
from retrieval.query_rewriter import rewrite_for_web
from retrieval.web_search import WebSearchProvider
from utils.crag_cache import CRAGCache

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────────────────────

class CRAGState(TypedDict, total=False):
    # Inputs
    query: str
    original_query: str
    rrf_docs: list[HybridResult]

    # Runtime deps (passed through — not part of persisted state semantics)
    llm: ChatGroq
    settings: Settings
    crag_cache: Optional[CRAGCache]
    web_search: Optional[WebSearchProvider]

    # Working values
    rrf_grade: str            # Correct | Incorrect | Ambiguous
    rrf_reason: str
    refined_docs: list[HybridResult]
    web_query: str            # possibly-rewritten query used for web search
    web_docs: list[HybridResult]
    web_grade: str
    web_reason: str
    web_attempts: int

    # Output
    final_docs: list[HybridResult]
    trace: list[str]          # human-readable event log for observability


# ── Helper ───────────────────────────────────────────────────────────────────

def _log(state: CRAGState, msg: str) -> None:
    state.setdefault("trace", []).append(msg)
    logger.info("CRAG-graph | %s", msg)


# ── Nodes ────────────────────────────────────────────────────────────────────

async def node_grade_retrieval(state: CRAGState) -> CRAGState:
    query = state["query"]
    docs = state.get("rrf_docs", [])
    llm: ChatGroq = state["llm"]
    cache: Optional[CRAGCache] = state.get("crag_cache")
    settings: Settings = state["settings"]

    # ── Heuristic fast-path (no LLM call) ─────────────────────────────────────
    # If both retrievers independently surfaced the top document AND we have a
    # healthy number of candidates, the retrieval is very likely Correct. This
    # skips the grader LLM round-trip for the common "good retrieval" case.
    if (
        settings.crag_grade_heuristic_enabled
        and len(docs) >= settings.crag_grade_min_docs
        and docs[0].found_by == "both"
    ):
        _log(state, "grade_retrieval: HEURISTIC Correct (top doc found by both retrievers)")
        return {"rrf_grade": "Correct", "rrf_reason": "heuristic: retriever agreement on top doc"}

    cached = await cache.get_grade(query) if cache else None
    if cached:
        grade, reason = cached
        _log(state, f"grade_retrieval: CACHE_HIT {grade} ({reason})")
    else:
        grade, reason = await evaluate_retrieval(query, docs, llm)
        if cache:
            await cache.set_grade(query, grade, reason)
        _log(state, f"grade_retrieval: {grade} ({reason})")

    return {"rrf_grade": grade, "rrf_reason": reason}


async def node_refine(state: CRAGState) -> CRAGState:
    docs = refine_documents(state["query"], state.get("rrf_docs", []))
    _log(state, f"refine: kept {len(docs)} refined docs")
    return {"refined_docs": docs}


async def node_web_search(state: CRAGState) -> CRAGState:
    provider = state.get("web_search")
    settings: Settings = state["settings"]
    cache: Optional[CRAGCache] = state.get("crag_cache")
    q = state.get("web_query") or state["query"]
    attempts = state.get("web_attempts", 0) + 1

    if not provider:
        _log(state, "web_search: no provider configured — skipping")
        return {"web_docs": [], "web_attempts": attempts}

    # Cache lookup only on the FIRST attempt (rewritten queries are fresh)
    if attempts == 1 and cache:
        cached_web = await cache.get_web(q)
        if cached_web is not None:
            docs = [
                HybridResult(
                    chunk_id=d["chunk_id"],
                    content=d["content"],
                    metadata=d.get("metadata", {}),
                    rrf_score=d.get("rrf_score", 0.0),
                )
                for d in cached_web
            ]
            _log(state, f"web_search: CACHE_HIT ({len(docs)} docs)")
            return {"web_docs": docs, "web_attempts": attempts, "web_query": q}

    docs = await provider.search(q, top_k=settings.web_search_top_k)
    if cache and attempts == 1:
        await cache.set_web(q, [
            {"chunk_id": d.chunk_id, "content": d.content,
             "metadata": d.metadata, "rrf_score": d.rrf_score}
            for d in docs
        ])
    _log(state, f"web_search (attempt {attempts}) query={q!r} -> {len(docs)} docs")
    return {"web_docs": docs, "web_attempts": attempts, "web_query": q}


async def node_grade_web(state: CRAGState) -> CRAGState:
    llm: ChatGroq = state["llm"]
    docs = state.get("web_docs", [])
    grade, reason = await evaluate_retrieval(state["query"], docs, llm)
    _log(state, f"grade_web: {grade} ({reason})")
    return {"web_grade": grade, "web_reason": reason}


async def node_rewrite_query(state: CRAGState) -> CRAGState:
    llm: ChatGroq = state["llm"]
    prior = state.get("web_query", "")
    new_q = await rewrite_for_web(state["original_query"], llm, prior_attempt=prior)
    _log(state, f"rewrite_query: {new_q!r}")
    return {"web_query": new_q}


async def node_finalize(state: CRAGState) -> CRAGState:
    rrf_grade = state.get("rrf_grade", "")
    refined = state.get("refined_docs", [])
    web = state.get("web_docs", [])

    if rrf_grade == "Correct":
        final = refined
    elif rrf_grade == "Incorrect":
        final = web
    else:  # Ambiguous
        final = refined + web

    _log(state, f"finalize: {len(final)} docs (rrf_grade={rrf_grade})")
    return {"final_docs": final}


# ── Routers ──────────────────────────────────────────────────────────────────

def route_after_grade(state: CRAGState) -> str:
    grade = state.get("rrf_grade", "Ambiguous")
    if grade == "Correct":
        return "refine_only"
    if grade == "Incorrect":
        return "web_only"
    return "refine_and_web"


def route_after_web_grade(state: CRAGState) -> str:
    settings: Settings = state["settings"]
    attempts = state.get("web_attempts", 0)
    grade = state.get("web_grade", "Ambiguous")

    # Retry web search with a rewritten query if it was clearly bad
    if grade == "Incorrect" and attempts < settings.crag_max_web_attempts:
        return "retry"
    return "finalize"


# ── Graph builder ────────────────────────────────────────────────────────────

def build_crag_graph():
    g = StateGraph(CRAGState)

    g.add_node("grade_retrieval", node_grade_retrieval)
    g.add_node("refine", node_refine)
    g.add_node("web_search", node_web_search)
    g.add_node("grade_web", node_grade_web)
    g.add_node("rewrite_query", node_rewrite_query)
    g.add_node("refine_for_ambiguous", node_refine)  # separate instance for branching clarity
    g.add_node("finalize", node_finalize)

    g.set_entry_point("grade_retrieval")

    # After grading the retrieval, branch based on grade
    g.add_conditional_edges(
        "grade_retrieval",
        route_after_grade,
        {
            "refine_only": "refine",
            "web_only": "web_search",
            "refine_and_web": "refine_for_ambiguous",
        },
    )

    # Correct → refine → finalize
    g.add_edge("refine", "finalize")

    # Ambiguous → refine_for_ambiguous → web_search
    g.add_edge("refine_for_ambiguous", "web_search")

    # web_search → grade_web → (retry or finalize)
    g.add_edge("web_search", "grade_web")
    g.add_conditional_edges(
        "grade_web",
        route_after_web_grade,
        {"retry": "rewrite_query", "finalize": "finalize"},
    )
    g.add_edge("rewrite_query", "web_search")

    g.add_edge("finalize", END)
    return g.compile()
