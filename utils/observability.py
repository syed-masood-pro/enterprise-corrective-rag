"""
Observability middleware using OpenTelemetry-compatible custom spans.

Provides:
  - Detailed per-stage timing for the RAG pipeline
  - Token count tracking (estimated)
  - Retrieval quality metrics
  - All data exposed via the /health endpoint and structured logs

Usage: wrap any async/sync function call with `with rag_span(...):`
to get structured timing in logs without external trace collector dependency.

When OpenTelemetry collector IS configured:
  - Set OTEL_EXPORTER_OTLP_ENDPOINT in .env
  - Uncomment the opentelemetry lines in requirements.txt
  - Spans automatically flow to Jaeger, Zipkin, or any OTLP-compatible backend
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SpanMetrics:
    """Collected metrics for a single pipeline execution."""

    request_id: str = ""
    query: str = ""

    # Timing (seconds)
    sanitise_ms: float = 0.0
    cache_lookup_ms: float = 0.0
    bm25_search_ms: float = 0.0
    semantic_search_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    cache_store_ms: float = 0.0
    total_ms: float = 0.0

    # Counts
    bm25_candidates: int = 0
    semantic_candidates: int = 0
    rrf_candidates: int = 0
    reranked_candidates: int = 0
    final_candidates: int = 0

    # LLM
    prompt_tokens_est: int = 0   # estimated
    answer_length: int = 0

    # Cache
    cache_hit: bool = False
    cache_sim: Optional[float] = None

    # Extras
    extras: dict = field(default_factory=dict)

    def to_log_dict(self) -> dict:
        """Return a flat dict suitable for structured logging."""
        d = {
            "request_id": self.request_id,
            "query_preview": self.query[:60],
            "total_ms": round(self.total_ms, 1),
            "cache_hit": self.cache_hit,
        }
        if self.cache_hit:
            d["cache_sim"] = self.cache_sim
        else:
            d.update({
                "sanitise_ms":          round(self.sanitise_ms, 1),
                "cache_lookup_ms":      round(self.cache_lookup_ms, 1),
                "bm25_search_ms":       round(self.bm25_search_ms, 1),
                "semantic_search_ms":   round(self.semantic_search_ms, 1),
                "rrf_fusion_ms":        round(self.rrf_fusion_ms, 1),
                "rerank_ms":            round(self.rerank_ms, 1),
                "generation_ms":        round(self.generation_ms, 1),
                "bm25_candidates":      self.bm25_candidates,
                "semantic_candidates":  self.semantic_candidates,
                "rrf_candidates":       self.rrf_candidates,
                "reranked_candidates":  self.reranked_candidates,
                "prompt_tokens_est":    self.prompt_tokens_est,
                "answer_length":        self.answer_length,
            })
        d.update(self.extras)
        return d


@contextmanager
def rag_span(name: str, metrics: SpanMetrics, field: str):
    """
    Context manager that times a code block and stores the elapsed
    time (in ms) in the given SpanMetrics field.

    Usage:
        m = SpanMetrics()
        with rag_span("bm25_search", m, "bm25_search_ms"):
            results = bm25_index.search(query, top_k=10)

    Args:
        name:    Human-readable span label (for logging).
        metrics: SpanMetrics instance to record into.
        field:   Attribute name on SpanMetrics to store elapsed ms.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        setattr(metrics, field, elapsed_ms)
        logger.debug("span=%s elapsed=%.1fms", name, elapsed_ms)


def estimate_tokens(text: str) -> int:
    """Rough token count estimate: ~4 chars per token for English text."""
    return len(text) // 4
