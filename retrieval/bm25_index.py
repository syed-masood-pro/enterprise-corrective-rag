"""
In-memory BM25 keyword search index.

The index is built from the Pinecone corpus at application startup and
rebuilt in memory after every ingestion. No separate file persistence is
needed because Pinecone is the source of truth.

Thread-safety:  All public methods are protected by a reentrant lock.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """A single BM25 search hit."""

    chunk_id: str
    content: str
    metadata: dict
    score: float


class BM25Index:
    """Thread-safe in-memory BM25 index backed by a Pinecone corpus."""

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._lock = threading.RLock()

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """
        (Re)build the BM25 index from a corpus.

        Call at application startup and after every document ingestion.

        Args:
            ids:       Pinecone document IDs (== chunk_ids).
            documents: Raw text of each document.
            metadatas: Metadata dicts parallel to ids / documents.
        """
        with self._lock:
            if not documents:
                logger.warning("BM25 build called with an empty corpus")
                self._bm25 = None
                self._ids = []
                self._documents = []
                self._metadatas = []
                return

            self._ids = list(ids)
            self._documents = list(documents)
            self._metadatas = list(metadatas) if metadatas else [{} for _ in ids]

            tokenized = [doc.lower().split() for doc in documents]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("BM25 index built with %d documents", len(documents))

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        """
        Score all documents against *query* and return the top *top_k* hits.

        Returns:
            List of BM25Result sorted by score descending.
            Empty list when the index has not been built yet.
        """
        with self._lock:
            if self._bm25 is None or not self._documents:
                logger.warning("BM25 search called on an empty index")
                return []

            scores = self._bm25.get_scores(query.lower().split())

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:top_k]

            results: list[BM25Result] = []
            for i in ranked_indices:
                if scores[i] <= 0:
                    continue  # zero-score docs add no signal
                results.append(
                    BM25Result(
                        chunk_id=self._ids[i],
                        content=self._documents[i],
                        metadata=self._metadatas[i],
                        score=float(scores[i]),
                    )
                )

            return results

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def document_count(self) -> int:
        with self._lock:
            return len(self._documents)

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._bm25 is not None and len(self._documents) > 0
