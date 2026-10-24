"""
Semantic Query Cache — Valkey-native HNSW Vector Search.

Previous approach (in-memory NumPy):
  ❌ Cache vectors isolated per worker instance
  ❌ Cannot share state across replicas
  ❌ Lost on restart until warm_up rebuilds

Current approach (Valkey 8+ Vector Search):
  ✅ All similarity search runs INSIDE Valkey via FT.SEARCH KNN
  ✅ Every replica sees the same index — no per-worker isolation
  ✅ Scales horizontally without any in-memory coordination
  ✅ HNSW index automatically maintained by Valkey on insert/delete

How it works
------------
1.  FT.CREATE — called once at startup to ensure the HNSW index exists on the
    Valkey HASH namespace ``rag:semcache:*``.  Fields indexed:
      • ``embedding``  VECTOR HNSW  (dim=384, COSINE distance)
      • ``cached_at``  NUMERIC      (for TTL housekeeping queries)
2.  On cache SET: store a HASH with query, answer, sources (JSON), embedding
    (raw bytes), cached_at.  Apply EXPIREAT for TTL eviction.
3.  On cache GET: embed the incoming query → FT.SEARCH with KNN top-1.
    If distance < (1 − threshold) → cache HIT.  Valkey COSINE distance =
    1 − cosine_similarity, so threshold=0.92 → max_distance=0.08.
4.  On cache CLEAR: drop & recreate the index.

No NumPy, no asyncio.Lock, no in-memory state.  The only state is inside Valkey.
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

_INDEX_NAME = "idx:semcache"
_KEY_PREFIX = "rag:semcache:"
_EMBED_DIM = 384  # all-MiniLM-L6-v2


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _vec_to_bytes(vec: list[float]) -> bytes:
    """Pack a float32 list into raw little-endian bytes for Valkey VECTOR field."""
    return struct.pack(f"<{len(vec)}f", *vec)



# ── Main class ────────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Valkey-native HNSW vector search cache.

    All similarity computation happens server-side inside Valkey —
    no in-memory NumPy matrix, no per-worker state, fully replica-safe.

    Args:
        valkey_client:  Connected async Valkey client.
        embeddings:     Shared HuggingFaceEmbeddings instance
                        (normalize_embeddings=True so cosine distance is valid).
        threshold:      Min cosine similarity for a cache hit (0.0–1.0).
        ttl:            Seconds before Valkey auto-evicts an entry (EXPIRE).
        embed_dim:      Embedding dimension (default 384 for MiniLM).
    """

    def __init__(
        self,
        valkey_client: Any,
        embeddings: Any,
        threshold: float = 0.92,
        ttl: int = 300,
        embed_dim: int = _EMBED_DIM,
    ) -> None:
        self._client = valkey_client
        self._embeddings = embeddings
        self._threshold = threshold
        self._max_distance = 1.0 - threshold  # COSINE distance = 1 − similarity
        self._ttl = ttl
        self._embed_dim = embed_dim

    # ── Index creation ────────────────────────────────────────────────────────

    async def ensure_index(self) -> None:
        """
        Create the HNSW vector index if it doesn't already exist.
        Safe to call on every startup — Valkey returns an error if it exists
        and we silently ignore it.
        """
        try:
            await self._client.execute_command(
                "FT.CREATE", _INDEX_NAME,
                "ON", "HASH",
                "PREFIX", "1", _KEY_PREFIX,
                "SCHEMA",
                "embedding", "VECTOR", "HNSW", "6",
                    "TYPE", "FLOAT32",
                    "DIM", str(self._embed_dim),
                    "DISTANCE_METRIC", "COSINE",
                "cached_at", "NUMERIC", "SORTABLE",
            )
            logger.info("Valkey HNSW index '%s' created", _INDEX_NAME)
        except Exception as exc:
            msg = str(exc).lower()
            if "index already exists" in msg or "already exists" in msg:
                logger.info("Valkey HNSW index '%s' already exists — reusing", _INDEX_NAME)
            else:
                logger.error("Failed to create Valkey vector index: %s", exc)
                raise

    # ── Get ───────────────────────────────────────────────────────────────────

    async def get(self, query: str) -> Optional[dict]:
        """
        Embed the query and run a KNN-1 vector search inside Valkey.

        Returns cached answer dict on hit (cosine similarity ≥ threshold),
        or None on miss.
        """
        # Embed in thread executor (CPU-bound)
        loop = asyncio.get_event_loop()
        query_vec: list[float] = await loop.run_in_executor(
            None, self._embed, query
        )
        query_bytes = _vec_to_bytes(query_vec)

        try:
            # FT.SEARCH idx:semcache "*=>[KNN 1 @embedding $vec AS dist]"
            # PARAMS 2 vec <bytes> SORTBY dist ASC LIMIT 0 1 DIALECT 2
            raw = await self._client.execute_command(
                "FT.SEARCH", _INDEX_NAME,
                "*=>[KNN 1 @embedding $vec AS dist]",
                "PARAMS", "2", "vec", query_bytes,
                "SORTBY", "dist", "ASC",
                "LIMIT", "0", "1",
                "DIALECT", "2",
            )
        except Exception as exc:
            logger.warning("Valkey FT.SEARCH error — cache miss: %s", exc)
            return None

        # Parse response: [total_results, key, [field, value, ...], ...]
        if not raw or raw[0] == 0:
            return None

        # Extract field-value pairs from the result
        fields = self._parse_ft_result(raw)
        if not fields:
            return None

        distance = float(fields.get("dist", 1.0))
        similarity = 1.0 - distance

        if similarity < self._threshold:
            logger.debug(
                "Semantic cache MISS: sim=%.4f threshold=%.4f query=%r",
                similarity, self._threshold, query[:50],
            )
            return None

        # Reconstruct cached answer
        try:
            cached_query = fields.get("query", "")
            if isinstance(cached_query, bytes):
                cached_query = cached_query.decode("utf-8")
            answer = fields.get("answer", "")
            if isinstance(answer, bytes):
                answer = answer.decode("utf-8")
            sources_raw = fields.get("sources", "[]")
            if isinstance(sources_raw, bytes):
                sources_raw = sources_raw.decode("utf-8")
            sources = json.loads(sources_raw)
        except Exception as exc:
            logger.warning("Failed to parse cached entry: %s", exc)
            return None

        logger.info(
            "Semantic cache HIT: sim=%.4f matched=%r query=%r",
            similarity, cached_query[:50], query[:50],
        )
        return {
            "query":      cached_query,
            "answer":     answer,
            "sources":    sources,
            "cached":     True,
            "cache_type": "semantic",
            "cache_sim":  round(similarity, 4),
        }

    # ── Set ───────────────────────────────────────────────────────────────────

    async def set(self, query: str, result: dict) -> None:
        """
        Store a pipeline result as a Valkey HASH with an embedded vector.
        The HNSW index is automatically updated by Valkey.
        """
        loop = asyncio.get_event_loop()
        query_vec: list[float] = await loop.run_in_executor(
            None, self._embed, query
        )
        embedding_bytes = _vec_to_bytes(query_vec)

        entry_key = f"{_KEY_PREFIX}{uuid.uuid4().hex}"
        sources_json = json.dumps(result.get("sources", []), ensure_ascii=False)

        try:
            await self._client.hset(
                entry_key,
                mapping={
                    "query":     query,
                    "answer":    result.get("answer", ""),
                    "sources":   sources_json,
                    "embedding": embedding_bytes,
                    "cached_at": str(time.time()),
                },
            )
            # Auto-evict after TTL
            await self._client.expire(entry_key, self._ttl)
        except Exception as exc:
            logger.warning("Valkey HSET error — result NOT cached: %s", exc)
            return

        logger.debug("Semantic cache SET: key=%s query=%r", entry_key, query[:50])

    # ── Clear ─────────────────────────────────────────────────────────────────

    async def clear(self) -> int:
        """Drop the HNSW index + delete all cache keys, then recreate the index."""
        count = 0
        try:
            # Scan and delete all cache keys
            cursor = 0
            while True:
                cursor, keys = await self._client.scan(
                    cursor, match=f"{_KEY_PREFIX}*", count=200
                )
                if keys:
                    await self._client.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break

            # Drop and recreate the index
            try:
                await self._client.execute_command("FT.DROPINDEX", _INDEX_NAME)
            except Exception:
                pass
            await self.ensure_index()
        except Exception as exc:
            logger.warning("Semantic cache clear error: %s", exc)

        logger.info("Semantic cache cleared: %d entries removed", count)
        return count

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Approximate entry count from the FT.INFO num_docs field."""
        try:
            # Use synchronous fallback for property
            import asyncio as _aio
            loop = _aio.get_event_loop()
            if loop.is_running():
                return self._size_sync()
            return loop.run_until_complete(self._async_size())
        except Exception:
            return -1

    def _size_sync(self) -> int:
        """Fallback — return -1 when we can't run async from a property."""
        return -1  # caller should use async_size() in async context

    async def async_size(self) -> int:
        """Get entry count from Valkey FT.INFO (async-safe)."""
        try:
            info = await self._client.execute_command("FT.INFO", _INDEX_NAME)
            # FT.INFO returns a flat list: [..., "num_docs", "42", ...]
            if isinstance(info, list):
                for i, item in enumerate(info):
                    val = item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    if val == "num_docs" and i + 1 < len(info):
                        nxt = info[i + 1]
                        return int(nxt.decode("utf-8") if isinstance(nxt, bytes) else nxt)
            return 0
        except Exception:
            return 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def ttl(self) -> int:
        return self._ttl

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Synchronously embed a single string. Called via run_in_executor."""
        return self._embeddings.embed_documents([text])[0]

    @staticmethod
    def _parse_ft_result(raw: list) -> dict:
        """
        Parse the flat field-value list returned by FT.SEARCH into a dict.
        FT.SEARCH returns: [total, key1, [f1, v1, f2, v2, ...], ...]
        """
        if len(raw) < 3:
            return {}

        pairs = raw[2]
        if not isinstance(pairs, (list, tuple)):
            return {}

        result = {}
        for i in range(0, len(pairs) - 1, 2):
            key = pairs[i]
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            result[key] = pairs[i + 1]
        return result
