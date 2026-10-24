"""
Cross-replica BM25 synchronisation via Valkey pub/sub.

Problem
-------
The BM25 keyword index lives *in memory* inside each replica and is rebuilt
from the Pinecone corpus at startup. When one replica ingests a new document,
only THAT replica's BM25 index updates — every other replica keeps serving
keyword search against a stale corpus until it restarts.

Solution
--------
After a successful ingest, the receiving replica publishes a lightweight
"reload" event on a shared Valkey channel. Every replica runs a background
subscriber that, on receiving a *peer* event, rebuilds its own BM25 index from
Pinecone (the source of truth). The publishing replica ignores its own event
because it has already rebuilt in-process.

This keeps all replicas eventually-consistent for keyword search with a single
extra pub/sub message per ingest — no shared BM25 service required.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from retrieval.bm25_index import BM25Index
from vectorstore.store import get_all_documents

logger = logging.getLogger(__name__)

BM25_RELOAD_CHANNEL = "rag:bm25:reload"


def rebuild_bm25_from_store(store: Any, bm25_index: BM25Index) -> int:
    """
    Rebuild the in-memory BM25 index from the full Pinecone corpus.

    This is a synchronous (CPU/IO-bound) operation — call it inside an
    executor when invoked from async code.

    Returns:
        The number of documents indexed after the rebuild.
    """
    corpus = get_all_documents(store)
    bm25_index.build(
        ids=corpus.get("ids", []),
        documents=corpus.get("documents", []),
        metadatas=corpus.get("metadatas", []),
    )
    return bm25_index.document_count


async def publish_bm25_reload(valkey_client: Any, source_id: str) -> None:
    """
    Announce that the corpus changed so peer replicas rebuild their BM25 index.

    Failure to publish is logged but never raised — ingestion must still
    succeed on the local replica even if the pub/sub broker is unavailable.
    """
    try:
        await valkey_client.publish(BM25_RELOAD_CHANNEL, source_id)
        logger.info("Published BM25 reload event (source=%s)", source_id)
    except Exception as exc:
        logger.warning("Failed to publish BM25 reload event: %s", exc)


async def _listen(
    pubsub: Any,
    store: Any,
    bm25_index: BM25Index,
    self_id: str,
) -> None:
    """Background loop: rebuild BM25 whenever a *peer* publishes a reload event."""
    loop = asyncio.get_event_loop()
    try:
        async for message in pubsub.listen():
            if not message or message.get("type") != "message":
                continue

            data = message.get("data")
            if isinstance(data, bytes):
                data = data.decode("utf-8", "ignore")

            if data == self_id:
                # We triggered this event and already rebuilt in-process.
                logger.debug("Ignoring self-originated BM25 reload event")
                continue

            try:
                count = await loop.run_in_executor(
                    None, rebuild_bm25_from_store, store, bm25_index
                )
                logger.info(
                    "BM25 reloaded from peer event (source=%s) -> %d docs",
                    data, count,
                )
            except Exception as exc:
                logger.error("BM25 peer reload failed: %s", exc, exc_info=True)
    except asyncio.CancelledError:
        logger.info("BM25 subscriber loop cancelled")
        raise
    except Exception as exc:
        logger.error("BM25 subscriber loop crashed: %s", exc, exc_info=True)


async def start_bm25_subscriber(
    valkey_client: Any,
    store: Any,
    bm25_index: BM25Index,
    self_id: str,
) -> tuple[Optional[asyncio.Task], Optional[Any]]:
    """
    Subscribe to the reload channel and start the background listener task.

    Returns:
        (task, pubsub) — both are None if subscription failed (degrades to
        the previous single-replica behaviour without crashing startup).
    """
    try:
        pubsub = valkey_client.pubsub()
        await pubsub.subscribe(BM25_RELOAD_CHANNEL)
        task = asyncio.create_task(_listen(pubsub, store, bm25_index, self_id))
        logger.info(
            "BM25 pub/sub subscriber started (channel=%s self=%s)",
            BM25_RELOAD_CHANNEL, self_id,
        )
        return task, pubsub
    except Exception as exc:
        logger.warning(
            "Could not start BM25 subscriber (%s) — running without "
            "cross-replica BM25 sync", exc,
        )
        return None, None


async def stop_bm25_subscriber(
    task: Optional[asyncio.Task],
    pubsub: Optional[Any],
) -> None:
    """Cancel the listener task and unsubscribe/close the pubsub connection."""
    if task is not None:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    if pubsub is not None:
        try:
            await pubsub.unsubscribe(BM25_RELOAD_CHANNEL)
            await pubsub.aclose()
        except Exception as exc:
            logger.debug("Error closing BM25 pubsub: %s", exc)
