"""
Pinecone vector store operations.

Provides:
  - Serverless index creation (idempotent — safe to call on every startup)
  - Deduplicating upsert
  - Full-corpus fetch (for BM25 index building)
  - Document listing, deletion, chunk retrieval
  - Collection statistics
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

from chunking.base import Chunk

logger = logging.getLogger(__name__)

_pc_index: Optional[object] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None
_lock = threading.Lock()


def get_vector_store(
    api_key: str,
    index_name: str,
    cloud: str,
    region: str,
    embeddings: HuggingFaceEmbeddings,
    dimension: int = 384,
) -> object:
    """
    Return (or create) the singleton Pinecone index.

    Creates a serverless index if it doesn't exist (idempotent).

    Args:
        api_key:     Pinecone API key.
        index_name:  Name of the Pinecone index.
        cloud:       Cloud provider (aws, gcp, azure).
        region:      Region (us-east-1, etc.).
        embeddings:  Shared embedding function (stored for query-time embedding).
        dimension:   Embedding dimension (384 for MiniLM).

    Returns:
        Pinecone Index object ready for read/write.
    """
    global _pc_index, _embeddings

    if _pc_index is None:
        with _lock:
            if _pc_index is None:
                logger.info(
                    "Connecting to Pinecone (index=%s, cloud=%s, region=%s)",
                    index_name, cloud, region,
                )
                pc = Pinecone(api_key=api_key)

                # Create index if it doesn't exist
                existing = [idx.name for idx in pc.list_indexes()]
                if index_name not in existing:
                    logger.info("Creating Pinecone serverless index '%s'...", index_name)
                    pc.create_index(
                        name=index_name,
                        dimension=dimension,
                        metric="cosine",
                        spec=ServerlessSpec(cloud=cloud, region=region),
                    )
                    # Wait for index to be ready
                    while not pc.describe_index(index_name).status.get("ready", False):
                        logger.info("Waiting for Pinecone index to be ready...")
                        time.sleep(2)

                _pc_index = pc.Index(index_name)
                _embeddings = embeddings
                logger.info("Pinecone ready: index=%s", index_name)

    return _pc_index


def upsert_chunks(store: object, chunks: list[Chunk]) -> int:
    """
    Add chunks to Pinecone, skipping any already-present IDs.

    Each vector is stored with metadata containing:
      - chunk_id, doc_name, file_type, extension, chunk_index, word_count, text

    Args:
        store:  Pinecone Index object.
        chunks: List of Chunk objects to insert.

    Returns:
        Number of new chunks actually written.
    """
    if not chunks:
        return 0

    # Check which IDs already exist
    ids = [c.chunk_id for c in chunks]
    
    existing: set[str] = set()
    fetch_batch = 1000
    for i in range(0, len(ids), fetch_batch):
        batch_ids = ids[i : i + fetch_batch]
        try:
            fetch_result = store.fetch(ids=batch_ids)
            if fetch_result.vectors:
                existing.update(fetch_result.vectors.keys())
        except Exception:
            # On a fetch error, assume none in this batch exist rather than
            # aborting the whole upsert.
            continue

    new_chunks = [c for c in chunks if c.chunk_id not in existing]
    if not new_chunks:
        logger.info("All %d chunks already exist — skipping upsert", len(chunks))
        return 0

    # Embed all new chunks
    texts = [c.content for c in new_chunks]
    vectors = _embeddings.embed_documents(texts)

    # Build upsert records
    records = []
    for chunk, vec in zip(new_chunks, vectors):
        meta = {**chunk.metadata}
        # Store text in metadata for BM25 corpus retrieval
        meta["text"] = chunk.content[:39000]  # Pinecone metadata limit ~40KB
        meta["chunk_id"] = chunk.chunk_id
        records.append({
            "id": chunk.chunk_id,
            "values": vec,
            "metadata": meta,
        })

    # Upsert in batches of 100 (Pinecone recommended)
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        store.upsert(vectors=batch)

    logger.info("Inserted %d new chunks into Pinecone", len(new_chunks))
    return len(new_chunks)


def get_all_documents(store: object) -> dict:
    """
    Fetch every stored document for BM25 index building.

    Uses Pinecone's list + fetch to retrieve all vectors with metadata.

    Returns:
        dict with keys: 'ids', 'documents', 'metadatas'
    """
    try:
        all_ids: list[str] = []
        all_documents: list[str] = []
        all_metadatas: list[dict] = []

        # List all vector IDs
        id_list = []
                
        for list_response in store.list():
            if hasattr(list_response,'vectors'):
                for item in list_response.vectors:
                    if hasattr(item,'id'):
                        id_list.append(item.id)
            elif isinstance(list_response,(list,tuple)):
                id_list.extend(list_response)

        if not id_list:
            return {"ids": [], "documents": [], "metadatas": []}

        # Fetch in batches of 1000
        batch_size = 100
        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i : i + batch_size]
            result = store.fetch(ids=batch_ids)
            if result.vectors:
                for vid, vdata in result.vectors.items():
                    meta = vdata.metadata or {}
                    all_ids.append(vid)
                    all_documents.append(meta.get("text", ""))
                    all_metadatas.append(meta)

        return {"ids": all_ids, "documents": all_documents, "metadatas": all_metadatas}

    except Exception as exc:
        logger.error("Failed to fetch corpus from Pinecone: %s", exc)
        return {"ids": [], "documents": [], "metadatas": []}


def get_collection_stats(store: object) -> dict:
    """Return a lightweight health-check dict for the /health endpoint."""
    try:
        stats = store.describe_index_stats()
        count = stats.total_vector_count or 0
        return {"document_count": count, "status": "ok"}
    except Exception as exc:
        return {"document_count": 0, "status": f"error: {exc}"}


# ── Document lifecycle ────────────────────────────────────────────────────────

def list_documents(store: object) -> list[dict]:
    """
    Return a deduplicated list of ingested documents with chunk counts.
    """
    try:
        corpus = get_all_documents(store)
        docs: dict[str, dict] = {}
        for meta in corpus.get("metadatas", []):
            if not meta:
                continue
            name = meta.get("doc_name", "unknown")
            if name not in docs:
                docs[name] = {
                    "doc_name": name,
                    "file_type": meta.get("file_type", ""),
                    "extension": meta.get("extension", ""),
                    "chunk_count": 0,
                }
            docs[name]["chunk_count"] += 1
        return list(docs.values())
    except Exception as exc:
        logger.error("Failed to list documents: %s", exc)
        return []


def delete_document(store: object, doc_name: str) -> int:
    """
    Delete ALL chunks belonging to a specific document by doc_name.

    Args:
        store:    Pinecone Index object.
        doc_name: Exact doc_name metadata value (e.g. "main.py").

    Returns:
        Number of chunks deleted.
    """
    try:
        corpus = get_all_documents(store)
        ids_to_delete = []
        for cid, meta in zip(corpus.get("ids", []), corpus.get("metadatas", [])):
            if meta and meta.get("doc_name") == doc_name:
                ids_to_delete.append(cid)

        if not ids_to_delete:
            logger.info("No chunks found for doc_name='%s'", doc_name)
            return 0

        # Delete in batches of 1000 (Pinecone limit)
        batch_size = 1000
        for i in range(0, len(ids_to_delete), batch_size):
            batch = ids_to_delete[i : i + batch_size]
            store.delete(ids=batch)

        logger.info("Deleted %d chunks for doc_name='%s'", len(ids_to_delete), doc_name)
        return len(ids_to_delete)

    except Exception as exc:
        logger.error("Failed to delete doc '%s': %s", doc_name, exc)
        raise RuntimeError(f"Failed to delete document '{doc_name}': {exc}") from exc


def get_document_chunks(store: object, doc_name: str) -> list[dict]:
    """
    Return all chunks for a specific document with their metadata.
    """
    try:
        corpus = get_all_documents(store)
        results = []
        for cid, meta, doc in zip(
            corpus.get("ids", []),
            corpus.get("metadatas", []),
            corpus.get("documents", []),
        ):
            if meta and meta.get("doc_name") == doc_name:
                results.append({
                    "chunk_id": cid,
                    "chunk_index": meta.get("chunk_index", -1),
                    "content_preview": (doc[:200] + "…") if doc and len(doc) > 200 else doc,
                    "word_count": meta.get("word_count", 0),
                    "file_type": meta.get("file_type", ""),
                })

        results.sort(key=lambda x: x["chunk_index"])
        return results
    except Exception as exc:
        logger.error("Failed to get chunks for doc '%s': %s", doc_name, exc)
        return []
