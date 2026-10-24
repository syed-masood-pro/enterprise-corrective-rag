"""
Document lifecycle management routes.

  GET    /api/v1/documents           — list all ingested documents
  GET    /api/v1/documents/{name}    — get chunks for a specific document
  DELETE /api/v1/documents/{name}    — delete a document + all its chunks
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from utils.bm25_sync import publish_bm25_reload
from vectorstore.store import (
    delete_document,
    get_all_documents,
    get_document_chunks,
    list_documents,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Documents"])


@router.get(
    "/documents",
    summary="List all ingested documents",
)
async def list_docs(request: Request) -> dict:
    """
    Return a list of all documents that have been ingested, with chunk counts.
    Useful for auditing which files are in the knowledge base.
    """
    store = request.app.state.store
    docs = list_documents(store)
    return {
        "total_documents": len(docs),
        "documents": docs,
    }


@router.get(
    "/documents/{doc_name}",
    summary="Get chunks for a specific document",
)
async def get_doc_chunks(request: Request, doc_name: str) -> dict:
    """
    Return all chunks belonging to a specific document, including
    content previews and metadata.
    """
    store = request.app.state.store
    chunks = get_document_chunks(store, doc_name)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document found with name '{doc_name}'.",
        )

    return {
        "doc_name": doc_name,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


@router.delete(
    "/documents/{doc_name}",
    summary="Delete a document and all its chunks",
)
async def delete_doc(request: Request, doc_name: str) -> dict:
    """
    Permanently delete a document and all its chunks from:
    1. Pinecone vector store
    2. BM25 in-memory keyword index (rebuilt automatically)

    This is irreversible. To re-index the document, upload it again
    via the /api/v1/ingest endpoint.
    """
    store = request.app.state.store
    bm25_index = request.app.state.bm25_index

    try:
        deleted_count = delete_document(store, doc_name)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document found with name '{doc_name}'.",
        )

    # Rebuild BM25 index after deletion
    corpus = get_all_documents(store)
    bm25_index.build(
        ids=corpus.get("ids", []),
        documents=corpus.get("documents", []),
        metadatas=corpus.get("metadatas", []),
    )

    logger.info(
        "Document deleted: doc=%s chunks=%d bm25_total=%d",
        doc_name, deleted_count, bm25_index.document_count,
    )

    # Notify peer replicas to rebuild their in-memory BM25 index so the deleted
    # document stops appearing in keyword search fleet-wide (mirrors ingest).
    valkey_client = getattr(request.app.state, "valkey_client", None)
    instance_id = getattr(request.app.state, "instance_id", "")
    if valkey_client is not None:
        await publish_bm25_reload(valkey_client, instance_id)
    
    return {
        "doc_name": doc_name,
        "chunks_deleted": deleted_count,
        "total_indexed": bm25_index.document_count,
        "message": f"Document '{doc_name}' and {deleted_count} chunks deleted successfully.",
    }
