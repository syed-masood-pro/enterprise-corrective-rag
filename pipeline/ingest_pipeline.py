# file -> chunks -> embed->store-> rebuild bm25
#called from /api/v1/ingest route handler

from __future__ import annotations

import logging
from pathlib import Path

from chunking.document_processor import process_document
from config.settings import Settings
from retrieval.bm25_index import BM25Index
from vectorstore.store import get_all_documents,upsert_chunks

logger=logging.getLogger(__name__)

async def run_ingest(
    file_path:str|Path,
    store:object,
    bm25_index:BM25Index,
    settings:Settings
)->dict:
    #full ingestion pipeline for a single uploaded file.
    
    #process: detect file type, load with langchain, split into chunks
    #upsert: add new chunks to pincone(deduplication by chunk_id)
    #rebuild: refersh inmemory BM25 index from the updated corpus
    
    #returns summary dict contains doc_name,chunks_produced,chunks_inserted,chunks_skipped,total_indexed.
    
    path=Path(file_path)
    doc_name=path.name
    
    logger.info("Ingestion started: %s",doc_name)
    
    #process
    chunks=process_document(path,settings)
    
    #upsert into pincome
    inserted=upsert_chunks(store,chunks)
    
    #rebuild bm25 for updated corpus
    corpus=get_all_documents(store)
    bm25_index.build(
        ids=corpus.get('ids',[]),
        documents=corpus.get('documents',[]),
        metadatas=corpus.get('metadatas',[])
    )
    
    skipped=len(chunks)-inserted
    logger.info("Ingestion done: doc=%s produced=%d inserted=%s skipped=%d total=%d",doc_name,len(chunks),inserted,skipped,bm25_index.document_count)
    
    return{
        'doc_name':doc_name,
        'chunks_produced':len(chunks),
        'chunks_inserted':inserted,
        'chunks_skipped':skipped,
        'total_indexed':bm25_index.document_count,
    }