"""
Thread-safe singleton embedding model.

Uses HuggingFace sentence-transformers (local, free, no API calls).
The model is loaded exactly once at application startup and reused
across all concurrent requests.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_instance: Optional[HuggingFaceEmbeddings] = None
_lock = threading.Lock()


def get_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Return the singleton HuggingFaceEmbeddings instance.
    Thread-safe double-checked locking — model is loaded only once.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Shared HuggingFaceEmbeddings instance.
    """
    global _instance

    if _instance is None:
        with _lock:
            if _instance is None:
                logger.info("Loading embedding model: %s", model_name)
                _instance = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": 64,
                    },
                )
                logger.info("Embedding model loaded successfully")

    return _instance
