#groq llm client via langchain

#singleton chatgrop instance, loaded once at startup
#async invocation (non blocking for all concurrent user)
#exponential back-off retry on transient errors/groq rate limits(https 429)

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential
)

logger=logging.getLogger(__name__)

_groq_instance:Optional[ChatGroq]=None
_lock=threading.Lock()

# Substrings that indicate a *transient* failure worth retrying. Anything else
# (401 auth, 400 bad request, malformed prompt, etc.) is permanent — retrying
# would just waste latency and quota, so we let it raise immediately.
_TRANSIENT_MARKERS: tuple[str, ...] = (
    "429", "rate", "timeout", "timed out", "connection", "connect",
    "temporarily", "unavailable", "503", "502", "500", "overloaded",
)


def _is_transient(exc: BaseException) -> bool:
    """Return True only for retryable/transient Groq or network errors."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def get_groq_client(
    api_key:str,
    model:str='llama-3.1-8b-instant',
    max_tokens:int=1024,
    temperature:float=0.1,
)->ChatGroq:
    #returns or create a shared singleton ChatGroq client instance
    
    global _groq_instance
    
    if _groq_instance is None:
        with _lock:
            if _groq_instance is None:
                logger.info("Initialising Groq (model=%s)",model)
                _groq_instance=ChatGroq(
                    api_key=api_key,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                logger.info("Groq client ready")
    return _groq_instance

@retry(
    retry=retry_if_exception(Exception),
    wait=wait_exponential(multiplier=1,min=2,max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def generate_answer(messages:list[BaseMessage],llm:ChatGroq)->str:
    #invoke the groq model async with automatic retry
    #backs off exponentially on rate limit(429) or transient network errors.
    #raises the original excception after 3 failed attempts.
    
    try:
        response=await llm.ainvoke(messages)
        return response.content.strip()
    except Exception as exc:
        err=str(exc).lower()
        if '429' in err or 'rate' in err:
            logger.warning("Groq rate limit hit - will retry with back-off")
        else:
            logger.error("Groq generation error: %s",exc)
        raise
    
async def stream_answer(messages: list[BaseMessage], llm: ChatGroq):
    """
    Async-generator variant of generate_answer for token streaming.

    Yields incremental content chunks as they arrive from Groq.

    Retry policy: we can't safely retry *after* tokens have been sent (that
    would duplicate output), but a transient failure while establishing the
    stream OR before the first token is safe to retry. So we retry (with
    exponential backoff, up to 3 attempts) only until the first token is
    yielded; once streaming has started, errors propagate to the caller.

    Args:
        messages: Formatted prompt message list (system + human).
        llm:      Shared ChatGroq instance.

    Yields:
        str: successive token/text fragments.
    """
    attempt = 0
    max_attempts = 3
    while True:
        attempt += 1
        first_token_sent = False
        try:
            async for chunk in llm.astream(messages):
                text = getattr(chunk, "content", "") or ""
                if text:
                    first_token_sent = True
                    yield text
            return  # stream completed successfully
        except Exception as exc:
            # If we've already emitted tokens, we must NOT retry (would dupe).
            if first_token_sent or not _is_transient(exc) or attempt >= max_attempts:
                logger.error(
                    "Streaming failed (attempt %d, first_token_sent=%s): %s",
                    attempt, first_token_sent, exc,
                )
                raise
            backoff = min(2 ** attempt, 10)
            logger.warning(
                "Transient streaming error before first token (attempt %d) — "
                "retrying in %ds: %s", attempt, backoff, exc,
            )
            await asyncio.sleep(backoff)