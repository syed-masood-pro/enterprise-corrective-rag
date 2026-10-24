"""
Per-request structured logging middleware.

Attaches a short unique request_id to every request, logs the HTTP method,
path, response status, and wall-clock latency, and exposes the request_id
in the X-Request-ID response header for client-side correlation.
"""
from __future__ import annotations

import logging
import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """Log every request with a unique ID and latency."""

    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:10]
        request.state.request_id = request_id
        
        #bind request_id into structlog contextvars so EVERY log line emitted
        #while handling this request is automatically tagged with it
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "request_id=%s method=%s path=%s error=%s latency_ms=%.1f",
                request_id, request.method, request.url.path, exc, latency_ms,
            )
            structlog.contextvars.clear_contextvars()
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request_id=%s method=%s path=%s status=%d latency_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )

        response.headers["X-Request-ID"] = request_id
        structlog.contextvars.clear_contextvars()
        return response
