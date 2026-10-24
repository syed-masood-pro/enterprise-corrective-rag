"""
API-key authentication middleware.

Every request (except the public paths below) must carry a valid
X-API-Key header.  Invalid or missing keys receive 401.
"""
from __future__ import annotations

import hmac
import logging

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED

logger = logging.getLogger(__name__)

# Paths that do NOT require authentication
_PUBLIC_PATHS: frozenset[str] = frozenset({
    "/health",
    "/livez",
    "/docs",
    "/redoc",
    "/openapi.json",
})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that validates the X-API-Key header."""

    def __init__(self, app, valid_keys: set[str]) -> None:
        super().__init__(app)
        self._valid_keys = valid_keys
    
    def _is_valid(self, api_key: str) -> bool:
        """Constant-time comparison against every configured key.

        Using hmac.compare_digest prevents timing attacks that a plain
        `api_key in self._valid_keys` set-membership check would leak.
        We always compare against all keys (no early exit) to keep timing
        independent of how many keys are configured.
        """
        matched = False
        for key in self._valid_keys:
            if hmac.compare_digest(api_key, key):
                matched = True
        return matched

    async def dispatch(self, request: Request, call_next):
        if request.method=="OPTIONS":
            return await call_next(request)
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "").strip()

        if not api_key or not self._is_valid(api_key):
            client_ip = request.client.host if request.client else "unknown"
            logger.warning(
                "Unauthorised request: path=%s ip=%s",
                request.url.path,
                client_ip,
            )
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key."},
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return await call_next(request)
