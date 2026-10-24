"""
Rate-limiter setup using slowapi (per remote IP address).

The limiter instance is imported by route handlers that need limiting.
The handler for RateLimitExceeded is registered in main.py.
"""
from __future__ import annotations

import logging

from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{_settings.rate_limit_per_minute}/minute"],
    storage_uri=_settings.valkey_storage_uri,
    # Don't hard-crash if Valkey is momentarily unavailable at boot;
    # slowapi falls back to in-memory storage and logs a warning.
    in_memory_fallback_enabled=True,
)

logger.info(
    "Rate limiter initialised: %d/minute (storage=valkey)",
    _settings.rate_limit_per_minute,
)
