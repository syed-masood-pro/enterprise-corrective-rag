"""
Async Valkey connection helper.

Valkey is an open-source Redis fork (Linux Foundation).
The `valkey-py` client is API-compatible with redis-py.

Start Valkey locally:
    docker run -d --name valkey -p 6379:6379 valkey/valkey:latest

Or on Windows without Docker, use the Windows Subsystem for Linux (WSL):
    wsl -- docker run -d --name valkey -p 6379:6379 valkey/valkey:latest
"""
from __future__ import annotations

import logging
from typing import Optional

import valkey.asyncio as aioValkey

logger = logging.getLogger(__name__)


async def create_valkey_client(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
) -> aioValkey.Valkey:
    """
    Create a verified async Valkey connection.

    Performs a PING on startup to surface misconfiguration early.

    Args:
        host:     Valkey server hostname or IP.
        port:     Valkey server port (default 6379).
        db:       Valkey logical database index (0–15).
        password: Optional AUTH password (leave empty if none configured).

    Returns:
        Connected async Valkey client ready for use.

    Raises:
        ConnectionError: If Valkey is not reachable at the given host/port.
    """
    client = aioValkey.Valkey(
        host=host,
        port=port,
        db=db,
        password=password or None,
        decode_responses=False,       # we handle bytes manually
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
        health_check_interval=30,
    )

    try:
        await client.ping()
        logger.info("Valkey connected: %s:%d  db=%d", host, port, db)
    except Exception as exc:
        await client.aclose()
        raise ConnectionError(
            f"\n\n❌ Cannot connect to Valkey at {host}:{port}.\n"
            f"   Start Valkey with:\n"
            f"   docker run -d --name valkey -p 6379:6379 valkey/valkey:latest\n"
            f"   Original error: {exc}\n"
        ) from exc

    return client


async def close_valkey_client(client: aioValkey.Valkey) -> None:
    """Gracefully close the Valkey connection on application shutdown."""
    try:
        await client.aclose()
        logger.info("Valkey connection closed")
    except Exception as exc:
        logger.warning("Error closing Valkey connection: %s", exc)
