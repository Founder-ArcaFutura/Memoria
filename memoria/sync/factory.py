"""Factory helpers for constructing sync backends based on configuration."""

from __future__ import annotations

from typing import Any

from loguru import logger

from .base import NullSyncBackend, SyncBackend
from .inmemory import InMemorySyncBackend
from .postgres import PostgresSyncBackend
from .redis import RedisSyncBackend


def create_sync_backend(settings: Any) -> SyncBackend:
    """Instantiate an appropriate backend based on settings metadata."""

    enabled = bool(getattr(settings, "enabled", False))
    backend_value = getattr(settings, "backend", None)
    if hasattr(backend_value, "value"):
        backend_value = backend_value.value
    if backend_value is None:
        backend_value = "memory" if enabled else "none"
    backend_name = str(backend_value or ("memory" if enabled else "none")).lower()
    if backend_name in {"none", "null", "disabled"}:
        return InMemorySyncBackend() if enabled else NullSyncBackend()
    if backend_name in {"memory", "inmemory", "in-memory"}:
        return InMemorySyncBackend()

    if backend_name == "redis":
        connection_url = getattr(settings, "connection_url", None)
        if not connection_url:
            logger.warning(
                "Redis sync backend requested but no connection URL provided; falling back to null backend.",
            )
            return NullSyncBackend()
        channel = getattr(settings, "channel", None) or "memoria-sync"
        options = getattr(settings, "options", {}) or {}
        client_kwargs = dict(getattr(settings, "client_kwargs", {}) or {})
        if isinstance(options, dict):
            client_kwargs.update(options)
        try:
            return RedisSyncBackend(
                connection_url,
                channel=channel,
                client_kwargs=client_kwargs,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to initialise Redis sync backend; falling back to null backend.",
            )
            return NullSyncBackend()

    if backend_name in {"postgres", "postgresql"}:
        dsn = getattr(settings, "connection_url", None)
        if not dsn:
            logger.warning(
                "Postgres sync backend requested but no connection URL provided; falling back to null backend.",
            )
            return NullSyncBackend()
        channel = getattr(settings, "channel", None) or "memoria_sync"
        table = getattr(settings, "table", None) or "memoria_sync_events"
        connect_kwargs = dict(getattr(settings, "connect_kwargs", {}) or {})
        try:
            return PostgresSyncBackend(
                dsn,
                channel=channel,
                table=table,
                connect_kwargs=connect_kwargs,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to initialise Postgres sync backend; falling back to null backend.",
            )
            return NullSyncBackend()

    logger.warning(
        "Sync backend '%s' is not implemented in the open-source build; falling back to null backend.",
        backend_name,
    )
    return NullSyncBackend()
