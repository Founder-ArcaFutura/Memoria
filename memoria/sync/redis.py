"""Redis-based synchronization backend for multi-instance deployments."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

from loguru import logger

from .base import NullSubscription, SyncBackend, SyncEvent, SyncSubscription

try:  # pragma: no cover - optional dependency resolution
    import redis
except Exception:  # pragma: no cover - handled during backend creation
    redis = None  # type: ignore[assignment]


class _RedisSubscription(SyncSubscription):
    """Listen for Redis pub/sub messages and dispatch them to a handler."""

    def __init__(
        self,
        client: redis.Redis[Any],
        channel: str,
        handler: Callable[[SyncEvent], None],
    ) -> None:
        self._client = client
        self._channel = channel
        self._handler = handler
        self._pubsub = client.pubsub(ignore_subscribe_messages=True)
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread_started = False

    def start(self) -> None:
        if self._thread_started:
            return
        try:
            self._pubsub.subscribe(self._channel)
        except Exception:  # pragma: no cover - redis error handling
            logger.opt(exception=True).warning(
                "RedisSyncBackend failed to subscribe to channel '%s'", self._channel
            )
            raise
        self._thread_started = True
        self._thread.start()

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                message = self._pubsub.get_message(timeout=1.0)
            except Exception:  # pragma: no cover - redis runtime guard
                logger.opt(exception=True).warning("Redis pubsub listener error")
                break
            if message is None:
                continue
            if message.get("type") != "message":
                continue
            data = message.get("data")
            if not isinstance(data, str):
                continue
            try:
                decoded = json.loads(data)
                event = SyncEvent.from_dict(decoded)
            except Exception:  # pragma: no cover - malformed payloads
                logger.opt(exception=True).warning(
                    "Failed to decode Redis sync payload"
                )
                continue
            try:
                self._handler(event)
            except Exception:  # pragma: no cover - defensive subscriber handling
                logger.opt(exception=True).warning("Sync handler raised an exception")

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._pubsub.close()
        except Exception:  # pragma: no cover - redis shutdown guard
            logger.opt(exception=True).debug("Error closing Redis pubsub")
        self._thread.join(timeout=1.0)


class RedisSyncBackend(SyncBackend):
    """Publish/subscribe backend backed by Redis."""

    def __init__(
        self,
        connection_url: str,
        *,
        channel: str = "memoria-sync",
        client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if redis is None:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "redis package is required to use RedisSyncBackend. Install redis>=4.0."
            )
        if not connection_url:
            raise ValueError("RedisSyncBackend requires a connection URL")
        kwargs = dict(client_kwargs or {})
        kwargs.setdefault("decode_responses", True)
        try:
            self._client: redis.Redis[Any] = redis.from_url(connection_url, **kwargs)
        except Exception as exc:  # pragma: no cover - connection failures
            raise RuntimeError(
                f"Failed to connect to Redis at {connection_url}: {exc}"
            ) from exc
        self._channel = channel or "memoria-sync"
        self._subscriptions: list[_RedisSubscription] = []
        self._lock = threading.Lock()

    def publish(self, event: SyncEvent) -> None:
        payload = json.dumps(event.to_dict())
        try:
            self._client.publish(self._channel, payload)
        except Exception:  # pragma: no cover - redis publish guard
            logger.opt(exception=True).warning("Failed to publish sync event via Redis")

    def subscribe(self, handler: Callable[[SyncEvent], None]) -> SyncSubscription:
        subscription = _RedisSubscription(self._client, self._channel, handler)
        try:
            subscription.start()
        except Exception:
            return NullSubscription()
        with self._lock:
            self._subscriptions.append(subscription)
        return subscription

    def close(self) -> None:
        with self._lock:
            subscriptions = list(self._subscriptions)
            self._subscriptions.clear()
        for subscription in subscriptions:
            try:
                subscription.close()
            except Exception:  # pragma: no cover - shutdown guard
                logger.opt(exception=True).debug(
                    "Error while closing Redis subscription"
                )
        try:
            self._client.close()
        except Exception:  # pragma: no cover - redis close guard
            logger.opt(exception=True).debug("Error while closing Redis client")
