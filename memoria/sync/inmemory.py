"""In-memory sync backend for tests and single-process deployments."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from .base import NullSubscription, SyncBackend, SyncEvent, SyncSubscription


class _InMemorySubscription:
    """Background worker that processes events for a subscriber."""

    def __init__(
        self,
        backend: InMemorySyncBackend,
        handler: Callable[[SyncEvent], None],
    ) -> None:
        self._backend = backend
        self._handler = handler
        self._queue: queue.Queue[SyncEvent | None] = queue.Queue()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, event: SyncEvent | None) -> None:
        if not self._closed.is_set():
            self._queue.put(event)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(None)
        self._backend._remove_subscription(self)
        self._thread.join(timeout=1)

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if event is None:
                break
            try:
                self._handler(event)
            except Exception:  # pragma: no cover - defensive logging via backend
                self._backend._handle_handler_error()


class InMemorySyncBackend(SyncBackend):
    """Thread-based pub/sub backend used for tests."""

    def __init__(self) -> None:
        self._subscribers: list[_InMemorySubscription] = []
        self._lock = threading.Lock()
        self._error_callback: Callable[[Exception], None] | None = None

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        self._error_callback = callback

    def publish(self, event: SyncEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscription in subscribers:
            subscription.enqueue(event)

    def subscribe(self, handler: Callable[[SyncEvent], None]) -> SyncSubscription:
        subscription = _InMemorySubscription(self, handler)
        with self._lock:
            self._subscribers.append(subscription)
        return subscription

    def _remove_subscription(self, subscription: _InMemorySubscription) -> None:
        with self._lock:
            try:
                self._subscribers.remove(subscription)
            except ValueError:  # pragma: no cover - defensive
                pass

    def _handle_handler_error(self) -> None:  # pragma: no cover - defensive
        if self._error_callback is None:
            return
        try:
            self._error_callback(Exception("sync handler raised an exception"))
        except Exception:
            return

    def close(self) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        for subscription in subscribers:
            subscription.close()


class DisabledInMemoryBackend(InMemorySyncBackend):  # pragma: no cover - legacy alias
    def subscribe(self, handler: Callable[[SyncEvent], None]) -> SyncSubscription:
        return NullSubscription()

    def publish(self, event: SyncEvent) -> None:
        return None
