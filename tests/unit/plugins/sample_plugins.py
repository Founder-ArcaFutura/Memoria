"""Sample plugin implementations for unit tests."""

from __future__ import annotations

from typing import Any


class RecordingPlugin:
    """Record the Memoria instance received during initialization."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.initialized_with = None
        self.ready_with = None
        self.shutdown_called = False
        self.storage_seen = False

    def initialize(self, memoria) -> None:  # pragma: no cover - exercised in tests
        self.initialized_with = memoria
        self.storage_seen = bool(getattr(memoria, "storage_service", None))

    def on_memoria_ready(
        self, memoria
    ) -> None:  # pragma: no cover - exercised in tests
        self.ready_with = memoria

    def shutdown(self) -> None:  # pragma: no cover - exercised in tests
        self.shutdown_called = True


class FailingPlugin:
    """Plugin missing the required interface to trigger graceful failure."""

    def __init__(self, **_: dict[str, Any]) -> None:
        pass


class ShutdownTrackerPlugin(RecordingPlugin):
    """Extends :class:`RecordingPlugin` to expose shutdown behaviour."""

    name = "shutdown-tracker"
