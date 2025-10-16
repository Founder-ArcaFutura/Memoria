"""Abstract interfaces for Memoria synchronization backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol


class SyncEventAction(str, Enum):
    """Canonical actions emitted by storage synchronization hooks."""

    MEMORY_CREATED = "memory.created"
    MEMORY_UPDATED = "memory.updated"
    MEMORY_DELETED = "memory.deleted"
    THREAD_UPDATED = "thread.updated"


@dataclass(slots=True)
class SyncEvent:
    """Structured payload exchanged between Memoria instances."""

    action: str
    entity_type: str
    namespace: str
    entity_id: str | None = None
    payload: dict[str, object] = field(default_factory=dict)
    origin: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of the event."""

        data: dict[str, object] = {
            "action": self.action,
            "entity_type": self.entity_type,
            "namespace": self.namespace,
            "created_at": self.created_at.isoformat(),
        }
        if self.entity_id is not None:
            data["entity_id"] = self.entity_id
        if self.payload:
            data["payload"] = self.payload
        if self.origin is not None:
            data["origin"] = self.origin
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SyncEvent:
        """Reconstruct a :class:`SyncEvent` from a dictionary payload."""

        action = str(payload.get("action"))
        entity_type = str(payload.get("entity_type"))
        namespace = str(payload.get("namespace"))
        entity_id_value = payload.get("entity_id")
        entity_id = str(entity_id_value) if entity_id_value is not None else None
        raw_created_at = payload.get("created_at")
        created_at: datetime
        if isinstance(raw_created_at, datetime):
            created_at = raw_created_at.astimezone(timezone.utc)
        elif isinstance(raw_created_at, str):
            try:
                created_at = datetime.fromisoformat(raw_created_at)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                else:
                    created_at = created_at.astimezone(timezone.utc)
            except ValueError:
                created_at = datetime.now(timezone.utc)
        else:
            created_at = datetime.now(timezone.utc)

        payload_value = payload.get("payload")
        payload_dict: dict[str, object]
        if isinstance(payload_value, Mapping):
            payload_dict = dict(payload_value)
        else:
            payload_dict = {}

        origin_value = payload.get("origin")
        origin = str(origin_value) if origin_value is not None else None

        return cls(
            action=action,
            entity_type=entity_type,
            namespace=namespace,
            entity_id=entity_id,
            payload=payload_dict,
            origin=origin,
            created_at=created_at,
        )


SyncEventHandler = Callable[[SyncEvent], None]


class SyncSubscription(Protocol):
    """Active subscription returned by :func:`SyncBackend.subscribe`."""

    def close(self) -> None:
        """Terminate the subscription and release any resources."""


class SyncBackend(Protocol):
    """Protocol describing publisher/subscriber behaviour."""

    def publish(self, event: SyncEvent) -> None:
        """Broadcast an event to all subscribers."""

    def subscribe(self, handler: SyncEventHandler) -> SyncSubscription:
        """Register a handler and return a subscription handle."""

    def close(self) -> None:
        """Release all backend resources (idempotent)."""


class NullSubscription:
    """Subscription implementation that performs no work."""

    def close(self) -> None:  # pragma: no cover - trivial
        return None


class NullSyncBackend:
    """Fallback backend that disables cross-instance synchronisation."""

    def publish(self, event: SyncEvent) -> None:  # pragma: no cover - trivial
        return None

    def subscribe(self, handler: SyncEventHandler) -> SyncSubscription:
        return NullSubscription()

    def close(self) -> None:  # pragma: no cover - trivial
        return None
