"""Coordinator glue connecting Memoria instances to a sync backend."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from .base import (
    SyncBackend,
    SyncEvent,
    SyncEventAction,
    SyncEventHandler,
    SyncSubscription,
)


@dataclass
class SyncCoordinator:
    """Wrap a backend and provide convenience helpers for Memoria."""

    backend: SyncBackend
    namespace: str
    origin: str
    handler: SyncEventHandler

    def __post_init__(self) -> None:
        self._subscription: SyncSubscription | None = None

    def start(self) -> None:
        if self._subscription is not None:
            return

        def _wrapped(event: SyncEvent) -> None:
            if event.origin and event.origin == self.origin:
                return
            if event.namespace and event.namespace != self.namespace:
                return
            try:
                self.handler(event)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.opt(exception=True).warning("Sync handler failure: %s", exc)

        self._subscription = self.backend.subscribe(_wrapped)

    def publish_event(
        self,
        action: SyncEventAction | str,
        entity_type: str,
        entity_id: str | None,
        payload: dict[str, object] | None = None,
    ) -> None:
        if isinstance(action, SyncEventAction):
            action_value = action.value
        else:
            action_value = action
        event = SyncEvent(
            action=action_value,
            entity_type=entity_type,
            entity_id=entity_id,
            namespace=self.namespace,
            payload=payload or {},
            origin=self.origin,
        )
        try:
            self.backend.publish(event)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.opt(exception=True).warning("Failed to publish sync event: %s", exc)

    def close(self) -> None:
        if self._subscription is not None:
            try:
                self._subscription.close()
            except Exception:  # pragma: no cover - defensive logging
                logger.opt(exception=True).debug(
                    "Error while closing sync subscription"
                )
            finally:
                self._subscription = None
