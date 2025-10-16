"""Tests covering the lightweight stub sync backend used in unit tests."""

from __future__ import annotations

import pytest

from memoria import Memoria
from memoria.config.settings import SyncSettings
from memoria.sync import SyncEvent, SyncEventAction
from memoria.sync.base import SyncBackend, SyncSubscription


class _StubSubscription(SyncSubscription):
    def __init__(self, backend: StubSyncBackend, handler):
        self._backend = backend
        self._handler = handler
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._backend._remove_handler(self._handler)
        self._closed = True


class StubSyncBackend(SyncBackend):
    """Synchronous backend that immediately dispatches events to subscribers."""

    def __init__(self) -> None:
        self._handlers: list = []

    def publish(self, event: SyncEvent) -> None:
        for handler in list(self._handlers):
            handler(event)

    def subscribe(self, handler):
        self._handlers.append(handler)
        return _StubSubscription(self, handler)

    def _remove_handler(self, handler) -> None:
        try:
            self._handlers.remove(handler)
        except ValueError:
            pass

    def close(self) -> None:
        self._handlers.clear()


@pytest.fixture()
def stub_backend() -> StubSyncBackend:
    return StubSyncBackend()


@pytest.fixture()
def sync_settings() -> SyncSettings:
    return SyncSettings(enabled=True, backend="memory")


def _create_pair(tmp_path, backend: StubSyncBackend, settings: SyncSettings):
    db_path = tmp_path / "stub-sync.db"
    mem_a = Memoria(
        database_connect=f"sqlite:///{db_path}",
        enable_short_term=False,
        sync_backend=backend,
        sync_settings=settings,
    )
    mem_b = Memoria(
        database_connect=f"sqlite:///{db_path}",
        enable_short_term=False,
        sync_backend=backend,
        sync_settings=settings,
    )
    return mem_a, mem_b, backend


def test_stub_backend_propagates_memory_events(tmp_path, stub_backend, sync_settings):
    mem_a, mem_b, backend = _create_pair(tmp_path, stub_backend, sync_settings)
    try:
        memory_id = mem_a.storage_service.store_memory(
            anchor="alpha",
            text="draft",
            tokens=3,
            x_coord=0.0,
            y=0.0,
            z=0.0,
        )
        snapshot = mem_b.storage_service.get_memory_snapshot(memory_id)
        assert snapshot and snapshot["text"] == "draft"

        mem_a.storage_service.update_memory(memory_id, {"text": "final"})
        snapshot = mem_b.storage_service.get_memory_snapshot(memory_id)
        assert snapshot and snapshot["text"] == "final"

        mem_a.storage_service.delete_memory(memory_id)
        assert mem_b.storage_service.get_memory_snapshot(memory_id) is None
    finally:
        mem_a.cleanup()
        mem_b.cleanup()
        backend.close()


def test_stub_backend_dispatches_thread_updates(tmp_path, stub_backend, sync_settings):
    mem_a, mem_b, backend = _create_pair(tmp_path, stub_backend, sync_settings)
    try:
        first_id = mem_a.storage_service.store_memory(
            anchor="thread",
            text="start",
            tokens=1,
            x_coord=0.0,
            y=0.0,
            z=0.0,
        )
        thread_id = "thread-1"
        mem_a.storage_service.store_thread(
            thread_id,
            message_links=[
                {"memory_id": first_id, "sequence_index": 0, "role": "user"}
            ],
        )
        snapshot = mem_b.storage_service.get_thread(thread_id)
        assert snapshot is not None
        assert len(snapshot["messages"]) == 1

        second_id = mem_a.storage_service.store_memory(
            anchor="thread",
            text="reply",
            tokens=1,
            x_coord=0.0,
            y=0.0,
            z=0.0,
        )
        mem_a.storage_service.store_thread(
            thread_id,
            message_links=[
                {"memory_id": first_id, "sequence_index": 0, "role": "user"},
                {"memory_id": second_id, "sequence_index": 1, "role": "assistant"},
            ],
        )
        updated = mem_b.storage_service.get_thread(thread_id)
        assert updated is not None
        assert len(updated["messages"]) == 2
        assert updated["messages"][1]["memory_id"] == second_id
    finally:
        mem_a.cleanup()
        mem_b.cleanup()
        backend.close()


def test_sync_event_from_dict_roundtrip():
    event = SyncEvent(
        action=SyncEventAction.MEMORY_CREATED.value,
        entity_type="memory",
        namespace="default",
        entity_id="abc123",
        payload={"memory_id": "abc123", "text": "hello"},
        origin="origin-1",
    )
    restored = SyncEvent.from_dict(event.to_dict())
    assert restored.action == event.action
    assert restored.entity_type == event.entity_type
    assert restored.entity_id == event.entity_id
    assert restored.payload == event.payload
    assert restored.origin == event.origin
