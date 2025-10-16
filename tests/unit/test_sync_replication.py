import time

import pytest

from memoria import Memoria
from memoria.config.settings import SyncSettings
from memoria.sync import InMemorySyncBackend


@pytest.mark.parametrize("namespace", ["default"])
def test_memory_sync_updates_caches(tmp_path, namespace):
    db_path = tmp_path / "sync_memories.db"
    backend = InMemorySyncBackend()
    settings = SyncSettings(enabled=True, backend="memory")

    mem_a = Memoria(
        database_connect=f"sqlite:///{db_path}",
        enable_short_term=False,
        namespace=namespace,
        sync_settings=settings,
        sync_backend=backend,
    )
    mem_b = Memoria(
        database_connect=f"sqlite:///{db_path}",
        enable_short_term=False,
        namespace=namespace,
        sync_settings=settings,
        sync_backend=backend,
    )

    try:
        memory_id = mem_a.storage_service.store_memory(
            anchor="alpha",
            text="first draft",
            tokens=3,
            x_coord=0.0,
            y=0.0,
            z=0.0,
        )

        snapshot_b = mem_b.storage_service.get_memory_snapshot(memory_id)
        assert snapshot_b is not None
        assert snapshot_b["text"] == "first draft"

        mem_a.storage_service.update_memory(memory_id, {"text": "final draft"})

        for _ in range(40):
            cached = mem_b.storage_service.get_memory_snapshot(memory_id)
            if cached and cached["text"] == "final draft":
                break
            time.sleep(0.05)
        cached = mem_b.storage_service.get_memory_snapshot(memory_id)
        assert cached is not None
        assert cached["text"] == "final draft"

        mem_a.storage_service.delete_memory(memory_id)
        for _ in range(40):
            cached = mem_b.storage_service.get_memory_snapshot(memory_id)
            if cached is None:
                break
            time.sleep(0.05)
        assert mem_b.storage_service.get_memory_snapshot(memory_id) is None
    finally:
        mem_a.cleanup()
        mem_b.cleanup()
        backend.close()


def test_thread_sync_refreshes_cache(tmp_path):
    db_path = tmp_path / "threads.db"
    backend = InMemorySyncBackend()
    settings = SyncSettings(enabled=True, backend="memory")

    mem_a = Memoria(
        database_connect=f"sqlite:///{db_path}",
        enable_short_term=False,
        sync_settings=settings,
        sync_backend=backend,
    )
    mem_b = Memoria(
        database_connect=f"sqlite:///{db_path}",
        enable_short_term=False,
        sync_settings=settings,
        sync_backend=backend,
    )

    try:
        first_id = mem_a.storage_service.store_memory(
            anchor="thread",
            text="opening",
            tokens=1,
            x_coord=0.0,
            y=0.0,
            z=0.0,
        )

        thread_id = "conversation-1"
        mem_a.storage_service.store_thread(
            thread_id,
            message_links=[
                {"memory_id": first_id, "sequence_index": 0, "role": "user"}
            ],
        )

        thread_snapshot = mem_b.storage_service.get_thread(thread_id)
        assert thread_snapshot is not None
        assert len(thread_snapshot["messages"]) == 1

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

        for _ in range(40):
            updated = mem_b.storage_service.get_thread(thread_id)
            if updated and len(updated["messages"]) == 2:
                break
            time.sleep(0.05)
        updated = mem_b.storage_service.get_thread(thread_id)
        assert updated is not None
        assert len(updated["messages"]) == 2
        assert updated["messages"][1]["memory_id"] == second_id
    finally:
        mem_a.cleanup()
        mem_b.cleanup()
        backend.close()
