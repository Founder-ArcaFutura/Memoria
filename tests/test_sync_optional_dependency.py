from __future__ import annotations

import time
from pathlib import Path

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older runtimes
    import tomli as tomllib  # type: ignore[no-redef]

from memoria import Memoria
from memoria.config.settings import SyncBackendType, SyncSettings
from memoria.sync import InMemorySyncBackend


def _load_pyproject() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    pyproject_path = project_root / "pyproject.toml"
    with pyproject_path.open("rb") as fp:
        return tomllib.load(fp)


def test_sync_extra_declares_redis_dependency() -> None:
    pyproject = _load_pyproject()
    extras = pyproject["project"]["optional-dependencies"]

    assert "sync" in extras, "sync extra is missing from pyproject.toml"
    assert any(dep.startswith("redis") for dep in extras["sync"]), (
        "sync extra must include redis dependency",
    )
    assert any(dep.startswith("redis") for dep in extras["all"]), (
        "aggregate 'all' extra must include redis dependency",
    )


def test_sync_extra_allows_backend_import() -> None:
    """Ensure the Redis backend can be imported when optional deps are installed."""
    pyproject = _load_pyproject()
    extras = pyproject["project"]["optional-dependencies"]
    assert "sync" in extras

    # Import succeeds even when redis isn't installed; the optional dependency is
    # resolved when constructing the backend at runtime.
    from memoria.sync import RedisSyncBackend

    assert RedisSyncBackend.__name__ == "RedisSyncBackend"


def _wait_for(predicate, *, timeout: float = 3.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_realtime_sync_replication_converges(tmp_path) -> None:
    backend = InMemorySyncBackend()
    db1 = tmp_path / "mem1.db"
    db2 = tmp_path / "mem2.db"
    settings_a = SyncSettings(
        enabled=True,
        backend=SyncBackendType.MEMORY,
        realtime_replication=True,
        privacy_floor=0.0,
    )
    settings_b = SyncSettings(
        enabled=True,
        backend=SyncBackendType.MEMORY,
        realtime_replication=True,
        privacy_floor=0.0,
    )

    mem1 = Memoria(
        database_connect=f"sqlite:///{db1}",
        namespace="sync-test",
        sync_settings=settings_a,
        sync_backend=backend,
        schema_init=True,
    )
    mem2 = Memoria(
        database_connect=f"sqlite:///{db2}",
        namespace="sync-test",
        sync_settings=settings_b,
        sync_backend=backend,
        schema_init=True,
    )

    try:
        memory_id = mem1.storage_service.store_memory(
            anchor="test",
            text="hello world",
            tokens=3,
            y=5.0,
        )

        def _remote_snapshot() -> dict | None:
            return mem2.storage_service.get_memory_snapshot(memory_id, refresh=True)

        assert _wait_for(lambda: _remote_snapshot() is not None)
        remote = _remote_snapshot()
        assert remote is not None
        assert remote.get("text") == "hello world"

        mem1.storage_service.update_memory(
            memory_id,
            {
                "text": "updated text",
                "y_coord": 6.0,
            },
        )

        def _is_updated() -> bool:
            snapshot = _remote_snapshot()
            return bool(snapshot and snapshot.get("text") == "updated text")

        assert _wait_for(_is_updated)
        refreshed = _remote_snapshot()
        assert refreshed is not None
        assert refreshed.get("y_coord") == 6.0

        mem1.storage_service.delete_memory(memory_id)
        assert _wait_for(lambda: _remote_snapshot() is None)
    finally:
        mem1.cleanup()
        mem2.cleanup()
        backend.close()


def test_realtime_sync_respects_privacy_filters(tmp_path) -> None:
    backend = InMemorySyncBackend()
    settings_public = SyncSettings(
        enabled=True,
        backend=SyncBackendType.MEMORY,
        realtime_replication=True,
        privacy_floor=0.0,
    )
    settings_private = SyncSettings(
        enabled=True,
        backend=SyncBackendType.MEMORY,
        realtime_replication=True,
        privacy_floor=0.0,
    )

    mem1 = Memoria(
        database_connect=f"sqlite:///{tmp_path / 'privacy1.db'}",
        namespace="privacy",
        sync_settings=settings_public,
        sync_backend=backend,
        schema_init=True,
    )
    mem2 = Memoria(
        database_connect=f"sqlite:///{tmp_path / 'privacy2.db'}",
        namespace="privacy",
        sync_settings=settings_private,
        sync_backend=backend,
        schema_init=True,
    )

    try:
        public_id = mem1.storage_service.store_memory(
            anchor="public",
            text="broadcast me",
            tokens=3,
            y=5.0,
        )
        private_id = mem1.storage_service.store_memory(
            anchor="private",
            text="keep quiet",
            tokens=3,
            y=-12.0,
        )

        def _public_remote() -> dict | None:
            return mem2.storage_service.get_memory_snapshot(public_id, refresh=True)

        def _private_remote() -> dict | None:
            return mem2.storage_service.get_memory_snapshot(private_id, refresh=True)

        assert _wait_for(lambda: _public_remote() is not None)
        assert not _wait_for(lambda: _private_remote() is not None, timeout=0.5)
    finally:
        mem1.cleanup()
        mem2.cleanup()
        backend.close()
