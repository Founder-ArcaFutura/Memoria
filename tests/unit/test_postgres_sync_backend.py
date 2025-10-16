"""Unit tests for the PostgresSyncBackend."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import pytest

from memoria.config.settings import SyncBackendType
from memoria.sync import (
    NullSyncBackend,
    SyncEvent,
    SyncEventAction,
    create_sync_backend,
)
from memoria.sync.postgres import PostgresSyncBackend


@dataclass
class _Notify:
    channel: str
    payload: str | None


class _FakeCursor:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def execute(self, sql: str, params: tuple | None = None) -> None:
        statement = sql.strip()
        upper = statement.upper()
        if upper.startswith("CREATE TABLE"):
            self._connection.driver.created_tables.append(statement)
            return
        if upper.startswith("INSERT"):
            payload = params[0] if params else None
            self._connection.driver.inserted_payloads.append(payload)
            return
        if upper.startswith("LISTEN"):
            channel = statement.split()[1].strip(';"')
            self._connection.driver.register_listener(channel, self._connection)
            return
        if upper.startswith("NOTIFY"):
            channel = statement.split()[1].strip(',;"')
            payload = params[0] if params else None
            self._connection.driver.notify(channel, payload)
            return

    def close(self) -> None:
        return None


class _FakeConnection:
    def __init__(self, driver: _FakePsycopg) -> None:
        self.driver = driver
        self.autocommit = False
        self.closed = False
        self.notifies: queue.Queue[_Notify] = queue.Queue()

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def close(self) -> None:
        self.closed = True

    def rollback(self) -> None:
        self.driver.rollbacks += 1


class _FakePsycopg:
    def __init__(self) -> None:
        self.listeners: dict[str, set[_FakeConnection]] = {}
        self.created_tables: list[str] = []
        self.inserted_payloads: list[str | None] = []
        self.rollbacks = 0
        self.connections: list[_FakeConnection] = []

    def connect(self, dsn: str, **_: object) -> _FakeConnection:
        connection = _FakeConnection(self)
        self.connections.append(connection)
        return connection

    def register_listener(self, channel: str, connection: _FakeConnection) -> None:
        self.listeners.setdefault(channel, set()).add(connection)

    def notify(self, channel: str, payload: str | None) -> None:
        notification = _Notify(channel=channel, payload=payload)
        for listener in list(self.listeners.get(channel, set())):
            if listener.closed:
                continue
            listener.notifies.put(notification)


def test_postgres_sync_backend_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_driver = _FakePsycopg()
    monkeypatch.setattr("memoria.sync.postgres.psycopg", fake_driver)
    monkeypatch.setattr("memoria.sync.postgres.psycopg2", None)

    backend = PostgresSyncBackend("postgresql://example", channel="memoria_sync")
    received: list[SyncEvent] = []
    ready = threading.Event()

    def handler(event: SyncEvent) -> None:
        received.append(event)
        ready.set()

    subscription = backend.subscribe(handler)
    try:
        backend.publish(
            SyncEvent(
                action=SyncEventAction.MEMORY_CREATED.value,
                entity_type="memory",
                namespace="primary",
                entity_id="123",
                payload={"source": "unit-test"},
            )
        )
        assert ready.wait(timeout=1.0), "listener did not receive event"
        assert len(received) == 1
        assert received[0].action == SyncEventAction.MEMORY_CREATED.value
        assert fake_driver.inserted_payloads, "event payload was not persisted"
    finally:
        subscription.close()
        backend.close()


def test_create_sync_backend_postgres_missing_dsn() -> None:
    class Settings:
        enabled = True
        backend = SyncBackendType.POSTGRES
        connection_url = ""

    backend = create_sync_backend(Settings())
    assert isinstance(backend, NullSyncBackend)


def test_create_sync_backend_postgres_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Settings:
        enabled = True
        backend = SyncBackendType.POSTGRES
        connection_url = "postgresql://example"

    def raiser(*_: object, **__: object) -> PostgresSyncBackend:
        raise RuntimeError("boom")

    monkeypatch.setattr("memoria.sync.factory.PostgresSyncBackend", raiser)

    backend = create_sync_backend(Settings())
    assert isinstance(backend, NullSyncBackend)
