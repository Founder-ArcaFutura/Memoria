"""Unit tests for API helper modules extracted from the app factory."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask

from memoria_server.api.scheduler import schedule_daily_decrement
from memoria_server.api.spatial_setup import init_spatial_db


def test_schedule_daily_decrement_registers_timer(monkeypatch, tmp_path):
    """The scheduler should register a cancellable timer and persist the run date."""

    app = Flask(__name__)
    db_path = tmp_path / "memory.db"
    app.config.update(
        {
            "DB_PATH": str(db_path),
            "DECREMENT_TZ": "UTC",
        }
    )

    class DummyStorage:
        def __init__(self) -> None:
            self.calls = 0

        def decrement_x_coords(self) -> None:
            self.calls += 1

    class DummyMemoria:
        def __init__(self) -> None:
            self.storage_service = DummyStorage()

    dummy_memoria = DummyMemoria()
    app.config["memoria"] = dummy_memoria

    init_spatial_db(app)

    created_timers: list[object] = []

    class DummyTimer:
        def __init__(self, interval: float, function):
            self.interval = interval
            self.function = function
            self.started = False
            self.cancelled = False

        def start(self):
            self.started = True
            created_timers.append(self)
            return self

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr("memoria_server.api.scheduler.threading.Timer", DummyTimer)

    schedule_daily_decrement(app)

    registry = app.config.get("memoria_background_timers")
    assert registry is not None and "daily_decrement" in registry
    timer = registry["daily_decrement"]
    assert isinstance(timer, DummyTimer)
    assert timer.started is True
    assert created_timers, "timer.start should have been invoked"
    assert dummy_memoria.storage_service.calls == 1

    expected_day = datetime.now(ZoneInfo("UTC")).date()
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT value FROM service_metadata WHERE key='last_decrement_date'"
        )
        row = cur.fetchone()
    finally:
        conn.close()
    assert row is not None and row[0]
    assert datetime.fromisoformat(row[0]).date() == expected_day


def test_init_spatial_db_creates_schema(tmp_path):
    """The spatial setup helper should ensure required tables and columns exist."""

    app = Flask(__name__)
    db_path = tmp_path / "memory.db"
    app.config["DB_PATH"] = str(db_path)

    init_spatial_db(app)

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        assert {"spatial_metadata", "service_metadata"}.issubset(tables)

        cur = conn.execute("PRAGMA table_info(spatial_metadata)")
        columns = {row[1] for row in cur.fetchall()}
        assert {"memory_id", "namespace", "symbolic_anchors"}.issubset(columns)
    finally:
        conn.close()
