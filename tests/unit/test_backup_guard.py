import pytest
from sqlalchemy import create_engine, text

from memoria.database.backup_guard import backup_guard
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager


def _prepare_table(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE allowed (id INTEGER PRIMARY KEY, value TEXT)"))
        conn.execute(text("INSERT INTO allowed (value) VALUES ('a')"))


def test_backup_guard_restores_on_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conn_str = f"sqlite:///{tmp_path / 'guard.db'}"
    engine = create_engine(conn_str)

    _prepare_table(engine)

    restore_calls: list[str] = []
    original_restore = SQLAlchemyDatabaseManager.restore_database

    def tracking_restore(self, backup_path):
        restore_calls.append(str(backup_path))
        return original_restore(self, backup_path)

    monkeypatch.setattr(
        "memoria.database.sqlalchemy_manager.SQLAlchemyDatabaseManager.restore_database",
        tracking_restore,
    )
    monkeypatch.setattr("memoria.database.backup_guard._get_db_size", lambda manager: 0)

    with backup_guard(conn_str):
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO allowed (value) VALUES ('b')"))

    assert restore_calls

    engine.dispose()
    engine = create_engine(conn_str)
    with engine.begin() as conn:
        values = [
            row[0]
            for row in conn.execute(text("SELECT value FROM allowed ORDER BY id"))
        ]

    assert values == ["a"]

    engine.dispose()


def test_backup_guard_respects_ignored_tables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conn_str = f"sqlite:///{tmp_path / 'guard.db'}"
    engine = create_engine(conn_str)

    _prepare_table(engine)

    restore_called = False

    def fake_restore(self, backup_path):
        nonlocal restore_called
        restore_called = True

    monkeypatch.setattr(
        "memoria.database.sqlalchemy_manager.SQLAlchemyDatabaseManager.restore_database",
        fake_restore,
    )
    monkeypatch.setattr("memoria.database.backup_guard._get_db_size", lambda manager: 0)

    with backup_guard(conn_str, ignore_tables={"allowed"}):
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO allowed (value) VALUES ('b')"))

    assert restore_called is False

    with engine.begin() as conn:
        values = [
            row[0]
            for row in conn.execute(text("SELECT value FROM allowed ORDER BY id"))
        ]

    assert values == ["a", "b"]

    engine.dispose()


def test_backup_guard_can_disable_auto_restore(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conn_str = f"sqlite:///{tmp_path / 'guard.db'}"
    engine = create_engine(conn_str)

    _prepare_table(engine)

    def fail_if_called(self, backup_path):
        pytest.fail(
            "restore_database should not be called when auto restore is disabled"
        )

    monkeypatch.setattr(
        "memoria.database.sqlalchemy_manager.SQLAlchemyDatabaseManager.restore_database",
        fail_if_called,
    )
    monkeypatch.setattr("memoria.database.backup_guard._get_db_size", lambda manager: 0)

    with backup_guard(conn_str, auto_restore_on_mismatch=False):
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO allowed (value) VALUES ('b')"))

    with engine.begin() as conn:
        values = [
            row[0]
            for row in conn.execute(text("SELECT value FROM allowed ORDER BY id"))
        ]

    assert values == ["a", "b"]

    engine.dispose()
