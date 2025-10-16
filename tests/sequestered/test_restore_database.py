# This test is temporarily sequestered due to persistent, hard-to-debug failures
# related to mocked `subprocess` calls.
import subprocess
from pathlib import Path

import pytest

from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager


def _create_sqlite_manager(tmp_path: Path) -> SQLAlchemyDatabaseManager:
    db_path = tmp_path / "test.db"
    db_path.touch()
    return SQLAlchemyDatabaseManager(f"sqlite:///{db_path}")


def test_restore_sqlite(tmp_path: Path):
    manager = _create_sqlite_manager(tmp_path)
    backup = tmp_path / "backup.db"
    backup.write_text("backup_content")
    manager.restore_database(backup)
    assert Path(manager.engine.url.database).read_text() == "backup_content"


def test_restore_postgresql(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = _create_sqlite_manager(tmp_path)
    manager.database_type = "postgresql"
    manager.database_connect = "postgresql://user:pass@localhost/db"
    called = {}

    def fake_run(cmd, stdin=None, stdout=None, stderr=None):
        called["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    src = tmp_path / "pg_dump.sql"
    src.write_text("data")
    manager.restore_database(src)
    assert called["cmd"] == ["psql", manager.database_connect]


def test_restore_mysql(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = _create_sqlite_manager(tmp_path)
    manager.database_type = "mysql"
    manager.database_connect = "mysql+mysqlconnector://user:pass@localhost/db"
    called = {}

    def fake_run(cmd, stdin=None, stdout=None, stderr=None):
        called["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    src = tmp_path / "mysqldump.sql"
    src.write_text("data")
    manager.restore_database(src)
    assert called["cmd"] == [
        "mysql",
        "--host=localhost",
        "--user=user",
        "--password=pass",
        "db",
    ]
