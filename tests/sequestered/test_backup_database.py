# This test is temporarily sequestered due to persistent, hard-to-debug failures
# related to mocked `subprocess` calls.
import subprocess
from pathlib import Path

import pytest

from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager


def _create_sqlite_manager(tmp_path: Path) -> SQLAlchemyDatabaseManager:
    db_path = tmp_path / "test.db"
    return SQLAlchemyDatabaseManager(f"sqlite:///{db_path}")


def _fake_run_factory():
    def _fake_run(cmd, stdout=None, stderr=None, check=False):
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    return _fake_run


def test_backup_sqlite(tmp_path: Path):
    manager = _create_sqlite_manager(tmp_path)
    dest = tmp_path / "backup.db"
    manager.backup_database(dest)
    assert dest.exists()


def test_backup_postgresql(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = _create_sqlite_manager(tmp_path)
    manager.database_type = "postgresql"
    manager.database_connect = "postgresql://user:pass@localhost/db"
    monkeypatch.setattr(subprocess, "run", _fake_run_factory())
    dest = tmp_path / "pg_dump.sql"
    manager.backup_database(dest)


def test_backup_mysql(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = _create_sqlite_manager(tmp_path)
    manager.database_type = "mysql"
    manager.database_connect = "mysql+mysqlconnector://user:pass@localhost/db"
    monkeypatch.setattr(subprocess, "run", _fake_run_factory())
    dest = tmp_path / "mysqldump.sql"
    manager.backup_database(dest)
