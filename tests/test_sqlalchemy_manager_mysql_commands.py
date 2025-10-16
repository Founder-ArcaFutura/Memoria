import subprocess
from unittest.mock import patch

import pytest
from sqlalchemy.engine.url import make_url

from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager


def _build_mysql_manager(connection_string: str) -> SQLAlchemyDatabaseManager:
    manager = object.__new__(SQLAlchemyDatabaseManager)
    manager.database_type = "mysql"
    manager.database_connect = connection_string
    return manager


@pytest.mark.parametrize(
    "connection_string,expected_flags",
    [
        (
            "mysql+pymysql://test_user:test_pass@db.example.com:3307/test_db",
            [
                "--host=db.example.com",
                "--port=3307",
                "--user=test_user",
                "--password=test_pass",
            ],
        ),
        (
            "mysql://user_only@localhost/test_schema",
            [
                "--host=localhost",
                "--user=user_only",
            ],
        ),
    ],
)
def test_backup_database_mysql_command(tmp_path, connection_string, expected_flags):
    manager = _build_mysql_manager(connection_string)
    destination = tmp_path / "backup.sql"

    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=b"", stderr=b""
    )

    with patch(
        "memoria.database.sqlalchemy_manager.subprocess.run", return_value=completed
    ) as mock_run:
        manager.backup_database(destination)

    database_name = make_url(connection_string).database
    expected_command = ["mysqldump", *expected_flags, database_name]
    actual_command = mock_run.call_args[0][0]

    assert actual_command == expected_command


@pytest.mark.parametrize(
    "connection_string,expected_flags",
    [
        (
            "mysql+pymysql://test_user:test_pass@db.example.com:3307/test_db",
            [
                "--host=db.example.com",
                "--port=3307",
                "--user=test_user",
                "--password=test_pass",
            ],
        ),
        (
            "mysql://user_only@localhost/test_schema",
            [
                "--host=localhost",
                "--user=user_only",
            ],
        ),
    ],
)
def test_restore_database_mysql_command(tmp_path, connection_string, expected_flags):
    manager = _build_mysql_manager(connection_string)
    backup_path = tmp_path / "backup.sql"
    backup_path.write_bytes(b"test")

    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=b"", stderr=b""
    )

    with patch(
        "memoria.database.sqlalchemy_manager.subprocess.run", return_value=completed
    ) as mock_run:
        manager.restore_database(backup_path)

    database_name = make_url(connection_string).database
    expected_command = ["mysql", *expected_flags, database_name]
    actual_command = mock_run.call_args[0][0]

    assert actual_command == expected_command
