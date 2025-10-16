import pytest
from sqlalchemy import text

from memoria.database.connectors.base_connector import DatabaseType
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager
from memoria.utils.transaction_manager import (
    TransactionManager,
    bulk_insert_transaction,
)


class _RecordingCursor:
    """Lightweight cursor mock that records executed statements."""

    def __init__(self, executed_queries):
        self._executed = executed_queries
        self.rowcount = 0
        self.description = None

    def execute(self, query, params=None):
        normalized = query.strip().upper()
        if normalized == "BEGIN":
            return

        self._executed.append((query, params))
        self.rowcount = 1

    def fetchall(self):
        return []


class _RecordingConnection:
    """Mock connection that exposes cursor/commit/rollback hooks."""

    def __init__(self, database_type):
        self.database_type = database_type
        self.executed = []
        self.commits = 0
        self.rollbacks = 0
        self.closes = 0

    def cursor(self):
        return _RecordingCursor(self.executed)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closes += 1


class _RecordingConnector:
    """Connector stub that mimics database connectors for placeholder tests."""

    def __init__(self, database_type):
        self.database_type = database_type
        self._connection = _RecordingConnection(database_type)

    def get_connection(self):
        return self._connection


class _FailingConnector:
    """Connector stub whose connection acquisition always fails."""

    database_type = DatabaseType.SQLITE

    def get_connection(self):
        raise RuntimeError("connector unavailable")


class _SQLAlchemyAdapter:
    """Simple adapter exposing DB-API connection for the transaction manager."""

    def __init__(self, manager: SQLAlchemyDatabaseManager):
        self._manager = manager
        self.database_type = manager.database_type

    def get_connection(self):
        return self._manager.engine.raw_connection()


class _SQLAlchemyContextAdapter:
    """Adapter that returns a context manager producing DB-API connections."""

    def __init__(self, manager: SQLAlchemyDatabaseManager):
        self._manager = manager
        self.database_type = manager.database_type

        from contextlib import contextmanager

        @contextmanager
        def connection_context():
            conn = self._manager.engine.raw_connection()
            try:
                yield conn
            finally:
                conn.close()

        self._context_factory = connection_context

    def get_connection(self):
        return self._context_factory()


@pytest.mark.parametrize(
    "database_type, expected_placeholder",
    [
        (DatabaseType.SQLITE, "?"),
        ("sqlite", "?"),
        (DatabaseType.POSTGRESQL, "%s"),
        ("postgresql", "%s"),
        ("mysql", "%s"),
    ],
)
def test_bulk_insert_transaction_placeholder_normalization(
    database_type, expected_placeholder
):
    connector = _RecordingConnector(database_type)

    result = bulk_insert_transaction(
        connector,
        "test_table",
        [{"value": "alpha"}],
        batch_size=1,
    )

    assert result.success

    insert_statements = [
        query
        for query, _ in connector._connection.executed
        if "INSERT" in query.upper()
    ]
    assert insert_statements, "Expected INSERT statements to be recorded"
    assert expected_placeholder in insert_statements[0]


def test_bulk_insert_transaction_sqlalchemy_sqlite(tmp_path):
    db_path = tmp_path / "transaction_manager.db"
    manager = SQLAlchemyDatabaseManager(
        f"sqlite:///{db_path}",
        enable_short_term=False,
    )

    try:
        with manager.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE test_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        value TEXT NOT NULL
                    )
                    """
                )
            )

        adapter = _SQLAlchemyAdapter(manager)
        result = bulk_insert_transaction(
            adapter,
            "test_entries",
            [{"value": "one"}, {"value": "two"}],
            batch_size=1,
        )

        assert result.success
        assert result.operations_completed == 2

        with manager.engine.begin() as connection:
            rows = [
                row[0]
                for row in connection.execute(
                    text("SELECT value FROM test_entries ORDER BY id")
                )
            ]

        assert rows == ["one", "two"]
    finally:
        if getattr(manager, "_scheduler", None):
            manager._scheduler.shutdown(wait=False)
        manager.engine.dispose()


def test_transaction_manager_handles_raw_connection_cleanup():
    connector = _RecordingConnector(DatabaseType.SQLITE)
    manager = TransactionManager(connector)

    with manager.transaction() as tx:
        tx.execute("UPDATE dummy SET value = 1")

    assert connector._connection.commits == 1
    assert connector._connection.rollbacks == 0
    assert connector._connection.closes == 1


def test_transaction_manager_accepts_context_manager_connector(tmp_path):
    db_path = tmp_path / "transaction_manager_ctx.db"
    manager = SQLAlchemyDatabaseManager(
        f"sqlite:///{db_path}",
        enable_short_term=False,
    )

    adapter = _SQLAlchemyContextAdapter(manager)
    txn_manager = TransactionManager(adapter)

    try:
        with txn_manager.transaction() as tx:
            tx.execute(
                """
                CREATE TABLE IF NOT EXISTS ctx_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    value TEXT NOT NULL
                )
                """
            )
            tx.execute(
                "INSERT INTO ctx_entries (value) VALUES (?)",
                ["alpha"],
            )

        with manager.engine.begin() as connection:
            rows = [
                row[0]
                for row in connection.execute(text("SELECT value FROM ctx_entries"))
            ]

        assert rows == ["alpha"]
    finally:
        if getattr(manager, "_scheduler", None):
            manager._scheduler.shutdown(wait=False)
        manager.engine.dispose()


def test_transaction_manager_propagates_connector_errors():
    failing_connector = _FailingConnector()
    manager = TransactionManager(failing_connector)

    with pytest.raises(RuntimeError) as exc_info:
        with manager.transaction():
            pass

    assert "connector unavailable" in str(exc_info.value)
