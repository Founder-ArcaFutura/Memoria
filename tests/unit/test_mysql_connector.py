from __future__ import annotations

import sys
import types

import pytest

from memoria.database.connectors.mysql_connector import MySQLConnector


class _DummyCursor:
    def __init__(self, connection: _DummyConnection, dictionary: bool):
        self.connection = connection
        self.dictionary = dictionary
        self._results = connection.results
        self.executed: list[tuple[str, list | tuple | None]] = []
        self.closed = False

    def execute(self, query: str, params=None) -> None:
        self.executed.append((query, params))

    def fetchall(self):
        return self._results

    def close(self) -> None:
        self.closed = True

    @property
    def lastrowid(self) -> int:
        return 1

    @property
    def rowcount(self) -> int:
        return len(self.executed)


class _DummyConnection:
    def __init__(self, config: dict[str, object], results: list[dict[str, object]]):
        self.config = config
        self.results = list(results)
        self.cursor_calls: list[_DummyCursor] = []
        self.closed = False
        self.commits = 0
        self.transactions_started = 0
        self.rollbacks = 0

    def __enter__(self) -> _DummyConnection:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.closed = True

    def cursor(self, dictionary: bool = False) -> _DummyCursor:
        cursor = _DummyCursor(self, dictionary)
        self.cursor_calls.append(cursor)
        return cursor

    def commit(self) -> None:
        self.commits += 1

    def start_transaction(self) -> None:
        self.transactions_started += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def stub_mysql_module(monkeypatch):
    connections: list[_DummyConnection] = []
    results = [{"value": 42}]

    def connect(**config):
        connection = _DummyConnection(config, results)
        connections.append(connection)
        return connection

    mysql_module = types.ModuleType("mysql")
    connector_module = types.ModuleType("mysql.connector")
    connector_module.connect = connect
    mysql_module.connector = connector_module

    monkeypatch.setitem(sys.modules, "mysql", mysql_module)
    monkeypatch.setitem(sys.modules, "mysql.connector", connector_module)

    yield {"connections": connections, "results": results}

    monkeypatch.delitem(sys.modules, "mysql.connector", raising=False)
    monkeypatch.delitem(sys.modules, "mysql", raising=False)


def test_parse_connection_string_with_pymysql_driver(stub_mysql_module):
    connection_string = "mysql+pymysql://user:pass@localhost:3306/test_db"
    connector = MySQLConnector(connection_string)

    parsed = connector._parse_connection_string(connection_string)

    assert parsed == {
        "host": "localhost",
        "port": 3306,
        "user": "user",
        "password": "pass",
        "database": "test_db",
    }


def test_parse_connection_string_with_mysqlconnector_driver(stub_mysql_module):
    connection_string = "mysql+mysqlconnector://user:secret@db.example.com/sample"
    connector = MySQLConnector(connection_string)

    parsed = connector._parse_connection_string(connection_string)

    assert parsed == {
        "host": "db.example.com",
        "port": 3306,
        "user": "user",
        "password": "secret",
        "database": "sample",
    }


def test_dictionary_configuration_is_used_directly(stub_mysql_module):
    config = {
        "host": "127.0.0.1",
        "port": 3307,
        "user": "memoria",
        "password": "memoria_pass",
        "database": "memoria",
    }
    connector = MySQLConnector(config)

    connection = connector.get_connection()

    assert connection.config["host"] == "127.0.0.1"
    assert connection.config["port"] == 3307
    assert connection.config["user"] == "memoria"
    assert connection.config["password"] == "memoria_pass"
    assert connection.config["database"] == "memoria"


def test_execute_query_with_driver_qualified_url(stub_mysql_module):
    connector = MySQLConnector("mysql+mysqlconnector://user:pass@localhost/testdb")

    results = connector.execute_query("SELECT 1")

    assert results == stub_mysql_module["results"]

    connection = stub_mysql_module["connections"][-1]
    assert connection.config["charset"] == "utf8mb4"
    assert connection.config["collation"] == "utf8mb4_unicode_ci"
    assert connection.config["autocommit"] is False
    assert connection.config["use_pure"] is True

    # First cursor used for session configuration should not request dictionary rows.
    assert not connection.cursor_calls[0].dictionary
    # Final cursor executing the query should request dictionary rows and execute the query.
    assert connection.cursor_calls[-1].dictionary is True
    assert connection.cursor_calls[-1].executed == [("SELECT 1", None)]
