import pytest

from memoria.utils.exceptions import DatabaseError, SecurityError
from memoria.utils.pydantic_models import AgentPermissions
from memoria.utils.query_builder import DatabaseDialect
from memoria.utils.security_integration import SecureMemoriaDatabase


class DummyCursor:
    def __init__(self, executed_queries: list[tuple[str, tuple]]):
        self._executed_queries = executed_queries
        self.rowcount = 1
        self.description = None
        self._last_query = ""

    def execute(self, query: str, params=None):
        self._last_query = query
        normalized = query.strip().upper()
        if normalized in {"BEGIN", "SET TRANSACTION READ ONLY"}:
            return

        bound_params = tuple(params or [])
        self._executed_queries.append((query, bound_params))

        if normalized.startswith("SELECT"):
            self.description = [("dummy", None, None, None, None, None, None)]
        else:
            self.description = None
            self.rowcount = 1

    def fetchall(self):
        if self._last_query.strip().upper().startswith("SELECT"):
            return []
        return []


class DummyConnection:
    def __init__(self, executed_queries: list[tuple[str, tuple]]):
        self._executed_queries = executed_queries
        self.committed = False
        self.rolled_back = False
        self.autocommit = False

    def cursor(self):
        return DummyCursor(self._executed_queries)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class DummyConnector:
    def __init__(self):
        self.executed_queries: list[tuple[str, tuple]] = []

    def execute_query(self, query: str, params: list):
        self.executed_queries.append((query, tuple(params)))
        return [{"id": 1}]

    def get_connection(self):
        return DummyConnection(self.executed_queries)


@pytest.fixture
def secure_db() -> SecureMemoriaDatabase:
    connector = DummyConnector()
    return SecureMemoriaDatabase(connector, DatabaseDialect.SQLITE)


def test_secure_insert_executes_transaction(secure_db: SecureMemoriaDatabase) -> None:
    result = secure_db.secure_insert("memories", {"title": "hello"})

    assert result == "1"
    connector_queries = secure_db.connector.executed_queries
    assert any(
        query.startswith("INSERT OR REPLACE INTO memories")
        for query, _ in connector_queries
    )


def test_secure_insert_blocked_without_permission(
    secure_db: SecureMemoriaDatabase,
) -> None:
    with pytest.raises(SecurityError):
        secure_db.secure_insert(
            "memories",
            {"title": "hello"},
            permissions=AgentPermissions(can_write=False),
        )


def test_security_audit_blocks_sensitive_insert(
    secure_db: SecureMemoriaDatabase,
) -> None:
    with pytest.raises(DatabaseError) as exc_info:
        secure_db.secure_insert("accounts", {"password": "secret"})

    assert "SECURITY_ERROR" in str(exc_info.value)


def test_secure_delete_executes_with_permissions(
    secure_db: SecureMemoriaDatabase,
) -> None:
    affected = secure_db.secure_delete(
        "memories", {"id": 1}, permissions=AgentPermissions(can_edit=True)
    )

    assert affected == 1
    connector_queries = secure_db.connector.executed_queries
    assert any(
        query.startswith("DELETE FROM memories") for query, _ in connector_queries
    )
