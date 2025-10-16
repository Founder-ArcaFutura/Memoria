import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager


def test_search_memories_fallback_returns_structured(monkeypatch):
    manager = SQLAlchemyDatabaseManager(
        database_connect="sqlite:///:memory:", enable_short_term=False
    )

    class DummySearchService:
        def __init__(self):
            self.calls = 0
            self.session = types.SimpleNamespace(close=lambda: None)

        def search_memories(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("primary failed")
            return []

    monkeypatch.setattr(
        manager, "_get_search_service", lambda **kwargs: DummySearchService()
    )

    result = manager.search_memories("test")

    assert isinstance(result, dict)
    assert result["results"] == []
    assert "hint" in result
    assert "error" in result
