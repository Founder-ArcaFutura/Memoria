import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.agents.search_executor import SearchExecutor
from memoria.utils.pydantic_models import MemorySearchQuery


class FuzzyDBManager:
    def __init__(self):
        self.calls = []
        self.now = datetime.utcnow()

    def search_memories(self, query, namespace, limit, use_fuzzy=False, **kwargs):
        self.calls.append(use_fuzzy)
        if use_fuzzy:
            return [
                {
                    "memory_id": "fuzzy",
                    "importance_score": 0.5,
                    "created_at": self.now.isoformat(),
                }
            ]
        return []


def test_execute_search_falls_back_to_fuzzy():
    executor = SearchExecutor()
    search_plan = MemorySearchQuery(query_text="test", intent="test")
    db = FuzzyDBManager()

    results = executor.execute_search("test", search_plan, db)

    assert db.calls == [False, True]
    assert [r["memory_id"] for r in results] == ["fuzzy"]


class EmptyDBManager:
    def __init__(self):
        self.calls = []

    def search_memories(self, query, namespace, limit, use_fuzzy=False, **kwargs):
        self.calls.append(use_fuzzy)
        return []


def test_execute_search_suggests_query_when_empty():
    executor = SearchExecutor()
    search_plan = MemorySearchQuery(query_text="complicated", intent="test")
    db = EmptyDBManager()

    results = executor.execute_search("complicated", search_plan, db)

    assert db.calls == [False, True]
    assert len(results) == 1
    assert results[0]["suggested_query"].startswith("Try: anchor:")
