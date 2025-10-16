import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.agents.search_executor import SearchExecutor
from memoria.utils.pydantic_models import MemorySearchQuery


class DummyDBManager:
    def __init__(self):
        self.now = datetime.utcnow()

    def search_memories(self, *args, **kwargs):
        return [
            {
                "memory_id": "older",
                "importance_score": 0.5,
                "created_at": (self.now - timedelta(hours=12)).isoformat(),
            },
            {
                "memory_id": "newer",
                "importance_score": 0.5,
                "created_at": self.now.isoformat(),
            },
        ]


def test_execute_search_sorts_by_fractional_recency():
    executor = SearchExecutor()
    search_plan = MemorySearchQuery(query_text="test", intent="test")
    db = DummyDBManager()

    results = executor.execute_search("test", search_plan, db)

    assert [r["memory_id"] for r in results][:2] == ["newer", "older"]
