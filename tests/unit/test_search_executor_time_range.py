import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.agents.search_executor import SearchExecutor
from memoria.utils.pydantic_models import MemorySearchQuery


class RecordingDB:
    def __init__(self):
        self.calls = []

        self.retrieve_calls = []
        self.dataset = [
            {
                "memory_id": "before",
                "text": "project kickoff",
                "created_at": "2023-12-31T10:00:00",
                "importance_score": 0.3,
                "x": -10.0,
            },
            {
                "memory_id": "inside",
                "text": "project milestone",
                "created_at": "2024-01-01T12:00:00",
                "importance_score": 0.8,
                "x": -3.0,
            },
            {
                "memory_id": "after",
                "text": "project wrap up",
                "created_at": "2024-01-03T09:00:00",
                "importance_score": 0.6,
                "x": 2.0,
            },
        ]

    def search_memories(self, query, namespace, limit, use_fuzzy=False, **kwargs):
        start = kwargs.get("start_timestamp")
        end = kwargs.get("end_timestamp")

        self.calls.append(
            {
                "start_timestamp": start,
                "end_timestamp": end,
                "use_fuzzy": use_fuzzy,
            }
        )
        results = []
        for item in self.dataset:
            if query and "project" not in item["text"]:
                continue
            created_at = datetime.fromisoformat(item["created_at"])
            if start and created_at < start:
                continue
            if end and created_at > end:
                continue
            results.append(dict(item))
        return results[:limit]

    def retrieve_memories_by_time_range(
        self,
        start_timestamp=None,
        end_timestamp=None,
        start_x=None,
        end_x=None,
    ):
        self.retrieve_calls.append(
            {
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "start_x": start_x,
                "end_x": end_x,
            }
        )
        filtered = []
        for item in self.dataset:
            if start_x is not None and item["x"] < start_x:
                continue
            if end_x is not None and item["x"] > end_x:
                continue
            filtered.append({"memory_id": item["memory_id"]})
        return filtered


def test_execute_search_forwards_timestamp_bounds():
    executor = SearchExecutor()
    search_plan = MemorySearchQuery(
        query_text="project",
        intent="test",
        time_range="2024-01-01..2024-01-02",
    )
    db = RecordingDB()

    results = executor.execute_search("project", search_plan, db, namespace="default")

    assert db.calls, "search_memories should be invoked"
    start = db.calls[0]["start_timestamp"]
    end = db.calls[0]["end_timestamp"]
    assert start == datetime(2024, 1, 1)
    assert end >= datetime(2024, 1, 2)
    returned_ids = [item["memory_id"] for item in results if "memory_id" in item]
    assert returned_ids == ["inside"], "results should be limited to the provided range"


def test_execute_search_filters_by_x_range():
    executor = SearchExecutor()
    search_plan = MemorySearchQuery(
        query_text="project",
        intent="test",
        time_range="x:-5..0",
    )
    db = RecordingDB()

    results = executor.execute_search("project", search_plan, db, namespace="default")

    assert (
        db.retrieve_calls
    ), "retrieve_memories_by_time_range should be used for x bounds"
    coords = db.retrieve_calls[0]
    assert coords["start_x"] == -5
    assert coords["end_x"] == 0
    returned_ids = [item["memory_id"] for item in results if "memory_id" in item]
    assert returned_ids == ["inside"], "only memories within the x range should remain"
