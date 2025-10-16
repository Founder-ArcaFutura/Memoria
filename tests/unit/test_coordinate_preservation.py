import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import openai

from memoria.agents.retrieval_agent import MemorySearchEngine
from memoria.tools.memory_tool import MemoryTool
from memoria.utils.pydantic_models import MemorySearchQuery


class DummyDBManager:
    def search_memories(self, *args, **kwargs):
        return [
            {
                "memory_id": "1",
                "summary": "Test",
                "importance_score": 0.5,
                "created_at": "2023-01-01T00:00:00",
                "x_coord": 1.0,
                "y_coord": 2.0,
                "z_coord": 3.0,
            }
        ]


def test_search_engine_preserves_coordinates(monkeypatch):
    class DummyOpenAI:
        def __init__(self, api_key=None):
            pass

    monkeypatch.setattr(openai, "OpenAI", DummyOpenAI)

    engine = MemorySearchEngine()
    engine.plan_search = lambda query, **kwargs: MemorySearchQuery(
        query_text=query, intent="test"
    )
    db = DummyDBManager()

    response = engine.execute_search("test", db)
    results = response.get("results", [])

    assert results
    assert results[0]["x"] == 1.0
    assert results[0]["y"] == 2.0
    assert results[0]["z"] == 3.0


def test_memory_tool_formats_coordinates(monkeypatch):
    class DummySearchEngine:
        def __init__(self, *args, **kwargs):
            pass

        def execute_search(self, query, db_manager, namespace, limit):
            return {
                "results": [
                    {
                        "summary": "Test",
                        "importance_score": 0.5,
                        "created_at": "2023-01-01T00:00:00",
                        "search_reasoning": "reason",
                        "x": 1.0,
                        "y": 2.0,
                        "z": 3.0,
                    }
                ],
                "hint": None,
                "error": None,
            }

    monkeypatch.setattr(
        "memoria.agents.retrieval_agent.MemorySearchEngine", DummySearchEngine
    )

    class DummyMemoria:
        def __init__(self):
            self.db_manager = None
            self.namespace = "default"
            self.provider_config = None

    tool = MemoryTool(DummyMemoria())
    output = tool.execute(query="test")

    assert "X: 1.00" in output
    assert "Y: 2.00" in output
    assert "Z: 3.00" in output
