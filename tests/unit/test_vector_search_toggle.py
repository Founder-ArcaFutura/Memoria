from datetime import datetime

from memoria.agents.search_executor import SearchExecutor
from memoria.config.manager import ConfigManager
from memoria.utils.pydantic_models import MemorySearchQuery


class StubDBManager:
    def __init__(self, results):
        self.results = list(results)

    def search_memories(self, query, namespace, limit, use_fuzzy=False, **kwargs):
        return {"results": list(self.results)}


def _reset_vector_search(original: bool) -> None:
    ConfigManager().update_setting("enable_vector_search", original)


def test_vector_search_prioritizes_semantic_match(monkeypatch):
    cfg = ConfigManager()
    original = getattr(cfg.get_settings(), "enable_vector_search", False)
    cfg.update_setting("enable_vector_search", True)
    try:
        now = datetime.utcnow().isoformat()
        results = [
            {
                "memory_id": "semantic",
                "importance_score": 0.1,
                "created_at": now,
                "embedding": [1.0, 0.0],
                "memory_type": "long_term",
                "summary": "semantic",
            },
            {
                "memory_id": "important",
                "importance_score": 0.9,
                "created_at": now,
                "embedding": [0.0, 1.0],
                "memory_type": "long_term",
                "summary": "important",
            },
        ]
        db_manager = StubDBManager(results)
        executor = SearchExecutor()
        monkeypatch.setattr(executor, "_embed_text", lambda text: [1.0, 0.0])
        plan = MemorySearchQuery(query_text="hello", intent="general")

        ranked = executor.execute_search("hello", plan, db_manager)
        assert ranked[0]["memory_id"] == "semantic"
    finally:
        _reset_vector_search(original)


def test_vector_search_falls_back_when_disabled(monkeypatch):
    cfg = ConfigManager()
    original = getattr(cfg.get_settings(), "enable_vector_search", False)
    cfg.update_setting("enable_vector_search", False)
    try:
        now = datetime.utcnow().isoformat()
        results = [
            {
                "memory_id": "semantic",
                "importance_score": 0.1,
                "created_at": now,
                "embedding": [1.0, 0.0],
                "memory_type": "long_term",
                "summary": "semantic",
            },
            {
                "memory_id": "important",
                "importance_score": 0.9,
                "created_at": now,
                "embedding": [0.0, 1.0],
                "memory_type": "long_term",
                "summary": "important",
            },
        ]
        db_manager = StubDBManager(results)
        executor = SearchExecutor()
        monkeypatch.setattr(executor, "_embed_text", lambda text: [1.0, 0.0])
        plan = MemorySearchQuery(query_text="hello", intent="general")

        ranked = executor.execute_search("hello", plan, db_manager)
        assert ranked[0]["memory_id"] == "important"
    finally:
        _reset_vector_search(original)
