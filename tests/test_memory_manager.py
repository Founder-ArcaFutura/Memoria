import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memoria.config.manager import ConfigManager
from memoria.config.memory_manager import MemoryManager


class FakeMemoria:
    def __init__(
        self,
        *,
        namespace: str = "default",
        storage_service: Any | None = None,
        db_manager: Any | None = None,
    ):
        self.namespace = namespace
        self.storage_service = storage_service
        self.db_manager = db_manager
        self.sovereign_ingest = False
        self.recorded_calls: list[dict[str, Any]] = []

    def record_conversation(
        self,
        *,
        user_input: str,
        ai_output: Any = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.recorded_calls.append(
            {
                "user_input": user_input,
                "ai_output": ai_output,
                "model": model,
                "metadata": metadata,
            }
        )
        return "bound-chat-id"


class FakeDBManager:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def store_chat_history(
        self,
        *,
        chat_id: str,
        user_input: str,
        ai_output: str,
        timestamp,
        session_id: str,
        model: str,
        namespace: str,
        metadata: dict[str, Any],
    ) -> None:
        self.calls.append(
            {
                "chat_id": chat_id,
                "user_input": user_input,
                "ai_output": ai_output,
                "timestamp": timestamp,
                "session_id": session_id,
                "model": model,
                "namespace": namespace,
                "metadata": metadata,
            }
        )


class SearchRecordingDBManager(FakeDBManager):
    def __init__(self, results: list[dict[str, Any]]):
        super().__init__()
        self.search_results = results
        self.search_calls: list[dict[str, Any]] = []

    def search_memories(self, **kwargs):
        self.search_calls.append(kwargs)
        return {"results": list(self.search_results)}


class FakeStorageService:
    def __init__(self, results: list[dict[str, Any]], namespace: str = "storage-ns"):
        self.results = results
        self.namespace = namespace
        self.calls: list[dict[str, Any]] = []

    def search_memories(self, **kwargs):
        self.calls.append(kwargs)
        return {"results": list(self.results)}


class IncompleteMemoria:
    """A Memoria-like object missing record_conversation for fallback testing."""

    def __init__(self, namespace: str = "fallback") -> None:
        self.namespace = namespace
        self.sovereign_ingest = False
        self.storage_service = None
        self.db_manager = None


def test_record_conversation_delegates_to_memoria_instance():
    manager = MemoryManager()
    fake_memoria = FakeMemoria()
    manager.set_memoria_instance(fake_memoria)

    chat_id = manager.record_conversation(user_input="hello", ai_output="hi there")

    settings = ConfigManager().get_settings()
    assert chat_id == "bound-chat-id"
    assert fake_memoria.recorded_calls == [
        {
            "user_input": "hello",
            "ai_output": "hi there",
            "model": settings.agents.default_model,
            "metadata": {},
        }
    ]


def test_memory_filters_mapping_is_preserved():
    filters = {"importance_threshold": 0.5, "namespaces": ["alpha", "beta"]}

    manager = MemoryManager(memory_filters=filters)

    # Ensure an internal copy is created so downstream changes don't mutate input
    assert manager.memory_filters == filters
    assert manager.memory_filters is not filters

    health = manager.get_health()
    assert health["memory_filters"]["configured"] is True
    assert health["memory_filters"]["count"] == len(filters)
    assert health["memory_filters"]["values"] == filters


def test_record_conversation_falls_back_to_db_manager():
    manager = MemoryManager(namespace="custom")
    manager.db_manager = FakeDBManager()

    chat_id = manager.record_conversation(user_input="hello", ai_output="hi there")

    settings = ConfigManager().get_settings()
    assert len(manager.db_manager.calls) == 1
    call = manager.db_manager.calls[0]
    assert call["chat_id"] == chat_id
    assert call["user_input"] == "hello"
    assert call["ai_output"] == "hi there"
    assert call["model"] == settings.agents.default_model
    assert call["namespace"] == "custom"
    assert call["metadata"] == {}


def test_record_conversation_falls_back_when_memoria_missing_method():
    manager = MemoryManager(namespace="fallback")
    manager.set_memoria_instance(IncompleteMemoria())
    manager.db_manager = FakeDBManager()

    chat_id = manager.record_conversation(user_input="ping", ai_output="pong")

    call = manager.db_manager.calls[0]
    assert call["chat_id"] == chat_id
    assert call["namespace"] == "fallback"


def test_record_conversation_without_persistence_path_raises():
    manager = MemoryManager()
    with pytest.raises(RuntimeError):
        manager.record_conversation(user_input="hello", ai_output="hi there")


def test_search_memories_delegates_to_db_manager_and_normalizes():
    search_results = [{"memory_id": "1"}, {"memory_id": "2"}]
    db_manager = SearchRecordingDBManager(search_results)
    fake_memoria = FakeMemoria(namespace="memoria-ns", db_manager=db_manager)

    manager = MemoryManager(namespace="preferred")
    manager.set_memoria_instance(fake_memoria)

    results = manager.search_memories(
        query="status update",
        limit=2,
        memory_types=["long_term"],
        categories=["work"],
        min_importance=0.4,
    )

    assert results == search_results
    assert len(db_manager.search_calls) == 1
    call = db_manager.search_calls[0]
    assert call["query"] == "status update"
    assert call["limit"] == 2
    assert call["namespace"] == "preferred"
    assert call["memory_types"] == ["long_term"]
    assert call["category_filter"] == ["work"]
    assert call["min_importance"] == 0.4


def test_search_memories_prefers_storage_service_namespace():
    search_results = [{"memory_id": "storage"}]
    storage_service = FakeStorageService(search_results, namespace="storage-namespace")
    fake_memoria = FakeMemoria(storage_service=storage_service)

    manager = MemoryManager()
    manager.set_memoria_instance(fake_memoria)

    results = manager.search_memories(
        query="note",
        limit=3,
        memory_types=["short_term"],
        categories=["personal"],
        min_importance=0.1,
    )

    assert results == search_results
    assert len(storage_service.calls) == 1
    call = storage_service.calls[0]
    assert call["namespace"] == "storage-namespace"
    assert call["query"] == "note"
    assert call["limit"] == 3
    assert call["memory_types"] == ["short_term"]
    assert call["category_filter"] == ["personal"]
    assert call["min_importance"] == 0.1
