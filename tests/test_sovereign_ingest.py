import sys
from types import SimpleNamespace

from memoria import Memoria
from memoria.config import ConfigManager
from memoria.storage.service import StorageService


class DummyDBManager:
    def __init__(self):
        self.calls = []

    def search_memories(self, query, namespace, limit, **kwargs):
        self.calls.append({"query": query, "namespace": namespace, "limit": limit})
        return [
            {
                "memory_id": "db-result",
                "searchable_content": "fallback",
            }
        ]


def test_sovereign_mode_skips_llm_initialization(monkeypatch):
    def _raise(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("LLM agent should not be constructed in sovereign mode")

    monkeypatch.setattr("memoria.agents.memory_agent.MemoryAgent", _raise)
    monkeypatch.setattr("memoria.agents.retrieval_agent.MemorySearchEngine", _raise)

    mem = Memoria(database_connect="sqlite:///:memory:", sovereign_ingest=True)

    assert mem.memory_agent is None
    assert mem.search_engine is None
    assert mem.storage_service.search_engine is None


def test_config_sovereign_mode_skips_llm_initialization(monkeypatch, request):
    manager = ConfigManager.get_instance()
    settings = manager.get_settings()
    original_flag = settings.memory.sovereign_ingest
    settings.memory.sovereign_ingest = True

    def restore():
        settings.memory.sovereign_ingest = original_flag

    request.addfinalizer(restore)

    def _raise(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("LLM agent should not be constructed in sovereign mode")

    monkeypatch.setattr("memoria.agents.memory_agent.MemoryAgent", _raise)
    monkeypatch.setattr("memoria.agents.retrieval_agent.MemorySearchEngine", _raise)

    mem = Memoria(database_connect="sqlite:///:memory:")

    assert mem.sovereign_ingest is True
    assert mem.memory_agent is None
    assert mem.search_engine is None
    assert mem.storage_service.search_engine is None


def test_non_sovereign_initializes_llm_agents(monkeypatch):
    dummy_openai = SimpleNamespace(
        OpenAI=lambda *args, **kwargs: None,
        AsyncOpenAI=lambda *args, **kwargs: None,
        AzureOpenAI=lambda *args, **kwargs: None,
        AsyncAzureOpenAI=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "openai", dummy_openai)

    class DummyAgent:
        def __init__(self, *args, **kwargs):
            DummyAgent.created += 1

    class DummySearch:
        def __init__(self, *args, **kwargs):
            DummySearch.created += 1

        def execute_search(self, *args, **kwargs):  # pragma: no cover - defensive
            return {"results": []}

    DummyAgent.created = 0
    DummySearch.created = 0

    monkeypatch.setattr("memoria.agents.memory_agent.MemoryAgent", DummyAgent)
    monkeypatch.setattr(
        "memoria.agents.retrieval_agent.MemorySearchEngine", DummySearch
    )

    mem = Memoria(database_connect="sqlite:///:memory:", sovereign_ingest=False)

    assert isinstance(mem.memory_agent, DummyAgent)
    assert isinstance(mem.search_engine, DummySearch)
    assert DummyAgent.created == 1
    assert DummySearch.created == 1
    assert mem.storage_service.search_engine is mem.search_engine


def test_storage_service_falls_back_to_db_when_no_search_engine():
    db_manager = DummyDBManager()
    service = StorageService(
        db_manager=db_manager,
        namespace="default",
        search_engine=None,
        conscious_ingest=False,
    )

    context = service.retrieve_context("hello", limit=2)

    assert db_manager.calls and db_manager.calls[0]["query"] == "hello"
    assert context and context[0]["memory_id"] == "db-result"
