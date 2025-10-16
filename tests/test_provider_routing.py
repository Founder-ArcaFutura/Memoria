import asyncio
from datetime import datetime
from types import SimpleNamespace

from memoria.agents.memory_agent import MemoryAgent
from memoria.agents.retrieval_agent import MemorySearchEngine
from memoria.agents.search_executor import SearchExecutor
from memoria.core import provider_setup
from memoria.core.conversation import ConversationManager
from memoria.core.providers import (
    ProviderRegistry,
    ProviderRouteMetadata,
    ProviderType,
    ProviderUnavailableError,
    RegisteredProvider,
    TaskRouteSpec,
)
from memoria.utils.pydantic_models import (
    MemoryClassification,
    MemoryImportanceLevel,
    MemorySearchQuery,
    ProcessedLongTermMemory,
)


def test_memory_agent_falls_back_to_openai_when_anthropic_unsupported(monkeypatch):
    registry = ProviderRegistry()
    monkeypatch.setattr(
        "memoria.agents.memory_agent.openai",
        SimpleNamespace(
            OpenAI=lambda **kwargs: object(),
            AsyncOpenAI=lambda **kwargs: object(),
        ),
    )
    registry.register(
        RegisteredProvider(
            name="anthropic",
            provider_type=ProviderType.ANTHROPIC,
            metadata=ProviderRouteMetadata(
                preferred_tasks=("memory_ingest",),
                supports_structured_outputs=False,
            ),
        ),
        aliases=["anthropic"],
    )
    registry.register(
        RegisteredProvider(
            name="openai",
            provider_type=ProviderType.OPENAI,
            model="gpt-4o-mini",
            metadata=ProviderRouteMetadata(
                preferred_tasks=("memory_ingest",),
                supports_structured_outputs=True,
            ),
            client_factory=lambda: object(),
            async_client_factory=lambda: object(),
        ),
        aliases=["openai", "primary"],
        is_primary=True,
    )
    registry.set_task_routes(
        {"memory_ingest": TaskRouteSpec(provider="anthropic", fallback=("openai",))}
    )

    agent = MemoryAgent(
        model="gpt-4o-mini",
        provider_registry=registry,
        task_routes={
            "memory_ingest": TaskRouteSpec(provider="anthropic", fallback=("openai",))
        },
    )
    manager = ConversationManager()
    agent.set_conversation_manager(manager)

    selected_provider: dict[str, str | None] = {"name": None}

    async def fake_structured(
        self, chat_id, system_prompt, conversation_text, context_info, selection=None
    ):
        selected_provider["name"] = selection.provider_name if selection else None
        return ProcessedLongTermMemory(
            content="ok",
            summary="ok",
            classification=MemoryClassification.CONVERSATIONAL,
            importance=MemoryImportanceLevel.MEDIUM,
            conversation_id=chat_id,
            extraction_timestamp=datetime.now(),
        )

    monkeypatch.setattr(
        MemoryAgent,
        "process_with_structured_output",
        fake_structured,
    )

    result = asyncio.run(agent.process_conversation_async("session-1", "hi", "hello"))
    assert isinstance(result, ProcessedLongTermMemory)
    assert selected_provider["name"] == "openai"

    route = manager.get_last_model_route("session-1", "memory_ingest")
    assert route is not None
    assert route["provider"] == "openai"
    assert route["fallback_used"] is True
    assert route["success"] is True


def test_setup_provider_components_calls_configure_gemini_once(monkeypatch):
    gemini_config = object()
    gemini_client = object()
    call_counter = {"count": 0}

    def fake_configure_gemini(inputs):
        call_counter["count"] += 1
        return gemini_config, gemini_client

    captured_optional_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        provider_setup,
        "_configure_gemini",
        fake_configure_gemini,
    )
    monkeypatch.setattr(
        provider_setup,
        "_configure_anthropic",
        lambda inputs: (None, None, None),
    )
    monkeypatch.setattr(
        provider_setup,
        "_configure_primary_provider",
        lambda inputs: (SimpleNamespace(model=None), "test-key"),
    )
    monkeypatch.setattr(
        provider_setup,
        "_register_primary_provider",
        lambda *args, **kwargs: None,
    )

    def fake_register_optional(
        registry,
        *,
        anthropic_config,
        anthropic_client,
        anthropic_async_client,
        gemini_config,
        gemini_client,
    ):
        captured_optional_kwargs["gemini_config"] = gemini_config
        captured_optional_kwargs["gemini_client"] = gemini_client

    monkeypatch.setattr(
        provider_setup,
        "_register_optional_providers",
        fake_register_optional,
    )
    monkeypatch.setattr(
        provider_setup,
        "_initialize_agents",
        lambda **kwargs: (None, None, None, False, False),
    )

    inputs = provider_setup.ProviderSetupInputs(default_model="gpt-4o-mini")
    result = provider_setup.setup_provider_components(inputs)

    assert call_counter["count"] == 1
    assert captured_optional_kwargs["gemini_config"] is gemini_config
    assert captured_optional_kwargs["gemini_client"] is gemini_client
    assert result.gemini_config is gemini_config
    assert result.gemini_client is gemini_client


def test_search_engine_falls_back_when_provider_unavailable(monkeypatch):
    registry = ProviderRegistry()

    def _raise_unavailable():
        raise ProviderUnavailableError("anthropic down")

    monkeypatch.setattr(
        "memoria.agents.retrieval_agent.openai",
        SimpleNamespace(
            OpenAI=lambda **kwargs: object(),
            AsyncOpenAI=lambda **kwargs: object(),
        ),
    )

    registry.register(
        RegisteredProvider(
            name="anthropic",
            provider_type=ProviderType.ANTHROPIC,
            metadata=ProviderRouteMetadata(
                preferred_tasks=("search_planning",),
                supports_structured_outputs=False,
            ),
            client_factory=_raise_unavailable,
        ),
        aliases=["anthropic"],
    )

    registry.register(
        RegisteredProvider(
            name="openai",
            provider_type=ProviderType.OPENAI,
            model="gpt-4o-mini",
            metadata=ProviderRouteMetadata(
                preferred_tasks=("search_planning",),
                supports_structured_outputs=True,
            ),
            client_factory=lambda: object(),
        ),
        aliases=["openai", "primary"],
        is_primary=True,
    )

    registry.set_task_routes(
        {"search_planning": TaskRouteSpec(provider="anthropic", fallback=("openai",))}
    )

    class RecordingPlanner:
        created_types: list[ProviderType | None] = []

        def __init__(self, client, model, provider_config=None, permissions=None):
            self.client = client
            self.model = model
            self.provider_type = None

        def set_provider_type(self, provider_type: ProviderType | None) -> None:
            self.provider_type = provider_type

        def plan_search(
            self, query: str, context: str | None = None
        ) -> MemorySearchQuery:
            RecordingPlanner.created_types.append(self.provider_type)
            return MemorySearchQuery(
                query_text=query,
                intent="Test",
                entity_filters=[],
                category_filters=[],
                search_strategy=["keyword_search"],
                expected_result_types=["any"],
            )

    monkeypatch.setattr(
        SearchExecutor,
        "execute_search",
        lambda self, *args, **kwargs: [],
    )

    engine = MemorySearchEngine(
        model="gpt-4o-mini",
        provider_registry=registry,
        task_routes={
            "search_planning": TaskRouteSpec(provider="anthropic", fallback=("openai",))
        },
        planner_factory=RecordingPlanner,
    )
    manager = ConversationManager()
    engine.set_conversation_manager(manager)

    response = engine.execute_search(
        "find", db_manager=SimpleNamespace(), session_id="session-2"
    )

    assert response == {"results": [], "hint": None, "error": None}
    assert RecordingPlanner.created_types[-1] == ProviderType.OPENAI
    route = manager.get_last_model_route("session-2", "search_planning")
    assert route is not None
    assert route["provider"] == "openai"
    assert route["fallback_used"] is True
    assert route["success"] is True
