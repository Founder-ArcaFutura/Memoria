import sys
from types import SimpleNamespace

import pytest

from memoria.config.settings import MemoriaSettings
from memoria.core.memory import Memoria
from memoria.core.provider_setup import (
    ProviderSetupInputs,
    setup_provider_components,
)
from memoria.core.providers import AnthropicConfig, GeminiConfig, ProviderConfig
from memoria.integrations.openai_client import create_openai_client
from memoria_server.api.app_factory import build_provider_options


def _make_memoria_stub(default_model: str = "test-model") -> Memoria:
    memoria = object.__new__(Memoria)
    memoria.default_model = default_model
    memoria._sync_origin = "stub"
    memoria._sync_coordinator = None
    memoria._sync_backend = None
    memoria._sync_backend_owned = False
    memoria._sync_settings = None
    memoria._sync_settings_snapshot = None
    memoria._sync_enabled = False
    return memoria


def test_create_openai_client_uses_provider_config(monkeypatch):
    memoria = _make_memoria_stub()
    provider_config = ProviderConfig.from_custom(
        base_url="https://custom.endpoint", api_key="provider-key"
    )

    captured_kwargs: dict[str, object] = {}

    class DummyOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.kwargs = kwargs

    dummy_openai_module = SimpleNamespace(OpenAI=DummyOpenAI)

    monkeypatch.setitem(sys.modules, "openai", dummy_openai_module)

    registered_instances: list[Memoria] = []
    monkeypatch.setattr(
        "memoria.integrations.openai_client.register_memoria_instance",
        lambda instance: registered_instances.append(instance),
    )

    client = create_openai_client(
        memoria, provider_config=provider_config, timeout=30, api_key="override"
    )

    assert isinstance(client, DummyOpenAI)
    assert registered_instances == [memoria]
    assert captured_kwargs == {
        "base_url": "https://custom.endpoint",
        "api_key": "override",
        "timeout": 30,
    }


def _make_provider_inputs(**overrides) -> ProviderSetupInputs:
    base_kwargs = {
        "default_model": "test-model",
        "provider_config": None,
        "api_type": None,
        "base_url": None,
        "azure_endpoint": None,
        "api_key": None,
        "openai_api_key": None,
        "azure_deployment": None,
        "api_version": None,
        "azure_ad_token": None,
        "organization": None,
        "project": None,
        "model": None,
        "sovereign_ingest": False,
        "conscious_ingest": False,
        "auto_ingest": False,
        "enable_short_term": True,
        "use_lightweight_conscious_ingest": False,
        "anthropic_config": None,
        "anthropic_api_key": None,
        "anthropic_base_url": None,
        "anthropic_model": None,
        "gemini_config": None,
        "gemini_api_key": None,
        "gemini_model": None,
    }
    base_kwargs.update(overrides)
    return ProviderSetupInputs(**base_kwargs)


def test_setup_provider_components_explicit_azure():
    inputs = _make_provider_inputs(
        api_type="azure",
        azure_endpoint="https://example.openai.azure.com",
        api_key="azure-key",
        azure_deployment="deployment-name",
        api_version="2023-05-15",
        azure_ad_token="aad-token",
    )

    result = setup_provider_components(inputs)

    assert isinstance(result.provider_config, ProviderConfig)
    assert result.provider_config.api_type == "azure"
    assert result.provider_config.azure_endpoint == "https://example.openai.azure.com"
    assert result.provider_config.azure_deployment == "deployment-name"
    assert result.provider_config.model == "test-model"
    assert result.api_key == "azure-key"


def test_setup_provider_components_explicit_custom_endpoint():
    inputs = _make_provider_inputs(
        api_type="custom",
        base_url="https://custom.endpoint",
        api_key="custom-key",
    )

    result = setup_provider_components(inputs)

    assert isinstance(result.provider_config, ProviderConfig)
    assert result.provider_config.api_type == "custom"
    assert result.provider_config.base_url == "https://custom.endpoint"
    assert result.provider_config.model == "test-model"
    assert result.api_key == "custom-key"


def test_setup_provider_components_defaults_to_openai():
    inputs = _make_provider_inputs(
        default_model="default-model",
        openai_api_key="openai-key",
        organization="org-id",
        project="project-id",
    )

    result = setup_provider_components(inputs)

    assert isinstance(result.provider_config, ProviderConfig)
    assert result.provider_config.api_type == "openai"
    assert result.provider_config.organization == "org-id"
    assert result.provider_config.project == "project-id"
    assert result.provider_config.model == "default-model"
    assert result.api_key == "openai-key"


def test_setup_provider_components_prefers_preconfigured_provider():
    preconfigured = ProviderConfig.from_custom(
        base_url="https://existing.endpoint",
        api_key="preconfigured-key",
        model="preconfigured-model",
    )

    inputs = _make_provider_inputs(
        provider_config=preconfigured,
        api_key="fallback-key",
        openai_api_key="fallback-openai-key",
    )

    result = setup_provider_components(inputs)

    assert result.provider_config is preconfigured
    assert result.api_key == "preconfigured-key"
    assert result.provider_config.model == "preconfigured-model"


def test_setup_provider_components_initializes_anthropic_clients(monkeypatch):
    sentinel_sync = object()
    sentinel_async = object()

    def fake_create_client(self):
        fake_create_client.called = True
        return sentinel_sync

    fake_create_client.called = False

    def fake_create_async_client(self):
        fake_create_async_client.called = True
        return sentinel_async

    fake_create_async_client.called = False

    monkeypatch.setattr(AnthropicConfig, "create_client", fake_create_client)
    monkeypatch.setattr(
        AnthropicConfig, "create_async_client", fake_create_async_client
    )

    inputs = _make_provider_inputs(
        openai_api_key="openai-key",
        anthropic_api_key="anth-key",
        anthropic_base_url="https://anthropic.example",
        anthropic_model="claude-3-haiku",
    )

    result = setup_provider_components(inputs)

    assert isinstance(result.provider_config, ProviderConfig)
    assert result.api_key == "openai-key"
    assert result.anthropic_config is not None
    assert result.anthropic_config.api_key == "anth-key"
    assert result.anthropic_config.base_url == "https://anthropic.example"
    assert result.anthropic_config.model == "claude-3-haiku"
    assert result.anthropic_client is sentinel_sync
    assert result.anthropic_async_client is sentinel_async
    assert fake_create_client.called is True
    assert fake_create_async_client.called is True


def test_setup_provider_components_initializes_gemini_client(monkeypatch):
    sentinel_client = object()

    def fake_create_client(self):
        fake_create_client.calls.append({"api_key": self.api_key, "model": self.model})
        return sentinel_client

    fake_create_client.calls = []

    monkeypatch.setattr(GeminiConfig, "create_client", fake_create_client)

    inputs = _make_provider_inputs(
        openai_api_key="openai-key",
        gemini_api_key="gem-key",
        gemini_model="gemini-1.5-pro",
    )

    result = setup_provider_components(inputs)

    assert isinstance(result.provider_config, ProviderConfig)
    assert result.api_key == "openai-key"
    assert result.gemini_config is not None
    assert result.gemini_config.api_key == "gem-key"
    assert result.gemini_config.model == "gemini-1.5-pro"
    assert result.gemini_client is sentinel_client
    assert fake_create_client.calls


def test_setup_provider_components_respects_sovereign_ingest(monkeypatch):
    inputs = _make_provider_inputs(
        sovereign_ingest=True,
        conscious_ingest=True,
        openai_api_key="api-key",
        model="configured-model",
    )

    class DummyMemoryAgent:
        called = False

        def __init__(self, *args, **kwargs):
            DummyMemoryAgent.called = True

    class DummySearchEngine:
        called = False

        def __init__(self, *args, **kwargs):
            DummySearchEngine.called = True

    class DummyConsciousAgent:
        called = False

        def __init__(self, use_heuristics: bool):
            DummyConsciousAgent.called = True
            self.use_heuristics = use_heuristics

    monkeypatch.setattr("memoria.agents.memory_agent.MemoryAgent", DummyMemoryAgent)
    monkeypatch.setattr(
        "memoria.agents.retrieval_agent.MemorySearchEngine", DummySearchEngine
    )
    monkeypatch.setattr(
        "memoria.agents.conscious_agent.ConsciousAgent", DummyConsciousAgent
    )

    result = setup_provider_components(inputs)

    assert result.memory_agent is None
    assert result.search_engine is None
    assert DummyMemoryAgent.called is False
    assert DummySearchEngine.called is False
    assert isinstance(result.conscious_agent, DummyConsciousAgent)
    assert DummyConsciousAgent.called is True
    assert result.conscious_ingest is True


def test_setup_provider_components_initializes_agents_with_provider(monkeypatch):
    provider_config = SimpleNamespace(model="provider-model")
    inputs = _make_provider_inputs(
        provider_config=provider_config,
        model="configured-model",
        sovereign_ingest=False,
        conscious_ingest=False,
        auto_ingest=False,
    )

    class RecordingMemoryAgent:
        calls: list[dict[str, object]] = []

        def __init__(self, **kwargs):
            RecordingMemoryAgent.calls.append(kwargs)
            self.kwargs = kwargs

    class RecordingSearchEngine:
        calls: list[dict[str, object]] = []

        def __init__(self, **kwargs):
            RecordingSearchEngine.calls.append(kwargs)
            self.kwargs = kwargs

    monkeypatch.setattr("memoria.agents.memory_agent.MemoryAgent", RecordingMemoryAgent)
    monkeypatch.setattr(
        "memoria.agents.retrieval_agent.MemorySearchEngine", RecordingSearchEngine
    )

    class FailingConsciousAgent:
        def __init__(self, *args, **kwargs):  # pragma: no cover - defensive guard
            raise AssertionError("ConsciousAgent should not be created")

    monkeypatch.setattr(
        "memoria.agents.conscious_agent.ConsciousAgent", FailingConsciousAgent
    )

    result = setup_provider_components(inputs)

    assert isinstance(result.memory_agent, RecordingMemoryAgent)
    assert isinstance(result.search_engine, RecordingSearchEngine)
    assert len(RecordingMemoryAgent.calls) == 1
    memory_kwargs = RecordingMemoryAgent.calls[0]
    assert memory_kwargs["provider_config"] is provider_config
    assert memory_kwargs["model"] == "configured-model"
    assert "provider_registry" in memory_kwargs
    assert "task_routes" in memory_kwargs

    assert len(RecordingSearchEngine.calls) == 1
    search_kwargs = RecordingSearchEngine.calls[0]
    assert search_kwargs["provider_config"] is provider_config
    assert search_kwargs["model"] == "configured-model"
    assert "provider_registry" in search_kwargs
    assert "task_routes" in search_kwargs
    assert result.conscious_agent is None


def _make_settings(**sections):
    return SimpleNamespace(**sections)


def test_build_provider_options_prefers_agent_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_TYPE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_NAME", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_ORGANIZATION", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT", raising=False)

    settings = _make_settings(
        agents=SimpleNamespace(
            conscious_ingest=True,
            default_model="agent-model",
            openai_api_key="agent-openai-key",
            anthropic_api_key="agent-anthropic-key",
            anthropic_base_url="https://anthropic.agent",
            anthropic_model="claude-3-sonnet",
            gemini_api_key="agent-gemini-key",
            gemini_model="gemini-1.5-flash",
        ),
        memory=SimpleNamespace(
            context_injection=True,
            namespace="test-namespace",
            shared_memory=True,
            sovereign_ingest=True,
        ),
        plugins={"enabled": True},
        sync={"interval": 5},
    )

    options = build_provider_options(settings)

    assert options["conscious_ingest"] is True
    assert options["auto_ingest"] is True
    assert options["namespace"] == "test-namespace"
    assert options["shared_memory"] is True
    assert options["sovereign_ingest"] is True
    assert options["openai_api_key"] == "agent-openai-key"
    assert options["api_key"] == "agent-openai-key"
    assert options["model"] == "agent-model"
    assert options["anthropic_api_key"] == "agent-anthropic-key"
    assert options["anthropic_base_url"] == "https://anthropic.agent"
    assert options["anthropic_model"] == "claude-3-sonnet"
    assert options["gemini_api_key"] == "agent-gemini-key"
    assert options["gemini_model"] == "gemini-1.5-flash"
    assert options["plugin_settings"] == {"enabled": True}
    assert options["sync_settings"] == {"interval": 5}


@pytest.mark.parametrize(
    "env_name,expected_api_key",
    [
        ("OPENAI_API_KEY", "env-openai"),
        ("AZURE_OPENAI_API_KEY", "env-azure"),
    ],
)
def test_build_provider_options_env_fallbacks(monkeypatch, env_name, expected_api_key):
    monkeypatch.setenv(env_name, expected_api_key)
    # Ensure the alternate variable doesn't interfere with precedence.
    for other in {"OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"} - {env_name}:
        monkeypatch.delenv(other, raising=False)

    monkeypatch.setenv("OPENAI_API_TYPE", "azure")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://base.url")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.endpoint")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "deployment-name")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "aad-token")
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org")
    monkeypatch.setenv("OPENAI_PROJECT", "project")

    settings = _make_settings(agents=None, memory=None, plugins=None, sync=None)

    options = build_provider_options(settings)

    assert options["api_key"] == expected_api_key
    assert options["openai_api_key"] == (
        expected_api_key if env_name == "OPENAI_API_KEY" else None
    )
    assert options["api_type"] == "azure"
    assert options["base_url"] == "https://base.url"
    assert options["anthropic_api_key"] is None
    assert options["gemini_api_key"] is None
    assert options["azure_endpoint"] == "https://azure.endpoint"
    assert options["azure_deployment"] == "deployment-name"
    assert options["api_version"] == "2024-06-01"
    assert options["azure_ad_token"] == "aad-token"
    assert options["organization"] == "org"
    assert options["project"] == "project"


def test_build_provider_options_uses_config_defaults():
    settings = MemoriaSettings()

    options = build_provider_options(settings)

    assert options["model"] == settings.agents.default_model
    assert options["conscious_ingest"] is settings.agents.conscious_ingest
    assert options["auto_ingest"] is settings.memory.context_injection
    assert options["namespace"] == settings.memory.namespace
    assert options["shared_memory"] is settings.memory.shared_memory
    assert options["sovereign_ingest"] is settings.memory.sovereign_ingest


def test_build_provider_options_anthropic_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anth")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.local")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-sonnet")

    settings = _make_settings(agents=None, memory=None, plugins=None, sync=None)

    options = build_provider_options(settings)

    assert options["anthropic_api_key"] == "env-anth"
    assert options["anthropic_base_url"] == "https://anthropic.local"
    assert options["anthropic_model"] == "claude-3-sonnet"


def test_build_provider_options_gemini_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GOOGLE_GEMINI_MODEL", raising=False)

    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-flash")

    settings = _make_settings(agents=None, memory=None, plugins=None, sync=None)

    options = build_provider_options(settings)

    assert options["gemini_api_key"] == "env-gemini"
    assert options["gemini_model"] == "gemini-1.5-flash"


def test_memoria_settings_loads_optional_agent_fields():
    settings = MemoriaSettings(
        agents={
            "openai_api_key": "sk-agent-1234567890abcdef",
            "anthropic_api_key": "anth-key",
            "anthropic_model": " claude-3-haiku ",
            "anthropic_base_url": " https://anthropic.internal ",
            "gemini_api_key": "gem-key",
            "gemini_model": " gemini-1.5-pro ",
        }
    )

    agent_settings = settings.agents
    assert agent_settings.anthropic_api_key == "anth-key"
    assert agent_settings.anthropic_model == "claude-3-haiku"
    assert agent_settings.anthropic_base_url == "https://anthropic.internal"
    assert agent_settings.gemini_api_key == "gem-key"
    assert agent_settings.gemini_model == "gemini-1.5-pro"
