import sys
from types import SimpleNamespace

from memoria.core.providers import AnthropicConfig, GeminiConfig


def test_anthropic_config_builds_clients(monkeypatch):
    captured: dict[str, dict[str, object]] = {}

    class SyncClient:
        def __init__(self, **kwargs):
            captured["sync"] = kwargs

    class AsyncClient:
        def __init__(self, **kwargs):
            captured["async"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=SyncClient, AsyncAnthropic=AsyncClient),
    )

    config = AnthropicConfig(
        api_key="anth-key",
        base_url="https://anthropic.example",
        timeout=30.0,
        default_headers={"User-Agent": "memoria-test"},
    )

    sync_client = config.create_client()
    async_client = config.create_async_client()

    assert isinstance(sync_client, SyncClient)
    assert isinstance(async_client, AsyncClient)
    assert captured["sync"] == {
        "api_key": "anth-key",
        "base_url": "https://anthropic.example",
        "timeout": 30.0,
        "default_headers": {"User-Agent": "memoria-test"},
    }
    assert captured["async"] == captured["sync"]


def test_gemini_config_configures_sdk(monkeypatch):
    configure_kwargs: dict[str, object] = {}
    model_kwargs: dict[str, object] = {}

    class FakeGenerativeModel:
        def __init__(self, model_name: str, **kwargs):
            model_kwargs["model_name"] = model_name
            model_kwargs["kwargs"] = kwargs

    fake_genai = SimpleNamespace(
        configure=lambda **kwargs: configure_kwargs.update(kwargs),
        GenerativeModel=FakeGenerativeModel,
    )

    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(generativeai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    config = GeminiConfig(
        api_key="gem-key",
        model="gemini-pro",
        generation_config={"temperature": 0.2},
        safety_settings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
    )

    client = config.create_client()

    assert isinstance(client, FakeGenerativeModel)
    assert configure_kwargs == {"api_key": "gem-key"}
    assert model_kwargs["model_name"] == "gemini-pro"
    assert model_kwargs["kwargs"] == {
        "generation_config": {"temperature": 0.2},
        "safety_settings": [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
    }

    # Async creation should reuse the synchronous implementation
    second_client = config.create_async_client()
    assert isinstance(second_client, FakeGenerativeModel)
    assert model_kwargs["model_name"] == "gemini-pro"
