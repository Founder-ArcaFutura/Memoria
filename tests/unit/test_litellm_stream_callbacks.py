import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace


class StubMemoria:
    """Minimal Memoria stub for streaming tests."""

    def __init__(self):
        self.auto_ingest = True
        self.conscious_ingest = False
        self._enabled = True
        self.injected_modes: list[str] = []
        self.recorded: list[dict] = []

    @property
    def is_enabled(self):  # noqa: D401 - production parity
        """Return whether recording is enabled."""

        return self._enabled

    def _inject_litellm_context(self, kwargs, *, mode):  # noqa: D401 - signature parity
        self.injected_modes.append(mode)
        updated = dict(kwargs)
        messages = list(updated.get("messages", []))
        messages.append({"role": "system", "content": f"context:{mode}"})
        updated["messages"] = messages
        return updated

    def record_conversation(self, user_input, ai_output, model, metadata):
        self.recorded.append(
            {
                "user_input": user_input,
                "ai_output": ai_output,
                "model": model,
                "metadata": metadata,
            }
        )


def create_fake_litellm(stream_chunks, async_stream_chunks):
    calls: dict[str, list] = {"stream_completion": [], "acompletion_stream": []}

    def fake_stream_completion(*args, **kwargs):
        calls["stream_completion"].append(kwargs)

        def generator():
            yield from stream_chunks

        return generator()

    def fake_acompletion_stream(*args, **kwargs):
        calls["acompletion_stream"].append(kwargs)

        async def async_generator():
            for chunk in async_stream_chunks:
                yield chunk

        return async_generator()

    fake = ModuleType("litellm")
    fake.success_callback = []
    fake.completion = lambda *a, **k: None  # pragma: no cover - unused fallback
    fake.acompletion = lambda *a, **k: None  # pragma: no cover - unused fallback
    fake.stream_completion = fake_stream_completion
    fake.acompletion_stream = fake_acompletion_stream

    return fake, calls


def _reload_integration(monkeypatch, fake_litellm):
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.delitem(
        sys.modules, "memoria.integrations.litellm_integration", raising=False
    )
    integration = importlib.import_module("memoria.integrations.litellm_integration")
    monkeypatch.setattr(integration, "LITELLM_AVAILABLE", True, raising=False)
    return integration


def test_stream_completion_records_after_consumption(monkeypatch):
    stream_chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {
            "choices": [
                {
                    "delta": {"content": " world"},
                    "message": {"content": "Hello world"},
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            },
        },
    ]

    async_chunks = [
        {"choices": [{"delta": {"content": "unused"}}]},
    ]

    fake_litellm, calls = create_fake_litellm(stream_chunks, async_chunks)
    integration = _reload_integration(monkeypatch, fake_litellm)
    LiteLLMCallbackManager = integration.LiteLLMCallbackManager
    fake_litellm = integration.litellm  # type: ignore[assignment]

    memoria = StubMemoria()
    manager = LiteLLMCallbackManager(memoria)

    original_stream = fake_litellm.stream_completion
    assert manager.register_callbacks() is True
    assert fake_litellm.stream_completion is not original_stream

    kwargs = {
        "model": "stream-model",
        "messages": [
            {"role": "user", "content": "Hi"},
        ],
    }

    chunks = list(fake_litellm.stream_completion(**kwargs))
    assert chunks == stream_chunks

    injected_kwargs = calls["stream_completion"][0]
    assert injected_kwargs is not kwargs
    assert any(
        msg["content"].startswith("context:") for msg in injected_kwargs["messages"]
    )
    assert memoria.injected_modes == ["auto"]

    assert memoria.recorded
    recorded = memoria.recorded[0]
    assert recorded["user_input"] == "Hi"
    assert recorded["ai_output"] == "Hello world"
    assert recorded["model"] == "stream-model"
    assert recorded["metadata"]["api_type"] == "stream_completion"
    assert recorded["metadata"]["tokens_used"] == 7
    assert recorded["metadata"]["prompt_tokens"] == 3
    assert recorded["metadata"]["completion_tokens"] == 4
    assert recorded["metadata"]["stream"] is True

    assert manager.unregister_callbacks() is True
    assert fake_litellm.stream_completion is original_stream


def test_acompletion_stream_records(monkeypatch):
    stream_chunks = [
        {"choices": [{"delta": {"content": "ignored"}}]},
    ]

    async_chunks = [
        {"choices": [{"delta": {"content": "Async"}}]},
        {
            "choices": [{"delta": {"content": " stream"}}],
            "usage": SimpleNamespace(
                prompt_tokens=2,
                completion_tokens=2,
                total_tokens=4,
            ),
        },
    ]

    fake_litellm, calls = create_fake_litellm(stream_chunks, async_chunks)
    integration = _reload_integration(monkeypatch, fake_litellm)
    LiteLLMCallbackManager = integration.LiteLLMCallbackManager
    fake_litellm = integration.litellm  # type: ignore[assignment]

    memoria = StubMemoria()
    manager = LiteLLMCallbackManager(memoria)

    original_async_stream = fake_litellm.acompletion_stream
    assert manager.register_callbacks() is True
    assert fake_litellm.acompletion_stream is not original_async_stream

    kwargs = {
        "model": "async-stream-model",
        "messages": [
            {"role": "system", "content": "intro"},
            {"role": "user", "content": "Stream please"},
        ],
    }

    async def consume():
        collected = []
        async for chunk in fake_litellm.acompletion_stream(**kwargs):
            collected.append(chunk)
        return collected

    collected_chunks = asyncio.run(consume())
    assert collected_chunks == async_chunks

    injected_kwargs = calls["acompletion_stream"][0]
    assert injected_kwargs is not kwargs
    assert any(
        msg["content"].startswith("context:") for msg in injected_kwargs["messages"]
    )
    assert memoria.injected_modes == ["auto"]

    assert memoria.recorded
    recorded = memoria.recorded[0]
    assert recorded["user_input"] == "Stream please"
    assert recorded["ai_output"] == "Async stream"
    assert recorded["model"] == "async-stream-model"
    assert recorded["metadata"]["api_type"] == "acompletion_stream"
    assert recorded["metadata"]["tokens_used"] == 4
    assert recorded["metadata"]["prompt_tokens"] == 2
    assert recorded["metadata"]["completion_tokens"] == 2

    assert manager.unregister_callbacks() is True
    assert fake_litellm.acompletion_stream is original_async_stream
