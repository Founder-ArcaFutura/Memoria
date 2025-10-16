"""Tests for LiteLLM async context injection wrappers."""

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace


class StubMemoria:
    """Minimal Memoria stub for testing LiteLLM async helpers."""

    def __init__(self):
        self.auto_ingest = True
        self.conscious_ingest = False
        self._enabled = True
        self.injected_modes = []
        self.recorded = []

    @property
    def is_enabled(self):  # noqa: D401 - parity with production property
        """Return whether recording is enabled."""

        return self._enabled

    def _inject_litellm_context(
        self, kwargs, *, mode
    ):  # noqa: D401 - production signature
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


def test_async_completion_injects_context_and_records(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="AI output"))],
        usage=SimpleNamespace(
            total_tokens=5,
            prompt_tokens=2,
            completion_tokens=3,
        ),
    )

    calls = {"completion": [], "acompletion": []}

    def _run_callbacks(kwargs):
        for callback in list(fake_litellm.success_callback):
            callback(kwargs, response, 1.0, 2.0)

    def fake_completion(*args, **kwargs):
        calls["completion"].append(kwargs)
        _run_callbacks(kwargs)
        return response

    async def fake_acompletion(*args, **kwargs):
        calls["acompletion"].append(kwargs)
        _run_callbacks(kwargs)
        return response

    fake_litellm = ModuleType("litellm")
    fake_litellm.success_callback = []
    fake_litellm.completion = fake_completion
    fake_litellm.acompletion = fake_acompletion

    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.delitem(
        sys.modules, "memoria.integrations.litellm_integration", raising=False
    )

    integration = importlib.import_module("memoria.integrations.litellm_integration")
    LiteLLMCallbackManager = integration.LiteLLMCallbackManager
    fake_litellm = integration.litellm  # type: ignore[assignment]
    monkeypatch.setattr(integration, "LITELLM_AVAILABLE", True, raising=False)

    memoria = StubMemoria()
    manager = LiteLLMCallbackManager(memoria)

    original_completion = fake_litellm.completion
    original_acompletion = fake_litellm.acompletion

    assert manager.register_callbacks() is True
    assert fake_litellm.acompletion is not original_acompletion

    kwargs = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "existing"},
            {"role": "user", "content": "Hello"},
        ],
    }

    result = asyncio.run(fake_litellm.acompletion(**kwargs))

    assert result is response
    assert calls["acompletion"], "Async helper should be invoked"

    injected_kwargs = calls["acompletion"][0]
    assert (
        injected_kwargs is not kwargs
    ), "Wrapper should avoid mutating original kwargs"
    assert any(
        msg["content"].startswith("context:") for msg in injected_kwargs["messages"]
    )
    assert memoria.injected_modes == ["auto"]

    assert memoria.recorded
    recorded = memoria.recorded[0]
    assert recorded["user_input"] == "Hello"
    assert recorded["ai_output"] == "AI output"
    assert recorded["model"] == "test-model"
    assert recorded["metadata"]["tokens_used"] == 5

    assert manager.unregister_callbacks() is True
    assert fake_litellm.completion is original_completion
    assert fake_litellm.acompletion is original_acompletion
