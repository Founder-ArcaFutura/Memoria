import importlib
import sys
import types

import pytest

from memoria.config.manager import ConfigManager
from memoria.tools.memory_tool import MemoryTool, record_conversation
from memoria.utils.exceptions import ConfigurationError

# Provide a stub Memoria class to avoid heavy dependencies during import
sys.modules.setdefault("memoria.core.memory", types.SimpleNamespace(Memoria=object))


@pytest.fixture
def stub_settings(monkeypatch):
    settings = types.SimpleNamespace(
        agents=types.SimpleNamespace(default_model="stub-model")
    )

    class StubManager:
        def __init__(self, configured_settings):
            self._settings = configured_settings

        def get_settings(self):
            return self._settings

    monkeypatch.setattr(
        ConfigManager,
        "get_instance",
        classmethod(lambda cls: StubManager(settings)),
    )

    return settings


class DummyMemoria:
    def __init__(self):
        self.calls = []
        self.spatial_calls = []

    def record_conversation(self, **kwargs):
        self.calls.append(kwargs)
        return "cid"

    def retrieve_memories_near_2d(self, y, z, max_distance, anchor=None, limit=None):
        self.spatial_calls.append(
            {
                "method": "2d",
                "y": y,
                "z": z,
                "max_distance": max_distance,
                "anchor": anchor,
                "limit": limit,
            }
        )
        return [{"text": "2d memory result", "distance": 1.0}]

    def retrieve_memories_near(
        self,
        x,
        y,
        z,
        max_distance,
        *,
        anchor=None,
        limit=None,
        dimensions="3d",
    ):
        self.spatial_calls.append(
            {
                "method": "3d",
                "x": x,
                "y": y,
                "z": z,
                "max_distance": max_distance,
                "anchor": anchor,
                "limit": limit,
                "dimensions": dimensions,
            }
        )
        return []


def test__record_conversation_uses_resolved_model(stub_settings):
    dummy = DummyMemoria()
    tool = MemoryTool(dummy)

    tool._record_conversation(user_input="hi", ai_output="yo")
    assert dummy.calls[0]["model"] == stub_settings.agents.default_model

    tool._record_conversation(user_input="hi", ai_output="yo", model="override")
    assert dummy.calls[1]["model"] == "override"


def test_decorator_passes_resolved_model(stub_settings):
    dummy = DummyMemoria()

    @record_conversation(dummy)
    def chat(messages, model=None):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message=types.SimpleNamespace(content="response"))
            ]
        )

    chat(messages=[{"role": "user", "content": "hi"}])
    assert dummy.calls[0]["model"] == stub_settings.agents.default_model

    chat(messages=[{"role": "user", "content": "hi"}], model="custom")
    assert dummy.calls[1]["model"] == "custom"


def test_execute_spatial_2d_routes_to_2d_retrieval():
    dummy = DummyMemoria()
    tool = MemoryTool(dummy)

    result = tool.execute(
        operation="spatial", dimensions="2d", y=1.25, max_distance=3.0
    )

    assert "Error" not in result
    assert "2d memory result" in result
    assert dummy.spatial_calls
    call = dummy.spatial_calls[0]
    assert call["method"] == "2d"
    assert call["y"] == 1.25
    assert call["z"] == 0.0
    assert call["max_distance"] == 3.0


def test_memory_tool_import_without_config(monkeypatch):
    module_name = "memoria.tools.memory_tool"
    original_module = sys.modules.pop(module_name, None)

    class RaisingManager:
        def get_settings(self):
            raise ConfigurationError("Configuration not loaded")

    monkeypatch.setattr(
        ConfigManager,
        "get_instance",
        classmethod(lambda cls: RaisingManager()),
    )

    try:
        module = importlib.import_module(module_name)
        assert hasattr(module, "MemoryTool")
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
