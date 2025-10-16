import importlib
import sys
import types

import pytest

from memoria.config.manager import ConfigManager
from memoria.utils.exceptions import ConfigurationError


@pytest.fixture
def reload_memory_tool():
    """Ensure the memory tool module is re-imported fresh for each test."""

    module_name = "memoria.tools.memory_tool"
    original_module = sys.modules.pop(module_name, None)
    try:
        yield
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module


def test_memory_tool_import_without_configuration(monkeypatch, reload_memory_tool):
    """Importing the tool should not require configuration to be pre-loaded."""

    monkeypatch.setitem(
        sys.modules,
        "memoria.core.memory",
        types.SimpleNamespace(Memoria=object),
    )

    class RaisingManager:
        def get_settings(self):
            raise ConfigurationError("Configuration not loaded")

    monkeypatch.setattr(
        ConfigManager,
        "get_instance",
        classmethod(lambda cls: RaisingManager()),
    )

    module = importlib.import_module("memoria.tools.memory_tool")

    assert hasattr(module, "MemoryTool")
