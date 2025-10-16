import sys
from types import ModuleType

import pytest

sys.modules.setdefault("litellm", ModuleType("litellm_stub"))

from memoria.config import ConfigManager
from memoria.integrations import (
    anthropic_integration,
    get_default_model,
    litellm_integration,
    openai_integration,
)


@pytest.fixture
def configured_default_model(monkeypatch):
    for env_var in ("LLM_MODEL", "OPENAI_MODEL", "LITELLM_MODEL", "ANTHROPIC_MODEL"):
        monkeypatch.delenv(env_var, raising=False)

    ConfigManager._instance = None
    ConfigManager._settings = None

    manager = ConfigManager()
    manager.update_setting("agents.default_model", "integration-configured-model")

    try:
        yield "integration-configured-model"
    finally:
        ConfigManager._instance = None
        ConfigManager._settings = None


def test_get_default_model_reads_configured_value(configured_default_model):
    assert get_default_model() == configured_default_model


def test_openai_helper_reads_configured_value(configured_default_model):
    assert openai_integration.get_default_openai_model() == configured_default_model


def test_anthropic_helper_reads_configured_value(configured_default_model):
    assert (
        anthropic_integration.get_default_anthropic_model() == configured_default_model
    )


def test_litellm_helper_reads_configured_value(configured_default_model):
    assert litellm_integration.get_default_litellm_model() == configured_default_model
