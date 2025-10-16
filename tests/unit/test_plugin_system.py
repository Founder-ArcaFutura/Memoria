import json

import pytest

from memoria.config import ConfigManager, MemoriaSettings
from memoria.core.memory import Memoria

PLUGIN_BASE = "tests.unit.plugins.sample_plugins"


def _reset_plugins(config_manager: ConfigManager) -> None:
    config_manager.update_setting("plugins", [])


@pytest.fixture
def config_manager():
    manager = ConfigManager()
    _reset_plugins(manager)
    yield manager
    _reset_plugins(manager)


def test_plugins_receive_memoria_instance(config_manager):
    config_manager.update_setting(
        "plugins",
        [
            {
                "name": "recorder",
                "import_path": f"{PLUGIN_BASE}:RecordingPlugin",
                "options": {"token": "alpha"},
            }
        ],
    )

    memoria = Memoria(
        database_connect="sqlite:///:memory:",
        schema_init=False,
        sovereign_ingest=True,
    )
    try:
        plugin = memoria.get_plugin("recorder")
        assert plugin is not None
        assert plugin.initialized_with is memoria
        assert plugin.ready_with is memoria
        assert plugin.storage_seen is True
        assert plugin.token == "alpha"
    finally:
        memoria.disable()


def test_plugin_shutdown_called_on_disable(config_manager):
    config_manager.update_setting(
        "plugins",
        [
            {
                "name": "shutdown-tracker",
                "import_path": f"{PLUGIN_BASE}:ShutdownTrackerPlugin",
                "options": {"token": "beta"},
            }
        ],
    )

    memoria = Memoria(
        database_connect="sqlite:///:memory:",
        schema_init=False,
        sovereign_ingest=True,
    )
    plugin = memoria.get_plugin("shutdown-tracker")
    assert plugin is not None
    assert plugin.shutdown_called is False

    memoria.disable()

    assert plugin.shutdown_called is True


def test_misconfigured_plugins_reported(config_manager):
    config_manager.update_setting(
        "plugins",
        [
            {
                "name": "valid",
                "import_path": f"{PLUGIN_BASE}:RecordingPlugin",
            },
            {
                "name": "broken",
                "import_path": f"{PLUGIN_BASE}:FailingPlugin",
            },
            {
                "name": "missing",
                "import_path": "tests.unit.plugins.missing:Plugin",
            },
        ],
    )

    memoria = Memoria(
        database_connect="sqlite:///:memory:",
        schema_init=False,
        sovereign_ingest=True,
    )
    try:
        assert len(memoria.plugins) == 1
        assert memoria.plugins[0].__class__.__name__ == "RecordingPlugin"

        failures = memoria.plugin_failures
        assert "broken" in failures
        assert "missing" in failures
        assert "initialize" in failures["broken"].lower()
    finally:
        memoria.disable()


def test_settings_parse_plugin_list_from_env(monkeypatch):
    payload = json.dumps(
        [
            {
                "name": "env-plugin",
                "import_path": f"{PLUGIN_BASE}:RecordingPlugin",
                "options": {"token": "gamma"},
            },
            {
                "import_path": f"{PLUGIN_BASE}:ShutdownTrackerPlugin",
                "enabled": False,
            },
        ]
    )
    monkeypatch.setenv("MEMORIA_PLUGINS", payload)

    settings = MemoriaSettings.from_env()

    monkeypatch.delenv("MEMORIA_PLUGINS", raising=False)

    assert len(settings.plugins) == 2
    assert settings.plugins[0].name == "env-plugin"
    assert settings.plugins[0].options["token"] == "gamma"
    assert settings.plugins[1].enabled is False
