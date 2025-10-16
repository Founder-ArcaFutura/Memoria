import json

import pytest

from memoria.config.manager import ConfigManager
from memoria.core.memory import Memoria


def _reset_config_manager():
    ConfigManager._instance = None
    ConfigManager._settings = None


@pytest.fixture(autouse=True)
def _cleanup_config_manager():
    try:
        yield
    finally:
        _reset_config_manager()


def test_conscious_manager_uses_interval_from_env(monkeypatch):
    monkeypatch.setenv(
        "MEMORIA_MEMORY__CONSCIOUS_ANALYSIS_INTERVAL_SECONDS",
        "180",
    )

    manager = ConfigManager()
    manager.load_from_env()

    memoria = Memoria(conscious_ingest=False, auto_ingest=False, openai_api_key=None)

    assert memoria.conscious_analysis_interval_seconds == 180
    assert memoria.conscious_manager._analysis_interval == 180


def test_conscious_manager_uses_interval_from_file(tmp_path):
    config_path = tmp_path / "memoria.json"
    config_path.write_text(
        json.dumps({"memory": {"conscious_analysis_interval_seconds": 240}})
    )

    manager = ConfigManager()
    manager.load_from_file(config_path)

    memoria = Memoria(conscious_ingest=False, auto_ingest=False, openai_api_key=None)

    assert memoria.conscious_analysis_interval_seconds == 240
    assert memoria.conscious_manager._analysis_interval == 240
