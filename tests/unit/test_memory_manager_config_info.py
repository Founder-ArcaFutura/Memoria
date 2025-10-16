from memoria.config.manager import ConfigManager
from memoria.config.memory_manager import MemoryManager


def test_memory_manager_config_info_from_config_manager(monkeypatch):
    manager = MemoryManager()

    expected = {
        "loaded": True,
        "sources": ["defaults", "environment"],
        "version": "2.1.0",
        "debug_mode": True,
        "is_production": False,
    }

    monkeypatch.setattr(ConfigManager, "get_config_info", lambda self: expected)

    assert manager.get_config_info() == expected


def test_memory_manager_config_info_fallback_to_memoria(monkeypatch):
    manager = MemoryManager()

    class DummyMemoria:
        def __init__(self, info):
            self._config_info = info

        def get_config_info(self):
            return dict(self._config_info)

    fallback_info = {
        "loaded": True,
        "sources": ("runtime", "manual"),
        "version": "1.5.0",
        "debug_mode": False,
        "is_production": True,
    }

    manager.memoria_instance = DummyMemoria(fallback_info)

    def raising_get_config_info(_self):
        raise RuntimeError("test failure")

    monkeypatch.setattr(ConfigManager, "get_config_info", raising_get_config_info)

    result = manager.get_config_info()

    assert result == {
        "loaded": True,
        "sources": ["runtime", "manual"],
        "version": "1.5.0",
        "debug_mode": False,
        "is_production": True,
    }
