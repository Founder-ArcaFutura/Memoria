import pytest
from pydantic import ValidationError

from memoria.config.manager import ConfigManager
from memoria.config.settings import DatabaseSettings, MemoriaSettings


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg2://user:pass@localhost:5432/memoria",
        "postgres://user:pass@localhost:5432/memoria",
        "mysql+pymysql://user:pass@localhost:3306/memoria",
        "sqlite:///memoria.db",
    ],
)
def test_database_settings_accepts_extended_schemes(url):
    settings = DatabaseSettings(connection_string=url)
    assert settings.connection_string == url


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/memoria",
        "redis://localhost:6379",
        "mariadb://user:pass@localhost:3306/memoria",
        "mssql+pyodbc://user:pass@localhost:1433/memoria",
    ],
)
def test_database_settings_rejects_unknown_scheme(url):
    with pytest.raises(ValidationError):
        DatabaseSettings(connection_string=url)


def test_memoria_settings_default_conscious_interval():
    settings = MemoriaSettings()
    assert settings.memory.conscious_analysis_interval_seconds == 6 * 60 * 60


def test_memoria_settings_neutral_default_models():
    settings = MemoriaSettings()
    assert settings.agents.default_model == "gpt-4o-mini"
    assert settings.agents.fallback_model == "gpt-3.5-turbo"


def test_memoria_settings_opt_in_ingest_flags():
    settings = MemoriaSettings()
    assert settings.agents.conscious_ingest is False
    assert settings.memory.context_injection is False


def test_config_manager_loads_nested_env(monkeypatch):
    monkeypatch.setenv("MEMORIA_DATABASE__CONNECTION_STRING", "sqlite:///env.db")
    monkeypatch.setenv("MEMORIA_MEMORY__NAMESPACE", "env-namespace")

    # Reset singleton state for isolated testing
    ConfigManager._instance = None
    ConfigManager._settings = None

    manager = ConfigManager()

    try:
        manager.load_from_env()
        settings = manager.get_settings()

        assert settings.database.connection_string == "sqlite:///env.db"
        assert settings.memory.namespace == "env-namespace"
    finally:
        ConfigManager._instance = None
        ConfigManager._settings = None


def test_config_manager_loads_conscious_interval_from_env(monkeypatch):
    monkeypatch.setenv("MEMORIA_MEMORY__CONSCIOUS_ANALYSIS_INTERVAL_SECONDS", "120")

    ConfigManager._instance = None
    ConfigManager._settings = None

    manager = ConfigManager()

    try:
        manager.load_from_env()
        settings = manager.get_settings()

        assert settings.memory.conscious_analysis_interval_seconds == 120
    finally:
        ConfigManager._instance = None
        ConfigManager._settings = None


def test_config_manager_auto_load_alias_env(monkeypatch):
    monkeypatch.setenv("MEMORIA_DB_URL", "sqlite:///alias.db")

    ConfigManager._instance = None
    ConfigManager._settings = None

    manager = ConfigManager()

    try:
        manager.auto_load()
        settings = manager.get_settings()

        assert settings.database.connection_string == "sqlite:///alias.db"

        info = manager.get_config_info()
        assert "environment" in info["sources"]
        assert "MEMORIA_DB_URL" in info["env_overrides"]
    finally:
        ConfigManager._instance = None
        ConfigManager._settings = None
