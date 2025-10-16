from __future__ import annotations

from types import SimpleNamespace

import pytest

from memoria.config import ConfigManager
from memoria.config.settings import TeamMode
from memoria_server.api.app_factory import create_app


class FakeConsciousManager:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def is_running(self) -> bool:
        return self.started and not self.stopped


class FakeMemoryManager:
    def __init__(self) -> None:
        self._enabled = False
        self.conscious_ingest = False
        self.sovereign_ingest = False
        self.enable_calls = 0
        self.disable_calls = 0
        self.set_calls = 0
        self.memoria = None

    def set_memoria_instance(self, memoria) -> None:
        self.memoria = memoria
        self.set_calls += 1

    def enable(self, interceptors=None) -> dict[str, object]:
        self._enabled = True
        self.enable_calls += 1
        return {"success": True, "enabled_interceptors": interceptors or []}

    def disable(self) -> dict[str, object]:
        if self._enabled:
            self.disable_calls += 1
        self._enabled = False
        return {"success": True}


class DummyStorageService(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.publisher = None

    def set_sync_publisher(self, publisher) -> None:
        self.publisher = publisher


class FakeMemoria:
    enable_short_term = True

    def __init__(self, *args, **kwargs) -> None:
        self.conscious_ingest = False
        self.sovereign_ingest = False
        self.storage_service = DummyStorageService(conscious_ingest=False)
        self.memory_manager = FakeMemoryManager()
        self.memory_manager._enabled = True
        self.conscious_manager = FakeConsciousManager()
        self.conscious_agent = object()
        self.auto_ingest = kwargs.get("auto_ingest", False)
        self.retention_service = SimpleNamespace(
            cluster_enabled=False,
            config=SimpleNamespace(cluster_gravity_lambda=0.5),
        )
        db_url = kwargs.get("database_connect", "sqlite:///:memory:")
        self.db_manager = SimpleNamespace(
            get_database_info=lambda: {
                "database_type": "sqlite",
                "database_url": db_url,
            }
        )
        self._enabled = False
        self.context_limit = 3
        self.context_orchestrator = None
        self._context_orchestration_config = SimpleNamespace(
            enabled=False,
            token_budget=0,
            max_limit=3,
        )
        self._init_context_calls = 0
        self.memory_manager.set_memoria_instance(self)
        self.sync_state: dict[str, object] = {"enabled": False, "backend": None}
        self._sync_calls: list = []
        self.active_team = kwargs.get("active_team_id")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self, interceptors=None):
        self._enabled = True
        return self.memory_manager.enable(interceptors)

    def configure_sync(self, sync_settings=None, *, backend_override=None):
        self._sync_calls.append((sync_settings, backend_override))
        enabled = (
            bool(getattr(sync_settings, "enabled", False)) if sync_settings else False
        )
        backend = getattr(sync_settings, "backend", None) if sync_settings else None
        if hasattr(backend, "value"):
            backend = backend.value
        self.sync_state = {"enabled": enabled, "backend": backend}
        publisher = object() if enabled else None
        self.storage_service.set_sync_publisher(publisher)

    def _init_context_orchestrator(self) -> None:
        self._init_context_calls += 1
        self.context_orchestrator = object()

    def get_active_team(self):
        return self.active_team


@pytest.fixture
def settings_app(monkeypatch):
    config = ConfigManager()
    config.reset_to_defaults()

    monkeypatch.setenv("MEMORIA_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setattr(
        "memoria_server.api.app_factory.schedule_daily_decrement", lambda app: None
    )
    monkeypatch.setattr(
        "memoria_server.api.app_factory.schedule_roster_verification", lambda app: None
    )
    monkeypatch.setattr(
        "memoria_server.api.app_factory.memoria_pkg.Memoria", FakeMemoria
    )

    app = create_app()
    app.config["TESTING"] = True

    yield app

    config.reset_to_defaults()


@pytest.fixture
def authed_client(settings_app):
    with settings_app.test_client() as client:
        yield client


def test_get_settings_hides_sensitive_fields(settings_app, authed_client):
    response = authed_client.get("/settings", headers={"X-API-Key": "secret"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    agents = data["settings"].get("agents", {})
    openai_secret = data["secrets"]["agents.openai_api_key"]
    if openai_secret:
        assert agents.get("openai_api_key") == "***"
    else:
        assert agents.get("openai_api_key") in {"", None}
    assert data["settings"]["database"]["connection_string"] == "***"
    database_meta = data["meta"].get("database", {})
    assert database_meta.get("summary")
    assert "SQLite" in database_meta.get("summary", "")
    assert database_meta.get("display_url")
    assert database_meta.get("configured") is True
    assert database_meta.get("masked_connection") == "***"
    sync = data["settings"].get("sync", {})
    assert sync.get("enabled") is False
    assert sync.get("backend") in {"none", None}
    assert sync.get("connection_url") in {"", "***"}
    sync_meta = data["meta"].get("sync", {})
    assert sync_meta.get("enabled") is False
    assert sync_meta.get("backend") in {"none", None}
    assert sync_meta.get("connection") == ""
    assert data["secrets"]["sync.connection_url"] is False
    assert "enable_vector_search" in data["settings"]
    memory_settings = data["settings"].get("memory", {})
    assert "context_orchestration" in memory_settings
    assert "context_token_budget" in memory_settings


def test_patch_updates_runtime_state(settings_app, authed_client):
    payload = {
        "agents.conscious_ingest": False,
        "memory.sovereign_ingest": True,
        "memory.context_injection": False,
        "memory.context_orchestration": True,
        "memory.context_token_budget": 450,
        "memory.context_small_query_limit": 2,
        "memory.context_large_query_limit": 5,
        "enable_vector_clustering": True,
        "enable_vector_search": True,
        "integrations.litellm_enabled": True,
        "sync.enabled": True,
        "sync.backend": "memory",
        "sync.connection_url": "redis://localhost/0",
    }
    response = authed_client.patch(
        "/settings",
        json=payload,
        headers={"X-API-Key": "secret"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    for key, value in payload.items():
        if key == "sync.connection_url":
            assert data["updated"][key] == "***"
        else:
            assert data["updated"][key] == value

    settings = data["settings"]
    assert settings["enable_vector_clustering"] is True
    assert settings["enable_vector_search"] is True
    assert settings["agents"]["conscious_ingest"] is False
    assert settings["memory"]["sovereign_ingest"] is True
    assert settings["sync"]["enabled"] is True
    assert settings["sync"]["backend"] == "memory"
    assert settings["sync"]["connection_url"] == "***"
    assert settings["memory"]["context_token_budget"] == 450
    assert settings["memory"]["context_small_query_limit"] == 2
    assert settings["memory"]["context_large_query_limit"] == 5

    runtime = data["runtime"]
    assert runtime["conscious_ingest"] is False
    assert runtime["sovereign_ingest"] is True
    assert runtime["context_injection"] is False
    assert runtime["integrations"]["litellm_enabled"] is True
    assert runtime["sync"]["enabled"] is True
    assert runtime["sync"]["backend"] == "memory"
    assert runtime["context_settings"]["orchestration"] is True
    assert runtime["context_settings"]["token_budget"] == 450

    sync_meta = data["meta"]["sync"]
    assert sync_meta["enabled"] is True
    assert sync_meta["backend"] == "memory"
    assert sync_meta["connection"] == "redis://localhost/0"
    assert data["secrets"]["sync.connection_url"] is True

    memoria = settings_app.config["memoria"]
    assert memoria.conscious_ingest is False
    assert memoria.storage_service.conscious_ingest is False
    assert memoria.sovereign_ingest is True
    assert memoria.auto_ingest is False
    assert memoria.sync_state["enabled"] is True
    assert memoria.sync_state["backend"] == "memory"
    assert memoria.context_orchestrator is not None
    assert memoria._init_context_calls >= 1
    assert memoria.storage_service.publisher is not None
    assert memoria.retention_service.cluster_enabled is True
    assert memoria.retention_service.config.cluster_gravity_lambda == pytest.approx(
        ConfigManager().get_settings().cluster_gravity_lambda
    )
    assert memoria.conscious_manager.stopped is True

    mm = memoria.memory_manager
    assert mm.conscious_ingest is False
    assert mm.sovereign_ingest is True
    assert mm.disable_calls >= 1
    assert mm.enable_calls >= 1
    assert mm.set_calls >= 2

    assert runtime["cluster_enabled"] is True


def test_patch_allows_context_injection_toggle(settings_app, authed_client):
    memoria = settings_app.config["memoria"]
    starting_value = bool(memoria.auto_ingest)
    target_value = not starting_value

    response = authed_client.patch(
        "/settings",
        json={"memory.context_injection": target_value},
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["updated"]["memory.context_injection"] is target_value
    assert payload["settings"]["memory"]["context_injection"] is target_value

    runtime = payload["runtime"]
    assert runtime["context_injection"] is target_value

    memoria = settings_app.config["memoria"]
    assert memoria.auto_ingest is target_value


def test_create_app_with_team_configuration(monkeypatch):
    config = ConfigManager()
    config.reset_to_defaults()

    monkeypatch.setenv("MEMORIA_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MEMORIA_MEMORY__TEAM_MODE", TeamMode.OPTIONAL.value)
    monkeypatch.setenv("MEMORIA_MEMORY__TEAM_DEFAULT_ID", "team-xyz")
    monkeypatch.setenv("MEMORIA_MEMORY__TEAM_SHARE_BY_DEFAULT", "1")
    monkeypatch.setenv("MEMORIA_MEMORY__TEAM_ENFORCE_MEMBERSHIP", "0")
    monkeypatch.setattr(
        "memoria_server.api.app_factory.schedule_daily_decrement", lambda app: None
    )
    monkeypatch.setattr(
        "memoria_server.api.app_factory.schedule_roster_verification", lambda app: None
    )
    monkeypatch.setattr(
        "memoria_server.api.app_factory.memoria_pkg.Memoria", FakeMemoria
    )

    app = create_app()

    try:
        team_config = app.config.get("TEAM_CONFIG")
        assert team_config is not None
        assert team_config["mode"] == TeamMode.OPTIONAL.value
        assert team_config["default_team_id"] == "team-xyz"
        assert team_config["share_by_default"] is True
        assert team_config["enforce_membership"] is False
        assert app.config.get("ACTIVE_TEAM_ID") is None
    finally:
        config.reset_to_defaults()


def test_patch_rejects_invalid_database_scheme(settings_app, authed_client):
    response = authed_client.patch(
        "/settings",
        json={"database.connection_string": "ftp://example.com/db"},
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["status"] == "error"
    assert "scheme" in payload["message"].lower()


def test_patch_rebinds_database_connection(settings_app, authed_client, monkeypatch):
    captured_logs = []

    def capture_log(message, *args, **kwargs):
        captured_logs.append(str(message))

    monkeypatch.setattr("memoria.config.manager.logger.debug", capture_log)

    # Mock the scheduler function to prevent it from running during this test
    monkeypatch.setattr(
        "memoria_server.api.scheduler.reschedule_daily_decrement",
        lambda app: None,
    )
    # Also mock the spatial DB init to prevent it from running with a mock engine
    monkeypatch.setattr(
        "memoria_server.api.spatial_setup.init_spatial_db",
        lambda app: None,
    )

    # Mock the binding function to avoid a real network call
    def mock_refresh_binding(app, settings, database_url):
        fake_memoria = FakeMemoria(database_connect=database_url)
        engine = SimpleNamespace()  # A simple object is enough now
        team_config = {}
        # Manually update the app config, as the real function would
        app.config["DATABASE_URL"] = database_url
        app.config["memoria"] = fake_memoria
        app.config["ENGINE"] = engine
        return (fake_memoria, database_url, None, engine, team_config)

    monkeypatch.setattr(
        "memoria_server.api.app_factory.refresh_memoria_binding",
        mock_refresh_binding,
    )

    original_memoria = settings_app.config["memoria"]
    new_url = "postgresql://user:supersecret@db.local/memoria"

    response = authed_client.patch(
        "/settings",
        json={"database.connection_string": new_url},
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["updated"]["database.connection_string"] == "***"
    assert "supersecret" not in response.get_data(as_text=True)
    assert not any(new_url in message for message in captured_logs)

    rebound_info = body["runtime"].get("database", {})
    assert rebound_info.get("rebound") is True

    db_meta = body["meta"]["database"]
    assert db_meta["configured"] is True
    assert db_meta["masked_connection"] == "***"
    assert "supersecret" not in db_meta["display_url"]

    assert settings_app.config["DATABASE_URL"] == new_url
    rebound_memoria = settings_app.config["memoria"]
    assert rebound_memoria is not original_memoria
    info = rebound_memoria.db_manager.get_database_info()
    assert info["database_url"] == new_url


def test_patch_rejects_disallowed_paths(settings_app, authed_client):
    response = authed_client.patch(
        "/settings",
        json={"agents.openai_api_key": "sk-secret"},
        headers={"X-API-Key": "secret"},
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "OpenAI API key" in data["message"]


def test_patch_requires_object_payload(settings_app, authed_client):
    response = authed_client.patch(
        "/settings",
        json=["enable_cluster_indexing"],
        headers={"X-API-Key": "secret"},
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "JSON body must be an object" in data["message"]


def test_database_summary_masks_credentials(settings_app, authed_client):
    config = ConfigManager()
    config.update_setting(
        "database.connection_string",
        "postgresql://user:secret@localhost:5432/memoria",
    )
    config.update_setting("database.database_type", "postgresql")

    memoria = settings_app.config["memoria"]
    if hasattr(memoria, "db_manager"):
        delattr(memoria, "db_manager")

    response = authed_client.get("/settings", headers={"X-API-Key": "secret"})
    assert response.status_code == 200
    payload = response.get_json()
    database_meta = payload["meta"]["database"]

    assert database_meta["type"] == "postgresql"
    assert database_meta["label"] == "PostgreSQL"
    assert database_meta["display_url"] == "postgresql://localhost:5432/memoria"
    assert database_meta["summary"].startswith("Using PostgreSQL")
    assert "secret" not in database_meta["summary"]
    assert "user" not in database_meta["summary"]


def test_get_settings_schema_includes_metadata(settings_app, authed_client):
    response = authed_client.get("/settings/schema", headers={"X-API-Key": "secret"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    schema = payload["schema"]
    agents = schema["properties"]["agents"]["properties"]
    openai_field = agents["openai_api_key"]
    assert openai_field["x-memoria"]["secret"] is True

    memory_props = schema["properties"]["memory"]["properties"]
    team_mode = memory_props["team_mode"]
    assert team_mode["enum"]
    enum_meta = team_mode["x-memoria"]["enum"]
    assert set(enum_meta["values"]) >= {mode.value for mode in TeamMode}


def test_patch_secret_placeholder_preserves_value(settings_app, authed_client):
    config = ConfigManager()
    secret_url = "redis://user:password@localhost/0"
    config.update_setting("sync.connection_url", secret_url)

    response = authed_client.patch(
        "/settings",
        json={"sync.connection_url": "***"},
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert "sync.connection_url" not in payload["updated"]
    assert payload["settings"]["sync"]["connection_url"] == "***"
    assert payload["meta"]["sync"]["connection"] == "redis://localhost/0"
    assert payload["secrets"]["sync.connection_url"] is True
