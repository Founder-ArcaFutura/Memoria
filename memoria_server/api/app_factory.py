"""Flask application factory and setup helpers for the Memoria API."""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, current_app, g
from flask_cors import CORS
from loguru import logger
from sqlalchemy.engine import make_url

import memoria as memoria_pkg
from memoria.config import ConfigManager
from memoria.config.settings import MemoriaSettings, TeamMode
from memoria.core.context_orchestration import ContextOrchestrationConfig
from memoria.core.memory import build_provider_options as core_build_provider_options
from memoria.policy.enforcement import PolicyEnforcementEngine
from memoria.cli import (
    check_provider_capabilities,
    get_provider_capability_status,
    get_structural_migration_status,
    run_structural_migrations,
)

from .admin_routes import admin_bp
from .memory_routes import memory_bp
from .search_routes import search_bp
from .utility_routes import utility_bp
from .manual_entry import manual_entry_bp
from .settings_routes import settings_bp
from .policy_routes import policy_bp
from .governance_routes import governance_bp
from .ui_routes import (
    UI_SESSION_COOKIE_NAME,
    ui_bp,
    validate_ui_session_token,
)

from .scheduler import (
    cancel_daily_decrement_timer,
    reschedule_daily_decrement,
    restart_daily_decrement_schedule,
    schedule_daily_decrement,
    schedule_roster_rotation,
    schedule_roster_verification,
    schedule_override_expiry,
    set_service_metadata_value,
)
from .spatial_setup import init_spatial_db


_TEAM_ROUTE_PREFIXES = ("/memory/teams", "/memory/workspaces")


def _parse_header_list(raw_value: str | None) -> list[str]:
    """Return a normalised list of header names from configuration strings."""

    if not raw_value:
        return []

    return [segment.strip() for segment in raw_value.split(",") if segment.strip()]


def _derive_sqlite_path(database_url: str) -> str | None:
    """Return a filesystem path for SQLite connection strings."""

    if not isinstance(database_url, str):
        return None

    working = database_url.strip()
    if not working.startswith("sqlite"):
        return None

    try:
        url = make_url(working)
    except Exception:
        return None

    database = (url.database or "").strip()
    if not database or database == ":memory:":
        return None

    candidate = Path(database).expanduser()
    if candidate.is_absolute():
        normalized = candidate
    else:
        normalized = (Path.cwd() / candidate).resolve()

    return str(normalized)


def _resolve_database_url(
    settings: MemoriaSettings | None, override: str | None = None
) -> str:
    """Determine the preferred database URL from overrides, env, or settings."""

    if isinstance(override, str) and override.strip():
        return override.strip()

    env_value = os.getenv("DATABASE_URL")
    if isinstance(env_value, str) and env_value.strip():
        return env_value.strip()

    if settings is not None:
        try:
            candidate = settings.get_database_url()
        except Exception:  # pragma: no cover - defensive
            candidate = None
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    return "sqlite:///memoria.db"


def build_provider_options(settings: MemoriaSettings | None) -> dict[str, object | None]:
    """Collect settings and environment driven options for :class:`memoria.Memoria`."""

    return core_build_provider_options(settings)


def refresh_memoria_binding(
    app: Flask,
    *,
    settings: MemoriaSettings | None,
    database_url: str | None,
) -> tuple[
    memoria_pkg.Memoria,
    str,
    str | None,
    object | None,
    dict[str, object | None],
]:
    """(Re)initialize the Memoria instance for the provided database URL."""

    resolved_url = (database_url or "").strip() or "sqlite:///memoria.db"
    db_path = _derive_sqlite_path(resolved_url)
    if db_path:
        path_obj = Path(db_path)
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - defensive
            pass
        db_path = str(path_obj)

    provider_options = build_provider_options(settings)

    auto_enabled = True
    if settings is not None and getattr(settings, "database", None):
        auto_enabled = bool(getattr(settings.database, "migration_auto", True))

    run_structural_migrations(resolved_url, auto_enabled=auto_enabled, stream=None)
    check_provider_capabilities(settings, interactive=False, stream=None)

    memoria_instance = memoria_pkg.Memoria(
        database_connect=resolved_url,
        **provider_options,
    )

    team_config: dict[str, object | None] = {
        "mode": TeamMode.DISABLED.value,
        "default_team_id": None,
        "share_by_default": False,
        "enforce_membership": True,
    }
    workspace_config: dict[str, object | None] = {
        "mode": TeamMode.DISABLED.value,
        "default_workspace_id": None,
    }
    if settings is not None:
        memory_cfg = getattr(settings, "memory", None)
        if memory_cfg is not None:
            raw_mode = getattr(memory_cfg, "team_mode", TeamMode.DISABLED)
            if isinstance(raw_mode, TeamMode):
                team_config["mode"] = raw_mode.value
            elif raw_mode is not None:
                team_config["mode"] = str(raw_mode)
            team_config["default_team_id"] = getattr(
                memory_cfg, "team_default_id", None
            )
            team_config["share_by_default"] = bool(
                getattr(memory_cfg, "team_share_by_default", False)
            )
            team_config["enforce_membership"] = bool(
                getattr(memory_cfg, "team_enforce_membership", True)
            )
            workspace_mode = getattr(
                memory_cfg, "workspace_mode", TeamMode.DISABLED
            )
            if isinstance(workspace_mode, TeamMode):
                workspace_config["mode"] = workspace_mode.value
            elif workspace_mode is not None:
                workspace_config["mode"] = str(workspace_mode)
            workspace_config["default_workspace_id"] = getattr(
                memory_cfg, "workspace_default_id", None
            )

    memoria_instance = _apply_runtime_settings(app, memoria_instance, settings)

    engine = getattr(getattr(memoria_instance, "db_manager", None), "engine", None)
    app.config.update(
        {
            "DATABASE_URL": resolved_url,
            "ENGINE": engine,
            "TEAM_CONFIG": team_config,
            "WORKSPACE_CONFIG": workspace_config,
        }
    )
    if db_path:
        app.config["DB_PATH"] = db_path
    else:
        app.config.pop("DB_PATH", None)

    app.config["STRUCTURAL_MIGRATIONS"] = get_structural_migration_status()
    app.config["PROVIDER_CAPABILITIES"] = get_provider_capability_status()

    return memoria_instance, resolved_url, db_path, engine, team_config


def _sync_runtime_toggles(app: Flask) -> dict:
    """Synchronize runtime toggles on the active :class:`memoria.Memoria` instance."""

    memoria = app.config.get("memoria")
    config_manager = app.config.get("config_manager")

    if memoria is None or config_manager is None:
        return {}

    try:
        settings = config_manager.get_settings()
    except Exception:  # pragma: no cover - defensive
        return {}

    applied: dict[str, object] = {}

    agents_settings = getattr(settings, "agents", None)
    memory_settings = getattr(settings, "memory", None)

    conscious_target = bool(
        getattr(memoria, "enable_short_term", True)
        and (
            getattr(agents_settings, "conscious_ingest", False)
            if agents_settings is not None
            else False
        )
    )
    if getattr(memoria, "conscious_ingest", None) != conscious_target:
        memoria.conscious_ingest = conscious_target
    storage_service = getattr(memoria, "storage_service", None)
    if storage_service is not None:
        setattr(storage_service, "conscious_ingest", conscious_target)
    memory_manager = getattr(memoria, "memory_manager", None)
    if memory_manager is not None:
        setattr(memory_manager, "conscious_ingest", conscious_target)
    if conscious_target and getattr(memoria, "conscious_agent", None):
        manager = getattr(memoria, "conscious_manager", None)
        if manager is not None and hasattr(manager, "start"):
            manager.start()
    elif not conscious_target:
        manager = getattr(memoria, "conscious_manager", None)
        if manager is not None and hasattr(manager, "stop"):
            manager.stop()
    applied["conscious_ingest"] = conscious_target

    sovereign_target = bool(
        getattr(memory_settings, "sovereign_ingest", False)
        if memory_settings is not None
        else False
    )
    current_sovereign = getattr(memoria, "sovereign_ingest", False)
    if current_sovereign != sovereign_target:
        memoria.sovereign_ingest = sovereign_target
    if memory_manager is not None:
        setattr(memory_manager, "sovereign_ingest", sovereign_target)

    mm_enabled = False
    if memory_manager is not None:
        mm_enabled = bool(getattr(memory_manager, "_enabled", False))
        disable_fn = getattr(memory_manager, "disable", None)
        if mm_enabled and callable(disable_fn):
            try:
                disable_fn()
            except Exception:  # pragma: no cover - defensive
                pass

    if memory_manager is not None:
        set_instance = getattr(memory_manager, "set_memoria_instance", None)
        if callable(set_instance):
            try:
                set_instance(memoria)
            except Exception:  # pragma: no cover - defensive
                pass
        if mm_enabled:
            enable_fn = getattr(memory_manager, "enable", None)
            if callable(enable_fn):
                try:
                    enable_fn()
                except Exception:  # pragma: no cover - defensive
                    pass

    applied["sovereign_ingest"] = sovereign_target

    context_target = bool(
        getattr(memory_settings, "context_injection", False)
        if memory_settings is not None
        else False
    )
    if getattr(memoria, "auto_ingest", None) != context_target:
        memoria.auto_ingest = context_target
    applied["context_injection"] = context_target

    context_applied: dict[str, object] = {}
    if memory_settings is not None:
        if hasattr(memoria, "context_limit"):
            memoria.context_limit = getattr(
                memory_settings,
                "context_limit",
                getattr(memoria, "context_limit", 3),
            )
            context_applied["context_limit"] = memoria.context_limit

        try:
            new_context_config = ContextOrchestrationConfig.from_settings(
                memory_settings
            )
        except Exception:  # pragma: no cover - defensive
            new_context_config = None

        if new_context_config is not None:
            setattr(memoria, "_context_orchestration_config", new_context_config)
            init_fn = getattr(memoria, "_init_context_orchestrator", None)
            if new_context_config.enabled and callable(init_fn):
                init_fn()
            elif not new_context_config.enabled:
                setattr(memoria, "context_orchestrator", None)
            context_applied.update(
                {
                    "orchestration": bool(new_context_config.enabled),
                    "token_budget": new_context_config.token_budget,
                    "max_limit": new_context_config.max_limit,
                }
            )

    if context_applied:
        applied["context_settings"] = context_applied

    cluster_enabled = bool(
        getattr(settings, "enable_cluster_indexing", False)
        and getattr(settings, "use_db_clusters", False)
    )
    retention_service = getattr(memoria, "retention_service", None)
    if retention_service is not None:
        setattr(retention_service, "cluster_enabled", cluster_enabled)
        config = getattr(retention_service, "config", None)
        if config is not None:
            setattr(config, "cluster_gravity_lambda", getattr(settings, "cluster_gravity_lambda", 0.0))
    applied["cluster_enabled"] = cluster_enabled

    integrations = getattr(settings, "integrations", None)
    if integrations is not None:
        applied["integrations"] = {
            "litellm_enabled": getattr(integrations, "litellm_enabled", False),
            "openai_wrapper_enabled": getattr(
                integrations, "openai_wrapper_enabled", False
            ),
            "anthropic_wrapper_enabled": getattr(
                integrations, "anthropic_wrapper_enabled", False
            ),
        }

    sync_settings = getattr(settings, "sync", None)
    configure_sync = getattr(memoria, "configure_sync", None)
    if callable(configure_sync):
        try:
            configure_sync(sync_settings)
        except Exception:  # pragma: no cover - defensive
            logger.opt(exception=True).debug("Failed to apply sync runtime configuration")
    backend_value = getattr(sync_settings, "backend", None)
    if hasattr(backend_value, "value"):
        backend_value = backend_value.value
    applied["sync"] = {
        "enabled": bool(getattr(sync_settings, "enabled", False)),
        "backend": backend_value,
    }

    return applied


def _apply_runtime_settings(
    app: Flask, memoria: memoria_pkg.Memoria, settings: MemoriaSettings | None
) -> memoria_pkg.Memoria:
    """Apply runtime configuration toggles to a Memoria instance."""

    sovereign_ingest = False
    litellm_enabled = True
    conscious_ingest_enabled = False
    context_injection_enabled = False

    if settings is not None:
        litellm_enabled = False
        memory_settings = getattr(settings, "memory", None)
        integrations = getattr(settings, "integrations", None)
        agent_settings = getattr(settings, "agents", None)

        if memory_settings is not None:
            sovereign_ingest = bool(getattr(memory_settings, "sovereign_ingest", False))
            context_injection_enabled = bool(
                getattr(memory_settings, "context_injection", False)
            )
            memoria.auto_ingest = context_injection_enabled
        if integrations is not None:
            litellm_enabled = bool(getattr(integrations, "litellm_enabled", False))
        if agent_settings is not None:
            conscious_ingest_enabled = bool(
                getattr(agent_settings, "conscious_ingest", False)
            )

    memoria.sovereign_ingest = sovereign_ingest

    should_enable = (
        settings is None
        or sovereign_ingest
        or litellm_enabled
        or conscious_ingest_enabled
        or context_injection_enabled
    )

    if should_enable and not memoria.is_enabled:
        memoria.enable()
    elif not should_enable and memoria.is_enabled:
        memoria.disable()

    app.config["memoria"] = memoria
    return memoria


def create_app() -> Flask:
    """Application factory for the Memoria API."""

    app = Flask(__name__)

    expected_api_key = os.getenv("MEMORIA_API_KEY")
    secret_key = (
        os.getenv("FLASK_SECRET_KEY")
        or os.getenv("MEMORIA_UI_SESSION_SECRET")
        or expected_api_key
    )
    if not secret_key:
        secret_key = secrets.token_hex(32)
    app.config["SECRET_KEY"] = secret_key

    allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*")
    if allowed_origins != "*":
        allowed_origins = [o.strip() for o in allowed_origins.split(",") if o.strip()]

    CORS(
        app,
        resources={r"/clusters.*": {"origins": allowed_origins, "methods": ["GET", "POST"]}},
    )

    configured_headers: list[str] = []
    configured_headers.extend(
        _parse_header_list(os.getenv("MEMORIA_USER_HEADER"))
    )
    configured_headers.extend(
        _parse_header_list(os.getenv("MEMORIA_USER_HEADER_FALLBACKS"))
    )
    defaults = ["X-Memoria-User", "X-User-Id"]
    for header in defaults:
        if header not in configured_headers:
            configured_headers.append(header)
    app.config["USER_ID_HEADER_NAMES"] = configured_headers
    app.config["TEAM_ROUTE_PREFIXES"] = _TEAM_ROUTE_PREFIXES

    config_manager = ConfigManager()
    try:
        config_manager.auto_load()
    except Exception:  # pragma: no cover - defensive
        logger.opt(exception=True).debug(
            "Config auto-load failed; continuing with defaults"
        )

    try:
        settings: MemoriaSettings | None = config_manager.get_settings()
    except Exception:  # pragma: no cover - defensive
        settings = None

    database_url = _resolve_database_url(settings)
    (
        memoria_instance,
        database_url,
        db_path,
        engine,
        team_config,
    ) = refresh_memoria_binding(
        app,
        settings=settings,
        database_url=database_url,
    )

    get_active_team = getattr(memoria_instance, "get_active_team", None)
    active_team_id = get_active_team() if callable(get_active_team) else None

    decrement_tz = os.getenv("DECREMENT_TZ", "UTC")
    serve_ui = os.getenv("MEMORIA_SERVE_UI", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    default_ui_path = (Path(__file__).resolve().parent.parent / "dashboard").resolve()
    ui_path_env = os.getenv("MEMORIA_UI_PATH")
    if ui_path_env:
        try:
            ui_path = Path(ui_path_env).expanduser().resolve()
        except Exception:  # pragma: no cover - defensive
            ui_path = default_ui_path
    else:
        ui_path = default_ui_path
    cookie_name = os.getenv("MEMORIA_UI_SESSION_COOKIE", UI_SESSION_COOKIE_NAME)

    app.config.update(
        {
            "memoria": memoria_instance,
            "config_manager": config_manager,
            "DATABASE_URL": database_url,
            "EXPECTED_API_KEY": expected_api_key,
            "DECREMENT_TZ": decrement_tz,
            "ENGINE": engine,
            "SERVE_UI": serve_ui,
            "UI_PATH": str(ui_path),
            "UI_SESSION_COOKIE_NAME": cookie_name,
            "TEAM_CONFIG": team_config,
            "ACTIVE_TEAM_ID": active_team_id,
        }
    )
    app.config["policy_metrics_collector"] = PolicyEnforcementEngine.get_global().metrics
    if db_path:
        app.config["DB_PATH"] = db_path

    if serve_ui:
        app.register_blueprint(ui_bp)

    def _request_targets_team_operations() -> bool:
        if request.method == "OPTIONS":
            return False

        endpoint = (request.endpoint or "").lower()
        blueprint = (request.blueprint or "").lower()

        if blueprint in {"ui"} or endpoint.startswith("ui."):
            return False
        if blueprint == "admin" or endpoint.startswith("admin."):
            return False

        prefixes = current_app.config.get("TEAM_ROUTE_PREFIXES") or _TEAM_ROUTE_PREFIXES
        path = request.path or ""
        if any(path.startswith(prefix) for prefix in prefixes):
            return True

        return blueprint in {"memory", "search"}

    def _extract_request_user() -> str | None:
        header_names = current_app.config.get("USER_ID_HEADER_NAMES") or []
        for header_name in header_names:
            value = request.headers.get(header_name)
            if isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    return candidate
        return None

    @app.before_request
    def require_api_key():
        expected = current_app.config.get("EXPECTED_API_KEY")
        if not expected:
            return None

        if request.method == "OPTIONS":
            return None

        endpoint = request.endpoint
        if endpoint is None:
            return None

        endpoint_name = endpoint or ""
        blueprint = request.blueprint

        if endpoint_name == "admin.create_session":
            return None

        ui_endpoints = {"ui.index", "ui.asset", "ui.session"}
        if blueprint == "ui" or endpoint_name in ui_endpoints:
            return None

        provided = request.headers.get("X-API-Key")
        if provided == expected:
            return None

        cookie_name = current_app.config.get("UI_SESSION_COOKIE_NAME", UI_SESSION_COOKIE_NAME)
        session_token = request.cookies.get(cookie_name)
        if validate_ui_session_token(current_app, session_token, expected):
            return None

        return jsonify({"status": "error", "message": "Unauthorized"}), 401


    @app.before_request
    def bind_memoria_user():
        memoria = current_app.config.get("memoria")
        if memoria is None:
            return None

        storage_service = getattr(memoria, "storage_service", None)
        previous_state = {
            "user_id": getattr(memoria, "user_id", None),
            "namespace": getattr(memoria, "namespace", None),
        }
        if storage_service is not None:
            previous_state["storage_namespace"] = getattr(
                storage_service, "namespace", None
            )
        g._memoria_request_state = previous_state

        request_user = _extract_request_user()
        memoria.user_id = request_user

        if storage_service is not None and hasattr(memoria, "namespace"):
            storage_service.namespace = getattr(memoria, "namespace")

        team_config = current_app.config.get("TEAM_CONFIG") or {}
        enforce_membership = bool(team_config.get("enforce_membership", True))
        team_mode_value = str(team_config.get("mode") or "").lower()
        if not team_mode_value:
            team_mode_value = TeamMode.DISABLED.value
        team_enabled = bool(getattr(memoria, "team_memory_enabled", False)) or (
            team_mode_value != TeamMode.DISABLED.value
        )

        if (
            enforce_membership
            and team_enabled
            and _request_targets_team_operations()
            and not request_user
        ):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "A user identity header is required for team operations.",
                    }
                ),
                400,
            )

        return None

    @app.teardown_request
    def restore_memoria_user(_exc: BaseException | None) -> None:
        memoria = current_app.config.get("memoria")
        state = getattr(g, "_memoria_request_state", None)
        if memoria is None or state is None:
            return None

        try:
            del g._memoria_request_state
        except Exception:  # pragma: no cover - defensive cleanup
            pass

        memoria.user_id = state.get("user_id")
        if "namespace" in state:
            memoria.namespace = state.get("namespace")
        storage_service = getattr(memoria, "storage_service", None)
        if storage_service is not None and "storage_namespace" in state:
            storage_service.namespace = state.get("storage_namespace")

        return None

    app.register_blueprint(admin_bp)
    app.register_blueprint(memory_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(utility_bp)
    app.register_blueprint(manual_entry_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(policy_bp)
    app.register_blueprint(governance_bp)

    init_spatial_db(app)
    schedule_daily_decrement(app)
    schedule_roster_verification(app)
    schedule_roster_rotation(app)
    schedule_override_expiry(app)

    bootstrap_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    rotation_bootstrap = {
        "status": "scheduled",
        "scheduled_at": bootstrap_time,
        "generated_at": None,
        "next_check_at": None,
        "cadence_minutes": 60,
    }
    override_bootstrap = {
        "status": "scheduled",
        "scheduled_at": bootstrap_time,
        "generated_at": None,
        "next_check_at": None,
        "cadence_minutes": 60,
    }

    app.config["ROSTER_ROTATION_STATUS"] = rotation_bootstrap
    app.config["OVERRIDE_EXPIRY_STATUS"] = override_bootstrap

    try:
        set_service_metadata_value(app, "roster_rotation_status", json.dumps(rotation_bootstrap))
    except Exception:  # pragma: no cover - defensive bootstrap
        logger.opt(exception=True).debug("Failed to persist initial roster rotation status")

    try:
        set_service_metadata_value(app, "override_expiry_status", json.dumps(override_bootstrap))
    except Exception:  # pragma: no cover - defensive bootstrap
        logger.opt(exception=True).debug("Failed to persist initial override expiry status")

    _sync_runtime_toggles(app)

    return app
