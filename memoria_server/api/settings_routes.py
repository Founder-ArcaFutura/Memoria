"""Blueprint exposing runtime configuration operations."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from flask import Blueprint, current_app, jsonify, request

from memoria.config.settings import MemoriaSettings
from memoria.utils.exceptions import ConfigurationError

settings_bp = Blueprint("settings", __name__)

_SCHEMA_CACHE_KEY = "_memoria_settings_schema"
_SECRET_STATE_KEY = "_memoria_settings_secret_state"
_SECRET_PLACEHOLDER = "***"

_SECRET_SETTING_PATHS = {
    "database.connection_string",
    "agents.openai_api_key",
    "agents.anthropic_api_key",
    "agents.gemini_api_key",
    "sync.connection_url",
}

_DYNAMIC_UPDATE_PREFIXES = ("agents.task_model_routes.",)

_ALLOWED_DATABASE_SCHEMES = (
    "sqlite",
    "postgres",
    "postgresql",
    "mysql",
)


def _validate_connection_string(value: Any) -> str:
    """Validate and normalize a database connection string."""

    if not isinstance(value, str):
        raise ValueError("Database connection string must be provided as a string.")

    candidate = value.strip()
    if not candidate:
        raise ValueError("Database connection string cannot be empty.")

    if "://" not in candidate:
        raise ValueError(
            "Database connection string must include a URI scheme (e.g. sqlite:///memory.db)."
        )

    scheme = candidate.split(":", 1)[0].lower()
    base_scheme = scheme.split("+", 1)[0]
    if not any(base_scheme.startswith(prefix) for prefix in _ALLOWED_DATABASE_SCHEMES):
        raise ValueError(
            "Unsupported database scheme. Use sqlite, postgres, or mysql URLs."
        )

    return candidate


def _normalize_database_type(raw_type: Any | None) -> str:
    """Return a normalized lower-case database type string."""

    if isinstance(raw_type, str):
        return raw_type.strip().lower()
    if hasattr(raw_type, "value"):
        return str(raw_type.value).strip().lower()
    return ""


def _humanize_database_label(db_type: str) -> str:
    """Return a user-facing label for the database type."""

    mapping = {
        "sqlite": "SQLite",
        "postgresql": "PostgreSQL",
        "mysql": "MySQL",
    }
    normalized = db_type.strip().lower() if db_type else ""
    if not normalized:
        return "Database"
    return mapping.get(normalized, normalized.replace("_", " ").title())


def _sanitize_database_url(url: str | None, db_type: str | None) -> str:
    """Remove credentials and return a display-safe database location."""

    if not isinstance(url, str) or not url.strip():
        return ""

    working = url.strip()
    normalized_type = _normalize_database_type(db_type)
    if "://" not in working and normalized_type:
        working = f"{normalized_type}://{working}"

    parsed = urlparse(working)
    scheme = parsed.scheme or normalized_type

    if scheme and scheme.startswith("sqlite"):
        path = parsed.path or ""
        if path.startswith("//"):
            path = "/" + path.lstrip("/")
        if path.startswith("/:"):
            path = path[1:]
        if not path and parsed.netloc:
            path = parsed.netloc
        return path or ""

    hostname = parsed.hostname or ""
    port = parsed.port
    path = parsed.path or ""
    netloc = hostname
    if port:
        netloc = f"{hostname}:{port}" if hostname else str(port)
    if not netloc and not path:
        return scheme or ""

    sanitized = ""
    if scheme:
        sanitized += f"{scheme}://"
    sanitized += netloc
    if path:
        sanitized += path
    if parsed.query:
        sanitized += f"?{parsed.query}"
    if parsed.fragment:
        sanitized += f"#{parsed.fragment}"
    return sanitized


def _sanitize_sync_connection(url: str | None) -> str:
    """Mask credentials and return a display-friendly sync connection description."""

    if not isinstance(url, str) or not url.strip():
        return ""

    parsed = urlparse(url.strip())
    hostname = parsed.hostname or ""
    port = parsed.port
    path = parsed.path or ""
    netloc = hostname
    if port:
        netloc = f"{hostname}:{port}" if hostname else str(port)

    components = []
    scheme = parsed.scheme or ""
    if scheme:
        components.append(f"{scheme}://")
    components.append(netloc)
    components.append(path)
    if parsed.query:
        components.append(f"?{parsed.query}")
    if parsed.fragment:
        components.append(f"#{parsed.fragment}")

    return "".join(part for part in components if part)


def _normalize_task_route_payload(value: Any) -> dict[str, Any]:
    """Validate and normalize a task routing update payload."""

    if not isinstance(value, dict):
        raise ValueError("Task model route updates must be provided as an object.")

    provider = value.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("Task model routes require a non-empty provider value.")

    model = value.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError("Model overrides must be strings when provided.")

    fallback_raw = value.get("fallback", [])
    if isinstance(fallback_raw, str):
        fallback = [part.strip() for part in fallback_raw.split(",") if part.strip()]
    elif isinstance(fallback_raw, (list, tuple)):
        fallback = [str(item).strip() for item in fallback_raw if str(item).strip()]
    else:
        fallback = []

    return {"provider": provider.strip(), "model": model, "fallback": fallback}


def _resolve_database_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """Build a UI-safe database descriptor from config and runtime info."""

    database_raw = raw.get("database") if isinstance(raw, dict) else {}
    runtime_info: dict[str, Any] = {}
    try:
        memoria = current_app.config.get("memoria")  # type: ignore[arg-type]
    except RuntimeError:  # pragma: no cover - happens outside request context
        memoria = None
    if memoria is not None:
        db_manager = getattr(memoria, "db_manager", None)
        if db_manager is not None:
            try:
                runtime_info = db_manager.get_database_info() or {}
            except Exception:  # pragma: no cover - defensive runtime guard
                runtime_info = {}

    configured_type = None
    configured_url = None
    if isinstance(database_raw, dict):
        configured_type = database_raw.get("database_type")
        configured_url = database_raw.get("connection_string")

    runtime_type = runtime_info.get("database_type")
    runtime_url = runtime_info.get("database_url")

    normalized_type = _normalize_database_type(runtime_type or configured_type)
    display_url = _sanitize_database_url(runtime_url, normalized_type)
    if not display_url:
        display_url = _sanitize_database_url(configured_url, normalized_type)

    label = _humanize_database_label(normalized_type)
    if display_url:
        summary = f"Using {label}: {display_url}" if label != "Database" else f"Database: {display_url}"
    else:
        summary = (
            f"Using {label}" if label != "Database" else "Database configuration unavailable"
        )

    configured = False
    if isinstance(configured_url, str) and configured_url.strip():
        configured = True
    elif isinstance(runtime_url, str) and runtime_url.strip():
        configured = True

    masked_connection = "***" if configured else ""

    return {
        "type": normalized_type,
        "label": label,
        "display_url": display_url,
        "summary": summary,
        "configured": configured,
        "masked_connection": masked_connection,
    }


def _sanitize_sync_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal safe snapshot of the sync configuration."""

    enabled = bool(raw.get("enabled")) if isinstance(raw, dict) else False
    backend = ""
    if isinstance(raw, dict):
        backend = raw.get("backend")
    if hasattr(backend, "value"):
        backend = backend.value
    backend = backend or ("memory" if enabled else "none")
    connection_url = raw.get("connection_url") if isinstance(raw, dict) else None
    sanitized_connection = _sanitize_sync_connection(connection_url)
    options = raw.get("options") if isinstance(raw, dict) else None
    options_configured = bool(options)
    channel = raw.get("channel") if isinstance(raw, dict) else ""
    if hasattr(channel, "value"):
        channel = str(channel.value)
    channel_str = str(channel) if channel else ""

    return {
        "enabled": enabled,
        "backend": backend,
        "connection": sanitized_connection,
        "has_connection": bool(connection_url),
        "options_configured": options_configured,
        "channel": channel_str,
    }


def _jsonify_settings(value: Any) -> Any:
    """Return JSON-friendly data by normalising enums, paths, and containers."""

    if isinstance(value, dict):
        return {str(key): _jsonify_settings(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_settings(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _get_secret_state() -> dict[str, bool]:
    """Return (and initialise) the stored secret placeholders map."""

    store = current_app.config.setdefault(_SECRET_STATE_KEY, {})
    if not isinstance(store, dict):
        store = {}
        current_app.config[_SECRET_STATE_KEY] = store
    return store


def _mask_secret_fields(
    json_settings: dict[str, Any],
    originals: dict[str, Any],
    secret_paths: set[str],
    placeholders: dict[str, bool],
) -> dict[str, Any]:
    """Replace secret values with placeholders while updating stored state."""

    masked = deepcopy(json_settings)
    for path in secret_paths:
        parts = path.split(".") if path else []
        if not parts:
            continue

        masked_cursor: Any = masked
        original_cursor: Any = originals
        missing = False

        for segment in parts[:-1]:
            if not isinstance(masked_cursor, dict) or not isinstance(original_cursor, dict):
                missing = True
                break
            masked_cursor = masked_cursor.get(segment)
            original_cursor = original_cursor.get(segment)

        if missing or not isinstance(masked_cursor, dict):
            placeholders[path] = False
            continue

        key = parts[-1]
        original_value = None
        if isinstance(original_cursor, dict):
            original_value = original_cursor.get(key)

        has_value = bool(original_value)
        placeholders[path] = has_value
        masked_cursor[key] = _SECRET_PLACEHOLDER if has_value else ""

    return masked


def _secret_state_snapshot(secret_paths: set[str], placeholders: dict[str, bool]) -> dict[str, bool]:
    """Return a serialisable representation of secret placeholder state."""

    return {path: bool(placeholders.get(path)) for path in secret_paths}


def _collect_schema_context() -> dict[str, Any]:
    """Return cached schema metadata including secret and enum flags."""

    cache = current_app.config.get(_SCHEMA_CACHE_KEY)
    if cache:
        return cache

    schema = MemoriaSettings.model_json_schema()
    field_index: dict[str, dict[str, Any]] = {}
    leaf_index: dict[str, dict[str, Any]] = {}
    secret_paths: set[str] = set()

    def _walk(node: Any, path: list[str]) -> None:
        if not isinstance(node, dict):
            return

        path_str = ".".join(path) if path else ""
        if path:
            field_index[path_str] = node
            if node.get("type") != "object" or not node.get("properties"):
                leaf_index[path_str] = node

        meta: dict[str, Any] = dict(node.get("x-memoria", {}))
        if path_str in _SECRET_SETTING_PATHS:
            meta["secret"] = True
            secret_paths.add(path_str)

        if "enum" in node:
            enum_values = list(node.get("enum") or [])
            enum_meta: dict[str, Any] = {"values": enum_values}
            enum_names = node.get("enumNames")
            if isinstance(enum_names, (list, tuple)):
                enum_meta["labels"] = list(enum_names)
            meta.setdefault("enum", enum_meta)

        if meta:
            node["x-memoria"] = meta
        else:
            node.pop("x-memoria", None)

        for key, child in (node.get("properties") or {}).items():
            _walk(child, path + [key])

        items = node.get("items")
        if isinstance(items, dict):
            _walk(items, path + ["[]"])

    _walk(schema, [])

    cache = {
        "schema": schema,
        "fields": field_index,
        "leaf": leaf_index,
        "secret_paths": secret_paths,
    }
    current_app.config[_SCHEMA_CACHE_KEY] = cache
    return cache


def _prepare_settings_payload(
    settings_obj,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    """Return masked settings, metadata, and secret placeholder state."""

    schema_context = _collect_schema_context()
    placeholders = _get_secret_state()

    raw_settings = settings_obj.dict()
    json_settings = _jsonify_settings(raw_settings)
    masked_settings = _mask_secret_fields(
        json_settings,
        raw_settings,
        schema_context["secret_paths"],
        placeholders,
    )

    sync_raw = raw_settings.get("sync") if isinstance(raw_settings, dict) else {}
    metadata = {
        "database": _resolve_database_settings(raw_settings),
        "sync": _sanitize_sync_settings(sync_raw if isinstance(sync_raw, dict) else {}),
    }

    migrations_meta = current_app.config.get("STRUCTURAL_MIGRATIONS")
    if not isinstance(migrations_meta, dict):
        try:
            from memoria.cli import get_structural_migration_status

            migrations_meta = get_structural_migration_status()
        except Exception:  # pragma: no cover - defensive import fallback
            migrations_meta = {}

    capabilities_meta = current_app.config.get("PROVIDER_CAPABILITIES")
    if not isinstance(capabilities_meta, list):
        try:
            from memoria.cli import get_provider_capability_status

            capabilities_meta = get_provider_capability_status()
        except Exception:  # pragma: no cover - defensive import fallback
            capabilities_meta = []

    metadata["migrations"] = migrations_meta
    metadata["capabilities"] = capabilities_meta

    secrets = _secret_state_snapshot(schema_context["secret_paths"], placeholders)
    return masked_settings, metadata, secrets


def _apply_runtime_settings(app) -> dict[str, Any]:
    from .app_factory import _sync_runtime_toggles as apply_helper

    return apply_helper(app)


@settings_bp.route("/settings/schema", methods=["GET"])
def get_settings_schema():
    """Return the settings schema including UI metadata."""

    schema_context = _collect_schema_context()
    return jsonify({"status": "ok", "schema": deepcopy(schema_context["schema"])})


@settings_bp.route("/settings", methods=["GET"])
def get_settings():
    """Return a sanitized view of the current configuration."""

    config_manager = current_app.config.get("config_manager")
    if config_manager is None:
        return jsonify({"status": "error", "message": "Config manager unavailable"}), 503

    settings_obj = config_manager.get_settings()
    settings, metadata, secrets = _prepare_settings_payload(settings_obj)

    return jsonify({"status": "ok", "settings": settings, "meta": metadata, "secrets": secrets})


@settings_bp.route("/settings", methods=["PATCH"])
def update_settings():
    """Apply dot-path configuration updates and refresh runtime state."""

    config_manager = current_app.config.get("config_manager")
    if config_manager is None:
        return jsonify({"status": "error", "message": "Config manager unavailable"}), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "JSON body must be an object with at least one setting.",
                }
            ),
            400,
        )

    schema_context = _collect_schema_context()
    leaf_fields: dict[str, Any] = schema_context["leaf"]

    invalid_keys = [
        key
        for key in payload
        if key not in leaf_fields
        and not any(key.startswith(prefix) for prefix in _DYNAMIC_UPDATE_PREFIXES)
    ]
    if invalid_keys:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Unsupported settings: {', '.join(sorted(invalid_keys))}",
                }
            ),
            400,
        )

    applied: dict[str, Any] = {}
    connection_update: str | None = None
    secret_state = _get_secret_state()
    try:
        for key, value in payload.items():
            if any(key.startswith(prefix) for prefix in _DYNAMIC_UPDATE_PREFIXES):
                try:
                    normalized_route = _normalize_task_route_payload(value)
                except ValueError as exc:
                    return jsonify({"status": "error", "message": str(exc)}), 400
                config_manager.update_setting(key, normalized_route)
                applied[key] = normalized_route
                continue

            field_schema = leaf_fields.get(key, {})
            field_meta = field_schema.get("x-memoria", {})
            is_secret = bool(field_meta.get("secret"))

            if key == "database.connection_string":
                try:
                    normalized = _validate_connection_string(value)
                except ValueError as exc:
                    return jsonify({"status": "error", "message": str(exc)}), 400
                config_manager.update_setting(key, normalized)
                applied[key] = _SECRET_PLACEHOLDER
                secret_state[key] = True
                connection_update = normalized
                continue

            if is_secret and isinstance(value, str):
                trimmed = value.strip()
                if trimmed == _SECRET_PLACEHOLDER:
                    # Client is signalling the stored secret should remain unchanged
                    continue
                if not trimmed:
                    config_manager.update_setting(key, None)
                    applied[key] = ""
                    secret_state[key] = False
                    continue
                value = trimmed

            config_manager.update_setting(key, value)
            if is_secret:
                applied[key] = _SECRET_PLACEHOLDER if value else ""
                secret_state[key] = bool(value)
            else:
                applied[key] = value
    except ConfigurationError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    try:
        config_manager.save_configuration()
    except ConfigurationError as exc:
        # Non-fatal, but should be reported
        current_app.logger.error(f"Failed to save settings: {exc}")

    settings_obj = config_manager.get_settings()

    rebound = False
    if connection_update is not None:
        from .app_factory import refresh_memoria_binding
        from .scheduler import reschedule_daily_decrement
        from .spatial_setup import init_spatial_db

        refresh_memoria_binding(
            current_app,
            settings=settings_obj,
            database_url=connection_update,
        )
        init_spatial_db(current_app)
        reschedule_daily_decrement(current_app)
        rebound = True

    applied_runtime = _apply_runtime_settings(current_app)
    if rebound:
        applied_runtime.setdefault("database", {})["rebound"] = True

    settings, metadata, secrets = _prepare_settings_payload(settings_obj)

    return jsonify(
        {
            "status": "ok",
            "updated": applied,
            "runtime": applied_runtime,
            "settings": settings,
            "meta": metadata,
            "secrets": secrets,
        }
    )
