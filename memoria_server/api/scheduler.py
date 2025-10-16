"""Background scheduling helpers for Memoria's Flask API."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from flask import Flask
from loguru import logger
from sqlalchemy import create_engine, text
from zoneinfo import ZoneInfo

from memoria.policy.roster import apply_rotation_metadata, verify_escalation_contacts
from memoria.policy.schemas import apply_override_expiry


_BACKGROUND_TIMER_REGISTRY = "memoria_background_timers"
_DAILY_DECREMENT_TIMER = "daily_decrement"
_ROSTER_VERIFICATION_TIMER = "roster_verification"
_ROSTER_ROTATION_TIMER = "roster_rotation"
_OVERRIDE_EXPIRY_TIMER = "override_expiry"
_ROSTER_METADATA_KEY = "roster_verification_status"
_ROSTER_CONFIG_KEY = "ROSTER_VERIFICATION_STATUS"
_ROTATION_METADATA_KEY = "roster_rotation_status"
_ROTATION_CONFIG_KEY = "ROSTER_ROTATION_STATUS"
_OVERRIDE_METADATA_KEY = "override_expiry_status"
_OVERRIDE_CONFIG_KEY = "OVERRIDE_EXPIRY_STATUS"


def _resolve_flask_app(app: Flask) -> Flask:
    """Return the underlying Flask app when a LocalProxy is provided."""

    get_current = getattr(app, "_get_current_object", None)
    if callable(get_current):
        try:
            return get_current()
        except RuntimeError:  # pragma: no cover - fall back to provided instance
            pass
    return app


def _ensure_daemon(timer: threading.Timer) -> None:
    """Best-effort request that a timer thread runs as a daemon."""

    try:
        timer.daemon = True
    except Exception:  # pragma: no cover - attribute may be missing on dummies
        pass


def _isoformat_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _cancel_registered_timer(app: Flask, name: str) -> None:
    """Cancel and remove a previously registered background timer."""

    app = _resolve_flask_app(app)
    registry = app.config.get(_BACKGROUND_TIMER_REGISTRY)
    if not registry:
        return

    timer = registry.pop(name, None)
    if timer is None:
        return

    cancel = getattr(timer, "cancel", None)
    if callable(cancel):
        try:
            cancel()
        except Exception:  # pragma: no cover - defensive guard
            logger.opt(exception=True).debug("Failed to cancel timer '%s'", name)


def _start_cancellable_timer(
    app: Flask, name: str, delay: float, function: Callable[[], None]
) -> threading.Timer:
    """Start a daemon timer and register it for later cancellation."""

    app = _resolve_flask_app(app)
    timer = threading.Timer(delay, function)
    _ensure_daemon(timer)

    registry = app.config.setdefault(_BACKGROUND_TIMER_REGISTRY, {})
    existing = registry.get(name)
    if existing is not None:
        cancel = getattr(existing, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:  # pragma: no cover - defensive guard
                logger.opt(exception=True).debug(
                    "Failed to cancel existing timer '%s'", name
                )

    registry[name] = timer
    timer.start()
    return timer


def set_service_metadata_value(app: Flask, key: str, value: str) -> None:
    """Persist a key/value pair into the ``service_metadata`` table."""

    app = _resolve_flask_app(app)
    engine = app.config.get("ENGINE")
    db_path = app.config.get("DB_PATH")

    if engine is not None:
        dialect_name = getattr(getattr(engine, "dialect", None), "name", "")
        if dialect_name == "sqlite":
            upsert_sql = "INSERT OR REPLACE INTO service_metadata (key, value) VALUES (:key, :value)"
        elif dialect_name == "mysql":
            upsert_sql = (
                "INSERT INTO service_metadata (key, value) VALUES (:key, :value) "
                "ON DUPLICATE KEY UPDATE value = VALUES(value)"
            )
        else:
            upsert_sql = (
                "INSERT INTO service_metadata (key, value) VALUES (:key, :value) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
            )
        with engine.begin() as conn:
            conn.execute(text(upsert_sql), {"key": key, "value": value})
        return

    if db_path:
        conn = sqlite3.connect(db_path)
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO service_metadata (key, value) VALUES (?, ?)",
                    (key, value),
                )
        finally:
            conn.close()


def get_service_metadata_value(app: Flask, key: str) -> str | None:
    """Return a stored ``service_metadata`` value when available."""

    app = _resolve_flask_app(app)
    engine = app.config.get("ENGINE")
    db_path = app.config.get("DB_PATH")

    if engine is not None:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT value FROM service_metadata WHERE key = :key"),
                {"key": key},
            )
            row = result.fetchone()
            if row:
                return row[0]
        return None

    if db_path:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute(
                "SELECT value FROM service_metadata WHERE key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row:
                return row[0]
        finally:
            conn.close()
    return None


def cancel_daily_decrement_timer(app: Flask) -> None:
    """Cancel any in-flight decrement timer stored on the Flask config."""

    app = _resolve_flask_app(app)
    timer = app.config.pop("_decrement_timer", None)
    if timer is None:
        return

    cancel = getattr(timer, "cancel", None)
    if callable(cancel):
        try:
            cancel()
        except Exception:  # pragma: no cover - defensive guard
            logger.opt(exception=True).debug("Failed to cancel in-flight decrement timer")


def schedule_daily_decrement(app: Flask) -> None:
    """Schedule daily decrement of temporal coordinates."""

    app = _resolve_flask_app(app)
    memoria = app.config["memoria"]

    def _cancel_existing_timer() -> None:
        cancel_daily_decrement_timer(app)

    if not hasattr(memoria, "storage_service"):
        _cancel_registered_timer(app, _DAILY_DECREMENT_TIMER)

        return

    db_path = app.config.get("DB_PATH")
    engine = app.config.get("ENGINE")
    decrement_tz = app.config.get("DECREMENT_TZ", "UTC")

    local_engine = engine
    if local_engine is None:
        if not db_path:
            return
        local_engine = create_engine(f"sqlite:///{db_path}")

    def get_last_run() -> date | None:
        try:
            with local_engine.begin() as conn:
                result = conn.execute(
                    text(
                        "SELECT value FROM service_metadata WHERE key='last_decrement_date'"
                    )
                )
                row = result.fetchone()
        except Exception:  # pragma: no cover - defensive
            return None
        if row and row[0]:
            try:
                return datetime.fromisoformat(row[0]).date()
            except ValueError:
                return None
        return None

    def set_last_run(day: date) -> None:
        dialect_name = getattr(getattr(local_engine, "dialect", None), "name", "")
        if dialect_name == "sqlite":
            upsert_sql = "INSERT OR REPLACE INTO service_metadata (key, value) VALUES (:key, :value)"
        elif dialect_name == "mysql":
            upsert_sql = (
                "INSERT INTO service_metadata (key, value) VALUES (:key, :value) "
                "ON DUPLICATE KEY UPDATE value = VALUES(value)"
            )
        else:
            upsert_sql = (
                "INSERT INTO service_metadata (key, value) VALUES (:key, :value) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
            )
        try:
            with local_engine.begin() as conn:
                conn.execute(
                    text(upsert_sql),
                    {"key": "last_decrement_date", "value": day.isoformat()},
                )
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to persist last decrement date")
            raise

    def schedule_next() -> None:
        now = datetime.now(ZoneInfo(decrement_tz))
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        delay = (next_midnight - now).total_seconds()
        _start_cancellable_timer(app, _DAILY_DECREMENT_TIMER, delay, _run)

    def _run() -> None:
        try:
            memoria.storage_service.decrement_x_coords()
            set_last_run(datetime.now(ZoneInfo(decrement_tz)).date())
            logger.info("Daily temporal coordinates decremented")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to decrement x coords: {exc}")
        finally:
            _cancel_existing_timer()
            schedule_next()

    today = datetime.now(ZoneInfo(decrement_tz)).date()
    last_run = get_last_run()
    if last_run is None or last_run < today:
        try:
            memoria.storage_service.decrement_x_coords()
            set_last_run(today)
            logger.info("Startup temporal coordinates decremented")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to decrement x coords on startup: {exc}")

    schedule_next()


def reschedule_daily_decrement(app: Flask) -> None:
    """Replace the scheduled decrement timer with one bound to current state."""

    app = _resolve_flask_app(app)
    cancel_daily_decrement_timer(app)
    _cancel_registered_timer(app, _DAILY_DECREMENT_TIMER)
    schedule_daily_decrement(app)


def restart_daily_decrement_schedule(app: Flask) -> None:
    """Restart the daily decrement scheduler with the current app bindings."""

    reschedule_daily_decrement(app)


def _store_roster_status(app: Flask, payload: dict[str, object]) -> None:
    app = _resolve_flask_app(app)
    app.config[_ROSTER_CONFIG_KEY] = payload
    try:
        set_service_metadata_value(app, _ROSTER_METADATA_KEY, json.dumps(payload))
    except Exception:  # pragma: no cover - defensive persistence guard
        logger.opt(exception=True).warning("Failed to persist roster verification status")


def _store_rotation_status(app: Flask, payload: dict[str, object]) -> None:
    app = _resolve_flask_app(app)
    app.config[_ROTATION_CONFIG_KEY] = payload
    try:
        set_service_metadata_value(app, _ROTATION_METADATA_KEY, json.dumps(payload))
    except Exception:  # pragma: no cover - defensive persistence guard
        logger.opt(exception=True).warning("Failed to persist roster rotation status")


def _store_override_status(app: Flask, payload: dict[str, object]) -> None:
    app = _resolve_flask_app(app)
    app.config[_OVERRIDE_CONFIG_KEY] = payload
    try:
        set_service_metadata_value(app, _OVERRIDE_METADATA_KEY, json.dumps(payload))
    except Exception:  # pragma: no cover - defensive persistence guard
        logger.opt(exception=True).warning("Failed to persist override expiry status")


def schedule_roster_verification(app: Flask, *, cadence_minutes: int = 60) -> None:
    """Schedule periodic verification for escalation rosters."""

    app = _resolve_flask_app(app)
    cadence_minutes = max(int(cadence_minutes or 60), 5)
    cadence_seconds = cadence_minutes * 60

    def _load_contacts() -> list:
        manager = app.config.get("config_manager")
        if manager is None:
            return []
        get_settings = getattr(manager, "get_settings", None)
        if not callable(get_settings):
            return []
        try:
            settings = get_settings()
        except Exception:  # pragma: no cover - defensive guard
            logger.opt(exception=True).warning("Unable to load settings for roster verification")
            return []
        definitions = getattr(settings, "policy", None)
        if definitions is None:
            return []
        contacts = getattr(definitions, "escalation_contacts", []) or []
        return list(contacts)

    def _run_once() -> None:
        contacts = _load_contacts()
        if not contacts:
            logger.debug("Roster verification skipped – no escalation contacts configured")
            return
        try:
            payload = verify_escalation_contacts(contacts, cadence_minutes=cadence_minutes)
        except Exception:  # pragma: no cover - defensive guard
            logger.opt(exception=True).warning("Roster verification failed")
            return
        _store_roster_status(app, payload)

    def _tick() -> None:
        try:
            _run_once()
        finally:
            _start_cancellable_timer(app, _ROSTER_VERIFICATION_TIMER, cadence_seconds, _tick)

    _cancel_registered_timer(app, _ROSTER_VERIFICATION_TIMER)
    try:
        _run_once()
    finally:
        _start_cancellable_timer(app, _ROSTER_VERIFICATION_TIMER, cadence_seconds, _tick)


def schedule_roster_rotation(app: Flask, *, cadence_minutes: int = 60) -> None:
    """Schedule periodic rotation metadata updates for escalation contacts."""

    app = _resolve_flask_app(app)
    cadence_minutes = max(int(cadence_minutes or 60), 5)
    cadence_seconds = cadence_minutes * 60

    def _load_contacts() -> list:
        manager = app.config.get("config_manager")
        if manager is None:
            return []
        get_settings = getattr(manager, "get_settings", None)
        if not callable(get_settings):
            return []
        try:
            settings = get_settings()
        except Exception:  # pragma: no cover - defensive guard
            logger.opt(exception=True).warning("Unable to load settings for roster rotation")
            return []
        definitions = getattr(settings, "policy", None)
        if definitions is None:
            return []
        contacts = getattr(definitions, "escalation_contacts", []) or []
        return list(contacts)

    def _persist_contacts(contacts: list) -> None:
        manager = app.config.get("config_manager")
        if manager is None:
            return
        payload = [contact.dict() for contact in contacts]
        try:
            manager.update_setting("policy.escalation_contacts", payload)
        except Exception:  # pragma: no cover - defensive guard
            logger.opt(exception=True).warning("Unable to persist rotated escalation contacts")

    def _run_once() -> None:
        contacts = _load_contacts()
        reference = datetime.now(timezone.utc)
        rotation_records: list[dict[str, object]] = []
        updated_contacts: list = []
        metadata_updates = 0
        overdue_contacts = 0

        for contact in contacts:
            updated, record = apply_rotation_metadata(contact, reference_time=reference)
            updated_contacts.append(updated)
            if record.get("metadata_updated"):
                metadata_updates += 1
            if record.get("overdue_windows"):
                overdue_contacts += 1
            rotation_records.append(record)

        if metadata_updates:
            _persist_contacts(updated_contacts)

        payload = {
            "generated_at": _isoformat_utc(reference),
            "next_check_at": _isoformat_utc(reference + timedelta(seconds=cadence_seconds)),
            "cadence_minutes": cadence_minutes,
            "summary": {
                "total_contacts": len(contacts),
                "metadata_updates": metadata_updates,
                "overdue_contacts": overdue_contacts,
            },
            "contacts": rotation_records,
        }
        _store_rotation_status(app, payload)

    def _tick() -> None:
        try:
            _run_once()
        finally:
            _start_cancellable_timer(app, _ROSTER_ROTATION_TIMER, cadence_seconds, _tick)

    _cancel_registered_timer(app, _ROSTER_ROTATION_TIMER)
    try:
        _run_once()
    finally:
        _start_cancellable_timer(app, _ROSTER_ROTATION_TIMER, cadence_seconds, _tick)


def schedule_override_expiry(app: Flask, *, cadence_minutes: int = 60) -> None:
    """Schedule automatic expiry checks for policy overrides."""

    app = _resolve_flask_app(app)
    cadence_minutes = max(int(cadence_minutes or 60), 5)
    cadence_seconds = cadence_minutes * 60

    def _load_overrides() -> list:
        manager = app.config.get("config_manager")
        if manager is None:
            return []
        get_settings = getattr(manager, "get_settings", None)
        if not callable(get_settings):
            return []
        try:
            settings = get_settings()
        except Exception:  # pragma: no cover - defensive guard
            logger.opt(exception=True).warning("Unable to load settings for override expiry")
            return []
        definitions = getattr(settings, "policy", None)
        if definitions is None:
            return []
        overrides = getattr(definitions, "overrides", []) or []
        return list(overrides)

    def _persist_overrides(overrides: list) -> None:
        manager = app.config.get("config_manager")
        if manager is None:
            return
        payload: list[dict[str, object]] = []
        for override in overrides:
            data = override.dict()
            action = data.get("action")
            stage = data.get("stage")
            expires_at = data.get("expires_at")
            if hasattr(action, "value"):
                data["action"] = action.value
            if hasattr(stage, "value"):
                data["stage"] = stage.value
            if isinstance(expires_at, datetime):
                data["expires_at"] = _isoformat_utc(expires_at)
            payload.append(data)
        try:
            manager.update_setting("policy.overrides", payload)
        except Exception:  # pragma: no cover - defensive guard
            logger.opt(exception=True).warning("Unable to persist override expiry updates")

    def _run_once() -> None:
        overrides = _load_overrides()
        reference = datetime.now(timezone.utc)
        updated_overrides: list = []
        automation_records: list[dict[str, object]] = []
        metadata_updates = 0
        expired_count = 0

        for override in overrides:
            updated, record = apply_override_expiry(override, reference_time=reference)
            updated_overrides.append(updated)
            if record.get("metadata_updated"):
                metadata_updates += 1
            if record.get("status") == "expired":
                expired_count += 1
            automation_records.append(record)

        if metadata_updates:
            _persist_overrides(updated_overrides)

        payload = {
            "generated_at": _isoformat_utc(reference),
            "next_check_at": _isoformat_utc(reference + timedelta(seconds=cadence_seconds)),
            "cadence_minutes": cadence_minutes,
            "summary": {
                "total_overrides": len(overrides),
                "metadata_updates": metadata_updates,
                "expired_overrides": expired_count,
            },
            "overrides": automation_records,
        }
        _store_override_status(app, payload)

    def _tick() -> None:
        try:
            _run_once()
        finally:
            _start_cancellable_timer(app, _OVERRIDE_EXPIRY_TIMER, cadence_seconds, _tick)

    _cancel_registered_timer(app, _OVERRIDE_EXPIRY_TIMER)
    try:
        _run_once()
    finally:
        _start_cancellable_timer(app, _OVERRIDE_EXPIRY_TIMER, cadence_seconds, _tick)


def cancel_roster_verification_timer(app: Flask) -> None:
    """Cancel any scheduled roster verification task."""

    _cancel_registered_timer(app, _ROSTER_VERIFICATION_TIMER)


def cancel_roster_rotation_timer(app: Flask) -> None:
    """Cancel any scheduled roster rotation task."""

    _cancel_registered_timer(app, _ROSTER_ROTATION_TIMER)


def cancel_override_expiry_timer(app: Flask) -> None:
    """Cancel any scheduled override expiry task."""

    _cancel_registered_timer(app, _OVERRIDE_EXPIRY_TIMER)


__all__ = [
    "schedule_daily_decrement",
    "reschedule_daily_decrement",
    "restart_daily_decrement_schedule",
    "cancel_daily_decrement_timer",
    "schedule_roster_verification",
    "schedule_roster_rotation",
    "schedule_override_expiry",
    "cancel_roster_verification_timer",
    "cancel_roster_rotation_timer",
    "cancel_override_expiry_timer",
    "set_service_metadata_value",
    "get_service_metadata_value",
    "_start_cancellable_timer",
]
