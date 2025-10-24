from __future__ import annotations

import json
import sqlite3
from flask import Blueprint, jsonify, request, current_app
from typing import Any, Mapping
from loguru import logger
from pydantic import BaseModel, ValidationError

from memoria.schemas import (
    MemoryEntry,
    PersonalMemoryEntry,
    canonicalize_symbolic_anchors,
    validate_memory_entry,
)
from memoria.utils.exceptions import MemoriaError
from memoria.utils.pydantic_compat import model_validate
from memoria.config.settings import IngestMode

from memoria.database.models import (
    SpatialMetadata,
    LinkMemoryThread,
    LongTermMemory,
    ShortTermMemory,
)

from .spatial_utils import upsert_spatial_metadata
from .request_parsers import (
    CoordinateParsingError,
    parse_optional_coordinate,
    timestamp_from_x,
)


memory_bp = Blueprint("memory", __name__)
_MISSING = object()


class ConversationRecord(BaseModel):
    user_input: str
    ai_output: Any | None = None
    model: str | None = None
    metadata: dict[str, Any] | None = None


def _parse_positive_int(
    name: str,
    raw_value: str | None,
    *,
    default: int,
    allow_zero: bool = False,
) -> tuple[int | None, str | None]:
    """Parse a positive integer query parameter, returning an error message on failure."""

    if raw_value in (None, ""):
        return default, None

    message = (
        f"Parameter '{name}' must be a non-negative integer"
        if allow_zero
        else f"Parameter '{name}' must be a positive integer"
    )

    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None, message

    if allow_zero:
        if value < 0:
            return None, message
    else:
        if value <= 0:
            return None, message

    return value, None


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    try:
        return bool(value)
    except Exception:  # pragma: no cover - defensive
        return None


def _coerce_identifier(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _normalize_identifier_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, (list, tuple, set)):
        results: list[str] = []
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    results.append(cleaned)
        return results
    return None


def _normalize_member_mapping(value: Mapping[str, Any]) -> dict[str, Any] | None:
    user_value = value.get("user_id") or value.get("id") or value.get("user")
    user_id = _coerce_identifier(user_value)
    if not user_id:
        return None
    payload: dict[str, Any] = {"user_id": user_id}
    if "is_agent" in value:
        payload["is_agent"] = bool(value.get("is_agent", False))
    preferred_value = value.get("preferred_model")
    if preferred_value is not None:
        if isinstance(preferred_value, str):
            preferred_clean = preferred_value.strip()
        else:
            preferred_clean = str(preferred_value).strip()
        if preferred_clean:
            payload["preferred_model"] = preferred_clean
    model_value = value.get("last_edited_by_model")
    if model_value is not None:
        if isinstance(model_value, str):
            model_clean = model_value.strip()
        else:
            model_clean = str(model_value).strip()
        if model_clean:
            payload["last_edited_by_model"] = model_clean
    return payload


def _normalize_member_entries(
    value: Any,
) -> list[str | dict[str, Any]] | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, Mapping):
        payload = _normalize_member_mapping(value)
        return [payload] if payload else []
    if isinstance(value, (list, tuple, set)):
        results: list[str | dict[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    results.append(cleaned)
            elif isinstance(item, Mapping):
                payload = _normalize_member_mapping(item)
                if payload:
                    results.append(payload)
        return results
    return None


def _team_error(message: str, status: int = 400):
    return jsonify({"status": "error", "message": message}), status


def _ensure_thread_table(cur: sqlite3.Cursor) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS link_memory_threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_memory_id TEXT NOT NULL,
            target_memory_id TEXT NOT NULL,
            relation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


@memory_bp.route("/memory", methods=["POST"])
def store_memory():
    """Store a memory entry via :class:`memoria.Memoria`."""
    try:
        memoria = current_app.config["memoria"]
        db_path = current_app.config.get("DB_PATH")
        db_manager = getattr(memoria, "db_manager", None)
        namespace = getattr(memoria, "namespace", "default") or "default"

        original = dict(request.json or {})
        data = dict(original)
        promotion_weights = data.pop("promotion_weights", None)
        workspace_id = data.pop("workspace_id", None)
        team_id = data.pop("team_id", None)
        share_param = data.pop("share_with_team", None)
        target_namespace = data.pop("namespace", None)
        ingest_mode_value = data.pop("ingest_mode", None)
        edited_by_model = _coerce_identifier(data.pop("last_edited_by_model", None))
        team_id = _coerce_identifier(team_id)
        workspace_id = _coerce_identifier(workspace_id)
        if team_id is None:
            team_id = workspace_id
        if isinstance(target_namespace, str):
            target_namespace = target_namespace.strip() or None
        share_with_team = _coerce_optional_bool(share_param)
        try:
            for field in ("x_coord", "y_coord", "z_coord"):
                if field in data:
                    data[field] = parse_optional_coordinate(data[field], field)
        except CoordinateParsingError as err:
            return jsonify({"status": "error", "message": str(err)}), 400
        ts_str = data.get("timestamp")
        if ts_str in (None, ""):
            derived_ts = timestamp_from_x(data.get("x_coord"))
            if derived_ts is not None:
                data["timestamp"] = derived_ts

        if "symbolic_anchors" in data:
            data["symbolic_anchors"] = canonicalize_symbolic_anchors(
                data.get("symbolic_anchors")
            )

        requested_mode: IngestMode | None = None
        if isinstance(ingest_mode_value, IngestMode):
            requested_mode = ingest_mode_value
        elif isinstance(ingest_mode_value, str) and ingest_mode_value:
            cleaned_mode = ingest_mode_value.strip().lower()
            if cleaned_mode:
                try:
                    requested_mode = IngestMode(cleaned_mode)
                except ValueError:
                    requested_mode = None

        resolved_mode = requested_mode or getattr(
            memoria, "ingest_mode", IngestMode.STANDARD
        )

        if resolved_mode == IngestMode.PERSONAL:
            entry = model_validate(PersonalMemoryEntry, data)
            cleaned_anchors = canonicalize_symbolic_anchors(entry.symbolic_anchors)
            entry.symbolic_anchors = cleaned_anchors
            result = memoria.store_memory(
                anchor=entry.anchor,
                text=entry.text,
                tokens=entry.tokens,
                timestamp=entry.timestamp,
                x_coord=entry.x_coord,
                y=entry.y_coord,
                z=entry.z_coord,
                symbolic_anchors=cleaned_anchors,
                chat_id=entry.chat_id,
                metadata=entry.metadata,
                documents=entry.documents,
                promotion_weights=promotion_weights,
                return_status=True,
                namespace=target_namespace,
                team_id=team_id,
                share_with_team=share_with_team,
                workspace_id=workspace_id,
                ingest_mode=requested_mode or IngestMode.PERSONAL,
                last_edited_by_model=edited_by_model,
            )
            return jsonify(result)

        entry = validate_memory_entry(data)
        cleaned_anchors = canonicalize_symbolic_anchors(entry.symbolic_anchors)
        entry.symbolic_anchors = cleaned_anchors

        result = memoria.store_memory(
            anchor=entry.anchor,
            text=entry.text,
            tokens=entry.tokens,
            timestamp=entry.timestamp,
            x_coord=entry.x_coord,
            y=entry.y_coord,
            z=entry.z_coord,
            symbolic_anchors=cleaned_anchors,
            promotion_weights=promotion_weights,
            return_status=True,
            namespace=target_namespace,
            team_id=team_id,
            share_with_team=share_with_team,
            workspace_id=workspace_id,
            ingest_mode=requested_mode,
            last_edited_by_model=edited_by_model,
        )

        print(
            "[MEMORY STORED]"
            f" {result['memory_id']} | {entry.anchor} | {entry.tokens} tokens"
            f" | status={result['status']} score={result['promotion_score']:.2f}"
        )
        response_payload = {
            "status": result["status"],
            "anchor": entry.anchor,
            "memory_id": result["memory_id"],
            "short_term_id": result["short_term_id"],
            "long_term_id": result["long_term_id"],
            "promotion_score": result["promotion_score"],
            "threshold": result["threshold"],
            "promoted": result["promoted"],
            "namespace": result.get("namespace"),
        }
        if result.get("team_id"):
            response_payload["team_id"] = result["team_id"]
            response_payload.setdefault("workspace_id", result["team_id"])
        if any(
            v is not None
            for v in [
                entry.x_coord,
                entry.y_coord,
                entry.z_coord,
                entry.symbolic_anchors,
            ]
        ):
            upsert_spatial_metadata(
                memory_id=result["memory_id"],
                namespace=namespace,
                db_path=db_path,
                db_manager=db_manager,
                timestamp=entry.timestamp,
                x=entry.x_coord,
                y=entry.y_coord,
                z=entry.z_coord,
                symbolic_anchors=cleaned_anchors or [],
                ensure_timestamp=True,
            )
        return jsonify(response_payload)

    except ValidationError as e:  # pragma: no cover - validated in tests
        # ``ValidationError.errors()`` can contain values that are not
        # JSON-serializable (e.g. ``datetime`` instances). Converting via
        # ``e.json()`` and loading ensures the payload is purely composed of
        # standard JSON types.
        return jsonify({"status": "error", "message": json.loads(e.json())}), 422
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"[ERROR] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@memory_bp.route("/memory/conversation", methods=["POST"])
def record_conversation():
    try:
        memoria = current_app.config["memoria"]

        payload = ConversationRecord(**(request.json or {}))
        chat_id = memoria.record_conversation(
            user_input=payload.user_input,
            ai_output=payload.ai_output,
            model=payload.model,
            metadata=payload.metadata,
        )

        return jsonify({"status": "ok", "chat_id": chat_id})

    except ValidationError as e:  # pragma: no cover - validated in tests
        return jsonify({"status": "error", "message": json.loads(e.json())}), 422
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"[ERROR] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@memory_bp.route("/memory/teams", methods=["GET"])
def list_team_spaces():
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    include_members = bool(
        _coerce_optional_bool(request.args.get("include_members"))
    )
    try:
        teams = memoria.list_team_spaces(include_members=include_members)
    except MemoriaError as exc:
        return _team_error(str(exc))
    return jsonify({"status": "ok", "teams": teams})


@memory_bp.route("/memory/teams", methods=["POST"])
def create_or_update_team_space():
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    payload = request.get_json(silent=True) or {}
    team_id = payload.get("team_id")
    if not isinstance(team_id, str) or not team_id.strip():
        return _team_error("team_id is required", 400)

    include_members = bool(_coerce_optional_bool(payload.get("include_members")))
    members = _normalize_member_entries(payload.get("members"))
    admins = _normalize_member_entries(payload.get("admins"))
    share_by_default = _coerce_optional_bool(payload.get("share_by_default"))
    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return _team_error("metadata must be an object", 400)

    try:
        team = memoria.register_team_space(
            team_id,
            namespace=payload.get("namespace"),
            display_name=payload.get("display_name"),
            members=members,
            admins=admins,
            share_by_default=share_by_default,
            metadata=metadata,
            include_members=include_members,
        )
    except MemoriaError as exc:
        return _team_error(str(exc))

    return jsonify({"status": "ok", "team": team})


@memory_bp.route("/memory/teams/<team_id>", methods=["GET"])
def get_team_space(team_id: str):
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    include_members = bool(
        _coerce_optional_bool(request.args.get("include_members"))
    )
    try:
        team = memoria.get_team_space(team_id, include_members=include_members)
    except MemoriaError as exc:
        return _team_error(str(exc))
    return jsonify({"status": "ok", "team": team})


@memory_bp.route("/memory/teams/<team_id>/members", methods=["POST"])
def add_team_members(team_id: str):
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    payload = request.get_json(silent=True) or {}
    members = _normalize_member_entries(payload.get("members")) or []
    if not members:
        return _team_error("members must include at least one identifier", 400)

    role = (payload.get("role") or "member").lower()
    as_admin = bool(_coerce_optional_bool(payload.get("as_admin")))
    if role == "admin":
        as_admin = True

    include_members = bool(_coerce_optional_bool(payload.get("include_members")))

    try:
        team = memoria.add_team_members(
            team_id,
            members,
            as_admin=as_admin,
            include_members=include_members,
        )
    except MemoriaError as exc:
        return _team_error(str(exc))

    return jsonify({"status": "ok", "team": team})


@memory_bp.route("/memory/teams/<team_id>/members", methods=["PUT"])
def set_team_members(team_id: str):
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    payload = request.get_json(silent=True) or {}
    include_members = bool(_coerce_optional_bool(payload.get("include_members")))

    members_raw = payload.get("members", _MISSING)
    admins_raw = payload.get("admins", _MISSING)
    members = (
        _normalize_member_entries(members_raw)
        if members_raw is not _MISSING
        else None
    )
    admins = (
        _normalize_member_entries(admins_raw)
        if admins_raw is not _MISSING
        else None
    )

    try:
        team = memoria.set_team_members(
            team_id,
            members=members,
            admins=admins,
            include_members=include_members,
        )
    except MemoriaError as exc:
        return _team_error(str(exc))

    return jsonify({"status": "ok", "team": team})


@memory_bp.route(
    "/memory/teams/<team_id>/members/<user_id>", methods=["DELETE"]
)
def remove_team_member(team_id: str, user_id: str):
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    include_members = bool(
        _coerce_optional_bool(request.args.get("include_members"))
    )
    try:
        team = memoria.remove_team_member(
            team_id, user_id, include_members=include_members
        )
    except MemoriaError as exc:
        return _team_error(str(exc))

    return jsonify({"status": "ok", "team": team})


@memory_bp.route("/memory/teams/<team_id>/activate", methods=["POST"])
def activate_team(team_id: str):
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    payload = request.get_json(silent=True) or {}
    enforce_membership = _coerce_optional_bool(payload.get("enforce_membership"))
    try:
        memoria.set_active_team(
            team_id,
            enforce_membership=False if enforce_membership is False else True,
        )
    except MemoriaError as exc:
        return _team_error(str(exc))

    return jsonify({"status": "ok", "active_team": memoria.get_active_team()})


@memory_bp.route("/memory/teams/active", methods=["DELETE"])
def clear_active_team():
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    memoria.clear_active_team()
    return jsonify({"status": "ok", "active_team": None})


@memory_bp.route("/memory/teams/active", methods=["GET"])
def get_active_team():
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    return jsonify({"status": "ok", "active_team": memoria.get_active_team()})


@memory_bp.route("/memory/teams/namespaces", methods=["GET"])
def list_accessible_namespaces():
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    try:
        contexts = memoria.get_accessible_contexts()
        namespaces = sorted(
            {ctx["namespace"] for ctx in contexts if ctx.get("namespace")}
        )
    except MemoriaError as exc:
        return _team_error(str(exc))
    return jsonify({"status": "ok", "namespaces": namespaces, "contexts": contexts})


@memory_bp.route("/memory/workspaces", methods=["GET"])
def list_workspaces():
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    include_members = bool(
        _coerce_optional_bool(request.args.get("include_members"))
    )
    try:
        workspaces = memoria.list_workspaces(include_members=include_members)
    except MemoriaError as exc:
        return _team_error(str(exc))
    return jsonify({"status": "ok", "workspaces": workspaces})


@memory_bp.route("/memory/workspaces", methods=["POST"])
def create_or_update_workspace():
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    payload = request.get_json(silent=True) or {}
    workspace_id = payload.get("workspace_id") or payload.get("team_id")
    workspace_id = _coerce_identifier(workspace_id)
    if not workspace_id:
        return _team_error("workspace_id is required", 400)

    include_members = bool(_coerce_optional_bool(payload.get("include_members")))
    members = _normalize_member_entries(payload.get("members"))
    admins = _normalize_member_entries(payload.get("admins"))
    share_by_default = _coerce_optional_bool(payload.get("share_by_default"))
    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return _team_error("metadata must be an object", 400)

    try:
        workspace = memoria.register_workspace(
            workspace_id,
            namespace=_coerce_identifier(payload.get("namespace")),
            display_name=_coerce_identifier(payload.get("display_name")),
            members=members,
            admins=admins,
            share_by_default=share_by_default,
            metadata=metadata,
            include_members=include_members,
        )
    except MemoriaError as exc:
        return _team_error(str(exc))

    return jsonify({"status": "ok", "workspace": workspace})


@memory_bp.route("/memory/workspaces/<workspace_id>", methods=["GET"])
def get_workspace(workspace_id: str):
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    include_members = bool(
        _coerce_optional_bool(request.args.get("include_members"))
    )
    try:
        workspace = memoria.get_workspace(
            workspace_id, include_members=include_members
        )
    except MemoriaError as exc:
        return _team_error(str(exc))
    return jsonify({"status": "ok", "workspace": workspace})


@memory_bp.route("/memory/workspaces/<workspace_id>/members", methods=["POST"])
def add_workspace_members(workspace_id: str):
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    payload = request.get_json(silent=True) or {}
    include_members = bool(_coerce_optional_bool(payload.get("include_members")))
    as_admin = bool(_coerce_optional_bool(payload.get("as_admin")))
    members = _normalize_member_entries(payload.get("members"))
    if not members:
        return _team_error("members must include at least one identifier", 400)

    try:
        workspace = memoria.add_workspace_members(
            workspace_id,
            members,
            as_admin=as_admin,
            include_members=include_members,
        )
    except MemoriaError as exc:
        return _team_error(str(exc))
    return jsonify({"status": "ok", "workspace": workspace})


@memory_bp.route("/memory/workspaces/<workspace_id>/members", methods=["PUT"])
def set_workspace_members(workspace_id: str):
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    payload = request.get_json(silent=True) or {}
    include_members = bool(_coerce_optional_bool(payload.get("include_members")))
    members = _normalize_member_entries(payload.get("members"))
    admins = _normalize_member_entries(payload.get("admins"))

    try:
        workspace = memoria.set_workspace_members(
            workspace_id,
            members=members,
            admins=admins,
            include_members=include_members,
        )
    except MemoriaError as exc:
        return _team_error(str(exc))
    return jsonify({"status": "ok", "workspace": workspace})


@memory_bp.route(
    "/memory/workspaces/<workspace_id>/members/<user_id>", methods=["DELETE"]
)
def remove_workspace_member(workspace_id: str, user_id: str):
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    include_members = bool(
        _coerce_optional_bool(request.args.get("include_members"))
    )
    try:
        workspace = memoria.remove_workspace_member(
            workspace_id,
            user_id,
            include_members=include_members,
        )
    except MemoriaError as exc:
        return _team_error(str(exc))
    return jsonify({"status": "ok", "workspace": workspace})


@memory_bp.route("/memory/workspaces/<workspace_id>/activate", methods=["POST"])
def activate_workspace(workspace_id: str):
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    payload = request.get_json(silent=True) or {}
    allow_guest = bool(_coerce_optional_bool(payload.get("allow_guest")))
    try:
        memoria.set_active_workspace(
            workspace_id,
            enforce_membership=not allow_guest,
        )
    except MemoriaError as exc:
        return _team_error(str(exc))
    return jsonify({"status": "ok", "active_workspace": memoria.get_active_workspace()})


@memory_bp.route("/memory/workspaces/active", methods=["DELETE"])
def clear_active_workspace():
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    memoria.clear_active_workspace()
    return jsonify({"status": "ok", "active_workspace": None})


@memory_bp.route("/memory/workspaces/active", methods=["GET"])
def get_active_workspace():
    memoria = current_app.config.get("memoria")
    if memoria is None:
        return _team_error("Memoria not configured", 503)

    return jsonify({"status": "ok", "active_workspace": memoria.get_active_workspace()})


@memory_bp.route("/memory/ingestion", methods=["POST"])
def trigger_daily_ingestion():
    try:
        memoria = current_app.config["memoria"]
    except KeyError:  # pragma: no cover - defensive
        return jsonify({"status": "error", "message": "Memoria not configured"}), 500

    payload = dict(request.json or {})
    promotion_weights = payload.get("promotion_weights")

    try:
        results = memoria.run_daily_ingestion(promotion_weights=promotion_weights)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"Daily ingestion trigger failed: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500

    promoted = [item for item in results if item.get("promoted")]
    return jsonify(
        {
            "status": "ok",
            "promoted_count": len(promoted),
            "results": results,
        }
    )


@memory_bp.route("/memory/ingestion", methods=["GET"])
def get_daily_ingestion_report():
    try:
        memoria = current_app.config["memoria"]
    except KeyError:  # pragma: no cover - defensive
        return jsonify({"status": "error", "message": "Memoria not configured"}), 500

    report = memoria.get_ingestion_report()
    promoted = [item for item in report if item.get("promoted")]
    return jsonify(
        {
            "status": "ok",
            "promoted_count": len(promoted),
            "results": report,
        }
    )


@memory_bp.route("/memory/<memory_id>", methods=["PUT"])
def update_memory(memory_id: str):
    """Update a memory entry and synchronize spatial metadata."""

    try:
        memoria = current_app.config["memoria"]
        db_path = current_app.config.get("DB_PATH")
        db_manager = getattr(memoria, "db_manager", None)
        namespace = getattr(memoria, "namespace", "default") or "default"

        original = dict(request.json or {})
        data = dict(original)
        try:
            for field in ("x_coord", "y_coord", "z_coord"):
                if field in data:
                    data[field] = parse_optional_coordinate(data[field], field)
        except CoordinateParsingError as err:
            return jsonify({"status": "error", "message": str(err)}), 400

        ts_str = data.get("timestamp")
        if ts_str in (None, ""):
            derived_ts = timestamp_from_x(data.get("x_coord"))
            if derived_ts is not None:
                data["timestamp"] = derived_ts

        entry = validate_memory_entry(data)
        entry.symbolic_anchors = canonicalize_symbolic_anchors(entry.symbolic_anchors)

        updates: dict[str, Any] = {"anchor": entry.anchor}
        if "text" in original:
            updates["text"] = entry.text
        if "tokens" in original:
            updates["tokens"] = entry.tokens
        if "timestamp" in original or "x_coord" in original:
            updates["timestamp"] = entry.timestamp
            updates["x_coord"] = entry.x_coord
        if "y_coord" in original:
            updates["y_coord"] = entry.y_coord
        if "z_coord" in original:
            updates["z_coord"] = entry.z_coord
        if "symbolic_anchors" in original:
            updates["symbolic_anchors"] = entry.symbolic_anchors

        if not memoria.update_memory(memory_id, updates):
            return jsonify({"status": "error", "message": "Memory not found"}), 404

        spatial_keys = {"timestamp", "x_coord", "y_coord", "z_coord", "symbolic_anchors"}
        if any(key in updates for key in spatial_keys):
            resolved_timestamp = None
            resolved_x = None
            resolved_y = None
            resolved_z = None
            resolved_anchors = None

            if db_manager and getattr(db_manager, "SessionLocal", None):
                with db_manager.SessionLocal() as session:
                    source = (
                        session.query(LongTermMemory)
                        .filter_by(memory_id=memory_id, namespace=namespace)
                        .one_or_none()
                    )
                    if source is None and getattr(db_manager, "enable_short_term", False):
                        source = (
                            session.query(ShortTermMemory)
                            .filter_by(memory_id=memory_id, namespace=namespace)
                            .one_or_none()
                        )
                    if source is not None:
                        resolved_timestamp = getattr(source, "timestamp", None)
                        resolved_x = getattr(source, "x_coord", None)
                        resolved_y = getattr(source, "y_coord", None)
                        resolved_z = getattr(source, "z_coord", None)
                        resolved_anchors = getattr(source, "symbolic_anchors", None)

            if "timestamp" in updates:
                resolved_timestamp = updates.get("timestamp")
            if "x_coord" in updates:
                resolved_x = updates.get("x_coord")
            if "y_coord" in updates:
                resolved_y = updates.get("y_coord")
            if "z_coord" in updates:
                resolved_z = updates.get("z_coord")
            if "symbolic_anchors" in updates:
                resolved_anchors = updates.get("symbolic_anchors")

            anchors_value = canonicalize_symbolic_anchors(resolved_anchors) or []

            upsert_spatial_metadata(
                memory_id=memory_id,
                namespace=namespace,
                db_path=db_path,
                db_manager=db_manager,
                timestamp=resolved_timestamp,
                x=resolved_x,
                y=resolved_y,
                z=resolved_z,
                symbolic_anchors=anchors_value,
                ensure_timestamp=bool(db_manager and getattr(db_manager, "SessionLocal", None)),
                prefer_db_manager=bool(
                    not db_path and db_manager and getattr(db_manager, "SessionLocal", None)
                ),
            )

        return jsonify({"status": "updated", "memory_id": memory_id})

    except ValidationError as e:  # pragma: no cover - validated in tests
        return jsonify({"status": "error", "message": json.loads(e.json())}), 422
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"[ERROR] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@memory_bp.route("/memory/<memory_id>", methods=["DELETE"])
def delete_memory(memory_id: str):
    """Delete a memory and its spatial metadata."""
    try:
        memoria = current_app.config["memoria"]
        db_path = current_app.config.get("DB_PATH")
        db_manager = getattr(memoria, "db_manager", None)
        success = memoria.delete_memory(memory_id)

        if db_path:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM spatial_metadata WHERE memory_id = ?", (memory_id,)
            )
            conn.commit()
            conn.close()
        elif db_manager and getattr(db_manager, "SessionLocal", None):
            with db_manager.SessionLocal() as session:
                session.query(SpatialMetadata).filter_by(memory_id=memory_id).delete()
                session.commit()

        if success:
            return jsonify({"status": "deleted", "memory_id": memory_id})
        return jsonify({"status": "error", "message": "Memory not found"}), 404
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"status": "error", "message": str(e)}), 500


@memory_bp.route("/memory/recent", methods=["GET"])
def get_recent_memories():
    """Return recent memories with optional pagination."""
    try:
        memoria = current_app.config["memoria"]
        limit_raw = request.args.get("limit")
        limit, error = _parse_positive_int("limit", limit_raw, default=10)
        if error:
            return jsonify({"status": "error", "message": error}), 400

        offset_raw = request.args.get("offset")
        offset, error = _parse_positive_int(
            "offset", offset_raw, default=0, allow_zero=True
        )
        if error:
            return jsonify({"status": "error", "message": error}), 400

        raw_memories = memoria.search_memories(query="", limit=limit + offset)
        if isinstance(raw_memories, dict):
            raw_list = raw_memories.get("results", [])
        else:
            raw_list = raw_memories
        memories = raw_list[offset : offset + limit]
        response = {"memories": memories}
        if isinstance(raw_memories, dict):
            if raw_memories.get("hint"):
                response["hint"] = raw_memories["hint"]
            if raw_memories.get("error"):
                response["error"] = raw_memories["error"]
        return jsonify(response)
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"status": "error", "message": str(e)}), 500


@memory_bp.route("/memory/dashboard", methods=["GET"])
def get_memory_dashboard():
    """Return aggregated memory counts for a bird's-eye dashboard."""
    try:
        memoria = current_app.config["memoria"]
        limit_raw = request.args.get("limit")
        limit, error = _parse_positive_int("limit", limit_raw, default=100)
        if error:
            return jsonify({"status": "error", "message": error}), 400
        category = request.args.get("category")
        category_filter = category.split(",") if category else None
        memory_types = request.args.get("memory_types")
        memory_types_list = memory_types.split(",") if memory_types else None
        summary = memoria.get_memory_dashboard(
            limit=limit,
            category_filter=category_filter,
            memory_types=memory_types_list,
        )
        return jsonify(summary)
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"status": "error", "message": str(e)}), 500


@memory_bp.route("/memory/thread", methods=["POST"])
def ingest_memory_thread():
    """Ingest a conversational thread payload and return promotion results."""

    try:
        memoria = current_app.config["memoria"]
        payload = dict(request.json or {})
        promotion_weights = payload.pop("promotion_weights", None)
        result = memoria.ingest_thread(
            payload,
            promotion_weights=promotion_weights,
        )
        response = {"status": "processed"}
        response.update(result)
        return jsonify(response)
    except ValidationError as exc:
        return jsonify({"status": "error", "message": json.loads(exc.json())}), 422
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"Thread ingestion failed: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


@memory_bp.route("/memory/<memory_id>/threads", methods=["GET"])
def fetch_memory_threads(memory_id: str):
    """Return thread memberships for a memory identifier."""

    try:
        memoria = current_app.config["memoria"]
        threads = memoria.get_threads_for_memory(memory_id)
        return jsonify({"threads": threads})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"Failed to fetch threads for {memory_id}: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


@memory_bp.route("/memory/<memory_id>/threads/traverse", methods=["GET"])
def traverse_memory_threads(memory_id: str):
    """Traverse threads starting from a memory up to a given depth."""
    try:
        depth = int(request.args.get("depth", 1))
        db_path = current_app.config.get("DB_PATH")
        memoria = current_app.config["memoria"]
        db_manager = getattr(memoria, "db_manager", None)
        visited = {memory_id}
        frontier = [memory_id]
        edges: list[dict[str, str]] = []
        if db_path:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            _ensure_thread_table(cur)
            for _ in range(depth):
                next_frontier = []
                for mid in frontier:
                    cur.execute(
                        """
                        SELECT source_memory_id, target_memory_id, relation
                        FROM link_memory_threads
                        WHERE source_memory_id = ? OR target_memory_id = ?
                        """,
                        (mid, mid),
                    )
                    for src, tgt, rel in cur.fetchall():
                        edges.append({"source": src, "target": tgt, "relation": rel})
                        other = tgt if src == mid else src
                        if other not in visited:
                            visited.add(other)
                            next_frontier.append(other)
                frontier = next_frontier
            conn.close()
        elif db_manager and getattr(db_manager, "SessionLocal", None):
            engine = getattr(db_manager, "engine", None)
            if engine:
                LinkMemoryThread.__table__.create(engine, checkfirst=True)
            with db_manager.SessionLocal() as session:
                for _ in range(depth):
                    next_frontier = []
                    for mid in frontier:
                        rows = session.query(LinkMemoryThread).filter(
                            (LinkMemoryThread.source_memory_id == mid)
                            | (LinkMemoryThread.target_memory_id == mid)
                        ).all()
                        for r in rows:
                            edges.append(
                                {
                                    "source": r.source_memory_id,
                                    "target": r.target_memory_id,
                                    "relation": r.relation,
                                }
                            )
                            other = (
                                r.target_memory_id
                                if r.source_memory_id == mid
                                else r.source_memory_id
                            )
                            if other not in visited:
                                visited.add(other)
                                next_frontier.append(other)
                    frontier = next_frontier
        return jsonify({"nodes": list(visited), "links": edges})
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"status": "error", "message": str(e)}), 500
