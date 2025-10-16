from __future__ import annotations

import json
from datetime import datetime

from typing import Any

from flask import Blueprint, jsonify, request, current_app

from memoria.utils.exceptions import MemoriaError

from .memory_routes import _coerce_optional_bool, _coerce_identifier

search_bp = Blueprint("search", __name__)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str | None, default: int = 10) -> int:
    """Parse an integer from a string, returning ``default`` on failure."""
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed <= 0:
            return default
        return parsed
    except ValueError:
        return default


def _parse_anchor_params(raw_params: list[str]) -> list[str]:
    """Normalize anchor query parameters into a flat list of strings."""

    anchors: list[str] = []
    for raw in raw_params:
        if not raw:
            continue
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid anchor format") from exc
            if not isinstance(parsed, list):
                raise ValueError("Anchors must be an array")
            anchors.extend(
                trimmed for trimmed in (str(a).strip() for a in parsed) if trimmed
            )
        elif "," in raw:
            anchors.extend(part.strip() for part in raw.split(",") if part.strip())
        else:
            anchors.append(raw)
    return anchors


@search_bp.route("/memory/anchor", methods=["GET"])
def get_memory_by_anchor():
    """Retrieve memories tagged with one or more symbolic anchors."""
    try:
        memoria = current_app.config["memoria"]
        try:
            anchors = _parse_anchor_params(request.args.getlist("anchor"))
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        namespace = request.args.get("namespace")
        team_id_raw = request.args.get("team_id")
        workspace_raw = request.args.get("workspace_id")
        team_id = _coerce_identifier(team_id_raw)
        workspace_id = _coerce_identifier(workspace_raw)
        if team_id is None:
            team_id = workspace_id
        share_with_team = _coerce_optional_bool(request.args.get("share_with_team"))
        try:
            result = memoria.retrieve_memories_by_anchor(
                anchors,
                namespace=namespace,
                team_id=team_id,
                share_with_team=share_with_team,
            )
        except MemoriaError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 403
        return jsonify(result)
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"status": "error", "message": str(e)}), 500


@search_bp.route("/memory/search", methods=["GET"])
def search_memory():
    """Search memories with optional filters."""
    try:
        memoria = current_app.config["memoria"]

        query = request.args.get("query") or request.args.get("q", "")
        limit = _parse_int(request.args.get("limit"), 10)

        keywords = request.args.getlist("keyword") or None
        categories = request.args.getlist("category") or None
        anchors = request.args.getlist("anchor") or None

        namespace = request.args.get("namespace")
        team_id_raw = request.args.get("team_id")
        workspace_raw = request.args.get("workspace_id")
        team_id = _coerce_identifier(team_id_raw)
        workspace_id = _coerce_identifier(workspace_raw)
        if team_id is None:
            team_id = workspace_id
        share_with_team = _coerce_optional_bool(request.args.get("share_with_team"))

        start_ts = _parse_datetime(request.args.get("start_timestamp"))
        end_ts = _parse_datetime(request.args.get("end_timestamp"))
        min_importance = _parse_float(request.args.get("min_importance"))

        x = _parse_float(request.args.get("x"))
        y = _parse_float(request.args.get("y"))
        z = _parse_float(request.args.get("z"))
        max_distance = _parse_float(request.args.get("max_distance"))

        search_kwargs: dict[str, Any] = {
            "keywords": keywords,
            "category_filter": categories,
        }
        if anchors:
            search_kwargs["anchors"] = anchors
        if start_ts:
            search_kwargs["start_timestamp"] = start_ts
        if end_ts:
            search_kwargs["end_timestamp"] = end_ts
        if min_importance is not None:
            search_kwargs["min_importance"] = min_importance
        if x is not None:
            search_kwargs["x"] = x
        if y is not None:
            search_kwargs["y"] = y
        if z is not None:
            search_kwargs["z"] = z
        if max_distance is not None:
            search_kwargs["max_distance"] = max_distance

        try:
            results = memoria.search_memories(
                query=query,
                limit=limit,
                namespace=namespace,
                team_id=team_id,
                share_with_team=share_with_team,
                **search_kwargs,
            )
        except MemoriaError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 403
        response_results = results if isinstance(results, list) else results.get("results", [])

        applied_filters: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "keywords": keywords,
            "category": categories,
            "anchors": anchors,
            "start_timestamp": start_ts.isoformat() if start_ts else None,
            "end_timestamp": end_ts.isoformat() if end_ts else None,
            "min_importance": min_importance,
            "x": x,
            "y": y,
            "z": z,
            "max_distance": max_distance,
        }
        applied_filters = {k: v for k, v in applied_filters.items() if v is not None}

        response: dict[str, Any] = {"memories": response_results, "applied_filters": applied_filters}
        if isinstance(results, dict):
            if results.get("hint"):
                response["hint"] = results["hint"]
            if results.get("error"):
                response["error"] = results["error"]
        return jsonify(response)
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"status": "error", "message": str(e)}), 500


@search_bp.route("/memory/spatial", methods=["GET"])
def get_nearby_memories():
    """Retrieve memories near a point in conceptual space."""
    try:
        memoria = current_app.config["memoria"]
        mode = (request.args.get("mode") or "3d").lower()
        if mode not in {"3d", "2d"}:
            return (
                jsonify({"status": "error", "message": "mode must be '3d' or '2d'"}),
                400,
            )

        limit = _parse_int(request.args.get("limit"), 10)

        def _parse_param(
            name: str, *, required: bool = False, default: float | None = None
        ):
            raw = request.args.get(name)
            if raw is None or raw == "":
                if required:
                    return None, f"Parameter '{name}' is required"
                return default, None
            parsed = _parse_float(raw)
            if parsed is None:
                return None, f"Parameter '{name}' must be a valid float"
            return parsed, None

        x = None
        if mode == "3d":
            x, error = _parse_param("x", required=True)
            if error:
                return jsonify({"status": "error", "message": error}), 400
        elif request.args.get("x") not in (None, ""):
            x, error = _parse_param("x")
            if error:
                return jsonify({"status": "error", "message": error}), 400

        y, error = _parse_param("y", required=True)
        if error:
            return jsonify({"status": "error", "message": error}), 400

        z = None
        if mode == "3d":
            z, error = _parse_param("z", required=True)
            if error:
                return jsonify({"status": "error", "message": error}), 400
        elif request.args.get("z") not in (None, ""):
            z, error = _parse_param("z")
            if error:
                return jsonify({"status": "error", "message": error}), 400

        max_distance, error = _parse_param("max_distance", default=5.0)
        if error:
            return jsonify({"status": "error", "message": error}), 400

        try:
            anchors = _parse_anchor_params(request.args.getlist("anchor"))
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        anchor = anchors or None
        namespace = request.args.get("namespace")
        team_id_raw = request.args.get("team_id")
        workspace_raw = request.args.get("workspace_id")
        team_id = _coerce_identifier(team_id_raw)
        workspace_id = _coerce_identifier(workspace_raw)
        if team_id is None:
            team_id = workspace_id
        share_with_team = _coerce_optional_bool(request.args.get("share_with_team"))
        if mode == "2d":
            try:
                results = memoria.retrieve_memories_near_2d(
                    y=y,
                    z=z if z is not None else 0.0,
                    max_distance=max_distance,
                    anchor=anchor,
                    namespace=namespace,
                    team_id=team_id,
                    share_with_team=share_with_team,
                    limit=limit,
                )
            except MemoriaError as exc:
                return jsonify({"status": "error", "message": str(exc)}), 403
        else:
            retrieve_kwargs = {
                "y": y,
                "z": z if z is not None else 0.0,
                "max_distance": max_distance,
                "anchor": anchor,
                "limit": limit,
            }
            if x is not None:
                retrieve_kwargs["x"] = x
            retrieve_kwargs["namespace"] = namespace
            retrieve_kwargs["team_id"] = team_id
            retrieve_kwargs["share_with_team"] = share_with_team
            try:
                results = memoria.retrieve_memories_near(**retrieve_kwargs)
            except MemoriaError as exc:
                return jsonify({"status": "error", "message": str(exc)}), 403

        return jsonify(results)
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"status": "error", "message": str(e)}), 500


@search_bp.route("/memory/time-range", methods=["GET"])
def get_memories_by_time_range():
    """Retrieve memories within a temporal window."""
    try:
        memoria = current_app.config["memoria"]
        start_ts_raw = request.args.get("start_timestamp")
        end_ts_raw = request.args.get("end_timestamp")
        start_x_raw = request.args.get("start_x")
        end_x_raw = request.args.get("end_x")

        start_dt = _parse_datetime(start_ts_raw)
        if start_ts_raw not in (None, "") and start_dt is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Parameter 'start_timestamp' must be a valid ISO 8601 timestamp",
                    }
                ),
                400,
            )

        end_dt = _parse_datetime(end_ts_raw)
        if end_ts_raw not in (None, "") and end_dt is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Parameter 'end_timestamp' must be a valid ISO 8601 timestamp",
                    }
                ),
                400,
            )

        start_x_val = _parse_float(start_x_raw)
        if start_x_raw not in (None, "") and start_x_val is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Parameter 'start_x' must be a valid float",
                    }
                ),
                400,
            )

        end_x_val = _parse_float(end_x_raw)
        if end_x_raw not in (None, "") and end_x_val is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Parameter 'end_x' must be a valid float",
                    }
                ),
                400,
            )
        namespace = request.args.get("namespace")
        team_id_raw = request.args.get("team_id")
        workspace_raw = request.args.get("workspace_id")
        team_id = _coerce_identifier(team_id_raw)
        workspace_id = _coerce_identifier(workspace_raw)
        if team_id is None:
            team_id = workspace_id
        share_with_team = _coerce_optional_bool(request.args.get("share_with_team"))
        try:
            results = memoria.retrieve_memories_by_time_range(
                start_timestamp=start_dt,
                end_timestamp=end_dt,
                start_x=start_x_val,
                end_x=end_x_val,
                namespace=namespace,
                team_id=team_id,
                share_with_team=share_with_team,
            )
        except MemoriaError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 403
        return jsonify(results)
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"status": "error", "message": str(e)}), 500
