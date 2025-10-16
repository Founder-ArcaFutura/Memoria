"""Routes and helpers for serving the optional Memoria dashboard UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    request,
    send_from_directory,
)
from itsdangerous import BadData, BadSignature, URLSafeSerializer


UI_SESSION_COOKIE_NAME = "memoria_ui_session"
_SERIALIZER_SALT = "memoria-ui-session"


ui_bp = Blueprint("ui", __name__, url_prefix="/ui")


def _get_ui_root() -> Path:
    """Return the directory that contains the UI assets."""

    root = current_app.config.get("UI_PATH")
    if not root:
        abort(404)
    return Path(root)


def _ui_enabled() -> bool:
    """Return whether the UI has been enabled via configuration."""

    return bool(current_app.config.get("SERVE_UI"))


def _get_serializer(app: Any) -> URLSafeSerializer:
    """Build a serializer for the UI session cookie."""

    secret_key = app.config.get("SECRET_KEY")
    if not secret_key:  # pragma: no cover - configuration guard
        raise RuntimeError("SECRET_KEY must be configured to issue UI sessions")
    return URLSafeSerializer(secret_key, salt=_SERIALIZER_SALT)


def issue_ui_session_token(app: Any, expected_api_key: str) -> str:
    """Return a signed token that authorizes UI requests."""

    serializer = _get_serializer(app)
    return serializer.dumps({"api_key": expected_api_key})


def validate_ui_session_token(app: Any, token: str, expected_api_key: str | None) -> bool:
    """Validate a previously issued UI session token."""

    if not token or not expected_api_key:
        return False
    serializer = _get_serializer(app)
    try:
        data = serializer.loads(token)
    except (BadSignature, BadData):
        return False
    return data.get("api_key") == expected_api_key


def _send_ui_asset(filename: str):
    if not _ui_enabled():
        abort(404)
    ui_root = _get_ui_root()
    return send_from_directory(ui_root, filename)


@ui_bp.route("/session", methods=["POST"], endpoint="session")
def create_ui_session():
    """Validate the API key and issue a browser session cookie."""

    if not _ui_enabled():
        abort(404)

    expected = current_app.config.get("EXPECTED_API_KEY")
    if not expected:
        return (
            jsonify({"status": "error", "message": "API key authentication is not configured"}),
            400,
        )

    payload = request.get_json(silent=True) or {}
    provided = payload.get("api_key")
    if provided != expected:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    token = issue_ui_session_token(current_app, expected)
    cookie_name = current_app.config.get("UI_SESSION_COOKIE_NAME", UI_SESSION_COOKIE_NAME)
    response = jsonify({"status": "ok"})
    response.set_cookie(
        cookie_name,
        token,
        httponly=True,
        secure=request.is_secure,
        samesite="Lax",
        path="/",
    )
    return response


@ui_bp.route("/", methods=["GET"], endpoint="index", strict_slashes=False)
def serve_index():
    """Serve the UI entry point."""

    return _send_ui_asset("index.html")


@ui_bp.route("/<path:filename>", methods=["GET"], endpoint="asset")
def serve_asset(filename: str):
    """Serve additional UI assets (JavaScript, CSS, etc.)."""

    return _send_ui_asset(filename)


__all__ = ["UI_SESSION_COOKIE_NAME", "ui_bp", "validate_ui_session_token"]
