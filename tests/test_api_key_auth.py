import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memoria_server.api.app_factory import create_app


class DummyMemoria:

    db_manager = None

    def __init__(self):
        self.is_enabled = False
        self.auto_ingest = False
        self.sovereign_ingest = False
        self.conscious_ingest = False

    def enable(self):
        self.is_enabled = True
        return None

    def search_memories(self, query="", limit=10):
        return [
            {
                "id": "m1",
                "anchor": "a",
                "text": "t",
                "tokens": 1,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            }
        ]


def build_test_app(
    monkeypatch,
    serve_ui: bool,
    *,
    set_custom_path: bool = True,
    set_api_key: bool = True,
):
    if set_api_key:
        monkeypatch.setenv("MEMORIA_API_KEY", "secret")
    else:
        monkeypatch.delenv("MEMORIA_API_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    if serve_ui:
        monkeypatch.setenv("MEMORIA_SERVE_UI", "1")
        if set_custom_path:
            monkeypatch.setenv(
                "MEMORIA_UI_PATH",
                str(
                    Path(__file__).resolve().parents[1] / "memoria_server" / "dashboard"
                ),
            )
        else:
            monkeypatch.delenv("MEMORIA_UI_PATH", raising=False)
    else:
        monkeypatch.delenv("MEMORIA_SERVE_UI", raising=False)
        monkeypatch.delenv("MEMORIA_UI_PATH", raising=False)
    monkeypatch.setattr(
        "memoria_server.api.app_factory.schedule_daily_decrement", lambda app: None
    )
    monkeypatch.setattr(
        "memoria_server.api.app_factory.schedule_roster_verification", lambda app: None
    )
    monkeypatch.setattr(
        "memoria_server.api.app_factory.memoria_pkg.Memoria",
        lambda *a, **kw: DummyMemoria(),
    )
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(monkeypatch):
    app = build_test_app(monkeypatch, serve_ui=False, set_api_key=True)
    with app.test_client() as client:
        yield client


def test_missing_api_key_rejected(client):
    resp = client.get("/memory/recent")
    assert resp.status_code == 401


def test_invalid_api_key_rejected(client):
    resp = client.get("/memory/recent", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_valid_api_key_allowed(client):
    resp = client.get("/memory/recent", headers={"X-API-Key": "secret"})
    assert resp.status_code == 200


def test_ui_routes_disabled_without_toggle(monkeypatch):
    app = build_test_app(monkeypatch, serve_ui=False)
    with app.test_client() as client:
        resp = client.get("/ui/")
        assert resp.status_code == 404

        resp = client.post("/ui/session", json={"api_key": "secret"})
        assert resp.status_code == 404


def test_ui_session_cookie_allows_access(monkeypatch):
    app = build_test_app(monkeypatch, serve_ui=True)
    with app.test_client() as client:
        index_resp = client.get("/ui/", headers={})
        assert index_resp.status_code == 200

        asset_resp = client.get("/ui/index.html", headers={})
        assert asset_resp.status_code == 200

        pre_auth = client.get("/memory/recent")
        assert pre_auth.status_code == 401

        bad_login = client.post("/ui/session", json={"api_key": "wrong"}, headers={})
        assert bad_login.status_code == 401
        assert "Set-Cookie" not in bad_login.headers

        login = client.post("/ui/session", json={"api_key": "secret"}, headers={})
        assert login.status_code == 200
        cookie_name = app.config["UI_SESSION_COOKIE_NAME"]
        cookie_header = login.headers.get("Set-Cookie", "")
        assert cookie_name in cookie_header
        assert "HttpOnly" in cookie_header
        assert "SameSite=Lax" in cookie_header
        assert "Path=/" in cookie_header
        assert "Secure" not in cookie_header

        authed = client.get("/memory/recent")
        assert authed.status_code == 200


def test_default_dashboard_served_without_custom_path(monkeypatch):
    app = build_test_app(monkeypatch, serve_ui=True, set_custom_path=False)
    with app.test_client() as client:
        response = client.get("/ui/index.html", headers={})
        assert response.status_code == 200
        assert b"<" in response.data
    default_path = Path(__file__).resolve().parents[1] / "memoria_server" / "dashboard"
    assert Path(app.config["UI_PATH"]).resolve() == default_path.resolve()


def test_ui_session_rejected_when_key_not_configured(monkeypatch):
    app = build_test_app(monkeypatch, serve_ui=True, set_api_key=False)
    with app.test_client() as client:
        response = client.post("/ui/session", json={"api_key": "secret"}, headers={})
        assert response.status_code == 400
        payload = response.get_json()
        assert payload["status"] == "error"
        assert "not configured" in payload["message"].lower()
