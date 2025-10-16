import importlib

import pytest

pytest.importorskip("flask")


try:
    app_factory = importlib.import_module("memoria_server.api.app_factory")
except Exception as exc:  # pragma: no cover - defensive
    pytest.skip(f"memoria_server.api.app_factory import failed: {exc}")
else:
    create_app = app_factory.create_app


class DummyMemoria:
    def __init__(self):
        self.last_anchor = None
        self.called_method = None
        self.call_args = None
        self.response = []
        self.is_enabled = True

    def retrieve_memories_near(
        self, x, y, z, max_distance, anchor=None, limit=10, **kwargs
    ):
        self.called_method = "3d"
        self.last_anchor = anchor
        self.call_args = {
            "x": x,
            "y": y,
            "z": z,
            "max_distance": max_distance,
            "limit": limit,
        }
        return self.response

    def retrieve_memories_near_2d(
        self, y, z, max_distance, anchor=None, limit=10, **kwargs
    ):
        self.called_method = "2d"
        self.last_anchor = anchor
        self.call_args = {
            "y": y,
            "z": z,
            "max_distance": max_distance,
            "limit": limit,
        }
        return self.response

    def enable(self):
        pass


def _setup_client(monkeypatch):
    dummy = DummyMemoria()
    app = create_app()
    app.config["TESTING"] = True
    app.config["memoria"] = dummy
    client = app.test_client()
    return client, dummy


def test_anchor_json_array(monkeypatch):
    client, dummy = _setup_client(monkeypatch)
    resp = client.get(
        "/memory/spatial",
        query_string=[("x", "0"), ("y", "0"), ("z", "0"), ("anchor", '["one","two"]')],
    )
    assert resp.status_code == 200
    assert dummy.last_anchor == ["one", "two"]
    assert dummy.called_method == "3d"


def test_anchor_comma_separated(monkeypatch):
    client, dummy = _setup_client(monkeypatch)
    resp = client.get(
        "/memory/spatial",
        query_string=[
            ("x", "0"),
            ("y", "0"),
            ("z", "0"),
            ("anchor", "foo,bar"),
            ("anchor", "baz"),
        ],
    )
    assert resp.status_code == 200
    assert dummy.last_anchor == ["foo", "bar", "baz"]
    assert dummy.called_method == "3d"


def test_anchor_malformed(monkeypatch):
    client, dummy = _setup_client(monkeypatch)
    resp = client.get(
        "/memory/spatial",
        query_string=[("x", "0"), ("y", "0"), ("z", "0"), ("anchor", "[broken")],
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert dummy.last_anchor is None

    assert dummy.called_method is None


def test_mode_2d_calls_new_service(monkeypatch):
    client, dummy = _setup_client(monkeypatch)
    dummy.response = [{"x": 1.0, "y": 2.0, "distance": 0.5}]
    resp = client.get(
        "/memory/spatial",
        query_string={"mode": "2d", "y": "2"},
    )
    assert resp.status_code == 200
    assert resp.get_json() == dummy.response
    assert dummy.called_method == "2d"
    assert dummy.call_args == {
        "y": 2.0,
        "z": 0.0,
        "max_distance": 5.0,
        "limit": 10,
    }
    assert dummy.last_anchor is None


def test_mode_invalid_rejected(monkeypatch):
    client, dummy = _setup_client(monkeypatch)
    resp = client.get(
        "/memory/spatial",
        query_string={"mode": "flat", "x": "0", "y": "0"},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert dummy.called_method is None
