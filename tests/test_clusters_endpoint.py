import importlib
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memoria.cli_support import heuristic_clusters as hc


@pytest.fixture
def client(monkeypatch, tmp_path):
    class DummyMemoria:
        def __init__(self):
            self._enabled = False
            self.sovereign_ingest = False
            self.auto_ingest = False
            self.conscious_ingest = False

        def enable(self):
            self._enabled = True

        @property
        def is_enabled(self):
            return self._enabled

    # Prevent background timers from running
    class DummyTimer:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            pass

    monkeypatch.setattr("threading.Timer", lambda *a, **kw: DummyTimer())
    monkeypatch.setattr("memoria.Memoria", lambda *a, **kw: DummyMemoria())

    # Stub out index building module to avoid heavy imports
    import types

    fake_ic = types.ModuleType("index_clusters")
    # Return an empty list to match the real build_index return type

    fake_ic.build_index = lambda *args, **kwargs: []
    fake_ic.INDEX_STATUS = {
        "state": "running",
        "current": 0,
        "total": 0,
        "error": None,
    }

    def _fake_get_status():
        # Yield to other threads so tests updating INDEX_STATUS can take effect
        time.sleep(0.005)
        return fake_ic.INDEX_STATUS

    fake_ic.get_status = _fake_get_status
    fake_scripts = types.ModuleType("scripts")
    fake_scripts.index_clusters = fake_ic
    monkeypatch.setitem(sys.modules, "scripts", fake_scripts)
    monkeypatch.setitem(sys.modules, "scripts.index_clusters", fake_ic)

    # Ensure utility routes pick up the stubbed module each time
    import memoria_server.api.utility_routes as ur

    importlib.reload(ur)
    monkeypatch.setitem(sys.modules, "memoria_server.api.utility_routes", ur)

    # Seed clusters in database
    from sqlalchemy import create_engine

    from memoria.config.manager import ConfigManager
    from memoria.database.models import Base
    from memoria.database.queries.cluster_queries import replace_clusters

    cfg = ConfigManager()
    db_path = tmp_path / "memory.db"
    cfg.update_setting("database.connection_string", f"sqlite:///{db_path}")
    engine = create_engine(cfg.get_settings().database.connection_string)
    Base.metadata.create_all(engine)

    now = datetime.now(timezone.utc)
    replace_clusters(
        [
            {
                "summary": "Trip to Paris",
                "centroid": [0.1, 0.2, 0.3],
                "members": [
                    {"memory_id": "m1", "anchor": "paris", "summary": "Paris trip"},
                    {"memory_id": "m2", "anchor": "travel", "summary": "Travel diary"},
                ],
                "emotions": {"polarity": 0.8},
                "size": 2,
                "avg_importance": 0.7,
                "update_count": 5,
                "last_updated": (now - timedelta(seconds=100)).isoformat(),
                "weight": 1.5,
            },
            {
                "summary": "Work meeting",
                "centroid": [1.0, 1.1, 1.2],
                "members": [
                    {
                        "memory_id": "m3",
                        "anchor": "work",
                        "summary": "Project discussion",
                    }
                ],
                "emotions": {"polarity": -0.1},
                "size": 1,
                "avg_importance": 0.2,
                "update_count": 1,
                "last_updated": (now - timedelta(seconds=10000)).isoformat(),
                "weight": 0.2,
            },
        ]
    )

    import memoria_api

    importlib.reload(memoria_api)
    memoria_api.DB_PATH = str(db_path)
    memoria_api.app.config["DB_PATH"] = memoria_api.DB_PATH
    memoria_api.init_spatial_db()

    return memoria_api.app.test_client()


def _ids(resp):
    return [c["id"] for c in resp.get_json()["clusters"]]


@pytest.fixture
def patched_index(monkeypatch):
    calls = {"build": 0, "query": 0}
    clusters = [{"id": 42}]

    def fake_build_index(*args, **kwargs):
        calls["build"] += 1
        return clusters

    def fake_query_cluster_index(*args, **kwargs):
        calls["query"] += 1
        return clusters

    monkeypatch.setattr("memoria_api.build_index", fake_build_index)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_index", fake_build_index
    )
    monkeypatch.setattr("memoria_api.query_cluster_index", fake_query_cluster_index)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.query_cluster_index",
        fake_query_cluster_index,
    )
    return calls, clusters


def test_clusters_filters_and_fields(client):
    resp = client.get("/clusters")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(_ids(resp)) == {0, 1}
    first = data["clusters"][0]
    for field in [
        "id",
        "summary",
        "centroid",
        "y_centroid",
        "z_centroid",
        "emotions",
        "size",
        "avg_importance",
        "update_count",
        "last_updated",
        "weight",
    ]:
        assert field in first
    assert "members" not in first
    assert "polarity" in first["emotions"]

    assert _ids(client.get("/clusters?keyword=paris")) == [0]
    assert _ids(client.get("/clusters?min_polarity=0")) == [0]
    assert _ids(client.get("/clusters?max_polarity=0")) == [1]
    assert _ids(client.get("/clusters?min_size=2")) == [0]
    assert _ids(client.get("/clusters?max_size=1")) == [1]
    assert _ids(client.get("/clusters?min_importance=0.6")) == [0]
    assert _ids(client.get("/clusters?max_importance=0.3")) == [1]
    assert _ids(client.get("/clusters?min_weight=1")) == [0]
    assert _ids(client.get("/clusters?max_weight=0.5")) == [1]
    assert _ids(client.get("/clusters?max_age_seconds=500")) == [0]
    assert _ids(client.get("/clusters?min_age_seconds=5000")) == [1]
    assert _ids(client.get("/clusters?sort_by=weight")) == [0, 1]


def test_cluster_activity_endpoint(client):
    resp = client.get("/clusters/activity?top_n=1&fading_threshold=0.3")
    assert resp.status_code == 200
    data = resp.get_json()
    assert [c["id"] for c in data["active"]] == [0]
    assert [c["id"] for c in data["fading"]] == [1]


def test_cluster_activity_invalid_top_n(client):
    resp = client.get("/clusters/activity?top_n=0")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["message"] == "Parameter 'top_n' must be a positive integer"


def test_cluster_activity_invalid_threshold(client):
    resp = client.get("/clusters/activity?fading_threshold=-0.5")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["message"] == "Parameter 'fading_threshold' must be a positive number"


def test_vector_clusters_with_patched_index(client, patched_index):
    calls, clusters = patched_index
    import memoria_api

    memoria_api.config_manager.update_setting("enable_vector_clustering", True)
    memoria_api.config_manager.update_setting("enable_heuristic_clustering", False)
    resp = client.post("/clusters?mode=vector")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["clusters"] == clusters
    assert calls["build"] == 1
    assert calls["query"] == 0


def test_post_rebuilds_vector_cluster_index(client, monkeypatch):
    calls = {"build": 0, "query": 0}

    def fake_build_index(*args, **kwargs):
        calls["build"] += 1
        return [{"id": 2}]

    def fake_query_cluster_index(*args, **kwargs):
        calls["query"] += 1
        return [{"id": 2}]

    monkeypatch.setattr("memoria_api.build_index", fake_build_index)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_index", fake_build_index
    )
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_index", fake_build_index
    )
    monkeypatch.setattr("memoria_api.query_cluster_index", fake_query_cluster_index)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.query_cluster_index",
        fake_query_cluster_index,
    )

    import memoria_api

    memoria_api.config_manager.update_setting("enable_vector_clustering", True)
    memoria_api.config_manager.update_setting("enable_heuristic_clustering", False)

    resp = client.post("/clusters?mode=vector")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["clusters"] == [{"id": 2}]
    assert calls["build"] == 1
    assert calls["query"] == 0


def test_vector_cluster_sources_payload(client, monkeypatch):
    captured: dict[str, list[str] | None] = {"sources": None}

    def fake_build_index(*args, **kwargs):
        captured["sources"] = kwargs.get("sources")
        return [{"id": 9}]

    monkeypatch.setattr("memoria_api.build_index", fake_build_index)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_index", fake_build_index
    )

    import memoria_api

    memoria_api.config_manager.update_setting("enable_vector_clustering", True)
    memoria_api.config_manager.update_setting("enable_heuristic_clustering", False)

    resp = client.post(
        "/clusters?mode=vector",
        json={"sources": ["ShortTermMemory"]},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["clusters"] == [{"id": 9}]
    assert captured["sources"] == ["ShortTermMemory"]


def test_handle_heuristic_cluster_rebuild_helper(monkeypatch, client):
    import memoria_api
    import memoria_server.api.utility_routes as ur

    memoria_api.config_manager.update_setting("enable_heuristic_clustering", True)
    memoria_api.config_manager.update_setting("enable_vector_clustering", False)

    captured: dict[str, object] = {"connection": None, "clusters": None}

    def fake_build(connection_string):
        captured["connection"] = connection_string
        return ([{"id": 55, "anchor": "helper", "members": []}], {"notes": "ok"})

    def fake_replace(clusters):
        captured["clusters"] = clusters

    monkeypatch.setattr(ur, "build_heuristic_clusters", fake_build)
    monkeypatch.setattr(
        "memoria.database.queries.cluster_queries.replace_clusters",
        fake_replace,
    )

    settings = memoria_api.config_manager.get_settings()
    with client.application.app_context():
        response = ur._handle_heuristic_cluster_rebuild(settings, None)

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["clusters"] == [{"id": 55, "anchor": "helper", "members": []}]
    assert data["summary"] == {"notes": "ok"}
    assert captured["connection"]
    assert captured["clusters"] == [{"id": 55, "anchor": "helper", "members": []}]


def test_handle_vector_cluster_rebuild_helper(monkeypatch, client):
    import memoria_api
    import memoria_server.api.utility_routes as ur

    memoria_api.config_manager.update_setting("enable_vector_clustering", True)
    memoria_api.config_manager.update_setting("enable_heuristic_clustering", False)

    captured: dict[str, object] = {"sources": None}

    def fake_build_index(*, sources=None):
        captured["sources"] = sources
        return [{"id": 66}]

    monkeypatch.setattr(ur, "build_index", fake_build_index)

    settings = memoria_api.config_manager.get_settings()
    with client.application.app_context():
        response = ur._handle_vector_cluster_rebuild(settings, ["ShortTermMemory"])

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["clusters"] == [{"id": 66}]
    assert captured["sources"] == ["ShortTermMemory"]


def test_vector_clustering_respects_disabled_setting(client, monkeypatch):
    calls = {"build": 0}

    def fake_build_index(*args, **kwargs):
        calls["build"] += 1

    monkeypatch.setattr("memoria_api.build_index", fake_build_index)
    import memoria_api

    memoria_api.config_manager.update_setting("enable_vector_clustering", False)
    memoria_api.config_manager.update_setting("enable_heuristic_clustering", True)

    resp = client.post("/clusters?mode=vector")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "disabled"
    assert data["message"] == "Vector clustering is disabled"
    assert isinstance(data.get("clusters"), list) and data["clusters"]
    assert calls["build"] == 0


def test_vector_clustering_handles_missing_build_index(client, monkeypatch):
    import memoria_api

    memoria_api.config_manager.update_setting("enable_vector_clustering", True)
    memoria_api.config_manager.update_setting("enable_heuristic_clustering", False)
    monkeypatch.setattr("memoria_api.build_index", None)
    monkeypatch.setattr("memoria_server.api.utility_routes.build_index", None)

    resp = client.post("/clusters?mode=vector")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert data["message"] == "Cluster indexing dependencies missing"
    assert isinstance(data.get("clusters"), list)


def test_vector_build_index_failure_returns_500(client, monkeypatch):
    calls = {"memoria_query": 0, "utility_query": 0}

    def boom_index(*args, **kwargs):
        raise RuntimeError("boom")

    def fake_memoria_query_cluster_index(*args, **kwargs):
        calls["memoria_query"] += 1
        return []

    def fake_api_query_cluster_index(*args, **kwargs):
        calls["utility_query"] += 1
        return []

    monkeypatch.setattr("memoria_api.build_index", boom_index)
    monkeypatch.setattr("memoria_server.api.utility_routes.build_index", boom_index)
    monkeypatch.setattr(
        "memoria_api.query_cluster_index", fake_memoria_query_cluster_index
    )
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.query_cluster_index",
        fake_api_query_cluster_index,
    )

    import memoria_api

    memoria_api.config_manager.update_setting("enable_vector_clustering", True)
    memoria_api.config_manager.update_setting("enable_heuristic_clustering", False)

    resp = client.post("/clusters?mode=vector")
    assert resp.status_code == 500
    data = resp.get_json()
    assert data["status"] == "error"
    assert "boom" in data["message"].lower()
    assert calls["memoria_query"] == 0
    assert calls["utility_query"] == 0


def test_default_runs_heuristic_clusters(client, monkeypatch):
    calls = {"heuristic": 0, "index": 0}

    def fake_heuristic(*args, **kwargs):
        calls["heuristic"] += 1
        return [{"id": 3, "anchor": "z", "members": []}]

    def fake_index(*args, **kwargs):
        calls["index"] += 1

    monkeypatch.setattr("memoria_api.build_heuristic_clusters", fake_heuristic)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_heuristic_clusters", fake_heuristic
    )
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_heuristic_clusters", fake_heuristic
    )
    monkeypatch.setattr("memoria_api.build_index", fake_index)
    monkeypatch.setattr("memoria_server.api.utility_routes.build_index", fake_index)
    import memoria_api

    memoria_api.config_manager.update_setting("enable_heuristic_clustering", True)
    memoria_api.config_manager.update_setting("enable_vector_clustering", False)

    resp = client.post("/clusters")
    assert resp.status_code == 200
    assert resp.get_json() == {
        "status": "success",
        "clusters": [{"id": 3, "anchor": "z", "members": []}],
    }
    assert calls["heuristic"] == 1
    assert calls["index"] == 0


def test_default_runs_vector_clusters(client, monkeypatch):
    calls = {"build": 0, "query": 0}

    def fake_build_index(*args, **kwargs):
        calls["build"] += 1
        return [{"id": 4}]

    def fake_query_cluster_index(*args, **kwargs):
        calls["query"] += 1
        return [{"id": 4}]

    monkeypatch.setattr("memoria_api.build_index", fake_build_index)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_index", fake_build_index
    )
    monkeypatch.setattr("memoria_api.query_cluster_index", fake_query_cluster_index)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.query_cluster_index",
        fake_query_cluster_index,
    )

    import memoria_api

    memoria_api.config_manager.update_setting("enable_heuristic_clustering", False)
    memoria_api.config_manager.update_setting("enable_vector_clustering", True)

    resp = client.post("/clusters")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["clusters"] == [{"id": 4}]
    assert calls["build"] == 1
    assert calls["query"] == 0


def test_default_clustering_disabled(client):
    import memoria_api

    memoria_api.config_manager.update_setting("enable_heuristic_clustering", False)
    memoria_api.config_manager.update_setting("enable_vector_clustering", False)

    resp = client.post("/clusters")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "disabled"
    assert data["message"] == "Clustering is disabled"
    assert isinstance(data.get("clusters"), list)


def test_heuristic_clustering_respects_disabled_setting(client, monkeypatch):
    calls = {"heuristic": 0}

    def fake_heuristic(*args, **kwargs):
        calls["heuristic"] += 1

    monkeypatch.setattr("memoria_api.build_heuristic_clusters", fake_heuristic)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_heuristic_clusters", fake_heuristic
    )
    import memoria_api

    memoria_api.config_manager.update_setting("enable_heuristic_clustering", False)
    memoria_api.config_manager.update_setting("enable_vector_clustering", True)

    resp = client.post("/clusters?mode=heuristic")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "disabled"
    assert data["message"] == "Heuristic clustering is disabled"
    assert isinstance(data.get("clusters"), list)
    assert calls["heuristic"] == 0


def test_post_heuristic_clusters(client, monkeypatch):
    calls = {"heuristic": 0, "index": 0}

    def fake_heuristic(*args, **kwargs):
        calls["heuristic"] += 1
        return [{"id": 1, "anchor": "x", "members": []}]

    def fake_index(*args, **kwargs):
        calls["index"] += 1

    monkeypatch.setattr("memoria_api.build_heuristic_clusters", fake_heuristic)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_heuristic_clusters", fake_heuristic
    )
    monkeypatch.setattr("memoria_api.build_index", fake_index)
    monkeypatch.setattr("memoria_server.api.utility_routes.build_index", fake_index)
    import memoria_api

    memoria_api.config_manager.update_setting("enable_heuristic_clustering", True)
    memoria_api.config_manager.update_setting("enable_vector_clustering", True)

    resp = client.post("/clusters?mode=heuristic")
    assert resp.status_code == 200
    assert resp.get_json() == {
        "status": "success",
        "clusters": [{"id": 1, "anchor": "x", "members": []}],
    }
    assert calls["heuristic"] == 1
    assert calls["index"] == 0


def test_post_heuristic_clusters_body(client, monkeypatch):
    calls = {"heuristic": 0, "index": 0}

    def fake_heuristic(*args, **kwargs):
        calls["heuristic"] += 1
        return [{"id": 2, "anchor": "y", "members": []}]

    def fake_index(*args, **kwargs):
        calls["index"] += 1

    monkeypatch.setattr("memoria_api.build_heuristic_clusters", fake_heuristic)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_heuristic_clusters", fake_heuristic
    )
    monkeypatch.setattr("memoria_api.build_index", fake_index)
    monkeypatch.setattr("memoria_server.api.utility_routes.build_index", fake_index)
    import memoria_api

    memoria_api.config_manager.update_setting("enable_heuristic_clustering", True)
    memoria_api.config_manager.update_setting("enable_vector_clustering", True)

    resp = client.post("/clusters", json={"mode": "heuristic"})
    assert resp.status_code == 200
    assert resp.get_json() == {
        "status": "success",
        "clusters": [{"id": 2, "anchor": "y", "members": []}],
    }
    assert calls["heuristic"] == 1
    assert calls["index"] == 0


def test_post_heuristic_clusters_includes_status_and_clusters(client, monkeypatch):
    """Heuristic clustering should immediately return clusters with success status."""

    sample_clusters = [{"id": 42, "anchor": "foo", "members": []}]

    def fake_heuristic(*args, **kwargs):
        return sample_clusters

    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_heuristic_clusters", fake_heuristic
    )

    import memoria_api

    memoria_api.config_manager.update_setting("enable_cluster_indexing", True)
    memoria_api.config_manager.update_setting("enable_heuristic_clustering", True)

    resp = client.post("/clusters?mode=heuristic")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "success", "clusters": sample_clusters}


@pytest.fixture
def mock_vector_cluster(monkeypatch, client):
    calls = {"build": 0, "query": 0}

    def fake_build_index(*args, **kwargs):
        calls["build"] += 1
        return [{"id": 7}]

    def fake_query_cluster_index(*args, **kwargs):
        calls["query"] += 1
        return [{"id": 7}]

    monkeypatch.setattr("memoria_api.build_index", fake_build_index)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_index", fake_build_index
    )
    monkeypatch.setattr("memoria_api.query_cluster_index", fake_query_cluster_index)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.query_cluster_index",
        fake_query_cluster_index,
    )
    return calls


def test_post_clusters_uses_vector(mock_vector_cluster, client):
    import memoria_api

    memoria_api.config_manager.update_setting("enable_heuristic_clustering", False)
    memoria_api.config_manager.update_setting("enable_vector_clustering", True)

    resp = client.post("/clusters")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["clusters"] == [{"id": 7}]
    assert mock_vector_cluster["build"] == 1
    assert mock_vector_cluster["query"] == 0


def test_post_clusters_build_index_error(client, monkeypatch):
    def bad_build_index(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("memoria_api.build_index", bad_build_index)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_index", bad_build_index
    )
    import memoria_api

    memoria_api.config_manager.update_setting("enable_heuristic_clustering", False)
    memoria_api.config_manager.update_setting("enable_vector_clustering", True)

    resp = client.post("/clusters")
    assert resp.status_code == 500
    body = resp.get_json()
    assert body["status"] == "error"
    assert "boom" in body["message"]


def test_cluster_status_reports_progress(client, monkeypatch):
    status = {"state": "idle", "current": 0, "total": 0, "error": None}
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.get_cluster_status", lambda: status
    )

    status.update(state="running", total=2, current=1)
    resp = client.get("/clusters/status")
    assert resp.status_code == 200
    assert resp.get_json()["state"] == "running"

    status.update(state="complete", current=2)
    resp = client.get("/clusters/status")
    data = resp.get_json()
    assert data["state"] == "complete"
    assert data["current"] == 2
    assert data["total"] == 2


def test_cluster_status_error_after_failure(client):
    import memoria_server.api.utility_routes as ur

    ur.get_cluster_status = lambda: {
        "state": "error",
        "current": 1,
        "total": 5,
        "error": "boom",
    }
    status = client.get("/clusters/status").get_json()
    assert status["state"] == "error"
    assert "boom" in status.get("error", "")


def test_heuristic_clusters_succeed_without_vector_deps(client, monkeypatch):
    calls = {"heuristic": 0}

    def fake_heuristic(*args, **kwargs):
        calls["heuristic"] += 1
        return [{"id": 99, "anchor": "h", "members": []}]

    monkeypatch.setattr("memoria_api.build_heuristic_clusters", fake_heuristic)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_heuristic_clusters", fake_heuristic
    )
    monkeypatch.setattr("memoria_api.build_index", None)
    monkeypatch.setattr("memoria_server.api.utility_routes.build_index", None)

    import memoria_api

    memoria_api.config_manager.update_setting("enable_vector_clustering", False)
    memoria_api.config_manager.update_setting("enable_heuristic_clustering", True)

    resp = client.post("/clusters?mode=heuristic")
    assert resp.status_code == 200
    assert resp.get_json() == {
        "status": "success",
        "clusters": [{"id": 99, "anchor": "h", "members": []}],
    }
    assert calls["heuristic"] == 1


def test_vector_clustering_missing_deps_warns(client, monkeypatch):
    import memoria_api

    memoria_api.config_manager.update_setting("enable_vector_clustering", True)
    memoria_api.config_manager.update_setting("enable_heuristic_clustering", True)
    monkeypatch.setattr("memoria_api.build_index", None)
    monkeypatch.setattr("memoria_server.api.utility_routes.build_index", None)

    resp = client.post("/clusters")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "unavailable"
    assert "missing" in data["message"].lower()


def test_vector_clustering_runs_with_deps_present(client, monkeypatch):
    calls = {"build": 0}

    def fake_build_index(*args, **kwargs):
        calls["build"] += 1
        return [{"id": 123}]

    monkeypatch.setattr("memoria_api.build_index", fake_build_index)
    monkeypatch.setattr(
        "memoria_server.api.utility_routes.build_index", fake_build_index
    )

    import memoria_api

    memoria_api.config_manager.update_setting("enable_vector_clustering", True)
    memoria_api.config_manager.update_setting("enable_heuristic_clustering", True)

    resp = client.post("/clusters?mode=vector")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["clusters"] == [{"id": 123}]
    assert calls["build"] == 1


def test_clusters_invalid_mode_query(client):
    resp = client.post("/clusters?mode=invalid")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert "mode" in data["message"].lower()


def test_clusters_invalid_mode_body(client):
    resp = client.post("/clusters", json={"mode": "wrong"})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert "mode" in data["message"].lower()


def test_build_heuristic_clusters_preserves_existing_sinks(monkeypatch):
    messages: list[str] = []

    def dummy_sink(message):
        messages.append(message.record["message"])

    sink_id = logger.add(dummy_sink, level="INFO")
    sample_rows = [
        {
            "memory_id": "m1",
            "anchors": ["alpha"],
            "summary": "Sample summary",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
    ]
    monkeypatch.setattr(hc, "fetch_texts", lambda *a, **kw: sample_rows)
    monkeypatch.setattr(hc, "_load_lambda", lambda: 0.1)

    try:
        clusters, summary = hc.build_heuristic_clusters(verbose=True)
        assert clusters and summary.startswith("Built")
        logger.info("after-call message")
    finally:
        logger.remove(sink_id)

    assert any("Fetched" in msg for msg in messages)
    assert any("after-call message" in msg for msg in messages)
