import datetime
import json
import pathlib
import re
import sqlite3
import sys
import types

import flask
import pytest
import sqlalchemy

# Ensure project root in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from memoria import Memoria
from memoria.config.settings import IngestMode
from memoria.database.models import (
    ChatHistory,
    LongTermMemory,
    ShortTermMemory,
    SpatialMetadata,
)
from memoria_server.api.manual_entry import manual_entry_bp
from memoria_server.api.memory_routes import memory_bp
from memoria_server.api.spatial_utils import upsert_spatial_metadata
from memoria_server.api.utility_routes import utility_bp


def create_app(tmp_path, use_db_path: bool):
    app = flask.Flask(__name__)
    db_file = tmp_path / "main.db"
    db_url = f"sqlite:///{db_file}"
    memoria = Memoria(
        database_connect=db_url, conscious_ingest=False, openai_api_key=None
    )
    memoria.enable()
    app.config["memoria"] = memoria
    if use_db_path:
        app.config["DB_PATH"] = str(db_file)
        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.execute("DROP TABLE IF EXISTS spatial_metadata")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spatial_metadata (
                memory_id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL DEFAULT 'default',
                team_id TEXT,
                workspace_id TEXT,
                timestamp TEXT,
                x REAL, y REAL, z REAL,
                symbolic_anchors TEXT
            )
            """
        )
        conn.commit()
        conn.close()
    else:
        app.config["DB_PATH"] = None
        engine = memoria.db_manager.engine
        with engine.begin() as conn:
            conn.execute(
                sqlalchemy.text(
                    """
                    CREATE TABLE IF NOT EXISTS spatial_metadata (
                        memory_id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL DEFAULT 'default',
                        team_id TEXT,
                        workspace_id TEXT,
                        timestamp TEXT,
                        x REAL, y REAL, z REAL,
                        symbolic_anchors JSON
                    )
                    """
                )
            )
    app.register_blueprint(memory_bp)
    app.register_blueprint(manual_entry_bp)
    app.register_blueprint(utility_bp)
    return app, memoria


def test_store_memory_with_db_path(tmp_path):
    app, _ = create_app(tmp_path, use_db_path=True)
    client = app.test_client()
    resp = client.post(
        "/memory",
        json={
            "anchor": "a",
            "text": "b",
            "tokens": 1,
            "x_coord": 1.0,
            "symbolic_anchors": [" X ", "Y"],
        },
    )
    assert resp.status_code == 200
    memory_id = resp.get_json()["memory_id"]
    conn = sqlite3.connect(app.config["DB_PATH"])
    row = conn.execute(
        "SELECT x, symbolic_anchors FROM spatial_metadata WHERE memory_id=?",
        (memory_id,),
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == pytest.approx(1.0)
    assert json.loads(row[1]) == ["X", "Y"]


def test_store_memory_without_db_path(tmp_path):
    app, memoria = create_app(tmp_path, use_db_path=False)
    client = app.test_client()
    resp = client.post(
        "/memory",
        json={
            "anchor": "a",
            "text": "b",
            "tokens": 1,
            "x_coord": 2.0,
            "symbolic_anchors": ["alpha"],
        },
    )
    assert resp.status_code == 200
    memory_id = resp.get_json()["memory_id"]
    with memoria.db_manager.SessionLocal() as session:
        row = session.query(SpatialMetadata).filter_by(memory_id=memory_id).one()
        assert row.x == pytest.approx(2.0)
        assert row.symbolic_anchors == ["alpha"]


def test_list_anchors_without_db_path(tmp_path):
    app, _ = create_app(tmp_path, use_db_path=False)
    client = app.test_client()
    post_resp = client.post(
        "/memory",
        json={
            "anchor": "a",
            "text": "t",
            "tokens": 1,
            "symbolic_anchors": ["X"],
        },
    )
    assert post_resp.status_code == 200
    resp = client.get("/debug/anchors")
    assert resp.status_code == 200
    assert "X" in resp.get_json()["anchors"]


def test_openapi_spec_route(tmp_path):
    app, _ = create_app(tmp_path, use_db_path=False)
    client = app.test_client()
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body.get("openapi") == "3.1.0"
    assert body.get("info", {}).get("title") == "Symbolic Memory API"


def test_manual_entry_without_db_path(tmp_path):
    app, memoria = create_app(tmp_path, use_db_path=False)
    client = app.test_client()
    resp = client.post(
        "/manual",
        data={"anchor": "m", "text": "hello", "x_coord": 1.0},
    )
    assert resp.status_code == 200
    with memoria.db_manager.SessionLocal() as session:
        assert session.query(SpatialMetadata).count() == 1


def test_personal_ingest_route_links_chat_and_documents(tmp_path):
    app, memoria = create_app(tmp_path, use_db_path=False)
    memoria.ingest_mode = IngestMode.PERSONAL
    memoria.personal_mode_enabled = True
    memoria.personal_documents_enabled = True

    client = app.test_client()
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    payload = {
        "anchor": "personal-anchor",
        "text": "Captured directly from a chat session.",
        "tokens": 12,
        "timestamp": now.isoformat(),
        "x_coord": 0.0,
        "y_coord": 2.5,
        "z_coord": -1.0,
        "symbolic_anchors": ["personal", "demo"],
        "chat_id": "chat-personal-123",
        "metadata": {"topic": "personal-capture"},
        "documents": [
            {
                "document_id": "doc-personal-1",
                "title": "Personal Transcript",
                "url": "https://example.com/transcript",
            }
        ],
        "ingest_mode": "personal",
    }

    response = client.post("/memory", json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "stored"
    assert body["chat_id"] == payload["chat_id"]

    memory_id = body["memory_id"]
    with memoria.db_manager.SessionLocal() as session:
        long_term = session.query(LongTermMemory).filter_by(memory_id=memory_id).one()
        assert long_term.original_chat_id == payload["chat_id"]
        documents = long_term.processed_data.get("documents")
        assert isinstance(documents, list) and documents
        assert documents[0].get("document_id") == "doc-personal-1"

        assert session.query(ShortTermMemory).count() == 0

        spatial_entry = (
            session.query(SpatialMetadata).filter_by(memory_id=memory_id).one()
        )
        assert spatial_entry.x == pytest.approx(payload["x_coord"])
        assert spatial_entry.y == pytest.approx(payload["y_coord"])
        assert "personal" in spatial_entry.symbolic_anchors

        chat_entry = (
            session.query(ChatHistory).filter_by(chat_id=payload["chat_id"]).one()
        )
        assert chat_entry.namespace == long_term.namespace


def test_manual_entry_rejects_invalid_x_coord(tmp_path):
    app, _ = create_app(tmp_path, use_db_path=False)
    client = app.test_client()
    resp = client.post(
        "/manual",
        data={"anchor": "m", "text": "hello", "x_coord": "not-a-number"},
    )
    assert resp.status_code == 400
    body = resp.get_data(as_text=True)
    assert "x_coord must be a numeric value" in body


def test_manual_entry_with_db_path_timestamp(tmp_path):
    app, _ = create_app(tmp_path, use_db_path=True)
    client = app.test_client()
    resp = client.post(
        "/manual",
        data={"anchor": "m", "text": "hello world", "x_coord": 1.0},
    )
    assert resp.status_code == 200

    body = resp.get_data(as_text=True)
    match = re.search(r"Stored memory ([^<\s]+)", body)
    assert match is not None
    memory_id = match.group(1)

    conn = sqlite3.connect(app.config["DB_PATH"])
    row = conn.execute(
        "SELECT timestamp FROM spatial_metadata WHERE memory_id=?",
        (memory_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] is not None
    # Should be ISO formatted
    datetime.datetime.fromisoformat(row[0].replace("Z", "+00:00"))


def test_update_memory_with_db_path(tmp_path):
    app, _ = create_app(tmp_path, use_db_path=True)
    client = app.test_client()
    post_resp = client.post(
        "/memory",
        json={
            "anchor": "a",
            "text": "base text",
            "tokens": 2,
            "x_coord": 0.0,
            "symbolic_anchors": ["base"],
        },
    )
    assert post_resp.status_code == 200
    memory_id = post_resp.get_json()["memory_id"]

    update_resp = client.put(
        f"/memory/{memory_id}",
        json={
            "anchor": "a",
            "text": "base text updated",
            "tokens": 3,
            "y_coord": 5.5,
            "symbolic_anchors": ["base", "updated"],
        },
    )
    assert update_resp.status_code == 200

    conn = sqlite3.connect(app.config["DB_PATH"])
    row = conn.execute(
        "SELECT y, symbolic_anchors FROM spatial_metadata WHERE memory_id=?",
        (memory_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == pytest.approx(5.5)
    assert json.loads(row[1]) == ["base", "updated"]


def test_upsert_spatial_metadata_without_timestamp(tmp_path):
    db_file = tmp_path / "meta.db"
    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        CREATE TABLE spatial_metadata (
            memory_id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL DEFAULT 'default',
            x REAL, y REAL, z REAL,
            symbolic_anchors TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    upsert_spatial_metadata(
        memory_id="mid",
        namespace="default",
        db_path=str(db_file),
        db_manager=None,
        timestamp=None,
        x=1.5,
        y=2.5,
        z=-1.0,
        symbolic_anchors=["demo"],
    )

    conn = sqlite3.connect(db_file)
    row = conn.execute(
        "SELECT x, y, symbolic_anchors FROM spatial_metadata WHERE memory_id=?",
        ("mid",),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == pytest.approx(1.5)
    assert row[1] == pytest.approx(2.5)
    assert json.loads(row[2]) == ["demo"]


def test_rebuild_clusters_uses_db_path(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    app, memoria = create_app(tmp_path, use_db_path=True)
    db_path = pathlib.Path(app.config["DB_PATH"])
    memory_id = memoria.store_memory(
        anchor="alpha",
        text="Cluster source entry",
        tokens=5,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        symbolic_anchors=["alpha"],
    )

    conn_str = f"sqlite:///{db_path}"

    class DummySettings:
        def __init__(self, url: str):
            self._url = url
            self.enable_heuristic_clustering = True
            self.enable_vector_clustering = False
            self.database = types.SimpleNamespace(connection_string=url)

        def get_database_url(self) -> str:
            return self._url

    class DummyConfigManager:
        def __init__(self, url: str):
            self._settings = DummySettings(url)

        def get_settings(self):
            return self._settings

    app.config["config_manager"] = DummyConfigManager(conn_str)

    import scripts.heuristic_clusters as heuristics

    monkeypatch.setattr(
        heuristics,
        "_load_lambda",
        lambda: heuristics.DEFAULT_LAMBDA,
        raising=False,
    )
    monkeypatch.setattr(heuristics, "datetime", datetime.datetime, raising=False)
    monkeypatch.setattr(heuristics, "timezone", datetime.timezone, raising=False)

    client = app.test_client()
    resp = client.post("/clusters?mode=heuristic")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    clusters = data.get("clusters", [])
    assert clusters
    member_ids = {
        member.get("memory_id")
        for cluster in clusters
        for member in cluster.get("members", [])
    }
    assert memory_id in member_ids
