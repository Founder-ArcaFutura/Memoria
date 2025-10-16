"""Tests for cluster centroid tracking."""

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memoria.config.manager import ConfigManager
from memoria.database.models import Base, Cluster, LongTermMemory
from scripts.index_clusters import build_index


@pytest.fixture
def configured_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = ConfigManager()
    settings = cfg.get_settings()
    original_conn = settings.database.connection_string
    original_vector = getattr(settings, "enable_vector_clustering", False)
    db_path = tmp_path / "centroids.db"
    connection = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", connection)
    cfg.update_setting("database.connection_string", connection)
    cfg.update_setting("enable_vector_clustering", False)

    engine = create_engine(connection)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    try:
        yield Session
    finally:
        engine.dispose()
        cfg.update_setting("database.connection_string", original_conn)
        cfg.update_setting("enable_vector_clustering", original_vector)


def _add_memory(session, memory_id: str, y: float, z: float) -> None:
    session.add(
        LongTermMemory(
            memory_id=memory_id,
            processed_data={},
            importance_score=0.5,
            category_primary="test",
            searchable_content=f"content {memory_id}",
            summary=f"summary {memory_id}",
            symbolic_anchors=["shared"],
            y_coord=y,
            z_coord=z,
        )
    )


@pytest.mark.parametrize("coords", [((-5.0, 1.0), (5.0, -3.0))])
def test_cluster_centroids_update_with_new_members(
    configured_session, coords, monkeypatch
):
    monkeypatch.setenv("DATABASE_URL", str(configured_session.kw["bind"].url))
    Session = configured_session
    with Session() as session:
        _add_memory(session, "m1", coords[0][0], coords[0][1])
        session.commit()

    clusters = build_index()
    assert clusters
    cluster = clusters[0]
    assert pytest.approx(coords[0][0]) == cluster.get("y_centroid")
    assert pytest.approx(coords[0][1]) == cluster.get("z_centroid")

    with Session() as session:
        _add_memory(session, "m2", coords[1][0], coords[1][1])
        session.commit()

    clusters = build_index()
    expected_y = sum(c[0] for c in coords) / len(coords)
    expected_z = sum(c[1] for c in coords) / len(coords)
    assert pytest.approx(expected_y) == clusters[0].get("y_centroid")
    assert pytest.approx(expected_z) == clusters[0].get("z_centroid")

    with Session() as session:
        stored = session.query(Cluster).one()
        assert pytest.approx(expected_y) == stored.y_centroid
        assert pytest.approx(expected_z) == stored.z_centroid
