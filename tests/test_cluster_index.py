"""Tests for ``query_cluster_index`` utility."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from memoria.config.manager import ConfigManager
from memoria.database.models import Base
from memoria.database.queries.cluster_queries import replace_clusters
from memoria.utils import query_cluster_index


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    cfg = ConfigManager()
    db_path = tmp_path / "mem.db"
    cfg.update_setting("database.connection_string", f"sqlite:///{db_path}")
    engine = create_engine(cfg.get_settings().database.connection_string)
    Base.metadata.create_all(engine)
    yield


def test_query_cluster_index_returns_time_since_update_and_weight(
    tmp_path: Path,
) -> None:
    """Clusters include ``time_since_update`` (days) and ``weight`` fields."""
    now = datetime.now(timezone.utc)
    clusters = [
        {
            "summary": "one day old",
            "emotions": {"polarity": 0.1},
            "size": 1,
            "avg_importance": 0.5,
            "last_updated": (now - timedelta(days=1)).isoformat(),
            "weight": 2.5,
        },
        {
            "summary": "missing weight",
            "emotions": {"polarity": 0.0},
            "size": 1,
            "avg_importance": 0.2,
            "last_updated": (now - timedelta(hours=12)).isoformat(),
        },
        {
            "summary": "missing last updated",
            "emotions": {"polarity": -0.2},
            "size": 1,
            "avg_importance": 0.3,
            "weight": 1.0,
        },
    ]
    replace_clusters(clusters)

    result = query_cluster_index()
    assert len(result) == 3
    for cluster in result:
        assert "y_centroid" in cluster
        assert "z_centroid" in cluster

    one_day = result[0]
    assert 0.9 <= one_day["time_since_update"] <= 1.1
    assert one_day["weight"] == 2.5

    missing_weight = result[1]
    assert 0.4 <= missing_weight["time_since_update"] <= 0.6
    assert missing_weight["weight"] == 0.0

    missing_last = result[2]
    assert missing_last["time_since_update"] is None
    assert missing_last["weight"] == 1.0


@pytest.mark.parametrize(
    "cluster,expected_age",
    [
        ({"last_updated": None}, None),
        ({"last_updated": "invalid"}, None),
    ],
)
def test_query_cluster_index_handles_missing_last_updated(
    tmp_path: Path, cluster: dict, expected_age: float | None
) -> None:
    """Gracefully handles clusters without ``last_updated``."""
    cluster.update(
        {
            "summary": "edge",
            "emotions": {"polarity": 0.0},
            "size": 1,
            "avg_importance": 0.1,
            "weight": 0.5,
        }
    )
    replace_clusters([cluster])

    result = query_cluster_index()
    assert result[0]["time_since_update"] is expected_age


def test_query_cluster_index_accepts_list_format(tmp_path: Path) -> None:
    """Old index files stored a bare list; ensure we still parse them."""
    cluster = {
        "summary": "legacy",
        "emotions": {"polarity": 0.0},
        "size": 1,
        "avg_importance": 0.5,
    }
    replace_clusters([cluster])

    result = query_cluster_index()
    assert len(result) == 1
