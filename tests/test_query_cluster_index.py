from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine

from memoria.config.manager import ConfigManager
from memoria.database.models import Base
from memoria.database.queries.cluster_queries import replace_clusters
from memoria.utils.cluster_index import query_cluster_index


def _setup(tmp_path):
    cfg = ConfigManager()
    db = tmp_path / "mem.db"
    cfg.update_setting("database.connection_string", f"sqlite:///{db}")
    engine = create_engine(cfg.get_settings().database.connection_string)
    Base.metadata.create_all(engine)


def test_filters_and_sorting(tmp_path):
    _setup(tmp_path)
    now = datetime.now(timezone.utc)
    clusters = [
        {
            "summary": "recent",
            "members": [],
            "emotions": {"polarity": 0.1},
            "size": 1,
            "avg_importance": 0.5,
            "weight": 1.0,
            "update_count": 1,
            "last_updated": (now - timedelta(days=1)).isoformat(),
        },
        {
            "summary": "older",
            "members": [],
            "emotions": {"polarity": 0.1},
            "size": 1,
            "avg_importance": 0.5,
            "weight": 3.0,
            "update_count": 2,
            "last_updated": (now - timedelta(days=5)).isoformat(),
        },
    ]
    replace_clusters(clusters)

    res = query_cluster_index(weight_range=(2.0, 5.0))
    assert len(res) == 1 and res[0]["weight"] == 3.0

    res = query_cluster_index(time_since_update_range=(0, 2))
    assert len(res) == 1 and abs(res[0]["time_since_update"] - 1.0) < 0.1

    res = query_cluster_index(sort_by="weight")
    assert [round(c["weight"], 1) for c in res] == [3.0, 1.0]

    res = query_cluster_index(sort_by="time_since_update")
    assert abs(res[0]["time_since_update"] - 1.0) < 0.1

    first = res[0]
    assert {"weight", "update_count", "last_updated", "time_since_update"} <= set(first)
