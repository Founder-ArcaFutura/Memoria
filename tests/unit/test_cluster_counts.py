from sqlalchemy import create_engine

from memoria.config.manager import ConfigManager
from memoria.database.models import Base
from memoria.database.queries.cluster_queries import query_clusters, replace_clusters


def test_replace_clusters_computes_counts(tmp_path):
    cfg = ConfigManager()
    db = tmp_path / "mem.db"
    cfg.update_setting("database.connection_string", f"sqlite:///{db}")
    engine = create_engine(cfg.get_settings().database.connection_string)
    Base.metadata.create_all(engine)

    cluster = {
        "summary": "test cluster",
        "members": [
            {"memory_id": "m1", "anchor": "a", "summary": "hello world"},
            {"memory_id": "m2", "anchor": "b", "summary": "foo"},
        ],
        "emotions": {"polarity": 0.0},
        "size": 2,
        "avg_importance": 0.5,
    }
    replace_clusters([cluster])

    result = query_clusters(include_members=True)
    assert result[0]["token_count"] == 3
    assert result[0]["char_count"] == len("hello world") + len("foo")
    members = {m["memory_id"]: m for m in result[0]["members"]}
    assert members["m1"]["tokens"] == 2
    assert members["m2"]["chars"] == len("foo")
