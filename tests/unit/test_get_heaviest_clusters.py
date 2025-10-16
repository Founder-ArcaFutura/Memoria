from sqlalchemy import create_engine

from memoria.config.manager import ConfigManager
from memoria.database.models import Base
from memoria.database.queries.cluster_queries import replace_clusters
from memoria.utils.cluster_index import get_heaviest_clusters


def test_get_heaviest_clusters(tmp_path):
    cfg = ConfigManager()
    db = tmp_path / "mem.db"
    cfg.update_setting("database.connection_string", f"sqlite:///{db}")
    engine = create_engine(cfg.get_settings().database.connection_string)
    Base.metadata.create_all(engine)

    data = [
        {"weight": 1.0, "members": [{"memory_id": "a", "summary": "", "anchor": ""}]},
        {"weight": 3.0, "members": [{"memory_id": "b", "summary": "", "anchor": ""}]},
        {"weight": 2.0, "members": [{"memory_id": "c", "summary": "", "anchor": ""}]},
        {"members": [{"memory_id": "d", "summary": "", "anchor": ""}]},
    ]
    replace_clusters(data)

    clusters = get_heaviest_clusters(top_n=3)
    weights = [round(c.get("weight", 0), 1) for c in clusters]
    assert weights == [3.0, 2.0, 1.0]
    assert all("members" not in c for c in clusters)

    clusters_with_members = get_heaviest_clusters(top_n=1, include_members=True)
    assert clusters_with_members[0]["members"]
