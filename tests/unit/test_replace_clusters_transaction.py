import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memoria.config.manager import ConfigManager
from memoria.database.models import Base, Cluster
from memoria.database.queries.cluster_queries import query_clusters, replace_clusters


def test_replace_clusters_rolls_back_on_failure(tmp_path, monkeypatch):
    cfg = ConfigManager()
    db = tmp_path / "mem.db"
    cfg.update_setting("database.connection_string", f"sqlite:///{db}")
    engine = create_engine(cfg.get_settings().database.connection_string)
    Base.metadata.create_all(engine)

    initial = {
        "summary": "initial",
        "members": [{"memory_id": "m1", "anchor": "a", "summary": ""}],
    }
    replace_clusters([initial])

    session = sessionmaker(bind=engine)()
    original_add = session.add

    def fail_on_cluster(obj):
        if isinstance(obj, Cluster):
            raise RuntimeError("boom")
        return original_add(obj)

    monkeypatch.setattr(session, "add", fail_on_cluster)

    new_cluster = {
        "summary": "new",
        "members": [{"memory_id": "m2", "anchor": "a", "summary": ""}],
    }

    with pytest.raises(RuntimeError):
        replace_clusters([new_cluster], session=session)

    remaining = query_clusters(include_members=True)
    assert len(remaining) == 1
    assert remaining[0]["members"][0]["memory_id"] == "m1"
    session.close()
