from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memoria.config.manager import ConfigManager
from memoria.database.models import Base, LongTermMemory, ShortTermMemory
from scripts.index_clusters import build_index


def test_build_index_includes_memories_without_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = ConfigManager()
    db_path = tmp_path / "mem.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    cfg.update_setting("database.connection_string", db_url)
    cfg.update_setting("enable_vector_clustering", False)

    engine = create_engine(cfg.get_settings().database.connection_string)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add_all(
            [
                LongTermMemory(
                    memory_id="m1",
                    processed_data={},
                    importance_score=0.5,
                    category_primary="test",
                    searchable_content="with summary",
                    summary="summary",
                ),
                LongTermMemory(
                    memory_id="m2",
                    processed_data={},
                    importance_score=0.5,
                    category_primary="test",
                    searchable_content="fallback text",
                    summary="",
                ),
            ]
        )
        session.commit()

    clusters = build_index()
    ids = [m["memory_id"] for c in clusters for m in c["members"]]
    assert "m1" in ids
    assert "m2" in ids


def test_build_index_respects_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = ConfigManager()
    db_path = tmp_path / "short.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    cfg.update_setting("database.connection_string", db_url)
    cfg.update_setting("enable_vector_clustering", False)

    engine = create_engine(cfg.get_settings().database.connection_string)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.add_all(
            [
                LongTermMemory(
                    memory_id="ltm",
                    processed_data={},
                    importance_score=0.1,
                    category_primary="test",
                    searchable_content="ltm",
                    summary="ltm summary",
                ),
                ShortTermMemory(
                    memory_id="stm",
                    processed_data={},
                    importance_score=0.9,
                    category_primary="test",
                    searchable_content="stm",
                    summary="stm summary",
                ),
            ]
        )
        session.commit()

    clusters = build_index(sources=["ShortTermMemory"])
    ids = [m["memory_id"] for c in clusters for m in c["members"]]
    assert "stm" in ids
    assert "ltm" not in ids
