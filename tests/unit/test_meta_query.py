import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.database.models import Base, LongTermMemory, ShortTermMemory
from memoria.database.search_service import SearchService


def _setup_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    s1 = ShortTermMemory(
        memory_id="s1",
        processed_data={"text": "a"},
        importance_score=0.5,
        category_primary="general",
        namespace="default",
        searchable_content="a",
        summary="a",
        created_at=datetime.utcnow(),
        symbolic_anchors=["alpha"],
        y_coord=0.0,
    )
    l1 = LongTermMemory(
        memory_id="l1",
        processed_data={"text": "b"},
        importance_score=0.5,
        category_primary="general",
        namespace="default",
        searchable_content="b",
        summary="b",
        created_at=datetime.utcnow(),
        symbolic_anchors=["beta"],
        y_coord=5.0,
    )
    l2 = LongTermMemory(
        memory_id="l2",
        processed_data={"text": "c"},
        importance_score=0.5,
        category_primary="general",
        namespace="default",
        searchable_content="c",
        summary="c",
        created_at=datetime.utcnow(),
        symbolic_anchors=["alpha", "beta"],
        y_coord=-5.0,
    )
    session.add_all([s1, l1, l2])

    session.commit()
    return session


def test_meta_query_counts():
    session = _setup_session()
    service = SearchService(session, "sqlite")

    counts = service.meta_query(["alpha", "beta"], (-10.0, 10.0))
    assert counts["by_anchor"]["alpha"] == 2
    assert counts["by_anchor"]["beta"] == 2
    assert counts["total_memories"] == 3

    narrow = service.meta_query(["alpha"], (-2.0, 2.0))
    assert narrow["by_anchor"]["alpha"] == 1
    assert narrow["total_memories"] == 1
    session.close()
