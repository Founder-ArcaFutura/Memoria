import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.database.models import Base, ShortTermMemory
from memoria.database.search_service import SearchService


def _setup_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    m1 = ShortTermMemory(
        memory_id="m1",
        processed_data={"text": "hello"},
        importance_score=0.9,
        category_primary="general",
        namespace="default",
        searchable_content="hello world",
        summary="hello",
        created_at=datetime(2024, 3, 1),
        x_coord=0.0,
        y_coord=0.0,
        z_coord=0.0,
        symbolic_anchors=["alpha"],
    )
    m2 = ShortTermMemory(
        memory_id="m2",
        processed_data={"text": "hello"},
        importance_score=0.5,
        category_primary="general",
        namespace="default",
        searchable_content="hello world",
        summary="hello",
        created_at=datetime(2024, 1, 1),
        x_coord=10.0,
        y_coord=10.0,
        z_coord=10.0,
        symbolic_anchors=["beta"],
    )
    session.add_all([m1, m2])
    session.commit()
    return session


def test_combined_filters():
    session = _setup_session()
    service = SearchService(session, "sqlite")

    results = service.search_memories(
        query="hello",
        namespace="default",
        category_filter=["general"],
        keywords=["hello"],
        start_timestamp=datetime(2024, 2, 1),
        end_timestamp=datetime(2024, 4, 1),
        min_importance=0.8,
        anchors=["alpha"],
        x=0.0,
        y=0.0,
        z=0.0,
        max_distance=5.0,
    )["results"]

    assert len(results) == 1
    assert results[0]["memory_id"] == "m1"
    session.close()
