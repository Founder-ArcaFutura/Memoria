import sys
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.database.models import Base, LongTermMemory, ShortTermMemory
from memoria.database.search_service import SearchService


@pytest.fixture
def search_service():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    now = datetime.utcnow()

    session.add_all(
        [
            ShortTermMemory(
                memory_id="short_source",
                processed_data={"text": "short source"},
                importance_score=0.5,
                category_primary="general",
                namespace="default",
                searchable_content="short source",
                summary="short source",
                created_at=now,
                x_coord=0.0,
                y_coord=0.0,
                z_coord=0.0,
            ),
            ShortTermMemory(
                memory_id="short_neighbor",
                processed_data={"text": "short neighbor"},
                importance_score=0.5,
                category_primary="general",
                namespace="default",
                searchable_content="short neighbor",
                summary="short neighbor",
                created_at=now,
                x_coord=0.5,
                y_coord=0.5,
                z_coord=0.0,
            ),
            ShortTermMemory(
                memory_id="short_far",
                processed_data={"text": "short far"},
                importance_score=0.5,
                category_primary="general",
                namespace="default",
                searchable_content="short far",
                summary="short far",
                created_at=now,
                x_coord=5.0,
                y_coord=0.0,
                z_coord=0.0,
            ),
            ShortTermMemory(
                memory_id="short_near_long",
                processed_data={"text": "short near long"},
                importance_score=0.5,
                category_primary="general",
                namespace="default",
                searchable_content="short near long",
                summary="short near long",
                created_at=now,
                x_coord=10.5,
                y_coord=0.0,
                z_coord=0.0,
            ),
            LongTermMemory(
                memory_id="long_source",
                processed_data={"text": "long source"},
                importance_score=0.5,
                category_primary="general",
                namespace="default",
                searchable_content="long source",
                summary="long source",
                created_at=now,
                timestamp=now,
                x_coord=10.0,
                y_coord=0.0,
                z_coord=0.0,
            ),
            LongTermMemory(
                memory_id="long_neighbor",
                processed_data={"text": "long neighbor"},
                importance_score=0.5,
                category_primary="general",
                namespace="default",
                searchable_content="long neighbor",
                summary="long neighbor",
                created_at=now,
                timestamp=now,
                x_coord=10.0,
                y_coord=0.5,
                z_coord=0.0,
            ),
            LongTermMemory(
                memory_id="long_far",
                processed_data={"text": "long far"},
                importance_score=0.5,
                category_primary="general",
                namespace="default",
                searchable_content="long far",
                summary="long far",
                created_at=now,
                timestamp=now,
                x_coord=-10.0,
                y_coord=-10.0,
                z_coord=-10.0,
            ),
            LongTermMemory(
                memory_id="long_near_short",
                processed_data={"text": "long near short"},
                importance_score=0.5,
                category_primary="general",
                namespace="default",
                searchable_content="long near short",
                summary="long near short",
                created_at=now,
                timestamp=now,
                x_coord=0.0,
                y_coord=1.0,
                z_coord=0.0,
            ),
        ]
    )

    session.commit()
    service = SearchService(session, "sqlite")
    try:
        yield service
    finally:
        session.close()
        engine.dispose()


def test_get_neighbors_short_term(search_service):
    neighbors = search_service.get_neighbors("short_source", 1.5)

    assert [n["memory_id"] for n in neighbors] == [
        "short_neighbor",
        "long_near_short",
    ]
    assert neighbors[0]["memory_type"] == "short_term"
    assert neighbors[1]["memory_type"] == "long_term"
    assert neighbors[0]["distance"] == pytest.approx((0.5**2 + 0.5**2) ** 0.5)
    assert neighbors[1]["distance"] == pytest.approx(1.0)


def test_get_neighbors_long_term(search_service):
    neighbors = search_service.get_neighbors("long_source", 1.0)

    assert len(neighbors) == 2
    assert [n["memory_type"] for n in neighbors] == ["short_term", "long_term"]
    assert {n["memory_id"] for n in neighbors} == {"short_near_long", "long_neighbor"}
    for neighbor in neighbors:
        assert neighbor["distance"] == pytest.approx(0.5)


def test_get_neighbors_excludes_source(search_service):
    neighbors = search_service.get_neighbors("short_source", 5.0)

    ids = {n["memory_id"] for n in neighbors}
    assert "short_source" not in ids
