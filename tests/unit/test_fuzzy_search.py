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

    st = ShortTermMemory(
        memory_id="st1",
        processed_data={"text": "apples are tasty"},
        importance_score=0.5,
        category_primary="general",
        namespace="default",
        searchable_content="apples are tasty",
        summary="apples",
        created_at=datetime.utcnow(),
        x_coord=1.0,
        y_coord=2.0,
        z_coord=3.0,
    )
    lt = LongTermMemory(
        memory_id="lt1",
        processed_data={"text": "bananas are yellow"},
        importance_score=0.5,
        category_primary="general",
        namespace="default",
        searchable_content="bananas are yellow",
        summary="bananas",
        created_at=datetime.utcnow(),
        x_coord=4.0,
        y_coord=5.0,
        z_coord=6.0,
    )
    session.add_all([st, lt])
    session.commit()
    return session


def test_fuzzy_search_matches_misspelled_queries():
    session = _setup_session()
    service = SearchService(session, "sqlite")

    short_results = service.search_memories(
        query="appls",
        namespace="default",
        limit=5,
        use_fuzzy=True,
        fuzzy_min_similarity=30,
    )["results"]
    long_results = service.search_memories(
        query="bananna",
        namespace="default",
        limit=5,
        use_fuzzy=True,
        fuzzy_min_similarity=30,
    )["results"]

    st_entry = next(r for r in short_results if r["memory_id"] == "st1")
    lt_entry = next(r for r in long_results if r["memory_id"] == "lt1")
    assert st_entry["x"] == 1.0 and st_entry["y"] == 2.0 and st_entry["z"] == 3.0
    assert lt_entry["x"] == 4.0 and lt_entry["y"] == 5.0 and lt_entry["z"] == 6.0
    session.close()


def test_adaptive_min_similarity():
    session = _setup_session()
    service = SearchService(session, "sqlite")

    query = "apples are somewhat flavorful tasty fruit"

    adaptive_results = service.search_memories(
        query=query,
        namespace="default",
        limit=5,
        use_fuzzy=True,
        fuzzy_min_similarity=60,
    )["results"]

    fixed_results = service.search_memories(
        query=query,
        namespace="default",
        limit=5,
        use_fuzzy=True,
        fuzzy_min_similarity=60,
        adaptive_min_similarity=False,
    )["results"]

    assert any(r["memory_id"] == "st1" for r in adaptive_results)
    assert not any(r["memory_id"] == "st1" for r in fixed_results)

    session.close()
