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

    anchor_mem = ShortTermMemory(
        memory_id="anchor1",
        processed_data={"text": "foo"},
        importance_score=0.5,
        category_primary="general",
        namespace="default",
        searchable_content="foo",
        summary="foo",
        created_at=datetime.utcnow(),
        symbolic_anchors=["Alpha"],
    )
    text_mem = ShortTermMemory(
        memory_id="text1",
        processed_data={"text": "Alpha mentioned"},
        importance_score=0.5,
        category_primary="general",
        namespace="default",
        searchable_content="Alpha mentioned",
        summary="Alpha mentioned",
        created_at=datetime.utcnow(),
    )
    session.add_all([anchor_mem, text_mem])
    session.commit()
    return session


def test_anchor_lookup_branch():
    session = _setup_session()
    service = SearchService(session, "sqlite")

    results = service.search_memories(
        "Alpha", namespace="default", limit=1, use_anchor=True
    )["results"]
    assert results and results[0]["memory_id"] == "anchor1"
    assert results[0]["search_strategy"] == "symbolic_anchor"

    no_anchor = service.search_memories(
        "Alpha", namespace="default", limit=1, use_anchor=False
    )["results"]
    assert no_anchor and no_anchor[0]["search_strategy"] != "symbolic_anchor"
    session.close()


def test_anchor_search_ignores_namespace():
    session = _setup_session()
    session.add(
        ShortTermMemory(
            memory_id="anchor_other",
            processed_data={"text": "foo other"},
            importance_score=0.5,
            category_primary="general",
            namespace="other",
            searchable_content="foo other",
            summary="foo other",
            created_at=datetime.utcnow(),
            symbolic_anchors=["Alpha"],
        )
    )
    session.commit()

    service = SearchService(session, "sqlite")
    results = service.search_memories(
        "Alpha", namespace="default", limit=5, use_anchor=True
    )["results"]

    memory_ids = {item["memory_id"] for item in results}
    assert "anchor1" in memory_ids
    assert "anchor_other" in memory_ids

    session.close()
