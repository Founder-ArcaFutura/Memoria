import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria import Memoria
from memoria.database.models import LongTermMemory


def _create_memory_manager_with_data():
    mem = Memoria(database_connect="sqlite:///:memory:", enable_short_term=False)

    high_work_id = mem.store_memory(
        anchor="project_alpha",
        text="Discussed alpha project roadmap and deliverables",
        tokens=5,
        x_coord=0.0,
        y=0.0,
        z=0.0,
    )
    low_work_id = mem.store_memory(
        anchor="project_beta",
        text="Reviewed beta project tasks",
        tokens=5,
        x_coord=1.0,
        y=0.0,
        z=0.0,
    )
    personal_id = mem.store_memory(
        anchor="personal_note",
        text="Planned weekend hiking trip",
        tokens=5,
        x_coord=0.0,
        y=10.0,
        z=0.0,
    )

    with mem.db_manager.SessionLocal() as session:
        session.query(LongTermMemory).filter_by(memory_id=high_work_id).update(
            {"category_primary": "work", "importance_score": 0.85}
        )
        session.query(LongTermMemory).filter_by(memory_id=low_work_id).update(
            {"category_primary": "work", "importance_score": 0.45}
        )
        session.query(LongTermMemory).filter_by(memory_id=personal_id).update(
            {"category_primary": "personal", "importance_score": 0.9}
        )
        session.commit()

    return mem.memory_manager, {
        "high_work": high_work_id,
        "low_work": low_work_id,
        "personal": personal_id,
    }


def test_memory_manager_search_returns_normalized_results():
    manager, ids = _create_memory_manager_with_data()

    results = manager.search_memories("project", limit=5)

    assert isinstance(results, list)
    assert ids["high_work"] in {item["memory_id"] for item in results}
    for item in results:
        assert isinstance(item, dict)
        assert "memory_id" in item


def test_memory_manager_search_honors_filters():
    manager, ids = _create_memory_manager_with_data()

    filtered = manager.search_memories(
        "project",
        limit=5,
        memory_types=["long_term"],
        categories=["work"],
        min_importance=0.6,
    )

    assert filtered
    filtered_ids = {item["memory_id"] for item in filtered}
    assert filtered_ids == {ids["high_work"]}

    only_short_term = manager.search_memories(
        "project", memory_types=["short_term"], categories=["work"]
    )

    assert only_short_term == []
