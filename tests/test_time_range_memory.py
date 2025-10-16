import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memoria import Memoria
from memoria.database.models import LongTermMemory


def test_retrieve_memories_by_time_range():
    mem = Memoria(database_connect="sqlite:///:memory:")
    now = datetime.datetime.utcnow()
    with mem.db_manager.SessionLocal() as session:
        m1 = LongTermMemory(
            memory_id="m1",
            processed_data={"text": "old"},
            importance_score=0.5,
            category_primary="test",
            namespace="default",
            created_at=now - datetime.timedelta(days=3),
            searchable_content="old",
            summary="old",
            x_coord=-3.0,
        )
        m2 = LongTermMemory(
            memory_id="m2",
            processed_data={"text": "new"},
            importance_score=0.5,
            category_primary="test",
            namespace="default",
            created_at=now - datetime.timedelta(days=1),
            searchable_content="new",
            summary="new",
            x_coord=-1.0,
        )
        m3 = LongTermMemory(
            memory_id="m3",
            processed_data={"text": "fractional"},
            importance_score=0.5,
            category_primary="test",
            namespace="default",
            created_at=now - datetime.timedelta(days=2),
            searchable_content="fractional",
            summary="fractional",
            x_coord=-2.4,
        )
        session.add_all([m1, m2, m3])
        session.commit()
    recent = mem.retrieve_memories_by_time_range(
        start_timestamp=now - datetime.timedelta(days=1, hours=12),
        end_timestamp=now,
    )
    assert len(recent) == 1
    assert recent[0]["text"] == "new"
    by_x = mem.retrieve_memories_by_time_range(start_x=-3, end_x=-2.6)
    assert len(by_x) == 1
    assert by_x[0]["text"] == "old"

    fractional = mem.retrieve_memories_by_time_range(start_x=-2.6, end_x=-2.2)
    assert [mem_record["text"] for mem_record in fractional] == ["fractional"]
