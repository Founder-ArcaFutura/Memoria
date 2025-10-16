import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memoria import Memoria
from memoria.database.models import LongTermMemory


def test_recompute_x_coord_from_timestamp(tmp_path: Path):
    db_path = tmp_path / "test.db"
    mem = Memoria(database_connect=f"sqlite:///{db_path}")
    ts = datetime.datetime.utcnow() - datetime.timedelta(days=2)
    result = mem.store_memory(
        anchor="a",
        text="t",
        tokens=1,
        timestamp=ts,
        return_status=True,
    )
    memory_id = result["memory_id"]
    expected = float((ts.date() - datetime.datetime.utcnow().date()).days)
    with mem.storage_service.db_manager.SessionLocal() as session:
        row = session.query(LongTermMemory).filter_by(memory_id=memory_id).first()
        row.x_coord = None
        session.commit()
    new_x = mem.storage_service.recompute_x_coord_from_timestamp(memory_id)
    assert new_x == expected
