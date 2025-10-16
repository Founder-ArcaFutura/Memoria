import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from memoria.core.memory import Memoria
from memoria.database.models import LongTermMemory, ShortTermMemory


@pytest.fixture
def tmp_path_factory(tmp_path_factory):
    return tmp_path_factory


def _create_memoria(tmp_path_factory=None, **kwargs) -> Memoria:
    if tmp_path_factory:
        db_path = tmp_path_factory.mktemp("data") / "test.db"
        mem = Memoria(database_connect=f"sqlite:///{db_path}", **kwargs)
    else:
        mem = Memoria(database_connect="sqlite:///:memory:", **kwargs)
    mem._enabled = True
    return mem


def test_sovereign_mode_stages_without_promotion():
    mem = _create_memoria(sovereign_ingest=True)

    chat_id = mem.record_conversation(
        user_input="User mentions a fleeting idea",
        ai_output="Assistant acknowledges",
        model="test-model",
        metadata={
            "timestamp": datetime.utcnow().isoformat(),
            "summary": "User shared a fleeting idea",
            "tokens_used": 8,
        },
    )

    assert chat_id

    with mem.db_manager.SessionLocal() as session:
        long_term = session.query(LongTermMemory).all()
        short_term = session.query(ShortTermMemory).all()

    assert len(long_term) == 0
    assert len(short_term) == 1


def test_heuristic_processing_promotes_when_threshold_met(tmp_path_factory):
    mem = _create_memoria(tmp_path_factory=tmp_path_factory)

    chat_id = "heuristic-chat"
    metadata = {
        "anchor": "tea_appreciation",
        "symbolic_anchors": ["tea", "preference"],
        "summary": "User loves oolong tea",
        "x_coord": 0.0,
        "y_coord": 10.0,
        "z_coord": 5.0,
        "user_priority": 1.0,
        "tokens_used": 6,
    }

    result = mem.process_recorded_conversation_heuristic(
        chat_id,
        "I really enjoy oolong tea",
        "That's great to know!",
        model="test-model",
        metadata=metadata,
    )

    assert result["promoted"] is True
    assert result["long_term_id"]

    with mem.db_manager.SessionLocal() as session:
        long_term = session.query(LongTermMemory).all()
        short_term = session.query(ShortTermMemory).all()

    assert len(long_term) == 1
    assert len(short_term) == 0
    assert long_term[0].summary == "User loves oolong tea"
