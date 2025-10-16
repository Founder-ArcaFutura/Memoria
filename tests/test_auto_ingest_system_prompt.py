import datetime
import os
import sys
import uuid

import pytest

# Ensure the memoria package is importable when tests are executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memoria.core.memory import Memoria
from memoria.database.models import LongTermMemory


@pytest.fixture
def mem_with_memories():
    """Create Memoria instance populated with sample memories."""
    mem = Memoria(database_connect="sqlite:///:memory:")

    # Insert sample memories directly into the database with essential categories
    with mem.db_manager.SessionLocal() as session:
        now = datetime.datetime.utcnow()
        pizza = LongTermMemory(
            memory_id=str(uuid.uuid4()),
            processed_data={"text": "User loves Hawaiian pizza"},
            importance_score=0.5,
            category_primary="essential_preference",
            retention_type="long_term",
            namespace="default",
            created_at=now,
            searchable_content="User loves Hawaiian pizza",
            summary="User loves Hawaiian pizza",
        )
        code = LongTermMemory(
            memory_id=str(uuid.uuid4()),
            processed_data={"text": "Favorite programming language is Python"},
            importance_score=0.5,
            category_primary="essential_fact",
            retention_type="long_term",
            namespace="default",
            created_at=now,
            searchable_content="Favorite programming language is Python",
            summary="Favorite programming language is Python",
        )
        session.add_all([pizza, code])
        session.commit()
    return mem


@pytest.mark.parametrize(
    "query,expected,unexpected",
    [
        ("pizza", "Hawaiian pizza", "Python"),
        ("Python", "Python", "Hawaiian pizza"),
    ],
)
def test_auto_ingest_prompt_returns_relevant_memories(
    mem_with_memories, query, expected, unexpected
):
    prompt = mem_with_memories.get_auto_ingest_system_prompt(query)
    assert expected in prompt
    assert unexpected not in prompt
