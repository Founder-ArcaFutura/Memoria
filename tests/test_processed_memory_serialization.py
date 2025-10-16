"""Regression tests for JSON-safe persistence of processed memories."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from memoria.core.database import DatabaseManager
from memoria.database.models import LongTermMemory, ShortTermMemory
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager
from memoria.utils.pydantic_models import (
    MemoryClassification,
    MemoryImportanceLevel,
    ProcessedLongTermMemory,
)


def _build_processed_memory() -> ProcessedLongTermMemory:
    """Construct a processed memory containing timezone-aware datetimes."""

    now = datetime.now(timezone.utc)
    return ProcessedLongTermMemory(
        content="Memoria learns that datetime serialization must be robust.",
        summary="Datetime serialization regression case",
        classification=MemoryClassification.ESSENTIAL,
        importance=MemoryImportanceLevel.HIGH,
        topic="serialization",
        entities=["Memoria"],
        keywords=["datetime", "json"],
        is_user_context=False,
        conversation_id="conv-json-safe",
        classification_reason="Regression coverage for JSON-safe persistence",
        extraction_timestamp=now,
        last_accessed=now,
        x_coord=0.1,
        y_coord=-0.2,
        z_coord=0.3,
        symbolic_anchors=["regression"],
        promotion_eligible=True,
    )


def test_sqlalchemy_manager_stores_json_safe_payload(tmp_path) -> None:
    """Ensure SQLAlchemy-backed persistence handles datetime fields."""

    db_path = tmp_path / "sqlalchemy-json-safe.db"
    manager = SQLAlchemyDatabaseManager(f"sqlite:///{db_path}", enable_short_term=True)
    manager.initialize_schema()

    memory = _build_processed_memory()

    long_id = manager.store_long_term_memory_enhanced(memory, chat_id="chat-long")
    short_id = manager.store_short_term_memory(memory, chat_id="chat-short")

    with manager.SessionLocal() as session:
        stored_long = session.query(LongTermMemory).filter_by(memory_id=long_id).one()
        stored_short = (
            session.query(ShortTermMemory).filter_by(memory_id=short_id).one()
        )

    assert isinstance(stored_long.processed_data["extraction_timestamp"], str)
    assert isinstance(stored_long.processed_data["last_accessed"], str)
    assert isinstance(stored_short.processed_data["extraction_timestamp"], str)
    assert isinstance(stored_short.processed_data["last_accessed"], str)


def test_transactional_fallback_stores_json_safe_payload(tmp_path) -> None:
    """Ensure the transactional SQLite fallback stores JSON-safe payloads."""

    db_path = tmp_path / "fallback-json-safe.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")
    manager.initialize_schema()

    memory = _build_processed_memory()
    memory_id = manager.store_long_term_memory_enhanced(memory, chat_id="chat-fallback")

    with sqlite3.connect(str(db_path)) as connection:
        cursor = connection.execute(
            "SELECT processed_data FROM long_term_memory WHERE memory_id = ?",
            (memory_id,),
        )
        stored_payload = cursor.fetchone()[0]

    payload_dict = json.loads(stored_payload)
    assert isinstance(payload_dict["extraction_timestamp"], str)
    assert isinstance(payload_dict["last_accessed"], str)


def test_schema_fallback_handles_missing_template(tmp_path) -> None:
    """The basic fallback schema should support enhanced long-term inserts."""

    db_path = tmp_path / "missing-template-fallback.db"
    manager = DatabaseManager(f"sqlite:///{db_path}", template="nonexistent-template")

    manager.get_connection = manager._get_connection  # type: ignore[attr-defined]
    manager.initialize_schema()

    memory = _build_processed_memory()
    memory_id = manager.store_long_term_memory_enhanced(
        memory, chat_id="chat-missing-template"
    )

    with sqlite3.connect(str(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT
                classification,
                memory_importance,
                timestamp,
                x_coord,
                y_coord,
                z_coord,
                symbolic_anchors,
                extraction_timestamp
            FROM long_term_memory
            WHERE memory_id = ?
            """,
            (memory_id,),
        ).fetchone()

    assert row is not None
    assert row["classification"] == memory.classification.value
    assert row["memory_importance"] == memory.importance.value
    assert row["timestamp"]
    assert row["x_coord"] == memory.x_coord
    assert row["y_coord"] == memory.y_coord
    assert row["z_coord"] == memory.z_coord
    assert json.loads(row["symbolic_anchors"]) == memory.symbolic_anchors
    assert row["extraction_timestamp"] == memory.extraction_timestamp.isoformat()
