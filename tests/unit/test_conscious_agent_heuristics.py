import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.agents.conscious_agent import ConsciousAgent
from memoria.conscious.constants import CONSCIOUS_CONTEXT_CATEGORY
from memoria.core.memory import Memoria
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager


def test_sqlalchemy_manager_get_connection_context_manager(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'compat.db'}"
    manager = SQLAlchemyDatabaseManager(db_url)
    manager.initialize_schema()

    with manager.get_connection() as connection:
        result = connection.execute(text("SELECT 1")).scalar()

    assert result == 1


def _insert_long_term_memory(connection, **kwargs):
    kwargs = dict(kwargs)
    kwargs.setdefault("timestamp", kwargs.get("created_at", datetime.utcnow()))
    kwargs.setdefault("memory_importance", "medium")
    kwargs.setdefault("extraction_timestamp", kwargs["timestamp"])
    connection.execute(
        text(
            """INSERT INTO long_term_memory (
            memory_id, processed_data, summary, searchable_content,
            importance_score, category_primary, retention_type,
            namespace, classification, conscious_processed, created_at, timestamp, memory_importance, extraction_timestamp
        ) VALUES (
            :memory_id, :processed_data, :summary, :searchable_content,
            :importance_score, :category_primary, :retention_type,
            :namespace, :classification, :conscious_processed, :created_at, :timestamp, :memory_importance, :extraction_timestamp
        )"""
        ),
        kwargs,
    )


def test_lightweight_heuristics_filters_memories(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'heuristics.db'}"
    manager = SQLAlchemyDatabaseManager(db_url, enable_short_term=True)
    manager.initialize_schema()

    agent = ConsciousAgent(use_heuristics=True)
    namespace = "heuristic"
    now = datetime.utcnow()

    base_content = (
        "Alpha project update: delivered final module and awaiting QA review for "
        "deployment approval."
    )

    with manager.get_connection() as connection:
        _insert_long_term_memory(
            connection,
            memory_id="base-001",
            processed_data=json.dumps({"text": base_content}),
            summary="Alpha project delivery update with QA pending.",
            searchable_content=base_content,
            importance_score=0.5,
            category_primary="project",
            retention_type="long_term",
            namespace=namespace,
            classification="project-log",
            conscious_processed=False,
            created_at=now,
        )

        good_summary = "Alpha project timeline update covering final module delivery and upcoming QA checkpoint."
        good_content = (
            f"{base_content} Additional insights: QA kickoff scheduled for next Tuesday "
            "with dedicated compliance support and customer readiness tasks."
        )
        _insert_long_term_memory(
            connection,
            memory_id="conscious-good",
            processed_data=json.dumps({"text": good_content}),
            summary=good_summary,
            searchable_content=good_content,
            importance_score=0.9,
            category_primary="conscious",
            retention_type="long_term",
            namespace=namespace,
            classification="conscious-info",
            conscious_processed=False,
            created_at=now - timedelta(minutes=10),
        )

        _insert_long_term_memory(
            connection,
            memory_id="conscious-short",
            processed_data=json.dumps({"text": "Short"}),
            summary="Too short",
            searchable_content="Short",
            importance_score=0.4,
            category_primary="conscious",
            retention_type="long_term",
            namespace=namespace,
            classification="conscious-info",
            conscious_processed=False,
            created_at=now - timedelta(minutes=5),
        )

        stale_content = "Alpha project compliance review outcome and archival notice for the legacy module."
        _insert_long_term_memory(
            connection,
            memory_id="conscious-old",
            processed_data=json.dumps({"text": stale_content}),
            summary="Alpha project compliance review with archival notice for prior work.",
            searchable_content=stale_content,
            importance_score=0.7,
            category_primary="conscious",
            retention_type="long_term",
            namespace=namespace,
            classification="conscious-info",
            conscious_processed=False,
            created_at=now - timedelta(days=3),
        )

        connection.commit()

    copied = asyncio.run(agent.run_conscious_ingest(manager, namespace))
    assert copied is True

    with manager.get_connection() as connection:
        rows = connection.execute(
            text(
                """SELECT summary, searchable_content FROM short_term_memory
                WHERE namespace = :namespace AND category_primary = :category"""
            ),
            {"namespace": namespace, "category": CONSCIOUS_CONTEXT_CATEGORY},
        ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == good_summary
    assert good_content in rows[0][1]


def test_initialize_existing_respects_lightweight_heuristics(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'init.db'}"
    manager = SQLAlchemyDatabaseManager(db_url, enable_short_term=True)
    manager.initialize_schema()

    agent = ConsciousAgent(use_heuristics=True)
    namespace = "init"
    now = datetime.utcnow()

    base_content = "Client onboarding journal: captured initial goals and scheduled training session."

    with manager.get_connection() as connection:
        _insert_long_term_memory(
            connection,
            memory_id="base-ctx",
            processed_data=json.dumps({"text": base_content}),
            summary="Client onboarding journal entry with goals and training schedule.",
            searchable_content=base_content,
            importance_score=0.6,
            category_primary="journal",
            retention_type="long_term",
            namespace=namespace,
            classification="journal",
            conscious_processed=False,
            created_at=now,
        )

        good_summary = "Client onboarding preferences expanding on training cadence and motivation cues."
        good_content = (
            f"{base_content} Extended notes include weekly motivation prompts and "
            "preferred resources for sustained engagement."
        )
        _insert_long_term_memory(
            connection,
            memory_id="conscious-pass",
            processed_data=json.dumps({"text": good_content}),
            summary=good_summary,
            searchable_content=good_content,
            importance_score=0.85,
            category_primary="conscious",
            retention_type="long_term",
            namespace=namespace,
            classification="conscious-info",
            conscious_processed=False,
            created_at=now - timedelta(minutes=20),
        )

        _insert_long_term_memory(
            connection,
            memory_id="conscious-mismatch",
            processed_data=json.dumps({"text": "Mismatch memory"}),
            summary="Mismatch memory",
            searchable_content="Mismatch memory",
            importance_score=0.3,
            category_primary="conscious",
            retention_type="long_term",
            namespace=namespace,
            classification="conscious-info",
            conscious_processed=False,
            created_at=now - timedelta(minutes=15),
        )

        connection.commit()

    initialized = asyncio.run(
        agent.initialize_existing_conscious_memories(manager, namespace)
    )
    assert initialized is True

    with manager.get_connection() as connection:
        rows = connection.execute(
            text(
                """SELECT summary FROM short_term_memory
                WHERE namespace = :namespace AND category_primary = :category"""
            ),
            {"namespace": namespace, "category": CONSCIOUS_CONTEXT_CATEGORY},
        ).fetchall()

    assert [row[0] for row in rows] == [good_summary]


def test_memoria_threads_lightweight_flag(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'memoria.db'}"
    memoria = Memoria(
        database_connect=db_url,
        template="basic",
        conscious_ingest=False,
        auto_ingest=True,
        use_lightweight_conscious_ingest=True,
        enable_short_term=True,
        namespace="toggle",
        openai_api_key="test-key",
    )

    try:
        assert memoria.use_lightweight_conscious_ingest is True
        assert memoria.conscious_agent is not None
        assert memoria.conscious_agent.use_heuristics is True
    finally:
        memoria.conscious_manager.stop()
        memoria.db_manager.engine.dispose()
