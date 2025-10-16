"""End-to-end tests for the JSON/NDJSON export and import helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from memoria.database.models import (
    Base,
    ChatHistory,
    Cluster,
    ClusterMember,
    LinkMemoryThread,
    LongTermMemory,
    MemoryAccessEvent,
    ShortTermMemory,
    SpatialMetadata,
    ThreadEvent,
    ThreadMessageLink,
)
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager


def _create_manager(tmp_path: Path, name: str) -> SQLAlchemyDatabaseManager:
    db_url = f"sqlite:///{tmp_path / name}"
    manager = SQLAlchemyDatabaseManager(db_url)
    Base.metadata.create_all(manager.engine)
    return manager


def _seed_source(manager: SQLAlchemyDatabaseManager) -> None:
    timestamp = datetime.utcnow()
    with manager.SessionLocal() as session:
        chat = ChatHistory(
            chat_id="chat-1",
            user_input="hello",
            ai_output="hi there",
            model="gpt-test",
            timestamp=timestamp,
            session_id="session-1",
            namespace="default",
            tokens_used=42,
            metadata={"source": "unit-test"},
        )
        session.add(chat)

        short_term = ShortTermMemory(
            memory_id="stm-1",
            chat_id=chat.chat_id,
            processed_data={"text": "short"},
            importance_score=0.9,
            category_primary="greeting",
            retention_type="short_term",
            namespace="default",
            created_at=timestamp,
            expires_at=None,
            searchable_content="short content",
            summary="short summary",
            is_permanent_context=False,
            x_coord=0.1,
            y_coord=0.2,
            z_coord=0.3,
            symbolic_anchors=["hello"],
        )
        session.add(short_term)

        long_term = LongTermMemory(
            memory_id="ltm-1",
            original_chat_id=chat.chat_id,
            processed_data={"text": "long"},
            importance_score=0.8,
            category_primary="greeting",
            retention_type="long_term",
            namespace="default",
            timestamp=timestamp,
            created_at=timestamp,
            searchable_content="long content",
            summary="long summary",
            novelty_score=0.1,
            relevance_score=0.2,
            actionability_score=0.3,
            x_coord=0.4,
            y_coord=0.5,
            z_coord=0.6,
            symbolic_anchors=["memory"],
            classification="conversational",
            memory_importance="high",
            topic="testing",
            entities_json=["tester"],
            keywords_json=["unit"],
            is_user_context=True,
            is_preference=False,
            is_skill_knowledge=False,
            is_current_project=False,
            promotion_eligible=False,
            duplicate_of=None,
            supersedes_json=[],
            related_memories_json=[],
            confidence_score=0.99,
            extraction_timestamp=timestamp,
            classification_reason="seed",
            processed_for_duplicates=False,
            conscious_processed=False,
        )
        session.add(long_term)

        spatial = SpatialMetadata(
            memory_id=long_term.memory_id,
            namespace="default",
            timestamp=timestamp,
            x=0.4,
            y=0.5,
            z=0.6,
            symbolic_anchors=["memory"],
        )
        session.add(spatial)

        access = MemoryAccessEvent(
            memory_id=long_term.memory_id,
            namespace="default",
            accessed_at=timestamp,
            access_type="retrieval",
            source="unit",
            metadata_json={"count": 1},
        )
        session.add(access)

        link = LinkMemoryThread(
            source_memory_id=long_term.memory_id,
            target_memory_id=short_term.memory_id,
            relation="related",
            created_at=timestamp,
        )
        session.add(link)

        thread = ThreadEvent(
            thread_id="thread-1",
            namespace="default",
            symbolic_anchors=["conversation"],
            ritual_name="test",
            ritual_phase="start",
            ritual_metadata={"meta": True},
            centroid_x=0.1,
            centroid_y=0.2,
            centroid_z=0.3,
            created_at=timestamp,
            updated_at=timestamp,
        )
        session.add(thread)
        session.flush()

        thread_link = ThreadMessageLink(
            thread_id=thread.thread_id,
            memory_id=long_term.memory_id,
            namespace="default",
            sequence_index=1,
            role="assistant",
            anchor="hello",
            timestamp=timestamp,
        )
        session.add(thread_link)

        cluster = Cluster(
            summary="cluster summary",
            centroid={"x": 0.4},
            y_centroid=0.5,
            z_centroid=0.6,
            polarity=0.1,
            subjectivity=0.2,
            size=1,
            avg_importance=0.8,
            update_count=0,
            weight=0.5,
            total_tokens=10,
            total_chars=20,
        )
        session.add(cluster)
        session.flush()

        member = ClusterMember(
            cluster_id=cluster.id,
            memory_id=long_term.memory_id,
            anchor="hello",
            summary="cluster member",
            tokens=10,
            chars=20,
        )
        session.add(member)

        session.commit()


def _collect_counts(manager: SQLAlchemyDatabaseManager) -> dict[str, int]:
    with manager.SessionLocal() as session:
        return {
            "chat_history": session.query(ChatHistory).count(),
            "short_term_memory": session.query(ShortTermMemory).count(),
            "long_term_memory": session.query(LongTermMemory).count(),
            "spatial_metadata": session.query(SpatialMetadata).count(),
            "memory_access_events": session.query(MemoryAccessEvent).count(),
            "link_memory_threads": session.query(LinkMemoryThread).count(),
            "thread_events": session.query(ThreadEvent).count(),
            "thread_message_links": session.query(ThreadMessageLink).count(),
            "clusters": session.query(Cluster).count(),
            "cluster_members": session.query(ClusterMember).count(),
        }


def test_export_import_json(tmp_path: Path):
    source = _create_manager(tmp_path, "source.db")
    _seed_source(source)

    export_path = tmp_path / "backup.json"
    result = source.export_dataset(destination=export_path, format="json")
    assert export_path.exists()
    assert any(table["name"] == "chat_history" for table in result.metadata["tables"])

    target = _create_manager(tmp_path, "target.db")
    target.import_dataset(export_path)

    assert _collect_counts(source) == _collect_counts(target)

    with target.SessionLocal() as session:
        loaded = session.get(LongTermMemory, "ltm-1")
        assert loaded is not None
        assert loaded.summary == "long summary"


def test_export_import_ndjson(tmp_path: Path):
    source = _create_manager(tmp_path, "source_nd.db")
    _seed_source(source)

    result = source.export_dataset(format="ndjson")
    target = _create_manager(tmp_path, "target_nd.db")
    metadata = target.import_dataset(result.content, format="ndjson")

    assert metadata.get("format") == "ndjson"
    assert _collect_counts(source) == _collect_counts(target)


def test_reimport_overwrites_changes(tmp_path: Path):
    source = _create_manager(tmp_path, "source_overwrite.db")
    _seed_source(source)
    export_path = tmp_path / "backup_overwrite.ndjson"
    source.export_dataset(destination=export_path, format="ndjson")

    target = _create_manager(tmp_path, "target_overwrite.db")
    target.import_dataset(export_path)

    with target.SessionLocal() as session:
        record = session.get(ChatHistory, "chat-1")
        record.user_input = "mutated"
        session.commit()

    target.import_dataset(export_path, truncate=True)

    with target.SessionLocal() as session:
        record = session.get(ChatHistory, "chat-1")
        assert record.user_input == "hello"


@pytest.mark.parametrize("format", ["json", "ndjson"])
def test_import_subset_tables(tmp_path: Path, format: str):
    source = _create_manager(tmp_path, f"subset_src_{format}.db")
    _seed_source(source)
    result = source.export_dataset(format=format)

    target = _create_manager(tmp_path, f"subset_dest_{format}.db")
    target.import_dataset(
        result.content, format=format, tables=["chat_history", "long_term_memory"]
    )

    counts = _collect_counts(target)
    assert counts["chat_history"] == 1
    assert counts["long_term_memory"] == 1
    assert counts["short_term_memory"] == 0
