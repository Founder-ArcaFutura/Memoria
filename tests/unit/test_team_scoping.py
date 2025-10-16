from datetime import datetime

from memoria.database.models import ChatHistory, LongTermMemory
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager
from memoria.storage.service import StorageService


def _create_manager(tmp_path) -> SQLAlchemyDatabaseManager:
    db_path = tmp_path / "team_scoping.sqlite"
    manager = SQLAlchemyDatabaseManager(f"sqlite:///{db_path}")
    manager.initialize_schema()
    return manager


def test_database_manager_team_filtered_history(tmp_path):
    manager = _create_manager(tmp_path)
    timestamp = datetime.utcnow()

    manager.store_chat_history(
        chat_id="chat-team-a",
        user_input="Hi A",
        ai_output="Hello A",
        timestamp=timestamp,
        session_id="session-a",
        model="test-model",
        namespace="alpha",
        team_id="team-a",
    )
    manager.store_chat_history(
        chat_id="chat-team-b",
        user_input="Hi B",
        ai_output="Hello B",
        timestamp=timestamp,
        session_id="session-b",
        model="test-model",
        namespace="alpha",
        team_id="team-b",
    )
    manager.store_chat_history(
        chat_id="chat-legacy",
        user_input="Legacy",
        ai_output="Legacy",
        timestamp=timestamp,
        session_id="session-legacy",
        model="test-model",
        namespace="alpha",
    )

    team_a_history = manager.get_chat_history(namespace="alpha", team_id="team-a")
    assert {item["chat_id"] for item in team_a_history} == {"chat-team-a"}

    namespace_history = manager.get_chat_history(namespace="alpha")
    assert {item["chat_id"] for item in namespace_history} == {
        "chat-team-a",
        "chat-team-b",
        "chat-legacy",
    }

    with manager.SessionLocal() as session:
        stored = (
            session.query(ChatHistory)
            .filter(ChatHistory.chat_id == "chat-team-a")
            .one()
        )
        assert stored.team_id == "team-a"


def test_storage_service_team_scoped_store_and_fetch(tmp_path):
    manager = _create_manager(tmp_path)
    service_team_a = StorageService(
        db_manager=manager,
        namespace="alpha",
        team_id="team-a",
    )

    memory_id = service_team_a.store_memory(
        anchor="project",
        text="Team scoped entry",
        tokens=5,
    )

    with manager.SessionLocal() as session:
        long_term = (
            session.query(LongTermMemory)
            .filter(LongTermMemory.memory_id == memory_id)
            .one()
        )
        assert long_term.team_id == "team-a"

        chat = (
            session.query(ChatHistory)
            .filter(ChatHistory.chat_id == long_term.original_chat_id)
            .one()
        )
        assert chat.team_id == "team-a"

    history_team_a = service_team_a.get_conversation_history(
        session_id=chat.session_id,
        shared_memory=False,
        limit=5,
    )
    assert [entry["chat_id"] for entry in history_team_a] == [chat.chat_id]

    service_team_b = StorageService(
        db_manager=manager,
        namespace="alpha",
        team_id="team-b",
    )
    history_team_b = service_team_b.get_conversation_history(
        session_id=chat.session_id,
        shared_memory=False,
        limit=5,
    )
    assert history_team_b == []

    legacy_service = StorageService(db_manager=manager, namespace="alpha")
    legacy_memory = legacy_service.store_memory(
        anchor="legacy",
        text="Legacy namespace entry",
        tokens=4,
    )

    with manager.SessionLocal() as session:
        legacy_record = (
            session.query(LongTermMemory)
            .filter(LongTermMemory.memory_id == legacy_memory)
            .one()
        )
        assert legacy_record.team_id is None


def test_storage_service_cache_is_team_scoped(tmp_path):
    manager = _create_manager(tmp_path)
    service = StorageService(
        db_manager=manager,
        namespace="team-alpha",
        team_id="team-a",
    )

    memory_id = service.store_memory(
        anchor="team-a-anchor",
        text="Team A cached memory",
        tokens=5,
    )

    snapshot = service.get_memory_snapshot(memory_id)
    assert snapshot and snapshot["memory_id"] == memory_id

    service.namespace = "team-bravo"
    service.team_id = "team-b"

    assert service.get_memory_snapshot(memory_id) is None

    service.namespace = "team-alpha"
    service.team_id = "team-a"

    thread_id = "shared-thread"
    service.store_thread(
        thread_id,
        message_links=[{"memory_id": memory_id, "sequence_index": 0}],
    )

    cached_thread = service.get_thread(thread_id)
    assert cached_thread and cached_thread["thread_id"] == thread_id

    service.namespace = "team-bravo"
    service.team_id = "team-b"

    assert service.get_thread(thread_id) is None
