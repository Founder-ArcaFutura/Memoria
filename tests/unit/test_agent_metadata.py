from pathlib import Path

import pytest

from memoria.core.memory import Memoria
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager
from memoria.database.models import ChatHistory
from memoria.storage.service import StorageService


@pytest.fixture()
def sqlite_db(tmp_path: Path) -> str:
    db_url = f"sqlite:///{tmp_path/'agents.sqlite'}"
    manager = SQLAlchemyDatabaseManager(db_url)
    manager.initialize_schema()
    return db_url


def test_agent_registration_and_lookup(sqlite_db: str) -> None:
    manager = SQLAlchemyDatabaseManager(sqlite_db)
    profile = manager.register_agent(
        "agent-primary",
        name="Primary Agent",
        role="orchestrator",
        preferred_model="gpt-4o",
        metadata={"custom": True},
    )
    assert profile["preferred_model"] == "gpt-4o"
    fetched = manager.get_agent("agent-primary")
    assert fetched and fetched["name"] == "Primary Agent"

    storage = StorageService(db_manager=manager, namespace="alpha")
    cached = storage.get_agent("agent-primary")
    assert cached and cached["role"] == "orchestrator"

    updated = storage.register_agent(
        "agent-primary",
        name="Primary Agent",
        role="planner",
        preferred_model="gpt-4o-mini",
    )
    assert updated["role"] == "planner"
    assert storage.get_agent("agent-primary")["preferred_model"] == "gpt-4o-mini"


def test_team_space_tracks_agents(sqlite_db: str) -> None:
    manager = SQLAlchemyDatabaseManager(sqlite_db)
    manager.register_agent("agent-shared", name="Shared Agent", preferred_model="gpt-4")
    storage = StorageService(db_manager=manager, namespace="alpha")
    team = storage.register_team_space(
        "team-alpha",
        members=["human-user"],
        agents=["agent-shared"],
        metadata={"purpose": "demo"},
    )
    payload = team.to_dict()
    assert "agent-shared" in payload.get("agents", [])
    assert storage.user_has_team_access("team-alpha", "agent-shared")


def test_memoria_uses_agent_preferred_model(sqlite_db: str) -> None:
    manager = SQLAlchemyDatabaseManager(sqlite_db)
    manager.register_agent("agent-model", name="Model Agent", preferred_model="gpt-test")
    memoria = Memoria(database_connect=sqlite_db, agent_id="agent-model", user_id="user-1")
    memoria.enable()
    assert memoria.default_model == "gpt-test"
    assert memoria.model == "gpt-test"

    chat_id = memoria.record_conversation("hello", {"content": "world", "model": "gpt-test"})
    with memoria.db_manager.SessionLocal() as session:
        record = session.query(ChatHistory).filter(ChatHistory.chat_id == chat_id).one()
        assert record.last_edited_by_model == "gpt-test"

    memoria.disable()
