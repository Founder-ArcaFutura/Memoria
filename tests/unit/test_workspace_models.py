import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from memoria.core.memory import Memoria
from memoria.database.models import Base, Workspace, WorkspaceMember
from memoria.utils.exceptions import MemoriaError


@pytest.fixture()
def db_session():
    """Provide an in-memory SQLAlchemy session for model assertions."""

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def test_workspace_member_relationships_and_cascade(db_session):
    workspace = Workspace(
        workspace_id="ws-1",
        name="Workspace One",
        slug="workspace-one",
    )
    member = WorkspaceMember(user_id="user-1", role="member")
    workspace.members.append(member)

    db_session.add(workspace)
    db_session.commit()

    assert workspace.members[0] is member
    assert member.workspace is workspace

    db_session.delete(workspace)
    db_session.commit()

    remaining = db_session.query(WorkspaceMember).all()
    assert remaining == []


def test_workspace_member_uniqueness_enforced(db_session):
    workspace = Workspace(
        workspace_id="ws-dup",
        name="Workspace Duplicate",
        slug="workspace-duplicate",
    )
    db_session.add(workspace)
    db_session.flush()

    db_session.add_all(
        [
            WorkspaceMember(workspace_id="ws-dup", user_id="user-dup"),
            WorkspaceMember(workspace_id="ws-dup", user_id="user-dup"),
        ]
    )

    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


@pytest.fixture()
def memoria_instance():
    mem = Memoria(
        database_connect="sqlite:///:memory:",
        team_memory_enabled=True,
        enable_short_term=False,
    )
    try:
        yield mem
    finally:
        if mem._ingestion_scheduler:
            mem._ingestion_scheduler.stop()
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()
        if getattr(mem, "db_manager", None) and getattr(mem.db_manager, "engine", None):
            mem.db_manager.engine.dispose()


def test_workspace_wrappers_return_workspace_payload(memoria_instance):
    payload = memoria_instance.register_workspace(
        "alpha",
        namespace="team-alpha",
        display_name="Alpha Workspace",
        members=["bob", "alice"],
        admins=["dave", "carol"],
        share_by_default=True,
        metadata={"tier": "gold"},
    )

    assert payload["workspace_id"] == "alpha"
    assert payload["namespace"] == "team-alpha"
    assert payload["members"] == ["alice", "bob"]
    assert payload["admins"] == ["carol", "dave"]
    assert "team_id" not in payload

    listing = memoria_instance.list_workspaces(include_members=True)
    assert len(listing) == 1
    assert listing[0]["workspace_id"] == "alpha"
    assert listing[0]["members"] == ["alice", "bob"]
    assert listing[0]["admins"] == ["carol", "dave"]
    assert "team_id" not in listing[0]

    fetched = memoria_instance.get_workspace("alpha", include_members=True)
    assert fetched["workspace_id"] == "alpha"
    assert fetched["members"] == ["alice", "bob"]
    assert fetched["admins"] == ["carol", "dave"]
    assert "team_id" not in fetched

    updated = memoria_instance.set_workspace_members(
        "alpha",
        members=["eve", "alice"],
        admins=["dave", "carol"],
        include_members=True,
    )
    assert updated["members"] == ["alice", "eve"]
    assert updated["admins"] == ["carol", "dave"]
    assert "team_id" not in updated

    removed = memoria_instance.remove_workspace_member(
        "alpha", "carol", include_members=True
    )
    assert removed["admins"] == ["dave"]
    assert removed["members"] == ["alice", "eve"]
    assert "team_id" not in removed

    after_update = memoria_instance.get_workspace("alpha", include_members=True)
    assert after_update["members"] == ["alice", "eve"]
    assert after_update["admins"] == ["dave"]
    assert "team_id" not in after_update


def test_workspace_wrapper_failures(memoria_instance):
    memoria_instance.register_workspace("alpha", members=["alice"], admins=["carol"])

    with pytest.raises(MemoriaError):
        memoria_instance.set_workspace_members("alpha", members=["alice", "alice"])

    with pytest.raises(MemoriaError):
        memoria_instance.get_workspace("unknown")

    with pytest.raises(MemoriaError):
        memoria_instance.remove_workspace_member("unknown", "ghost")
