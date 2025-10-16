import sys
import threading
from pathlib import Path

import pytest
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memoria import Memoria
from memoria.database.auto_creator import DatabaseAutoCreator
from memoria.database.models import LongTermMemory
from memoria.database.query_translator import QueryParameterTranslator
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager


@pytest.fixture
def sample_client(monkeypatch) -> tuple:
    """Create a test client with deterministic in-memory data."""

    class DaemonTimer(threading.Timer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.daemon = True

    monkeypatch.setattr(threading, "Timer", DaemonTimer)

    mem = Memoria(database_connect="sqlite:///:memory:", enable_short_term=False)
    mem.enable()

    id_work_recent = mem.store_memory(
        anchor="a",
        text="alpha project update",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
    )
    id_personal = mem.store_memory(
        anchor="a",
        text="alpha hobby note",
        tokens=1,
        x_coord=0.0,
        y=10.0,
        z=0.0,
    )
    id_work_old = mem.store_memory(
        anchor="a",
        text="alpha project old",
        tokens=1,
        x_coord=10.0,
        y=0.0,
        z=0.0,
    )

    with mem.db_manager.SessionLocal() as session:
        session.query(LongTermMemory).filter_by(memory_id=id_work_recent).update(
            {"category_primary": "work"}
        )
        session.query(LongTermMemory).filter_by(memory_id=id_personal).update(
            {"category_primary": "personal"}
        )
        session.query(LongTermMemory).filter_by(memory_id=id_work_old).update(
            {"category_primary": "work"}
        )
        session.commit()

    # The original `memoria_api` module has been replaced by an app factory.
    # We now create a new app instance for each test and inject our
    # test-specific, pre-populated Memoria instance into it.
    from memoria_server.api.app_factory import create_app

    # Set the DATABASE_URL to in-memory to prevent file-based DB creation during tests
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    # Disable UI to avoid potential issues with static file paths in tests
    monkeypatch.setenv("MEMORIA_SERVE_UI", "false")

    app = create_app()
    app.config["memoria"] = mem
    client = app.test_client()
    ids: dict[str, str] = {
        "work_recent": id_work_recent,
        "personal": id_personal,
        "work_old": id_work_old,
    }
    return client, ids


@pytest.fixture(params=["sqlite", "postgresql"], name="team_memoria_context")
def team_memoria_context(tmp_path, monkeypatch, request):
    """Provision a Memoria instance with two team spaces for integration tests."""

    backend = request.param
    db_file = tmp_path / f"team_{backend}.db"
    sqlite_url = f"sqlite:///{db_file}"
    connect_url = sqlite_url

    monkeypatch.setattr(
        Memoria, "_init_sync", lambda self, settings, backend: None, raising=False
    )

    if backend == "postgresql":
        connect_url = f"postgresql+psycopg:///{db_file}"

        def _skip_validation(self, database_connect):
            return None

        def _forced_engine(self, database_connect):
            return create_engine(sqlite_url)

        monkeypatch.setattr(
            SQLAlchemyDatabaseManager,
            "_validate_database_dependencies",
            _skip_validation,
        )
        monkeypatch.setattr(SQLAlchemyDatabaseManager, "_create_engine", _forced_engine)
        monkeypatch.setattr(
            DatabaseAutoCreator,
            "ensure_database_exists",
            lambda self, url: sqlite_url,
        )

    mem = Memoria(
        database_connect=connect_url,
        enable_short_term=True,
        user_id="alice",
        team_memory_enabled=True,
        team_enforce_membership=True,
        team_namespace_prefix="team",
    )
    mem.enable()
    mem.storage_service.configure_team_policy(
        namespace_prefix="team", enforce_membership=True
    )
    mem._sync_coordinator = None
    mem._sync_backend = None
    mem._sync_backend_owned = False
    mem._sync_settings = None

    teams = {
        "ops": mem.register_team_space(
            "ops",
            members=["alice", "bob"],
            admins=["alice"],
            share_by_default=True,
        ),
        "research": mem.register_team_space(
            "research",
            members=["alice"],
            admins=["alice"],
            share_by_default=False,
        ),
    }

    if backend == "postgresql":
        mem.db_manager.database_type = "postgresql"
        mem.db_manager.query_translator = QueryParameterTranslator("postgresql")

    try:
        yield mem, teams, backend
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()
        if getattr(mem.db_manager, "engine", None):
            mem.db_manager.engine.dispose()
