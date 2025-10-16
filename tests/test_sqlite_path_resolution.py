import sys
from pathlib import Path

import pytest
from flask import Flask

# Ensure project root on path for direct module imports when pytest changes cwd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_memoria_bindings():
    from memoria_server.api.app_factory import refresh_memoria_binding
    from memoria_server.api.spatial_setup import init_spatial_db

    return refresh_memoria_binding, init_spatial_db


refresh_memoria_binding, init_spatial_db = _load_memoria_bindings()


def _dispose_engine(engine) -> None:
    if engine is not None:
        try:
            engine.dispose()
        except Exception:
            pass


def test_refresh_memoria_binding_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = Flask(__name__)

    (
        memoria_instance,
        resolved_url,
        db_path,
        engine,
        team_config,
    ) = refresh_memoria_binding(
        app,
        settings=None,
        database_url="sqlite:///my_memory.db",
    )

    try:
        assert resolved_url == "sqlite:///my_memory.db"
        assert db_path is not None
        assert Path(db_path).parent == tmp_path
        assert app.config["DB_PATH"] == str(Path(db_path))
        assert team_config == app.config["TEAM_CONFIG"]
    finally:
        _dispose_engine(engine)
        dispose = getattr(getattr(memoria_instance, "db_manager", None), "engine", None)
        if dispose is not None:
            _dispose_engine(dispose)


@pytest.mark.parametrize("kind", ["relative", "absolute"])
def test_init_spatial_db_handles_paths(tmp_path, monkeypatch, kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    if kind == "relative":
        database_url = "sqlite:///nested/memory.db"
        expected_parent = workspace / "nested"
    else:
        absolute_target = tmp_path / "absolute" / "data" / "memory.db"
        url_path = absolute_target.as_posix().lstrip("/")
        database_url = f"sqlite:////{url_path}"
        expected_parent = absolute_target.parent

    app = Flask(__name__)

    memoria_instance, _, db_path, engine, _ = refresh_memoria_binding(
        app,
        settings=None,
        database_url=database_url,
    )

    try:
        assert db_path is not None
        path_obj = Path(db_path)
        assert path_obj.parent == expected_parent
        assert path_obj.parent.exists()

        init_spatial_db(app)

        assert path_obj.exists()
    finally:
        _dispose_engine(engine)
        dispose = getattr(getattr(memoria_instance, "db_manager", None), "engine", None)
        if dispose is not None:
            _dispose_engine(dispose)
