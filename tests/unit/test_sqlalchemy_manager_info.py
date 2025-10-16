import pytest

from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager


@pytest.mark.parametrize("schema_init", [True, False])
def test_get_database_info_includes_auto_creation(tmp_path, schema_init):
    db_path = tmp_path / "info.db"
    manager = SQLAlchemyDatabaseManager(
        f"sqlite:///{db_path}",
        schema_init=schema_init,
    )

    info = manager.get_database_info()

    assert "auto_creation_enabled" in info
    assert info["auto_creation_enabled"] is bool(schema_init)

    if getattr(manager, "_scheduler", None):
        manager._scheduler.shutdown(wait=False)
    manager.engine.dispose()
