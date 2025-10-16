import importlib
import importlib.util
import sys
import types

import pytest


@pytest.fixture()
def sqlalchemy_manager_class(monkeypatch):
    fake_config_module = types.ModuleType("memoria.config.manager")

    class DummyConfigManager:
        @classmethod
        def get_instance(cls):
            return cls()

        def get_settings(self):  # pragma: no cover - not used but provided for safety
            return types.SimpleNamespace(
                database=types.SimpleNamespace(backup_enabled=False)
            )

    fake_config_module.ConfigManager = DummyConfigManager
    monkeypatch.setitem(sys.modules, "memoria.config.manager", fake_config_module)

    # Ensure the module under test is re-imported using the stubbed dependency
    sys.modules.pop("memoria.database.sqlalchemy_manager", None)
    module = importlib.import_module("memoria.database.sqlalchemy_manager")
    return module.SQLAlchemyDatabaseManager


def test_validate_database_dependencies_accepts_psycopg(
    monkeypatch, sqlalchemy_manager_class
):
    manager = sqlalchemy_manager_class.__new__(sqlalchemy_manager_class)

    def fake_find_spec(name):
        if name == "psycopg":
            return object()
        return None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    # Should not raise when psycopg (v3) is available even if psycopg2/asyncpg are missing
    manager._validate_database_dependencies("postgresql://user:pass@localhost:5432/db")
