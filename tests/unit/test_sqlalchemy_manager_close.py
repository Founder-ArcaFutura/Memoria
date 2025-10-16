import sys
import types
from types import SimpleNamespace

from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager


def test_close_shuts_down_scheduler_when_backups_enabled(tmp_path, monkeypatch):
    scheduler_instances = []

    class FakeScheduler:
        def __init__(self, *args, **kwargs):
            self.shutdown_calls = []
            scheduler_instances.append(self)

        def add_job(self, *args, **kwargs):
            return None

        def start(self):
            return None

        def shutdown(self, wait=True):
            self.shutdown_calls.append(wait)

    fake_background_module = types.ModuleType("apscheduler.schedulers.background")
    fake_background_module.BackgroundScheduler = FakeScheduler

    monkeypatch.setitem(sys.modules, "apscheduler", types.ModuleType("apscheduler"))
    monkeypatch.setitem(
        sys.modules,
        "apscheduler.schedulers",
        types.ModuleType("apscheduler.schedulers"),
    )
    monkeypatch.setitem(
        sys.modules,
        "apscheduler.schedulers.background",
        fake_background_module,
    )

    settings = SimpleNamespace(
        database=SimpleNamespace(
            backup_enabled=True,
            backup_interval_hours=1,
        )
    )

    monkeypatch.setattr(
        SQLAlchemyDatabaseManager,
        "_get_settings",
        staticmethod(lambda: settings),
    )

    db_path = tmp_path / "scheduler.db"
    manager = SQLAlchemyDatabaseManager(f"sqlite:///{db_path}")

    assert scheduler_instances, "Expected backup scheduler to be created"
    scheduler = scheduler_instances[0]

    manager.close()

    assert scheduler.shutdown_calls == [False]
    assert manager._scheduler is None
