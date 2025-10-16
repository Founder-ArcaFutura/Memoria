from memoria import Memoria
from memoria.config.manager import ConfigManager
from memoria.config.settings import MemoriaSettings, TeamMode


def test_workspace_default_activation(monkeypatch, tmp_path):
    manager = ConfigManager.get_instance()
    original_settings = manager.get_settings()

    test_settings = MemoriaSettings()
    test_settings.memory.team_memory_enabled = True
    test_settings.memory.team_mode = TeamMode.OPTIONAL
    test_settings.memory.team_default_id = None
    test_settings.memory.workspace_mode = TeamMode.OPTIONAL
    test_settings.memory.workspace_default_id = "alpha"

    monkeypatch.setattr(manager, "_settings", test_settings, raising=False)
    monkeypatch.setattr(manager, "_config_sources", ["defaults"], raising=False)
    monkeypatch.setattr(manager, "_env_overrides", [], raising=False)

    original_apply = Memoria._apply_default_workspace

    def preload_workspace(self, workspace_id, *, enforce_membership):
        if workspace_id and self.storage_service.get_team_space(workspace_id) is None:
            self.register_workspace(
                workspace_id,
                members=[self.user_id] if self.user_id else None,
                admins=[self.user_id] if self.user_id else None,
            )
        original_apply(self, workspace_id, enforce_membership=enforce_membership)

    monkeypatch.setattr(Memoria, "_apply_default_workspace", preload_workspace)

    mem: Memoria | None = None
    try:
        db_path = tmp_path / "workspace-default.db"
        mem = Memoria(
            database_connect=f"sqlite:///{db_path}",
            enable_short_term=False,
            user_id="alice",
            team_memory_enabled=True,
            team_enforce_membership=True,
        )

        assert mem.default_workspace_id == "alpha"
        assert mem.get_active_workspace() == "alpha"
    finally:
        if mem is not None:
            if mem._retention_scheduler:
                mem._retention_scheduler.stop()
            if getattr(mem.db_manager, "engine", None):
                mem.db_manager.engine.dispose()

    monkeypatch.setattr(manager, "_settings", original_settings, raising=False)
