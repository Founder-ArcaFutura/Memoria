"""Tests for the runtime toggle synchronization helper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from flask import Flask

from memoria_server.api.app_factory import _sync_runtime_toggles


class DummyMemoria:
    """Minimal stub for the memoria runtime object used by `_sync_runtime_toggles`."""

    def __init__(self) -> None:
        self.enable_short_term = True
        self.conscious_ingest = False
        self.conscious_agent = None
        self.conscious_manager = SimpleNamespace(stop=lambda: None)
        self.storage_service = SimpleNamespace(conscious_ingest=False)
        self.memory_manager = SimpleNamespace(
            conscious_ingest=False,
            sovereign_ingest=False,
            _enabled=False,
            disable=lambda: None,
            enable=lambda: None,
            set_memoria_instance=lambda _mem: None,
        )
        self.sovereign_ingest = False
        self.auto_ingest = False
        self.retention_service = SimpleNamespace(
            cluster_enabled=False,
            config=SimpleNamespace(cluster_gravity_lambda=0.0),
        )


@pytest.mark.parametrize(
    ("settings_overrides", "expected"),
    [
        (
            {
                "memory": SimpleNamespace(
                    sovereign_ingest=True,
                    context_injection=True,
                )
            },
            {
                "conscious_ingest": False,
                "sovereign_ingest": True,
                "context_injection": True,
            },
        ),
        (
            {
                "agents": SimpleNamespace(
                    conscious_ingest=True,
                )
            },
            {
                "conscious_ingest": True,
                "sovereign_ingest": False,
                "context_injection": False,
            },
        ),
        (
            {},
            {
                "conscious_ingest": False,
                "sovereign_ingest": False,
                "context_injection": False,
            },
        ),
    ],
)
def test_sync_runtime_toggles_handles_missing_sections(
    settings_overrides, expected
) -> None:
    """Ensure `_sync_runtime_toggles` tolerates missing settings sections."""

    app = Flask(__name__)
    memoria = DummyMemoria()

    base_settings = SimpleNamespace(
        enable_cluster_indexing=False,
        use_db_clusters=False,
        integrations=None,
    )
    for key, value in settings_overrides.items():
        setattr(base_settings, key, value)

    config_manager = SimpleNamespace(get_settings=lambda: base_settings)

    app.config["memoria"] = memoria
    app.config["config_manager"] = config_manager

    result = _sync_runtime_toggles(app)

    assert result["conscious_ingest"] is expected["conscious_ingest"]
    assert memoria.conscious_ingest is expected["conscious_ingest"]

    assert result["sovereign_ingest"] is expected["sovereign_ingest"]
    assert memoria.sovereign_ingest is expected["sovereign_ingest"]

    assert result["context_injection"] is expected["context_injection"]
    assert memoria.auto_ingest is expected["context_injection"]

    assert result["cluster_enabled"] is False
    assert memoria.retention_service.cluster_enabled is False
