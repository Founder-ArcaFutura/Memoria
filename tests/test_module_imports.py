"""Smoke tests for the public Flask bootstrap modules."""

from __future__ import annotations

import importlib


def test_memoria_api_module_is_canonical_alias():
    import memoria.api as canonical
    import memoria_api as shim

    assert shim.app is canonical.app
    assert shim.init_spatial_db is canonical.init_spatial_db
    assert shim._schedule_daily_decrement is canonical._schedule_daily_decrement
    assert shim._sync_module_state is canonical._sync_module_state


def test_reloading_legacy_shim_preserves_singletons():
    import memoria.api as canonical
    import memoria_api as shim

    reloaded = importlib.reload(shim)

    assert reloaded.app is canonical.app
    assert reloaded.config_manager is canonical.config_manager
