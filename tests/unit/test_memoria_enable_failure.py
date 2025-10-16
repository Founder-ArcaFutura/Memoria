from memoria.core.memory import Memoria


def test_enable_failure_does_not_set_enabled(monkeypatch):
    memoria = Memoria(conscious_ingest=True)

    # Ensure conscious ingest prerequisites are satisfied for the start check
    memoria.conscious_ingest = True
    memoria.conscious_agent = object()

    start_called = {"value": False}

    def fake_start():
        start_called["value"] = True

    monkeypatch.setattr(memoria.conscious_manager, "start", fake_start)

    def failing_enable(_interceptors=None):
        return {"success": False, "message": "test failure"}

    monkeypatch.setattr(memoria.memory_manager, "enable", failing_enable)

    result = memoria.enable()

    assert result["success"] is False
    assert memoria._enabled is False
    assert memoria.conscious_manager.is_running() is False
    assert start_called["value"] is False
