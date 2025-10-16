from unittest.mock import Mock

from memoria.core.memory import Memoria


def _build_memoria_stub() -> Memoria:
    memoria = Memoria.__new__(Memoria)

    memoria.sovereign_ingest = True
    memoria.memory_manager = Mock()
    memoria.memory_manager._enabled = False
    memoria.memory_manager.enable.return_value = {
        "success": True,
        "enabled_interceptors": [],
    }
    memoria.memory_manager.disable.return_value = {"success": True}

    memoria.conscious_ingest = False
    memoria.conscious_agent = None
    memoria.conscious_manager = Mock()
    memoria.conscious_manager.start = Mock()
    memoria.conscious_manager.stop = Mock()
    memoria.conscious_manager.is_running = Mock(return_value=False)

    retention_scheduler = Mock()
    retention_scheduler.start = Mock()
    retention_scheduler.stop = Mock()
    ingestion_scheduler = Mock()
    ingestion_scheduler.start = Mock()
    ingestion_scheduler.stop = Mock()

    memoria._retention_scheduler = retention_scheduler
    memoria._ingestion_scheduler = ingestion_scheduler

    memoria._enabled = False
    memoria._session_id = "test-session"

    return memoria


def test_schedulers_restart_after_disable_enable():
    memoria = _build_memoria_stub()

    memoria._enabled = True
    memoria.disable()

    memoria._retention_scheduler.stop.assert_called_once()
    memoria._ingestion_scheduler.stop.assert_called_once()

    memoria._retention_scheduler.stop.reset_mock()
    memoria._ingestion_scheduler.stop.reset_mock()
    memoria._retention_scheduler.start.reset_mock()
    memoria._ingestion_scheduler.start.reset_mock()

    memoria.enable()

    memoria._retention_scheduler.start.assert_called_once()
    memoria._ingestion_scheduler.start.assert_called_once()
