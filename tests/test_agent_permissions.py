import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memoria.agents.memory_agent import MemoryAgent
from memoria.config.memory_manager import MemoryManager
from memoria.utils.pydantic_models import AgentPermissions


class DummyStorageService:
    def __init__(self):
        self.stored = False
        self.deleted = None

    def store_memory(self, *args, **kwargs):
        self.stored = True
        return "ok"

    def delete_memory(self, memory_id):
        self.deleted = memory_id
        return True


def test_store_memory_requires_write_permission():
    perms = AgentPermissions(can_read=True, can_write=False, can_edit=False)
    agent = MemoryAgent(api_key="test", permissions=perms)
    manager = MemoryManager()
    manager.storage_service = DummyStorageService()
    with pytest.raises(PermissionError):
        manager.store_memory("a", "t", 1, permissions=agent.permissions)


def test_store_memory_with_permission():
    perms = AgentPermissions(can_read=True, can_write=True, can_edit=True)
    agent = MemoryAgent(api_key="test", permissions=perms)
    manager = MemoryManager()
    storage = DummyStorageService()
    manager.storage_service = storage
    result = manager.store_memory("a", "t", 1, permissions=agent.permissions)
    assert result == "ok"
    assert storage.stored is True


def test_delete_memory_requires_edit_permission():
    perms = AgentPermissions(can_read=True, can_write=True, can_edit=False)
    agent = MemoryAgent(api_key="test", permissions=perms)
    manager = MemoryManager()
    manager.storage_service = DummyStorageService()
    with pytest.raises(PermissionError):
        manager.delete_memory("123", permissions=agent.permissions)


def test_delete_memory_with_permission():
    perms = AgentPermissions(can_read=True, can_write=True, can_edit=True)
    agent = MemoryAgent(api_key="test", permissions=perms)
    manager = MemoryManager()
    storage = DummyStorageService()
    manager.storage_service = storage
    assert manager.delete_memory("456", permissions=agent.permissions)
    assert storage.deleted == "456"
