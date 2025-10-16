"""Smoke tests for the Memoria command line interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from memoria.config.settings import MemoriaSettings
from memoria.database.models import LongTermMemory
from memoria.database.sqlalchemy_manager import (
    SQLAlchemyDatabaseManager as DatabaseManager,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_help() -> None:
    """The CLI should render a help message without errors."""

    result = subprocess.run(
        [sys.executable, "-m", "memoria.cli", "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in result.stdout
    assert "runserver" in result.stdout


def test_import_memories_cli(tmp_path) -> None:
    """The import script should load exported JSON into the configured database."""

    db_url = f"sqlite:///{tmp_path/'cli.db'}"
    manager = DatabaseManager(db_url)
    manager.initialize_schema()

    payload = {
        "metadata": {"exported_at": datetime.utcnow().isoformat()},
        "payload": {
            "long_term_memories": [
                {
                    "memory_id": "cli-1",
                    "text": "CLI import test",
                    "created_at": datetime.utcnow().replace(microsecond=0).isoformat(),
                    "anchors": ["cli"],
                }
            ]
        },
    }
    export_path = tmp_path / "cli_export.json"
    export_path.write_text(json.dumps(payload), encoding="utf-8")

    config_path = tmp_path / "memoria.json"
    config_path.write_text(
        json.dumps({"database": {"connection_string": db_url}}),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["MEMORIA_CONFIG_PATH"] = str(config_path)
    for proxy_key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    ):
        env.pop(proxy_key, None)

    subprocess.run(
        [
            sys.executable,
            "scripts/import_memories.py",
            str(export_path),
            "--namespace",
            "cli",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    with manager.SessionLocal() as session:
        stored = (
            session.query(LongTermMemory)
            .filter(LongTermMemory.namespace == "cli")
            .all()
        )

    assert len(stored) == 1
    assert stored[0].summary == "CLI import test"


def test_assign_task_model_persists_configuration(tmp_path) -> None:
    """assign-task-model should update the config file on disk."""

    config_path = tmp_path / "memoria.json"
    MemoriaSettings().to_file(config_path)

    env = os.environ.copy()
    env["MEMORIA_CONFIG_PATH"] = str(config_path)
    for proxy_key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    ):
        env.pop(proxy_key, None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "memoria.cli",
            "assign-task-model",
            "memory_ingest",
            "--provider",
            "anthropic",
            "--model",
            "claude-3-opus",
            "--fallback",
            "openai:gpt-4o-mini",
            "--fallback",
            "openai",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    route = payload["agents"]["task_model_routes"]["memory_ingest"]
    assert route["provider"] == "anthropic"
    assert route["model"] == "claude-3-opus"
    assert route["fallback"] == ["openai:gpt-4o-mini", "openai"]
    assert "Persisted task routing update" in result.stdout
