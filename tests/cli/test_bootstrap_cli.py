from __future__ import annotations

import json
from pathlib import Path

import pytest

from memoria.cli import main
from memoria.config.manager import ConfigManager


@pytest.fixture(autouse=True)
def reset_config() -> None:
    manager = ConfigManager.get_instance()
    manager.reset_to_defaults()
    yield
    manager.reset_to_defaults()


def test_bootstrap_non_interactive_creates_files(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    config_path = tmp_path / "memoria.json"
    db_path = tmp_path / "data" / "memoria.db"

    config_payload = {
        "env": {
            "MEMORIA_API_KEY": "test-api-key",
            "MEMORIA_AGENTS__OPENAI_API_KEY": "openai-key",
        },
        "database": {
            "backend": "sqlite",
            "path": str(db_path),
        },
    }

    config_json = tmp_path / "bootstrap.json"
    config_json.write_text(json.dumps(config_payload), encoding="utf-8")

    exit_code = main(
        [
            "bootstrap",
            "--non-interactive",
            "--env-file",
            str(env_path),
            "--config",
            str(config_path),
            "--config-json",
            str(config_json),
        ]
    )

    assert exit_code == 0
    assert env_path.exists()
    assert config_path.exists()
    assert db_path.exists()

    env_lines = env_path.read_text(encoding="utf-8").splitlines()
    env_map = dict(line.split("=", 1) for line in env_lines if "=" in line)

    expected_url = f"sqlite:///{db_path.as_posix()}"
    assert env_map.get("DATABASE_URL") == expected_url
    assert env_map.get("MEMORIA_DB_URL") == expected_url
    assert env_map.get("MEMORIA_API_KEY") == "test-api-key"
    assert env_map.get("MEMORIA_AGENTS__OPENAI_API_KEY") == "openai-key"

    stored_config = json.loads(config_path.read_text(encoding="utf-8"))
    database_settings = stored_config.get("database", {})
    assert database_settings.get("connection_string") == expected_url
    assert database_settings.get("database_type") == "sqlite"
