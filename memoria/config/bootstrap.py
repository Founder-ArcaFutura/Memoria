"""Interactive configuration bootstrapper for Memoria deployments."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from getpass import getpass
from pathlib import Path
from typing import Any

from loguru import logger

from memoria.config.manager import ConfigManager
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager
from memoria.utils.exceptions import ConfigurationError


class BootstrapError(RuntimeError):
    """Raised when the bootstrap wizard cannot complete successfully."""


@dataclass(slots=True)
class DatabaseOptions:
    """Structured database configuration for non-interactive bootstraps."""

    backend: str = "sqlite"
    path: str | None = None
    url: str | None = None
    skip_verification: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> DatabaseOptions:
        if not payload:
            return cls()

        backend = str(payload.get("backend", cls.backend)).strip() or "sqlite"
        path = payload.get("path")
        url = payload.get("url")
        skip = bool(payload.get("skip_verification", False))
        return cls(backend=backend, path=path, url=url, skip_verification=skip)


@dataclass(slots=True)
class BootstrapConfig:
    """Structured configuration used to bypass interactive prompts."""

    env: dict[str, str] = field(default_factory=dict)
    database: DatabaseOptions = field(default_factory=DatabaseOptions)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> BootstrapConfig:
        env_payload = payload.get("env", {})
        if not isinstance(env_payload, Mapping):
            raise BootstrapError(
                "'env' section must be a mapping of environment variables"
            )
        env = {str(key): str(value) for key, value in env_payload.items()}

        database = DatabaseOptions.from_mapping(payload.get("database"))
        return cls(env=env, database=database)


class BootstrapWizard:
    """Guide operators through first-time configuration of Memoria."""

    def __init__(
        self,
        *,
        env_path: Path | str = ".env",
        config_path: Path | str = "memoria.json",
        force: bool = False,
        output_stream=sys.stdout,
        config: BootstrapConfig | None = None,
    ) -> None:
        self.env_path = Path(env_path)
        self.config_path = Path(config_path)
        self.force = force
        self.stream = output_stream
        self.manager = ConfigManager.get_instance()
        self.manager.auto_load()
        self.config = config
        self.non_interactive = config is not None
        self.env_values: dict[str, str] = dict(config.env) if config else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute the bootstrap wizard interactively."""

        self._write_heading("Memoria bootstrap wizard")
        self._write(
            "This tool will collect API credentials, provision a storage backend, "
            "and generate configuration files for local development."
        )
        self._write("")

        if self.non_interactive:
            self._run_non_interactive()
            return

        try:
            self._collect_provider_credentials()
            connection_url = self._configure_database()
            self._verify_database(connection_url)
            self._persist_configuration(connection_url)
            self._write("")
            self._write("✅ Bootstrap complete. You can now run 'docker compose up'.")
        except KeyboardInterrupt as exc:  # pragma: no cover - interactive guard
            self._write("\nBootstrap aborted by user.")
            raise BootstrapError("Bootstrap aborted") from exc

    def _run_non_interactive(self) -> None:
        if not self.config:
            raise BootstrapError(
                "Non-interactive mode requires a configuration payload"
            )

        self._write("Running in non-interactive mode.")
        self._apply_non_interactive_env()
        connection_url = self._configure_database_from_config()

        if not self.config.database.skip_verification:
            self._verify_database(connection_url)

        self._persist_configuration(connection_url)
        self._write("")
        self._write("✅ Bootstrap complete. You can now run 'docker compose up'.")

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def _apply_non_interactive_env(self) -> None:
        assert self.config is not None

        existing_api_key = os.getenv("MEMORIA_API_KEY")
        generated_key = secrets.token_urlsafe(24)
        api_key = (
            self.env_values.get("MEMORIA_API_KEY") or existing_api_key or generated_key
        )
        self.env_values["MEMORIA_API_KEY"] = api_key

    def _configure_database_from_config(self) -> str:
        assert self.config is not None

        backend = (self.config.database.backend or "sqlite").strip().lower()
        if backend in {"sqlite", "sqlite3"}:
            return self._configure_sqlite_from_options(self.config.database)
        if backend in {"postgres", "postgresql"}:
            return self._configure_postgres_from_options(self.config.database)

        if self.config.database.url:
            connection_url = str(self.config.database.url)
            self.manager.update_setting("database.connection_string", connection_url)
            self.manager.update_setting("database.database_type", backend or "sqlite")
            self._write(f"Using {backend or 'custom'} database at {connection_url}")
            return connection_url

        raise BootstrapError(
            f"Unsupported database backend: {self.config.database.backend}"
        )

    def _configure_sqlite_from_options(self, options: DatabaseOptions) -> str:
        sqlite_path = options.path or "memoria.db"
        path = Path(sqlite_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection_url = options.url or f"sqlite:///{path.as_posix()}"

        self.manager.update_setting("database.connection_string", connection_url)
        self.manager.update_setting("database.database_type", "sqlite")

        self._write(f"Using SQLite database at {path.resolve()}")
        return connection_url

    def _configure_postgres_from_options(self, options: DatabaseOptions) -> str:
        connection_url = (
            options.url or "postgresql://memoria:memoria@localhost:5432/memoria"
        )
        self.manager.update_setting("database.connection_string", connection_url)
        self.manager.update_setting("database.database_type", "postgresql")

        self._write(f"Using PostgreSQL database at {connection_url}")
        return connection_url

    def _prompt(self, message: str, *, default: str | None = None) -> str:
        prompt = message
        if default:
            prompt += f" [{default}]"
        prompt += ": "
        response = input(prompt).strip()
        if not response and default is not None:
            return default
        return response

    def _prompt_secret(self, message: str, *, allow_empty: bool = True) -> str:
        response = getpass(f"{message}: ")
        if not response and not allow_empty:
            self._write("Value required; please try again.")
            return self._prompt_secret(message, allow_empty=allow_empty)
        return response.strip()

    def _confirm(self, message: str, *, default: bool = False) -> bool:
        suffix = "[Y/n]" if default else "[y/N]"
        response = self._prompt(f"{message} {suffix}")
        if not response:
            return default
        return response.strip().lower() in {"y", "yes"}

    # ------------------------------------------------------------------
    # Credential handling
    # ------------------------------------------------------------------
    def _collect_provider_credentials(self) -> None:
        self._write("Collecting provider credentials (press Enter to skip).")
        provider_map: Iterable[tuple[str, str, str]] = (
            ("OpenAI", "MEMORIA_AGENTS__OPENAI_API_KEY", "OPENAI_API_KEY"),
            ("Anthropic", "MEMORIA_AGENTS__ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
            ("Google Gemini", "MEMORIA_AGENTS__GEMINI_API_KEY", "GEMINI_API_KEY"),
        )
        for label, memoria_key, public_key in provider_map:
            secret = self._prompt_secret(f"{label} API key", allow_empty=True)
            if not secret:
                continue
            self.env_values[memoria_key] = secret
            self.env_values[public_key] = secret

        existing_api_key = os.getenv("MEMORIA_API_KEY")
        generated_key = secrets.token_urlsafe(24)
        if existing_api_key:
            suggested = existing_api_key
        elif self.env_values.get("MEMORIA_API_KEY"):
            suggested = self.env_values["MEMORIA_API_KEY"]
        else:
            suggested = generated_key

        self._write(
            "A Memoria API key is required for REST requests. Press Enter to "
            "accept the generated value."
        )
        api_key = self._prompt_secret(
            "Memoria API key (leave blank to accept generated)", allow_empty=True
        )
        api_key = api_key or suggested
        self.env_values["MEMORIA_API_KEY"] = api_key

    # ------------------------------------------------------------------
    # Database configuration
    # ------------------------------------------------------------------
    def _configure_database(self) -> str:
        self._write("")
        self._write("Configure storage backend")
        default_connection = self.manager.get_setting(
            "database.connection_string", "sqlite:///memoria.db"
        )
        self._write("Choose a backend:")
        self._write("  1) SQLite (local file, zero dependencies)")
        self._write("  2) Dockerised Postgres (uses docker compose db service)")
        selection = self._prompt("Selection [1/2]", default="1")

        if selection.strip() in {"2", "postgres", "postgresql"}:
            return self._configure_postgres()
        return self._configure_sqlite(default_connection)

    def _configure_sqlite(self, default_connection: str) -> str:
        default_path = "memoria.db"
        if default_connection.startswith("sqlite:///"):
            default_path = default_connection.replace("sqlite:///", "", 1)
        sqlite_path = self._prompt(
            "SQLite file path", default=str(Path(default_path).as_posix())
        )
        sqlite_path = sqlite_path.strip() or default_path
        sqlite_path = Path(sqlite_path).expanduser()
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connection_url = f"sqlite:///{sqlite_path.as_posix()}"

        self.manager.update_setting("database.connection_string", connection_url)
        self.manager.update_setting("database.database_type", "sqlite")

        self._write(f"Using SQLite database at {sqlite_path.resolve()}")
        return connection_url

    def _configure_postgres(self) -> str:
        compose_command = self._resolve_compose_command()
        if compose_command is None:
            raise BootstrapError(
                "Docker Compose is required to provision Postgres. Install Docker "
                "and try again, or choose SQLite."
            )

        self._write(
            "Starting docker compose service 'db' (user=memoria password=memoria)."
        )
        try:
            subprocess.run(
                [*compose_command, "up", "-d", "db"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - system dep
            logger.error(exc.stdout)
            raise BootstrapError(
                "Failed to start Postgres via docker compose."
            ) from exc

        connection_url = "postgresql://memoria:memoria@localhost:5432/memoria"
        self.manager.update_setting("database.connection_string", connection_url)
        self.manager.update_setting("database.database_type", "postgresql")

        self._write("Waiting for Postgres to accept connections...")
        return connection_url

    def _resolve_compose_command(self) -> list[str] | None:
        docker_path = shutil.which("docker")
        compose_path = shutil.which("docker-compose")

        if docker_path:
            try:
                subprocess.run(
                    [docker_path, "compose", "version"],
                    capture_output=True,
                    check=True,
                    text=True,
                )
                return [docker_path, "compose"]
            except subprocess.CalledProcessError:  # pragma: no cover - env dependent
                pass

        if compose_path:
            return [compose_path]
        return None

    # ------------------------------------------------------------------
    # Verification and persistence
    # ------------------------------------------------------------------
    def _verify_database(self, connection_url: str) -> None:
        retries = 6
        delay = 2.0
        last_error: Exception | None = None
        for _attempt in range(1, retries + 1):
            try:
                manager = SQLAlchemyDatabaseManager(connection_url)
                manager.close()
                self._write(f"Database connection verified ({connection_url}).")
                return
            except Exception as exc:  # pragma: no cover - requires DB failure
                last_error = exc
                time.sleep(delay)
        raise BootstrapError(f"Unable to verify database connectivity: {last_error}")

    def _persist_configuration(self, connection_url: str) -> None:
        self.env_values["DATABASE_URL"] = connection_url
        self.env_values["MEMORIA_DB_URL"] = connection_url

        self._write("")
        self._write("Writing configuration files...")
        self._write_env_file()
        self._write_config_file()

        # Reload configuration to ensure consistency with generated files
        try:
            self.manager.auto_load()
        except ConfigurationError as exc:  # pragma: no cover - defensive guard
            raise BootstrapError(f"Configuration reload failed: {exc}") from exc

    def _write_env_file(self) -> None:
        if self.env_path.exists() and not self.force:
            if self.non_interactive:
                raise BootstrapError(
                    f"{self.env_path} exists and overwrite was not forced in non-interactive mode"
                )
            if not self._confirm(f"{self.env_path} exists. Overwrite?", default=False):
                self._write("Skipping .env creation; existing file retained.")
                return

        lines = [f"{key}={value}" for key, value in sorted(self.env_values.items())]
        content = "\n".join(lines) + "\n"
        self.env_path.write_text(content, encoding="utf-8")
        self._write(f"Wrote environment variables to {self.env_path}.")

    def _write_config_file(self) -> None:
        if self.config_path.exists() and not self.force:
            if self.non_interactive:
                raise BootstrapError(
                    f"{self.config_path} exists and overwrite was not forced in non-interactive mode"
                )
            if not self._confirm(
                f"{self.config_path} exists. Overwrite?", default=False
            ):
                self._write("Skipping config creation; existing file retained.")
                return

        data = self.manager.get_settings().export(include_sensitive=True)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        self._write(f"Wrote configuration to {self.config_path}.")

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def _write_heading(self, message: str) -> None:
        self._write(message)
        self._write("=" * len(message))

    def _write(self, message: str) -> None:
        print(message, file=self.stream)


def run_bootstrap(
    *,
    env_path: str | Path = ".env",
    config_path: str | Path = "memoria.json",
    force: bool = False,
    config: BootstrapConfig | None = None,
) -> None:
    """Convenience entry point used by the CLI."""

    wizard = BootstrapWizard(
        env_path=env_path,
        config_path=config_path,
        force=force,
        config=config,
    )
    wizard.run()
