"""Helpers for discovering and executing bundled migration scripts."""

from __future__ import annotations

import ast
import os
import runpy
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class MigrationError(Exception):
    """Base exception for migration runner failures."""


class MigrationNotFoundError(MigrationError):
    """Raised when a requested migration cannot be located."""


class MigrationExecutionError(MigrationError):
    """Raised when executing a migration script fails."""


@dataclass(frozen=True)
class MigrationScript:
    """Metadata describing an available migration script."""

    name: str
    path: Path
    description: str | None
    archived: bool = False


_MIGRATIONS_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "migrations"


def _read_docstring(path: Path) -> str | None:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    return ast.get_docstring(module)


def _iter_migration_files(base: Path) -> Iterable[Path]:
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*.py") if not p.name.startswith("__"))


def discover_migrations(*, include_archived: bool = False) -> list[MigrationScript]:
    """Return discovered migration scripts sorted by filename."""

    migrations: list[MigrationScript] = []
    for path in _iter_migration_files(_MIGRATIONS_ROOT):
        archived = "archive" in path.parts
        if archived and not include_archived:
            continue
        description = _read_docstring(path)
        migrations.append(
            MigrationScript(
                name=path.stem,
                path=path,
                description=description,
                archived=archived,
            )
        )
    return migrations


def get_migration(name: str, *, include_archived: bool = False) -> MigrationScript:
    """Look up a migration by name or filename."""

    matches = []
    target = name.removesuffix(".py")
    for migration in discover_migrations(include_archived=include_archived):
        if migration.name == target or migration.path.name == name:
            matches.append(migration)
    if not matches:
        raise MigrationNotFoundError(f"Unknown migration: {name}")
    if len(matches) > 1:
        raise MigrationExecutionError(
            f"Multiple migrations matched '{name}'. Use the full filename to disambiguate."
        )
    return matches[0]


@contextmanager
def _patched_environ(overrides: Sequence[tuple[str, str]]):
    previous: dict[str, str | None] = {}
    try:
        for key, value in overrides:
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def run_migration_script(
    migration: MigrationScript,
    *,
    database_url: str,
    dry_run: bool = False,
) -> None:
    """Execute a migration script with the provided database URL."""

    if dry_run:
        return

    overrides = [("DATABASE_URL", database_url)]
    try:
        with _patched_environ(overrides):
            runpy.run_path(str(migration.path), run_name="__main__")
    except SystemExit as exc:  # pragma: no cover - depends on script behaviour
        raise MigrationExecutionError(
            f"Migration '{migration.name}' exited with status {exc.code}"
        ) from None
    except Exception as exc:  # pragma: no cover - depends on script behaviour
        raise MigrationExecutionError(
            f"Migration '{migration.name}' failed: {exc}"
        ) from exc
