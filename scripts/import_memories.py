#!/usr/bin/env python3
"""Import a migration snapshot produced by ``export_memories.py``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memoria.config.manager import ConfigManager
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager
from memoria.storage.service import StorageService
from memoria.tools import migration


def _load_config() -> None:
    config = ConfigManager.get_instance()
    config_path = os.getenv("MEMORIA_CONFIG_PATH")
    if config_path:
        config.load_from_file(config_path)
        return
    if hasattr(config, "auto_load"):
        try:  # pragma: no cover - configuration bootstrap
            config.auto_load()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Configuration auto-load failed: %s", exc)


def import_memories(
    source: Path,
    *,
    namespace: str,
    namespace_map: dict[str, str],
    allow_duplicates: bool,
    dry_run: bool,
    validate: bool,
) -> migration.MigrationReport:
    _load_config()
    config = ConfigManager.get_instance().get_settings()
    manager = SQLAlchemyDatabaseManager(config.database.connection_string)
    manager.initialize_schema()

    service = StorageService(db_manager=manager, namespace=namespace)
    snapshot = migration.load_snapshot(source)

    validation = None
    if validate:
        validation = migration.validate_payload(
            service,
            snapshot,
            namespace_map=namespace_map,
            default_namespace=namespace,
            dedupe=not allow_duplicates,
        )
        if validation.errors:
            logger.warning(
                "Validation reported %d errors; import may skip these entries",
                len(validation.errors),
            )

    report = migration.import_snapshot(
        manager,
        service,
        snapshot,
        namespace_map=namespace_map,
        default_namespace=namespace,
        dedupe=not allow_duplicates,
        dry_run=dry_run,
        validation=validation,
    )

    return report


def _summarise(report: migration.MigrationReport) -> None:
    status = "Preview" if report.dry_run else "Import"
    logger.info("%s results:", status)
    for category, values in report.inserted.items():
        logger.info("  %s inserted: %d", category.replace("_", " "), len(values))

    if report.skipped:
        logger.warning("%d entries skipped", len(report.skipped))
        for entry in report.skipped[:10]:
            identifier = entry.get("memory_id") or entry.get("id") or entry.get("relationship_id")
            logger.warning("  Skipped %s: %s", entry.get("type", "record"), identifier)
        if len(report.skipped) > 10:
            logger.warning("  ... (%d additional skips not shown)", len(report.skipped) - 10)

    if report.errors:
        logger.error("%d errors encountered", len(report.errors))
        for entry in report.errors[:10]:
            identifier = entry.get("memory_id") or entry.get("id") or entry.get("relationship_id")
            logger.error("  Error %s: %s", identifier or entry.get("type", "record"), entry.get("error"))
        if len(report.errors) > 10:
            logger.error("  ... (%d additional errors not shown)", len(report.errors) - 10)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import an exported Memoria snapshot into the configured database",
    )
    parser.add_argument("input", help="Path to the exported JSON snapshot")
    parser.add_argument(
        "--namespace",
        default="default",
        help="Target namespace for imported memories (default: %(default)s)",
    )
    parser.add_argument(
        "--namespace-map",
        nargs="*",
        default=[],
        help="Optional namespace remapping entries in the form source=target",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Insert memories even if matching IDs or timestamps already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report without persisting changes",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip preflight validation before importing",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    source_path = Path(args.input).expanduser().resolve()
    if not source_path.exists():
        parser.error(f"Export file not found: {source_path}")

    try:
        namespace_map = migration.parse_namespace_mappings(args.namespace_map)
    except ValueError as exc:
        parser.error(str(exc))

    report = import_memories(
        source_path,
        namespace=args.namespace,
        namespace_map=namespace_map,
        allow_duplicates=args.allow_duplicates,
        dry_run=args.dry_run,
        validate=not args.no_validate,
    )

    _summarise(report)
    return 0 if not report.errors else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())

