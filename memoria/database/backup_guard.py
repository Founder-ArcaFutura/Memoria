from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import inspect, text

from .sqlalchemy_manager import SQLAlchemyDatabaseManager


def _get_db_size(manager: SQLAlchemyDatabaseManager) -> int:
    """Return the size of the database in bytes."""
    try:
        return manager.get_db_size()
    except Exception as exc:  # pragma: no cover - depends on backend
        logger.warning(f"Failed to determine database size: {exc}")
        return 0


def _get_table_counts(manager: SQLAlchemyDatabaseManager) -> dict[str, int]:
    """Return row counts for all tables in the database."""
    counts: dict[str, int] = {}
    inspector = inspect(manager.engine)
    with manager.engine.connect() as conn:
        for table in inspector.get_table_names():
            try:
                counts[table] = int(
                    conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                )
            except Exception as exc:  # pragma: no cover - table may not exist
                logger.warning(f"Failed to count table {table}: {exc}")
    return counts


@contextmanager
def backup_guard(
    connection_string: str,
    ignore_tables: Iterable[str] | None = None,
    *,
    auto_restore_on_mismatch: bool = True,
):
    """Guard database operations with a backup and integrity checks.

    Args:
        connection_string: SQLAlchemy-compatible database URL.
        ignore_tables: Iterable of table names to exclude from row count comparisons.
        auto_restore_on_mismatch: Restore the backup automatically when
            integrity checks fail. Disable for workflows that intentionally
            rewrite large tables (for example, rebuilding cluster indexes).
    """
    ignored_tables = set(ignore_tables or [])
    manager = SQLAlchemyDatabaseManager(connection_string)
    backup_dir = Path("backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"guard_{datetime.now().strftime('%Y%m%d%H%M%S')}.sql"

    manager.backup_database(backup_path)
    logger.info(f"Created database backup at {backup_path}")

    size_before = _get_db_size(manager)
    counts_before = _get_table_counts(manager)

    try:
        yield
    finally:
        size_after = _get_db_size(manager)
        counts_after = _get_table_counts(manager)

        filtered_counts_before = {
            table: count
            for table, count in counts_before.items()
            if table not in ignored_tables
        }
        filtered_counts_after = {
            table: count
            for table, count in counts_after.items()
            if table not in ignored_tables
        }

        logger.info(
            f"Database size before: {size_before} bytes, after: {size_after} bytes"
        )
        logger.info(f"Table counts before: {counts_before}")
        logger.info(f"Table counts after: {counts_after}")

        size_changed = size_before != size_after
        mismatched_tables = {
            table: (
                filtered_counts_before.get(table),
                filtered_counts_after.get(table),
            )
            for table in set(filtered_counts_before) | set(filtered_counts_after)
            if filtered_counts_before.get(table) != filtered_counts_after.get(table)
        }

        if size_changed or mismatched_tables:
            logger.error(
                "Database integrity check failed: size_changed=%s mismatched_tables=%s",
                size_changed,
                mismatched_tables,
            )
            if auto_restore_on_mismatch:
                logger.warning(
                    "Restoring database from %s due to integrity mismatch", backup_path
                )
                manager.restore_database(backup_path)
            else:
                logger.warning(
                    "Automatic restore disabled; leaving database in potentially inconsistent state"
                )
        else:
            logger.info("Database integrity check passed")
