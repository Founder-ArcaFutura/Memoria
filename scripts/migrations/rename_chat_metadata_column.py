"""Rename the chat_history metadata column to maintain compatibility."""

from __future__ import annotations

import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///memoria.db")

TABLE_NAME = "chat_history"
OLD_COLUMN = "metadata_json"
NEW_COLUMN = "metadata"


def table_exists(inspector, table: str) -> bool:
    """Return ``True`` when the requested table is present."""

    try:
        return table in inspector.get_table_names()
    except Exception:  # pragma: no cover - backend specific
        return False


def rename_metadata_column(engine) -> None:
    """Rename ``metadata_json`` to ``metadata`` when required."""

    inspector = inspect(engine)
    if not table_exists(inspector, TABLE_NAME):
        print("⚠️ chat_history table not found - nothing to migrate")
        return

    columns = {col["name"]: col for col in inspector.get_columns(TABLE_NAME)}

    if NEW_COLUMN in columns and OLD_COLUMN not in columns:
        print("✅ chat_history metadata column already up to date")
        return

    if OLD_COLUMN not in columns:
        print("⚠️ Legacy metadata_json column not found - skipping rename")
        return

    with engine.begin() as conn:
        dialect = engine.dialect.name

        if NEW_COLUMN not in columns:
            rename_stmt = f"ALTER TABLE {TABLE_NAME} RENAME COLUMN {OLD_COLUMN} TO {NEW_COLUMN}"
            try:
                conn.execute(text(rename_stmt))
                print("✅ Renamed metadata_json column to metadata")
                return
            except SQLAlchemyError as exc:
                if dialect in {"mysql", "mariadb"}:
                    column = columns[OLD_COLUMN]
                    column_type = column["type"].compile(dialect=engine.dialect)
                    nullable = "NULL" if column.get("nullable", True) else "NOT NULL"
                    change_stmt = (
                        f"ALTER TABLE {TABLE_NAME} CHANGE {OLD_COLUMN} {NEW_COLUMN} "
                        f"{column_type} {nullable}"
                    )
                    conn.execute(text(change_stmt))
                    print("✅ Renamed metadata_json column to metadata using CHANGE syntax")
                    return
                raise RuntimeError(
                    f"Failed to rename column {OLD_COLUMN} to {NEW_COLUMN}: {exc}"
                ) from exc

        # Both columns exist - copy data and drop the legacy column when possible.
        conn.execute(
            text(
                f"UPDATE {TABLE_NAME} SET {NEW_COLUMN} = {OLD_COLUMN} "
                f"WHERE {NEW_COLUMN} IS NULL"
            )
        )

        drop_stmt = f"ALTER TABLE {TABLE_NAME} DROP COLUMN {OLD_COLUMN}"
        try:
            conn.execute(text(drop_stmt))
            print("✅ Dropped legacy metadata_json column after migration")
        except SQLAlchemyError as exc:
            print(
                "⚠️ Failed to drop legacy metadata_json column automatically:",
                exc,
            )


def migrate() -> None:
    """Execute the metadata column migration."""

    engine = create_engine(DATABASE_URL)
    try:
        rename_metadata_column(engine)
    finally:
        engine.dispose()


if __name__ == "__main__":  # pragma: no cover - manual execution utility
    migrate()
