#!/usr/bin/env python
"""Add a namespace column to the spatial metadata table."""

from __future__ import annotations

import os

from sqlalchemy import create_engine, inspect, text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///memoria.db")
TABLE = "spatial_metadata"
COLUMN = "namespace"
INDEX = "idx_spatial_namespace"
DEFAULT = "default"


def column_exists(inspector, table: str, column: str) -> bool:
    try:
        return any(col["name"] == column for col in inspector.get_columns(table))
    except Exception:
        return False


def index_exists(inspector, table: str, index_name: str) -> bool:
    try:
        return any(idx["name"] == index_name for idx in inspector.get_indexes(table))
    except Exception:
        return False


def add_namespace_column(engine) -> None:
    dialect = engine.dialect.name
    if dialect == "sqlite":
        ddl = f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} TEXT NOT NULL DEFAULT :default"
        finalize = None
    elif dialect == "postgresql":
        ddl = f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} TEXT DEFAULT :default"
        finalize = f"ALTER TABLE {TABLE} ALTER COLUMN {COLUMN} SET NOT NULL"
    elif dialect in {"mysql", "mariadb"}:
        ddl = (
            f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(255) NOT NULL DEFAULT :default"
        )
        finalize = None
    else:
        ddl = f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(255) DEFAULT :default"
        finalize = None

    with engine.begin() as conn:
        conn.execute(text(ddl), {"default": DEFAULT})
        conn.execute(
            text(
                f"UPDATE {TABLE} SET {COLUMN} = :default "
                f"WHERE {COLUMN} IS NULL OR {COLUMN} = ''"
            ),
            {"default": DEFAULT},
        )
        if finalize:
            try:
                conn.execute(text(finalize))
            except Exception:
                pass


def create_namespace_index(engine) -> None:
    dialect = engine.dialect.name
    if dialect in {"sqlite", "postgresql"}:
        ddl = f"CREATE INDEX IF NOT EXISTS {INDEX} ON {TABLE} ({COLUMN})"
    else:
        ddl = f"CREATE INDEX {INDEX} ON {TABLE} ({COLUMN})"

    with engine.begin() as conn:
        try:
            conn.execute(text(ddl))
        except Exception:
            # Index might already exist or dialect may not support IF NOT EXISTS
            pass


def main() -> None:
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)

    if not column_exists(inspector, TABLE, COLUMN):
        try:
            add_namespace_column(engine)
            print(f"Added {COLUMN} column to {TABLE}")
        except Exception as exc:  # pragma: no cover - database specific
            print(f"Failed to add column: {exc}")
    else:
        print(f"{TABLE} already has {COLUMN} column")

    inspector = inspect(engine)
    if not index_exists(inspector, TABLE, INDEX):
        try:
            create_namespace_index(engine)
            print(f"Created index {INDEX} on {TABLE}")
        except Exception as exc:  # pragma: no cover - database specific
            print(f"Failed to create index {INDEX}: {exc}")
    else:
        print(f"Index {INDEX} already present on {TABLE}")

    engine.dispose()


if __name__ == "__main__":
    main()
