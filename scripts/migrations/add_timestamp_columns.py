#!/usr/bin/env python
"""Add `timestamp` column to memory tables if missing.

This migration adds a `timestamp` column to the `long_term_memory` and
`spatial_metadata` tables for databases created before this field was
introduced.
"""
from __future__ import annotations

import os
from sqlalchemy import create_engine, inspect, text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///memoria.db")
TABLES = ["long_term_memory", "spatial_metadata"]
COLUMN = "timestamp"


def column_exists(inspector, table: str, column: str) -> bool:
    try:
        return any(col["name"] == column for col in inspector.get_columns(table))
    except Exception:
        return False


def add_column(engine, table: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {COLUMN} TIMESTAMP"))
        conn.execute(
            text(
                f"UPDATE {table} SET {COLUMN} = CURRENT_TIMESTAMP WHERE {COLUMN} IS NULL"
            )
        )


def main() -> None:
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    for table in TABLES:
        if column_exists(inspector, table, COLUMN):
            print(f"{table} already has {COLUMN} column")
            continue
        try:
            add_column(engine, table)
            print(f"Added {COLUMN} to {table}")
        except Exception as exc:  # pragma: no cover - db specific
            print(f"Skipping {table}: {exc}")
    engine.dispose()


if __name__ == "__main__":
    main()
