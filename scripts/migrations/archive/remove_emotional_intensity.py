#!/usr/bin/env python
"""Legacy migration to drop ``emotional_intensity`` column.

This script is preserved for databases created before the
``emotional_intensity`` column was removed from the schema. New
installations do not need to run it.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine, inspect, text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///memoria.db")
TABLES = ["short_term_memory", "long_term_memory"]
COLUMN = "emotional_intensity"


def column_exists(inspector, table: str, column: str) -> bool:
    try:
        return any(col["name"] == column for col in inspector.get_columns(table))
    except Exception:
        return False


def drop_column(engine, table: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {COLUMN}"))


def main() -> None:
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    for table in TABLES:
        if not column_exists(inspector, table, COLUMN):
            print(f"{table} has no {COLUMN} column")
            continue
        try:
            drop_column(engine, table)
            print(f"Dropped {COLUMN} from {table}")
        except Exception as exc:  # pragma: no cover - database specific errors
            print(f"Skipping {table}: {exc}")
    engine.dispose()


if __name__ == "__main__":
    main()
