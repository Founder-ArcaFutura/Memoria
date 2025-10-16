#!/usr/bin/env python
"""Add embedding columns to the memory tables."""
from __future__ import annotations

import os

from sqlalchemy import create_engine, inspect, text

from memoria.database.models import Base, LongTermMemory, ShortTermMemory

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///memoria.db")
TABLES = {
    "short_term_memory": "embedding",
    "long_term_memory": "embedding",
}


def column_exists(inspector, table: str, column: str) -> bool:
    try:
        return any(col["name"] == column for col in inspector.get_columns(table))
    except Exception:
        return False


def determine_type(dialect: str) -> str:
    if dialect == "sqlite":
        return "TEXT"
    if dialect == "postgresql":
        return "JSONB"
    return "JSON"


def add_column(engine, table: str, column: str) -> None:
    column_type = determine_type(engine.dialect.name)
    statement = text(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
    with engine.begin() as conn:
        conn.execute(statement)


def main() -> None:
    engine = create_engine(DATABASE_URL)
    # Ensure base tables exist before altering
    Base.metadata.create_all(
        engine,
        tables=[ShortTermMemory.__table__, LongTermMemory.__table__],
    )

    inspector = inspect(engine)
    for table, column in TABLES.items():
        if column_exists(inspector, table, column):
            print(f"{table} already has {column} column")
            continue
        try:
            add_column(engine, table, column)
            print(f"Added {column} column to {table}")
        except Exception as exc:  # pragma: no cover - DB specific failures
            print(f"Skipping {table}.{column}: {exc}")
    engine.dispose()


if __name__ == "__main__":
    main()
