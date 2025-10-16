#!/usr/bin/env python
"""Add `total_tokens` and `total_chars` columns to the `clusters` table."""
from __future__ import annotations

import os

from sqlalchemy import create_engine, inspect, text

from memoria.database.models import Base, Cluster

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///memoria.db")
TABLE = "clusters"
COLUMNS = {
    "total_tokens": "INT",
    "total_chars": "INT",
}

def column_exists(inspector, table: str, column: str) -> bool:
    try:
        return any(col["name"] == column for col in inspector.get_columns(table))
    except Exception:
        return False

def add_column(engine, column: str, col_type: str) -> None:
    sql_type = "INTEGER" if engine.dialect.name == "sqlite" else col_type
    with engine.begin() as conn:
        conn.execute(
            text(
                f"ALTER TABLE {TABLE} ADD COLUMN {column} {sql_type} DEFAULT 0"
            )
        )

def main() -> None:
    engine = create_engine(DATABASE_URL)
    # Ensure the clusters table exists before attempting to alter it
    Base.metadata.create_all(engine, tables=[Cluster.__table__])

    inspector = inspect(engine)
    for column, col_type in COLUMNS.items():
        if column_exists(inspector, TABLE, column):
            print(f"{TABLE} already has {column} column")
            continue
        try:
            add_column(engine, column, col_type)
            print(f"Added {column} to {TABLE}")
        except Exception as exc:  # pragma: no cover - db specific
            print(f"Skipping {column}: {exc}")
    engine.dispose()

if __name__ == "__main__":
    main()
