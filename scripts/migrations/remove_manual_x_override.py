from __future__ import annotations

import os
from sqlalchemy import create_engine, inspect, text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///memoria.db")
TABLES = ["long_term_memory", "short_term_memory", "spatial_metadata"]
COLUMN = "manual_x_override"


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
            print(f"Removed {COLUMN} from {table}")
        except Exception as exc:  # pragma: no cover - db specific
            print(f"Skipping {table}: {exc}")
    engine.dispose()


if __name__ == "__main__":
    main()
