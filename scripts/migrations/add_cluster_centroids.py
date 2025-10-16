"""Add ``y_centroid`` and ``z_centroid`` columns to the clusters table."""

import os

from sqlalchemy import create_engine, inspect, text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///memoria.db")
TABLE = "clusters"
COLUMNS = {
    "y_centroid": "FLOAT",
    "z_centroid": "FLOAT",
}


def column_exists(inspector, table: str, column: str) -> bool:
    try:
        return any(col["name"] == column for col in inspector.get_columns(table))
    except Exception:
        return False


def add_column(engine, column: str, definition: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN {column} {definition}"))


def main() -> None:
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    for column, definition in COLUMNS.items():
        if column_exists(inspector, TABLE, column):
            print(f"{TABLE} already has {column} column")
            continue
        try:
            add_column(engine, column, definition)
            print(f"Added {column} to {TABLE}")
        except Exception as exc:  # pragma: no cover - database dependent
            print(f"Failed to add {column} to {TABLE}: {exc}")
    engine.dispose()


if __name__ == "__main__":
    main()
