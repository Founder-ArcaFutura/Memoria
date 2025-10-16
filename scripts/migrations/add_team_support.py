"""Add teams table and team_id foreign keys to existing deployments.

This migration is intentionally lightweight so it can be applied to SQLite,
PostgreSQL, and MySQL installations without Alembic. It creates the new
``teams`` table and adds nullable ``team_id`` columns plus supporting indexes
for chat and memory tables.
"""

from __future__ import annotations

import os
from typing import Iterable

from sqlalchemy import create_engine, inspect, text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///memoria.db")
TEAM_TABLE = "teams"
TEAM_COLUMN = "team_id"

FOREIGN_KEYS = {
    "chat_history": "fk_chat_history_team_id",
    "short_term_memory": "fk_short_term_team_id",
    "long_term_memory": "fk_long_term_team_id",
}

INDEX_DEFINITIONS: dict[str, list[tuple[str, Iterable[str], bool]]] = {
    TEAM_TABLE: [
        ("idx_team_slug", ("slug",), True),
        ("idx_team_created", ("created_at",), False),
    ],
    "chat_history": [("idx_chat_team", (TEAM_COLUMN,), False)],
    "short_term_memory": [
        ("idx_short_term_team", (TEAM_COLUMN,), False),
        ("idx_short_term_namespace_team", ("namespace", TEAM_COLUMN), False),
    ],
    "long_term_memory": [
        ("idx_long_term_team", (TEAM_COLUMN,), False),
        ("idx_long_term_namespace_team", ("namespace", TEAM_COLUMN), False),
    ],
}


def table_exists(inspector, table: str) -> bool:
    try:
        return table in inspector.get_table_names()
    except Exception:  # pragma: no cover - backend specific errors
        return False


def column_exists(inspector, table: str, column: str) -> bool:
    try:
        return any(col["name"] == column for col in inspector.get_columns(table))
    except Exception:  # pragma: no cover - backend specific errors
        return False


def index_exists(inspector, table: str, index: str) -> bool:
    try:
        return any(idx.get("name") == index for idx in inspector.get_indexes(table))
    except Exception:  # pragma: no cover - backend specific errors
        return False


def foreign_key_exists(inspector, table: str, fk_name: str) -> bool:
    try:
        return any(fk.get("name") == fk_name for fk in inspector.get_foreign_keys(table))
    except Exception:  # pragma: no cover - backend specific errors
        return False


def _create_team_table(engine) -> None:
    inspector = inspect(engine)
    if table_exists(inspector, TEAM_TABLE):
        print("✅ teams table already exists")
        return

    dialect = engine.dialect.name
    if dialect == "sqlite":
        ddl = f"""
        CREATE TABLE {TEAM_TABLE} (
            team_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            slug TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    elif dialect == "postgresql":
        ddl = f"""
        CREATE TABLE {TEAM_TABLE} (
            team_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            slug VARCHAR(255) NOT NULL,
            description TEXT,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    elif dialect in {"mysql", "mariadb"}:
        ddl = f"""
        CREATE TABLE {TEAM_TABLE} (
            team_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            slug VARCHAR(255) NOT NULL,
            description TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    else:
        ddl = f"""
        CREATE TABLE {TEAM_TABLE} (
            team_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            slug VARCHAR(255) NOT NULL,
            description TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """

    with engine.begin() as conn:
        conn.execute(text(ddl))
    print("✅ Created teams table")


def _ensure_index(engine, table: str, name: str, columns: Iterable[str], *, unique: bool = False) -> None:
    inspector = inspect(engine)
    if not table_exists(inspector, table):
        return
    if index_exists(inspector, table, name):
        return

    dialect = engine.dialect.name
    column_list = ", ".join(columns)
    prefix = "CREATE UNIQUE INDEX" if unique else "CREATE INDEX"
    if dialect in {"sqlite", "postgresql"}:
        prefix = f"{prefix} IF NOT EXISTS"

    statement = f"{prefix} {name} ON {table} ({column_list})"
    with engine.begin() as conn:
        conn.execute(text(statement))
    print(f"✅ Created index {name} on {table}")


def _add_team_column(engine, table: str, fk_name: str) -> None:
    inspector = inspect(engine)
    if not table_exists(inspector, table):
        print(f"⚠️ Table {table} not found, skipping")
        return
    if column_exists(inspector, table, TEAM_COLUMN):
        print(f"ℹ️ {table}.{TEAM_COLUMN} already exists")
        for index_name, columns, is_unique in INDEX_DEFINITIONS.get(table, []):
            _ensure_index(engine, table, index_name, columns, unique=is_unique)
        return

    dialect = engine.dialect.name
    column_type = "TEXT" if dialect == "sqlite" else "VARCHAR(255)"
    fk_clause = f" REFERENCES {TEAM_TABLE}({TEAM_COLUMN}) ON DELETE SET NULL"

    with engine.begin() as conn:
        if dialect == "sqlite":
            conn.execute(
                text(
                    f"ALTER TABLE {table} ADD COLUMN {TEAM_COLUMN} {column_type}{fk_clause}"
                )
            )
        else:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {TEAM_COLUMN} {column_type}"))

    print(f"✅ Added {TEAM_COLUMN} column to {table}")

    # Refresh inspector state before adding foreign keys or indexes
    inspector = inspect(engine)
    if dialect in {"postgresql", "mysql", "mariadb"}:
        if not foreign_key_exists(inspector, table, fk_name):
            fk_statement = (
                f"ALTER TABLE {table} ADD CONSTRAINT {fk_name} "
                f"FOREIGN KEY ({TEAM_COLUMN}) REFERENCES {TEAM_TABLE}({TEAM_COLUMN}) ON DELETE SET NULL"
            )
            with engine.begin() as conn:
                conn.execute(text(fk_statement))
            print(f"✅ Added foreign key {fk_name} on {table}.{TEAM_COLUMN}")

    for index_name, columns, is_unique in INDEX_DEFINITIONS.get(table, []):
        _ensure_index(engine, table, index_name, columns, unique=is_unique)


def main() -> None:
    engine = create_engine(DATABASE_URL)

    try:
        _create_team_table(engine)
        for index_name, columns, is_unique in INDEX_DEFINITIONS.get(TEAM_TABLE, []):
            _ensure_index(engine, TEAM_TABLE, index_name, columns, unique=is_unique)

        for table, fk_name in FOREIGN_KEYS.items():
            _add_team_column(engine, table, fk_name)

        print("✅ Team support migration completed")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
