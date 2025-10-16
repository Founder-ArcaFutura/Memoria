"""Add workspace metadata tables and workspace scoping columns."""

from __future__ import annotations

import os
from typing import Iterable

from sqlalchemy import create_engine, inspect, text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///memoria.db")

WORKSPACE_TABLE = "workspaces"
WORKSPACE_MEMBERS_TABLE = "workspace_members"
WORKSPACE_COLUMN = "workspace_id"
TEAM_TABLE = "teams"
TEAM_COLUMN = "team_id"

FOREIGN_KEYS = {
    "chat_history": "fk_chat_history_workspace_id",
    "short_term_memory": "fk_short_term_workspace_id",
    "long_term_memory": "fk_long_term_workspace_id",
    "spatial_metadata": "fk_spatial_metadata_workspace_id",
    "memory_access_events": "fk_memory_access_events_workspace_id",
}

INDEX_DEFINITIONS: dict[str, list[tuple[str, Iterable[str], bool]]] = {
    WORKSPACE_TABLE: [
        ("idx_workspace_slug", ("slug",), True),
        ("idx_workspace_owner", ("owner_id",), False),
        ("idx_workspace_team", ("team_id",), False),
        ("idx_workspace_created", ("created_at",), False),
    ],
    WORKSPACE_MEMBERS_TABLE: [
        ("idx_workspace_member_workspace", (WORKSPACE_COLUMN,), False),
        ("idx_workspace_member_user", ("user_id",), False),
        ("uq_workspace_member", (WORKSPACE_COLUMN, "user_id"), True),
    ],
    "chat_history": [
        ("idx_chat_workspace", (WORKSPACE_COLUMN,), False),
        ("idx_chat_workspace_namespace", (WORKSPACE_COLUMN, "namespace"), False),
    ],
    "short_term_memory": [
        ("idx_short_term_workspace", (WORKSPACE_COLUMN,), False),
        (
            "idx_short_term_namespace_workspace",
            ("namespace", WORKSPACE_COLUMN),
            False,
        ),
    ],
    "long_term_memory": [
        ("idx_long_term_workspace", (WORKSPACE_COLUMN,), False),
        (
            "idx_long_term_namespace_workspace",
            ("namespace", WORKSPACE_COLUMN),
            False,
        ),
    ],
    "spatial_metadata": [
        (
            "idx_spatial_workspace_namespace",
            (WORKSPACE_COLUMN, "namespace"),
            False,
        ),
    ],
    "memory_access_events": [
        (
            "idx_access_events_workspace_namespace",
            (WORKSPACE_COLUMN, "namespace", "accessed_at"),
            False,
        ),
    ],
}


def table_exists(inspector, table: str) -> bool:
    try:
        return table in inspector.get_table_names()
    except Exception:  # pragma: no cover - backend specific
        return False


def column_exists(inspector, table: str, column: str) -> bool:
    try:
        return any(col["name"] == column for col in inspector.get_columns(table))
    except Exception:  # pragma: no cover - backend specific
        return False


def index_exists(inspector, table: str, index: str) -> bool:
    try:
        return any(idx.get("name") == index for idx in inspector.get_indexes(table))
    except Exception:  # pragma: no cover - backend specific
        return False


def foreign_key_exists(inspector, table: str, fk_name: str) -> bool:
    try:
        return any(fk.get("name") == fk_name for fk in inspector.get_foreign_keys(table))
    except Exception:  # pragma: no cover - backend specific
        return False


def _ensure_index(engine, table: str, name: str, columns: Iterable[str], *, unique: bool = False) -> None:
    inspector = inspect(engine)
    if not table_exists(inspector, table):
        return
    if index_exists(inspector, table, name):
        return

    column_list = ", ".join(columns)
    dialect = engine.dialect.name
    prefix = "CREATE UNIQUE INDEX" if unique else "CREATE INDEX"
    if dialect in {"sqlite", "postgresql"}:
        prefix = f"{prefix} IF NOT EXISTS"

    statement = f"{prefix} {name} ON {table} ({column_list})"
    with engine.begin() as conn:
        conn.execute(text(statement))
    print(f"✅ Created index {name} on {table}")


def _create_workspace_table(engine) -> None:
    inspector = inspect(engine)
    if table_exists(inspector, WORKSPACE_TABLE):
        print("✅ workspaces table already exists")
        return

    dialect = engine.dialect.name
    if dialect == "sqlite":
        ddl = f"""
        CREATE TABLE {WORKSPACE_TABLE} (
            workspace_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            slug TEXT NOT NULL,
            description TEXT,
            owner_id TEXT,
            team_id TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES {TEAM_TABLE} ({TEAM_COLUMN}) ON DELETE SET NULL
        )
        """
    elif dialect == "postgresql":
        ddl = f"""
        CREATE TABLE {WORKSPACE_TABLE} (
            workspace_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            slug VARCHAR(255) NOT NULL,
            description TEXT,
            owner_id VARCHAR(255),
            team_id VARCHAR(255),
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_workspaces_team FOREIGN KEY (team_id)
                REFERENCES {TEAM_TABLE} ({TEAM_COLUMN}) ON DELETE SET NULL
        )
        """
    elif dialect in {"mysql", "mariadb"}:
        ddl = f"""
        CREATE TABLE {WORKSPACE_TABLE} (
            workspace_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            slug VARCHAR(255) NOT NULL,
            description TEXT,
            owner_id VARCHAR(255),
            team_id VARCHAR(255),
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            CONSTRAINT fk_workspaces_team FOREIGN KEY (team_id)
                REFERENCES {TEAM_TABLE} ({TEAM_COLUMN}) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    else:
        ddl = f"""
        CREATE TABLE {WORKSPACE_TABLE} (
            workspace_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            slug VARCHAR(255) NOT NULL,
            description TEXT,
            owner_id VARCHAR(255),
            team_id VARCHAR(255),
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_workspaces_team FOREIGN KEY (team_id)
                REFERENCES {TEAM_TABLE} ({TEAM_COLUMN}) ON DELETE SET NULL
        )
        """

    with engine.begin() as conn:
        conn.execute(text(ddl))
    print("✅ Created workspaces table")


def _create_workspace_members_table(engine) -> None:
    inspector = inspect(engine)
    if table_exists(inspector, WORKSPACE_MEMBERS_TABLE):
        print("✅ workspace_members table already exists")
        return

    dialect = engine.dialect.name
    if dialect == "sqlite":
        ddl = f"""
        CREATE TABLE {WORKSPACE_MEMBERS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'member',
            is_admin BOOLEAN NOT NULL DEFAULT 0,
            joined_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workspace_id) REFERENCES {WORKSPACE_TABLE} ({WORKSPACE_COLUMN}) ON DELETE CASCADE,
            UNIQUE(workspace_id, user_id)
        )
        """
    elif dialect == "postgresql":
        ddl = f"""
        CREATE TABLE {WORKSPACE_MEMBERS_TABLE} (
            id SERIAL PRIMARY KEY,
            workspace_id VARCHAR(255) NOT NULL,
            user_id VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL DEFAULT 'member',
            is_admin BOOLEAN NOT NULL DEFAULT FALSE,
            joined_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_workspace_members_workspace FOREIGN KEY (workspace_id)
                REFERENCES {WORKSPACE_TABLE} ({WORKSPACE_COLUMN}) ON DELETE CASCADE,
            CONSTRAINT uq_workspace_member UNIQUE (workspace_id, user_id)
        )
        """
    elif dialect in {"mysql", "mariadb"}:
        ddl = f"""
        CREATE TABLE {WORKSPACE_MEMBERS_TABLE} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            workspace_id VARCHAR(255) NOT NULL,
            user_id VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL DEFAULT 'member',
            is_admin BOOLEAN NOT NULL DEFAULT FALSE,
            joined_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            CONSTRAINT fk_workspace_members_workspace FOREIGN KEY (workspace_id)
                REFERENCES {WORKSPACE_TABLE} ({WORKSPACE_COLUMN}) ON DELETE CASCADE,
            CONSTRAINT uq_workspace_member UNIQUE (workspace_id, user_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    else:
        ddl = f"""
        CREATE TABLE {WORKSPACE_MEMBERS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id VARCHAR(255) NOT NULL,
            user_id VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL DEFAULT 'member',
            is_admin BOOLEAN NOT NULL DEFAULT 0,
            joined_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_workspace_members_workspace FOREIGN KEY (workspace_id)
                REFERENCES {WORKSPACE_TABLE} ({WORKSPACE_COLUMN}) ON DELETE CASCADE,
            CONSTRAINT uq_workspace_member UNIQUE (workspace_id, user_id)
        )
        """

    with engine.begin() as conn:
        conn.execute(text(ddl))
    print("✅ Created workspace_members table")


def _add_workspace_column(engine, table: str, fk_name: str) -> None:
    inspector = inspect(engine)
    if not table_exists(inspector, table):
        print(f"⚠️ Table {table} not found, skipping")
        return
    if column_exists(inspector, table, WORKSPACE_COLUMN):
        print(f"ℹ️ {table}.{WORKSPACE_COLUMN} already exists")
        for index_name, columns, is_unique in INDEX_DEFINITIONS.get(table, []):
            _ensure_index(engine, table, index_name, columns, unique=is_unique)
        return

    dialect = engine.dialect.name
    column_type = "TEXT" if dialect == "sqlite" else "VARCHAR(255)"
    fk_clause = (
        f" REFERENCES {WORKSPACE_TABLE}({WORKSPACE_COLUMN}) ON DELETE SET NULL"
    )

    with engine.begin() as conn:
        if dialect == "sqlite":
            conn.execute(
                text(
                    f"ALTER TABLE {table} ADD COLUMN {WORKSPACE_COLUMN} {column_type}{fk_clause}"
                )
            )
        else:
            conn.execute(
                text(
                    f"ALTER TABLE {table} ADD COLUMN {WORKSPACE_COLUMN} {column_type}"
                )
            )

    print(f"✅ Added {WORKSPACE_COLUMN} column to {table}")

    inspector = inspect(engine)
    if dialect in {"postgresql", "mysql", "mariadb"}:
        if not foreign_key_exists(inspector, table, fk_name):
            statement = (
                f"ALTER TABLE {table} ADD CONSTRAINT {fk_name} "
                f"FOREIGN KEY ({WORKSPACE_COLUMN}) REFERENCES {WORKSPACE_TABLE}({WORKSPACE_COLUMN}) ON DELETE SET NULL"
            )
            with engine.begin() as conn:
                conn.execute(text(statement))
            print(f"✅ Added foreign key {fk_name} on {table}.{WORKSPACE_COLUMN}")

    for index_name, columns, is_unique in INDEX_DEFINITIONS.get(table, []):
        _ensure_index(engine, table, index_name, columns, unique=is_unique)


def _backfill_workspaces(engine) -> None:
    inspector = inspect(engine)
    if not table_exists(inspector, TEAM_TABLE):
        return

    with engine.begin() as conn:
        existing_workspace_ids = {
            row[0] for row in conn.execute(text(f"SELECT {WORKSPACE_COLUMN} FROM {WORKSPACE_TABLE}"))
        }
        teams = conn.execute(
            text(
                f"SELECT {TEAM_COLUMN}, name, slug, description, created_at, updated_at "
                f"FROM {TEAM_TABLE}"
            )
        )
        for team_id, name, slug, description, created_at, updated_at in teams:
            if team_id is None:
                continue
            if team_id in existing_workspace_ids:
                continue
            conn.execute(
                text(
                    f"""
                    INSERT INTO {WORKSPACE_TABLE} (
                        {WORKSPACE_COLUMN}, name, slug, description, owner_id, {TEAM_COLUMN}, created_at, updated_at
                    ) VALUES (:workspace_id, :name, :slug, :description, NULL, :team_id, :created_at, :updated_at)
                    """
                ),
                {
                    "workspace_id": team_id,
                    "name": name or team_id,
                    "slug": slug or team_id,
                    "description": description,
                    "team_id": team_id,
                    "created_at": created_at,
                    "updated_at": updated_at,
                },
            )
            existing_workspace_ids.add(team_id)
        print("✅ Backfilled workspace rows from teams")


def _backfill_workspace_ids(engine) -> None:
    with engine.begin() as conn:
        for table in FOREIGN_KEYS.keys():
            statement = text(
                f"""
                UPDATE {table}
                SET {WORKSPACE_COLUMN} = {TEAM_COLUMN}
                WHERE {WORKSPACE_COLUMN} IS NULL AND {TEAM_COLUMN} IS NOT NULL
                """
            )
            result = conn.execute(statement)
            print(
                f"ℹ️ Updated {result.rowcount} rows in {table} with workspace IDs from team IDs"
            )


def main() -> None:
    engine = create_engine(DATABASE_URL)

    _create_workspace_table(engine)
    _create_workspace_members_table(engine)

    for table in (WORKSPACE_TABLE, WORKSPACE_MEMBERS_TABLE):
        for index_name, columns, is_unique in INDEX_DEFINITIONS.get(table, []):
            _ensure_index(engine, table, index_name, columns, unique=is_unique)

    for table, fk_name in FOREIGN_KEYS.items():
        _add_workspace_column(engine, table, fk_name)

    for table, index_defs in INDEX_DEFINITIONS.items():
        if table in {WORKSPACE_TABLE, WORKSPACE_MEMBERS_TABLE}:
            continue
        for index_name, columns, is_unique in index_defs:
            _ensure_index(engine, table, index_name, columns, unique=is_unique)

    _backfill_workspaces(engine)
    _backfill_workspace_ids(engine)


if __name__ == "__main__":
    main()
