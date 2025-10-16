"""Create the ``policy_artifacts`` table used to persist governance policies."""

from __future__ import annotations

import os

from sqlalchemy import create_engine, inspect, text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///memoria.db")
TABLE = "policy_artifacts"
INDEXES = {
    "idx_policy_artifacts_type": ("artifact_type",),
    "idx_policy_artifacts_namespace": ("namespace",),
}


def table_exists(inspector, table: str) -> bool:
    try:
        return table in inspector.get_table_names()
    except Exception:  # pragma: no cover - backend specific
        return False


def index_exists(inspector, table: str, name: str) -> bool:
    try:
        return any(idx.get("name") == name for idx in inspector.get_indexes(table))
    except Exception:  # pragma: no cover - backend specific
        return False


def upgrade() -> None:
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)

    if table_exists(inspector, TABLE):
        print("✅ policy_artifacts table already exists")
    else:
        dialect = engine.dialect.name
        if dialect == "sqlite":
            ddl = f"""
            CREATE TABLE {TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT '*',
                payload TEXT NOT NULL,
                schema_version TEXT NOT NULL DEFAULT '1.0',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                workspace_id TEXT,
                team_id TEXT,
                CONSTRAINT uq_policy_artifacts_identity
                    UNIQUE (name, namespace, workspace_id),
                FOREIGN KEY (workspace_id)
                    REFERENCES workspaces (workspace_id) ON DELETE SET NULL,
                FOREIGN KEY (team_id)
                    REFERENCES teams (team_id) ON DELETE SET NULL
            )
            """
        elif dialect == "postgresql":
            ddl = f"""
            CREATE TABLE {TABLE} (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                artifact_type VARCHAR(50) NOT NULL,
                namespace VARCHAR(255) NOT NULL DEFAULT '*',
                payload JSONB NOT NULL,
                schema_version VARCHAR(50) NOT NULL DEFAULT '1.0',
                created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by VARCHAR(255),
                workspace_id VARCHAR(255),
                team_id VARCHAR(255),
                CONSTRAINT uq_policy_artifacts_identity
                    UNIQUE (name, namespace, workspace_id),
                CONSTRAINT fk_policy_artifact_workspace
                    FOREIGN KEY (workspace_id)
                    REFERENCES workspaces (workspace_id) ON DELETE SET NULL,
                CONSTRAINT fk_policy_artifact_team
                    FOREIGN KEY (team_id)
                    REFERENCES teams (team_id) ON DELETE SET NULL
            )
            """
        elif dialect in {"mysql", "mariadb"}:
            ddl = f"""
            CREATE TABLE {TABLE} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                artifact_type VARCHAR(50) NOT NULL,
                namespace VARCHAR(255) NOT NULL DEFAULT '*',
                payload JSON NOT NULL,
                schema_version VARCHAR(50) NOT NULL DEFAULT '1.0',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                created_by VARCHAR(255),
                workspace_id VARCHAR(255),
                team_id VARCHAR(255),
                CONSTRAINT uq_policy_artifacts_identity
                    UNIQUE (name, namespace, workspace_id),
                CONSTRAINT fk_policy_artifact_workspace
                    FOREIGN KEY (workspace_id)
                    REFERENCES workspaces (workspace_id) ON DELETE SET NULL,
                CONSTRAINT fk_policy_artifact_team
                    FOREIGN KEY (team_id)
                    REFERENCES teams (team_id) ON DELETE SET NULL
            )
            ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        else:  # pragma: no cover - unsupported backend during tests
            raise RuntimeError(f"Unsupported database dialect: {dialect}")

        with engine.begin() as conn:
            conn.execute(text(ddl))
        print("✅ Created policy_artifacts table")

    inspector = inspect(engine)
    for name, columns in INDEXES.items():
        if index_exists(inspector, TABLE, name):
            continue
        column_sql = ", ".join(columns)
        dialect = engine.dialect.name
        prefix = "CREATE INDEX"
        if dialect in {"sqlite", "postgresql"}:
            prefix = f"{prefix} IF NOT EXISTS"
        statement = f"{prefix} {name} ON {TABLE} ({column_sql})"
        with engine.begin() as conn:
            conn.execute(text(statement))
        print(f"✅ Created index {name} on {TABLE}")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    upgrade()
