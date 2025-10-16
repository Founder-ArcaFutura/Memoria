import sqlite3

from sqlalchemy import inspect

from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager


def test_cluster_member_tokens_and_chars_added(tmp_path):
    db_path = tmp_path / "memory.db"
    conn_str = f"sqlite:///{db_path}"

    # Create a fresh schema then close to modify the database manually
    manager = SQLAlchemyDatabaseManager(conn_str)
    manager.initialize_schema()
    manager.close()

    # Remove the tokens and chars columns to simulate an outdated schema
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE cluster_members RENAME TO cluster_members_old")
        conn.execute(
            """
            CREATE TABLE cluster_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id VARCHAR(255) NOT NULL,
                anchor VARCHAR(255),
                summary TEXT NOT NULL,
                cluster_id INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO cluster_members (id, memory_id, anchor, summary, cluster_id)
            SELECT id, memory_id, anchor, summary, cluster_id
            FROM cluster_members_old
            """
        )
        conn.execute("DROP TABLE cluster_members_old")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster ON cluster_members (cluster_id)"
        )
        conn.commit()

    # Ensure columns are indeed missing before re-initialization
    with sqlite3.connect(db_path) as conn:
        column_names = {
            row[1] for row in conn.execute("PRAGMA table_info('cluster_members')")
        }
        assert "tokens" not in column_names
        assert "chars" not in column_names

    # Re-initialize Memoria, which should trigger the column repair routine
    manager = SQLAlchemyDatabaseManager(conn_str)
    manager.initialize_schema()

    inspector = inspect(manager.engine)
    repaired_columns = {col["name"] for col in inspector.get_columns("cluster_members")}
    assert {"tokens", "chars"}.issubset(repaired_columns)

    manager.close()


def test_initialize_schema_repairs_embedding_columns(tmp_path):
    db_path = tmp_path / "legacy_embeddings.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE short_term_memory (memory_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE long_term_memory (memory_id TEXT PRIMARY KEY)")
        conn.commit()

    manager = SQLAlchemyDatabaseManager(f"sqlite:///{db_path}")
    manager.initialize_schema()

    inspector = inspect(manager.engine)
    short_columns = {col["name"] for col in inspector.get_columns("short_term_memory")}
    long_columns = {col["name"] for col in inspector.get_columns("long_term_memory")}

    assert "embedding" in short_columns
    assert "embedding" in long_columns

    manager.close()


def test_initialize_schema_sets_sqlite_embedding_type(tmp_path):
    db_path = tmp_path / "legacy_embedding_type.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE long_term_memory (memory_id TEXT PRIMARY KEY)")
        conn.commit()

    manager = SQLAlchemyDatabaseManager(f"sqlite:///{db_path}")
    manager.initialize_schema()

    inspector = inspect(manager.engine)
    long_columns = inspector.get_columns("long_term_memory")
    embedding_column = next(
        column for column in long_columns if column["name"] == "embedding"
    )

    assert str(embedding_column["type"]).upper() == "TEXT"

    manager.close()
