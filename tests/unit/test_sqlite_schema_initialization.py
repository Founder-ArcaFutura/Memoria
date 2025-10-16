from memoria.database.connectors.sqlite_connector import SQLiteConnector


def test_sqlite_connector_initializes_without_schema_sql(tmp_path):
    db_file = tmp_path / "memory.db"
    connector = SQLiteConnector({"database": str(db_file)})

    connector.initialize_schema()

    with connector.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = {row[0] for row in cursor.fetchall()}

    assert {"chat_history", "short_term_memory", "long_term_memory"}.issubset(
        table_names
    )
