from sqlalchemy import inspect

from memoria.core.memory import Memoria


def test_memoria_initializes_missing_sqlite_db(tmp_path):
    db_file = tmp_path / "subdir" / "memory.db"
    conn_str = f"sqlite:///{db_file}"

    assert not db_file.exists()

    mem = Memoria(database_connect=conn_str)

    inspector = inspect(mem.db_manager.engine)
    tables = set(inspector.get_table_names())
    assert {
        "chat_history",
        "short_term_memory",
        "long_term_memory",
        "clusters",
        "cluster_members",
    }.issubset(tables)

    mem.db_manager.close()
