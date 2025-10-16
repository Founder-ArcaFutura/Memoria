import datetime

from sqlalchemy import text

from memoria.database.models import (
    DatabaseManager,
    LongTermMemory,
    ShortTermMemory,
)


def _create_db():
    db = DatabaseManager("sqlite:///:memory:")
    db.create_tables()
    return db


def _fts_count(session, term):
    return session.execute(
        text(
            "SELECT count(*) FROM memory_search_fts WHERE memory_search_fts MATCH :term"
        ),
        {"term": term},
    ).scalar()


def test_short_term_update_refreshes_fts():
    db = _create_db()
    session = db.get_session()

    stm = ShortTermMemory(
        memory_id="s1",
        processed_data={},
        importance_score=0.5,
        category_primary="test",
        namespace="default",
        created_at=datetime.datetime.utcnow(),
        searchable_content="hello world",
        summary="hi",
        symbolic_anchors=["h"],
    )
    session.add(stm)
    session.commit()

    assert _fts_count(session, "hello") == 1

    stm.searchable_content = "goodbye world"
    session.commit()

    assert _fts_count(session, "hello") == 0
    assert _fts_count(session, "goodbye") == 1


def test_long_term_update_refreshes_fts():
    db = _create_db()
    session = db.get_session()

    ltm = LongTermMemory(
        memory_id="l1",
        processed_data={},
        importance_score=0.5,
        category_primary="test",
        namespace="default",
        created_at=datetime.datetime.utcnow(),
        searchable_content="hello galaxy",
        summary="hi",
        symbolic_anchors=["g"],
    )
    session.add(ltm)
    session.commit()

    assert _fts_count(session, "hello") == 1

    ltm.searchable_content = "goodbye galaxy"
    session.commit()

    assert _fts_count(session, "hello") == 0
    assert _fts_count(session, "goodbye") == 1
