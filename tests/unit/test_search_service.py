from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import Query, sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.database.models import Base  # noqa: E402
from memoria.database.search_service import SearchService  # noqa: E402


class CompileOnlyQuery(Query):
    """Query subclass that verifies MySQL compilation during count()."""

    def count(self):  # type: ignore[override]
        try:
            statement = self._statement_20()  # SQLAlchemy 1.4 compatibility helper
        except AttributeError:  # pragma: no cover - SQLAlchemy <1.4 compatibility
            statement = self.statement
        statement.compile(dialect=mysql.dialect())
        return 0


def _setup_mysql_like_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, query_cls=CompileOnlyQuery)
    return Session()


def test_meta_query_mysql_json_contains_compiles():
    session = _setup_mysql_like_session()
    try:
        service = SearchService(session, "mysql")

        counts = service.meta_query(["alpha", "gamma"], (-5.0, 5.0))

        assert counts == {"total_memories": 0, "by_anchor": {"alpha": 0, "gamma": 0}}
    finally:
        session.close()
