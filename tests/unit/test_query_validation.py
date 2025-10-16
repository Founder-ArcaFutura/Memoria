from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.utils.input_validator import InputValidator
from memoria.utils.query_builder import DatabaseDialect, QueryBuilder


@pytest.mark.parametrize(
    "raw_query",
    [
        "alpha & beta",
        "1 < 2",
        'He said "hello"',
        "It's complicated",
    ],
)
def test_validate_query_preserves_special_characters(raw_query: str) -> None:
    sanitized = InputValidator.validate_and_sanitize_query(raw_query)
    assert sanitized == raw_query


def test_validate_query_strips_control_characters() -> None:
    raw_query = "find\x00this\x1fmemory"
    sanitized = InputValidator.validate_and_sanitize_query(raw_query)
    assert sanitized == "findthismemory"


def test_query_builder_uses_preserved_query_text() -> None:
    qb = QueryBuilder(DatabaseDialect.SQLITE)
    raw_query = 'score < 100 & name="Ada"'

    query, params = qb.build_search_query(
        tables=["long_term_memory"],
        search_columns=["searchable_content"],
        query_text=raw_query,
        namespace="default",
    )

    assert "LIKE" in query
    assert params[0] == f"%{raw_query}%"
    assert params[1] == "default"
