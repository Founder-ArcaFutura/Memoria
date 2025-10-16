"""Tests for database adapter utilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_utils_module():
    base_path = Path(__file__).resolve().parents[2]
    module_path = base_path / "memoria" / "database" / "adapters" / "utils.py"
    spec = importlib.util.spec_from_file_location(
        "memoria.database.adapters.utils", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader  # nosec: B101 - required for tests
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


utils = _load_utils_module()
extract_index_name = utils.extract_index_name


def test_extract_index_name_from_create_statement():
    sql = "CREATE INDEX IF NOT EXISTS idx_sample ON some_table(column)"
    assert extract_index_name(sql) == "idx_sample"


def test_extract_index_name_from_fulltext_statement():
    sql = (
        "ALTER TABLE short_term_memory ADD FULLTEXT idx_st_fulltext "
        "(searchable_content, summary)"
    )
    assert extract_index_name(sql) == "idx_st_fulltext"


def test_extract_index_name_from_multiline_statement():
    sql = """
    CREATE INDEX IF NOT EXISTS idx_short_term_fts_gin
    ON short_term_memory
    USING gin(to_tsvector('english', searchable_content || ' ' || summary))
    """

    assert extract_index_name(sql) == "idx_short_term_fts_gin"


def test_extract_index_name_returns_none_when_missing():
    assert extract_index_name("CREATE TABLE example (id integer)") is None
