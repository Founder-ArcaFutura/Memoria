"""Ensure Python schema definitions stay aligned with the canonical SQL file."""

from memoria.database.queries.base_queries import (
    SchemaQueries,
    load_schema_from_sql,
)


def test_schema_queries_match_canonical_sql() -> None:
    """The in-code schema maps must match the canonical SQL definitions."""

    schema_components = load_schema_from_sql(SchemaQueries.SCHEMA_PATH)

    assert dict(SchemaQueries.TABLE_CREATION) == schema_components["tables"]
    assert dict(SchemaQueries.INDEX_CREATION) == schema_components["indexes"]
    assert dict(SchemaQueries.TRIGGER_CREATION) == schema_components["triggers"]

    # Sanity check: the schema must contain known core tables
    for required_table in ("chat_history", "short_term_memory", "long_term_memory"):
        assert (
            required_table in SchemaQueries.TABLE_CREATION
        ), f"Missing core table definition: {required_table}"
