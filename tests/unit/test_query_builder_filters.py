import datetime
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.utils.query_builder import DatabaseDialect, QueryBuilder


@pytest.mark.parametrize(
    "dialect, placeholder",
    [
        (DatabaseDialect.SQLITE, "?"),
        (DatabaseDialect.POSTGRESQL, "%s"),
        (DatabaseDialect.MYSQL, "%s"),
    ],
)
def test_build_search_query_mixed_filters(dialect, placeholder):
    qb = QueryBuilder(dialect)
    start = datetime.datetime(2024, 1, 1)
    end = datetime.datetime(2024, 12, 31)

    query, params = qb.build_search_query(
        tables=["long_term_memory"],
        search_columns=["searchable_content"],
        query_text="hello",
        namespace="default",
        category_filter=["general", "test"],
        limit=20,
        use_fts=False,
        time_range=(start, end),
        importance_min=0.4,
        coordinates={"x": (-5.0, 5.0), "y": (-5.0, 5.0), "z": (-5.0, 5.0)},
    )

    # Ensure placeholder style matches dialect and count equals params
    assert query.count(placeholder) == len(params)

    # Check that key filter clauses appear in the query
    assert "created_at BETWEEN" in query
    assert "importance_score >=" in query
    assert "x_coord BETWEEN" in query
    assert "y_coord BETWEEN" in query
    assert "z_coord BETWEEN" in query
    assert "category_primary IN" in query

    expected_params = [
        "%hello%",
        "default",
        "general",
        "test",
        start,
        end,
        0.4,
        -5.0,
        5.0,
        -5.0,
        5.0,
        -5.0,
        5.0,
        20,
    ]

    assert params == expected_params


@pytest.mark.parametrize(
    "dialect, placeholder",
    [
        (DatabaseDialect.SQLITE, "?"),
        (DatabaseDialect.POSTGRESQL, "%s"),
        (DatabaseDialect.MYSQL, "%s"),
    ],
)
def test_anchor_expression_and_privacy_range(dialect, placeholder):
    qb = QueryBuilder(dialect)
    expr = "alpha AND (beta OR gamma) AND NOT delta"

    query, params = qb.build_search_query(
        tables=["long_term_memory"],
        search_columns=["searchable_content"],
        query_text="hello",
        namespace="default",
        anchor_expression=expr,
        privacy_range=(-10, 10),
    )

    # Ensure placeholder style matches dialect and count equals params
    assert query.count(placeholder) == len(params)

    # Ensure anchor and privacy conditions are present
    assert "symbolic_anchors" in query
    assert "y_coord BETWEEN" in query

    # Params should include anchors and privacy bounds
    assert params[0] == "%hello%"
    assert params[1] == "default"
    assert params[2:6] == ["%alpha%", "%beta%", "%gamma%", "%delta%"]
    assert params[6:8] == [-10, 10]
    assert params[-1] == 10
