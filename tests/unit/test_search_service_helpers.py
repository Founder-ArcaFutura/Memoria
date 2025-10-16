import math
from datetime import UTC, datetime

import pytest

from memoria.database.models import ShortTermMemory
from memoria.database.search_service import SearchService


@pytest.fixture
def search_service() -> SearchService:
    return SearchService(session=None, database_type="sqlite")


def test_apply_distance_metadata_adds_distance(search_service: SearchService):
    data = [
        {"memory_id": "1", "x": 0.0, "y": 0.0, "z": 0.0},
        {"memory_id": "2", "x": None, "y": 1.0, "z": 2.0},
    ]

    enriched = search_service._apply_distance_metadata(data, 0.0, 0.0, 0.0)

    assert enriched[0]["distance"] == pytest.approx(0.0)
    assert "distance" not in enriched[1]


def test_search_by_anchor_ranks_and_limits(
    monkeypatch: pytest.MonkeyPatch, search_service: SearchService
):
    base_time = datetime.now(UTC)
    anchor_data = [
        {
            "memory_id": "a",
            "search_score": 0.1,
            "importance_score": 0.2,
            "created_at": base_time,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        },
        {
            "memory_id": "b",
            "search_score": 0.9,
            "importance_score": 0.8,
            "created_at": base_time,
            "x": 1.0,
            "y": 1.0,
            "z": 1.0,
        },
    ]

    monkeypatch.setattr(
        search_service,
        "_search_symbolic_anchor",
        lambda *args, **kwargs: anchor_data,
    )

    results = search_service._search_by_anchor(
        query="focus",
        namespace="default",
        category_filter=None,
        limit=1,
        search_short_term=True,
        search_long_term=True,
        start_timestamp=None,
        end_timestamp=None,
        min_importance=None,
        max_importance=None,
        anchors=None,
        x=0.0,
        y=0.0,
        z=0.0,
        max_distance=None,
        rank_weights=None,
    )

    assert len(results) == 1
    assert results[0]["memory_id"] == "b"
    assert results[0]["distance"] == pytest.approx(math.sqrt(3))


def test_search_by_keywords_returns_ranked_results(
    monkeypatch: pytest.MonkeyPatch, search_service: SearchService
):
    base_time = datetime.now(UTC)
    keyword_data = [
        {
            "memory_id": "a",
            "search_score": 0.2,
            "importance_score": 0.5,
            "created_at": base_time,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        },
        {
            "memory_id": "b",
            "search_score": 0.7,
            "importance_score": 0.9,
            "created_at": base_time,
            "x": 2.0,
            "y": 0.0,
            "z": 0.0,
        },
    ]

    monkeypatch.setattr(
        search_service,
        "_search_keyword_terms",
        lambda *args, **kwargs: keyword_data,
    )

    results = search_service._search_by_keywords(
        keywords=["focus"],
        namespace="default",
        category_filter=None,
        limit=5,
        search_short_term=True,
        search_long_term=True,
        start_timestamp=None,
        end_timestamp=None,
        min_importance=None,
        max_importance=None,
        anchors=None,
        x=0.0,
        y=0.0,
        z=0.0,
        max_distance=None,
        rank_weights=None,
    )

    assert [item["memory_id"] for item in results] == ["b", "a"]
    assert results[0]["distance"] == pytest.approx(2.0)


def test_run_fulltext_pipeline_executes_all_stages(
    monkeypatch: pytest.MonkeyPatch, search_service: SearchService
):
    fts_data = [
        {"memory_id": "fts", "x": 0.0, "y": 0.0, "z": 0.0},
    ]
    fuzzy_data = [
        {"memory_id": "fuzzy", "x": 1.0, "y": 1.0, "z": 1.0},
    ]
    like_data = [
        {"memory_id": "like", "x": 2.0, "y": 2.0, "z": 2.0},
    ]

    monkeypatch.setattr(
        search_service,
        "_search_sqlite_fts",
        lambda *args, **kwargs: fts_data,
    )
    monkeypatch.setattr(
        search_service,
        "_search_fuzzy",
        lambda *args, **kwargs: fuzzy_data,
    )
    monkeypatch.setattr(
        search_service,
        "_search_like_fallback",
        lambda *args, **kwargs: like_data,
    )

    results, attempted = search_service._run_fulltext_pipeline(
        query="focus",
        namespace="default",
        category_filter=None,
        limit=5,
        search_short_term=True,
        search_long_term=True,
        use_fuzzy=True,
        fuzzy_max_results=5,
        adjusted_min_similarity=50,
        start_timestamp=None,
        end_timestamp=None,
        min_importance=None,
        max_importance=None,
        anchors=None,
        x=0.0,
        y=0.0,
        z=0.0,
        max_distance=None,
        advanced_filters=False,
    )

    assert [item["memory_id"] for item in results] == ["fts", "fuzzy", "like"]
    assert attempted == ["fts", "fuzzy", "like"]
    assert results[1]["distance"] == pytest.approx(math.sqrt(3))


def test_run_fulltext_pipeline_skips_fts_with_advanced_filters(
    monkeypatch: pytest.MonkeyPatch, search_service: SearchService
):
    monkeypatch.setattr(
        search_service,
        "_search_sqlite_fts",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Should not call FTS")
        ),
    )
    monkeypatch.setattr(
        search_service,
        "_search_fuzzy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Should not call fuzzy")
        ),
    )
    like_data = [
        {"memory_id": "like", "x": 1.0, "y": 0.0, "z": 0.0},
    ]
    monkeypatch.setattr(
        search_service,
        "_search_like_fallback",
        lambda *args, **kwargs: like_data,
    )

    results, attempted = search_service._run_fulltext_pipeline(
        query="focus",
        namespace="default",
        category_filter=None,
        limit=3,
        search_short_term=True,
        search_long_term=True,
        use_fuzzy=False,
        fuzzy_max_results=0,
        adjusted_min_similarity=50,
        start_timestamp=None,
        end_timestamp=None,
        min_importance=None,
        max_importance=None,
        anchors=None,
        x=0.0,
        y=0.0,
        z=0.0,
        max_distance=None,
        advanced_filters=True,
    )

    assert [item["memory_id"] for item in results] == ["like"]
    assert attempted == ["like"]
    assert results[0]["distance"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("db_type", "sqlite_fallback", "expected_fragment"),
    [
        ("sqlite", False, "like '%'"),
        ("sqlite", True, "instr(lower(cast"),
        ("postgresql", False, "jsonb_exists_any"),
        ("mysql", False, "json_contains"),
    ],
)
def test_build_anchor_conditions_backend_variants(
    db_type: str, sqlite_fallback: bool, expected_fragment: str
):
    service = SearchService(session=None, database_type=db_type)
    conditions = service._build_anchor_conditions(
        ShortTermMemory.symbolic_anchors,
        ["alpha"],
        sqlite_fallback=sqlite_fallback,
        prefix=f"test_{db_type}",
    )

    assert conditions, f"Expected conditions for {db_type}"

    compiled = str(
        conditions[0].compile(compile_kwargs={"literal_binds": True})
    ).lower()
    assert expected_fragment in compiled
