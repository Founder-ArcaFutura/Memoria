import copy
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.database.search_service import SearchService


def _service():
    return SearchService(session=None, database_type="sqlite")


def test_anchor_boost_affects_ranking():
    service = _service()
    now = datetime.utcnow()
    results = [
        {
            "search_score": 0.2,
            "importance_score": 0.2,
            "created_at": now,
            "processed_data": {"symbolic_anchors": ["foo"]},
        },
        {
            "search_score": 0.2,
            "importance_score": 0.2,
            "created_at": now,
            "processed_data": {"symbolic_anchors": ["bar"]},
        },
    ]
    ranked = service._rank_and_limit_results(copy.deepcopy(results), 2, "foo")
    assert ranked[0]["matched_via_anchor"] is True
    assert ranked[0]["composite_score"] > ranked[1]["composite_score"]


def test_custom_weights_change_order():
    service = _service()
    now = datetime.utcnow()
    res1 = {
        "search_score": 0.9,
        "importance_score": 0.1,
        "created_at": now,
        "processed_data": {},
    }
    res2 = {
        "search_score": 0.1,
        "importance_score": 0.9,
        "created_at": now,
        "processed_data": {},
    }
    ranked_default = service._rank_and_limit_results(
        copy.deepcopy([res1, res2]), 2, "test"
    )
    assert ranked_default[0]["search_score"] == 0.9

    custom_weights = {"search": 0.1, "importance": 0.8, "recency": 0.1}
    ranked_custom = service._rank_and_limit_results(
        copy.deepcopy([res1, res2]), 2, "test", rank_weights=custom_weights
    )
    assert ranked_custom[0]["importance_score"] == 0.9


def test_search_memories_applies_custom_rank_weights():
    service = _service()
    now = datetime.utcnow()
    res1 = {
        "memory_id": 1,
        "memory_type": "short_term",
        "processed_data": {},
        "importance_score": 0.1,
        "created_at": now,
        "summary": "",
        "category_primary": "",
        "search_score": 0.9,
        "search_strategy": "sqlite_fts5",
    }
    res2 = {
        "memory_id": 2,
        "memory_type": "short_term",
        "processed_data": {},
        "importance_score": 0.9,
        "created_at": now,
        "summary": "",
        "category_primary": "",
        "search_score": 0.1,
        "search_strategy": "sqlite_fts5",
    }

    service._search_sqlite_fts = lambda *args, **kwargs: [res1, res2]
    service._search_fuzzy = lambda *args, **kwargs: []
    service._search_like_fallback = lambda *args, **kwargs: []

    default = service.search_memories("test", limit=2)["results"]
    assert [r["memory_id"] for r in default] == [1, 2]

    weights = {"search": 0.1, "importance": 0.8, "recency": 0.1}
    custom = service.search_memories("test", limit=2, rank_weights=weights)["results"]
    assert [r["memory_id"] for r in custom] == [2, 1]


def test_vector_similarity_influences_ranking():
    service = SearchService(
        session=None, database_type="sqlite", vector_search_enabled=True
    )
    now = datetime.utcnow()
    results = [
        {
            "memory_id": "semantic",
            "search_score": 0.1,
            "importance_score": 0.2,
            "created_at": now,
            "processed_data": {},
            "embedding": [1.0, 0.0],
        },
        {
            "memory_id": "important",
            "search_score": 0.2,
            "importance_score": 0.9,
            "created_at": now,
            "processed_data": {},
            "embedding": [0.0, 1.0],
        },
    ]

    ranked = service._rank_and_limit_results(
        copy.deepcopy(results),
        2,
        "query",
        None,
        query_embedding=[1.0, 0.0],
    )
    assert ranked[0]["memory_id"] == "semantic"
