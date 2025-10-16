from __future__ import annotations

import importlib
from datetime import datetime, timedelta

import pytest

from memoria import Memoria
from memoria.database.analytics import (
    get_analytics_summary,
    get_category_counts,
    get_retention_trends,
    get_usage_frequency,
)
from memoria.database.models import DatabaseManager, LongTermMemory, ShortTermMemory


def _seed_records(session, now: datetime, namespace: str = "default") -> None:
    session.add_all(
        [
            LongTermMemory(
                memory_id="lt-1",
                original_chat_id="chat-1",
                processed_data={"text": "work update"},
                category_primary="work",
                retention_type="long_term",
                namespace=namespace,
                timestamp=now - timedelta(days=2),
                created_at=now - timedelta(days=2),
                searchable_content="work update",
                summary="work update",
                access_count=5,
                last_accessed=now - timedelta(hours=6),
            ),
            LongTermMemory(
                memory_id="lt-2",
                original_chat_id="chat-2",
                processed_data={"text": "personal note"},
                category_primary="personal",
                retention_type="long_term",
                namespace=namespace,
                timestamp=now - timedelta(days=1),
                created_at=now - timedelta(days=1),
                searchable_content="personal note",
                summary="personal note",
                access_count=2,
                last_accessed=now - timedelta(hours=3),
            ),
            LongTermMemory(
                memory_id="lt-3",
                original_chat_id="chat-3",
                processed_data={"text": "work retro"},
                category_primary="work",
                retention_type="archival",
                namespace=namespace,
                timestamp=now - timedelta(days=3),
                created_at=now - timedelta(days=3),
                searchable_content="work retro",
                summary="work retro",
                access_count=1,
                last_accessed=now - timedelta(days=1),
            ),
            ShortTermMemory(
                memory_id="st-1",
                chat_id="chat-1",
                processed_data={"text": "standup summary"},
                category_primary="work",
                retention_type="short_term",
                namespace=namespace,
                created_at=now - timedelta(hours=12),
                expires_at=now + timedelta(days=1),
                searchable_content="standup summary",
                summary="standup summary",
                access_count=4,
                last_accessed=now - timedelta(hours=1),
            ),
            ShortTermMemory(
                memory_id="st-2",
                chat_id="chat-4",
                processed_data={"text": "shopping list"},
                category_primary="personal",
                retention_type="short_term",
                namespace=namespace,
                created_at=now - timedelta(hours=30),
                expires_at=now + timedelta(hours=6),
                searchable_content="shopping list",
                summary="shopping list",
                access_count=1,
                last_accessed=None,
            ),
        ]
    )
    session.commit()


@pytest.fixture
def seeded_session(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path/'analytics.db'}")
    manager.create_tables()
    session = manager.get_session()
    now = datetime.utcnow()
    _seed_records(session, now)
    try:
        yield session
    finally:
        session.close()


def test_category_counts_distinguishes_memory_types(seeded_session):
    counts = get_category_counts(seeded_session)
    long_term = {item.category: item.count for item in counts["long_term"]}
    short_term = {item.category: item.count for item in counts["short_term"]}

    assert long_term["work"] == 2
    assert long_term["personal"] == 1
    assert short_term["work"] == 1
    assert short_term["personal"] == 1


def test_retention_trends_returns_recent_window(seeded_session):
    trends = get_retention_trends(seeded_session, days=4)
    long_term_series = trends["long_term"]["series"]
    totals = {
        series.retention_type: sum(series.daily_counts.values())
        for series in long_term_series
    }
    assert totals["long_term"] == 2
    assert totals["archival"] == 1


def test_usage_frequency_orders_by_access_count(seeded_session):
    usage = get_usage_frequency(seeded_session, top_n=2)
    top_ids = [record.memory_id for record in usage["long_term"]["top_records"]]
    assert top_ids[0] == "lt-1"
    assert usage["long_term"]["total_records"] == 3
    assert usage["short_term"]["total_records"] == 2


def test_analytics_summary_combines_metrics(seeded_session):
    summary = get_analytics_summary(seeded_session, days=3, top_n=1)
    assert summary["category_counts"]["long_term"][0].category == "work"
    assert summary["usage_frequency"]["long_term"]["top_records"][0].memory_id == "lt-1"


@pytest.fixture
def analytics_client(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path/'analytics_api.db'}"
    mem = Memoria(database_connect=db_url, enable_short_term=True)
    mem.enable()
    now = datetime.utcnow()
    with mem.db_manager.SessionLocal() as session:
        _seed_records(session, now)

    import memoria_api

    importlib.reload(memoria_api)
    monkeypatch.setattr(memoria_api, "memoria", mem)
    memoria_api.app.config["memoria"] = mem
    client = memoria_api.app.test_client()
    try:
        yield client
    finally:
        if getattr(mem, "_retention_scheduler", None):
            mem._retention_scheduler.stop()


def test_summary_endpoint_returns_expected_shape(analytics_client):
    response = analytics_client.get("/analytics/summary?days=5&top=2")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert "category_counts" in payload
    assert "retention_trends" in payload
    assert "usage_frequency" in payload


def test_usage_endpoint_respects_top_parameter(analytics_client):
    response = analytics_client.get("/analytics/usage?top=1")
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["usage"]["long_term"]["top_records"]) == 1
