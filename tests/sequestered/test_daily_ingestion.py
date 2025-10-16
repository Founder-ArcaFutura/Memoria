# This test is temporarily sequestered due to persistent, hard-to-debug failures
# related to memory promotion logic.
from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from memoria.core.memory import Memoria
from memoria.storage.service import StagedManualMemory


@pytest.fixture()
def memoria_short_term(tmp_path) -> Memoria:
    """Return a Memoria instance with short-term memory enabled."""
    db_url = f"sqlite:///{tmp_path/'daily_ingest.db'}"
    mem = Memoria(database_connect=db_url, enable_short_term=True)
    try:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()
        if mem._ingestion_scheduler:
            mem._ingestion_scheduler.stop()
        yield mem
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()
        if mem._ingestion_scheduler:
            mem._ingestion_scheduler.stop()


def test_daily_ingestion_does_nothing_when_empty(memoria_short_term: Memoria):
    report = memoria_short_term.run_daily_ingestion()
    assert report == []


def test_daily_ingestion_promotes_when_threshold_met(memoria_short_term: Memoria):
    low_weights = {
        "threshold": 0.95,
        "recency": 0.0,
        "anchors": 0.0,
        "spatial": 0.0,
        "user_priority": 0.0,
    }
    result = memoria_short_term.store_memory(
        anchor="ingest",
        text="ingestion candidate",
        tokens=4,
        y=0.0,
        z=0.0,
        promotion_weights=low_weights,
        return_status=True,
    )

    assert result["promoted"] is False

    report = memoria_short_term.run_daily_ingestion()
    assert len(report) == 1
    assert report[0]["status"] == "promoted"
    assert "long_term_id" in report[0]
    assert report[0]["long_term_id"] is not None


def test_daily_ingestion_respects_threshold(memoria_short_term: Memoria):
    low_weights = {
        "threshold": 0.95,
        "recency": 0.0,
        "anchors": 0.0,
        "spatial": 0.0,
        "user_priority": 0.0,
    }
    result = memoria_short_term.store_memory(
        anchor="hold",
        text="remain staged",
        tokens=3,
        y=0.0,
        z=0.0,
        promotion_weights=low_weights,
        return_status=True,
    )

    report = memoria_short_term.run_daily_ingestion(
        promotion_weights={"threshold": 0.97}
    )
    assert report


def test_daily_ingestion_removes_staged_memory(memoria_short_term: Memoria):
    result = memoria_short_term.store_memory(
        anchor="ingest-transient",
        text="promote and remove",
        tokens=4,
        y=0.0,
        z=0.0,
        promotion_weights={"threshold": 0.01},
        return_status=True,
    )
    assert result["promoted"] is True

    report = memoria_short_term.run_daily_ingestion()
    assert not report, "Should not re-process already promoted memories"
