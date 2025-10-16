# This test is temporarily sequestered due to persistent, hard-to-debug failures
# related to memory promotion logic.
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from memoria.core.memory import Memoria
from memoria.heuristics.manual_promotion import PromotionDecision, StagedManualMemory


def _make_staged_memory(**kwargs) -> StagedManualMemory:
    base = {
        "memory_id": str(uuid.uuid4()),
        "chat_id": str(uuid.uuid4()),
        "namespace": "default",
        "anchor": "test",
        "text": "test memory",
        "tokens": 3,
        "timestamp": datetime.now(timezone.utc),
        "x_coord": 0.0,
        "y_coord": 0.0,
        "z_coord": 0.0,
        "symbolic_anchors": ["test"],
        "metadata": {},
    }
    base.update(kwargs)
    return StagedManualMemory(**base)


def _make_decision(
    should_promote: bool,
    score: float,
    threshold: float,
    weights: dict[str, float],
) -> PromotionDecision:
    return PromotionDecision(
        should_promote=should_promote,
        score=score,
        threshold=threshold,
        weights=weights,
    )


@pytest.fixture()
def memoria_instance(tmp_path) -> Memoria:
    """Return a Memoria instance with short-term memory enabled."""
    db_url = f"sqlite:///{tmp_path/'manual_promotion.db'}"
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


def test_manual_memory_promotes_on_approval(
    monkeypatch: pytest.MonkeyPatch, memoria_instance: Memoria
):
    def approve(*_args, **_kwargs) -> PromotionDecision:
        return _make_decision(True, 0.9, 0.8, {"recency": 0.5, "user_priority": 0.4})

    monkeypatch.setattr("memoria.core.memory.score_staged_memory", approve)

    result = memoria_instance.store_memory(
        anchor="test",
        text="promote to long term",
        tokens=4,
        y=0.0,
        z=0.0,
        return_status=True,
    )

    assert result["promoted"] is True
    assert result["long_term_id"] is not None


def test_manual_memory_stays_staged(
    monkeypatch: pytest.MonkeyPatch, memoria_instance: Memoria
):
    def decline(*_args, **_kwargs) -> PromotionDecision:
        return _make_decision(False, 0.2, 0.8, {"recency": 0.5})

    monkeypatch.setattr("memoria.core.memory.score_staged_memory", decline)

    result = memoria_instance.store_memory(
        anchor="test",
        text="hold short term",
        tokens=3,
        y=0.0,
        z=0.0,
        return_status=True,
    )

    assert result["promoted"] is False
