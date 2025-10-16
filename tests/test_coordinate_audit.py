from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import pytest

from memoria import Memoria
from memoria.database.models import LongTermMemory, SpatialMetadata
from memoria.rituals.coordinate_audit import (
    CoordinateAuditJob,
    CoordinateAuditResponse,
    CoordinateAuditResult,
)


class StubCompletions:
    def __init__(self, responses: Sequence[CoordinateAuditResponse]):
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def parse(
        self, *, model, messages, response_format, temperature
    ):  # noqa: D401
        self.calls.append(messages)
        response = (
            self._responses.pop(0) if self._responses else CoordinateAuditResponse()
        )
        return StubCompletion(response)


class StubMessage:
    def __init__(self, parsed: CoordinateAuditResponse):
        self.parsed = parsed
        self.refusal = None


class StubChoice:
    def __init__(self, parsed: CoordinateAuditResponse):
        self.message = StubMessage(parsed)


class StubCompletion:
    def __init__(self, parsed: CoordinateAuditResponse):
        self.choices = [StubChoice(parsed)]


class StubChat:
    def __init__(self, completions: StubCompletions):
        self.completions = completions


class StubBeta:
    def __init__(self, completions: StubCompletions):
        self.chat = StubChat(completions)


class StubAsyncClient:
    def __init__(self, responses: Sequence[CoordinateAuditResponse]):
        self.completions = StubCompletions(responses)
        self.beta = StubBeta(self.completions)


class StubMemoryAgent:
    def __init__(self, responses: Sequence[CoordinateAuditResponse]):
        self.async_client = StubAsyncClient(responses)
        self.model = "test-model"


@pytest.fixture()
def memoria_instance():
    mem = Memoria(
        database_connect="sqlite:///:memory:",
        enable_short_term=False,
        coordinate_audit_enabled=False,
    )
    try:
        yield mem
    finally:
        mem._stop_schedulers()


def test_coordinate_audit_window_selection(memoria_instance):
    mem = memoria_instance
    earlier_id = mem.store_memory(
        anchor="early",
        text="Old entry",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        symbolic_anchors=["history"],
    )
    recent_id = mem.store_memory(
        anchor="recent",
        text="Recent entry",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        symbolic_anchors=["today"],
    )

    job = CoordinateAuditJob(mem.storage_service, StubMemoryAgent([]), lookback_days=7)

    initial_candidates = job._collect_candidates(window_start=None)
    assert {c.memory_id for c in initial_candidates} == {earlier_id, recent_id}

    last_run = datetime.utcnow() - timedelta(days=1)
    mem.storage_service.set_service_metadata_value(
        job.METADATA_KEY, last_run.isoformat()
    )

    with mem.db_manager.SessionLocal() as session:
        older = session.query(LongTermMemory).filter_by(memory_id=earlier_id).one()
        boundary = datetime.utcnow() - timedelta(days=14)
        older.created_at = boundary
        older.timestamp = boundary
        session.commit()

    window_start = job._determine_window_start(last_run)
    delta_candidates = job._collect_candidates(window_start)
    assert [c.memory_id for c in delta_candidates] == [recent_id]


def test_coordinate_audit_updates_and_metadata(memoria_instance):
    mem = memoria_instance
    first_id = mem.store_memory(
        anchor="first",
        text="Initial text",
        tokens=3,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        symbolic_anchors=["alpha"],
    )
    second_id = mem.store_memory(
        anchor="second",
        text="Second text",
        tokens=3,
        x_coord=1.0,
        y=1.0,
        z=1.0,
        symbolic_anchors=["beta"],
    )

    response = CoordinateAuditResponse(
        audits=[
            CoordinateAuditResult(
                memory_id=first_id,
                temporal=1.5,
                privacy=2.5,
                cognitive=-3.5,
            ),
            CoordinateAuditResult(
                memory_id=second_id,
                temporal=-4.0,
                privacy=-2.0,
                cognitive=0.5,
                confidence=0.9,
            ),
        ]
    )
    agent = StubMemoryAgent([response])
    job = CoordinateAuditJob(mem.storage_service, agent, lookback_days=7)

    summary = job.run()
    assert summary == {"processed": 2, "updated": 2, "failed": 0}
    assert len(agent.async_client.completions.calls) == 1

    with mem.db_manager.SessionLocal() as session:
        first = session.query(LongTermMemory).filter_by(memory_id=first_id).one()
        second = session.query(LongTermMemory).filter_by(memory_id=second_id).one()
        assert first.x_coord == pytest.approx(1.5)
        assert first.y_coord == pytest.approx(2.5)
        assert first.z_coord == pytest.approx(-3.5)
        assert second.x_coord == pytest.approx(-4.0)
        assert second.y_coord == pytest.approx(-2.0)
        assert second.z_coord == pytest.approx(0.5)

        spatial_rows = (
            session.query(SpatialMetadata)
            .filter(SpatialMetadata.memory_id.in_([first_id, second_id]))
            .all()
        )
        assert len(spatial_rows) == 2
        coords = {row.memory_id: (row.x, row.y, row.z) for row in spatial_rows}
        assert coords[first_id] == pytest.approx((1.5, 2.5, -3.5))
        assert coords[second_id] == pytest.approx((-4.0, -2.0, 0.5))

    stored = mem.storage_service.get_service_metadata_value(job.METADATA_KEY)
    assert stored is not None
    recorded = datetime.fromisoformat(stored)
    assert datetime.utcnow() - recorded < timedelta(minutes=1)
