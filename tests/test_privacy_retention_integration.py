from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

import pytest

from memoria import Memoria
from memoria.config.settings import SyncSettings
from memoria.database.models import LongTermMemory, RetentionPolicyAudit
from memoria.heuristics.retention import (
    MemoryRetentionService,
    RetentionConfig,
    RetentionPolicyRule,
)


@pytest.fixture()
def memoria_privacy_fixture(tmp_path) -> Iterable[Memoria]:
    db_url = f"sqlite:///{tmp_path/'privacy_retention.db'}"
    mem = Memoria(database_connect=db_url, enable_short_term=False)
    if mem._retention_scheduler:
        mem._retention_scheduler.stop()
    if mem._ingestion_scheduler:
        mem._ingestion_scheduler.stop()
    try:
        yield mem
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()
        if mem._ingestion_scheduler:
            mem._ingestion_scheduler.stop()


def _insert_memory(
    mem: Memoria,
    *,
    memory_id: str,
    importance: float,
    y_coord: float,
    last_accessed: datetime,
) -> None:
    with mem.db_manager.SessionLocal() as session:
        record = LongTermMemory(
            memory_id=memory_id,
            original_chat_id="test-chat",
            processed_data={"text": "multi privacy"},
            importance_score=importance,
            category_primary="integration",
            retention_type="long_term",
            namespace=mem.namespace,
            team_id=None,
            workspace_id=None,
            timestamp=last_accessed,
            created_at=last_accessed,
            last_accessed=last_accessed,
            access_count=3,
            searchable_content="multi privacy",
            summary="multi privacy",
            novelty_score=0.5,
            relevance_score=0.5,
            actionability_score=0.5,
            x_coord=0.0,
            y_coord=y_coord,
            z_coord=0.0,
            symbolic_anchors=["integration"],
        )
        session.merge(record)
        session.commit()


def test_multi_privacy_ingestion_respects_sync_thresholds(memoria_privacy_fixture):
    mem = memoria_privacy_fixture
    events: list[tuple[str, str, str | None, dict[str, float | str]]] = []

    def _capture(action, entity_type, entity_id, payload):
        events.append((str(action), entity_type, entity_id, payload))

    mem.storage_service.set_sync_publisher(_capture)
    sync_settings = SyncSettings(
        realtime_replication=True,
        privacy_floor=-5.0,
        privacy_ceiling=10.0,
    )
    mem.storage_service.configure_sync_policy(sync_settings)

    private_id = mem.storage_service.store_memory(
        anchor="private-note",
        text="private details",
        tokens=5,
        y=-12.0,
        z=0.0,
    )
    public_id = mem.storage_service.store_memory(
        anchor="public-note",
        text="public announcement",
        tokens=7,
        y=6.0,
        z=1.0,
    )

    try:
        with mem.db_manager.SessionLocal() as session:
            stored = (
                session.query(LongTermMemory)
                .filter(LongTermMemory.memory_id.in_([private_id, public_id]))
                .all()
            )
            coords = {record.memory_id: record.y_coord for record in stored}
            assert coords[private_id] == pytest.approx(-12.0)
            assert coords[public_id] == pytest.approx(6.0)

        assert all(
            event[2] != private_id for event in events
        ), "Private ingestion should not emit sync events"
        public_events = [event for event in events if event[2] == public_id]
        assert public_events, "Public memory should emit at least one sync event"
        _, _, _, payload = public_events[-1]
        assert payload.get("privacy") == pytest.approx(6.0)
    finally:
        mem.storage_service.set_sync_publisher(None)


@pytest.fixture()
def retention_escalation_runner(memoria_privacy_fixture):
    mem = memoria_privacy_fixture

    def _run(policies: tuple[RetentionPolicyRule, ...]):
        audits: list[dict[str, object]] = []
        memory_id = "escalation-memory"
        past = datetime.utcnow() - timedelta(days=45)
        _insert_memory(
            mem,
            memory_id=memory_id,
            importance=0.35,
            y_coord=3.5,
            last_accessed=past,
        )
        service = MemoryRetentionService(
            db_manager=mem.db_manager,
            namespace=mem.namespace,
            config=RetentionConfig(
                decay_half_life_hours=24.0,
                reinforcement_bonus=0.0,
                privacy_shift=0.0,
                importance_floor=0.05,
                cluster_gravity_lambda=0.0,
                policies=policies,
            ),
            cluster_enabled=False,
            audit_callback=audits.append,
        )
        service.run_cycle()
        return memory_id, audits

    return _run


def test_retention_escalation_branching_includes_metadata(retention_escalation_runner):
    policies = (
        RetentionPolicyRule(
            name="lifecycle-review",
            namespaces=("*",),
            lifecycle_days=30.0,
            action="escalate",
            escalate_to="compliance",
            metadata={"severity": "critical"},
        ),
        RetentionPolicyRule(
            name="privacy-ceiling",
            namespaces=("*",),
            privacy_ceiling=2.0,
            action="block",
            metadata={"severity": "warning"},
        ),
    )

    memory_id, audits = retention_escalation_runner(policies)

    assert len(audits) == 1
    escalate_event = audits[0]

    assert escalate_event["action"] == "escalate"
    assert escalate_event["metadata"] == {"severity": "critical"}
    assert escalate_event["privacy"]["before"] == pytest.approx(3.5)
    assert escalate_event["age_days"] >= 30.0


def test_storage_service_records_policy_metadata(memoria_privacy_fixture):
    mem = memoria_privacy_fixture
    payload = {
        "memory_id": "policy-metadata",
        "namespace": mem.namespace,
        "policy_name": "privacy-audit",
        "action": "escalate",
        "escalate_to": "guardian",
        "violations": ["privacy 4.0 exceeds ceiling 3.0"],
        "importance": {"before": 0.3, "after": 0.3},
        "privacy": {"before": 4.0, "after": 4.0},
        "age_days": 45.0,
        "metadata": {"severity": "medium"},
    }

    mem.storage_service.record_retention_audit(payload)

    with mem.db_manager.SessionLocal() as session:
        audits = session.query(RetentionPolicyAudit).all()
        assert len(audits) == 1
        stored = audits[0]
        assert stored.policy_name == "privacy-audit"
        assert stored.escalate_to == "guardian"
        assert stored.details["metadata"] == {"severity": "medium"}
        assert stored.details["violations"] == payload["violations"]
