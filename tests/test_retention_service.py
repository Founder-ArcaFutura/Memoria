from datetime import datetime, timedelta
from typing import Any

import pytest

from memoria.core.memory import Memoria
from memoria.database.models import (
    Cluster,
    ClusterMember,
    LongTermMemory,
    MemoryAccessEvent,
)
from memoria.heuristics.retention import (
    MemoryRetentionService,
    RetentionConfig,
    RetentionPolicyRule,
)


@pytest.fixture()
def memoria_instance(tmp_path):
    db_url = f"sqlite:///{tmp_path/'retention.db'}"
    mem = Memoria(database_connect=db_url)
    # Stop automatic scheduler to keep tests deterministic
    if mem._retention_scheduler:
        mem._retention_scheduler.stop()
    if mem._ingestion_scheduler:
        mem._ingestion_scheduler.stop()
    yield mem
    if mem._retention_scheduler:
        mem._retention_scheduler.stop()
    if mem._ingestion_scheduler:
        mem._ingestion_scheduler.stop()


def _insert_long_term_memory(
    mem,
    *,
    memory_id="mem-1",
    importance=0.8,
    y_coord=0.0,
    access_count=0,
    last_accessed=None,
):
    now = datetime.utcnow()
    with mem.db_manager.SessionLocal() as session:
        record = LongTermMemory(
            memory_id=memory_id,
            original_chat_id="chat",
            processed_data={"text": "hello world"},
            importance_score=importance,
            category_primary="test",
            retention_type="long_term",
            namespace=mem.namespace,
            timestamp=now,
            created_at=now,
            access_count=access_count,
            last_accessed=last_accessed,
            searchable_content="hello world",
            summary="hello world",
            novelty_score=0.5,
            relevance_score=0.5,
            actionability_score=0.5,
            x_coord=0.0,
            y_coord=y_coord,
            z_coord=0.0,
            symbolic_anchors=["test"],
        )
        session.merge(record)
        session.commit()
    return memory_id


def test_retrieval_logs_access_events(memoria_instance):
    mem = memoria_instance
    memory_id = _insert_long_term_memory(mem)

    results = mem.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=1.0)
    assert any(result["memory_id"] == memory_id for result in results)

    with mem.db_manager.SessionLocal() as session:
        stored = (
            session.query(LongTermMemory)
            .filter(
                LongTermMemory.memory_id == memory_id,
                LongTermMemory.namespace == mem.namespace,
            )
            .one()
        )
        assert stored.access_count == 1
        assert stored.last_accessed is not None

        events = (
            session.query(MemoryAccessEvent)
            .filter(
                MemoryAccessEvent.memory_id == memory_id,
                MemoryAccessEvent.namespace == mem.namespace,
            )
            .all()
        )
        assert len(events) == 1
        assert events[0].access_type == "retrieval"


def test_retention_service_decays_importance(memoria_instance):
    mem = memoria_instance
    past = datetime.utcnow() - timedelta(days=7)
    memory_id = _insert_long_term_memory(mem, importance=0.9, last_accessed=past)

    service = MemoryRetentionService(
        db_manager=mem.db_manager,
        namespace=mem.namespace,
        config=RetentionConfig(
            decay_half_life_hours=24.0,
            reinforcement_bonus=0.0,
            privacy_shift=0.0,
            importance_floor=0.05,
            cluster_gravity_lambda=1.0,
        ),
        cluster_enabled=False,
    )
    service.run_cycle()

    with mem.db_manager.SessionLocal() as session:
        updated = (
            session.query(LongTermMemory)
            .filter(LongTermMemory.memory_id == memory_id)
            .one()
        )
        assert updated.importance_score < 0.9
        assert updated.importance_score >= 0.05


def test_retention_service_reinforces_recent_usage(memoria_instance):
    mem = memoria_instance
    now = datetime.utcnow()
    memory_id = _insert_long_term_memory(
        mem,
        importance=0.2,
        y_coord=0.0,
        access_count=12,
        last_accessed=now,
    )

    service = MemoryRetentionService(
        db_manager=mem.db_manager,
        namespace=mem.namespace,
        config=RetentionConfig(
            decay_half_life_hours=72.0,
            reinforcement_bonus=0.3,
            privacy_shift=1.0,
            importance_floor=0.05,
            cluster_gravity_lambda=0.0,
        ),
        cluster_enabled=False,
    )
    service.run_cycle()

    with mem.db_manager.SessionLocal() as session:
        updated = (
            session.query(LongTermMemory)
            .filter(LongTermMemory.memory_id == memory_id)
            .one()
        )
        assert updated.importance_score > 0.2
        assert updated.y_coord is not None
        assert updated.y_coord >= 0.0


def test_retention_policy_blocks_privacy_changes(memoria_instance):
    mem = memoria_instance
    now = datetime.utcnow()
    memory_id = _insert_long_term_memory(
        mem,
        importance=0.4,
        y_coord=2.5,
        access_count=6,
        last_accessed=now,
    )

    policy = RetentionPolicyRule(
        name="privacy-cap",
        namespaces=(mem.namespace,),
        privacy_ceiling=1.0,
        action="block",
    )
    audits: list[dict[str, Any]] = []

    service = MemoryRetentionService(
        db_manager=mem.db_manager,
        namespace=mem.namespace,
        config=RetentionConfig(
            decay_half_life_hours=72.0,
            reinforcement_bonus=0.4,
            privacy_shift=1.2,
            importance_floor=0.05,
            cluster_gravity_lambda=0.0,
            policies=(policy,),
        ),
        cluster_enabled=False,
        audit_callback=audits.append,
    )

    service.run_cycle()

    with mem.db_manager.SessionLocal() as session:
        updated = (
            session.query(LongTermMemory)
            .filter(LongTermMemory.memory_id == memory_id)
            .one()
        )
        assert updated.y_coord == pytest.approx(2.5)

    assert audits, "policy audit should be recorded"
    event = audits[0]
    assert event["policy_name"] == "privacy-cap"
    assert event["action"] == "block"
    assert event["privacy"]["after"] > policy.privacy_ceiling


def test_retention_policy_escalates_without_mutation(memoria_instance):
    mem = memoria_instance
    past = datetime.utcnow() - timedelta(days=60)
    memory_id = _insert_long_term_memory(
        mem,
        importance=0.2,
        y_coord=1.5,
        access_count=1,
        last_accessed=past,
    )

    policy = RetentionPolicyRule(
        name="aging-review",
        namespaces=(mem.namespace,),
        lifecycle_days=30.0,
        action="escalate",
        escalate_to="compliance",
    )
    audits: list[dict[str, Any]] = []

    service = MemoryRetentionService(
        db_manager=mem.db_manager,
        namespace=mem.namespace,
        config=RetentionConfig(
            decay_half_life_hours=24.0,
            reinforcement_bonus=0.0,
            privacy_shift=0.0,
            importance_floor=0.05,
            cluster_gravity_lambda=0.0,
            policies=(policy,),
        ),
        cluster_enabled=False,
        audit_callback=audits.append,
    )

    service.run_cycle()

    with mem.db_manager.SessionLocal() as session:
        updated = (
            session.query(LongTermMemory)
            .filter(LongTermMemory.memory_id == memory_id)
            .one()
        )
        assert updated.importance_score == pytest.approx(0.2)

    assert audits, "escalation should be logged"
    event = audits[0]
    assert event["action"] == "escalate"
    assert event["escalate_to"] == "compliance"
    assert any("lifecycle" in violation for violation in event["violations"])


def test_retention_updates_clusters(memoria_instance):
    mem = memoria_instance
    now = datetime.utcnow()
    memory_id = _insert_long_term_memory(
        mem,
        importance=0.3,
        y_coord=0.0,
        access_count=8,
        last_accessed=now,
    )

    with mem.db_manager.SessionLocal() as session:
        cluster = Cluster(
            summary="test cluster",
            centroid={"x": 0.0, "y": 0.0, "z": 0.0},
            y_centroid=0.0,
            z_centroid=0.0,
            size=1,
            avg_importance=0.1,
            weight=0.1,
            total_tokens=0,
            total_chars=0,
        )
        session.add(cluster)
        session.flush()
        session.add(
            ClusterMember(
                cluster_id=cluster.id,
                memory_id=memory_id,
                summary="hello world",
                tokens=0,
                chars=0,
            )
        )
        session.commit()

    service = MemoryRetentionService(
        db_manager=mem.db_manager,
        namespace=mem.namespace,
        config=RetentionConfig(
            decay_half_life_hours=72.0,
            reinforcement_bonus=0.25,
            privacy_shift=0.5,
            importance_floor=0.05,
            cluster_gravity_lambda=0.0,
        ),
        cluster_enabled=True,
    )
    service.run_cycle()

    with mem.db_manager.SessionLocal() as session:
        cluster = session.query(Cluster).one()
        assert cluster.avg_importance > 0.1
        assert cluster.weight >= cluster.avg_importance
        assert cluster.centroid["x"] == pytest.approx(0.0)
        assert cluster.centroid["y"] == pytest.approx(cluster.y_centroid)
        assert cluster.centroid["z"] == pytest.approx(cluster.z_centroid)
        assert cluster.update_count >= 1
        assert cluster.last_updated is not None
