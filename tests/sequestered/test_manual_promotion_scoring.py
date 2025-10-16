import uuid
from datetime import datetime

import pytest

from memoria import Memoria
from memoria.database.models import Cluster, ClusterMember, LongTermMemory
from memoria.heuristics.manual_promotion import score_staged_memory


@pytest.fixture()
def memoria_instance(tmp_path_factory) -> Memoria:
    db_path = tmp_path_factory.mktemp("data") / "test.db"
    mem = Memoria(database_connect=f"sqlite:///{db_path}")
    mem.enable()
    return mem


def test_cluster_gravity_increases_score(memoria_instance: Memoria) -> None:
    mem = memoria_instance
    with mem.db_manager.SessionLocal() as session:
        cluster = Cluster(
            summary="gravity cluster",
            centroid={"x": 0.0, "y": 0.0, "z": 0.0},
            y_centroid=0.0,
            z_centroid=0.0,
            weight=5.0,
            size=1,
        )
        session.add(cluster)
        session.flush()
        session.add(
            ClusterMember(
                memory_id="existing",
                anchor="shared",
                summary="existing summary",
                cluster_id=cluster.id,
            )
        )
        session.commit()

    staged_near = mem.storage_service.stage_manual_memory(
        "shared",
        "near cluster",
        12,
        x_coord=0.1,
        y=0.05,
        z=0.05,
        symbolic_anchors=["shared"],
    )
    decision_near = score_staged_memory(
        staged_near,
        storage_service=mem.storage_service,
    )
    mem.storage_service.remove_short_term_memory(staged_near.memory_id)

    staged_far = mem.storage_service.stage_manual_memory(
        "shared",
        "far cluster",
        12,
        x_coord=12.0,
        y=12.0,
        z=12.0,
        symbolic_anchors=["shared"],
    )
    decision_far = score_staged_memory(
        staged_far,
        storage_service=mem.storage_service,
    )

    assert decision_near.score > decision_far.score
    assert decision_near.should_promote is True
    assert decision_far.should_promote is False


def test_anchor_recurrence_increases_score(memoria_instance: Memoria) -> None:
    mem = memoria_instance
    with mem.db_manager.SessionLocal() as session:
        session.add(
            LongTermMemory(
                memory_id=str(uuid.uuid4()),
                processed_data={},
                importance_score=0.6,
                category_primary="test",
                namespace="default",
                timestamp=datetime.utcnow(),
                created_at=datetime.utcnow(),
                searchable_content="existing",
                summary="existing",
                symbolic_anchors=["shared"],
            )
        )
        session.commit()

    staged_shared = mem.storage_service.stage_manual_memory(
        "shared",
        "recurring anchor",
        6,
        symbolic_anchors=["shared"],
    )
    decision_shared = score_staged_memory(
        staged_shared,
        storage_service=mem.storage_service,
    )
    mem.storage_service.remove_short_term_memory(staged_shared.memory_id)

    staged_unique = mem.storage_service.stage_manual_memory(
        "novel",
        "isolated anchor",
        6,
        symbolic_anchors=["novel"],
    )
    decision_unique = score_staged_memory(
        staged_unique,
        storage_service=mem.storage_service,
    )

    assert decision_shared.score > decision_unique.score
    assert decision_shared.should_promote is True
    assert decision_unique.should_promote is False
