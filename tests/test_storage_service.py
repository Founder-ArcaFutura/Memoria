import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from memoria import Memoria
from memoria.config.manager import ConfigManager
from memoria.config.settings import (
    DatabaseSettings,
    IngestMode,
    MemoriaSettings,
    SyncSettings,
)
from memoria.core.database import DatabaseManager
from memoria.database.models import (
    LongTermMemory,
    MemoryAccessEvent,
    RetentionPolicyAudit,
    ShortTermMemory,
    SpatialMetadata,
    Team,
    Workspace,
)
from memoria.heuristics.retention import RetentionPolicyRule
from memoria.storage.service import StorageService
from memoria.sync import SyncEvent, SyncEventAction
from memoria.utils.exceptions import MemoriaError
from memoria.utils.pydantic_models import (
    MemoryClassification,
    MemoryImportanceLevel,
    ProcessedLongTermMemory,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script_module(name: str):
    script_path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class DummySearchEngine:
    def __init__(self, response):
        self.response = response

    def execute_search(self, query, db_manager, namespace, limit):
        return self.response


class DummyDBManager:
    def search_memories(self, *args, **kwargs):
        return []


def test_retrieve_context_merges_search_results_and_essential(monkeypatch):
    search_results = {"results": [{"memory_id": "search1"}]}
    essential = [{"memory_id": "essential1"}]

    db_manager = DummyDBManager()
    search_engine = DummySearchEngine(search_results)
    service = StorageService(
        db_manager, search_engine=search_engine, conscious_ingest=True
    )
    monkeypatch.setattr(service, "get_essential_conversations", lambda limit: essential)

    context = service.retrieve_context("hello")

    assert {"memory_id": "search1"} in context
    assert {"memory_id": "essential1"} in context
    assert len(context) == 2


def test_update_memory_updates_fields():
    mem = Memoria(database_connect="sqlite:///:memory:", enable_short_term=False)

    memory_id = mem.store_memory(
        anchor="initial",
        text="original text",
        tokens=3,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        symbolic_anchors=["initial"],
    )

    with mem.db_manager.SessionLocal() as session:
        record = session.query(LongTermMemory).filter_by(memory_id=memory_id).one()
        base_timestamp = record.timestamp

    updated = mem.storage_service.update_memory(
        memory_id,
        {
            "text": "updated text",
            "tokens": 7,
            "timestamp": base_timestamp + timedelta(days=2),
            "x_coord": 1.5,
            "y_coord": 4.5,
            "z_coord": -2.0,
            "symbolic_anchors": ["project", "updated"],
        },
    )

    assert updated is True

    with mem.db_manager.SessionLocal() as session:
        refreshed = session.query(LongTermMemory).filter_by(memory_id=memory_id).one()

        assert refreshed.summary == "updated text"
        assert refreshed.searchable_content == "updated text"
        assert refreshed.x_coord == 1.5
        assert refreshed.y_coord == 4.5
        assert refreshed.z_coord == -2.0
        assert refreshed.symbolic_anchors == ["project", "updated"]
        assert refreshed.timestamp == base_timestamp + timedelta(days=2)

        processed = refreshed.processed_data
        assert processed.get("text") == "updated text"
        assert processed.get("tokens") == 7


def test_store_memory_blocked_by_retention_policy():
    mem = Memoria(database_connect="sqlite:///:memory:", enable_short_term=False)
    try:
        policy = RetentionPolicyRule(
            name="privacy-hard-cap",
            namespaces=(mem.namespace,),
            privacy_ceiling=0.0,
            action="block",
        )
        mem.storage_service.configure_retention_policies((policy,))

        with pytest.raises(MemoriaError):
            mem.storage_service.store_memory(
                anchor="policy",
                text="should be blocked",
                tokens=5,
                x_coord=0.0,
                y=5.0,
                z=0.0,
                symbolic_anchors=["policy"],
            )

        with mem.db_manager.SessionLocal() as session:
            audits = session.query(RetentionPolicyAudit).all()
            assert len(audits) == 1
            assert audits[0].action == "block"
            assert audits[0].policy_name == "privacy-hard-cap"
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()
        if mem._ingestion_scheduler:
            mem._ingestion_scheduler.stop()


def test_store_memory_escalates_policy_violation(tmp_path):
    mem = Memoria(
        database_connect=f"sqlite:///{tmp_path/'policy.db'}", enable_short_term=False
    )
    try:
        policy = RetentionPolicyRule(
            name="review-required",
            namespaces=(mem.namespace,),
            privacy_ceiling=0.0,
            action="escalate",
            escalate_to="risk-team",
        )
        mem.storage_service.configure_retention_policies((policy,))

        memory_id = mem.storage_service.store_memory(
            anchor="policy",
            text="should trigger escalation",
            tokens=5,
            x_coord=0.0,
            y=4.0,
            z=0.0,
            symbolic_anchors=["policy"],
        )
        assert memory_id

        with mem.db_manager.SessionLocal() as session:
            audits = session.query(RetentionPolicyAudit).all()
            assert len(audits) == 1
            assert audits[0].action == "escalate"
            assert audits[0].escalate_to == "risk-team"
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()
        if mem._ingestion_scheduler:
            mem._ingestion_scheduler.stop()


def test_store_memory_with_images(tmp_path):
    mem = Memoria(
        database_connect=f"sqlite:///{tmp_path/'images.db'}", enable_short_term=False
    )
    try:
        image_root = tmp_path / "assets"
        mem.storage_service._image_storage_root = image_root
        image_root.mkdir(parents=True, exist_ok=True)

        base64_pixel = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wIAAgMBAp+B7aEAAAAASUVORK5CYII="
        )

        result = mem.store_memory(
            anchor="image-anchor",
            text="memory containing an image",
            tokens=5,
            return_status=True,
            images=[
                {
                    "filename": "pixel.png",
                    "mime_type": "image/png",
                    "data": base64_pixel,
                    "caption": "single pixel",
                }
            ],
        )

        memory_id = result["memory_id"]
        snapshot = mem.storage_service.get_memory_snapshot(memory_id, refresh=True)

        assert snapshot["includes_image"] is True
        images = snapshot.get("images")
        assert images and images[0]["mime_type"] == "image/png"

        stored_path = image_root / images[0]["file_path"]
        assert stored_path.exists()
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()
        if mem._ingestion_scheduler:
            mem._ingestion_scheduler.stop()


def test_store_personal_memory_skips_short_term_and_records_spatial_metadata(tmp_path):
    mem = Memoria(database_connect=f"sqlite:///{tmp_path/'personal.db'}")
    try:
        mem.ingest_mode = IngestMode.PERSONAL
        mem.personal_mode_enabled = True
        mem.personal_documents_enabled = True

        timestamp = datetime.now(timezone.utc).replace(microsecond=0)
        entry = {
            "anchor": "personal-note",
            "text": "Direct long-term capture without staging.",
            "tokens": 9,
            "timestamp": timestamp,
            "x_coord": 0.0,
            "y_coord": 1.0,
            "z_coord": -0.5,
            "symbolic_anchors": ["personal", "direct"],
            "chat_id": "chat-direct-001",
            "metadata": {"topic": "personal-capture"},
            "documents": [
                {
                    "document_id": "doc-direct-001",
                    "title": "Direct Capture Source",
                    "url": "https://example.com/direct",
                }
            ],
        }

        result = mem.storage_service.store_personal_memory(entry)
        assert result["status"] == "stored"

        with mem.db_manager.SessionLocal() as session:
            assert session.query(ShortTermMemory).count() == 0

            long_term = (
                session.query(LongTermMemory)
                .filter_by(memory_id=result["memory_id"])
                .one()
            )
            assert long_term.original_chat_id == entry["chat_id"]
            documents = long_term.processed_data.get("documents")
            assert isinstance(documents, list) and documents
            assert documents[0].get("document_id") == "doc-direct-001"

            spatial = (
                session.query(SpatialMetadata)
                .filter_by(memory_id=result["memory_id"])
                .one()
            )
            assert spatial.x == pytest.approx(entry["x_coord"])
            assert spatial.y == pytest.approx(entry["y_coord"])
            assert spatial.namespace == result["namespace"]
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()
        if mem._ingestion_scheduler:
            mem._ingestion_scheduler.stop()


def test_record_memory_touches_accepts_memory_type(tmp_path):
    db_url = f"sqlite:///{tmp_path/'context.db'}"
    mem = Memoria(database_connect=db_url)
    try:
        now = datetime.utcnow()
        memory_id = "ctx-1"
        with mem.db_manager.SessionLocal() as session:
            record = LongTermMemory(
                memory_id=memory_id,
                original_chat_id="chat",
                processed_data={"text": "contextual insights"},
                importance_score=0.5,
                category_primary="test",
                retention_type="long_term",
                namespace=mem.namespace,
                timestamp=now,
                created_at=now,
                access_count=0,
                searchable_content="contextual insights",
                summary="contextual insights",
                novelty_score=0.5,
                relevance_score=0.5,
                actionability_score=0.5,
                x_coord=0.0,
                y_coord=0.0,
                z_coord=0.0,
                symbolic_anchors=["context"],
            )
            session.merge(record)
            session.commit()

        mem.storage_service._record_memory_touches(
            [{"memory_id": memory_id, "memory_type": "long_term"}],
            event_source="context_query",
        )

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
            assert events[0].source == "context_query"
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()


def test_store_long_term_memory_enhanced_uses_public_connection(tmp_path):
    db_url = f"sqlite:///{tmp_path/'txn.db'}"
    manager = DatabaseManager(db_url)
    manager.initialize_schema()

    memory = ProcessedLongTermMemory(
        content="Transaction coverage",
        summary="Ensure transaction uses public connection",
        classification=MemoryClassification.ESSENTIAL,
        importance=MemoryImportanceLevel.HIGH,
        topic="transactions",
        conversation_id="conv-txn",
        classification_reason="Validate get_connection delegation",
    )

    original_get_connection = manager.get_connection
    with patch.object(manager, "get_connection", wraps=original_get_connection) as spy:
        memory_id = manager.store_long_term_memory_enhanced(
            memory,
            chat_id="chat-txn",
            namespace="default",
        )

    assert spy.call_count >= 1

    with manager.get_connection() as connection:
        row = connection.execute(
            "SELECT memory_id FROM long_term_memory WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()

    assert row["memory_id"] == memory_id


def test_import_round_trip_from_export(tmp_path, monkeypatch):
    export_module = _load_script_module("export_memories")

    source_db = f"sqlite:///{tmp_path/'source.db'}"
    source = Memoria(database_connect=source_db, enable_short_term=False)
    try:
        inserted_id = source.store_memory(
            anchor="roundtrip",
            text="Round trip memory",
            tokens=3,
            x_coord=0.0,
            y=2.0,
            z=-1.0,
            symbolic_anchors=["roundtrip"],
        )

        export_path = tmp_path / "export.json"
        settings_stub = MemoriaSettings(
            database=DatabaseSettings(
                connection_string=source_db,
                backup_enabled=False,
                backup_interval_hours=24,
            )
        )
        config_stub = SimpleNamespace(
            get_settings=lambda: settings_stub, auto_load=lambda: None
        )
        original_get_instance = ConfigManager.get_instance
        monkeypatch.setattr(ConfigManager, "get_instance", lambda: config_stub)
        try:
            export_module.export_memories(
                export_path,
                namespaces=[],
                include_short_term=False,
                include_clusters=False,
                include_relationships=False,
            )
        finally:
            monkeypatch.setattr(ConfigManager, "get_instance", original_get_instance)
    finally:
        if source._retention_scheduler:
            source._retention_scheduler.stop()

    payload = json.loads(export_path.read_text(encoding="utf-8"))
    long_term_payload = payload.get("payload", {}).get("long_term_memories", [])

    destination_db = f"sqlite:///{tmp_path/'dest.db'}"
    destination = Memoria(
        database_connect=destination_db, enable_short_term=False, namespace="imported"
    )
    try:
        result = destination.storage_service.import_memories_bulk(
            long_term_payload,
            namespace_map={"default": "imported"},
            default_namespace="imported",
        )

        assert inserted_id in result["inserted"]
        assert not result["errors"]

        with destination.db_manager.SessionLocal() as session:
            stored = (
                session.query(LongTermMemory)
                .filter(LongTermMemory.namespace == "imported")
                .all()
            )

        assert len(stored) == len(result["inserted"])
        assert stored[0].summary == "Round trip memory"
        assert stored[0].symbolic_anchors == ["roundtrip"]
    finally:
        if destination._retention_scheduler:
            destination._retention_scheduler.stop()


def test_import_memories_bulk_skips_duplicates(tmp_path):
    db_url = f"sqlite:///{tmp_path/'dupe.db'}"
    mem = Memoria(database_connect=db_url, enable_short_term=False)
    try:
        timestamp = datetime.utcnow().replace(microsecond=0)
        payload = [
            {
                "memory_id": "dupe-1",
                "text": "Duplicate detection",
                "created_at": timestamp.isoformat(),
                "anchors": ["duplicate"],
            }
        ]

        first = mem.storage_service.import_memories_bulk(payload)
        assert first["inserted"] == ["dupe-1"]
        assert not first["errors"]

        second = mem.storage_service.import_memories_bulk(payload)
        assert second["inserted"] == []
        assert second["skipped"]
        assert "existing" in second["skipped"][0]["reason"]
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()


def test_import_memories_bulk_records_spatial_metadata(tmp_path):
    db_url = f"sqlite:///{tmp_path/'spatial.db'}"
    mem = Memoria(database_connect=db_url, enable_short_term=False)
    try:
        timestamp = datetime.utcnow().replace(microsecond=0)
        payload = [
            {
                "memory_id": "spatial-1",
                "text": "Spatially indexed import",
                "created_at": timestamp.isoformat(),
                "timestamp": timestamp.isoformat(),
                "x_coord": 1.25,
                "y_coord": -3.5,
                "z_coord": 4.75,
                "symbolic_anchors": ["spatial", "import"],
            }
        ]

        result = mem.storage_service.import_memories_bulk(payload)

        assert result["inserted"] == ["spatial-1"]

        with mem.db_manager.SessionLocal() as session:
            spatial_record = (
                session.query(SpatialMetadata)
                .filter(
                    SpatialMetadata.memory_id == "spatial-1",
                    SpatialMetadata.namespace == mem.storage_service.namespace,
                )
                .one_or_none()
            )

        assert spatial_record is not None
        assert spatial_record.timestamp == timestamp
        assert spatial_record.x == pytest.approx(1.25)
        assert spatial_record.y == pytest.approx(-3.5)
        assert spatial_record.z == pytest.approx(4.75)
        assert spatial_record.symbolic_anchors == ["spatial", "import"]
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()


def test_team_namespace_isolation_and_access_control(tmp_path):
    db_url = f"sqlite:///{tmp_path/'team.db'}"
    mem = Memoria(database_connect=db_url, enable_short_term=False, user_id="alice")
    try:
        mem.team_memory_enabled = True
        mem.team_enforce_membership = True
        mem.storage_service.configure_team_policy(
            namespace_prefix="team", enforce_membership=True
        )

        team = mem.register_team_space("ops", members=["alice"], share_by_default=True)
        assert team["namespace"] == "team:ops"

        shared = mem.store_memory(
            anchor="ops-brief",
            text="Shared update for the ops team",
            tokens=5,
            share_with_team=True,
            team_id="ops",
            return_status=True,
        )
        assert shared["namespace"] == "team:ops"

        with mem.db_manager.SessionLocal() as session:
            stored = (
                session.query(LongTermMemory)
                .filter(LongTermMemory.namespace == "team:ops")
                .one()
            )
            assert stored.summary == "Shared update for the ops team"

        mem.user_id = "bob"
        with pytest.raises(MemoriaError):
            mem.store_memory(
                anchor="unauthorized",
                text="Should not be shared",
                tokens=2,
                team_id="ops",
                share_with_team=True,
            )

        private_id = mem.store_memory(
            anchor="personal",
            text="Bob keeps this personal",
            tokens=3,
            team_id="ops",
            share_with_team=False,
        )
        with mem.db_manager.SessionLocal() as session:
            personal = (
                session.query(LongTermMemory)
                .filter(LongTermMemory.memory_id == private_id)
                .one()
            )
            assert personal.namespace == mem.namespace
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()


def test_team_context_filters_and_counts(team_memoria_context):
    mem, teams, backend = team_memoria_context

    mem.namespace = mem.personal_namespace
    mem.storage_service.namespace = mem.personal_namespace

    shared_one = mem.store_memory(
        anchor="ops-shared-context",
        text="ops shared context entry",
        tokens=12,
        team_id="ops",
        promotion_weights={"threshold": 0.0},
        return_status=True,
    )
    shared_two = mem.store_memory(
        anchor="ops-shared-second",
        text="second shared ops entry",
        tokens=14,
        team_id="ops",
        share_with_team=True,
        promotion_weights={"threshold": 0.0},
        return_status=True,
    )
    personal = mem.store_memory(
        anchor="ops-private-context",
        text="personal ops memo",
        tokens=10,
        team_id="ops",
        share_with_team=False,
        promotion_weights={"threshold": 0.0},
        return_status=True,
    )
    research_shared = mem.store_memory(
        anchor="research-shared-context",
        text="research shared memo",
        tokens=11,
        team_id="research",
        share_with_team=True,
        promotion_weights={"threshold": 0.0},
        return_status=True,
    )

    if backend == "sqlite":
        mem.storage_service.namespace = teams["ops"]["namespace"]
        ops_counts = mem.storage_service.count_anchor_occurrences(
            ["ops-shared-context", "ops-private-context"]
        )
        assert ops_counts.get("ops-shared-context", 0) >= 1
        assert ops_counts.get("ops-private-context", 0) == 0

        mem.storage_service.namespace = mem.personal_namespace
        personal_counts = mem.storage_service.count_anchor_occurrences(
            ["ops-private-context", "research-shared-context"]
        )
        assert personal_counts.get("ops-private-context", 0) >= 1
        assert personal_counts.get("research-shared-context", 0) == 0

    with mem.db_manager.SessionLocal() as session:
        ops_total = (
            session.query(LongTermMemory)
            .filter_by(namespace=teams["ops"]["namespace"])
            .count()
        )
        personal_total = (
            session.query(LongTermMemory)
            .filter_by(namespace=mem.personal_namespace)
            .count()
        )
        research_total = (
            session.query(LongTermMemory)
            .filter_by(namespace=teams["research"]["namespace"])
            .count()
        )

    assert ops_total >= 2
    assert personal_total >= 1
    assert research_total >= 1

    accessible = mem.get_accessible_namespaces()
    assert teams["ops"]["namespace"] in accessible
    assert teams["research"]["namespace"] in accessible


def test_workspace_namespace_resolution_honors_share_defaults(tmp_path):
    db_url = f"sqlite:///{tmp_path/'workspace_defaults.db'}"
    mem = Memoria(
        database_connect=db_url,
        enable_short_term=True,
        user_id="lead",
        team_memory_enabled=True,
        team_enforce_membership=True,
        team_namespace_prefix="workspace",
    )
    try:
        service = mem.storage_service
        service.configure_team_policy(
            namespace_prefix="workspace",
            enforce_membership=True,
            share_by_default=False,
        )

        atlas = service.register_team_space(
            "atlas",
            members=["member_a", "member_b"],
            admins=["lead"],
            share_by_default=True,
        )
        zenith = service.register_team_space(
            "zenith",
            members=["member_c"],
            admins=["lead"],
            share_by_default=False,
        )

        assert atlas.namespace == "workspace:atlas"
        assert atlas.share_by_default is True
        assert atlas.members == {"member_a", "member_b"}
        assert atlas.iter_members() == {"member_a", "member_b", "lead"}

        assert zenith.namespace == "workspace:zenith"
        assert zenith.share_by_default is False
        assert zenith.admins == {"lead"}

        personal_namespace = mem.namespace
        assert (
            service.resolve_target_namespace(team_id="atlas", user_id="member_a")
            == "workspace:atlas"
        )
        assert (
            service.resolve_target_namespace(team_id="atlas", user_id="lead")
            == "workspace:atlas"
        )
        assert (
            service.resolve_target_namespace(team_id="zenith", user_id="member_c")
            == personal_namespace
        )
        assert (
            service.resolve_target_namespace(
                team_id="zenith", user_id="member_c", share_with_team=True
            )
            == "workspace:zenith"
        )
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()


def test_workspace_access_control_and_sync_events(tmp_path):
    db_url = f"sqlite:///{tmp_path/'workspace_access.db'}"
    mem = Memoria(
        database_connect=db_url,
        enable_short_term=True,
        user_id="member_a",
        team_memory_enabled=True,
        team_enforce_membership=True,
        team_namespace_prefix="workspace",
    )
    try:
        service = mem.storage_service
        service.configure_team_policy(
            namespace_prefix="workspace", enforce_membership=True
        )

        service.register_team_space(
            "atlas",
            members=["member_a", "observer"],
            admins=["lead"],
            share_by_default=True,
        )

        with pytest.raises(MemoriaError):
            service.resolve_target_namespace(team_id="atlas", user_id="mallory")
        with pytest.raises(MemoriaError):
            service.require_team_access("atlas", "mallory")

        atlas_namespace = service.resolve_target_namespace(
            team_id="atlas", user_id="member_a"
        )
        team_snapshot = service.require_team_access("atlas", "member_a")
        assert team_snapshot.team_id == "atlas"

        events: list[tuple[SyncEventAction | str, str, str | None, dict[str, Any]]] = []

        def _capture(action, entity_type, entity_id, payload):
            events.append((action, entity_type, entity_id, payload))

        service.namespace = atlas_namespace
        service.team_id = "atlas"
        service.set_sync_publisher(_capture)

        memory_id = service.store_memory(
            anchor="atlas-shared",
            text="atlas workspace sync memo",
            tokens=7,
        )

        assert events, "store_memory should emit a sync event for the workspace"
        action, entity_type, entity_id, payload = events[-1]
        assert action == SyncEventAction.MEMORY_CREATED
        assert entity_type == "memory"
        assert entity_id == memory_id
        assert payload.get("team_id") == "atlas"
        assert payload.get("namespace") == "workspace:atlas"

        with mem.db_manager.SessionLocal() as session:
            stored = (
                session.query(LongTermMemory)
                .filter(LongTermMemory.memory_id == memory_id)
                .one()
            )
            assert stored.namespace == "workspace:atlas"
            assert stored.team_id == "atlas"
    finally:
        service.set_sync_publisher(None)
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()


def test_workspace_namespace_resolution_and_access_controls(tmp_path):
    db_url = f"sqlite:///{tmp_path/'workspace_namespace.db'}"
    mem = Memoria(
        database_connect=db_url,
        enable_short_term=True,
        user_id="atlas_admin",
    )
    try:
        mem.team_memory_enabled = True
        mem.team_enforce_membership = True
        service = mem.storage_service
        service.configure_team_policy(
            namespace_prefix="workspace", enforce_membership=True
        )

        atlas_space = service.register_team_space(
            "atlas",
            members=["atlas_admin", "ops_member"],
            admins=["workspace_owner"],
            share_by_default=True,
        )
        zenith_space = service.register_team_space(
            "zenith",
            members=["zen_member"],
            admins=["workspace_owner"],
            share_by_default=False,
        )

        personal_namespace = mem.personal_namespace
        assert atlas_space.namespace == "workspace:atlas"
        assert atlas_space.share_by_default is True
        assert zenith_space.namespace == "workspace:zenith"
        assert zenith_space.share_by_default is False

        assert (
            service.resolve_target_namespace(team_id="atlas", user_id="atlas_admin")
            == atlas_space.namespace
        )
        assert (
            service.resolve_target_namespace(team_id="zenith", user_id="zen_member")
            == personal_namespace
        )
        assert (
            service.resolve_target_namespace(
                team_id="zenith", user_id="zen_member", share_with_team=True
            )
            == zenith_space.namespace
        )
        assert (
            service.resolve_target_namespace(
                team_id="zenith", user_id="zen_member", share_with_team=False
            )
            == personal_namespace
        )

        authorized_space = service.require_team_access("atlas", "atlas_admin")
        assert authorized_space.team_id == "atlas"
        assert "workspace_owner" in authorized_space.admins

        with pytest.raises(MemoriaError):
            service.require_team_access("atlas", "outsider")

        mem.user_id = "atlas_admin"
        shared_status = mem.store_memory(
            anchor="atlas-shared",
            text="atlas workspace shared memory",
            tokens=7,
            team_id="atlas",
            share_with_team=True,
            promotion_weights={"threshold": 0.0},
            return_status=True,
        )
        atlas_memory_id = shared_status["long_term_id"]
        assert shared_status["namespace"] == atlas_space.namespace

        with mem.db_manager.SessionLocal() as session:
            atlas_record = (
                session.query(LongTermMemory)
                .filter(LongTermMemory.memory_id == atlas_memory_id)
                .one()
            )
            assert atlas_record.namespace == atlas_space.namespace

        mem.user_id = "outsider"
        with pytest.raises(MemoriaError):
            mem.store_memory(
                anchor="atlas-denied",
                text="unauthorized attempt",
                tokens=3,
                team_id="atlas",
                share_with_team=True,
                promotion_weights={"threshold": 0.0},
            )

        mem.user_id = "zen_member"
        zenith_status = mem.store_memory(
            anchor="zenith-private",
            text="zenith keeps memory private",
            tokens=5,
            team_id="zenith",
            promotion_weights={"threshold": 0.0},
            return_status=True,
        )
        zenith_memory_id = zenith_status["long_term_id"]
        assert zenith_status["namespace"] == personal_namespace

        with mem.db_manager.SessionLocal() as session:
            zenith_record = (
                session.query(LongTermMemory)
                .filter(LongTermMemory.memory_id == zenith_memory_id)
                .one()
            )
            assert zenith_record.namespace == personal_namespace
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()


def test_workspace_sync_replication_preserves_isolation(tmp_path):
    db_url = f"sqlite:///{tmp_path/'workspace_sync.db'}"
    mem = Memoria(
        database_connect=db_url,
        enable_short_term=True,
        user_id="replicator",
        team_memory_enabled=True,
        team_enforce_membership=True,
        team_namespace_prefix="workspace",
    )
    try:
        service = mem.storage_service
        service.configure_team_policy(
            namespace_prefix="workspace", enforce_membership=True
        )
        service.configure_sync_policy(SyncSettings(realtime_replication=True))

        atlas_space = service.register_team_space(
            "atlas", members=["replicator"], admins=["lead"], share_by_default=True
        )
        zenith_space = service.register_team_space(
            "zenith", members=["replicator"], admins=["lead"], share_by_default=True
        )

        with mem.db_manager.SessionLocal() as session:
            session.add_all(
                [
                    Team(team_id="atlas", name="Atlas", slug="atlas"),
                    Team(team_id="zenith", name="Zenith", slug="zenith"),
                ]
            )
            session.add_all(
                [
                    Workspace(
                        workspace_id="atlas",
                        name="Atlas",
                        slug="atlas",
                        owner_id="lead",
                        team_id="atlas",
                    ),
                    Workspace(
                        workspace_id="zenith",
                        name="Zenith",
                        slug="zenith",
                        owner_id="lead",
                        team_id="zenith",
                    ),
                ]
            )
            session.commit()

        now = datetime.utcnow()

        service.namespace = atlas_space.namespace
        service.workspace_id = "atlas"
        service.team_id = "atlas"

        atlas_short_record = {
            "memory_id": "atlas-short-sync",
            "chat_id": "atlas-thread",
            "processed_data": {"text": "atlas short sync"},
            "importance_score": 0.75,
            "category_primary": "manual_staged",
            "retention_type": "short_term",
            "namespace": atlas_space.namespace,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(days=7)).isoformat(),
            "access_count": 0,
            "searchable_content": "atlas short sync",
            "summary": "atlas short sync",
            "is_permanent_context": False,
            "x_coord": 0.0,
            "y_coord": 0.0,
            "z_coord": 0.0,
            "symbolic_anchors": ["atlas", "sync"],
            "team_id": "atlas",
            "workspace_id": "atlas",
        }
        atlas_long_record = {
            "memory_id": "atlas-long-sync",
            "original_chat_id": "atlas-thread",
            "processed_data": {"text": "atlas long sync"},
            "importance_score": 0.6,
            "category_primary": "manual_staged",
            "retention_type": "long_term",
            "namespace": atlas_space.namespace,
            "timestamp": now.isoformat(),
            "created_at": now.isoformat(),
            "access_count": 0,
            "searchable_content": "atlas long sync",
            "summary": "atlas long sync",
            "novelty_score": 0.5,
            "relevance_score": 0.5,
            "actionability_score": 0.5,
            "x_coord": 0.0,
            "y_coord": 0.0,
            "z_coord": 0.0,
            "symbolic_anchors": ["atlas", "sync"],
            "team_id": "atlas",
            "workspace_id": "atlas",
        }

        atlas_short_event = SyncEvent(
            action=SyncEventAction.MEMORY_UPDATED.value,
            entity_type="memory",
            namespace=atlas_space.namespace,
            entity_id="atlas-short-sync",
            payload={
                "replica": {"table": "short_term", "record": atlas_short_record},
                "workspace_id": "atlas",
                "team_id": "atlas",
            },
        )
        atlas_long_event = SyncEvent(
            action=SyncEventAction.MEMORY_UPDATED.value,
            entity_type="memory",
            namespace=atlas_space.namespace,
            entity_id="atlas-long-sync",
            payload={
                "replica": {"table": "long_term", "record": atlas_long_record},
                "workspace_id": "atlas",
                "team_id": "atlas",
            },
        )

        service.apply_sync_event(atlas_short_event)
        service.apply_sync_event(atlas_long_event)

        service.namespace = zenith_space.namespace
        service.workspace_id = "zenith"
        service.team_id = "zenith"

        zenith_short_record = {
            "memory_id": "zenith-short-sync",
            "chat_id": "zenith-thread",
            "processed_data": {"text": "zenith short sync"},
            "importance_score": 0.8,
            "category_primary": "manual_staged",
            "retention_type": "short_term",
            "namespace": zenith_space.namespace,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(days=3)).isoformat(),
            "access_count": 0,
            "searchable_content": "zenith short sync",
            "summary": "zenith short sync",
            "is_permanent_context": False,
            "x_coord": 1.0,
            "y_coord": 0.0,
            "z_coord": -1.0,
            "symbolic_anchors": ["zenith", "sync"],
            "team_id": "zenith",
            "workspace_id": "zenith",
        }
        zenith_long_record = {
            "memory_id": "zenith-long-sync",
            "original_chat_id": "zenith-thread",
            "processed_data": {"text": "zenith long sync"},
            "importance_score": 0.65,
            "category_primary": "manual_staged",
            "retention_type": "long_term",
            "namespace": zenith_space.namespace,
            "timestamp": now.isoformat(),
            "created_at": now.isoformat(),
            "access_count": 0,
            "searchable_content": "zenith long sync",
            "summary": "zenith long sync",
            "novelty_score": 0.55,
            "relevance_score": 0.55,
            "actionability_score": 0.55,
            "x_coord": 1.0,
            "y_coord": 0.0,
            "z_coord": -1.0,
            "symbolic_anchors": ["zenith", "sync"],
            "team_id": "zenith",
            "workspace_id": "zenith",
        }

        zenith_short_event = SyncEvent(
            action=SyncEventAction.MEMORY_UPDATED.value,
            entity_type="memory",
            namespace=zenith_space.namespace,
            entity_id="zenith-short-sync",
            payload={
                "replica": {"table": "short_term", "record": zenith_short_record},
                "workspace_id": "zenith",
                "team_id": "zenith",
            },
        )
        zenith_long_event = SyncEvent(
            action=SyncEventAction.MEMORY_UPDATED.value,
            entity_type="memory",
            namespace=zenith_space.namespace,
            entity_id="zenith-long-sync",
            payload={
                "replica": {"table": "long_term", "record": zenith_long_record},
                "workspace_id": "zenith",
                "team_id": "zenith",
            },
        )

        service.apply_sync_event(zenith_short_event)
        service.apply_sync_event(zenith_long_event)

        cross_event = SyncEvent(
            action=SyncEventAction.MEMORY_UPDATED.value,
            entity_type="memory",
            namespace=zenith_space.namespace,
            entity_id="atlas-cross-sync",
            payload={
                "replica": {
                    "table": "long_term",
                    "record": {
                        "memory_id": "atlas-cross-sync",
                        "processed_data": {"text": "should not sync"},
                        "importance_score": 0.4,
                        "category_primary": "manual_staged",
                        "retention_type": "long_term",
                        "namespace": atlas_space.namespace,
                        "timestamp": now.isoformat(),
                        "created_at": now.isoformat(),
                        "searchable_content": "should not sync",
                        "summary": "should not sync",
                        "symbolic_anchors": ["atlas"],
                        "workspace_id": "atlas",
                    },
                },
                "workspace_id": "atlas",
            },
        )

        service.apply_sync_event(cross_event)

        with mem.db_manager.SessionLocal() as session:
            atlas_short = (
                session.query(ShortTermMemory)
                .filter(ShortTermMemory.memory_id == "atlas-short-sync")
                .one()
            )
            zenith_short = (
                session.query(ShortTermMemory)
                .filter(ShortTermMemory.memory_id == "zenith-short-sync")
                .one()
            )
            atlas_long = (
                session.query(LongTermMemory)
                .filter(LongTermMemory.memory_id == "atlas-long-sync")
                .one()
            )
            zenith_long = (
                session.query(LongTermMemory)
                .filter(LongTermMemory.memory_id == "zenith-long-sync")
                .one()
            )

            assert atlas_short.namespace == atlas_space.namespace
            assert atlas_short.workspace_id == "atlas"
            assert zenith_short.namespace == zenith_space.namespace
            assert zenith_short.workspace_id == "zenith"
            assert atlas_long.namespace == atlas_space.namespace
            assert atlas_long.workspace_id == "atlas"
            assert zenith_long.namespace == zenith_space.namespace
            assert zenith_long.workspace_id == "zenith"

            assert (
                session.query(LongTermMemory)
                .filter(LongTermMemory.memory_id == "atlas-cross-sync")
                .one_or_none()
                is None
            )
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()


def test_workspace_memory_isolation_across_sync(tmp_path):
    db_url = f"sqlite:///{tmp_path/'workspace_isolation.db'}"
    mem = Memoria(
        database_connect=db_url,
        enable_short_term=True,
        user_id="member_a",
        team_memory_enabled=True,
        team_enforce_membership=True,
        team_namespace_prefix="workspace",
    )
    try:
        service = mem.storage_service
        service.configure_team_policy(
            namespace_prefix="workspace", enforce_membership=True
        )

        service.register_team_space(
            "atlas",
            members=["member_a", "observer"],
            admins=["lead"],
            share_by_default=True,
        )
        service.register_team_space(
            "zenith",
            members=["member_c"],
            admins=["lead"],
            share_by_default=False,
        )

        atlas_stage = service.stage_manual_memory(
            "atlas-anchor",
            "atlas workspace staged memo",
            9,
            namespace=None,
            team_id="atlas",
            user_id="member_a",
            share_with_team=True,
        )
        zenith_stage = service.stage_manual_memory(
            "zenith-anchor",
            "zenith workspace staged memo",
            11,
            namespace=None,
            team_id="zenith",
            user_id="member_c",
            share_with_team=True,
        )

        atlas_processed = ProcessedLongTermMemory(
            content=atlas_stage.text,
            summary=atlas_stage.text,
            classification=MemoryClassification.CONTEXTUAL,
            importance=MemoryImportanceLevel.MEDIUM,
            topic=None,
            conversation_id=atlas_stage.chat_id,
            confidence_score=0.9,
            x_coord=atlas_stage.x_coord,
            y_coord=atlas_stage.y_coord,
            z_coord=atlas_stage.z_coord,
            symbolic_anchors=list(atlas_stage.symbolic_anchors or []),
            classification_reason="workspace isolation",
            promotion_eligible=True,
        )
        zenith_processed = ProcessedLongTermMemory(
            content=zenith_stage.text,
            summary=zenith_stage.text,
            classification=MemoryClassification.CONTEXTUAL,
            importance=MemoryImportanceLevel.MEDIUM,
            topic=None,
            conversation_id=zenith_stage.chat_id,
            confidence_score=0.9,
            x_coord=zenith_stage.x_coord,
            y_coord=zenith_stage.y_coord,
            z_coord=zenith_stage.z_coord,
            symbolic_anchors=list(zenith_stage.symbolic_anchors or []),
            classification_reason="workspace isolation",
            promotion_eligible=True,
        )

        atlas_long_id = mem.db_manager.store_long_term_memory_enhanced(
            atlas_processed,
            atlas_stage.chat_id,
            atlas_stage.namespace,
            team_id="atlas",
        )
        zenith_long_id = mem.db_manager.store_long_term_memory_enhanced(
            zenith_processed,
            zenith_stage.chat_id,
            zenith_stage.namespace,
            team_id="zenith",
        )

        with mem.db_manager.SessionLocal() as session:
            atlas_short = (
                session.query(ShortTermMemory)
                .filter(ShortTermMemory.memory_id == atlas_stage.memory_id)
                .one()
            )
            zenith_short = (
                session.query(ShortTermMemory)
                .filter(ShortTermMemory.memory_id == zenith_stage.memory_id)
                .one()
            )
            assert atlas_short.namespace == "workspace:atlas"
            assert atlas_short.team_id == "atlas"
            assert zenith_short.namespace == "workspace:zenith"
            assert zenith_short.team_id == "zenith"

            atlas_long = (
                session.query(LongTermMemory)
                .filter(LongTermMemory.memory_id == atlas_long_id)
                .one()
            )
            zenith_long = (
                session.query(LongTermMemory)
                .filter(LongTermMemory.memory_id == zenith_long_id)
                .one()
            )
            assert atlas_long.namespace == "workspace:atlas"
            assert atlas_long.team_id == "atlas"
            assert zenith_long.namespace == "workspace:zenith"
            assert zenith_long.team_id == "zenith"

            namespaces = {row.namespace for row in session.query(LongTermMemory).all()}
            assert namespaces == {"workspace:atlas", "workspace:zenith"}
    finally:
        if mem._retention_scheduler:
            mem._retention_scheduler.stop()
