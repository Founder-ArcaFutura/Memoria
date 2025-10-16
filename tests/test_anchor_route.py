from __future__ import annotations

import json
import sqlite3

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from memoria.database.models import LongTermMemory, SpatialMetadata
from memoria.utils.pydantic_models import (
    MemoryClassification,
    MemoryImportanceLevel,
    ProcessedLongTermMemory,
)


def test_comma_separated_anchor_query_returns_union(sample_client):
    client, _ = sample_client
    mem = client.application.config["memoria"]
    mem.store_memory(
        anchor="g1",
        text="Went to gym",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        symbolic_anchors=["  gym  "],
    )
    mem.store_memory(
        anchor="g2",
        text="Base gym session",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        symbolic_anchors=[" base_gym "],
    )
    resp = client.get("/memory/anchor", query_string={"anchor": "gym,base_gym"})
    assert resp.status_code == 200
    data = resp.get_json()
    texts = {m["text"] for m in data}
    anchor_lookup = {m["text"]: m["symbolic_anchors"] for m in data}
    assert texts == {"Went to gym", "Base gym session"}
    assert anchor_lookup["Went to gym"] == ["gym"]
    assert anchor_lookup["Base gym session"] == ["base_gym"]


def test_enhanced_anchor_storage_visible_via_anchor_route(sample_client):
    client, _ = sample_client
    mem = client.application.config["memoria"]

    processed = ProcessedLongTermMemory(
        content="Enhanced anchor memory",
        summary="Enhanced anchor memory",
        classification=MemoryClassification.CONTEXTUAL,
        importance=MemoryImportanceLevel.HIGH,
        conversation_id="enhanced-chat",
        classification_reason="regression test",
        x_coord=1.0,
        y_coord=2.0,
        z_coord=3.0,
        symbolic_anchors=["enhanced_anchor"],
    )

    memory_id = mem.db_manager.store_long_term_memory_enhanced(
        processed,
        chat_id=processed.conversation_id,
        namespace=mem.namespace,
    )

    resp = client.get(
        "/memory/anchor",
        query_string={"anchor": "enhanced_anchor"},
    )
    assert resp.status_code == 200
    data = resp.get_json()

    stored = next((item for item in data if item["memory_id"] == memory_id), None)
    assert stored is not None
    assert stored["text"] == "Enhanced anchor memory"
    assert stored["x"] == 1.0
    assert stored["y"] == 2.0
    assert stored["z"] == 3.0
    assert "enhanced_anchor" in stored["symbolic_anchors"]


def test_anchor_route_strips_padded_symbolic_anchors(sample_client):
    client, _ = sample_client
    mem = client.application.config["memoria"]

    payload = {
        "anchor": "spacing-check",
        "text": "Anchor spacing memory",
        "tokens": 1,
        "x_coord": 0.0,
        "y_coord": 0.0,
        "z_coord": 0.0,
        "symbolic_anchors": ["  padded_anchor  ", "   "],
    }

    post_resp = client.post("/memory", json=payload)
    assert post_resp.status_code == 200
    created = post_resp.get_json()
    memory_id = created["memory_id"]

    with mem.db_manager.SessionLocal() as session:
        long_term_entry = (
            session.query(LongTermMemory)
            .filter_by(memory_id=memory_id, namespace=mem.namespace)
            .one()
        )
        assert long_term_entry.symbolic_anchors == ["padded_anchor"]

    db_path = client.application.config.get("DB_PATH")
    if db_path:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT symbolic_anchors FROM spatial_metadata WHERE memory_id = ? AND namespace = ?",
                (memory_id, mem.namespace),
            )
            row = cur.fetchone()
        finally:
            conn.close()
        assert row is not None
        stored_anchors = json.loads(row[0]) if row[0] else []
    else:
        with mem.db_manager.SessionLocal() as session:
            spatial_entry = (
                session.query(SpatialMetadata)
                .filter_by(memory_id=memory_id, namespace=mem.namespace)
                .one_or_none()
            )
        assert spatial_entry is not None
        stored_anchors = spatial_entry.symbolic_anchors or []

    assert stored_anchors == ["padded_anchor"]

    resp = client.get("/memory/anchor", query_string={"anchor": "padded_anchor"})
    assert resp.status_code == 200
    payload = resp.get_json()
    stored = next((item for item in payload if item["memory_id"] == memory_id), None)
    assert stored is not None
    assert stored["symbolic_anchors"] == ["padded_anchor"]

    mem_results = mem.retrieve_memories_by_anchor([" padded_anchor ", "   "])
    match = next((item for item in mem_results if item["memory_id"] == memory_id), None)
    assert match is not None
    assert match["symbolic_anchors"] == ["padded_anchor"]


@pytest.mark.parametrize(
    ("query_string", "expected_texts"),
    [
        ([("anchor", "discipline")], {"Discipline practice"}),
        ([("anchor", " discipline ")], {"Discipline practice"}),
        ([("anchor", '["discipline"]')], {"Discipline practice"}),
        (
            [("anchor", '["discipline", " focus "]')],
            {"Discipline practice", "Focus exercise"},
        ),
    ],
)
def test_anchor_route_handles_varied_anchor_formats(
    sample_client, query_string, expected_texts
):
    client, _ = sample_client
    mem = client.application.config["memoria"]

    mem.store_memory(
        anchor="discipline",  # legacy anchor column
        text="Discipline practice",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        symbolic_anchors=[" discipline "],
    )
    mem.store_memory(
        anchor="focus",
        text="Focus exercise",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        symbolic_anchors=["focus   "],
    )

    resp = client.get("/memory/anchor", query_string=query_string)
    assert resp.status_code == 200
    data = resp.get_json()
    texts = {item["text"] for item in data}
    assert expected_texts.issubset(texts)
    for item in data:
        assert all(anchor == anchor.strip() for anchor in item["symbolic_anchors"])


def test_anchor_route_falls_back_when_json1_functions_missing(
    sample_client, monkeypatch
):
    client, _ = sample_client
    mem = client.application.config["memoria"]

    mem.store_memory(
        anchor="json-fallback",
        text="Fallback anchor memory",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        symbolic_anchors=["json_fallback_anchor"],
    )

    original_execute = Session.execute

    def failing_execute(self, statement, *args, **kwargs):
        sql_text = str(statement)
        if "json_each" in sql_text.lower():
            raise OperationalError(
                sql_text,
                None,
                sqlite3.OperationalError("no such function: json_each"),
            )
        return original_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(Session, "execute", failing_execute)

    resp = client.get(
        "/memory/anchor",
        query_string={"anchor": "json_fallback_anchor"},
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    texts = {item["text"] for item in payload}
    assert "Fallback anchor memory" in texts


def test_anchor_route_rejects_malformed_json(sample_client):
    client, _ = sample_client

    resp = client.get("/memory/anchor", query_string=[("anchor", "[not valid")])
    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["status"] == "error"
    assert payload["message"] == "Invalid anchor format"


def test_debug_anchor_route_reads_long_term_memory_only(sample_client):
    client, _ = sample_client
    mem = client.application.config["memoria"]

    processed = ProcessedLongTermMemory(
        content="Long term only anchor",  # avoid touching spatial_metadata
        summary="Long term only anchor",
        classification=MemoryClassification.CONTEXTUAL,
        importance=MemoryImportanceLevel.MEDIUM,
        conversation_id="debug-anchor-ltm",
        classification_reason="regression to ensure debug anchors query long term",
        x_coord=0.0,
        y_coord=0.0,
        z_coord=0.0,
        symbolic_anchors=["ltm_only_anchor"],
    )

    memory_id = mem.db_manager.store_long_term_memory_enhanced(
        processed,
        chat_id=processed.conversation_id,
        namespace=mem.namespace,
    )

    with mem.db_manager.SessionLocal() as session:
        spatial_entry = (
            session.query(SpatialMetadata).filter_by(memory_id=memory_id).one_or_none()
        )
        assert spatial_entry is None

    resp = client.get("/debug/anchors")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert "ltm_only_anchor" in payload["anchors"]
