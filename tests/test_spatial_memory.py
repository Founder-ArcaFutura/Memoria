import asyncio
import datetime
import json
import os
import sys

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session as ORMSession

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memoria.agents.conscious_agent import ConsciousAgent
from memoria.agents.memory_agent import MemoryAgent
from memoria.core.memory import Memoria
from memoria.database.models import (
    DatabaseManager,
    LongTermMemory,
    ShortTermMemory,
    SpatialMetadata,
)
from memoria.database.search_service import SearchService
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager
from memoria.storage.service import StorageService
from memoria.utils.pydantic_compat import model_dump
from tests.utils.factories import memoria_from_entries, memory_entry


def _create_db():
    db = DatabaseManager("sqlite:///:memory:")
    db.create_tables()
    return db


def test_short_term_memory_spatial_fields():
    db = _create_db()
    session = db.get_session()
    stm = ShortTermMemory(
        memory_id="s1",
        processed_data={},
        importance_score=0.5,
        category_primary="test",
        namespace="default",
        created_at=datetime.datetime.utcnow(),
        searchable_content="content",
        summary="summary",
        x_coord=1.0,
        y_coord=2.0,
        z_coord=3.0,
        symbolic_anchors=["anchor"],
    )
    session.add(stm)
    session.commit()
    retrieved = session.query(ShortTermMemory).filter_by(memory_id="s1").first()
    assert retrieved.x_coord == 1.0

    assert retrieved.symbolic_anchors == ["anchor"]


def test_long_term_memory_spatial_fields():
    db = _create_db()
    session = db.get_session()
    ltm = LongTermMemory(
        memory_id="l1",
        processed_data={},
        importance_score=0.5,
        category_primary="test",
        namespace="default",
        created_at=datetime.datetime.utcnow(),
        searchable_content="content",
        summary="summary",
        x_coord=4.0,
        y_coord=5.0,
        z_coord=6.0,
        symbolic_anchors=["anchor2"],
    )
    session.add(ltm)
    session.commit()
    retrieved = session.query(LongTermMemory).filter_by(memory_id="l1").first()
    assert retrieved.y_coord == 5.0

    assert retrieved.symbolic_anchors == ["anchor2"]


def test_retrieve_memories_near():
    mem = Memoria(database_connect="sqlite:///:memory:")
    mem.store_memory(
        anchor="a",
        text="near",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        symbolic_anchors=["A"],
    )
    mem.store_memory(
        anchor="a",
        text="far",
        tokens=1,
        x_coord=10.0,
        y=10.0,
        z=10.0,
        symbolic_anchors=["A"],
    )
    mem.store_memory(
        anchor="b",
        text="other",
        tokens=1,
        x_coord=1.0,
        y=1.0,
        z=1.0,
        symbolic_anchors=["B"],
    )
    results = mem.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=5.0)
    assert len(results) == 2
    assert all(isinstance(r["symbolic_anchors"], list) for r in results)
    anchored = mem.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=5.0, anchor="A")
    assert len(anchored) == 1
    assert anchored[0]["text"] == "near"
    assert anchored[0]["symbolic_anchors"] == ["A"]


def test_spatial_metadata_rows_do_not_mask_rich_text():
    mem = Memoria(database_connect="sqlite:///:memory:", enable_short_term=False)
    memory_id = mem.store_memory(
        anchor="pref", text="preferred", tokens=1, x_coord=0.05, y=0.0, z=0.0
    )

    with mem.db_manager.SessionLocal() as session:
        # The source of truth for symbolic anchors on a rich memory record
        # is the LongTermMemory table itself. Update it directly.
        long_entry = session.query(LongTermMemory).filter_by(memory_id=memory_id).one()
        long_entry.symbolic_anchors = ["meta"]
        session.commit()

    results = mem.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=1.0)
    assert results
    assert results[0]["memory_id"] == memory_id
    assert results[0]["text"] == "preferred"
    assert results[0]["symbolic_anchors"] == ["meta"]


def test_store_memory_populates_spatial_metadata_table():
    mem = Memoria(database_connect="sqlite:///:memory:", enable_short_term=False)
    timestamp = datetime.datetime.utcnow()

    memory_id = mem.storage_service.store_memory(
        anchor="spatial-sync",
        text="ensure spatial metadata mirrors long-term",
        tokens=5,
        timestamp=timestamp,
        x_coord=1.5,
        y=0.25,
        z=-0.75,
        symbolic_anchors=["sync"],
    )

    with mem.db_manager.SessionLocal() as session:
        spatial_entry = (
            session.query(SpatialMetadata)
            .filter(SpatialMetadata.memory_id == memory_id)
            .one()
        )

        assert spatial_entry.namespace == mem.namespace
        assert spatial_entry.team_id is None
        assert spatial_entry.workspace_id is None
        assert spatial_entry.x == pytest.approx(1.5)
        assert spatial_entry.y == pytest.approx(0.25)
        assert spatial_entry.z == pytest.approx(-0.75)
        assert spatial_entry.symbolic_anchors == ["sync"]
        assert spatial_entry.timestamp == timestamp


def test_retrieve_memories_near_filters_namespace(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'namespaced.db'}"
    mem_alpha = Memoria(database_connect=db_url, namespace="alpha")
    mem_beta = Memoria(database_connect=db_url, namespace="beta")

    mem_alpha.store_memory(
        anchor="a",
        text="alpha only",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
    )
    mem_beta.store_memory(
        anchor="b",
        text="beta hidden",
        tokens=1,
        x_coord=0.1,
        y=0.0,
        z=0.0,
    )

    results = mem_alpha.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=1.0)
    texts = {r["text"] for r in results}
    assert texts == {"alpha only"}


def test_retrieve_memories_near_with_namespace_argument():
    mem = Memoria(database_connect="sqlite:///:memory:")

    mem.store_memory(
        anchor="default",
        text="default scope",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
    )

    mem.store_memory(
        anchor="other",
        text="other scope",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        namespace="other",
    )

    default_results = mem.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=1.0)
    assert {result["text"] for result in default_results} == {"default scope"}

    other_results = mem.retrieve_memories_near(
        0.0,
        0.0,
        0.0,
        max_distance=1.0,
        namespace="other",
    )
    assert {result["text"] for result in other_results} == {"other scope"}


def test_manual_schema_processing_spatial_fields():
    agent = MemoryAgent(api_key="test")
    data = {
        "content": "spatial memory",
        "summary": "summary",
        "classification": "conversational",
        "importance": "medium",
        "x_coord": 7.0,
        "y_coord": 8.0,
        "z_coord": 9.0,
        "symbolic_anchors": ["anchor3"],
    }
    processed = agent._create_memory_from_dict(data, chat_id="c1")
    assert processed.x_coord == 7.0
    assert processed.symbolic_anchors == ["anchor3"]

    db = _create_db()
    session = db.get_session()
    ltm = LongTermMemory(
        memory_id="l2",
        processed_data=model_dump(processed),
        importance_score=processed.importance_score,
        category_primary="test",
        namespace="default",
        created_at=datetime.datetime.utcnow(),
        searchable_content=processed.content,
        summary=processed.summary,
        x_coord=processed.x_coord,
        y_coord=processed.y_coord,
        z_coord=processed.z_coord,
        symbolic_anchors=processed.symbolic_anchors,
        classification=processed.classification.value,
        memory_importance=processed.importance.value,
    )
    session.add(ltm)
    session.commit()
    retrieved = session.query(LongTermMemory).filter_by(memory_id="l2").first()
    assert retrieved.z_coord == 9.0

    assert retrieved.symbolic_anchors == ["anchor3"]


def _insert_conscious_long_term(
    db_manager: SQLAlchemyDatabaseManager,
    memory_id: str,
    text: str,
    *,
    x: float,
    y: float,
    z: float,
    anchors: list[str],
) -> None:
    namespace = "default"
    with db_manager.SessionLocal() as session:
        session.add(
            LongTermMemory(
                memory_id=memory_id,
                processed_data={"text": text, "tokens": len(text.split())},
                importance_score=0.9,
                category_primary="conscious",
                retention_type="long_term",
                namespace=namespace,
                created_at=datetime.datetime.utcnow(),
                timestamp=datetime.datetime.utcnow(),
                searchable_content=text,
                summary=text,
                classification="conscious-info",
                x_coord=x,
                y_coord=y,
                z_coord=z,
                symbolic_anchors=anchors,
            )
        )
        session.commit()


def test_conscious_promotion_preserves_spatial_metadata(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'conscious_preserve.db'}"
    db_manager = SQLAlchemyDatabaseManager(db_url)
    db_manager.initialize_schema()
    agent = ConsciousAgent()

    anchors = ["focus", "ritual"]
    _insert_conscious_long_term(
        db_manager,
        memory_id="lt-preserve",
        text="Preserve my context",
        x=1.25,
        y=-3.5,
        z=2.0,
        anchors=anchors,
    )

    asyncio.run(agent.run_conscious_ingest(db_manager, "default"))

    with db_manager.SessionLocal() as session:
        rows = session.query(ShortTermMemory).filter_by(namespace="default").all()

        assert len(rows) == 1
        promoted = rows[0]
        assert promoted.x_coord == pytest.approx(1.25)
        assert promoted.y_coord == pytest.approx(-3.5)
        assert promoted.z_coord == pytest.approx(2.0)

        anchors_value = promoted.symbolic_anchors
        if isinstance(anchors_value, str):
            anchors_value = json.loads(anchors_value)

        assert anchors_value == anchors


def test_conscious_promotion_enables_short_term_spatial_query(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'conscious_query.db'}"
    db_manager = SQLAlchemyDatabaseManager(db_url)
    db_manager.initialize_schema()
    agent = ConsciousAgent()
    storage = StorageService(db_manager, namespace="default", conscious_ingest=True)

    anchors = ["presence"]
    _insert_conscious_long_term(
        db_manager,
        memory_id="lt-query",
        text="Short-term spatial insight",
        x=0.0,
        y=0.0,
        z=0.0,
        anchors=anchors,
    )

    asyncio.run(agent.run_conscious_ingest(db_manager, "default"))

    with db_manager.SessionLocal() as session:
        session.query(LongTermMemory).filter_by(memory_id="lt-query").update(
            {"x_coord": 10.0, "y_coord": 10.0, "z_coord": 10.0}
        )
        session.commit()

    results = storage.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=5.0)
    assert len(results) == 1
    result = results[0]
    assert result["symbolic_anchors"] == anchors
    assert result["x"] == pytest.approx(0.0)
    assert result["y"] == pytest.approx(0.0)
    assert result["z"] == pytest.approx(0.0)


def test_retrieve_memories_by_anchor():
    mem = Memoria(database_connect="sqlite:///:memory:", enable_short_term=True)
    now = datetime.datetime.utcnow()

    with mem.db_manager.SessionLocal() as session:
        # Manually insert records to create a predictable database state for the test
        session.add_all(
            [
                LongTermMemory(
                    memory_id="long_id",
                    processed_data={"text": "long", "emotional_intensity": 0.7},
                    importance_score=0.8,
                    category_primary="test",
                    namespace="default",
                    created_at=now,
                    searchable_content="long",
                    summary="long",
                    x_coord=1.0,
                    y_coord=2.0,
                    z_coord=3.0,
                    symbolic_anchors=["A"],
                ),
                LongTermMemory(
                    memory_id="fallback_id",
                    processed_data="not valid json",
                    importance_score=0.8,
                    category_primary="test",
                    namespace="default",
                    created_at=now,
                    searchable_content="source text",
                    summary="fallback summary",
                    x_coord=1.5,
                    y_coord=2.5,
                    z_coord=3.5,
                    symbolic_anchors=["A"],
                ),
                ShortTermMemory(
                    memory_id="s_anchor",
                    processed_data={"text": "short"},
                    importance_score=0.5,
                    category_primary="test",
                    namespace="default",
                    created_at=now,
                    searchable_content="short",
                    summary="short",
                    symbolic_anchors=["A", "C"],
                ),
            ]
        )
        session.commit()

    # Test that anchor search retrieves from both long-term and short-term tables
    results = mem.retrieve_memories_by_anchor(["A"])
    assert len(results) == 3
    texts = {r["text"] for r in results}
    assert texts == {"long", "short", "fallback summary"}

    # Verify that the fallback to summary works when processed_data is invalid
    fallback_result = next(
        (r for r in results if r.get("summary") == "fallback summary"), None
    )
    assert fallback_result is not None
    assert fallback_result["text"] == "fallback summary"

    # Verify searching by an anchor that doesn't exist returns no results
    only_long = mem.retrieve_memories_by_anchor(["B"])
    assert len(only_long) == 0


def test_anchor_retrieval_cross_namespace():
    mem = Memoria(database_connect="sqlite:///:memory:")
    first_id = mem.store_memory(
        anchor="base",
        text="same namespace",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        symbolic_anchors=["shared"],
    )

    with mem.db_manager.SessionLocal() as session:
        other = LongTermMemory(
            memory_id="other_ns_anchor",
            processed_data={"text": "other namespace"},
            importance_score=0.5,
            category_primary="test",
            namespace="other",
            timestamp=datetime.datetime.utcnow(),
            created_at=datetime.datetime.utcnow(),
            searchable_content="other namespace",
            summary="other namespace",
            symbolic_anchors=["shared"],
        )
        session.add(other)
        session.commit()

    results = mem.retrieve_memories_by_anchor(["shared"])
    ids = {item["memory_id"] for item in results}
    assert first_id in ids
    assert "other_ns_anchor" in ids


def test_multi_anchor_retrieval_returns_union():
    mem = Memoria(database_connect="sqlite:///:memory:")
    mem.store_memory(
        anchor="g1",
        text="gym note",
        tokens=1,
        x_coord=1.0,
        y=2.0,
        z=3.0,
        symbolic_anchors=["gym"],
    )
    mem.store_memory(
        anchor="g2",
        text="base gym note",
        tokens=1,
        x_coord=1.0,
        y=2.0,
        z=3.0,
        symbolic_anchors=["base_gym"],
    )
    res = mem.retrieve_memories_by_anchor(["gym", "base_gym"])
    texts = {r["text"] for r in res}
    assert texts == {"gym note", "base gym note"}


def test_time_based_retrieval():
    mem = memoria_from_entries(
        [
            memory_entry("recent", "recent memory", x=-1.0, y=0.0, z=0.0),
            memory_entry("old", "old memory", x=-10.0, y=0.0, z=0.0),
        ]
    )
    results = mem.retrieve_memories_near(-2.0, 0.0, 0.0, max_distance=3.0)
    texts = {r["text"] for r in results}
    assert "recent memory" in texts
    assert "old memory" not in texts


def test_multi_anchor_spatial_query():
    mem = memoria_from_entries(
        [
            memory_entry(
                "m1", "multi", x=0.0, y=0.0, z=0.0, symbolic_anchors=["A", "B"]
            ),
            memory_entry("m2", "onlyA", x=0.0, y=0.0, z=0.0, symbolic_anchors=["A"]),
            memory_entry("m3", "onlyB", x=0.0, y=0.0, z=0.0, symbolic_anchors=["B"]),
        ]
    )
    res_a = mem.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=1.0, anchor="A")
    texts_a = {r["text"] for r in res_a}
    assert texts_a == {"multi", "onlyA"}
    res_b = mem.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=1.0, anchor="B")
    texts_b = {r["text"] for r in res_b}
    assert texts_b == {"multi", "onlyB"}
    res_ab = mem.retrieve_memories_near(
        0.0, 0.0, 0.0, max_distance=1.0, anchor=["A", "B"]
    )
    texts_ab = {r["text"] for r in res_ab}
    assert texts_ab == {"multi", "onlyA", "onlyB"}


def test_spatial_anchor_filter_falls_back_without_json1(monkeypatch):
    mem = memoria_from_entries(
        [
            memory_entry(
                "focus", "target", x=0.0, y=0.0, z=0.0, symbolic_anchors=["Focus"]
            ),
            memory_entry(
                "other", "distractor", x=0.0, y=0.0, z=0.0, symbolic_anchors=["Other"]
            ),
        ]
    )

    original_execute = ORMSession.execute
    call_state = {"count": 0}

    def patched_execute(self, statement, params=None, **kwargs):
        sql_text = str(statement).lower()
        if call_state["count"] == 0 and "symbolic_anchors" in sql_text:
            call_state["count"] += 1
            raise OperationalError(
                sql_text,
                params,
                Exception("no such function: json_each"),
            )
        return original_execute(self, statement, params=params, **kwargs)

    monkeypatch.setattr(ORMSession, "execute", patched_execute, raising=False)

    results = mem.retrieve_memories_near(
        0.0, 0.0, 0.0, max_distance=1.0, anchor="Focus"
    )
    texts = {item["text"] for item in results}
    assert texts == {"target"}
    assert call_state["count"] == 1


def test_anchor_search_falls_back_without_json1(monkeypatch):
    db = _create_db()
    session = db.get_session()
    now = datetime.datetime.utcnow()

    short = ShortTermMemory(
        memory_id="short_focus",
        processed_data={"symbolic_anchors": ["Focus"], "emotional_intensity": 0.3},
        importance_score=0.6,
        category_primary="search",
        namespace="default",
        created_at=now,
        searchable_content="focus short entry",
        summary="short focus",
        x_coord=0.0,
        y_coord=0.0,
        z_coord=0.0,
        symbolic_anchors=["Focus"],
    )

    long = LongTermMemory(
        memory_id="long_focus",
        processed_data={"symbolic_anchors": ["Focus"], "emotional_intensity": 0.5},
        importance_score=0.7,
        category_primary="search",
        namespace="default",
        created_at=now,
        searchable_content="focus long entry",
        summary="long focus",
        x_coord=0.5,
        y_coord=0.5,
        z_coord=0.5,
        symbolic_anchors=["Focus"],
    )

    session.add_all([short, long])
    session.commit()

    service = SearchService(session, "sqlite")

    original_execute = ORMSession.execute
    call_state = {"count": 0}

    def patched_execute(self, statement, params=None, **kwargs):
        sql_text = str(statement).lower()
        if (
            "symbolic_anchors" in sql_text
            and "instr(" not in sql_text
            and call_state["count"] < 2
        ):
            call_state["count"] += 1
            raise OperationalError(
                sql_text,
                params,
                Exception("no such function: json_each"),
            )
        return original_execute(self, statement, params=params, **kwargs)

    monkeypatch.setattr(ORMSession, "execute", patched_execute, raising=False)

    response = service.search_memories(
        query="Focus",
        namespace="default",
        limit=5,
        use_anchor=True,
    )

    ids = {item["memory_id"] for item in response["results"]}
    assert {"short_focus", "long_focus"} <= ids
    assert call_state["count"] == 2


def test_retrieval_from_spatial_metadata_only():
    mem = Memoria(database_connect="sqlite:///:memory:")
    with mem.db_manager.SessionLocal() as session:
        session.execute(
            text(
                "INSERT INTO spatial_metadata (memory_id, namespace, timestamp, x, y, z, symbolic_anchors) "
                "VALUES (:id, :ns, :ts, :x, :y, :z, :anchors)"
            ),
            {
                "id": "smeta1",
                "ns": "default",
                "ts": datetime.datetime.utcnow().isoformat(),
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "anchors": '["A"]',
            },
        )
        session.commit()

    results = mem.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=1.0)
    assert any(r["symbolic_anchors"] == ["A"] for r in results)

    anchored = mem.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=1.0, anchor="A")
    assert len(anchored) == 1
    assert anchored[0]["symbolic_anchors"] == ["A"]


def test_missing_or_invalid_coordinates():
    mem = memoria_from_entries(
        [
            memory_entry("good", "valid", x=0.0, y=0.0, z=0.0),
            memory_entry("missing", "missing coords", x=None, y=None, z=None),
        ]
    )
    results = mem.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=1.0)
    texts = {r["text"] for r in results}
    assert "valid" in texts
    assert "missing coords" not in texts
    with pytest.raises(Exception):
        mem.store_memory(
            anchor="bad",
            text="bad",
            tokens=1,
            x_coord="not-a-number",
            y=0.0,
            z=0.0,
        )


def test_search_memories_includes_symbolic_anchors():
    db = _create_db()
    session = db.get_session()
    now = datetime.datetime.utcnow()
    anchors = ["shared-tag"]

    short = ShortTermMemory(
        memory_id="short_search",
        processed_data={"symbolic_anchors": anchors, "emotional_intensity": 0.2},
        importance_score=0.6,
        category_primary="search",
        namespace="default",
        created_at=now,
        searchable_content="shared story short memory",
        summary="Shared story from short term",
        x_coord=0.1,
        y_coord=0.2,
        z_coord=0.3,
        symbolic_anchors=anchors,
    )

    long = LongTermMemory(
        memory_id="long_search",
        processed_data={"symbolic_anchors": anchors, "emotional_intensity": 0.4},
        importance_score=0.7,
        category_primary="search",
        namespace="default",
        created_at=now,
        searchable_content="shared story long memory",
        summary="Shared story from long term",
        x_coord=1.1,
        y_coord=1.2,
        z_coord=1.3,
        symbolic_anchors=anchors,
    )

    session.add_all([short, long])
    session.commit()

    service = SearchService(session, "sqlite")

    response = service.search_memories(
        query="shared story",
        namespace="default",
        use_anchor=False,
        use_fuzzy=True,
        fuzzy_min_similarity=0,
    )

    results = response["results"]
    assert results, "Expected search results for shared story"

    short_result = next(r for r in results if r["memory_type"] == "short_term")
    long_result = next(r for r in results if r["memory_type"] == "long_term")

    assert short_result["symbolic_anchors"] == anchors
    assert long_result["symbolic_anchors"] == anchors


def test_anchor_directive_search_returns_symbolic_matches():
    db = _create_db()
    session = db.get_session()
    now = datetime.datetime.utcnow()

    entry = LongTermMemory(
        memory_id="anchor_only",
        processed_data={"symbolic_anchors": ["discipline"], "emotional_intensity": 0.1},
        importance_score=0.5,
        category_primary="journal",
        namespace="default",
        created_at=now,
        searchable_content="gym discipline log",
        summary="Gym log entry",
        symbolic_anchors=["discipline"],
    )
    session.add(entry)
    session.commit()

    service = SearchService(session, "sqlite")

    response = service.search_memories(
        query="anchor:discipline training log",
        namespace="default",
        limit=5,
    )

    results = response["results"]
    assert any(result["memory_id"] == "anchor_only" for result in results)
    assert response.get("attempted") == ["anchor"]
    assert all(result.get("search_strategy") == "symbolic_anchor" for result in results)


def test_anchor_directive_supports_quoted_values():
    db = _create_db()
    session = db.get_session()
    now = datetime.datetime.utcnow()

    entry = LongTermMemory(
        memory_id="anchor_phrase",
        processed_data={
            "symbolic_anchors": ["cathedral scaffold"],
            "emotional_intensity": 0.2,
        },
        importance_score=0.6,
        category_primary="design",
        namespace="default",
        created_at=now,
        searchable_content="notes on cathedral scaffolding",
        summary="Design musings",
        symbolic_anchors=["cathedral scaffold"],
    )
    session.add(entry)
    session.commit()

    service = SearchService(session, "sqlite")
    response = service.search_memories(
        query='anchor:"cathedral scaffold"',
        namespace="default",
        limit=5,
    )

    results = response["results"]
    assert len(results) == 1
    result = results[0]
    assert result["memory_id"] == "anchor_phrase"
    assert result.get("symbolic_anchors") == ["cathedral scaffold"]
