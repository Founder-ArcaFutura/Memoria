from __future__ import annotations

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memoria.agents.memory_agent import MemoryAgent
from memoria.agents.search_executor import SearchExecutor
from memoria.database.models import Base, LinkMemoryThread, LongTermMemory
from memoria.database.search_service import SearchService
from memoria.heuristics.conversation_ingest import process_conversation_turn
from memoria.heuristics.manual_promotion import StagedManualMemory
from memoria.utils.pydantic_models import (
    AgentPermissions,
    MemoryClassification,
    MemoryImportanceLevel,
    MemorySearchQuery,
    ProcessedLongTermMemory,
)


class _RelationshipRecordingDB:
    def __init__(self):
        self.candidate_calls: list[dict[str, object]] = []
        self.link_calls: list[tuple[str, tuple[str, ...], str]] = []

    def find_related_memory_candidates(self, **kwargs):
        self.candidate_calls.append(kwargs)
        return [
            {"memory_id": "rel-1", "match_score": 0.9},
            {"memory_id": "rel-1", "match_score": 0.8},
            {"memory_id": "rel-2", "match_score": 0.75},
        ]

    def store_memory_links(self, source_memory_id, related_ids, relation="related"):
        self.link_calls.append((source_memory_id, tuple(related_ids), relation))


def test_memory_agent_relationship_detection_and_persistence():
    agent = MemoryAgent(api_key="test")
    memory = ProcessedLongTermMemory(
        content="Discussed alpha milestones",
        summary="Alpha roadmap update",
        classification=MemoryClassification.CONTEXTUAL,
        importance=MemoryImportanceLevel.MEDIUM,
        topic="alpha",
        entities=["alpha"],
        keywords=["roadmap"],
        conversation_id="chat-1",
        classification_reason="unit-test",
    )

    db = _RelationshipRecordingDB()

    candidates = agent.attach_related_memories(
        memory,
        db,
        namespace="default",
        limit=3,
    )

    assert [candidate["memory_id"] for candidate in candidates] == [
        "rel-1",
        "rel-1",
        "rel-2",
    ]
    assert memory.related_memories == ["rel-1", "rel-2"]

    agent.persist_relationship_links("mem-0", memory.related_memories, db)
    assert db.link_calls == [("mem-0", ("rel-1", "rel-2"), "related")]


class _RelationshipSearchDB:
    def __init__(self, results):
        self.results = results
        self.search_calls: list[dict[str, object]] = []
        self.relationship_calls: list[dict[str, object]] = []

    def search_memories(self, **kwargs):
        self.search_calls.append(kwargs)
        return {"results": list(self.results)}

    def get_related_memories(self, **kwargs):
        self.relationship_calls.append(kwargs)
        return [
            {
                "memory_id": "rel-1",
                "summary": "Follow-up alpha decision",
                "relationship_reason": "related",
            }
        ]


def test_search_executor_expands_with_relationships():
    executor = SearchExecutor(permissions=AgentPermissions())
    db = _RelationshipSearchDB(
        [{"memory_id": "base-1", "summary": "Alpha base", "importance_score": 0.5}]
    )

    search_plan = MemorySearchQuery(query_text="alpha", intent="general")

    results = executor.execute_search(
        "alpha",
        search_plan,
        db,
        namespace="default",
        limit=3,
    )

    assert [result.get("memory_id") for result in results] == [
        "base-1",
        "rel-1",
    ]
    assert results[0].get("related_memories") == ["rel-1"]
    assert results[1]["search_strategy"] == "relationship_graph"


class _StubStorageService:
    def __init__(self):
        self.namespace = "default"

    def stage_manual_memory(self, *args, **kwargs):
        return StagedManualMemory(
            memory_id="short-1",
            chat_id="chat-1",
            namespace=kwargs.get("namespace", self.namespace),
            anchor=kwargs.get("anchor", "alpha"),
            text=kwargs.get("text", "context"),
            tokens=kwargs.get("tokens", 5),
            timestamp=kwargs.get("timestamp"),
            x_coord=kwargs.get("x_coord"),
            y_coord=kwargs.get("y"),
            z_coord=kwargs.get("z"),
            symbolic_anchors=kwargs.get("symbolic_anchors", ["alpha"]),
            metadata=dict(kwargs.get("metadata") or {}),
        )

    def get_relationship_candidates(self, **kwargs):
        return [{"memory_id": "rel-1", "match_score": 0.8}]

    def compute_cluster_gravity(self, **kwargs):
        return 0.0

    def count_anchor_occurrences(self, *args, **kwargs):
        return {}


def test_conversation_ingest_collects_relationship_candidates():
    storage = _StubStorageService()
    result = process_conversation_turn(
        storage,
        chat_id="chat-1",
        user_input="Let's revisit alpha",
        ai_output="We confirmed the alpha milestone.",
    )

    assert result.related_candidates == [{"memory_id": "rel-1", "match_score": 0.8}]
    assert (
        result.staged.metadata["relationship_candidates"] == result.related_candidates
    )


def test_search_service_relationship_queries_round_trip():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    service = SearchService(session, "sqlite")

    now = datetime.utcnow()
    base_memory = LongTermMemory(
        memory_id="base",
        original_chat_id="chat-1",
        processed_data={"symbolic_anchors": ["alpha"], "content": "base"},
        importance_score=0.6,
        category_primary="context",
        retention_type="long_term",
        namespace="default",
        timestamp=now,
        created_at=now,
        searchable_content="alpha base update",
        summary="Alpha base update",
        related_memories_json=["rel-json"],
        symbolic_anchors=["alpha"],
    )
    related_json = LongTermMemory(
        memory_id="rel-json",
        original_chat_id="chat-2",
        processed_data={"symbolic_anchors": ["alpha"], "content": "related"},
        importance_score=0.5,
        category_primary="context",
        retention_type="long_term",
        namespace="default",
        timestamp=now,
        created_at=now,
        searchable_content="alpha related notes",
        summary="Alpha related notes",
        symbolic_anchors=["alpha"],
    )
    related_graph = LongTermMemory(
        memory_id="rel-graph",
        original_chat_id="chat-3",
        processed_data={"symbolic_anchors": ["beta"], "content": "graph"},
        importance_score=0.4,
        category_primary="context",
        retention_type="long_term",
        namespace="default",
        timestamp=now,
        created_at=now,
        searchable_content="beta follow up",
        summary="Beta follow up",
        symbolic_anchors=["beta"],
    )

    session.add_all([base_memory, related_json, related_graph])
    session.add_all(
        [
            LinkMemoryThread(
                source_memory_id="base",
                target_memory_id="rel-graph",
                relation="related",
            ),
            LinkMemoryThread(
                source_memory_id="rel-graph",
                target_memory_id="base",
                relation="related",
            ),
        ]
    )
    session.commit()

    related = service.get_related_memories("base", "default", limit=5)
    assert {item["memory_id"] for item in related} == {"rel-json", "rel-graph"}

    candidates = service.find_related_memory_candidates(
        namespace="default", symbolic_anchors=["alpha"], keywords=["update"], limit=3
    )
    assert candidates
    assert candidates[0]["memory_id"] in {"rel-json", "base"}

    session.close()
