import asyncio
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.agents.conscious_agent import ConsciousAgent
from memoria.conscious import CONSCIOUS_CONTEXT_CATEGORY
from memoria.core.memory import Memoria
from memoria.database import backup_guard
from memoria.database.models import LongTermMemory, ShortTermMemory
from memoria.heuristics.conversation_ingest import HeuristicConversationResult
from memoria.heuristics.manual_promotion import (
    PromotionDecision,
    StagedManualMemory,
)
from memoria.providers import anthropic_recorder, openai_recorder
from memoria.utils.pydantic_models import (
    MemoryClassification,
    MemoryImportanceLevel,
    ProcessedLongTermMemory,
)


def _create_memoria():
    return Memoria(database_connect="sqlite:///:memory:")


def test_memoria_initializes_with_in_memory_sqlite_url(tmp_path):
    mem = Memoria(database_connect="sqlite://")
    manager = mem.db_manager

    assert manager.database_type == "sqlite"

    size = manager.get_db_size()
    assert size == 0

    backup_destination = tmp_path / "memoria-backup.sqlite"
    manager.backup_database(backup_destination)
    assert not backup_destination.exists()

    helper_size = backup_guard._get_db_size(manager)
    assert helper_size == 0

    backup_source = tmp_path / "source.sqlite"
    backup_source.write_bytes(b"dummy")

    manager.restore_database(backup_source)


def _seed_short_term_memory(
    db_manager, *, namespace: str, summary: str, memory_id: str, importance: float = 0.9
):
    """Insert a short-term memory row for testing purposes."""

    now = datetime.utcnow()
    with db_manager.SessionLocal() as session:
        record = ShortTermMemory(
            memory_id=memory_id,
            processed_data={
                "text": summary,
                "searchable_content": summary,
                "emotional_intensity": 0.1,
            },
            importance_score=importance,
            category_primary="essential_user_profile",
            retention_type="short_term",
            namespace=namespace,
            created_at=now,
            expires_at=now + timedelta(days=7),
            access_count=0,
            searchable_content=summary,
            summary=summary,
        )
        session.add(record)
        session.commit()


def _seed_long_term_memory(
    db_manager,
    *,
    namespace: str,
    summary: str,
    memory_id: str,
    category: str = "profile",
    importance: float = 0.75,
):
    """Insert a long-term memory row for testing purposes."""

    now = datetime.utcnow()
    with db_manager.SessionLocal() as session:
        record = LongTermMemory(
            memory_id=memory_id,
            processed_data={
                "text": summary,
                "searchable_content": summary,
            },
            importance_score=importance,
            category_primary=category,
            retention_type="long_term",
            namespace=namespace,
            timestamp=now,
            created_at=now,
            searchable_content=summary,
            summary=summary,
        )
        session.add(record)
        session.commit()


class DummyMemoryAgent:
    async def process_conversation_async(
        self,
        chat_id: str,
        user_input: str,
        ai_output: str,
        context,
        existing_memories,
        **kwargs,
    ) -> ProcessedLongTermMemory:
        return ProcessedLongTermMemory(
            content=f"{user_input}\n{ai_output}",
            summary="User loves oolong tea",
            classification=MemoryClassification.ESSENTIAL,
            importance=MemoryImportanceLevel.HIGH,
            conversation_id=chat_id,
            classification_reason="test-classification",
            promotion_eligible=True,
            is_user_context=True,
            symbolic_anchors=["tea"],
            x_coord=0.5,
            y_coord=0.0,
            z_coord=0.25,
        )

    async def detect_duplicates(self, processed_memory, existing_memories):
        return None

    def should_filter_memory(self, processed_memory, filters):
        return False


def test_openai_recorder_records_conversation():
    mem = _create_memoria()
    kwargs = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-test",
    }
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Hi"))],
        usage=types.SimpleNamespace(total_tokens=5),
    )

    with patch.object(mem, "record_conversation") as mock_record:
        openai_recorder.record_conversation(mem, kwargs, response)

    mock_record.assert_called_once()
    called = mock_record.call_args.kwargs
    assert called["user_input"] == "Hello"
    assert called["ai_output"] == "Hi"
    assert called["model"] == "gpt-test"
    assert called["metadata"]["tokens_used"] == 5


def test_assemble_streamed_response_flattens_multimodal_chunks():
    chunks = [
        types.SimpleNamespace(
            id="chunk-1",
            created=123,
            model="gpt-stream",
            object="chat.completion.chunk",
            choices=[
                types.SimpleNamespace(
                    index=0,
                    delta=types.SimpleNamespace(role="assistant"),
                )
            ],
        ),
        types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    index=0,
                    delta=types.SimpleNamespace(
                        content=[
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": " "},
                            {"type": "text", "text": "world"},
                        ]
                    ),
                )
            ],
        ),
        types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    index=0,
                    delta=types.SimpleNamespace(
                        tool_calls=[
                            types.SimpleNamespace(
                                index=0,
                                id="call_1",
                                type="function",
                                function=types.SimpleNamespace(
                                    name="search",
                                    arguments={"query": "cats"},
                                ),
                            )
                        ]
                    ),
                )
            ],
        ),
        types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    delta=types.SimpleNamespace(),
                )
            ],
        ),
    ]

    original_arguments = chunks[2].choices[0].delta.tool_calls[0].function.arguments

    assembled = openai_recorder.assemble_streamed_response(chunks)

    assert assembled is not None
    choice = assembled.choices[0]
    assert choice.message.content == "Hello world"
    assert choice.finish_reason == "stop"

    tool_call = choice.message.tool_calls[0]
    assert tool_call.function.name == "search"
    assert isinstance(tool_call.function.arguments, str)
    assert '"cats"' in tool_call.function.arguments

    # Ensure the streaming assembler does not mutate the original chunk objects.
    assert original_arguments == {"query": "cats"}


def test_assemble_streamed_response_stringifies_function_call_arguments():
    chunks = [
        types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    index=0,
                    delta=types.SimpleNamespace(
                        function_call={
                            "name": "classify",
                            "arguments": {"label": "positive"},
                        }
                    ),
                )
            ],
        )
    ]

    assembled = openai_recorder.assemble_streamed_response(chunks)

    assert assembled is not None
    choice = assembled.choices[0]
    assert choice.message.content is None
    assert choice.message.function_call == {
        "name": "classify",
        "arguments": '{"label": "positive"}',
    }


def test_anthropic_recorder_records_conversation():
    mem = _create_memoria()
    kwargs = {
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "claude-test",
    }
    response = types.SimpleNamespace(
        content="Hello",
        usage=types.SimpleNamespace(input_tokens=2, output_tokens=3),
        stop_reason="end_turn",
        model="claude-test",
    )

    with patch.object(mem, "record_conversation") as mock_record:
        anthropic_recorder.record_conversation(mem, kwargs, response)

    mock_record.assert_called_once()
    called = mock_record.call_args.kwargs
    assert called["user_input"] == "Hi"
    assert called["ai_output"] == "Hello"
    assert called["model"] == "claude-test"
    assert called["metadata"]["tokens_used"] == 5


def test_conscious_promotion_uses_essential_category():
    mem = Memoria(database_connect="sqlite:///:memory:")
    agent = ConsciousAgent()

    summary = "User loves hiking"
    namespace = mem.namespace

    # Seed a conscious-info long-term memory that should be promoted.
    with mem.db_manager.SessionLocal() as session:
        session.add(
            LongTermMemory(
                memory_id="ltm-conscious-1",
                processed_data={
                    "text": summary,
                    "searchable_content": summary,
                    "emotional_intensity": 0.5,
                },
                importance_score=0.95,
                category_primary="profile",
                retention_type="long_term",
                namespace=namespace,
                timestamp=datetime.utcnow(),
                created_at=datetime.utcnow(),
                searchable_content=summary,
                summary=summary,
                classification="conscious-info",
                conscious_processed=False,
            )
        )
        session.commit()

    # Copy the memory into short-term storage via the conscious agent.
    rows = asyncio.run(agent._get_conscious_memories(mem.db_manager, namespace))
    assert rows
    copied = asyncio.run(
        agent._copy_memory_to_short_term(mem.db_manager, namespace, rows[0])
    )
    assert copied

    essential_conversations = mem.storage_service.get_essential_conversations(limit=5)
    assert essential_conversations
    assert essential_conversations[0]["category_primary"] == CONSCIOUS_CONTEXT_CATEGORY
    assert essential_conversations[0]["summary"] == summary

    conscious_prompt = mem.get_conscious_system_prompt()
    assert f"[ESSENTIAL_CONSCIOUS] {summary}" in conscious_prompt

    injected_messages = mem.conversation_manager.inject_context_with_history(
        session_id="conscious-essential-test",
        messages=[{"role": "user", "content": "Hello"}],
        memoria_instance=mem,
        mode="conscious",
    )
    system_messages = [
        message for message in injected_messages if message.get("role") == "system"
    ]
    assert system_messages
    assert (
        f"[{CONSCIOUS_CONTEXT_CATEGORY.upper()}] {summary}"
        in system_messages[0]["content"]
    )


def test_record_conversation_creates_long_and_short_term_entries():
    mem = _create_memoria()
    mem._enabled = True
    mem.memory_agent = DummyMemoryAgent()

    def immediate_schedule(chat_id, user_input, ai_output, model):
        asyncio.run(mem._process_memory_async(chat_id, user_input, ai_output, model))

    mem._schedule_memory_processing = immediate_schedule

    chat_id = mem.record_conversation(
        user_input="I love oolong tea", ai_output="Noted!", model="dummy"
    )
    assert chat_id

    with mem.db_manager.SessionLocal() as session:
        long_term_rows = session.query(LongTermMemory).all()
        short_term_rows = session.query(ShortTermMemory).all()

    assert len(long_term_rows) == 1
    assert len(short_term_rows) == 1

    essential = mem.get_essential_conversations(limit=5)
    assert essential
    assert any(conv["summary"] == "User loves oolong tea" for conv in essential)


def test_refresh_memory_last_access_helper_updates_without_touching_counts():
    mem = _create_memoria()
    namespace = mem.namespace

    long_memory_id = "ltm-refresh"
    short_memory_id = "stm-refresh"
    old_time = datetime.utcnow() - timedelta(days=10)

    with mem.db_manager.SessionLocal() as session:
        session.add(
            LongTermMemory(
                memory_id=long_memory_id,
                processed_data={"text": "Tea ritual"},
                importance_score=0.42,
                category_primary="profile",
                retention_type="long_term",
                namespace=namespace,
                timestamp=old_time,
                created_at=old_time,
                searchable_content="Tea ritual",
                summary="Tea ritual",
                access_count=5,
                last_accessed=old_time,
                symbolic_anchors=["tea"],
            )
        )
        session.add(
            ShortTermMemory(
                memory_id=short_memory_id,
                processed_data={"text": "Tea ritual"},
                importance_score=0.55,
                category_primary="manual_staged",
                retention_type="short_term",
                namespace=namespace,
                created_at=old_time,
                expires_at=old_time + timedelta(days=7),
                access_count=2,
                last_accessed=old_time,
                searchable_content="Tea ritual",
                summary="Tea ritual",
                symbolic_anchors=["tea"],
            )
        )
        session.commit()

    before_refresh = datetime.utcnow()
    mem.db_manager.refresh_memory_last_access(
        namespace, [long_memory_id, short_memory_id]
    )

    with mem.db_manager.SessionLocal() as session:
        refreshed_long = (
            session.query(LongTermMemory)
            .filter(
                LongTermMemory.namespace == namespace,
                LongTermMemory.memory_id == long_memory_id,
            )
            .one()
        )
        refreshed_short = (
            session.query(ShortTermMemory)
            .filter(
                ShortTermMemory.namespace == namespace,
                ShortTermMemory.memory_id == short_memory_id,
            )
            .one()
        )

    assert refreshed_long.last_accessed is not None
    assert refreshed_long.last_accessed >= before_refresh
    assert refreshed_long.access_count == 5
    assert refreshed_long.importance_score == 0.42

    assert refreshed_short.last_accessed is not None
    assert refreshed_short.last_accessed >= before_refresh
    assert refreshed_short.access_count == 2


def test_conversation_promotion_refreshes_related_memory_last_access():
    mem = _create_memoria()
    namespace = mem.namespace
    existing_id = "ltm-conversation-related"
    old_time = datetime.utcnow() - timedelta(days=3)

    with mem.db_manager.SessionLocal() as session:
        session.add(
            LongTermMemory(
                memory_id=existing_id,
                processed_data={"text": "Tea memories"},
                importance_score=0.61,
                category_primary="profile",
                retention_type="long_term",
                namespace=namespace,
                timestamp=old_time,
                created_at=old_time,
                searchable_content="Tea memories",
                summary="Tea memories",
                access_count=3,
                last_accessed=old_time,
                symbolic_anchors=["tea"],
            )
        )
        session.commit()

    staged = StagedManualMemory(
        memory_id="staged-convo",
        chat_id="chat-1",
        namespace=namespace,
        anchor="tea",
        text="Discussing tea",
        tokens=12,
        timestamp=datetime.now(timezone.utc),
        x_coord=0.0,
        y_coord=0.0,
        z_coord=0.0,
        symbolic_anchors=["tea"],
        metadata={"short_term_stored": False, "namespace": namespace},
    )
    decision = PromotionDecision(should_promote=True, score=0.9, threshold=0.5)
    heuristic_result = HeuristicConversationResult(
        staged=staged,
        decision=decision,
        summary="Discussing tea",
        anchor="tea",
        symbolic_anchors=["tea"],
        emotional_intensity=0.2,
        related_candidates=[],
    )

    before_call = datetime.utcnow()
    with patch(
        "memoria.core.memory.process_conversation_turn", return_value=heuristic_result
    ):
        response = mem.process_recorded_conversation_heuristic(
            chat_id="chat-1",
            user_input="Let's talk about tea",
            ai_output="Tea is delightful",
        )

    assert response["long_term_id"]

    with mem.db_manager.SessionLocal() as session:
        refreshed = (
            session.query(LongTermMemory)
            .filter(
                LongTermMemory.namespace == namespace,
                LongTermMemory.memory_id == existing_id,
            )
            .one()
        )

    assert refreshed.last_accessed is not None
    assert refreshed.last_accessed >= before_call
    assert refreshed.access_count == 3
    assert refreshed.importance_score == 0.61


def test_manual_promotion_refreshes_related_memory_last_access():
    mem = _create_memoria()
    namespace = mem.namespace
    existing_id = "ltm-manual-related"
    old_time = datetime.utcnow() - timedelta(days=4)

    with mem.db_manager.SessionLocal() as session:
        session.add(
            LongTermMemory(
                memory_id=existing_id,
                processed_data={"text": "Morning ritual"},
                importance_score=0.7,
                category_primary="profile",
                retention_type="long_term",
                namespace=namespace,
                timestamp=old_time,
                created_at=old_time,
                searchable_content="Morning ritual",
                summary="Morning ritual",
                access_count=4,
                last_accessed=old_time,
                symbolic_anchors=["ritual"],
            )
        )
        session.commit()

    decision = PromotionDecision(should_promote=True, score=0.92, threshold=0.5)
    before_call = datetime.utcnow()

    with patch("memoria.core.memory.score_staged_memory", return_value=decision):
        result_id = mem.store_memory(
            anchor="ritual",
            text="Daily ritual reflection",
            tokens=20,
            symbolic_anchors=["ritual"],
        )

    assert result_id

    with mem.db_manager.SessionLocal() as session:
        refreshed = (
            session.query(LongTermMemory)
            .filter(
                LongTermMemory.namespace == namespace,
                LongTermMemory.memory_id == existing_id,
            )
            .one()
        )

    assert refreshed.last_accessed is not None
    assert refreshed.last_accessed >= before_call
    assert refreshed.access_count == 4
    assert refreshed.importance_score == 0.7


def test_context_helpers_and_namespace_isolation(tmp_path):
    db_path = tmp_path / "memoria_context.sqlite"
    connection = f"sqlite:///{db_path}"

    mem_alpha = Memoria(
        database_connect=connection,
        namespace="alpha",
        conscious_ingest=True,
        auto_ingest=True,
    )

    mem_beta = None
    try:
        # Seed short-term essential memories (including duplicates to verify deduping)
        _seed_short_term_memory(
            mem_alpha.db_manager,
            namespace="alpha",
            summary="Alpha loves tea",
            memory_id="alpha-stm-1",
            importance=0.95,
        )
        _seed_short_term_memory(
            mem_alpha.db_manager,
            namespace="alpha",
            summary="Alpha loves tea",
            memory_id="alpha-stm-dup",
            importance=0.90,
        )
        _seed_short_term_memory(
            mem_alpha.db_manager,
            namespace="beta",
            summary="Beta loves coffee",
            memory_id="beta-stm-1",
        )

        # Seed long-term memories for both namespaces (with duplicates)
        _seed_long_term_memory(
            mem_alpha.db_manager,
            namespace="alpha",
            summary="Alpha works as engineer",
            memory_id="alpha-ltm-1",
        )
        _seed_long_term_memory(
            mem_alpha.db_manager,
            namespace="alpha",
            summary="Alpha works as engineer",
            memory_id="alpha-ltm-dup",
        )
        _seed_long_term_memory(
            mem_alpha.db_manager,
            namespace="beta",
            summary="Beta is designer",
            memory_id="beta-ltm-1",
        )

        mem_beta = Memoria(
            database_connect=connection,
            namespace="beta",
            conscious_ingest=True,
            auto_ingest=True,
        )

        # Conscious context retrieval returns deduplicated, namespace-specific data
        alpha_conscious_context = mem_alpha._get_conscious_context()
        assert alpha_conscious_context
        assert (
            sum(
                1
                for item in alpha_conscious_context
                if item.get("summary") == "Alpha loves tea"
            )
            == 1
        )
        assert all(
            item.get("summary") != "Beta loves coffee"
            for item in alpha_conscious_context
        )

        alpha_conscious_prompt = mem_alpha.get_conscious_system_prompt()
        assert "Alpha loves tea" in alpha_conscious_prompt
        assert alpha_conscious_prompt.lower().count("alpha loves tea") == 1

        # Auto-ingest context retrieval deduplicates long-term memories and ignores other namespaces
        alpha_auto_context = mem_alpha._get_auto_ingest_context(
            "Alpha works as engineer"
        )
        assert alpha_auto_context
        assert (
            sum(
                1
                for item in alpha_auto_context
                if item.get("summary") == "Alpha works as engineer"
            )
            == 1
        )
        assert all(
            item.get("summary") != "Beta is designer" for item in alpha_auto_context
        )

        alpha_auto_prompt = mem_alpha.get_auto_ingest_system_prompt(
            "Alpha works as engineer"
        )
        assert "Alpha works as engineer" in alpha_auto_prompt

        # Conversation manager injects context for both modes without errors
        auto_messages = mem_alpha.conversation_manager.inject_context_with_history(
            session_id="alpha-auto",
            messages=[{"role": "user", "content": "Alpha works as engineer"}],
            memoria_instance=mem_alpha,
            mode="auto",
        )
        auto_system_messages = [
            msg for msg in auto_messages if msg.get("role") == "system"
        ]
        assert auto_system_messages
        assert "Alpha works as engineer" in auto_system_messages[0]["content"]

        conscious_messages = mem_alpha.conversation_manager.inject_context_with_history(
            session_id="alpha-conscious",
            messages=[{"role": "user", "content": "Hello"}],
            memoria_instance=mem_alpha,
            mode="conscious",
        )
        conscious_system_messages = [
            msg for msg in conscious_messages if msg.get("role") == "system"
        ]
        assert conscious_system_messages
        assert "Alpha loves tea" in conscious_system_messages[0]["content"]

        # Namespace isolation: beta namespace only sees beta data
        beta_conscious_prompt = mem_beta.get_conscious_system_prompt()
        assert "Beta loves coffee" in beta_conscious_prompt
        assert "Alpha loves tea" not in beta_conscious_prompt

        beta_auto_context = mem_beta._get_auto_ingest_context("Beta is designer")
        assert beta_auto_context
        assert any(
            item.get("summary") == "Beta is designer" for item in beta_auto_context
        )
        assert all(
            item.get("summary") != "Alpha works as engineer"
            for item in beta_auto_context
        )

        beta_auto_prompt = mem_beta.get_auto_ingest_system_prompt("Beta is designer")
        assert "Beta is designer" in beta_auto_prompt
        assert "Alpha works as engineer" not in beta_auto_prompt

        beta_auto_messages = mem_beta.conversation_manager.inject_context_with_history(
            session_id="beta-auto",
            messages=[{"role": "user", "content": "Beta is designer"}],
            memoria_instance=mem_beta,
            mode="auto",
        )
        beta_system_messages = [
            msg for msg in beta_auto_messages if msg.get("role") == "system"
        ]
        assert beta_system_messages
        assert "Beta is designer" in beta_system_messages[0]["content"]
    finally:
        mem_alpha.cleanup()
        if mem_beta is not None:
            mem_beta.cleanup()
