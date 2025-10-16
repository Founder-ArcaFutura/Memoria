# This test is temporarily sequestered due to persistent, hard-to-debug failures
# related to database interactions and timing.
import datetime

import pytest

from memoria.database.models import ChatHistory, LongTermMemory, ShortTermMemory


def _apply_namespace(mem, namespace: str) -> None:
    """Synchronize Memoria and storage namespaces for targeted operations."""

    mem.namespace = namespace
    mem.storage_service.namespace = namespace


@pytest.mark.usefixtures("team_memoria_context")
class TestTeamMemoryFlows:
    def test_chat_crud_team_isolation(self, team_memoria_context):
        mem, teams, backend = team_memoria_context
        personal_namespace = mem.personal_namespace

        _apply_namespace(mem, teams["ops"]["namespace"])
        ops_chat_id = mem.record_conversation(
            user_input="ops status",
            ai_output="ack",
            metadata={"team": "ops", "backend": backend},
        )

        _apply_namespace(mem, personal_namespace)
        personal_chat_id = mem.record_conversation(
            user_input="personal reflection",
            ai_output="logged",
        )

        ops_history = mem.db_manager.get_chat_history(
            namespace=teams["ops"]["namespace"]
        )
        assert len(ops_history) == 1
        assert ops_history[0]["chat_id"] == ops_chat_id
        assert all(
            record["namespace"] == teams["ops"]["namespace"] for record in ops_history
        )

        personal_history = mem.db_manager.get_chat_history(namespace=personal_namespace)
        assert len(personal_history) == 1
        assert personal_history[0]["chat_id"] == personal_chat_id

        update_timestamp = datetime.datetime.utcnow()
        mem.db_manager.store_chat_history(
            chat_id=ops_chat_id,
            user_input="ops status",
            ai_output="ack-updated",
            timestamp=update_timestamp,
            session_id=mem.session_id,
            model="gpt-test",
            namespace=teams["ops"]["namespace"],
            metadata={"updated": True},
        )

        refreshed_ops = mem.db_manager.get_chat_history(
            namespace=teams["ops"]["namespace"]
        )
        assert refreshed_ops[0]["ai_output"] == "ack-updated"

        with mem.db_manager.SessionLocal() as session:
            session.query(ChatHistory).filter_by(chat_id=personal_chat_id).delete()
            session.commit()

        personal_after_delete = mem.db_manager.get_chat_history(
            namespace=personal_namespace
        )
        assert personal_after_delete == []

        remaining_ops = mem.db_manager.get_chat_history(
            namespace=teams["ops"]["namespace"]
        )
        assert len(remaining_ops) == 1

        _apply_namespace(mem, "legacy-namespace")
        legacy_chat_id = mem.record_conversation(
            user_input="legacy entry",
            ai_output="legacy response",
        )
        legacy_history = mem.db_manager.get_chat_history(namespace="legacy-namespace")
        assert len(legacy_history) == 1
        assert legacy_history[0]["chat_id"] == legacy_chat_id

        _apply_namespace(mem, personal_namespace)

    def test_short_term_crud_and_isolation(self, team_memoria_context):
        mem, teams, backend = team_memoria_context
        _apply_namespace(mem, mem.personal_namespace)

        shared = mem.store_memory(
            anchor="ops-shared",
            text="shared ops note",
            tokens=32,
            team_id="ops",
            share_with_team=True,
            promotion_weights={"threshold": 1.0},
            return_status=True,
        )
        assert shared["namespace"] == teams["ops"]["namespace"]

        withheld = mem.store_memory(
            anchor="ops-private",
            text="private ops note",
            tokens=24,
            team_id="ops",
            share_with_team=False,
            promotion_weights={"threshold": 1.0},
            return_status=True,
        )
        assert withheld["namespace"] == mem.personal_namespace

        research_default = mem.store_memory(
            anchor="research-default",
            text="research defaults to personal",
            tokens=16,
            team_id="research",
            promotion_weights={"threshold": 1.0},
            return_status=True,
        )
        assert research_default["namespace"] == mem.personal_namespace

        research_shared = mem.store_memory(
            anchor="research-shared",
            text="research override to share",
            tokens=28,
            team_id="research",
            share_with_team=True,
            promotion_weights={"threshold": 1.0},
            return_status=True,
        )
        assert research_shared["namespace"] == teams["research"]["namespace"]

        explicit_namespace = mem.store_memory(
            anchor="legacy-short",
            text="legacy namespace short term",
            tokens=20,
            namespace="legacy-short",
            promotion_weights={"threshold": 1.0},
            return_status=True,
        )
        assert explicit_namespace["namespace"] == "legacy-short"

        with mem.db_manager.SessionLocal() as session:
            shared_row = (
                session.query(ShortTermMemory)
                .filter_by(memory_id=shared["short_term_id"])
                .one()
            )
            assert shared_row.namespace == teams["ops"]["namespace"]

            withheld_row = (
                session.query(ShortTermMemory)
                .filter_by(memory_id=withheld["short_term_id"])
                .one()
            )
            assert withheld_row.namespace == mem.personal_namespace

            research_shared_row = (
                session.query(ShortTermMemory)
                .filter_by(memory_id=research_shared["short_term_id"])
                .one()
            )
            assert research_shared_row.namespace == teams["research"]["namespace"]

            legacy_row = (
                session.query(ShortTermMemory)
                .filter_by(memory_id=explicit_namespace["short_term_id"])
                .one()
            )
            assert legacy_row.namespace == "legacy-short"

        _apply_namespace(mem, mem.personal_namespace)
        assert (
            mem.update_memory(shared["short_term_id"], {"text": "no-update"}) is False
        )

        _apply_namespace(mem, teams["ops"]["namespace"])
        assert (
            mem.update_memory(shared["short_term_id"], {"text": "shared adjusted"})
            is True
        )

        _apply_namespace(mem, teams["ops"]["namespace"])
        assert mem.delete_memory(shared["short_term_id"]) is True

        with mem.db_manager.SessionLocal() as session:
            remaining_ops_stm = (
                session.query(ShortTermMemory)
                .filter_by(memory_id=shared["short_term_id"])
                .all()
            )
            assert remaining_ops_stm == []

            assert (
                session.query(ShortTermMemory)
                .filter_by(memory_id=withheld["short_term_id"])
                .count()
            ) == 1

        _apply_namespace(mem, mem.personal_namespace)

    def test_long_term_crud_and_isolation(self, team_memoria_context):
        mem, teams, backend = team_memoria_context
        _apply_namespace(mem, mem.personal_namespace)

        ops_shared = mem.store_memory(
            anchor="ops-long",
            text="ops shared long term",
            tokens=64,
            team_id="ops",
            promotion_weights={"threshold": 0.0},
            return_status=True,
        )
        assert ops_shared["namespace"] == teams["ops"]["namespace"]
        assert ops_shared["promoted"] is True

        research_private = mem.store_memory(
            anchor="research-long-private",
            text="research private long term",
            tokens=48,
            team_id="research",
            promotion_weights={"threshold": 0.0},
            return_status=True,
        )
        assert research_private["namespace"] == mem.personal_namespace
        assert research_private["promoted"] is True

        research_override = mem.store_memory(
            anchor="research-long-shared",
            text="research shared long term",
            tokens=52,
            team_id="research",
            share_with_team=True,
            promotion_weights={"threshold": 0.0},
            return_status=True,
        )
        assert research_override["namespace"] == teams["research"]["namespace"]

        legacy_long = mem.store_memory(
            anchor="legacy-long",
            text="legacy namespace long term",
            tokens=40,
            namespace="legacy-long",
            promotion_weights={"threshold": 0.0},
            return_status=True,
        )
        assert legacy_long["namespace"] == "legacy-long"
        assert legacy_long["promoted"] is True

        with mem.db_manager.SessionLocal() as session:
            ops_row = (
                session.query(LongTermMemory)
                .filter_by(memory_id=ops_shared["memory_id"])
                .one()
            )
            assert ops_row.namespace == teams["ops"]["namespace"]

            research_row = (
                session.query(LongTermMemory)
                .filter_by(memory_id=research_private["memory_id"])
                .one()
            )
            assert research_row.namespace == mem.personal_namespace

            research_shared_row = (
                session.query(LongTermMemory)
                .filter_by(memory_id=research_override["memory_id"])
                .one()
            )
            assert research_shared_row.namespace == teams["research"]["namespace"]

            legacy_row = (
                session.query(LongTermMemory)
                .filter_by(memory_id=legacy_long["memory_id"])
                .one()
            )
            assert legacy_row.namespace == "legacy-long"

        _apply_namespace(mem, mem.personal_namespace)
        assert (
            mem.update_memory(ops_shared["memory_id"], {"text": "no-op update"})
            is False
        )

        _apply_namespace(mem, teams["ops"]["namespace"])
        assert (
            mem.update_memory(ops_shared["memory_id"], {"text": "ops revised"}) is True
        )

        _apply_namespace(mem, teams["ops"]["namespace"])
        assert mem.delete_memory(ops_shared["memory_id"]) is True

        with mem.db_manager.SessionLocal() as session:
            remaining = (
                session.query(LongTermMemory)
                .filter_by(memory_id=ops_shared["memory_id"])
                .all()
            )
            assert remaining == []

            assert (
                session.query(LongTermMemory)
                .filter_by(memory_id=research_private["memory_id"])
                .count()
            ) == 1

        _apply_namespace(mem, mem.personal_namespace)
