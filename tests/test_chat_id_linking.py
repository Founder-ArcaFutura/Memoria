import uuid
from datetime import datetime

from memoria import Memoria
from memoria.database.models import ChatHistory, LongTermMemory
from memoria.storage.service import StorageService


def test_store_memory_with_supplied_chat_id_links_existing_history():
    mem = Memoria(database_connect="sqlite:///:memory:", enable_short_term=False)
    chat_id = f"chat-{uuid.uuid4()}"
    now = datetime.utcnow()

    mem.db_manager.store_chat_history(
        chat_id=chat_id,
        user_input="existing user input",
        ai_output="existing ai output",
        timestamp=now,
        session_id=chat_id,
        model="unit-test",
        namespace=mem.namespace,
    )

    memory_id = mem.store_memory(
        anchor="link",
        text="linked memory",
        tokens=3,
        chat_id=chat_id,
    )

    with mem.db_manager.SessionLocal() as session:
        memory = session.query(LongTermMemory).filter_by(memory_id=memory_id).one()
        chat_history = session.query(ChatHistory).filter_by(chat_id=chat_id).all()

    assert memory.original_chat_id == chat_id
    assert len(chat_history) == 1
    assert chat_history[0].user_input == "existing user input"
    assert chat_history[0].ai_output == "existing ai output"


def test_store_memory_generates_chat_history_when_missing():
    mem = Memoria(database_connect="sqlite:///:memory:", enable_short_term=False)

    memory_id = mem.store_memory(
        anchor="auto",
        text="auto memory",
        tokens=2,
    )

    with mem.db_manager.SessionLocal() as session:
        memory = session.query(LongTermMemory).filter_by(memory_id=memory_id).one()
        chat_history = (
            session.query(ChatHistory).filter_by(chat_id=memory.original_chat_id).one()
        )

    assert memory.original_chat_id is not None
    assert chat_history.chat_id == memory.original_chat_id
    assert chat_history.user_input == StorageService.AUTOGEN_USER_PLACEHOLDER
    assert chat_history.ai_output == StorageService.AUTOGEN_AI_PLACEHOLDER
    assert chat_history.namespace == mem.namespace
    assert chat_history.session_id == memory.original_chat_id
    assert chat_history.tokens_used == 0
    assert chat_history.metadata == {"auto_generated": True}


def test_chat_history_preserves_raw_html_like_content():
    mem = Memoria(database_connect="sqlite:///:memory:", enable_short_term=False)

    chat_id = f"chat-{uuid.uuid4()}"
    user_input = "Hello <tag>friend</tag> & welcome"
    ai_output = 'Response with <tag attr="value">details</tag>'

    mem.db_manager.store_chat_history(
        chat_id=chat_id,
        user_input=user_input,
        ai_output=ai_output,
        timestamp=datetime.utcnow(),
        session_id=chat_id,
        model="unit-test",
        namespace=mem.namespace,
    )

    history = mem.db_manager.get_chat_history(namespace=mem.namespace, limit=1)

    assert history[0]["user_input"] == user_input
    assert history[0]["ai_output"] == ai_output
