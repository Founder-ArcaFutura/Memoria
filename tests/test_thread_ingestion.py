from datetime import datetime, timedelta, timezone

from memoria import Memoria


def _create_thread_payload(thread_id: str) -> dict:
    base_time = datetime.now(timezone.utc)
    return {
        "thread_id": thread_id,
        "session_id": f"session-{thread_id}",
        "shared_symbolic_anchors": ["ritual", "shared-sigil"],
        "ritual": {
            "name": "dawn-circle",
            "phase": "opening",
            "location": "grove",
        },
        "messages": [
            {
                "role": "acolyte",
                "content": "We gather to light the first flame.",
                "anchor": "invocation",
                "tokens": 9,
                "timestamp": (base_time - timedelta(minutes=3)).isoformat(),
                "y_coord": -5.0,
                "z_coord": 4.0,
                "symbolic_anchors": ["fire", "dawn"],
            },
            {
                "role": "guide",
                "content": "The circle breathes together in harmony.",
                "anchor": "response",
                "tokens": 8,
                "timestamp": (base_time - timedelta(minutes=1)).isoformat(),
                "y_coord": -4.0,
                "z_coord": 3.0,
                "symbolic_anchors": ["breath"],
            },
        ],
    }


def test_ingest_thread_stages_messages_and_links(tmp_path):
    mem = Memoria(database_connect="sqlite:///:memory:", enable_short_term=True)
    mem.enable()

    payload = _create_thread_payload("thread-morning")
    result = mem.ingest_thread(payload)

    assert result["thread_id"] == "thread-morning"
    assert len(result["messages"]) == 2
    assert "ritual" in result and result["ritual"]["name"] == "dawn-circle"

    first_message = result["messages"][0]
    assert first_message["sequence_index"] == 0
    assert "shared-sigil" in first_message["symbolic_anchors"]

    memories = mem.retrieve_memories_by_anchor(["shared-sigil"])
    assert len(memories) >= 2

    thread_view = mem.get_thread("thread-morning")
    assert thread_view is not None
    assert thread_view["centroid"]["y"] is not None
    assert len(thread_view["messages"]) == 2
    assert thread_view["messages"][0]["role"] == "acolyte"


def test_get_threads_for_memory_returns_memberships(tmp_path):
    mem = Memoria(database_connect="sqlite:///:memory:", enable_short_term=True)
    mem.enable()

    payload = _create_thread_payload("thread-evening")
    result = mem.ingest_thread(payload)
    member_id = result["messages"][0]["memory_id"]

    memberships = mem.get_threads_for_memory(member_id)
    assert memberships
    assert memberships[0]["thread_id"] == "thread-evening"
    assert memberships[0]["sequence_index"] == 0
