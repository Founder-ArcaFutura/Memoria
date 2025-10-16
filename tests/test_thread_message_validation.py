from __future__ import annotations

from datetime import datetime, timezone

from memoria.schemas import ThreadMessage


def test_thread_message_accepts_alias_and_extra_fields():
    payload = {
        "role": "narrator",
        "content": "Ritual fire rises.",
        "anchor": "invocation",
        "tokens": 3,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sequence_index": 2,
    }

    message = ThreadMessage(**payload)

    assert message.text == "Ritual fire rises."
    assert message.content == "Ritual fire rises."
    assert message.sequence_index == 2


def test_thread_message_configuration_flags_exposed():
    config = getattr(ThreadMessage, "model_config", None)

    if config is not None:
        assert config.get("populate_by_name") is True
        assert config.get("extra") == "allow"
        assert config.get("validate_by_name") is True
    else:
        # Fallback for environments pinned to Pydantic v1
        legacy_config = ThreadMessage.Config
        assert legacy_config.allow_population_by_field_name is True
        assert legacy_config.extra == "allow"


def test_thread_message_text_to_content_round_trip():
    payload = {
        "role": "scribe",
        "text": "Participants speak in unison.",
        "anchor": "response",
        "tokens": 4,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    message = ThreadMessage(**payload)

    assert message.text == "Participants speak in unison."
    assert message.content == "Participants speak in unison."
