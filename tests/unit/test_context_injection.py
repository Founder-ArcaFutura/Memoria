"""Unit tests for provider specific context injection helpers."""

from copy import deepcopy

import pytest

from memoria.core.context_injection import (
    build_context_prompt,
    format_context_prompt,
    inject_anthropic_context,
    inject_gemini_context,
    inject_litellm_context,
    inject_openai_context,
)
from memoria.core.conversation import ConversationManager


class DummyMemoria:
    """Minimal stub of the Memoria client for context injection tests."""

    def __init__(
        self,
        *,
        auto_ingest=False,
        conscious_ingest=False,
        search_engine=False,
        context_rows=None,
    ):
        self.auto_ingest = auto_ingest
        self.conscious_ingest = conscious_ingest
        self.search_engine = search_engine
        self._conscious_context_injected = False
        self._context_rows = context_rows or []
        self._session_id = "test-session"
        self.conversation_manager = ConversationManager()
        self.last_auto_ingest_input = None

    # Provider hooks -----------------------------------------------------
    def _check_deferred_initialization(self):
        """No-op for tests."""

    def _get_conscious_context(self):
        return deepcopy(self._context_rows)

    def _get_auto_ingest_context(self, user_input):
        self.last_auto_ingest_input = user_input
        return deepcopy(self._context_rows)

    def retrieve_context(
        self, _user_input, limit=5
    ):  # noqa: ARG002 - parity with production signature
        return deepcopy(self._context_rows[:limit])

    def get_essential_conversations(
        self, limit=3
    ):  # noqa: ARG002 - mirror production signature
        return []

    def prepare_context_window(
        self, mode, query, *, provider_name=None
    ):  # noqa: ARG002 - test stub
        if mode == "conscious":
            context = self._get_conscious_context()
        elif mode == "auto":
            context = self._get_auto_ingest_context(query) if query else []
        else:
            context = []
        return context, None


def _prepare_messages(include_system=True, anthropic_style=False):
    system = {"role": "system", "content": "Existing system."}
    assistant = {"role": "assistant", "content": "Hello!"}
    if anthropic_style:
        user_content = [
            {"type": "text", "text": "Please help"},
            {"type": "text", "text": "with my task."},
        ]
    else:
        user_content = "Please help with my task."
    user = {"role": "user", "content": user_content}
    messages = [assistant, user]
    if include_system:
        messages.insert(0, system)
    return messages


def _extract_context_from_system_message(messages):
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content", "")
            if content.endswith("Existing system."):
                return content[: -len("Existing system.")]
            return content
    raise AssertionError("No system message found")


def _create_memoria(
    ingest_mode, context_rows, *, auto_ingest=None, conscious_ingest=None
):
    if auto_ingest is None:
        auto_ingest = ingest_mode == "auto"
    if conscious_ingest is None:
        conscious_ingest = ingest_mode == "conscious"
    return DummyMemoria(
        auto_ingest=auto_ingest,
        conscious_ingest=conscious_ingest,
        context_rows=context_rows,
    )


@pytest.mark.parametrize("ingest_mode", ["auto", "conscious"])
def test_anthropic_and_litellm_prompts_match(ingest_mode):
    context_rows = [
        {
            "searchable_content": "User loves green tea",
            "category_primary": "preference",
        },
        {"summary": "Critical allergy note", "category_primary": "essential_health"},
        {
            "searchable_content": "User loves green tea",
            "category_primary": "preference",
        },
    ]

    anthropic_memoria = _create_memoria(ingest_mode, context_rows)
    litellm_memoria = _create_memoria(ingest_mode, context_rows)

    anthropic_kwargs = {
        "messages": _prepare_messages(include_system=False, anthropic_style=True),
        "system": "Existing system.",
    }
    litellm_params = {"messages": _prepare_messages(include_system=True)}

    anthropic_result = inject_anthropic_context(
        anthropic_memoria, deepcopy(anthropic_kwargs)
    )
    litellm_result = inject_litellm_context(
        litellm_memoria, deepcopy(litellm_params), mode=ingest_mode
    )

    anthropic_prompt = anthropic_result["system"][: -len("Existing system.")]
    litellm_prompt = _extract_context_from_system_message(litellm_result["messages"])
    expected_prompt = format_context_prompt(ingest_mode, context_rows)

    assert anthropic_prompt == litellm_prompt == expected_prompt
    if ingest_mode == "conscious":
        assert "AUTHORIZED USER CONTEXT DATA" in anthropic_prompt
        assert anthropic_memoria._conscious_context_injected is True
        assert litellm_memoria._conscious_context_injected is True
    else:
        assert "AUTHORIZED USER CONTEXT DATA" not in anthropic_prompt
        assert anthropic_memoria._conscious_context_injected is False
        assert litellm_memoria._conscious_context_injected is False


@pytest.mark.parametrize("ingest_mode", ["auto", "conscious"])
def test_gemini_context_injection_with_contents(ingest_mode):
    context_rows = [
        {
            "searchable_content": "User prefers short replies",
            "category_primary": "preference",
        },
        {"summary": "Important allergy", "category_primary": "essential_health"},
    ]

    memoria = _create_memoria(ingest_mode, context_rows)
    kwargs = {
        "contents": [
            {"role": "user", "parts": [{"text": "Summarize my notes"}]},
            {"role": "model", "parts": [{"text": "Sure"}]},
        ],
        "system_instruction": "Existing system.",
    }

    result = inject_gemini_context(memoria, deepcopy(kwargs))
    prompt = format_context_prompt(ingest_mode, context_rows)

    assert "system_instruction" in result
    assert result["system_instruction"].startswith(prompt)
    assert result["system_instruction"].endswith("Existing system.")
    if ingest_mode == "conscious":
        assert memoria._conscious_context_injected is True
    else:
        assert memoria._conscious_context_injected is False


def test_gemini_context_injection_with_messages():
    context_rows = [
        {"searchable_content": "Knows Rust", "category_primary": "skill"},
        {"summary": "Needs concise answers", "category_primary": "preference"},
    ]

    memoria = _create_memoria("auto", context_rows)
    kwargs = {"messages": _prepare_messages(include_system=False)}

    result = inject_gemini_context(memoria, deepcopy(kwargs))
    messages = result["messages"]

    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(
        format_context_prompt("auto", context_rows)
    )


@pytest.mark.parametrize(
    "mode, expected_header, expected_markers",
    [
        (
            "auto",
            "--- Relevant Memory Context ---\n",
            ["[ESSENTIAL_HEALTH] Critical allergy note", "- Loves matcha tea"],
        ),
        (
            "conscious",
            "=== SYSTEM INSTRUCTION: AUTHORIZED USER CONTEXT DATA ===\n",
            [
                "[PREFERENCE] Loves matcha tea",
                "[ESSENTIAL_HEALTH] Critical allergy note",
                "=== END USER CONTEXT DATA ===",
            ],
        ),
    ],
)
def test_format_context_prompt_formats_entries(mode, expected_header, expected_markers):
    context_rows = [
        {"searchable_content": "Loves matcha tea", "category_primary": "preference"},
        {"summary": "Critical allergy note", "category_primary": "essential_health"},
        {"summary": "Loves matcha tea", "category_primary": "preference"},
    ]

    prompt = format_context_prompt(mode, context_rows)

    assert prompt.startswith(expected_header)
    for marker in expected_markers:
        assert marker in prompt
    # Duplicate entries should only appear once
    assert prompt.count("Loves matcha tea") == 1
    assert prompt.rstrip().endswith("-------------------------")


def test_build_context_prompt_wraps_format_helper():
    context_rows = [
        {"searchable_content": "Loves matcha tea", "category_primary": "preference"},
        {"summary": "Critical allergy note", "category_primary": "essential_health"},
    ]

    assert build_context_prompt("auto", context_rows) == format_context_prompt(
        "auto", context_rows
    )
    assert build_context_prompt("conscious", context_rows) == format_context_prompt(
        "conscious", context_rows
    )


@pytest.mark.parametrize("mode", ["auto", "conscious"])
def test_conversation_manager_uses_shared_prompt(mode):
    context_rows = [
        {"searchable_content": "Loves matcha tea", "category_primary": "preference"},
        {"summary": "Critical allergy note", "category_primary": "essential_health"},
    ]
    memoria = _create_memoria(mode, context_rows)
    manager = ConversationManager()

    messages = _prepare_messages(include_system=False)
    enhanced_messages = manager.inject_context_with_history(
        "session-123", deepcopy(messages), memoria, mode=mode
    )

    assert enhanced_messages[0]["role"] == "system"
    expected_prompt = format_context_prompt(mode, context_rows)
    assert enhanced_messages[0]["content"] == expected_prompt


def test_conversation_manager_normalizes_user_messages_for_auto_ingest():
    memoria = _create_memoria("auto", [])
    manager = ConversationManager()

    first_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Needs"},
                {"type": "text", "text": "analysis"},
            ],
        }
    ]

    # Should not raise even though the content is expressed as text segments.
    manager.inject_context_with_history(
        "session-lists", deepcopy(first_messages), memoria, mode="auto"
    )

    assert memoria.last_auto_ingest_input == "Needs analysis"

    session = manager.get_or_create_session("session-lists")
    history_messages = session.get_history_messages()
    assert history_messages[0]["content"] == "Needs analysis"

    manager.add_assistant_message("session-lists", "On it.")

    second_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Provide"},
                {"type": "text", "text": "details"},
            ],
        }
    ]

    enhanced_messages = manager.inject_context_with_history(
        "session-lists", deepcopy(second_messages), memoria, mode="auto"
    )

    assert memoria.last_auto_ingest_input == "Provide details"
    assert enhanced_messages[0]["role"] == "system"
    assert "User: Needs analysis" in enhanced_messages[0]["content"]
    assert not isinstance(enhanced_messages[0]["content"], list)


def test_openai_injects_conscious_context_when_both_modes_enabled():
    context_rows = [
        {"searchable_content": "Prefers jasmine tea", "category_primary": "preference"},
        {"summary": "Emergency contact: Ada", "category_primary": "essential_health"},
    ]
    memoria = _create_memoria(
        "conscious",
        context_rows,
        auto_ingest=True,
        conscious_ingest=True,
    )
    kwargs = {"messages": _prepare_messages(include_system=False)}

    result = inject_openai_context(memoria, deepcopy(kwargs))

    system_messages = [m for m in result["messages"] if m.get("role") == "system"]
    assert len(system_messages) == 1

    system_content = system_messages[0]["content"]
    assert "AUTHORIZED USER CONTEXT DATA" in system_content
    assert "--- Relevant Memory Context ---" not in system_content
    assert "Prefers jasmine tea" in system_content

    session = memoria.conversation_manager.sessions[memoria._session_id]
    assert len(session.messages) == 1
    assert session.messages[0].role == "user"
