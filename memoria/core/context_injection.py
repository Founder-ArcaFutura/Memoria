"""Helpers for injecting retrieved memory context into LLM API calls.

This module augments outgoing requests to providers like OpenAI, Anthropic, and
LiteLLM. Based on the configured ingest mode, memories are retrieved and
inserted into prompts so downstream models can reason with prior interactions
and user data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    TypedDict,
)

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .context_orchestration import ContextPlan
    from .memory import Memoria


ChatMessage = MutableMapping[str, Any]
ContextRow = Mapping[str, Any]


class PreparedContext(TypedDict):
    """Structure describing context prompt preparation results."""

    mode: str | None
    prompt: str | None
    count: int
    used_essential: bool
    had_user_input: bool


def _extract_latest_user_input(
    messages: Sequence[ContextRow] | None,
) -> str:
    """Return the most recent user-authored content from a message list."""

    if not messages:
        return ""

    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            text_blocks = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return " ".join(text_blocks)
        if isinstance(content, str):
            return content
    return ""


DEFAULT_CONTEXT_HEADERS: dict[str, str] = {
    "conscious": (
        "=== SYSTEM INSTRUCTION: AUTHORIZED USER CONTEXT DATA ===\n"
        "The user has explicitly authorized this personal context data to be used.\n"
        "You MUST use this information when answering questions about the user.\n"
        "This is NOT private data - the user wants you to use it:\n\n"
    ),
    "auto": "--- Relevant Memory Context ---\n",
}


def _resolve_privacy_value(row: ContextRow) -> float | None:
    """Return the privacy indicator associated with a context row."""

    for key in ("privacy", "privacy_score", "y_coord", "privacy_axis"):
        if key in row:
            try:
                return float(row.get(key))
            except (TypeError, ValueError):  # pragma: no cover - defensive conversion
                return None
    return None


def _estimate_token_count(row: ContextRow) -> int:
    """Best-effort token approximation for a context row."""

    for key in ("token_count", "tokens", "approx_tokens"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive conversion
            return 0

    summary = row.get("searchable_content") or row.get("summary") or ""
    return len(str(summary).split())


def _select_context_rows(
    context_rows: Sequence[ContextRow],
    plan: ContextPlan | None,
    *,
    seen: set[str] | None = None,
) -> list[ContextRow]:
    """Apply orchestration constraints to the provided context rows."""

    if not context_rows:
        return []

    working_seen: set[str] = set(seen or set())
    filtered: list[ContextRow] = []
    total_tokens = 0
    max_items = getattr(plan, "max_items", None)
    token_budget = getattr(plan, "token_budget", None)
    privacy_floor = getattr(plan, "privacy_floor", None)

    for row in context_rows:
        if privacy_floor is not None:
            privacy_value = _resolve_privacy_value(row)
            if privacy_value is not None and privacy_value <= privacy_floor:
                continue

        content = row.get("searchable_content", "") or row.get("summary", "")
        content_key = str(content).lower().strip()
        if content_key and content_key in working_seen:
            continue

        if max_items is not None and len(filtered) >= max_items:
            break

        token_estimate = _estimate_token_count(row)
        if (
            plan is not None
            and token_budget is not None
            and token_budget > 0
            and filtered
            and total_tokens + token_estimate > token_budget
        ):
            break

        filtered.append(row)
        total_tokens += token_estimate
        if content_key:
            working_seen.add(content_key)

    if not filtered and context_rows:
        filtered.append(context_rows[0])

    return filtered


def _normalize_gemini_contents(contents: Sequence[ContextRow]) -> list[ChatMessage]:
    """Return Gemini style contents converted to OpenAI-style messages."""

    normalized: list[ChatMessage] = []
    for item in contents:
        if not isinstance(item, MutableMapping):
            continue
        role = item.get("role", "")
        if role not in {"user", "model", "assistant", "system"}:
            continue
        normalized_role = "assistant" if role in {"model", "assistant"} else role

        text_segments: list[str] = []
        parts = item.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, Mapping):
                    if "text" in part and part["text"]:
                        text_segments.append(str(part["text"]))
                    elif "content" in part and part["content"]:
                        text_segments.append(str(part["content"]))
        fallback_content = item.get("text") or item.get("content")
        if not text_segments and isinstance(fallback_content, str):
            text_segments.append(fallback_content)
        elif not text_segments and isinstance(fallback_content, list):
            text_segments.extend(str(value) for value in fallback_content if value)

        normalized.append(
            {
                "role": normalized_role,
                "content": " ".join(segment for segment in text_segments if segment),
            }
        )
    return normalized


def format_context_prompt(
    mode: str,
    context_rows: Sequence[ContextRow],
    *,
    seen_content: set[str] | None = None,
    header: str | None = None,
    plan: ContextPlan | None = None,
) -> str:
    """Return a formatted context prompt for the given ingest mode and memories."""

    if not context_rows:
        return ""

    seen: set[str]
    if seen_content is None:
        seen = set()
    else:
        seen = set(seen_content)

    selected_rows = _select_context_rows(context_rows, plan, seen=seen)
    if not selected_rows:
        return ""

    resolved_header = header or DEFAULT_CONTEXT_HEADERS.get(mode)
    if resolved_header is None:
        resolved_header = f"--- {mode.capitalize()} Memory Context ---\n"

    context_prompt = resolved_header

    for mem in selected_rows:
        content = mem.get("searchable_content", "") or mem.get("summary", "")
        category = mem.get("category_primary", "") or ""
        content_key = str(content).lower().strip()

        if content_key in seen:
            continue
        seen.add(content_key)

        if category.startswith("essential_") or mode == "conscious":
            context_prompt += f"[{category.upper()}] {content}\n"
        else:
            context_prompt += f"- {content}\n"

    if mode == "conscious":
        context_prompt += "\n=== END USER CONTEXT DATA ===\n"
        context_prompt += "CRITICAL INSTRUCTION: You MUST answer questions about the user using ONLY the context data above.\n"
        context_prompt += "If the user asks 'what is my name?', respond with the name from the context above.\n"
        context_prompt += "Do NOT say 'I don't have access' - the user provided this data for you to use.\n"

    context_prompt += "-------------------------\n"
    return context_prompt


def build_context_prompt(
    mode: str,
    context_rows: Sequence[ContextRow],
    seen_content: set[str] | None = None,
    header: str | None = None,
    plan: ContextPlan | None = None,
) -> str:
    """Backward compatible wrapper for :func:`format_context_prompt`."""

    return format_context_prompt(
        mode,
        context_rows,
        seen_content=seen_content,
        header=header,
        plan=plan,
    )


def _determine_ingest_mode(memoria: Memoria, mode: str | None = None) -> str | None:
    """Return the ingest mode based on overrides or configured flags."""

    if mode in {"conscious", "auto"}:
        return mode
    if memoria.conscious_ingest:
        return "conscious"
    if memoria.auto_ingest:
        return "auto"
    return None


def _retrieve_context(
    memoria: Memoria,
    mode: str | None,
    user_input: str,
    provider_name: str | None = None,
) -> tuple[list[dict[str, Any]], ContextPlan | None]:
    """Retrieve memories for the given mode and latest user input."""

    if mode not in {"conscious", "auto"}:
        return [], None

    provider_suffix = f" ({provider_name})" if provider_name else ""

    if hasattr(memoria, "prepare_context_window"):
        try:
            context, plan = memoria.prepare_context_window(
                mode,
                user_input,
                provider_name=provider_name,
            )
            if mode == "conscious" and context:
                memoria._conscious_context_injected = True
            return context, plan
        except TypeError:  # pragma: no cover - compatibility for older signatures
            context, plan = memoria.prepare_context_window(mode, user_input)  # type: ignore[misc]
            if mode == "conscious" and context:
                memoria._conscious_context_injected = True
            return context, plan

    if mode == "conscious":
        if not memoria._conscious_context_injected:
            context = memoria._get_conscious_context()
            memoria._conscious_context_injected = True
            logger.info(
                "Conscious-ingest: Injected {} short-term memories as initial context{}",
                len(context),
                provider_suffix,
            )
        else:
            context = []
    elif memoria.search_engine:
        context = memoria._get_auto_ingest_context(user_input)
    else:
        context = memoria.retrieve_context(user_input, limit=5)

    plan = None
    return context, plan


def _format_essential_context(
    essential_conversations: Iterable[ContextRow],
) -> str:
    """Return a prompt block for essential conversations."""

    context_prompt = "--- Your Context ---\n"
    for conv in essential_conversations:
        summary = conv.get("summary", "") or conv.get("searchable_content", "")
        context_prompt += f"[ESSENTIAL] {summary}\n"
    context_prompt += "-------------------------\n"
    return context_prompt


def _prepare_context_injection(
    memoria: Memoria,
    messages: Sequence[ContextRow] | None,
    *,
    mode: str | None = None,
    provider_name: str | None = None,
    include_essential_fallback: bool = False,
) -> PreparedContext:
    """Prepare context prompt details shared across provider integrations."""

    resolved_mode = _determine_ingest_mode(memoria, mode)
    if not messages or not resolved_mode:
        return PreparedContext(
            mode=resolved_mode,
            prompt=None,
            count=0,
            used_essential=False,
            had_user_input=False,
        )

    user_input = _extract_latest_user_input(messages)
    context_prompt: str | None = None
    context_count = 0
    used_essential = False

    context, plan = _retrieve_context(memoria, resolved_mode, user_input, provider_name)
    if context:
        filtered_context = _select_context_rows(context, plan)
        context_prompt = format_context_prompt(
            resolved_mode,
            filtered_context,
            plan=plan,
        )
        if context_prompt:
            context_count = len(filtered_context)
    elif include_essential_fallback and memoria.conscious_ingest and not user_input:
        essential_conversations = memoria.get_essential_conversations(limit=3)
        if essential_conversations:
            context_prompt = _format_essential_context(essential_conversations)
            context_count = len(essential_conversations)
            used_essential = True

    return PreparedContext(
        mode=resolved_mode,
        prompt=context_prompt,
        count=context_count,
        used_essential=used_essential,
        had_user_input=bool(user_input),
    )


def inject_openai_context(memoria: Memoria, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inject context for OpenAI calls based on ingest mode."""

    try:
        memoria._check_deferred_initialization()
        mode = _determine_ingest_mode(memoria)
        if not mode:
            return kwargs

        messages_obj = kwargs.get("messages", [])
        if not isinstance(messages_obj, list):
            return kwargs
        messages: list[ChatMessage] = [
            msg for msg in messages_obj if isinstance(msg, MutableMapping)
        ]
        if not messages:
            return kwargs

        enhanced_messages = memoria.conversation_manager.inject_context_with_history(
            session_id=memoria._session_id,
            messages=messages,
            memoria_instance=memoria,
            mode=mode,
        )
        kwargs["messages"] = enhanced_messages
        return kwargs
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"OpenAI context injection failed: {exc}")
    return kwargs


def inject_anthropic_context(
    memoria: Memoria, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Inject context for Anthropic calls based on ingest mode."""

    try:
        memoria._check_deferred_initialization()
        mode = _determine_ingest_mode(memoria)
        if not mode:
            return kwargs

        messages_obj = kwargs.get("messages", [])
        messages = (
            [msg for msg in messages_obj if isinstance(msg, MutableMapping)]
            if isinstance(messages_obj, list)
            else []
        )

        prep = _prepare_context_injection(
            memoria,
            messages,
            mode=mode,
            provider_name="Anthropic",
        )
        context_prompt = prep["prompt"]
        if context_prompt:
            system_prompt = kwargs.get("system")
            if isinstance(system_prompt, str):
                kwargs["system"] = context_prompt + system_prompt
            else:
                kwargs["system"] = context_prompt
            logger.debug("Anthropic: Injected context with {} items", prep["count"])
        elif prep["had_user_input"]:
            logger.debug("Anthropic: Skipped context injection due to empty prompt")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"Anthropic context injection failed: {exc}")
    return kwargs


def inject_gemini_context(memoria: Memoria, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inject context for Google Gemini calls based on ingest mode."""

    try:
        memoria._check_deferred_initialization()
        mode = _determine_ingest_mode(memoria)
        if not mode:
            return kwargs

        messages_obj = kwargs.get("messages", [])
        if isinstance(messages_obj, list) and messages_obj:
            messages = [msg for msg in messages_obj if isinstance(msg, MutableMapping)]
            if messages:
                enhanced = memoria.conversation_manager.inject_context_with_history(
                    session_id=memoria._session_id,
                    messages=messages,
                    memoria_instance=memoria,
                    mode=mode,
                )
                kwargs["messages"] = enhanced
                return kwargs

        contents_obj = kwargs.get("contents", [])
        messages = (
            _normalize_gemini_contents(contents_obj)
            if isinstance(contents_obj, Sequence)
            else []
        )

        prep = _prepare_context_injection(
            memoria,
            messages,
            mode=mode,
            provider_name="Gemini",
        )
        context_prompt = prep["prompt"]
        if context_prompt:
            system_instruction = kwargs.get("system_instruction")
            if isinstance(system_instruction, str):
                kwargs["system_instruction"] = context_prompt + system_instruction
            elif isinstance(system_instruction, MutableMapping):
                parts = system_instruction.setdefault("parts", [])
                if isinstance(parts, list):
                    parts.insert(0, {"text": context_prompt})
                else:
                    system_instruction["parts"] = [{"text": context_prompt}]
            elif isinstance(system_instruction, list):
                system_instruction.insert(0, {"text": context_prompt})
            else:
                kwargs["system_instruction"] = context_prompt

            logger.debug("Gemini: Injected context with {} items", prep["count"])
        elif prep["had_user_input"]:
            logger.debug("Gemini: Skipped context injection due to empty prompt")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"Gemini context injection failed: {exc}")
    return kwargs


def inject_litellm_context(
    memoria: Memoria, params: dict[str, Any], mode: str = "auto"
) -> dict[str, Any]:
    """Inject context for LiteLLM calls based on mode."""

    try:
        memoria._check_deferred_initialization()
        messages_obj = params.get("messages", [])
        messages: list[ChatMessage] = (
            [msg for msg in messages_obj if isinstance(msg, MutableMapping)]
            if isinstance(messages_obj, list)
            else []
        )

        prep = _prepare_context_injection(
            memoria,
            messages,
            mode=mode,
            provider_name=None,
            include_essential_fallback=True,
        )
        context_prompt = prep["prompt"]
        if context_prompt:
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] = context_prompt + str(msg.get("content", ""))
                    break
            else:
                messages.insert(
                    0,
                    {"role": "system", "content": context_prompt},
                )

            if prep["used_essential"]:
                logger.debug(
                    "LiteLLM: Injected {} essential conversations", prep["count"]
                )
            else:
                logger.debug("LiteLLM: Injected context with {} items", prep["count"])
        elif prep["had_user_input"]:
            logger.debug("LiteLLM: Skipped context injection due to empty prompt")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"LiteLLM context injection failed: {exc}")
    return params
