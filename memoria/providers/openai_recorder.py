"""Utilities for recording OpenAI conversations."""

import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from loguru import logger


def _extract_openai_user_input(messages: list[dict]) -> str:
    """Extract user input from OpenAI messages with support for complex content types."""
    user_input = ""
    try:
        # Find the last user message
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")

                if isinstance(content, str):
                    # Simple string content
                    user_input = content
                elif isinstance(content, list):
                    # Complex content (vision, multiple parts)
                    text_parts = []
                    image_count = 0

                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") == "image_url":
                                image_count += 1

                    user_input = " ".join(text_parts)
                    # Add image indicator if present
                    if image_count > 0:
                        user_input += f" [Contains {image_count} image(s)]"

                break
    except Exception as e:
        logger.debug(f"Error extracting user input: {e}")
        user_input = "[Error extracting user input]"

    return user_input


def _extract_openai_ai_output(response) -> str:
    """Extract AI output from OpenAI response with support for various response types."""
    ai_output = ""
    try:
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]

            if hasattr(choice, "message") and choice.message:
                message = choice.message

                # Handle regular text content
                if hasattr(message, "content") and message.content:
                    ai_output = message.content

                # Handle function/tool calls
                elif hasattr(message, "tool_calls") and message.tool_calls:
                    tool_descriptions = []
                    for tool_call in message.tool_calls:
                        if hasattr(tool_call, "function"):
                            func_name = tool_call.function.name
                            func_args = tool_call.function.arguments
                            tool_descriptions.append(
                                f"Called {func_name} with {func_args}"
                            )
                    ai_output = "[Tool calls: " + "; ".join(tool_descriptions) + "]"

                # Handle function calls (legacy format)
                elif hasattr(message, "function_call") and message.function_call:
                    func_call = message.function_call
                    func_name = func_call.get("name", "unknown")
                    func_args = func_call.get("arguments", "{}")
                    ai_output = f"[Function call: {func_name} with {func_args}]"

                else:
                    ai_output = "[No content - possible function/tool call]"

    except Exception as e:
        logger.debug(f"Error extracting AI output: {e}")
        ai_output = "[Error extracting AI response]"

    return ai_output


def _extract_openai_metadata(
    kwargs: dict[str, Any], response, tokens_used: int
) -> dict[str, Any]:
    """Extract comprehensive metadata from OpenAI request and response."""
    metadata: dict[str, Any] = {
        "integration": "openai_auto",
        "api_type": "chat_completions",
        "tokens_used": tokens_used,
        "auto_recorded": True,
    }

    try:
        # Add request metadata
        if "temperature" in kwargs:
            metadata["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            metadata["max_tokens"] = kwargs["max_tokens"]
        if "tools" in kwargs:
            metadata["has_tools"] = True
            metadata["tool_count"] = len(kwargs["tools"])
        if "functions" in kwargs:
            metadata["has_functions"] = True
            metadata["function_count"] = len(kwargs["functions"])

        # Add response metadata
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "finish_reason"):
                metadata["finish_reason"] = choice.finish_reason

        # Add detailed token usage if available
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            metadata.update(
                {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
            )

        # Detect content types
        messages = kwargs.get("messages", [])
        has_images = False
        message_count = len(messages)

        for message in messages:
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            has_images = True
                            break
                if has_images:
                    break

        metadata["message_count"] = message_count
        metadata["has_images"] = has_images

    except Exception as e:
        logger.debug(f"Error extracting metadata: {e}")

    return metadata


def _get_attribute(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def _flatten_content_piece(content_piece: Any) -> str:
    """Convert streamed content fragments into a single string."""

    if content_piece is None:
        return ""

    if isinstance(content_piece, str):
        return content_piece

    if isinstance(content_piece, Sequence) and not isinstance(
        content_piece, (bytes, bytearray)
    ):
        return "".join(_flatten_content_piece(item) for item in content_piece)

    if isinstance(content_piece, dict):
        if "text" in content_piece:
            return _flatten_content_piece(content_piece["text"])
        if "content" in content_piece:
            return _flatten_content_piece(content_piece["content"])
        return json.dumps(content_piece, ensure_ascii=False)

    return str(content_piece)


def _stringify_argument(argument: Any) -> str:
    """Normalise streamed tool/function arguments into string form."""

    if argument is None:
        return ""

    if isinstance(argument, str):
        return argument

    if isinstance(argument, (bytes, bytearray)):
        return argument.decode()

    try:
        return json.dumps(argument, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(argument)


def assemble_streamed_response(chunks: Sequence[Any]) -> SimpleNamespace | None:
    """Assemble a streamed sequence of chunks into a ChatCompletion-like object."""

    if not chunks:
        return None

    aggregated_choices: dict[int, dict[str, Any]] = {}
    usage = None
    top_level: dict[str, Any] = {}

    for chunk in chunks:
        if chunk is None:
            continue

        for attr_name in ("id", "created", "model", "object"):
            if attr_name not in top_level:
                value = getattr(chunk, attr_name, None)
                if value is not None:
                    top_level[attr_name] = value

        if usage is None:
            usage_candidate = getattr(chunk, "usage", None)
            if usage_candidate is not None:
                usage = usage_candidate

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        for choice in choices:
            index = getattr(choice, "index", 0)
            choice_entry = aggregated_choices.setdefault(
                index,
                {
                    "role": None,
                    "content": [],
                    "tool_calls": {},
                    "function_call": {"name": None, "arguments": []},
                    "finish_reason": None,
                },
            )

            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason:
                choice_entry["finish_reason"] = finish_reason

            delta = getattr(choice, "delta", None)
            role = _get_attribute(delta, "role")
            if role:
                choice_entry["role"] = role

            content_piece = _get_attribute(delta, "content")
            if content_piece is not None:
                flattened_content = _flatten_content_piece(content_piece)
                if flattened_content:
                    choice_entry["content"].append(flattened_content)

            tool_calls = _get_attribute(delta, "tool_calls", []) or []
            for tool_delta in tool_calls:
                tool_index = _get_attribute(tool_delta, "index", 0)
                tool_entry = choice_entry["tool_calls"].setdefault(
                    tool_index,
                    {
                        "id": None,
                        "type": _get_attribute(tool_delta, "type"),
                        "function": {"name": None, "arguments": []},
                    },
                )

                tool_id = _get_attribute(tool_delta, "id")
                if tool_id:
                    tool_entry["id"] = tool_id

                tool_type = _get_attribute(tool_delta, "type")
                if tool_type:
                    tool_entry["type"] = tool_type

                function_delta = _get_attribute(tool_delta, "function", {})
                func_name = _get_attribute(function_delta, "name")
                if func_name:
                    tool_entry["function"]["name"] = func_name

                func_args = _get_attribute(function_delta, "arguments")
                if func_args is not None:
                    tool_entry["function"]["arguments"].append(
                        _stringify_argument(func_args)
                    )

            function_call_delta = _get_attribute(delta, "function_call")
            if function_call_delta:
                func_name = _get_attribute(function_call_delta, "name")
                if func_name:
                    choice_entry["function_call"]["name"] = func_name

                func_args = _get_attribute(function_call_delta, "arguments")
                if func_args is not None:
                    choice_entry["function_call"]["arguments"].append(
                        _stringify_argument(func_args)
                    )

    if not aggregated_choices:
        return None

    final_choices = []
    for index in sorted(aggregated_choices):
        entry = aggregated_choices[index]
        content = "".join(str(part) for part in entry["content"])

        tool_calls_list = []
        for tool_index in sorted(entry["tool_calls"]):
            tool_entry = entry["tool_calls"][tool_index]
            arguments = "".join(
                str(argument) for argument in tool_entry["function"]["arguments"]
            )
            tool_calls_list.append(
                SimpleNamespace(
                    id=tool_entry["id"],
                    type=tool_entry["type"],
                    function=SimpleNamespace(
                        name=tool_entry["function"]["name"],
                        arguments=arguments,
                    ),
                )
            )

        function_call_entry = entry["function_call"]
        function_call_value = None
        if function_call_entry["name"] or function_call_entry["arguments"]:
            function_call_value = {
                "name": function_call_entry["name"],
                "arguments": "".join(
                    str(argument) for argument in function_call_entry["arguments"]
                ),
            }

        message = SimpleNamespace(
            role=entry["role"] or "assistant",
            content=content if content else None,
            tool_calls=tool_calls_list or None,
            function_call=function_call_value,
        )

        final_choices.append(
            SimpleNamespace(
                index=index,
                message=message,
                finish_reason=entry["finish_reason"],
            )
        )

    usage_value = usage
    if isinstance(usage_value, dict):
        usage_value = SimpleNamespace(**usage_value)

    response_data = {"choices": final_choices, "usage": usage_value}
    response_data.update(top_level)

    return SimpleNamespace(**response_data)


def record_conversation(memoria, kwargs: dict[str, Any], response) -> None:
    """Record OpenAI conversation with enhanced content parsing."""
    try:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")

        # Extract user input with enhanced parsing
        user_input = _extract_openai_user_input(messages)

        # Extract AI response with enhanced parsing
        ai_output = _extract_openai_ai_output(response)

        # Calculate tokens
        tokens_used = 0
        if hasattr(response, "usage") and response.usage:
            tokens_used = getattr(response.usage, "total_tokens", 0)

        # Enhanced metadata extraction
        metadata = _extract_openai_metadata(kwargs, response, tokens_used)

        # Record conversation
        memoria.record_conversation(
            user_input=user_input,
            ai_output=ai_output,
            model=model,
            metadata=metadata,
        )

        # Also record AI response in conversation manager for history tracking
        if ai_output:
            memoria.conversation_manager.record_response(
                session_id=memoria._session_id,
                response=ai_output,
                metadata={"model": model, "tokens_used": tokens_used},
            )
    except Exception as e:
        logger.error(f"Failed to record OpenAI conversation: {e}")
