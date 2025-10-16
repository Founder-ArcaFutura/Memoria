"""Utilities for recording Anthropic conversations."""

import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from loguru import logger


def _extract_anthropic_user_input(messages: list[dict]) -> str:
    """Extract user input from Anthropic messages with support for complex content types."""
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
                    text_parts: list[str] = []
                    image_count = 0

                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "image":
                                image_count += 1

                    user_input = " ".join(text_parts)
                    # Add image indicator if present
                    if image_count > 0:
                        user_input += f" [Contains {image_count} image(s)]"

                break
    except Exception as e:
        logger.debug(f"Error extracting Anthropic user input: {e}")
        user_input = "[Error extracting user input]"

    return user_input


def _extract_anthropic_ai_output(response) -> str:
    """Extract AI output from Anthropic response with support for various response types."""
    ai_output = ""
    try:
        if hasattr(response, "content") and response.content:
            if isinstance(response.content, list):
                # Handle structured content (text blocks, tool use, etc.)
                text_parts: list[str] = []
                tool_uses: list[str] = []

                for block in response.content:
                    try:
                        # Handle text blocks
                        if hasattr(block, "text") and block.text:
                            text_parts.append(block.text)
                        # Handle tool use blocks
                        elif hasattr(block, "type"):
                            block_type = getattr(block, "type", None)
                            if block_type == "tool_use":
                                tool_name = getattr(block, "name", "unknown")
                                tool_input = getattr(block, "input", {})
                                tool_uses.append(f"Used {tool_name} with {tool_input}")
                        # Handle mock objects for testing
                        elif hasattr(block, "name") and hasattr(block, "input"):
                            tool_name = getattr(block, "name", "unknown")
                            tool_input = getattr(block, "input", {})
                            tool_uses.append(f"Used {tool_name} with {tool_input}")
                    except Exception as block_error:
                        logger.debug(f"Error processing block: {block_error}")
                        continue

                ai_output = " ".join(text_parts)
                if tool_uses:
                    if ai_output:
                        ai_output += " "
                    ai_output += "[Tool uses: " + "; ".join(tool_uses) + "]"

            elif isinstance(response.content, str):
                ai_output = response.content
            else:
                ai_output = str(response.content)

    except Exception as e:
        logger.debug(f"Error extracting Anthropic AI output: {e}")
        ai_output = "[Error extracting AI response]"

    return ai_output


def _extract_anthropic_tokens(response) -> int:
    """Extract token usage from Anthropic response."""
    tokens_used = 0
    try:
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
            tokens_used = input_tokens + output_tokens
    except Exception as e:
        logger.debug(f"Error extracting Anthropic tokens: {e}")

    return tokens_used


def _extract_anthropic_metadata(
    kwargs: dict[str, Any], response, tokens_used: int
) -> dict[str, Any]:
    """Extract comprehensive metadata from Anthropic request and response."""
    metadata: dict[str, Any] = {
        "integration": "anthropic_auto",
        "api_type": "messages",
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

        # Add response metadata
        if hasattr(response, "stop_reason"):
            metadata["stop_reason"] = response.stop_reason
        if hasattr(response, "model"):
            metadata["response_model"] = response.model

        # Add detailed token usage if available
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            metadata.update(
                {
                    "input_tokens": getattr(usage, "input_tokens", 0),
                    "output_tokens": getattr(usage, "output_tokens", 0),
                    "total_tokens": tokens_used,
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
                        if isinstance(item, dict) and item.get("type") == "image":
                            has_images = True
                            break
                if has_images:
                    break

        metadata["message_count"] = message_count
        metadata["has_images"] = has_images

    except Exception as e:
        logger.debug(f"Error extracting Anthropic metadata: {e}")

    return metadata


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def _to_dict(obj: Any) -> dict:
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    try:
        return dict(obj)
    except Exception:  # pragma: no cover - defensive
        pass

    try:
        return vars(obj)
    except Exception:  # pragma: no cover - defensive
        return {}


def _coerce_namespace(value: Any) -> Any:
    if value is None or isinstance(value, SimpleNamespace):
        return value

    if isinstance(value, dict):
        return SimpleNamespace(**value)

    return value


def _parse_partial_json(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:  # pragma: no cover - defensive
            return raw

    return raw


def assemble_streamed_response(chunks: Sequence[Any]) -> SimpleNamespace | None:
    """Assemble streamed Anthropic chunks into a response-like object."""

    if not chunks:
        return None

    block_order: list[int] = []
    blocks: dict[int, dict[str, Any]] = {}
    partial_tool_inputs: dict[int, str] = {}

    message_id = None
    model = None
    stop_reason = None
    role = None
    usage: Any = None
    top_type = None

    for chunk in chunks:
        if chunk is None:
            continue

        chunk_type = _get_attr(chunk, "type", None)
        model = _get_attr(chunk, "model", model)

        if chunk_type == "message_start":
            message = _get_attr(chunk, "message", {})
            message_dict = _to_dict(message)
            message_id = message_dict.get("id", message_id)
            model = message_dict.get("model", model)
            role = message_dict.get("role", role)
            top_type = message_dict.get("type", top_type)
            stop_reason = message_dict.get("stop_reason", stop_reason)
            if usage is None:
                usage = message_dict.get("usage")
        elif chunk_type == "message_delta":
            delta = _to_dict(_get_attr(chunk, "delta", {}))
            if delta.get("stop_reason") is not None:
                stop_reason = delta.get("stop_reason")
            if usage is None:
                usage = _get_attr(chunk, "usage", usage)
        elif chunk_type == "message_stop":
            stop_reason = _get_attr(chunk, "stop_reason", stop_reason)
        elif chunk_type == "content_block_start":
            index = _get_attr(chunk, "index", None)
            if index is None:
                continue

            block = _to_dict(_get_attr(chunk, "content_block", {}))
            block_type = block.get("type")
            if block_type is None:
                continue

            if index not in block_order:
                block_order.append(index)

            if block_type == "text":
                blocks[index] = {"type": "text", "text": block.get("text", "")}
            elif block_type == "tool_use":
                input_value = block.get("input") or {}
                if not input_value:
                    partial_tool_inputs[index] = ""
                blocks[index] = {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": input_value,
                }
            else:
                blocks[index] = block.copy()
        elif chunk_type == "content_block_delta":
            index = _get_attr(chunk, "index", None)
            if index is None or index not in blocks:
                continue

            delta = _to_dict(_get_attr(chunk, "delta", {}))
            block = blocks[index]

            if block.get("type") == "text":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif block.get("type") == "tool_use":
                if "partial_json" in delta and delta["partial_json"] is not None:
                    partial_tool_inputs[index] = partial_tool_inputs.get(
                        index, ""
                    ) + str(delta["partial_json"])
                if "input" in delta and isinstance(delta["input"], dict):
                    existing_input = block.setdefault("input", {})
                    if isinstance(existing_input, dict):
                        existing_input.update(delta["input"])
            else:
                if "text" in delta:
                    block["text"] = block.get("text", "") + delta.get("text", "")
        elif chunk_type == "content_block_stop":
            index = _get_attr(chunk, "index", None)
            if index in partial_tool_inputs:
                parsed = _parse_partial_json(partial_tool_inputs.pop(index))
                block = blocks.get(index)
                if block is not None:
                    block["input"] = parsed
        else:
            if usage is None:
                usage_candidate = _get_attr(chunk, "usage", None)
                if usage_candidate is not None:
                    usage = usage_candidate

    for index, raw in list(partial_tool_inputs.items()):
        block = blocks.get(index)
        if block is not None:
            block["input"] = _parse_partial_json(raw)
            partial_tool_inputs.pop(index, None)

    content_blocks: list[SimpleNamespace] = []
    for index in block_order:
        block = blocks.get(index)
        if block is None:
            continue

        if block.get("type") == "tool_use" and isinstance(block.get("input"), str):
            block["input"] = _parse_partial_json(block["input"])

        content_blocks.append(SimpleNamespace(**block))

    response_data = {
        "id": message_id,
        "model": model,
        "role": role,
        "type": top_type,
        "stop_reason": stop_reason,
        "usage": _coerce_namespace(usage),
        "content": content_blocks,
    }

    return SimpleNamespace(**response_data)


def record_conversation(memoria, kwargs: dict[str, Any], response) -> None:
    """Record Anthropic conversation with enhanced content parsing."""
    try:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "claude-unknown")

        # Extract user input with enhanced parsing
        user_input = _extract_anthropic_user_input(messages)

        # Extract AI response with enhanced parsing
        ai_output = _extract_anthropic_ai_output(response)

        # Calculate tokens
        tokens_used = _extract_anthropic_tokens(response)

        # Enhanced metadata extraction
        metadata = _extract_anthropic_metadata(kwargs, response, tokens_used)

        # Record conversation
        memoria.record_conversation(
            user_input=user_input,
            ai_output=ai_output,
            model=model,
            metadata=metadata,
        )
    except Exception as e:
        logger.error(f"Failed to record Anthropic conversation: {e}")
