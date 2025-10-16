"""
LiteLLM Integration - Native Callback System

This module handles LiteLLM native callback registration for automatic
memory recording. It uses LiteLLM's official callback mechanism instead
of monkey-patching.

Usage:
    from memoria import Memoria
    from memoria.config import ConfigManager
    import os

    memoria = Memoria(...)
    memoria.enable()  # Automatically registers LiteLLM callbacks

    # Load model name from config or environment (LITELLM_MODEL)
    config = ConfigManager()
    model_name = config.get_setting("agents.default_model", os.getenv("LITELLM_MODEL"))

    # Now use LiteLLM normally - conversations are auto-recorded
    from litellm import completion
    response = completion(model=model_name or "gpt-4o-mini", messages=[...])

Note:
    Configure the model via `agents.default_model` in your memoria.json or set the
    LITELLM_MODEL environment variable.
"""

import os
import time
from collections.abc import Callable
from functools import wraps

from loguru import logger

from memoria.config import ConfigManager

try:
    import litellm

    LITELLM_AVAILABLE = True

    # Check for modifying input callbacks (for context injection)
    HAS_MODIFYING_ROUTER = hasattr(litellm, "Router") and hasattr(
        litellm.Router, "pre_call_hook"
    )

except ImportError:
    LITELLM_AVAILABLE = False
    HAS_MODIFYING_ROUTER = False
    logger.warning("LiteLLM not available - native callback system disabled")


def get_default_litellm_model() -> str | None:
    """Return the configured default LiteLLM model name."""

    return ConfigManager().get_setting(
        "agents.default_model", os.getenv("LITELLM_MODEL")
    )


class LiteLLMCallbackManager:
    """
    Manages LiteLLM native callback registration and integration with Memoria.

    This class provides a clean interface for registering and managing
    LiteLLM callbacks that automatically record conversations into Memoria.
    """

    def __init__(self, memoria_instance):
        """
        Initialize LiteLLM callback manager.

        Args:
            memoria_instance: The Memoria instance to record conversations to
        """
        self.memoria_instance = memoria_instance
        self._callback_registered = False
        self._original_callbacks = None
        self._original_completion = None  # For context injection
        self._original_async_completions: dict[str, Callable | None] = {}
        self._original_stream_functions: dict[str, Callable | None] = {}

    def register_callbacks(self) -> bool:
        """
        Register LiteLLM native callbacks for automatic memory recording.

        Returns:
            True if registration successful, False otherwise
        """
        if not LITELLM_AVAILABLE:
            logger.error("LiteLLM not available - cannot register callbacks")
            return False

        if self._callback_registered:
            logger.warning("LiteLLM callbacks already registered")
            return True

        try:
            # Store original callbacks for restoration
            self._original_callbacks = getattr(litellm, "success_callback", [])

            # Register our success callback
            if not hasattr(litellm, "success_callback"):
                litellm.success_callback = []
            elif not isinstance(litellm.success_callback, list):
                litellm.success_callback = [litellm.success_callback]

            # Add our callback function
            litellm.success_callback.append(self._litellm_success_callback)

            # For context injection, we need to monkey-patch the completion function
            # This is the only reliable way to inject context before requests in LiteLLM
            if (
                self.memoria_instance.conscious_ingest
                or self.memoria_instance.auto_ingest
            ):
                self._setup_context_injection()

            self._callback_registered = True
            logger.info("LiteLLM native callbacks registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register LiteLLM callbacks: {e}")
            return False

    def unregister_callbacks(self) -> bool:
        """
        Unregister LiteLLM callbacks and restore original state.

        Returns:
            True if unregistration successful, False otherwise
        """
        if not LITELLM_AVAILABLE:
            return False

        if not self._callback_registered:
            logger.warning("LiteLLM callbacks not registered")
            return True

        try:
            # Remove our callback
            if hasattr(litellm, "success_callback") and isinstance(
                litellm.success_callback, list
            ):
                # Remove all instances of our callback
                litellm.success_callback = [
                    cb
                    for cb in litellm.success_callback
                    if cb != self._litellm_success_callback
                ]

                # If no callbacks left, restore original state
                if not litellm.success_callback:
                    if self._original_callbacks:
                        litellm.success_callback = self._original_callbacks
                    else:
                        delattr(litellm, "success_callback")

            # Restore original completion function if we modified it
            if self._original_completion is not None:
                litellm.completion = self._original_completion
                self._original_completion = None

            # Restore any async completion entry points we wrapped
            if self._original_async_completions:
                for attr, original in self._original_async_completions.items():
                    if original is not None:
                        setattr(litellm, attr, original)
                self._original_async_completions.clear()

            if self._original_stream_functions:
                for attr, original in self._original_stream_functions.items():
                    if original is not None:
                        setattr(litellm, attr, original)
                self._original_stream_functions.clear()

            self._callback_registered = False
            logger.info("LiteLLM native callbacks unregistered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister LiteLLM callbacks: {e}")
            return False

    def _litellm_success_callback(self, kwargs, response, start_time, end_time):
        """
        LiteLLM success callback that records conversations in Memoria.

        This function is automatically called by LiteLLM after successful completions.

        Args:
            kwargs: Original request parameters
            response: LiteLLM response object
            start_time: Request start time
            end_time: Request end time
        """
        try:
            if not self.memoria_instance or not self.memoria_instance.is_enabled:
                return

            # Handle context injection for conscious_ingest and auto_ingest
            if (
                self.memoria_instance.conscious_ingest
                or self.memoria_instance.auto_ingest
            ):
                # Note: Context injection happens BEFORE the request in LiteLLM
                # This callback is for recording AFTER the response
                pass

            # Extract user input
            user_input = ""
            messages = kwargs.get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_input = msg.get("content", "")
                    break

            # Extract AI output
            ai_output = ""
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    ai_output = choice.message.content or ""

            # Extract model
            model = kwargs.get("model", "litellm-unknown")

            # Calculate timing
            duration_ms = 0
            if start_time is not None and end_time is not None:
                try:
                    if isinstance(start_time, (int, float)) and isinstance(
                        end_time, (int, float)
                    ):
                        duration_ms = (end_time - start_time) * 1000
                except Exception:
                    pass

            # Extract token usage
            tokens_used = 0
            if hasattr(response, "usage") and response.usage:
                tokens_used = getattr(response.usage, "total_tokens", 0)

            # Prepare metadata
            metadata = {
                "integration": "litellm_native",
                "api_type": "completion",
                "tokens_used": tokens_used,
                "auto_recorded": True,
                "duration_ms": duration_ms,
            }

            # Add token details if available
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                metadata.update(
                    {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage, "completion_tokens", 0),
                    }
                )

            # Record the conversation
            if user_input and ai_output:
                self.memoria_instance.record_conversation(
                    user_input=user_input,
                    ai_output=ai_output,
                    model=model,
                    metadata=metadata,
                )
                logger.debug(
                    f"LiteLLM callback: Recorded conversation for model {model}"
                )

        except Exception as e:
            logger.error(f"LiteLLM callback failed: {e}")

    def _setup_context_injection(self):
        """Set up context injection by wrapping LiteLLM's completion function."""
        try:
            if self._original_completion is not None:
                # Already set up
                return

            # Store original completion function
            self._original_completion = litellm.completion

            @wraps(self._original_completion)
            def completion_with_context(*args, **kwargs):
                # Inject context if needed
                kwargs = self._inject_context(kwargs)
                # Call original completion function
                return self._original_completion(*args, **kwargs)

            # Replace LiteLLM's completion function
            litellm.completion = completion_with_context

            # Wrap async completion entry points if available
            async_completion_attrs = [
                "acompletion",
                "acompletion_with_retries",
            ]

            for attr in async_completion_attrs:
                if attr in self._original_async_completions:
                    continue

                original_async = getattr(litellm, attr, None)
                if original_async is None:
                    continue

                self._original_async_completions[attr] = original_async

                @wraps(original_async)
                async def async_wrapper(*args, _original=original_async, **kwargs):
                    kwargs = self._inject_context(kwargs)
                    return await _original(*args, **kwargs)

                setattr(litellm, attr, async_wrapper)

            logger.debug("Context injection wrapper set up for LiteLLM")

            streaming_attrs: list[tuple[str, bool]] = [
                ("stream_completion", False),
                ("stream_chat_completion", False),
                ("astream_completion", True),
                ("acompletion_stream", True),
            ]

            for attr, is_async in streaming_attrs:
                if attr in self._original_stream_functions:
                    continue

                original_stream = getattr(litellm, attr, None)
                if original_stream is None:
                    continue

                self._original_stream_functions[attr] = original_stream

                if is_async:

                    @wraps(original_stream)
                    def async_stream_wrapper(
                        *args, _attr=attr, _original=original_stream, **kwargs
                    ):
                        kwargs = self._inject_context(kwargs)
                        start_time = time.time()
                        stream = _original(*args, **kwargs)
                        return self._wrap_async_stream(
                            stream, kwargs, _attr, start_time
                        )

                    setattr(litellm, attr, async_stream_wrapper)
                else:

                    @wraps(original_stream)
                    def stream_wrapper(
                        *args, _attr=attr, _original=original_stream, **kwargs
                    ):
                        kwargs = self._inject_context(kwargs)
                        start_time = time.time()
                        stream = _original(*args, **kwargs)
                        return self._wrap_stream(stream, kwargs, _attr, start_time)

                    setattr(litellm, attr, stream_wrapper)

        except Exception as e:
            logger.error(f"Failed to set up context injection: {e}")

    def _inject_context(self, kwargs):
        """Inject memory context into LiteLLM request kwargs."""
        try:
            if not self.memoria_instance:
                return kwargs

            # Use the existing context injection methods from the Memoria instance
            if (
                self.memoria_instance.conscious_ingest
                or self.memoria_instance.auto_ingest
            ):
                logger.debug("LiteLLM: Starting context injection")

                # Determine mode
                if self.memoria_instance.conscious_ingest:
                    mode = "conscious"
                elif self.memoria_instance.auto_ingest:
                    mode = "auto"
                else:
                    mode = "auto"  # fallback

                # Extract user input first to debug what we're working with
                messages = kwargs.get("messages", [])
                user_input = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_input = msg.get("content", "")
                        break

                logger.debug(
                    f"LiteLLM: Injecting context in {mode} mode for input: {user_input[:100]}..."
                )

                # Use the existing _inject_litellm_context method
                kwargs = self.memoria_instance._inject_litellm_context(
                    kwargs, mode=mode
                )

                # Verify injection worked
                updated_messages = kwargs.get("messages", [])
                if len(updated_messages) > len(messages):
                    logger.debug(
                        f"LiteLLM: Context injection successful, message count increased from {len(messages)} to {len(updated_messages)}"
                    )
                else:
                    logger.debug(
                        "LiteLLM: Context injection completed, no new messages added (may be intended)"
                    )

        except Exception as e:
            logger.error(f"Context injection failed in LiteLLM wrapper: {e}")
            import traceback

            logger.debug(f"LiteLLM injection stack trace: {traceback.format_exc()}")

        return kwargs

    @property
    def is_registered(self) -> bool:
        """Check if callbacks are registered."""
        return self._callback_registered

    def _wrap_stream(self, stream, kwargs, api_type: str, start_time: float):
        """Wrap a synchronous LiteLLM stream to accumulate deltas."""

        if stream is None or not hasattr(stream, "__iter__"):
            return stream

        def generator():
            aggregated_parts: list[str] = []
            usage_info: dict[str, int] | None = None
            last_chunk = None
            try:
                for chunk in stream:
                    text = self._extract_stream_text(chunk)
                    if text:
                        aggregated_parts.append(text)

                    usage = self._extract_usage_tokens(chunk)
                    if usage:
                        usage_info = usage

                    last_chunk = chunk
                    yield chunk
            finally:
                end_time = time.time()
                self._finalize_stream_recording(
                    kwargs,
                    aggregated_parts,
                    usage_info,
                    start_time,
                    end_time,
                    last_chunk,
                    api_type,
                )

        return generator()

    def _wrap_async_stream(self, stream, kwargs, api_type: str, start_time: float):
        """Wrap an async LiteLLM stream to accumulate deltas."""

        if stream is None or not hasattr(stream, "__aiter__"):
            return stream

        async def async_generator():
            aggregated_parts: list[str] = []
            usage_info: dict[str, int] | None = None
            last_chunk = None
            try:
                async for chunk in stream:
                    text = self._extract_stream_text(chunk)
                    if text:
                        aggregated_parts.append(text)

                    usage = self._extract_usage_tokens(chunk)
                    if usage:
                        usage_info = usage

                    last_chunk = chunk
                    yield chunk
            finally:
                end_time = time.time()
                self._finalize_stream_recording(
                    kwargs,
                    aggregated_parts,
                    usage_info,
                    start_time,
                    end_time,
                    last_chunk,
                    api_type,
                )

        return async_generator()

    def _extract_stream_text(self, chunk) -> str:
        """Best-effort extraction of text content from a stream chunk."""

        if chunk is None:
            return ""

        choices = None
        if isinstance(chunk, dict):
            choices = chunk.get("choices")
        else:
            choices = getattr(chunk, "choices", None)

        if choices:
            first_choice = choices[0]
            text = self._extract_choice_text(first_choice)
            if text:
                return text

        if isinstance(chunk, dict):
            for key in ("content", "text"):
                value = chunk.get(key)
                if isinstance(value, str):
                    return value
        else:
            for attr in ("content", "text"):
                value = getattr(chunk, attr, None)
                if isinstance(value, str):
                    return value

        return ""

    def _extract_choice_text(self, choice) -> str:
        """Extract text from a LiteLLM choice entry."""

        if choice is None:
            return ""

        if isinstance(choice, dict):
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    return content

            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content

            text = choice.get("text")
            if isinstance(text, str):
                return text

            return ""

        delta = getattr(choice, "delta", None)
        if delta is not None:
            content = getattr(delta, "content", None)
            if isinstance(content, str):
                return content

        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content

        text = getattr(choice, "text", None)
        if isinstance(text, str):
            return text

        return ""

    def _extract_usage_tokens(self, chunk) -> dict[str, int] | None:
        """Extract token usage information from a stream chunk."""

        usage = None
        if isinstance(chunk, dict):
            usage = chunk.get("usage")
        else:
            usage = getattr(chunk, "usage", None)

        if not usage:
            return None

        values: dict[str, int] = {}

        if isinstance(usage, dict):
            items = usage.items()
        else:
            items = (
                ("prompt_tokens", getattr(usage, "prompt_tokens", None)),
                ("completion_tokens", getattr(usage, "completion_tokens", None)),
                ("total_tokens", getattr(usage, "total_tokens", None)),
            )

        for key, value in items:
            if value is None:
                continue
            try:
                values[key] = int(value)
            except (TypeError, ValueError):
                logger.debug(
                    f"LiteLLM stream usage value for {key} not numeric: {value}"
                )

        return values or None

    def _finalize_stream_recording(
        self,
        kwargs: dict,
        aggregated_parts: list[str],
        usage_info: dict[str, int] | None,
        start_time: float,
        end_time: float,
        last_chunk,
        api_type: str,
    ) -> None:
        """Record the accumulated streaming output once the stream finishes."""

        try:
            if not self.memoria_instance or not self.memoria_instance.is_enabled:
                return

            messages = kwargs.get("messages", [])
            user_input = ""
            for message in reversed(messages):
                if message.get("role") == "user":
                    user_input = message.get("content", "")
                    break

            ai_output = "".join(aggregated_parts)
            if not ai_output and last_chunk is not None:
                ai_output = self._extract_stream_text(last_chunk)

            if not user_input or not ai_output:
                return

            model = kwargs.get("model", "litellm-unknown")

            duration_ms = 0.0
            if start_time and end_time and end_time >= start_time:
                duration_ms = (end_time - start_time) * 1000.0

            metadata = {
                "integration": "litellm_native",
                "api_type": api_type,
                "auto_recorded": True,
                "duration_ms": duration_ms,
                "tokens_used": 0,
                "stream": True,
            }

            if usage_info:
                metadata["tokens_used"] = usage_info.get("total_tokens", 0)
                metadata.update(
                    {
                        "prompt_tokens": usage_info.get("prompt_tokens", 0),
                        "completion_tokens": usage_info.get("completion_tokens", 0),
                    }
                )

            self.memoria_instance.record_conversation(
                user_input=user_input,
                ai_output=ai_output,
                model=model,
                metadata=metadata,
            )

            logger.debug(
                "LiteLLM streaming callback recorded conversation for %s", api_type
            )

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to finalize LiteLLM stream recording: {exc}")


def setup_litellm_callbacks(memoria_instance) -> LiteLLMCallbackManager | None:
    """
    Convenience function to set up LiteLLM callbacks for a Memoria instance.

    Args:
        memoria_instance: The Memoria instance to record conversations to

    Returns:
        LiteLLMCallbackManager instance if successful, None otherwise
    """
    if not LITELLM_AVAILABLE:
        logger.error("LiteLLM not available - cannot set up callbacks")
        return None

    callback_manager = LiteLLMCallbackManager(memoria_instance)
    if callback_manager.register_callbacks():
        return callback_manager
    return None
