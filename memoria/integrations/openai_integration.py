"""
OpenAI Integration - Automatic Interception System

This module provides automatic interception of OpenAI API calls when Memoria is enabled.
Users can import and use the standard OpenAI client normally, and Memoria will automatically
record conversations when enabled.

Usage:
    from openai import OpenAI
    from memoria import Memoria
    from memoria.config import ConfigManager
    import os

    # Initialize Memoria and enable it
    openai_memory = Memoria(
        database_connect="sqlite:///openai_memory.db",
        conscious_ingest=True,
        verbose=True,
    )
    openai_memory.enable()

    # Pull model name from config or environment (OPENAI_MODEL)
    config = ConfigManager()
    model_name = config.get_setting("agents.default_model", os.getenv("OPENAI_MODEL"))

    # Use standard OpenAI client - automatically intercepted!
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name or "gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    # Conversation is automatically recorded to Memoria

Note:
    Configure the model via `agents.default_model` in your memoria.json or set the
    OPENAI_MODEL environment variable.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Sequence
from typing import Any

from loguru import logger

from memoria.config import ConfigManager
from memoria.providers import openai_recorder

# Global registry of enabled Memoria instances
_enabled_memoria_instances = []


def get_default_openai_model() -> str | None:
    """Return the configured default OpenAI model name."""

    return ConfigManager().get_setting(
        "agents.default_model", os.getenv("OPENAI_MODEL")
    )


class OpenAIInterceptor:
    """
    Automatic OpenAI interception system that patches the OpenAI module
    to automatically record conversations when Memoria instances are enabled.
    """

    _original_methods = {}
    _is_patched = False

    @classmethod
    def patch_openai(cls):
        """Patch OpenAI module to intercept API calls."""
        if cls._is_patched:
            return

        try:
            import openai

            # Patch sync OpenAI client
            if hasattr(openai, "OpenAI"):
                cls._patch_client_class(openai.OpenAI, "sync")

            # Patch async OpenAI client
            if hasattr(openai, "AsyncOpenAI"):
                cls._patch_async_client_class(openai.AsyncOpenAI, "async")

            # Patch Azure clients if available
            if hasattr(openai, "AzureOpenAI"):
                cls._patch_client_class(openai.AzureOpenAI, "azure_sync")

            if hasattr(openai, "AsyncAzureOpenAI"):
                cls._patch_async_client_class(openai.AsyncAzureOpenAI, "azure_async")

            cls._is_patched = True
            logger.debug("OpenAI module patched for automatic interception")

        except ImportError:
            logger.warning("OpenAI not available - skipping patch")
        except Exception as e:
            logger.error(f"Failed to patch OpenAI module: {e}")

    @classmethod
    def _patch_client_class(cls, client_class, client_type):
        """Patch a sync OpenAI client class."""
        # Store the original unbound method
        original_key = f"{client_type}_process_response"
        if original_key not in cls._original_methods:
            cls._original_methods[original_key] = client_class._process_response

        original_prepare_key = f"{client_type}_prepare_options"
        if original_prepare_key not in cls._original_methods and hasattr(
            client_class, "_prepare_options"
        ):
            cls._original_methods[original_prepare_key] = client_class._prepare_options

        # Get reference to original method to avoid recursion
        original_process = cls._original_methods[original_key]

        def patched_process_response(
            self, *, cast_to, options, response, stream, stream_cls, **kwargs
        ):
            # Call original method first with all kwargs
            result = original_process(
                self,
                cast_to=cast_to,
                options=options,
                response=response,
                stream=stream,
                stream_cls=stream_cls,
                **kwargs,
            )

            # Record conversation for enabled Memoria instances
            if stream:
                return cls._wrap_stream_result(result, options, client_type)

            cls._record_conversation_for_enabled_instances(options, result, client_type)

            return result

        client_class._process_response = patched_process_response

        # Patch prepare_options if it exists
        if original_prepare_key in cls._original_methods:
            original_prepare = cls._original_methods[original_prepare_key]

            def patched_prepare_options(self, options):
                # Call original method first
                options = original_prepare(self, options)

                # Inject context for enabled Memoria instances
                options = cls._inject_context_for_enabled_instances(
                    options, client_type
                )

                return options

            client_class._prepare_options = patched_prepare_options

    @classmethod
    def _patch_async_client_class(cls, client_class, client_type):
        """Patch an async OpenAI client class."""
        # Store the original unbound method
        original_key = f"{client_type}_process_response"
        if original_key not in cls._original_methods:
            cls._original_methods[original_key] = client_class._process_response

        original_prepare_key = f"{client_type}_prepare_options"
        if original_prepare_key not in cls._original_methods and hasattr(
            client_class, "_prepare_options"
        ):
            cls._original_methods[original_prepare_key] = client_class._prepare_options

        # Get reference to original method to avoid recursion
        original_process = cls._original_methods[original_key]

        async def patched_async_process_response(
            self, *, cast_to, options, response, stream, stream_cls, **kwargs
        ):
            # Call original method first with all kwargs
            result = await original_process(
                self,
                cast_to=cast_to,
                options=options,
                response=response,
                stream=stream,
                stream_cls=stream_cls,
                **kwargs,
            )

            # Record conversation for enabled Memoria instances
            if stream:
                return cls._wrap_async_stream_result(result, options, client_type)

            cls._record_conversation_for_enabled_instances(options, result, client_type)

            return result

        client_class._process_response = patched_async_process_response

        # Patch prepare_options if it exists
        if original_prepare_key in cls._original_methods:
            original_prepare = cls._original_methods[original_prepare_key]

            def patched_async_prepare_options(self, options):
                # Call original method first
                options = original_prepare(self, options)

                # Inject context for enabled Memoria instances
                options = cls._inject_context_for_enabled_instances(
                    options, client_type
                )

                return options

            client_class._prepare_options = patched_async_prepare_options

    @classmethod
    def _inject_context_for_enabled_instances(cls, options, client_type):
        """Inject context for all enabled Memoria instances with conscious/auto ingest."""
        for memoria_instance in _enabled_memoria_instances:
            if memoria_instance.is_enabled and (
                memoria_instance.conscious_ingest or memoria_instance.auto_ingest
            ):
                try:
                    # Get json_data from options - handle multiple attribute name possibilities
                    json_data = None
                    for attr_name in ["json_data", "_json_data", "data"]:
                        if hasattr(options, attr_name):
                            json_data = getattr(options, attr_name, None)
                            if json_data:
                                break

                    if not json_data:
                        # Try to reconstruct from other options attributes
                        json_data = {}
                        if hasattr(options, "messages"):
                            json_data["messages"] = options.messages
                        elif hasattr(options, "_messages"):
                            json_data["messages"] = options._messages

                    if json_data and "messages" in json_data:
                        # This is a chat completion request - inject context
                        logger.debug(
                            f"OpenAI: Injecting context for {client_type} with {len(json_data['messages'])} messages"
                        )
                        updated_data = memoria_instance._inject_openai_context(
                            {"messages": json_data["messages"]}
                        )

                        if updated_data.get("messages"):
                            # Update the options with modified messages
                            if hasattr(options, "json_data") and options.json_data:
                                options.json_data["messages"] = updated_data["messages"]
                            elif hasattr(options, "messages"):
                                options.messages = updated_data["messages"]

                            logger.debug(
                                f"OpenAI: Successfully injected context for {client_type}"
                            )
                    else:
                        logger.debug(
                            f"OpenAI: No messages found in options for {client_type}, skipping context injection"
                        )

                except Exception as e:
                    logger.error(f"Context injection failed for {client_type}: {e}")

        return options

    @classmethod
    def _is_internal_agent_call(cls, json_data):
        """Check if this is an internal agent processing call that should not be recorded."""
        try:
            messages = json_data.get("messages", [])
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    # Check for internal agent processing patterns
                    internal_patterns = [
                        "Process this conversation for enhanced memory storage:",
                        "User query:",
                        "Enhanced memory processing:",
                        "Memory classification:",
                        "Search for relevant memories:",
                        "Analyze conversation for:",
                        "Extract entities from:",
                        "Categorize the following conversation:",
                    ]

                    for pattern in internal_patterns:
                        if pattern in content:
                            return True

            return False

        except Exception as e:
            logger.debug(f"Failed to check internal agent call: {e}")
            return False

    @classmethod
    def _record_conversation_for_enabled_instances(cls, options, response, client_type):
        """Record conversation for all enabled Memoria instances."""
        for memoria_instance in _enabled_memoria_instances:
            if memoria_instance.is_enabled:
                try:
                    json_data = getattr(options, "json_data", None) or {}

                    if "messages" in json_data:
                        # Skip internal agent processing calls
                        if cls._is_internal_agent_call(json_data):
                            continue
                        # Chat completions
                        openai_recorder.record_conversation(
                            memoria_instance, json_data, response
                        )
                    elif "prompt" in json_data:
                        # Legacy completions
                        cls._record_legacy_completion(
                            memoria_instance, json_data, response, client_type
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to record conversation for {client_type}: {e}"
                    )

    @classmethod
    def _record_legacy_completion(
        cls, memoria_instance, request_data, response, client_type
    ):
        """Record legacy completion API calls."""
        try:
            prompt = request_data.get("prompt", "")
            model = request_data.get("model", "unknown")

            # Extract AI response
            ai_output = ""
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "text"):
                    ai_output = choice.text or ""

            # Calculate tokens
            tokens_used = 0
            if hasattr(response, "usage") and response.usage:
                tokens_used = getattr(response.usage, "total_tokens", 0)

            # Record conversation
            memoria_instance.record_conversation(
                user_input=prompt,
                ai_output=ai_output,
                model=model,
                metadata={
                    "integration": "openai_auto_intercept",
                    "client_type": client_type,
                    "api_type": "completions",
                    "tokens_used": tokens_used,
                    "auto_recorded": True,
                },
            )
        except Exception as e:
            logger.error(f"Failed to record legacy completion: {e}")

    @classmethod
    def _wrap_stream_result(cls, stream_result, options, client_type):
        """Wrap a streaming iterator to buffer chunks and record after completion."""

        def finalize(chunks: Sequence[Any]):
            cls._record_stream_conversation_for_enabled_instances(
                options, chunks, client_type
            )

        return _StreamingRecorderWrapper(stream_result, finalize)

    @classmethod
    def _wrap_async_stream_result(cls, stream_result, options, client_type):
        """Wrap an async streaming iterator to buffer chunks and record after completion."""

        async def finalize(chunks: Sequence[Any]):
            cls._record_stream_conversation_for_enabled_instances(
                options, chunks, client_type
            )

        return _AsyncStreamingRecorderWrapper(stream_result, finalize)

    @classmethod
    def _record_stream_conversation_for_enabled_instances(
        cls, options, chunks: Sequence[Any], client_type
    ) -> None:
        """Record streaming conversation for all enabled Memoria instances."""

        if not chunks:
            return

        for memoria_instance in _enabled_memoria_instances:
            if not memoria_instance.is_enabled:
                continue

            try:
                json_data = getattr(options, "json_data", None) or {}

                if "messages" not in json_data:
                    continue

                if cls._is_internal_agent_call(json_data):
                    continue

                assembled_response = openai_recorder.assemble_streamed_response(chunks)

                if assembled_response is None:
                    continue

                openai_recorder.record_conversation(
                    memoria_instance, json_data, assembled_response
                )

            except Exception as e:
                logger.error(
                    f"Failed to record streaming conversation for {client_type}: {e}"
                )

    @classmethod
    def unpatch_openai(cls):
        """Restore original OpenAI module methods."""
        if not cls._is_patched:
            return

        try:
            import openai

            # Restore sync OpenAI client
            if "sync_process_response" in cls._original_methods:
                openai.OpenAI._process_response = cls._original_methods[
                    "sync_process_response"
                ]

            if "sync_prepare_options" in cls._original_methods:
                openai.OpenAI._prepare_options = cls._original_methods[
                    "sync_prepare_options"
                ]

            # Restore async OpenAI client
            if "async_process_response" in cls._original_methods:
                openai.AsyncOpenAI._process_response = cls._original_methods[
                    "async_process_response"
                ]

            if "async_prepare_options" in cls._original_methods:
                openai.AsyncOpenAI._prepare_options = cls._original_methods[
                    "async_prepare_options"
                ]

            # Restore Azure clients
            if (
                hasattr(openai, "AzureOpenAI")
                and "azure_sync_process_response" in cls._original_methods
            ):
                openai.AzureOpenAI._process_response = cls._original_methods[
                    "azure_sync_process_response"
                ]

            if (
                hasattr(openai, "AzureOpenAI")
                and "azure_sync_prepare_options" in cls._original_methods
            ):
                openai.AzureOpenAI._prepare_options = cls._original_methods[
                    "azure_sync_prepare_options"
                ]

            if (
                hasattr(openai, "AsyncAzureOpenAI")
                and "azure_async_process_response" in cls._original_methods
            ):
                openai.AsyncAzureOpenAI._process_response = cls._original_methods[
                    "azure_async_process_response"
                ]

            if (
                hasattr(openai, "AsyncAzureOpenAI")
                and "azure_async_prepare_options" in cls._original_methods
            ):
                openai.AsyncAzureOpenAI._prepare_options = cls._original_methods[
                    "azure_async_prepare_options"
                ]

            cls._is_patched = False
            cls._original_methods.clear()
            logger.debug("OpenAI module patches removed")

        except ImportError:
            pass  # OpenAI not available
        except Exception as e:
            logger.error(f"Failed to unpatch OpenAI module: {e}")


def register_memoria_instance(memoria_instance):
    """
    Register a Memoria instance for automatic OpenAI interception.

    Args:
        memoria_instance: Memoria instance to register
    """
    global _enabled_memoria_instances

    if memoria_instance not in _enabled_memoria_instances:
        _enabled_memoria_instances.append(memoria_instance)
        logger.debug("Registered Memoria instance for OpenAI interception")

    # Ensure OpenAI is patched
    OpenAIInterceptor.patch_openai()


def unregister_memoria_instance(memoria_instance):
    """
    Unregister a Memoria instance from automatic OpenAI interception.

    Args:
        memoria_instance: Memoria instance to unregister
    """
    global _enabled_memoria_instances

    if memoria_instance in _enabled_memoria_instances:
        _enabled_memoria_instances.remove(memoria_instance)
        logger.debug("Unregistered Memoria instance from OpenAI interception")

    # If no more instances, unpatch OpenAI
    if not _enabled_memoria_instances:
        OpenAIInterceptor.unpatch_openai()


def get_enabled_instances():
    """Get list of currently enabled Memoria instances."""
    return _enabled_memoria_instances.copy()


def is_openai_patched():
    """Check if OpenAI module is currently patched."""
    return OpenAIInterceptor._is_patched


class _StreamingRecorderWrapper:
    """Wrapper that buffers streaming chunks before recording."""

    def __init__(self, stream, finalize_callback):
        self._stream = stream
        self._finalize_callback = finalize_callback
        self._buffer = []
        self._finalized = False

    def __iter__(self):
        completed = False
        try:
            for chunk in self._stream:
                self._buffer.append(chunk)
                yield chunk
            completed = True
        finally:
            self._finalize(completed)

    def _finalize(self, completed: bool) -> None:
        if self._finalized:
            return

        self._finalized = True

        if not completed or not self._buffer:
            return

        try:
            self._finalize_callback(tuple(self._buffer))
        except Exception as exc:  # pragma: no cover - logging safety
            logger.error(f"Failed to finalize streaming record: {exc}")

    def close(self) -> None:
        if hasattr(self._stream, "close"):
            self._stream.close()
        self._finalize(False)

    def __getattr__(self, item):
        return getattr(self._stream, item)


class _AsyncStreamingRecorderWrapper:
    """Async wrapper that buffers streaming chunks before recording."""

    def __init__(self, stream, finalize_callback):
        self._stream = stream
        self._finalize_callback = finalize_callback
        self._buffer = []
        self._finalized = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        completed = False
        try:
            async for chunk in self._stream:
                self._buffer.append(chunk)
                yield chunk
            completed = True
        finally:
            await self._finalize(completed)

    async def _finalize(self, completed: bool) -> None:
        if self._finalized:
            return

        self._finalized = True

        if not completed or not self._buffer:
            return

        try:
            result = self._finalize_callback(tuple(self._buffer))
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # pragma: no cover - logging safety
            logger.error(f"Failed to finalize async streaming record: {exc}")

    async def aclose(self) -> None:
        if hasattr(self._stream, "aclose"):
            await self._stream.aclose()
        await self._finalize(False)

    def __getattr__(self, item):
        return getattr(self._stream, item)


# Legacy wrapper logic has moved to ``openai_client`` to keep this module
# focused on interception utilities. Backward compatible attribute access is
# provided for existing imports.


def __getattr__(name):
    if name in {"MemoriaOpenAI", "MemoriaOpenAIInterceptor", "create_openai_client"}:
        from .openai_client import (
            MemoriaOpenAI,
            MemoriaOpenAIInterceptor,
            create_openai_client,
        )

        return {
            "MemoriaOpenAI": MemoriaOpenAI,
            "MemoriaOpenAIInterceptor": MemoriaOpenAIInterceptor,
            "create_openai_client": create_openai_client,
        }[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
