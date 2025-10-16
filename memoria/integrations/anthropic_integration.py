"""
Anthropic Integration - Clean wrapper without monkey-patching

RECOMMENDED: Use LiteLLM instead for unified API and native callback support.
This integration is provided for direct Anthropic SDK usage.

Usage:
    from memoria.integrations.anthropic_integration import MemoriaAnthropic
    from memoria.config import ConfigManager
    import os

    # Initialize with your memoria instance
    client = MemoriaAnthropic(memoria_instance, api_key="your-key")

    # Pull model name from config or environment (ANTHROPIC_MODEL)
    config = ConfigManager()
    model_name = config.get_setting("agents.default_model", os.getenv("ANTHROPIC_MODEL"))

    # Use exactly like Anthropic client
    response = client.messages.create(model=model_name or "claude-3-opus", messages=[...])

Note:
    Configure the model via `agents.default_model` in your memoria.json or set the
    ANTHROPIC_MODEL environment variable.
"""

import inspect
import os

from loguru import logger

from memoria.config import ConfigManager
from memoria.providers import anthropic_recorder


def get_default_anthropic_model() -> str | None:
    """Return the configured default Anthropic model name."""

    return ConfigManager().get_setting(
        "agents.default_model", os.getenv("ANTHROPIC_MODEL")
    )


class MemoriaAnthropic:
    """
    Clean Anthropic wrapper that automatically records conversations
    without monkey-patching. Drop-in replacement for Anthropic client.
    """

    def __init__(self, memoria_instance, api_key: str | None = None, **kwargs):
        """
        Initialize MemoriaAnthropic wrapper

        Args:
            memoria_instance: Memoria instance for recording conversations
            api_key: Anthropic API key
            **kwargs: Additional arguments passed to Anthropic client
        """
        try:
            import anthropic

            self._anthropic = anthropic.Anthropic(api_key=api_key, **kwargs)
            self._memoria = memoria_instance

            # Create wrapped messages
            self.messages = self._create_messages_wrapper()

            # Pass through other attributes
            for attr in dir(self._anthropic):
                if not attr.startswith("_") and attr not in ["messages"]:
                    setattr(self, attr, getattr(self._anthropic, attr))

        except ImportError as err:
            raise ImportError(
                "Anthropic package required: pip install anthropic"
            ) from err

    def _create_messages_wrapper(self):
        """Create wrapped messages"""

        class MessagesWrapper:
            def __init__(self, anthropic_client, memoria_instance):
                self._anthropic = anthropic_client
                self._memoria = memoria_instance

            def create(self, **kwargs):
                # Inject context if conscious ingestion is enabled
                if self._memoria.is_enabled and self._memoria.conscious_ingest:
                    kwargs = self._inject_context(kwargs)

                # Make the actual API call
                response = self._anthropic.messages.create(**kwargs)

                # Record conversation if memoria is enabled
                if self._memoria.is_enabled:
                    if kwargs.get("stream"):
                        response = self._wrap_stream_response(kwargs, response)
                    else:
                        self._record_conversation(kwargs, response)

                return response

            def _inject_context(self, kwargs):
                """Inject relevant context into messages"""
                try:
                    # Extract user input from messages
                    user_input = ""
                    for msg in reversed(kwargs.get("messages", [])):
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                # Handle content blocks
                                user_input = " ".join(
                                    [
                                        block.get("text", "")
                                        for block in content
                                        if isinstance(block, dict)
                                        and block.get("type") == "text"
                                    ]
                                )
                            else:
                                user_input = content
                            break

                    if user_input:
                        # Fetch relevant context
                        context = self._memoria.retrieve_context(user_input, limit=3)

                        if context:
                            # Create a context prompt
                            context_prompt = "--- Relevant Memories ---\n"
                            for mem in context:
                                if isinstance(mem, dict):
                                    summary = mem.get("summary", "") or mem.get(
                                        "content", ""
                                    )
                                    context_prompt += f"- {summary}\n"
                                else:
                                    context_prompt += f"- {str(mem)}\n"
                            context_prompt += "-------------------------\n"

                            # Inject context into the system parameter
                            if kwargs.get("system"):
                                # Prepend to existing system message
                                kwargs["system"] = context_prompt + kwargs["system"]
                            else:
                                # Add as system message
                                kwargs["system"] = context_prompt

                            logger.debug(f"Injected context: {len(context)} memories")
                except Exception as e:
                    logger.error(f"Context injection failed: {e}")

                return kwargs

            def _record_conversation(self, kwargs, response):
                """Record the conversation using shared recorder."""
                anthropic_recorder.record_conversation(self._memoria, kwargs, response)

            def _wrap_stream_response(self, kwargs, response):
                """Wrap streaming responses to buffer chunks before recording."""

                if response is None:
                    return response

                def finalize(chunks):
                    try:
                        assembled = anthropic_recorder.assemble_streamed_response(
                            chunks
                        )
                        if assembled is None:
                            return
                        self._record_conversation(kwargs, assembled)
                    except Exception as exc:  # pragma: no cover - logging safety
                        logger.error(
                            f"Failed to finalize Anthropic streaming record: {exc}"
                        )

                if hasattr(response, "__aiter__"):
                    return _AnthropicAsyncStreamWrapper(response, finalize)

                if hasattr(response, "__iter__"):
                    return _AnthropicStreamWrapper(response, finalize)

                return response

        return MessagesWrapper(self._anthropic, self._memoria)


class _AnthropicStreamWrapper:
    """Wrapper that buffers Anthropic streaming chunks before recording."""

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

        if completed and self._buffer:
            try:
                self._finalize_callback(tuple(self._buffer))
            except Exception as exc:  # pragma: no cover - logging safety
                logger.error(f"Failed to finalize Anthropic stream: {exc}")

        if hasattr(self._stream, "close"):
            try:
                self._stream.close()
            except Exception as exc:  # pragma: no cover - logging safety
                logger.debug(f"Error closing Anthropic stream: {exc}")

    def close(self) -> None:
        if hasattr(self._stream, "close"):
            self._stream.close()
        self._finalize(False)

    def __getattr__(self, item):
        return getattr(self._stream, item)


class _AnthropicAsyncStreamWrapper:
    """Async wrapper that buffers Anthropic streaming chunks before recording."""

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

        if completed and self._buffer:
            try:
                result = self._finalize_callback(tuple(self._buffer))
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # pragma: no cover - logging safety
                logger.error(f"Failed to finalize async Anthropic stream: {exc}")

        if hasattr(self._stream, "aclose"):
            try:
                await self._stream.aclose()
            except Exception as exc:  # pragma: no cover - logging safety
                logger.debug(f"Error closing async Anthropic stream: {exc}")

    async def aclose(self) -> None:
        if hasattr(self._stream, "aclose"):
            await self._stream.aclose()
        await self._finalize(False)

    def __getattr__(self, item):
        return getattr(self._stream, item)
