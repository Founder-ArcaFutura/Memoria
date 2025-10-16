import asyncio
import sys
from types import SimpleNamespace

import pytest


class _AnthropicStub:
    def __init__(self, *_, **__):
        self.messages = SimpleNamespace()


sys.modules.setdefault("anthropic", SimpleNamespace(Anthropic=_AnthropicStub))

from memoria.integrations.anthropic_integration import MemoriaAnthropic
from memoria.providers import anthropic_recorder


class DummyMemoria:
    def __init__(self):
        self.is_enabled = True
        self.conscious_ingest = False
        self.calls = []

    def retrieve_context(self, *_args, **_kwargs):
        return []

    def record_conversation(self, **payload):
        self.calls.append(payload)


@pytest.fixture
def memoria_instances():
    return DummyMemoria(), DummyMemoria()


def _make_client(memoria_instance):
    client = MemoriaAnthropic(memoria_instance, api_key="test")
    return client


def test_anthropic_streaming_sync_matches_persistence(memoria_instances):
    memoria_stream, memoria_baseline = memoria_instances
    client = _make_client(memoria_stream)

    messages = [{"role": "user", "content": "Hello Claude"}]
    chunks = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(id="msg_1", model="claude-3-opus"),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text"),
        ),
        SimpleNamespace(
            type="content_block_delta", index=0, delta=SimpleNamespace(text="Hello ")
        ),
        SimpleNamespace(
            type="content_block_delta", index=0, delta=SimpleNamespace(text="world")
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        ),
        SimpleNamespace(type="message_stop", stop_reason="end_turn"),
    ]

    def fake_create(**kwargs):
        assert kwargs.get("stream") is True
        return iter(chunks)

    client._anthropic.messages.create = fake_create

    request_kwargs = {"stream": True, "model": "claude-3-opus", "messages": messages}
    wrapped_stream = client.messages.create(**request_kwargs)

    assert memoria_stream.calls == []
    assert list(wrapped_stream) == chunks
    assert len(memoria_stream.calls) == 1

    baseline_response = SimpleNamespace(
        model="claude-3-opus",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="Hello world")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
    )

    baseline_kwargs = {"model": "claude-3-opus", "messages": messages}
    anthropic_recorder.record_conversation(
        memoria_baseline, baseline_kwargs, baseline_response
    )

    assert len(memoria_baseline.calls) == 1

    stream_call = memoria_stream.calls[0]
    baseline_call = memoria_baseline.calls[0]

    assert stream_call["user_input"] == baseline_call["user_input"]
    assert stream_call["ai_output"] == baseline_call["ai_output"]
    assert stream_call["model"] == baseline_call["model"]
    assert stream_call["metadata"] == baseline_call["metadata"]


def test_anthropic_streaming_async_tool_blocks(memoria_instances):
    memoria_stream, memoria_baseline = memoria_instances
    client = _make_client(memoria_stream)

    messages = [{"role": "user", "content": "Call a tool"}]
    chunks = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                id="msg_2", model="claude-3-haiku", role="assistant"
            ),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(
                type="tool_use", id="tool_1", name="lookup", input={}
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(partial_json='{"query":'),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(partial_json=' "value"}'),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="tool_use"),
            usage=SimpleNamespace(input_tokens=5, output_tokens=7),
        ),
        SimpleNamespace(type="message_stop", stop_reason="tool_use"),
    ]

    async def stream():
        for chunk in chunks:
            yield chunk

    def fake_create(**kwargs):
        assert kwargs.get("stream") is True
        return stream()

    client._anthropic.messages.create = fake_create

    request_kwargs = {"stream": True, "model": "claude-3-haiku", "messages": messages}
    wrapped_stream = client.messages.create(**request_kwargs)

    async def consume():
        async for _ in wrapped_stream:
            pass

    asyncio.run(consume())

    assert len(memoria_stream.calls) == 1

    baseline_response = SimpleNamespace(
        model="claude-3-haiku",
        stop_reason="tool_use",
        content=[
            SimpleNamespace(
                type="tool_use",
                id="tool_1",
                name="lookup",
                input={"query": "value"},
            )
        ],
        usage=SimpleNamespace(input_tokens=5, output_tokens=7),
    )

    baseline_kwargs = {"model": "claude-3-haiku", "messages": messages}
    anthropic_recorder.record_conversation(
        memoria_baseline, baseline_kwargs, baseline_response
    )

    assert len(memoria_baseline.calls) == 1

    stream_call = memoria_stream.calls[0]
    baseline_call = memoria_baseline.calls[0]

    assert stream_call["user_input"] == baseline_call["user_input"]
    assert stream_call["ai_output"] == baseline_call["ai_output"]
    assert stream_call["model"] == baseline_call["model"]
    assert stream_call["metadata"] == baseline_call["metadata"]
