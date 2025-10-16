import asyncio
from types import SimpleNamespace

from memoria.integrations import openai_integration as integration
from memoria.providers import openai_recorder


class DummyMemoria:
    def __init__(self):
        self.is_enabled = True


def _make_options(messages=None, **extra):
    data = {"messages": messages or [{"role": "user", "content": "Hi"}]}
    data.update(extra)
    return SimpleNamespace(json_data=data)


def test_streaming_sync_records_buffered_text(monkeypatch):
    memoria_instance = DummyMemoria()
    monkeypatch.setattr(integration, "_enabled_memoria_instances", [memoria_instance])

    recorded = []

    def fake_record(mem, kwargs, response):
        recorded.append((mem, kwargs, response))

    monkeypatch.setattr(openai_recorder, "record_conversation", fake_record)

    chunks = [
        SimpleNamespace(
            id="chunk-1",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(role="assistant", content="Hello"),
                    finish_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            id="chunk-2",
            usage=SimpleNamespace(total_tokens=5),
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content=" world"),
                    finish_reason="stop",
                )
            ],
        ),
    ]

    options = _make_options()
    wrapped = integration.OpenAIInterceptor._wrap_stream_result(
        iter(chunks), options, "sync"
    )

    assert list(wrapped) == chunks

    assert len(recorded) == 1
    recorded_mem, recorded_kwargs, response = recorded[0]
    assert recorded_mem is memoria_instance
    assert recorded_kwargs["messages"][0]["content"] == "Hi"
    assert response.choices[0].message.content == "Hello world"
    assert getattr(response.usage, "total_tokens", None) == 5


def test_streaming_async_records_tool_call(monkeypatch):
    memoria_instance = DummyMemoria()
    monkeypatch.setattr(integration, "_enabled_memoria_instances", [memoria_instance])

    recorded = []

    def fake_record(mem, kwargs, response):
        recorded.append((mem, kwargs, response))

    monkeypatch.setattr(openai_recorder, "record_conversation", fake_record)

    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(
                        role="assistant",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="tool_1",
                                type="function",
                                function=SimpleNamespace(
                                    name="lookup",
                                    arguments='{"query":',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                index=0, function=SimpleNamespace(arguments=' "value"}')
                            )
                        ],
                    ),
                    finish_reason="stop",
                )
            ],
        ),
    ]

    async def stream():
        for chunk in chunks:
            yield chunk

    async def run_test():
        options = _make_options()
        wrapped = integration.OpenAIInterceptor._wrap_async_stream_result(
            stream(), options, "async"
        )

        async for _ in wrapped:
            pass

    asyncio.run(run_test())

    assert len(recorded) == 1
    _, _, response = recorded[0]
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].id == "tool_1"
    assert tool_calls[0].function.name == "lookup"
    assert tool_calls[0].function.arguments == '{"query": "value"}'


def test_non_streaming_path_unchanged(monkeypatch):
    memoria_instance = DummyMemoria()
    monkeypatch.setattr(integration, "_enabled_memoria_instances", [memoria_instance])

    recorded = []

    def fake_record(mem, kwargs, response):
        recorded.append((mem, kwargs, response))

    monkeypatch.setattr(openai_recorder, "record_conversation", fake_record)

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Hello world"),
            )
        ]
    )

    options = _make_options()

    integration.OpenAIInterceptor._record_conversation_for_enabled_instances(
        options, response, "sync"
    )

    assert len(recorded) == 1
    assert recorded[0][2] is response
