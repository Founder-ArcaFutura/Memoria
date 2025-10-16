from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel

from memoria.utils.pydantic_compat import model_dump, model_dump_json


class ExampleModel(BaseModel):
    value: int
    timestamp: datetime


def _native_model_dump(model: Any, **kwargs: Any) -> Any:
    method = getattr(model, "model_dump", None)
    if callable(method):
        try:
            return method(**kwargs)
        except TypeError:
            return method()

    legacy = getattr(model, "dict", None)
    if callable(legacy):
        try:
            return legacy(**kwargs)
        except TypeError:
            return legacy()

    raise AttributeError("Model lacks serialisation helpers")


def _native_model_dump_json(model: Any, **kwargs: Any) -> str:
    method = getattr(model, "model_dump_json", None)
    if callable(method):
        try:
            return method(**kwargs)
        except TypeError:
            return method()

    legacy = getattr(model, "json", None)
    if callable(legacy):
        try:
            return legacy(**kwargs)
        except TypeError:
            return legacy()

    raise AttributeError("Model lacks JSON serialisation helpers")


def test_model_dump_matches_native_behaviour():
    model = ExampleModel(value=5, timestamp=datetime(2024, 1, 1))

    expected = _native_model_dump(model, mode="json")
    result = model_dump(model, mode="json")

    assert result == expected


def test_model_dump_json_matches_native_behaviour():
    model = ExampleModel(value=10, timestamp=datetime(2024, 2, 2))

    expected = _native_model_dump_json(model, mode="json")
    result = model_dump_json(model, mode="json")

    assert json.loads(result) == json.loads(expected)


def test_wrappers_forward_kwargs_to_model_methods():
    class DummyModel:
        def __init__(self) -> None:
            self.dump_kwargs: dict[str, Any] | None = None
            self.json_kwargs: dict[str, Any] | None = None

        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            self.dump_kwargs = dict(kwargs)
            return {"value": 1}

        def model_dump_json(self, **kwargs: Any) -> str:
            self.json_kwargs = dict(kwargs)
            return json.dumps({"value": 1})

    dummy = DummyModel()

    result = model_dump(dummy, mode="json", exclude={"ignored"})
    assert result == {"value": 1}
    assert dummy.dump_kwargs == {"mode": "json", "exclude": {"ignored"}}

    json_result = model_dump_json(dummy, mode="json", exclude={"ignored"})
    assert json.loads(json_result) == {"value": 1}
    assert dummy.json_kwargs == {"mode": "json", "exclude": {"ignored"}}


def test_wrappers_raise_when_methods_missing():
    class Noop:
        pass

    with pytest.raises(AttributeError):
        model_dump(Noop())

    with pytest.raises(AttributeError):
        model_dump_json(Noop())


def test_manual_memory_uses_helper(monkeypatch: pytest.MonkeyPatch):
    import scripts.manual_memory as manual_memory

    inputs = iter(["anchor", "some text", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    class DummyResponse:
        def json(self) -> dict[str, str]:
            return {"status": "ok"}

    captured: dict[str, Any] = {}

    def fake_post(
        url: str, json: Any = None, headers: dict[str, Any] | None = None
    ) -> DummyResponse:
        captured["json"] = json
        captured["headers"] = headers
        return DummyResponse()

    monkeypatch.setattr(manual_memory.requests, "post", fake_post)

    called: dict[str, Any] = {}

    def fake_model_dump(model: Any, **kwargs: Any) -> dict[str, str]:
        called["called"] = True
        called["kwargs"] = dict(kwargs)
        return {"payload": "ok"}

    monkeypatch.setattr(manual_memory, "model_dump", fake_model_dump)

    manual_memory.main()

    assert called["called"] is True
    assert captured["json"] == {"payload": "ok"}
    assert called["kwargs"] == {}
