"""Compatibility helpers for Pydantic 1.x and 2.x model serialisation."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, TypeVar


def _call_method(model: Any, name: str) -> Callable[..., Any] | None:
    """Return a callable attribute from *model* if it exists."""

    attr = getattr(model, name, None)
    return attr if callable(attr) else None


def model_dump(model: Any, **kwargs: Any) -> Any:
    """Return a serialised representation using modern Pydantic helpers."""

    method = _call_method(model, "model_dump")
    if method is not None:
        try:
            return method(**kwargs)
        except TypeError as exc:
            if "mode" in kwargs:
                trimmed_kwargs = dict(kwargs)
                trimmed_kwargs.pop("mode", None)
                return method(**trimmed_kwargs)
            raise exc

    v1_method = _call_method(model, "dict")
    if v1_method is not None:
        v1_kwargs = dict(kwargs)
        v1_kwargs.pop("mode", None)
        return v1_method(**v1_kwargs)

    raise AttributeError("Object does not provide a model_dump or dict interface")


def model_dump_json(model: Any, **kwargs: Any) -> str:
    """Return a JSON serialisation using modern Pydantic helpers."""

    method = _call_method(model, "model_dump_json")
    if method is not None:
        try:
            return method(**kwargs)
        except TypeError as exc:
            if "mode" in kwargs:
                trimmed_kwargs = dict(kwargs)
                trimmed_kwargs.pop("mode", None)
                return method(**trimmed_kwargs)
            raise exc

    v1_method = _call_method(model, "json")
    if v1_method is not None:
        v1_kwargs = dict(kwargs)
        v1_kwargs.pop("mode", None)
        return v1_method(**v1_kwargs)

    raise AttributeError("Object does not provide a model_dump_json or json interface")


def model_dump_json_safe(model: Any, **kwargs: Any) -> dict[str, Any]:
    """Return a JSON-safe mapping by leveraging ``model_dump_json`` compatibility."""

    json_kwargs = dict(kwargs)
    json_kwargs.setdefault("mode", "json")
    json_payload = model_dump_json(model, **json_kwargs)
    if isinstance(json_payload, bytes):
        json_payload = json_payload.decode("utf-8")
    return json.loads(json_payload)


ModelT = TypeVar("ModelT")


def model_validate(model_cls: type[ModelT], data: Mapping[str, Any] | Any) -> ModelT:
    """Validate *data* against *model_cls* supporting Pydantic 1.x and 2.x."""

    validator = getattr(model_cls, "model_validate", None)
    if callable(validator):
        return validator(data)

    parser = getattr(model_cls, "parse_obj", None)
    if callable(parser):
        return parser(data)

    if isinstance(data, Mapping):
        return model_cls(**data)

    raise TypeError(
        "Data for validation must be a mapping when `parse_obj` is unavailable."
    )
