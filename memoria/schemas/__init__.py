from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, root_validator, validator

from .constants import X_AXIS, Y_AXIS, Z_AXIS
from memoria.utils.pydantic_compat import model_validate

PYDANTIC_V2 = hasattr(BaseModel, "model_validate")


class XCoordMismatchError(ValueError):
    code = "x_coord_mismatch"
    msg_template = "x_coord likely mismatches timestamp-derived value"

    def __init__(self) -> None:
        super().__init__(self.msg_template)


_ORIGINAL_VALIDATION_ERROR_ERRORS = ValidationError.errors
_ORIGINAL_VALIDATION_ERROR_JSON = ValidationError.json


def _normalize_x_coord_error_details(
    details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure custom x_coord errors expose legacy metadata."""

    for item in details:
        if item.get("loc") == (
            "__root__",
            "x_coord",
        ) and "x_coord likely mismatches" in item.get("msg", ""):
            item["type"] = XCoordMismatchError.code
            item["msg"] = XCoordMismatchError.msg_template
            item["ctx"] = {"error": XCoordMismatchError.msg_template}
    return details


def _errors_with_custom_code(self: ValidationError) -> list[dict[str, Any]]:
    details = _ORIGINAL_VALIDATION_ERROR_ERRORS(self)
    return _normalize_x_coord_error_details(details)


def _json_with_custom_code(self: ValidationError, *args: Any, **kwargs: Any) -> str:
    try:
        original_json = _ORIGINAL_VALIDATION_ERROR_JSON(self, *args, **kwargs)
    except TypeError:
        return json.dumps(self.errors(), default=str)

    try:
        payload = json.loads(original_json)
    except json.JSONDecodeError:
        return original_json

    if isinstance(payload, list):
        for item in payload:
            if (
                isinstance(item, dict)
                and item.get("loc") == ["__root__", "x_coord"]
                and "x_coord likely mismatches" in item.get("msg", "")
            ):
                item["type"] = XCoordMismatchError.code
                item["msg"] = XCoordMismatchError.msg_template
                item["ctx"] = {"error": XCoordMismatchError.msg_template}

    return json.dumps(payload, default=str)


ValidationError.errors = _errors_with_custom_code  # type: ignore[assignment]
ValidationError.json = _json_with_custom_code  # type: ignore[assignment]


def canonicalize_symbolic_anchors(
    anchors: Iterable[str] | str | None,
) -> list[str] | None:
    """Return a whitespace-trimmed list of symbolic anchors."""

    if anchors is None:
        return None

    if isinstance(anchors, str):
        candidates = [anchors]
    else:
        candidates = list(anchors)

    cleaned: list[str] = []
    for anchor in candidates:
        if anchor is None:
            continue
        trimmed = str(anchor).strip()

        if trimmed:
            cleaned.append(trimmed)

    return cleaned


class PersonalMemoryDocument(BaseModel):
    """Structured document metadata linked to a personal memory."""

    document_id: str | None = Field(
        default=None,
        description="External identifier for the supporting document or chunk",
    )
    title: str | None = Field(
        default=None, description="Human-readable title for the source document"
    )
    url: str | None = Field(
        default=None,
        description="Optional locator or URL for the source document",
    )
    snippet: str | None = Field(
        default=None, description="Short excerpt that grounds the memory"
    )
    symbolic_anchors: list[str] | None = Field(
        default=None,
        description="Symbolic anchors describing the document context",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary metadata supplied by the ingestion pipeline",
    )

    @root_validator(pre=True)
    def _canonicalize_symbolic_anchors(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "symbolic_anchors" not in values:
            return values
        mutable = dict(values)
        mutable["symbolic_anchors"] = canonicalize_symbolic_anchors(
            mutable.get("symbolic_anchors")
        )
        return mutable

    @validator("document_id", "title", "url", "snippet", pre=True)
    def _strip_optional_strings(cls, value: Any) -> str | None:
        if value in (None, "", b""):
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        text = str(value).strip()
        return text or None

    @validator("metadata", pre=True)
    def _ensure_metadata_mapping(cls, value: Any) -> dict[str, Any] | None:
        if value in (None, "", {}):
            return None
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    "Document metadata must be valid JSON or a mapping"
                ) from exc
            if not isinstance(parsed, Mapping):
                raise ValueError("Document metadata JSON must represent an object")
            return dict(parsed)
        raise TypeError("Document metadata must be a mapping or JSON object")

    if PYDANTIC_V2:
        model_config = {"extra": "allow"}
    else:  # pragma: no cover - exercised in Pydantic v1 environments

        class Config:
            extra = "allow"


class MemoryEntry(BaseModel):
    """Schema for storing a memory entry with spatial and symbolic data."""

    anchor: str
    text: str
    tokens: int
    timestamp: datetime | None = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    x_coord: float | None = Field(default=None, description=X_AXIS.description)
    y_coord: float | None = Field(
        default=None,
        ge=Y_AXIS.min,
        le=Y_AXIS.max,
        description=Y_AXIS.description,
    )
    z_coord: float | None = Field(
        default=None,
        ge=Z_AXIS.min,
        le=Z_AXIS.max,
        description=Z_AXIS.description,
    )
    symbolic_anchors: list[str] | None = None

    @root_validator(pre=True)
    def _canonicalize_symbolic_anchors(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Trim whitespace from symbolic anchors before further validation."""

        if "symbolic_anchors" not in values:
            return values

        mutable_values = dict(values)
        mutable_values["symbolic_anchors"] = canonicalize_symbolic_anchors(
            mutable_values.get("symbolic_anchors")
        )
        return mutable_values

    @root_validator(pre=True)
    def _default_timestamp_and_x_coord(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Ensure timestamp/x_coord defaults mirror legacy behaviour."""

        values = dict(values)
        now = datetime.now(timezone.utc)
        timestamp_provided = "timestamp" in values and values["timestamp"] is not None
        x_provided = "x_coord" in values and values["x_coord"] is not None

        if not timestamp_provided:
            values["timestamp"] = now
            if not x_provided:
                values["x_coord"] = 0.0

        return values

    @root_validator(skip_on_failure=True)
    def _compute_x_coord(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Auto-compute and validate x_coord relative to timestamp."""

        timestamp = values.get("timestamp")
        x_coord = values.get("x_coord")
        if timestamp is None:
            return values

        now = datetime.now(timezone.utc)
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
            values["timestamp"] = ts

        if x_coord is None:
            delta_days = (ts.date() - now.date()).days
            values["x_coord"] = float(delta_days)
        else:
            derived_x = float((ts.date() - now.date()).days)
            if abs(x_coord - derived_x) > 1:
                if PYDANTIC_V2:
                    raise ValidationError.from_exception_data(
                        cls.__name__,
                        [
                            {
                                "loc": ("__root__", "x_coord"),
                                "msg": XCoordMismatchError.msg_template,
                                "type": "value_error",
                                "input": x_coord,
                                "ctx": {
                                    "error": XCoordMismatchError.msg_template,
                                    "expected": derived_x,
                                    "actual": x_coord,
                                },
                            }
                        ],
                    )
                else:  # pragma: no cover
                    raise XCoordMismatchError()

        return values


class PersonalMemoryEntry(MemoryEntry):
    """Memory entry enriched with chat linkage and optional documents."""

    chat_id: str = Field(
        ..., description="Chat session identifier associated with the memory"
    )
    documents: list[PersonalMemoryDocument] | None = Field(
        default=None,
        description="Optional supporting documents captured during ingestion",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata describing the personal memory entry",
    )

    @validator("chat_id")
    def _validate_chat_id(cls, value: Any) -> str:
        if value in (None, "", b""):
            raise ValueError("chat_id must be a non-empty string")
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        text = str(value).strip()
        if not text:
            raise ValueError("chat_id must be a non-empty string")
        return text

    @validator("metadata", pre=True)
    def _coerce_metadata(cls, value: Any) -> dict[str, Any] | None:
        if value in (None, "", {}):
            return None
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError("metadata must be valid JSON or mapping") from exc
            if not isinstance(parsed, Mapping):
                raise ValueError("metadata JSON must define an object")
            return dict(parsed)
        raise TypeError("metadata must be provided as a mapping or JSON object")

    @validator("documents", pre=True)
    def _coerce_documents(
        cls, value: Any
    ) -> list[PersonalMemoryDocument] | None | list[dict[str, Any]]:
        if value in (None, "", []):
            return None
        if isinstance(value, PersonalMemoryDocument):
            return [value]
        if isinstance(value, Mapping):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
        raise TypeError(
            "documents must be provided as a PersonalMemoryDocument, mapping, or sequence"
        )

    if PYDANTIC_V2:
        model_config = {"extra": "allow"}
    else:  # pragma: no cover - exercised in Pydantic v1 environments

        class Config:
            extra = "allow"


@lru_cache(maxsize=1)
def _read_hypertext_graph_schema() -> dict[str, Any]:
    """Load the hypertext graph schema from package resources."""

    schema_text = resources.read_text(
        __name__, "hypertext_graph.json", encoding="utf-8"
    )
    return json.loads(schema_text)


def load_hypertext_graph_schema() -> dict[str, Any]:
    """Return a copy of the hypertext graph JSON schema for validation."""

    return copy.deepcopy(_read_hypertext_graph_schema())


HYPERTEXT_GRAPH_SCHEMA: Mapping[str, Any] = MappingProxyType(
    _read_hypertext_graph_schema()
)


def validate_memory_entry(data: dict[str, Any]) -> MemoryEntry:
    """Validate raw payloads while supporting legacy Pydantic versions."""

    return model_validate(MemoryEntry, data)


class RitualMetadata(BaseModel):
    """Lightweight ritual metadata shared by thread participants."""

    name: str | None = None
    phase: str | None = None
    location: str | None = None
    attributes: dict[str, Any] | None = None

    class Config:
        extra = "allow"


class ThreadMessage(MemoryEntry):
    """Schema describing an ordered conversational message."""

    if PYDANTIC_V2:
        model_config = {
            "populate_by_name": True,
            "validate_by_name": True,
            "extra": "allow",
        }
    else:  # pragma: no cover - exercised in Pydantic v1 environments

        class Config:
            allow_population_by_field_name = True
            extra = "allow"

    role: str = Field(default="user", description="Actor delivering the message")
    message_id: str | None = Field(
        default=None, description="Upstream message identifier"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Arbitrary per-message metadata"
    )
    chat_id: str | None = Field(
        default=None, description="Optional chat session identifier"
    )
    emotional_intensity: float | None = Field(
        default=None, description="Optional emotional intensity signal"
    )

    @root_validator(pre=True)
    def _alias_text_field(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Support both ``content`` and ``text`` payload keys."""

        values = dict(values)
        if "content" in values and "text" not in values:
            values["text"] = values.get("content")
        elif "text" in values and "content" not in values:
            values["content"] = values.get("text")
        return values


class ThreadIngestion(BaseModel):
    """Thread ingestion payload capturing ordered ritual exchanges."""

    thread_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    shared_symbolic_anchors: list[str] | None = None
    ritual: RitualMetadata | None = None
    messages: list[ThreadMessage]
    metadata: dict[str, Any] | None = None

    @root_validator(pre=True)
    def _canonicalize_shared_anchors(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Normalize shared symbolic anchors before validation."""

        values = dict(values)
        if "shared_symbolic_anchors" in values:
            values["shared_symbolic_anchors"] = canonicalize_symbolic_anchors(
                values.get("shared_symbolic_anchors")
            )
        return values

    @root_validator(skip_on_failure=True)
    def _ensure_messages_present(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Guarantee at least one message is supplied for ingestion."""

        messages = values.get("messages") or []
        if not messages:
            raise ValidationError.from_exception_data(
                cls.__name__,
                [
                    {
                        "loc": ("messages",),
                        "msg": "Thread ingestion requires at least one message",
                        "type": "value_error",
                        "input": messages,
                        "ctx": {
                            "error": "Thread ingestion requires at least one message"
                        },
                    }
                ],
            )
        return values


__all__ = [
    "canonicalize_symbolic_anchors",
    "PersonalMemoryDocument",
    "PersonalMemoryEntry",
    "MemoryEntry",
    "ThreadMessage",
    "ThreadIngestion",
    "RitualMetadata",
    "HYPERTEXT_GRAPH_SCHEMA",
    "load_hypertext_graph_schema",
    "validate_memory_entry",
]
