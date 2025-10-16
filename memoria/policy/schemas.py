"""Declarative policy schema definitions and validation helpers."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from importlib import resources
from typing import (
    TYPE_CHECKING,
    Any,
)

try:  # pragma: no cover - optional dependency
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover - fallback when jsonschema isn't installed
    Draft202012Validator = None  # type: ignore[assignment]

from pydantic import BaseModel, Field, root_validator, validator

from ..schemas.constants import Y_AXIS
from ..utils.pydantic_compat import model_validate

if TYPE_CHECKING:  # pragma: no cover
    from .enforcement import EnforcementStage, PolicyAction


@dataclass(frozen=True)
class PolicyArtifactDefinition:
    """Metadata about a stored policy artifact."""

    name: str
    artifact_type: str
    payload: Mapping[str, Any]


_POLICY_NAMESPACE_WILDCARD = "*"
_ESCALATION_PRIORITIES = {"low", "normal", "high", "urgent"}
_POLICY_ARTIFACT_RESOURCE = "policy_artifacts.json"


def _ensure_utc_datetime(reference: datetime | None) -> datetime | None:
    if reference is None:
        return None
    if reference.tzinfo is None:
        return reference.replace(tzinfo=timezone.utc)
    return reference.astimezone(timezone.utc)


def _reference_now(reference: datetime | None = None) -> datetime:
    if reference is None:
        return datetime.now(timezone.utc)
    ensured = _ensure_utc_datetime(reference)
    assert ensured is not None  # for type checkers
    return ensured


def _isoformat_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    ensured = _ensure_utc_datetime(value)
    if ensured is None:
        return None
    return ensured.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalise_namespaces(value: Any) -> list[str]:
    if value in (None, "", []):
        return [_POLICY_NAMESPACE_WILDCARD]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return parts or [_POLICY_NAMESPACE_WILDCARD]
    cleaned: list[str] = []
    if isinstance(value, Iterable):
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                cleaned.append(text)
    if cleaned:
        return cleaned
    return [_POLICY_NAMESPACE_WILDCARD]


class RetentionCeiling(BaseModel):
    """Defines the maximum lifecycle window for a namespace."""

    name: str = Field(..., description="Identifier for the retention ceiling")
    namespaces: list[str] = Field(
        default_factory=lambda: [_POLICY_NAMESPACE_WILDCARD],
        description="Namespaces or glob patterns governed by this ceiling",
    )
    max_days: float = Field(
        ..., gt=0.0, description="Maximum retention window in days before escalation"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata forwarded with audit events",
    )

    @validator("namespaces", pre=True)
    def _coerce_namespaces(cls, value: Any) -> list[str]:
        return _normalise_namespaces(value)


class NamespacePrivacyFloor(BaseModel):
    """Declares the minimum privacy value allowed per namespace."""

    name: str = Field(..., description="Identifier for the privacy floor rule")
    namespaces: list[str] = Field(
        default_factory=lambda: [_POLICY_NAMESPACE_WILDCARD],
        description="Namespaces or glob patterns impacted by this floor",
    )
    privacy_floor: float = Field(
        ..., ge=Y_AXIS.min, le=Y_AXIS.max, description="Minimum allowed Y-axis value"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata forwarded with audit events",
    )

    @validator("namespaces", pre=True)
    def _coerce_namespaces(cls, value: Any) -> list[str]:
        return _normalise_namespaces(value)


class EscalationContact(BaseModel):
    """Represents an on-call escalation target for policy violations."""

    name: str = Field(..., description="Human readable escalation contact name")
    channel: str = Field(..., description="Transport used to notify the contact")
    target: str = Field(..., description="Endpoint or identifier for dispatch")
    priority: str | None = Field(
        default=None,
        description="Relative urgency communicated with the escalation",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured metadata for downstream tooling",
    )

    @validator("channel", pre=True)
    def _normalise_channel(cls, value: Any) -> str:
        if value is None:
            raise ValueError("Escalation channel must be provided")
        channel = str(value).strip().lower()
        if not channel:
            raise ValueError("Escalation channel must not be empty")
        return channel

    @validator("priority", pre=True)
    def _normalise_priority(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        priority = str(value).strip().lower()
        if priority and priority not in _ESCALATION_PRIORITIES:
            raise ValueError(
                f"Escalation priority must be one of {sorted(_ESCALATION_PRIORITIES)}"
            )
        return priority or None


class PolicyOverride(BaseModel):
    """Temporary exceptions or adjustments to policy enforcement."""

    name: str = Field(..., description="Identifier for the override")
    namespaces: list[str] = Field(
        default_factory=lambda: [_POLICY_NAMESPACE_WILDCARD],
        description="Namespaces or glob patterns covered by the override",
    )
    target_policies: list[str] = Field(
        default_factory=list,
        description="Specific policy rules the override affects",
    )
    action: PolicyAction = Field(
        default="allow",
        description="Action to enforce when override conditions are met",
    )
    stage: EnforcementStage | None = Field(
        default=None,
        description="Workflow stage where the override applies",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Optional timestamp when the override should lapse",
    )
    justification: str | None = Field(
        default=None,
        description="Reasoning captured for compliance reviews",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for audit and notification systems",
    )

    @validator("namespaces", pre=True)
    def _coerce_namespaces(cls, value: Any) -> list[str]:
        return _normalise_namespaces(value)

    @validator("target_policies", pre=True)
    def _coerce_targets(cls, value: Any) -> list[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, Iterable):
            cleaned: list[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned.append(text)
            return cleaned
        raise TypeError(
            "target_policies must be provided as a string or iterable of strings"
        )

    @validator("action", pre=True)
    def _coerce_action(cls, value: Any) -> PolicyAction:
        from .enforcement import PolicyAction as _PolicyAction

        if isinstance(value, _PolicyAction):
            return value
        if isinstance(value, str) and value.strip():
            normalized = value.strip().lower()
            for member in _PolicyAction:
                if member.value == normalized:
                    return member
        raise ValueError("Override action must reference a supported PolicyAction")

    @validator("stage", pre=True)
    def _coerce_stage(cls, value: Any) -> EnforcementStage | None:
        from .enforcement import EnforcementStage as _EnforcementStage

        if value in (None, ""):
            return None
        if isinstance(value, _EnforcementStage):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in _EnforcementStage:
                if member.value == normalized:
                    return member
        raise ValueError("Override stage must be a valid EnforcementStage")

    @validator("expires_at", pre=True)
    def _coerce_datetime(cls, value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("expires_at must be ISO-8601 formatted") from exc
        raise TypeError("expires_at must be a datetime or ISO-8601 string")


def summarise_override_window(
    override: PolicyOverride,
    *,
    reference_time: datetime | None = None,
    window_days: int = 14,
) -> dict[str, Any]:
    """Return a status snapshot for *override* relative to *reference_time*."""

    reference = _reference_now(reference_time)
    expires_at = _ensure_utc_datetime(override.expires_at)

    status = "indefinite"
    days_remaining: float | None = None
    if expires_at is not None:
        delta = expires_at - reference
        days_remaining = round(delta.total_seconds() / 86400, 2)
        if expires_at <= reference:
            status = "expired"
        elif delta <= timedelta(days=max(window_days, 0)):
            status = "expiring"
        else:
            status = "active"

    action_value = (
        override.action.value if hasattr(override.action, "value") else override.action
    )
    stage_value = override.stage.value if getattr(override, "stage", None) else None

    payload = {
        "name": override.name,
        "namespaces": list(override.namespaces or []),
        "target_policies": list(override.target_policies or []),
        "action": action_value,
        "stage": stage_value,
        "expires_at": _isoformat_utc(expires_at),
        "status": status,
        "days_remaining": days_remaining,
        "justification": override.justification,
        "metadata": override.metadata or {},
        "namespace_count": len(list(override.namespaces or [])),
        "policy_count": len(list(override.target_policies or [])),
        "evaluated_at": _isoformat_utc(reference),
    }
    return payload


def apply_override_expiry(
    override: PolicyOverride, *, reference_time: datetime | None = None
) -> tuple[PolicyOverride, dict[str, Any]]:
    """Annotate override metadata with automation status and expiry markers."""

    summary = summarise_override_window(
        override, reference_time=reference_time, window_days=0
    )
    metadata = dict(override.metadata or {})
    changed = False

    status = summary.get("status")
    evaluated_at = summary.get("evaluated_at")

    def _assign(key: str, value: Any) -> None:
        nonlocal changed
        if value is None:
            if key in metadata:
                metadata.pop(key, None)
                changed = True
            return
        if metadata.get(key) != value:
            metadata[key] = value
            changed = True

    _assign("override_status", status)
    _assign("override_checked_at", evaluated_at)

    if status == "expired":
        expired_at = summary.get("expires_at") or evaluated_at
        _assign("expired_at", expired_at)
        if not metadata.get("expired_by"):
            _assign("expired_by", "automation")
    else:
        if "expired_at" in metadata:
            metadata.pop("expired_at", None)
            changed = True
        if "expired_by" in metadata:
            metadata.pop("expired_by", None)
            changed = True

    updated_override = (
        override if not changed else override.copy(update={"metadata": metadata})
    )
    summary["metadata_updated"] = changed
    return updated_override, summary


_POLICY_LIST_FIELDS = (
    "retention_ceilings",
    "namespace_privacy_floors",
    "escalation_contacts",
    "overrides",
)


class PolicyDefinitions(BaseModel):
    """Container for structured policy definitions."""

    retention_ceilings: list[RetentionCeiling] = Field(default_factory=list)
    namespace_privacy_floors: list[NamespacePrivacyFloor] = Field(default_factory=list)
    escalation_contacts: list[EscalationContact] = Field(default_factory=list)
    overrides: list[PolicyOverride] = Field(default_factory=list)

    @validator(*_POLICY_LIST_FIELDS, pre=True)
    def _normalise_sequence(cls, value: Any) -> list[Any]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("Policy definitions must be valid JSON") from exc
            value = parsed
        if isinstance(value, Mapping):
            return [value]
        if isinstance(value, list):
            return value
        raise TypeError(
            "Policy definitions must be provided as a list, mapping, or JSON string"
        )

    @root_validator
    def _ensure_unique_names(
        cls, values: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        for field in _POLICY_LIST_FIELDS:
            seen: dict[str, str] = {}
            for item in values.get(field, []) or []:
                key = str(getattr(item, "name", "")).strip().lower()
                if not key:
                    raise ValueError(f"Entries in {field} must define a name")
                if key in seen:
                    raise ValueError(
                        f"Duplicate entry '{item.name}' detected in {field}; names must be unique"
                    )
                seen[key] = item.name
        return values

    def merge(self, other: PolicyDefinitions) -> PolicyDefinitions:
        """Return a new ``PolicyDefinitions`` combining values from *other*."""

        def _merge(left: Sequence[Any], right: Sequence[Any]) -> list[Any]:
            merged: OrderedDict[str, Any] = OrderedDict()
            for item in left:
                merged[str(item.name).lower()] = item
            for item in right:
                merged[str(item.name).lower()] = item
            return list(merged.values())

        return PolicyDefinitions(
            retention_ceilings=_merge(
                self.retention_ceilings, other.retention_ceilings
            ),
            namespace_privacy_floors=_merge(
                self.namespace_privacy_floors, other.namespace_privacy_floors
            ),
            escalation_contacts=_merge(
                self.escalation_contacts, other.escalation_contacts
            ),
            overrides=_merge(self.overrides, other.overrides),
        )

    def iter_artifacts(self) -> Iterable[PolicyArtifactDefinition]:
        """Yield persistent artifact metadata for all configured policies."""

        for item in self.retention_ceilings:
            yield PolicyArtifactDefinition(
                name=item.name,
                artifact_type="retention_ceiling",
                payload=_dump_model(item),
            )
        for item in self.namespace_privacy_floors:
            yield PolicyArtifactDefinition(
                name=item.name,
                artifact_type="namespace_privacy_floor",
                payload=_dump_model(item),
            )
        for item in self.escalation_contacts:
            yield PolicyArtifactDefinition(
                name=item.name,
                artifact_type="escalation_contact",
                payload=_dump_model(item),
            )
        for item in self.overrides:
            yield PolicyArtifactDefinition(
                name=item.name,
                artifact_type="override_rule",
                payload=_dump_model(item),
            )


POLICY_ARTIFACT_TYPES = frozenset(
    {
        "retention_ceiling",
        "namespace_privacy_floor",
        "escalation_contact",
        "override_rule",
    }
)


@lru_cache(maxsize=1)
def _read_policy_artifact_schema() -> Mapping[str, Any]:
    schema_text = resources.read_text(
        "memoria.schemas", _POLICY_ARTIFACT_RESOURCE, encoding="utf-8"
    )
    return json.loads(schema_text)


def load_policy_artifact_schema(artifact_type: str) -> Mapping[str, Any]:
    """Return the JSON schema fragment for *artifact_type*."""

    artifact_type = artifact_type.strip().lower()
    if artifact_type not in POLICY_ARTIFACT_TYPES:
        raise KeyError(f"Unsupported policy artifact type: {artifact_type}")

    schema = _read_policy_artifact_schema()
    definitions = schema.get("definitions", {})
    fragment = definitions.get(artifact_type)
    if not fragment:
        raise KeyError(f"Schema does not define type {artifact_type}")

    base: dict[str, Any] = {
        "$schema": schema.get(
            "$schema", "https://json-schema.org/draft/2020-12/schema"
        ),
    }
    base.update(fragment)
    return base


def validate_policy_artifact_payload(
    artifact_type: str, payload: Mapping[str, Any]
) -> None:
    """Validate *payload* against the JSON schema for *artifact_type*."""

    schema = load_policy_artifact_schema(artifact_type)
    if Draft202012Validator is not None:
        Draft202012Validator(schema).validate(dict(payload))
        return

    _manual_schema_validate(schema, dict(payload))


def merge_policy_sections(
    base: Mapping[str, Any] | PolicyDefinitions | None,
    override: Mapping[str, Any] | PolicyDefinitions | None,
) -> dict[str, Any]:
    """Return a merged dictionary representing combined policy definitions."""

    base_model = model_validate(PolicyDefinitions, base or {})
    override_model = model_validate(PolicyDefinitions, override or {})
    merged = base_model.merge(override_model)
    return _dump_model(merged)


def _dump_model(model: BaseModel) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        try:
            return dumper(mode="python")
        except TypeError:
            return dumper()
    return model.dict()


def _manual_schema_validate(
    schema: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Policy artifact payload must be a mapping")

    required = schema.get("required", [])
    for field in required:
        if field not in payload:
            raise ValueError(f"Missing required field '{field}'")

    properties: Mapping[str, Any] = schema.get("properties", {})
    additional_allowed = schema.get("additionalProperties", True)
    if not additional_allowed:
        for field in payload.keys():
            if field not in properties:
                raise ValueError(f"Unsupported field '{field}' for policy artifact")

    for field, spec in properties.items():
        if field not in payload:
            continue
        _validate_schema_value(field, spec, payload[field])


def _validate_schema_value(field: str, spec: Mapping[str, Any], value: Any) -> None:
    schema_type = spec.get("type")
    if schema_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"Field '{field}' must be a string")
        if "minLength" in spec and len(value) < spec["minLength"]:
            raise ValueError(
                f"Field '{field}' must be at least {spec['minLength']} characters"
            )
        if "maxLength" in spec and len(value) > spec["maxLength"]:
            raise ValueError(
                f"Field '{field}' must be at most {spec['maxLength']} characters"
            )
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"Field '{field}' must be one of {sorted(spec['enum'])}")
        if spec.get("format") == "date-time":
            _ensure_datetime(field, value)
        return

    if schema_type == "number":
        if not isinstance(value, (int, float)):
            raise ValueError(f"Field '{field}' must be numeric")
        numeric = float(value)
        if "minimum" in spec and numeric < spec["minimum"]:
            raise ValueError(
                f"Field '{field}' must be greater than or equal to {spec['minimum']}"
            )
        if "maximum" in spec and numeric > spec["maximum"]:
            raise ValueError(
                f"Field '{field}' must be less than or equal to {spec['maximum']}"
            )
        if "exclusiveMinimum" in spec and numeric <= spec["exclusiveMinimum"]:
            raise ValueError(
                f"Field '{field}' must be greater than {spec['exclusiveMinimum']}"
            )
        return

    if schema_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"Field '{field}' must be an array")
        if "minItems" in spec and len(value) < spec["minItems"]:
            raise ValueError(
                f"Field '{field}' must contain at least {spec['minItems']} items"
            )
        if spec.get("uniqueItems") and len(set(value)) != len(value):
            raise ValueError(f"Field '{field}' must contain unique items")
        item_spec = spec.get("items")
        if isinstance(item_spec, Mapping):
            for index, item in enumerate(value):
                _validate_schema_value(f"{field}[{index}]", item_spec, item)
        return

    if schema_type == "object":
        if not isinstance(value, Mapping):
            raise ValueError(f"Field '{field}' must be an object")
        _manual_schema_validate(spec, value)
        return

    if schema_type is None and "enum" in spec:
        if value not in spec["enum"]:
            raise ValueError(f"Field '{field}' must be one of {sorted(spec['enum'])}")
        return


def _ensure_datetime(field: str, value: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Field '{field}' must be ISO-8601 date-time") from exc


__all__ = [
    "EscalationContact",
    "NamespacePrivacyFloor",
    "PolicyArtifactDefinition",
    "PolicyDefinitions",
    "PolicyOverride",
    "RetentionCeiling",
    "POLICY_ARTIFACT_TYPES",
    "load_policy_artifact_schema",
    "merge_policy_sections",
    "validate_policy_artifact_payload",
]
