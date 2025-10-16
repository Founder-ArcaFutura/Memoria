"""Utility helpers for loading and serialising retention policy rules."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from memoria.config.settings import RetentionPolicyRuleSettings
from memoria.heuristics.retention import RetentionConfig, RetentionPolicyRule
from memoria.utils.exceptions import MemoriaError


class PolicyConfigurationError(MemoriaError):
    """Raised when a policy document cannot be parsed or validated."""


def _load_yaml(text: str) -> object:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise PolicyConfigurationError(
            "PyYAML is required to load .yml or .yaml policy files"
        ) from exc
    return yaml.safe_load(text)


def _coerce_policy_items(payload: object) -> list[Mapping[str, object]]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        if "policies" in payload and isinstance(payload["policies"], list):
            return [item for item in payload["policies"] if isinstance(item, Mapping)]
        if "rules" in payload and isinstance(payload["rules"], list):
            return [item for item in payload["rules"] if isinstance(item, Mapping)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    raise PolicyConfigurationError(
        "Policy documents must define an object or list of rule objects"
    )


def load_policy_document(
    source: str | Path,
    *,
    allow_stdin: bool = False,
    raw_text: str | None = None,
) -> list[RetentionPolicyRuleSettings]:
    """Load and validate policy rules from disk or stdin."""

    if isinstance(source, Path):
        path = source
    else:
        path = Path(str(source))

    if raw_text is None:
        if str(path) == "-" and allow_stdin:
            raw_text = sys.stdin.read()
        else:
            try:
                raw_text = path.read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                raise PolicyConfigurationError(
                    f"Policy file not found: {path}"
                ) from exc

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = _load_yaml(raw_text or "")
    elif suffix == ".ndjson":
        payload = [json.loads(line) for line in (raw_text or "").splitlines() if line]
    else:
        try:
            payload = json.loads(raw_text or "{}")
        except json.JSONDecodeError as exc:
            raise PolicyConfigurationError(
                f"Invalid JSON in policy file: {exc}"
            ) from exc

    items = _coerce_policy_items(payload)
    rules: list[RetentionPolicyRuleSettings] = []
    errors: list[str] = []
    for index, item in enumerate(items, start=1):
        try:
            rules.append(RetentionPolicyRuleSettings(**dict(item)))
        except ValidationError as exc:
            location = item.get("name") if isinstance(item, Mapping) else f"#{index}"
            errors.append(f"{location}: {exc}")
    if errors:
        joined = "; ".join(errors)
        raise PolicyConfigurationError(f"Invalid policy definitions: {joined}")

    return rules


def serialise_policy(
    rule: RetentionPolicyRuleSettings,
    *,
    include_sensitive: bool = True,
) -> dict[str, object]:
    """Return a JSON serialisable representation of a policy rule."""

    payload: dict[str, object] = {
        "name": rule.name,
        "namespaces": list(rule.namespaces or ["*"]),
        "action": (
            rule.action.value if hasattr(rule.action, "value") else str(rule.action)
        ),
    }
    if rule.privacy_ceiling is not None:
        payload["privacy_ceiling"] = rule.privacy_ceiling
    if rule.importance_floor is not None:
        payload["importance_floor"] = rule.importance_floor
    if rule.lifecycle_days is not None:
        payload["lifecycle_days"] = rule.lifecycle_days
    if include_sensitive and rule.escalate_to:
        payload["escalate_to"] = rule.escalate_to
    if include_sensitive and rule.metadata:
        payload["metadata"] = dict(rule.metadata)
    if not include_sensitive and rule.escalate_to:
        payload["has_escalation"] = True
    if not include_sensitive and rule.metadata:
        payload["metadata_keys"] = sorted(rule.metadata.keys())
    return payload


def settings_to_runtime(
    policies: Sequence[RetentionPolicyRuleSettings],
) -> tuple[RetentionPolicyRule, ...]:
    """Convert pydantic policy settings into runtime policy objects."""

    converted: list[RetentionPolicyRule] = []
    for rule in policies:
        action_value = getattr(rule.action, "value", rule.action)
        converted.append(
            RetentionPolicyRule(
                name=rule.name,
                namespaces=tuple(rule.namespaces or ("*",)),
                privacy_ceiling=rule.privacy_ceiling,
                importance_floor=rule.importance_floor,
                lifecycle_days=rule.lifecycle_days,
                action=str(action_value),
                escalate_to=rule.escalate_to,
                metadata=dict(rule.metadata or {}),
            )
        )
    return tuple(converted)


def apply_runtime_policies(
    retention_service: object | None,
    storage_service: object | None,
    policies: Sequence[RetentionPolicyRuleSettings],
) -> None:
    """Refresh runtime services after policy updates."""

    runtime_rules = settings_to_runtime(policies)
    if storage_service is not None and hasattr(
        storage_service, "configure_retention_policies"
    ):
        storage_service.configure_retention_policies(runtime_rules)

    if retention_service is None:
        return

    config = getattr(retention_service, "config", None)
    if isinstance(config, RetentionConfig):
        updated = replace(config, policies=runtime_rules)
        retention_service.config = updated
    if hasattr(retention_service, "_policies"):
        retention_service._policies = runtime_rules


def sanitise_timestamp(value: str | None) -> datetime | None:
    """Parse ISO-8601 timestamps that may lack timezone information."""

    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


__all__ = [
    "PolicyConfigurationError",
    "apply_runtime_policies",
    "load_policy_document",
    "sanitise_timestamp",
    "serialise_policy",
    "settings_to_runtime",
]
