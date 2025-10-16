import json

import pytest

try:  # pragma: no cover - optional dependency
    from jsonschema.exceptions import ValidationError
except Exception:  # pragma: no cover - fallback when jsonschema isn't installed
    ValidationError = ValueError  # type: ignore[assignment]

from memoria.config.manager import ConfigManager
from memoria.config.settings import MemoriaSettings
from memoria.policy.schemas import (
    NamespacePrivacyFloor,
    PolicyDefinitions,
    PolicyOverride,
    RetentionCeiling,
    validate_policy_artifact_payload,
)


def test_policy_definitions_merge_overrides() -> None:
    base = PolicyDefinitions(
        retention_ceilings=[RetentionCeiling(name="default", max_days=30)],
        overrides=[PolicyOverride(name="temp-allow")],
    )
    override = PolicyDefinitions(
        retention_ceilings=[RetentionCeiling(name="default", max_days=45)],
        namespace_privacy_floors=[
            NamespacePrivacyFloor(name="public", privacy_floor=-5.0)
        ],
    )

    merged = base.merge(override)

    assert len(merged.retention_ceilings) == 1
    assert merged.retention_ceilings[0].max_days == 45
    assert merged.namespace_privacy_floors[0].name == "public"
    assert merged.namespace_privacy_floors[0].privacy_floor == -5.0


def test_validate_policy_artifact_payload_accepts_valid_data() -> None:
    payload = {
        "name": "default",
        "namespaces": ["*"],
        "max_days": 30,
        "metadata": {"owner": "security"},
    }

    validate_policy_artifact_payload("retention_ceiling", payload)


def test_validate_policy_artifact_payload_rejects_invalid_data() -> None:
    payload = {"name": "bad", "max_days": -1}

    with pytest.raises(ValidationError):
        validate_policy_artifact_payload("retention_ceiling", payload)


def test_memoria_settings_policy_string_coercion() -> None:
    payload = json.dumps(
        {
            "escalation_contacts": [
                {
                    "name": "pagerduty",
                    "channel": "email",
                    "target": "secops@example.com",
                }
            ]
        }
    )

    settings = MemoriaSettings(policy=payload)

    assert settings.policy.escalation_contacts[0].name == "pagerduty"
    assert settings.policy.escalation_contacts[0].channel == "email"


def test_config_manager_merges_policy_definitions() -> None:
    ConfigManager._instance = None
    manager = ConfigManager()
    base = MemoriaSettings(
        policy={"retention_ceilings": [{"name": "default", "max_days": 30}]}
    )
    manager._settings = base

    override = MemoriaSettings(
        policy={
            "namespace_privacy_floors": [{"name": "public", "privacy_floor": -4.0}],
            "retention_ceilings": [{"name": "default", "max_days": 60}],
        }
    )
    manager._merge_settings(override)

    merged = manager.get_settings()
    assert merged.policy.retention_ceilings[0].max_days == 60
    assert merged.policy.namespace_privacy_floors[0].privacy_floor == -4.0

    ConfigManager._instance = None
