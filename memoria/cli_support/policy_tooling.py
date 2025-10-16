"""Helper utilities backing the CLI policy commands."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from memoria.config.manager import ConfigManager
from memoria.config.settings import RetentionPolicyRuleSettings
from memoria.policy.utils import (
    PolicyConfigurationError,
    load_policy_document,
    sanitise_timestamp,
)


@dataclass(slots=True)
class SimulationHit:
    """Represents a policy trigger during simulation."""

    sample_index: int
    memory_id: str | None
    reasons: tuple[str, ...]


@dataclass(slots=True)
class SimulationReport:
    """Aggregated simulation results for a single policy rule."""

    rule: RetentionPolicyRuleSettings
    hits: tuple[SimulationHit, ...]
    total_samples: int

    @property
    def trigger_count(self) -> int:
        return len(self.hits)


def _normalise_rule_payload(rule: RetentionPolicyRuleSettings) -> dict[str, object]:
    payload = rule.dict()
    action = payload.get("action")
    if hasattr(action, "value"):
        payload["action"] = action.value
    return payload


def load_policies_for_cli(path: str | Path) -> list[RetentionPolicyRuleSettings]:
    return load_policy_document(path, allow_stdin=True)


def lint_policies(
    policies: Sequence[RetentionPolicyRuleSettings],
) -> list[str]:
    """Return warnings discovered during linting."""

    warnings: list[str] = []
    seen: dict[str, RetentionPolicyRuleSettings] = {}
    for rule in policies:
        key = rule.name.strip().lower()
        if key in seen:
            warnings.append(
                f"Duplicate rule name '{rule.name}' also defined in '{seen[key].name}'"
            )
        else:
            seen[key] = rule
        if not rule.namespaces:
            warnings.append(
                f"Rule '{rule.name}' has no namespaces; default '*' assumed"
            )
        if rule.escalate_to and rule.action.value != "escalate":
            warnings.append(
                f"Rule '{rule.name}' defines escalate_to but action is '{rule.action.value}'"
            )
    return warnings


def _load_samples(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".ndjson":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    raise PolicyConfigurationError(
        "Sample payloads must be a JSON object, array, or NDJSON stream"
    )


def simulate_policies(
    policies: Sequence[RetentionPolicyRuleSettings],
    *,
    sample_path: Path | None = None,
) -> list[SimulationReport]:
    if not policies:
        return []

    samples: list[dict[str, object]] = []
    if sample_path is not None:
        samples = _load_samples(sample_path)

    if not samples:
        return [
            SimulationReport(rule=rule, hits=(), total_samples=0)
            for rule in policies
        ]

    from memoria.policy.utils import settings_to_runtime

    runtime_rules = settings_to_runtime(policies)

    reports: list[SimulationReport] = []
    now = max(
        [
            sanitise_timestamp(str(sample.get("reference_time")))
            for sample in samples
            if sample.get("reference_time") is not None
        ]
        or [None]
    )

    for idx, rule in enumerate(runtime_rules):
        hits: list[SimulationHit] = []
        for index, sample in enumerate(samples, start=1):
            namespace = str(sample.get("namespace") or "default")
            if not rule.matches(namespace):
                continue

            reasons: list[str] = []

            privacy = None
            if "y_coord" in sample:
                y_candidate = sample.get("y_coord")
                if y_candidate is not None:
                    privacy = y_candidate
            elif "privacy" in sample:
                privacy = sample.get("privacy")
            if (
                rule.privacy_ceiling is not None
                and privacy is not None
                and float(privacy) > float(rule.privacy_ceiling)
            ):
                reasons.append("privacy_ceiling")

            importance = None
            if "importance_score" in sample:
                importance_candidate = sample.get("importance_score")
                if importance_candidate is not None:
                    importance = importance_candidate
            elif "importance" in sample:
                importance = sample.get("importance")
            if (
                rule.importance_floor is not None
                and importance is not None
                and float(importance) < float(rule.importance_floor)
            ):
                reasons.append("importance_floor")

            lifecycle_days = rule.lifecycle_days
            if lifecycle_days is not None:
                created = None
                for key in ("last_accessed", "created_at", "timestamp"):
                    candidate = (
                        sanitise_timestamp(sample.get(key)) if sample.get(key) else None
                    )
                    if candidate is not None:
                        created = candidate
                        break
                reference = sanitise_timestamp(sample.get("reference_time")) or now
                if created is not None and reference is not None:
                    age_days = (reference - created).total_seconds() / 86400
                    if age_days > float(lifecycle_days):
                        reasons.append("lifecycle_days")

            if reasons:
                hits.append(
                    SimulationHit(
                        sample_index=index,
                        memory_id=(
                            str(sample.get("memory_id"))
                            if sample.get("memory_id")
                            else None
                        ),
                        reasons=tuple(reasons),
                    )
                )
        reports.append(
            SimulationReport(
                rule=policies[idx],
                hits=tuple(hits),
                total_samples=len(samples),
            )
        )

    return reports


def apply_policies(
    policies: Iterable[RetentionPolicyRuleSettings],
    *,
    dry_run: bool = False,
) -> None:
    manager = ConfigManager.get_instance()
    policy_list = list(policies)
    serialised = [_normalise_rule_payload(rule) for rule in policy_list]

    if dry_run:
        return

    manager.update_setting("memory.retention_policy_rules", serialised)


__all__ = [
    "SimulationHit",
    "SimulationReport",
    "apply_policies",
    "lint_policies",
    "load_policies_for_cli",
    "simulate_policies",
]
