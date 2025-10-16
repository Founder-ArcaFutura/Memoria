"""Governance helpers shared by CLI commands."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from memoria.config.manager import ConfigManager
from memoria.policy.roster import apply_rotation_metadata, verify_escalation_contacts
from memoria.policy.schemas import EscalationContact


def _ensure_manager(manager: ConfigManager | None = None) -> ConfigManager:
    if isinstance(manager, ConfigManager):
        return manager
    return ConfigManager.get_instance()


def load_escalation_contacts(
    manager: ConfigManager | None = None,
) -> list[EscalationContact]:
    """Return escalation contacts from the active configuration."""

    manager = _ensure_manager(manager)
    settings = manager.get_settings()
    policy_settings = getattr(settings, "policy", None)
    contacts = getattr(policy_settings, "escalation_contacts", None)
    if not contacts:
        return []
    return list(contacts)


def persist_escalation_contacts(
    contacts: Iterable[EscalationContact], manager: ConfigManager | None = None
) -> None:
    """Persist escalation contacts back to configuration storage."""

    manager = _ensure_manager(manager)
    payload = [contact.dict() for contact in contacts]
    manager.update_setting("policy.escalation_contacts", payload)


def build_roster_verification(
    *,
    cadence_minutes: int = 60,
    reference_time: datetime | None = None,
    manager: ConfigManager | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return verification payload and flattened rows for CSV export."""

    manager = _ensure_manager(manager)
    contacts = load_escalation_contacts(manager)
    payload = verify_escalation_contacts(
        contacts,
        reference_time=reference_time,
        cadence_minutes=cadence_minutes,
    )
    rows = [
        _flatten_verification_record(record) for record in payload.get("contacts", [])
    ]
    return payload, rows


@dataclass(slots=True)
class RotationReport:
    payload: dict[str, Any]
    rows: list[dict[str, Any]]
    updated_contacts: list[EscalationContact]
    metadata_updates: int
    overdue_contacts: int


def build_rotation_report(
    *,
    cadence_minutes: int = 60,
    reference_time: datetime | None = None,
    persist: bool = False,
    manager: ConfigManager | None = None,
) -> RotationReport:
    """Apply rotation metadata and build audit payload suitable for export."""

    manager = _ensure_manager(manager)
    contacts = load_escalation_contacts(manager)
    reference = reference_time or datetime.now(timezone.utc)

    updated_contacts: list[EscalationContact] = []
    records: list[dict[str, Any]] = []
    metadata_updates = 0
    overdue_contacts = 0

    for contact in contacts:
        updated_contact, record = apply_rotation_metadata(
            contact, reference_time=reference
        )
        updated_contacts.append(updated_contact)
        if record.get("metadata_updated"):
            metadata_updates += 1
        if record.get("overdue_windows"):
            overdue_contacts += 1
        records.append(record)

    if persist and metadata_updates:
        persist_escalation_contacts(updated_contacts, manager)

    cadence_minutes = max(int(cadence_minutes or 60), 5)
    payload = {
        "generated_at": reference.replace(microsecond=0)
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "next_check_at": (reference + timedelta(minutes=cadence_minutes))
        .replace(microsecond=0)
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "cadence_minutes": cadence_minutes,
        "summary": {
            "total_contacts": len(contacts),
            "metadata_updates": metadata_updates,
            "overdue_contacts": overdue_contacts,
        },
        "contacts": records,
    }
    rows = [_flatten_rotation_record(record) for record in records]
    return RotationReport(
        payload=payload,
        rows=rows,
        updated_contacts=updated_contacts,
        metadata_updates=metadata_updates,
        overdue_contacts=overdue_contacts,
    )


def _join_list(values: Sequence[Any] | None) -> str:
    if not values:
        return ""
    return "; ".join(str(item) for item in values if item not in (None, ""))


def _flatten_verification_record(record: Mapping[str, Any]) -> dict[str, Any]:
    namespaces = _join_list(record.get("namespaces"))
    triggers = _join_list(record.get("triggers"))
    issues = record.get("issues") or []
    rotation = record.get("rotation") or []
    next_rotation = record.get("next_rotation")

    return {
        "name": record.get("name"),
        "channel": record.get("channel"),
        "target": record.get("target"),
        "priority": record.get("priority"),
        "status": record.get("status"),
        "issue_count": len(issues),
        "issues": _join_list(issues),
        "coverage": record.get("coverage"),
        "namespaces": namespaces,
        "triggers": triggers,
        "next_rotation": next_rotation,
        "rotation_entries": _join_list(
            [entry.get("date") for entry in rotation if isinstance(entry, Mapping)]
        ),
        "integrations": _join_list(
            [
                f"{entry.get('type')}: {entry.get('status')}"
                for entry in record.get("integrations") or []
                if isinstance(entry, Mapping)
            ]
        ),
    }


def _extract_rotation_slot(
    payload: Mapping[str, Any] | None,
) -> tuple[str | None, str | None, str | None]:
    if not isinstance(payload, Mapping):
        return (None, None, None)
    return (
        str(payload.get("primary")) if payload.get("primary") is not None else None,
        str(payload.get("secondary")) if payload.get("secondary") is not None else None,
        str(payload.get("date")) if payload.get("date") is not None else None,
    )


def _flatten_rotation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    active_primary, active_secondary, active_date = _extract_rotation_slot(
        record.get("active_rotation")
    )
    next_primary, next_secondary, next_date = _extract_rotation_slot(
        record.get("next_rotation")
    )
    history = record.get("history") or []

    return {
        "name": record.get("name"),
        "channel": record.get("channel"),
        "target": record.get("target"),
        "checked_at": record.get("checked_at"),
        "active_primary": active_primary,
        "active_secondary": active_secondary,
        "active_date": active_date,
        "next_primary": next_primary,
        "next_secondary": next_secondary,
        "next_date": next_date,
        "overdue_windows": record.get("overdue_windows"),
        "invalid_entries": record.get("invalid_entries"),
        "rotation_count": record.get("rotation_count"),
        "metadata_updated": record.get("metadata_updated"),
        "history_entries": _join_list(
            [entry.get("date") for entry in history if isinstance(entry, Mapping)]
        ),
    }


__all__ = [
    "RotationReport",
    "build_roster_verification",
    "build_rotation_report",
    "load_escalation_contacts",
    "persist_escalation_contacts",
]
