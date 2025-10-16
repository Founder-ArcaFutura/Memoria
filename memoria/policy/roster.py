"""Roster verification utilities for escalation contacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from .schemas import EscalationContact

_STATUS_LABELS = {0: "ok", 1: "warning", 2: "error"}


def _ensure_utc(reference: datetime | None = None) -> datetime:
    reference = _utc_now(reference)
    return reference


def _isoformat_utc(value: datetime | None) -> str | None:
    return _isoformat(value)


def _utc_now(reference: datetime | None = None) -> datetime:
    if reference is None:
        return datetime.now(timezone.utc)
    if reference.tzinfo is None:
        return reference.replace(tzinfo=timezone.utc)
    return reference.astimezone(timezone.utc)


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _ensure_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [segment.strip() for segment in value.split(",") if segment.strip()]
    if isinstance(value, Sequence):
        cleaned: list[str] = []
        for item in value:
            if item in (None, ""):
                continue
            cleaned.append(str(item).strip())
        return cleaned
    return []


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalise_rotation_entries(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    rotation = metadata.get("rotation")
    if not isinstance(rotation, Sequence):
        return []
    entries: list[dict[str, Any]] = []
    for raw in rotation:
        if not isinstance(raw, Mapping):
            continue
        entry: dict[str, Any] = {
            "primary": raw.get("primary"),
            "secondary": raw.get("secondary"),
        }
        parsed = _parse_datetime(raw.get("date"))
        entry["date"] = _isoformat(parsed)
        entry["raw_date"] = raw.get("date")
        entries.append(entry)
    entries.sort(key=lambda item: item.get("date") or "")
    return entries


def _collect_rotation_windows(
    metadata: Mapping[str, Any],
) -> tuple[list[tuple[datetime, dict[str, Any]]], int]:
    entries: list[tuple[datetime, dict[str, Any]]] = []
    invalid = 0
    for raw in _normalise_rotation_entries(metadata):
        parsed = _parse_datetime(raw.get("raw_date") or raw.get("date"))
        if parsed is None:
            invalid += 1
            continue
        payload = {
            "date": _isoformat(parsed),
            "primary": raw.get("primary"),
            "secondary": raw.get("secondary"),
        }
        entries.append((parsed, payload))
    entries.sort(key=lambda item: item[0])
    return entries, invalid


def apply_rotation_metadata(
    contact: EscalationContact, *, reference_time: datetime | None = None
) -> tuple[EscalationContact, dict[str, Any]]:
    """Update roster metadata to reflect the active and next rotation windows."""

    reference = _ensure_utc(reference_time)
    metadata: Mapping[str, Any] = contact.metadata or {}
    windows, invalid_entries = _collect_rotation_windows(metadata)

    history: list[dict[str, Any]] = []
    upcoming: list[dict[str, Any]] = []
    for window_dt, payload in windows:
        if window_dt <= reference:
            history.append(payload)
        else:
            upcoming.append(payload)

    active_window = history[-1] if history else None
    completed_windows = history[:-1] if len(history) > 1 else []
    overdue_windows = max(0, len(history) - (1 if active_window else 0))
    next_window = upcoming[0] if upcoming else None

    condensed_rotation: list[dict[str, Any]] = []
    if active_window:
        condensed_rotation.append(active_window)
    condensed_rotation.extend(upcoming)

    metadata_dict = dict(metadata)
    changed = False

    def _assign(key: str, value: Any) -> None:
        nonlocal changed
        if value is None:
            if key in metadata_dict:
                metadata_dict.pop(key, None)
                changed = True
            return
        if metadata_dict.get(key) != value:
            metadata_dict[key] = value
            changed = True

    if metadata_dict.get("rotation") != condensed_rotation:
        metadata_dict["rotation"] = condensed_rotation
        changed = True

    _assign("rotation_active", active_window)
    _assign("rotation_next", next_window)
    truncated_history = completed_windows[-5:] if completed_windows else []
    _assign("rotation_history", truncated_history if truncated_history else None)
    _assign("rotation_overdue", bool(overdue_windows))
    check_timestamp = _isoformat_utc(reference)
    _assign("rotation_checked_at", check_timestamp)
    if invalid_entries:
        _assign("rotation_invalid_entries", invalid_entries)
    elif "rotation_invalid_entries" in metadata_dict:
        metadata_dict.pop("rotation_invalid_entries", None)
        changed = True

    updated_contact = (
        contact if not changed else contact.copy(update={"metadata": metadata_dict})
    )

    audit_payload = {
        "name": contact.name,
        "channel": contact.channel,
        "target": contact.target,
        "checked_at": check_timestamp,
        "active_rotation": active_window,
        "next_rotation": next_window,
        "overdue_windows": overdue_windows,
        "invalid_entries": invalid_entries,
        "history": truncated_history,
        "rotation_count": len(windows),
        "metadata_updated": changed,
    }

    return updated_contact, audit_payload


def _normalise_integrations(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    integrations = metadata.get("integrations")
    if isinstance(integrations, Mapping):
        integrations = [integrations]
    if not isinstance(integrations, Sequence):
        return []
    normalised: list[dict[str, Any]] = []
    for raw in integrations:
        if not isinstance(raw, Mapping):
            continue
        entry = {
            "type": str(raw.get("type") or "").strip() or "unknown",
            "target": str(raw.get("target") or "").strip(),
            "status": str(raw.get("status") or "").strip().lower() or "unknown",
            "metadata": (
                raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else {}
            ),
        }
        normalised.append(entry)
    return normalised


def _evaluate_contact(
    contact: EscalationContact, *, reference: datetime, cadence: timedelta
) -> dict[str, Any]:
    metadata: Mapping[str, Any] = contact.metadata or {}
    status_level = 0
    issues: list[str] = []

    namespaces = _ensure_list(metadata.get("namespaces"))
    if not namespaces:
        status_level = max(status_level, 1)
        issues.append("No namespaces assigned to escalation queue")

    triggers = _ensure_list(metadata.get("triggers"))
    if not triggers:
        status_level = max(status_level, 1)
        issues.append("No triggers configured for escalation queue")

    coverage = str(metadata.get("coverage") or "").strip()
    if not coverage:
        status_level = max(status_level, 1)
        issues.append("Coverage window not documented")

    rotation_entries = _normalise_rotation_entries(metadata)
    next_rotation = None
    overdue_windows = 0
    if rotation_entries:
        for entry in rotation_entries:
            parsed = _parse_datetime(entry.get("date"))
            if parsed is None:
                status_level = max(status_level, 1)
                issues.append("Rotation entry missing a valid date")
                continue
            if parsed + cadence < reference:
                overdue_windows += 1
                continue
            next_rotation = parsed
            break
        if overdue_windows:
            status_level = max(
                status_level, 1 if overdue_windows < len(rotation_entries) else 2
            )
            issues.append(f"{overdue_windows} rotation window(s) are past due")
        if next_rotation is None:
            status_level = max(status_level, 2)
            issues.append("No upcoming on-call coverage identified")
    else:
        status_level = max(status_level, 1)
        issues.append("No rotation schedule provided")

    sync_info = (
        metadata.get("sync") if isinstance(metadata.get("sync"), Mapping) else None
    )
    if sync_info:
        status = str(sync_info.get("status") or "").strip().lower()
        if status in {"error", "failed", "inactive"}:
            status_level = max(status_level, 2 if status == "failed" else 1)
            issues.append(f"Sync integration status: {status}")

    integrations = _normalise_integrations(metadata)
    for integration in integrations:
        integration_status = integration.get("status") or "unknown"
        if integration_status in {"error", "failed"}:
            status_level = max(status_level, 2)
            issues.append(
                f"Integration '{integration['type']}' reported status {integration_status}"
            )
        elif integration_status in {"degraded", "inactive", "unknown"}:
            status_level = max(status_level, 1)
            if integration_status != "unknown":
                issues.append(
                    f"Integration '{integration['type']}' reported status {integration_status}"
                )

    status_label = _STATUS_LABELS.get(status_level, "warning")

    return {
        "name": contact.name,
        "channel": contact.channel,
        "target": contact.target,
        "priority": contact.priority,
        "status": status_label,
        "issues": issues,
        "namespaces": namespaces,
        "triggers": triggers,
        "coverage": coverage,
        "rotation": rotation_entries,
        "next_rotation": _isoformat(next_rotation),
        "integrations": integrations,
        "sync": {
            "provider": sync_info.get("provider") if sync_info else None,
            "status": (sync_info.get("status") if sync_info else None),
        },
    }


def verify_escalation_contacts(
    contacts: Iterable[EscalationContact],
    *,
    reference_time: datetime | None = None,
    cadence_minutes: int = 60,
) -> dict[str, Any]:
    """Analyse the supplied contacts and return verification telemetry."""

    cadence_minutes = max(int(cadence_minutes or 60), 5)
    cadence_delta = timedelta(minutes=cadence_minutes)
    reference = _utc_now(reference_time)

    contact_results = [
        _evaluate_contact(contact, reference=reference, cadence=cadence_delta)
        for contact in contacts
    ]

    status_counts: dict[str, int] = {"ok": 0, "warning": 0, "error": 0}
    total_issues = 0
    for result in contact_results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        total_issues += len(result.get("issues", []))

    payload = {
        "generated_at": _isoformat(reference),
        "next_check_at": _isoformat(reference + cadence_delta),
        "cadence_minutes": cadence_minutes,
        "summary": {
            "total_contacts": len(contact_results),
            "status_counts": status_counts,
            "total_issues": total_issues,
        },
        "contacts": contact_results,
    }
    return payload


__all__ = ["verify_escalation_contacts", "apply_rotation_metadata"]
