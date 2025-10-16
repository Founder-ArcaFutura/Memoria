"""Governance-focused API routes for the admin dashboard."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
import queue
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from pydantic import ValidationError
from sqlalchemy.orm import Session

from memoria.config.manager import ConfigManager
from memoria.config.settings import RetentionPolicyRuleSettings
from memoria.database.models import DatabaseManager, RetentionPolicyAudit
from memoria.policy.enforcement import PolicyEnforcementEngine, PolicyMetricsSnapshot
from memoria.policy.roster import verify_escalation_contacts
from memoria.policy.schemas import (
    EscalationContact,
    NamespacePrivacyFloor,
    PolicyOverride,
    apply_override_expiry,
    summarise_override_window,
)
from memoria.policy.utils import (
    PolicyConfigurationError,
    sanitise_timestamp,
    serialise_policy,
    settings_to_runtime,
)

from .scheduler import get_service_metadata_value, set_service_metadata_value


governance_bp = Blueprint("governance", __name__)

_ROSTER_METADATA_KEY = "roster_verification_status"
_ROSTER_CONFIG_KEY = "ROSTER_VERIFICATION_STATUS"
_ROTATION_METADATA_KEY = "roster_rotation_status"
_ROTATION_CONFIG_KEY = "ROSTER_ROTATION_STATUS"
_OVERRIDE_METADATA_KEY = "override_expiry_status"
_OVERRIDE_CONFIG_KEY = "OVERRIDE_EXPIRY_STATUS"


def _get_config_manager() -> ConfigManager:
    manager = current_app.config.get("config_manager")
    if not isinstance(manager, ConfigManager):  # pragma: no cover - configuration guard
        raise RuntimeError("Config manager unavailable")
    return manager


def _get_database_manager() -> DatabaseManager | None:
    memoria_app = current_app.config.get("memoria")
    return getattr(memoria_app, "db_manager", None)


def _load_policy_rules(manager: ConfigManager) -> list[RetentionPolicyRuleSettings]:
    settings = manager.get_settings()
    rules = getattr(getattr(settings, "memory", None), "retention_policy_rules", None)
    return list(rules or [])


def _load_policy_definitions(manager: ConfigManager) -> Any:
    settings = manager.get_settings()
    return getattr(settings, "policy", None)


def _serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _persist_roster_status(payload: dict[str, Any]) -> None:
    current_app.config[_ROSTER_CONFIG_KEY] = payload
    try:
        set_service_metadata_value(current_app, _ROSTER_METADATA_KEY, json.dumps(payload))
    except Exception:  # pragma: no cover - defensive log guard
        current_app.logger.exception("Failed to persist roster verification status")


def _persist_rotation_status(payload: dict[str, Any]) -> None:
    current_app.config[_ROTATION_CONFIG_KEY] = payload
    try:
        set_service_metadata_value(current_app, _ROTATION_METADATA_KEY, json.dumps(payload))
    except Exception:  # pragma: no cover - defensive log guard
        current_app.logger.exception("Failed to persist roster rotation status")


def _persist_override_status(payload: dict[str, Any]) -> None:
    current_app.config[_OVERRIDE_CONFIG_KEY] = payload
    try:
        set_service_metadata_value(current_app, _OVERRIDE_METADATA_KEY, json.dumps(payload))
    except Exception:  # pragma: no cover - defensive log guard
        current_app.logger.exception("Failed to persist override expiry status")


def _load_roster_status() -> dict[str, Any] | None:
    cached = current_app.config.get(_ROSTER_CONFIG_KEY)
    if isinstance(cached, dict):
        return cached
    stored = get_service_metadata_value(current_app, _ROSTER_METADATA_KEY)
    if not stored:
        return None
    try:
        payload = json.loads(stored)
    except json.JSONDecodeError:
        return None
    current_app.config[_ROSTER_CONFIG_KEY] = payload
    return payload if isinstance(payload, dict) else None


def _load_rotation_status() -> dict[str, Any] | None:
    cached = current_app.config.get(_ROTATION_CONFIG_KEY)
    if isinstance(cached, dict):
        return cached
    stored = get_service_metadata_value(current_app, _ROTATION_METADATA_KEY)
    if not stored:
        return None
    try:
        payload = json.loads(stored)
    except json.JSONDecodeError:
        return None
    current_app.config[_ROTATION_CONFIG_KEY] = payload
    return payload if isinstance(payload, dict) else None


def _load_override_status() -> dict[str, Any] | None:
    cached = current_app.config.get(_OVERRIDE_CONFIG_KEY)
    if isinstance(cached, dict):
        return cached
    stored = get_service_metadata_value(current_app, _OVERRIDE_METADATA_KEY)
    if not stored:
        return None
    try:
        payload = json.loads(stored)
    except json.JSONDecodeError:
        return None
    current_app.config[_OVERRIDE_CONFIG_KEY] = payload
    return payload if isinstance(payload, dict) else None


def _parse_bool(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _normalize_owner_entries(metadata: Any) -> Iterable[str]:
    if not metadata:
        return []
    owners = []
    owner_value = metadata.get("owners") if isinstance(metadata, dict) else None
    if owner_value is None and isinstance(metadata, dict):
        owner_value = metadata.get("owner")
    if owner_value is None:
        return []
    if isinstance(owner_value, str):
        cleaned = owner_value.strip()
        if cleaned:
            owners.append(cleaned)
        return owners
    if isinstance(owner_value, (list, tuple, set)):
        for item in owner_value:
            if isinstance(item, str) and item.strip():
                owners.append(item.strip())
    return owners


def _summarise_namespaces(
    policies: list[RetentionPolicyRuleSettings],
    floors: list[NamespacePrivacyFloor],
    contacts: list[EscalationContact],
) -> list[dict[str, Any]]:
    namespace_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rule_count": 0,
            "escalations": set(),
            "owners": set(),
            "privacy_ceilings": [],
            "lifecycle_windows": [],
            "teams": set(),
            "workspaces": set(),
            "privacy_floors": [],
        }
    )

    for rule in policies:
        namespaces = list(rule.namespaces or ["*"])
        owners = set(_normalize_owner_entries(getattr(rule, "metadata", None)))
        metadata = getattr(rule, "metadata", None)
        teams = []
        workspaces = []
        if isinstance(metadata, dict):
            raw_teams = metadata.get("teams")
            if isinstance(raw_teams, str):
                teams = [segment.strip() for segment in raw_teams.split(",") if segment.strip()]
            elif isinstance(raw_teams, (list, tuple, set)):
                teams = [str(item).strip() for item in raw_teams if str(item).strip()]
            raw_workspaces = metadata.get("workspaces")
            if isinstance(raw_workspaces, str):
                workspaces = [segment.strip() for segment in raw_workspaces.split(",") if segment.strip()]
            elif isinstance(raw_workspaces, (list, tuple, set)):
                workspaces = [str(item).strip() for item in raw_workspaces if str(item).strip()]
        for namespace in namespaces:
            key = namespace or "*"
            bucket = namespace_data[key]
            bucket["rule_count"] += 1
            if rule.escalate_to:
                bucket["escalations"].add(rule.escalate_to)
            bucket["owners"].update(owners)
            if rule.privacy_ceiling is not None:
                bucket["privacy_ceilings"].append(float(rule.privacy_ceiling))
            if rule.lifecycle_days is not None:
                bucket["lifecycle_windows"].append(float(rule.lifecycle_days))
            bucket["teams"].update(teams)
            bucket["workspaces"].update(workspaces)

    for floor in floors:
        for namespace in floor.namespaces or ["*"]:
            bucket = namespace_data[namespace or "*"]
            if floor.privacy_floor is not None:
                bucket["privacy_floors"].append(float(floor.privacy_floor))

    contact_lookup: dict[str, set[str]] = defaultdict(set)
    for contact in contacts:
        namespaces = []
        if isinstance(contact.metadata, dict):
            raw = contact.metadata.get("namespaces")
            if isinstance(raw, str):
                namespaces = [segment.strip() for segment in raw.split(",") if segment.strip()]
            elif isinstance(raw, (list, tuple, set)):
                namespaces = [str(item).strip() for item in raw if str(item).strip()]
        if not namespaces:
            namespaces = ["*"]
        for namespace in namespaces:
            contact_lookup[namespace or "*"].add(contact.name)

    summaries: list[dict[str, Any]] = []
    for namespace, payload in namespace_data.items():
        privacy_ceilings = payload["privacy_ceilings"]
        lifecycle_windows = payload["lifecycle_windows"]
        def _privacy_band() -> str:
            if not privacy_ceilings:
                return "unbounded"
            minimum = min(privacy_ceilings)
            if minimum <= 3:
                return "strict"
            if minimum <= 7:
                return "balanced"
            return "permissive"

        def _lifecycle_band() -> str:
            if not lifecycle_windows:
                return "unbounded"
            minimum = min(lifecycle_windows)
            if minimum <= 7:
                return "short"
            if minimum <= 30:
                return "medium"
            return "extended"

        summaries.append(
            {
                "namespace": namespace,
                "rule_count": payload["rule_count"],
                "rule_density": min(payload["rule_count"] / 8.0, 1.0) if payload["rule_count"] else 0.0,
                "escalations": sorted(payload["escalations"]),
                "owners": sorted(payload["owners"]),
                "teams": sorted(payload["teams"]),
                "workspaces": sorted(payload["workspaces"]),
                "privacy_band": _privacy_band(),
                "lifecycle_band": _lifecycle_band(),
                "privacy_ceilings": privacy_ceilings,
                "lifecycle_windows": lifecycle_windows,
                "contacts": sorted(contact_lookup.get(namespace, set()) | contact_lookup.get("*", set())),
                "privacy_floors": payload["privacy_floors"],
            }
        )

    return sorted(summaries, key=lambda item: item["namespace"])


def _filter_segments(
    segments: list[dict[str, Any]],
    *,
    team: str | None,
    workspace: str | None,
    privacy_band: str | None,
    lifecycle_band: str | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for segment in segments:
        if team and team != "all" and team not in segment["teams"]:
            continue
        if workspace and workspace != "all" and workspace not in segment["workspaces"]:
            continue
        if privacy_band and privacy_band != "all" and segment["privacy_band"] != privacy_band:
            continue
        if lifecycle_band and lifecycle_band != "all" and segment["lifecycle_band"] != lifecycle_band:
            continue
        filtered.append(segment)
    return filtered


@governance_bp.route("/governance/namespaces", methods=["GET"])
def list_namespace_segments():
    manager = _get_config_manager()
    policies = _load_policy_rules(manager)
    definitions = _load_policy_definitions(manager)
    floors = list(getattr(definitions, "namespace_privacy_floors", []) or [])
    contacts = list(getattr(definitions, "escalation_contacts", []) or [])

    segments = _summarise_namespaces(
        policies,
        floors,
        contacts,
    )

    filter_team = request.args.get("team")
    filter_workspace = request.args.get("workspace")
    filter_privacy = request.args.get("privacy")
    filter_lifecycle = request.args.get("lifecycle")

    filtered_segments = _filter_segments(
        segments,
        team=filter_team,
        workspace=filter_workspace,
        privacy_band=filter_privacy,
        lifecycle_band=filter_lifecycle,
    )

    teams = sorted({team for segment in segments for team in segment["teams"]})
    workspaces = sorted({ws for segment in segments for ws in segment["workspaces"]})
    privacy_bands = sorted({segment["privacy_band"] for segment in segments})
    lifecycle_bands = sorted({segment["lifecycle_band"] for segment in segments})

    return jsonify(
        {
            "status": "ok",
            "namespaces": filtered_segments,
            "filters": {
                "teams": teams,
                "workspaces": workspaces,
                "privacy_bands": privacy_bands,
                "lifecycle_bands": lifecycle_bands,
            },
            "total": len(filtered_segments),
        }
    )


def _match_policies_to_namespace(
    namespace: str,
    policies: list[RetentionPolicyRuleSettings],
) -> list[RetentionPolicyRuleSettings]:
    runtime_rules = settings_to_runtime(policies)
    matched: list[RetentionPolicyRuleSettings] = []
    for runtime, settings_rule in zip(runtime_rules, policies):
        try:
            if runtime.matches(namespace):
                matched.append(settings_rule)
        except Exception:  # pragma: no cover - defensive guard
            namespaces = list(settings_rule.namespaces or [])
            if namespace in namespaces or "*" in namespaces:
                matched.append(settings_rule)
    return matched


@governance_bp.route("/governance/namespaces/<path:namespace>", methods=["GET"])
def describe_namespace(namespace: str):
    manager = _get_config_manager()
    policies = _load_policy_rules(manager)
    definitions = _load_policy_definitions(manager)
    namespace = namespace or "*"

    matching_rules = _match_policies_to_namespace(namespace, policies)

    floors = []
    contacts: list[EscalationContact] = []
    if definitions is not None:
        for floor in getattr(definitions, "namespace_privacy_floors", []) or []:
            if namespace in floor.namespaces or "*" in floor.namespaces:
                floors.append(
                    {
                        "name": floor.name,
                        "namespaces": list(floor.namespaces or []),
                        "privacy_floor": floor.privacy_floor,
                        "metadata": floor.metadata or {},
                    }
                )
        for contact in getattr(definitions, "escalation_contacts", []) or []:
            namespaces = []
            if isinstance(contact.metadata, dict):
                raw = contact.metadata.get("namespaces")
                if isinstance(raw, str):
                    namespaces = [segment.strip() for segment in raw.split(",") if segment.strip()]
                elif isinstance(raw, (list, tuple, set)):
                    namespaces = [str(item).strip() for item in raw if str(item).strip()]
            if not namespaces:
                namespaces = ["*"]
            if namespace in namespaces or "*" in namespaces:
                contacts.append(
                    {
                        "name": contact.name,
                        "channel": contact.channel,
                        "target": contact.target,
                        "priority": contact.priority,
                        "metadata": contact.metadata,
                    }
                )

    metadata_keys = sorted(
        {
            key
            for rule in matching_rules
            for key in (rule.metadata or {}).keys()
            if isinstance(rule.metadata, dict)
        }
    )

    payload = {
        "status": "ok",
        "namespace": namespace,
        "policy_count": len(matching_rules),
        "policies": [serialise_policy(rule, include_sensitive=True) for rule in matching_rules],
        "privacy_floors": floors,
        "escalations": contacts,
        "metadata_keys": metadata_keys,
    }
    return jsonify(payload)


def _parse_samples(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        import json

        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - validation guard
            raise PolicyConfigurationError(f"Invalid JSON payload: {exc}") from exc
        raw = parsed
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        samples: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                samples.append(item)
        return samples
    raise PolicyConfigurationError("Samples must be an object, list, or JSON string")


def _simulate_policies(
    policies: list[RetentionPolicyRuleSettings],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not policies:
        return []
    if not samples:
        return [
            {
                "policy": serialise_policy(policy, include_sensitive=True),
                "hits": [],
                "trigger_count": 0,
                "total_samples": 0,
            }
            for policy in policies
        ]

    runtime_rules = settings_to_runtime(policies)
    reports: list[dict[str, Any]] = []

    reference_times = [
        sanitise_timestamp(str(sample.get(key)))
        for sample in samples
        for key in ("reference_time", "timestamp", "created_at")
        if sample.get(key) is not None
    ]
    now = max([value for value in reference_times if value is not None] or [None])

    for runtime_rule, policy in zip(runtime_rules, policies):
        hits: list[dict[str, Any]] = []
        for index, sample in enumerate(samples, start=1):
            namespace = str(sample.get("namespace") or "default")
            try:
                if not runtime_rule.matches(namespace):
                    continue
            except Exception:  # pragma: no cover - defensive guard
                if namespace not in (policy.namespaces or ["*"]):
                    continue

            reasons: list[str] = []
            privacy = sample.get("privacy")
            if privacy is None:
                privacy = sample.get("y_coord")
            if (
                policy.privacy_ceiling is not None
                and privacy is not None
                and float(privacy) > float(policy.privacy_ceiling)
            ):
                reasons.append("privacy_ceiling")

            importance = sample.get("importance")
            if importance is None:
                importance = sample.get("importance_score")
            if (
                policy.importance_floor is not None
                and importance is not None
                and float(importance) < float(policy.importance_floor)
            ):
                reasons.append("importance_floor")

            lifecycle_days = policy.lifecycle_days
            if lifecycle_days is not None:
                created = None
                for key in ("last_accessed", "created_at", "timestamp"):
                    if sample.get(key) is None:
                        continue
                    candidate = sanitise_timestamp(str(sample.get(key)))
                    if candidate is not None:
                        created = candidate
                        break
                reference = sanitise_timestamp(str(sample.get("reference_time"))) or now
                if created is not None and reference is not None:
                    age_days = (reference - created).total_seconds() / 86400.0
                    if age_days > float(lifecycle_days):
                        reasons.append("lifecycle_days")

            if not reasons:
                continue

            hits.append(
                {
                    "sample_index": index,
                    "memory_id": sample.get("memory_id"),
                    "reasons": reasons,
                }
            )

        reports.append(
            {
                "policy": serialise_policy(policy, include_sensitive=True),
                "hits": hits,
                "trigger_count": len(hits),
                "total_samples": len(samples),
            }
        )

    return reports


@governance_bp.route("/governance/policies/dry-run", methods=["POST"])
def policy_dry_run():
    manager = _get_config_manager()
    policies = _load_policy_rules(manager)

    payload = request.get_json(silent=True) or {}
    sample_payload = payload.get("samples")
    try:
        samples = _parse_samples(sample_payload)
    except PolicyConfigurationError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    policy_names = payload.get("policies")
    if policy_names:
        lookup = {policy.name.strip().lower(): policy for policy in policies}
        selected: list[RetentionPolicyRuleSettings] = []
        for name in policy_names:
            key = str(name).strip().lower()
            policy = lookup.get(key)
            if policy:
                selected.append(policy)
        policies = selected

    reports = _simulate_policies(policies, samples)

    reason_counts: Counter[str] = Counter()
    for report in reports:
        for hit in report["hits"]:
            reason_counts.update(hit["reasons"])

    statistics = {
        "total_policies": len(reports),
        "total_samples": len(samples),
        "violations": {reason: count for reason, count in reason_counts.most_common()},
    }

    return jsonify({"status": "ok", "reports": reports, "statistics": statistics})


@governance_bp.route("/governance/telemetry", methods=["GET"])
def governance_telemetry():
    engine = PolicyEnforcementEngine.get_global()
    limit = _parse_positive_int(request.args.get("limit"), 25, maximum=200)

    snapshot = engine.metrics.capture_snapshot()
    payload = snapshot.to_payload(limit=limit, round_durations=3)
    payload["status"] = "ok"
    return jsonify(payload)


@governance_bp.route("/governance/telemetry/export", methods=["GET"])
def governance_telemetry_export():
    engine = PolicyEnforcementEngine.get_global()
    limit = _parse_positive_int(request.args.get("limit"), 25, maximum=200)
    buffer_size = _parse_positive_int(request.args.get("buffer"), 256, maximum=4096)

    collector = engine.metrics

    def _encode(snapshot: PolicyMetricsSnapshot) -> str:
        payload = snapshot.to_payload(limit=limit, round_durations=3)
        return json.dumps(payload, separators=(",", ":")) + "\n"

    def _event_stream():
        queue_size = buffer_size if buffer_size > 0 else 256
        event_queue: queue.Queue[PolicyMetricsSnapshot] = queue.Queue(maxsize=queue_size)
        stop_event = threading.Event()

        def _observer(snapshot: PolicyMetricsSnapshot) -> None:
            if stop_event.is_set():
                return
            try:
                event_queue.put_nowait(snapshot)
            except queue.Full:
                try:
                    event_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    event_queue.put_nowait(snapshot)
                except queue.Full:
                    current_app.logger.warning(
                        "Dropping policy telemetry snapshot; observer queue full",
                    )

        unsubscribe = collector.register_observer(_observer)

        try:
            yield _encode(collector.capture_snapshot(limit=limit))
            while not stop_event.is_set():
                try:
                    next_snapshot = event_queue.get(timeout=5.0)
                except queue.Empty:
                    continue
                yield _encode(next_snapshot)
        finally:
            stop_event.set()
            try:
                unsubscribe()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    response = Response(
        stream_with_context(_event_stream()), mimetype="application/x-ndjson"
    )
    response.headers["Cache-Control"] = "no-cache"
    return response


def _serialise_contact(contact: EscalationContact) -> dict[str, Any]:
    return {
        "name": contact.name,
        "channel": contact.channel,
        "target": contact.target,
        "priority": contact.priority,
        "metadata": contact.metadata or {},
    }


def _load_contacts(manager: ConfigManager) -> list[EscalationContact]:
    definitions = _load_policy_definitions(manager)
    if definitions is None:
        return []
    return list(getattr(definitions, "escalation_contacts", []) or [])


def _persist_contacts(manager: ConfigManager, contacts: Iterable[EscalationContact]) -> None:
    payload = [contact.dict() for contact in contacts]
    manager.update_setting("policy.escalation_contacts", payload)


def _load_overrides(manager: ConfigManager) -> list[PolicyOverride]:
    definitions = _load_policy_definitions(manager)
    if definitions is None:
        return []
    return list(getattr(definitions, "overrides", []) or [])


def _persist_overrides(manager: ConfigManager, overrides: Iterable[PolicyOverride]) -> None:
    payload: list[dict[str, Any]] = []
    for override in overrides:
        data = override.dict()
        action = data.get("action")
        stage = data.get("stage")
        expires_at = data.get("expires_at")
        if hasattr(action, "value"):
            data["action"] = action.value
        if hasattr(stage, "value"):
            data["stage"] = stage.value
        if isinstance(expires_at, datetime):
            data["expires_at"] = _serialize_datetime(expires_at)
        payload.append(data)
    manager.update_setting("policy.overrides", payload)


@governance_bp.route("/governance/escalations", methods=["GET"])
def list_escalations():
    manager = _get_config_manager()
    contacts = _load_contacts(manager)
    return jsonify(
        {
            "status": "ok",
            "contacts": [_serialise_contact(contact) for contact in contacts],
            "total": len(contacts),
        }
    )


@governance_bp.route("/governance/escalations", methods=["POST"])
def create_escalation_contact():
    manager = _get_config_manager()
    payload = request.get_json(silent=True) or {}
    contact_payload = payload.get("contact") if isinstance(payload, dict) else None
    if contact_payload is None:
        contact_payload = payload
    if not isinstance(contact_payload, dict):
        return (
            jsonify({"status": "error", "message": "Contact payload must be an object"}),
            400,
        )
    try:
        contact = EscalationContact(**contact_payload)
    except ValidationError as exc:
        return jsonify({"status": "error", "message": exc.errors()}), 400

    contacts = _load_contacts(manager)
    if any(existing.name.strip().lower() == contact.name.strip().lower() for existing in contacts):
        return (
            jsonify({"status": "error", "message": "Escalation contact with that name already exists"}),
            409,
        )

    contacts.append(contact)
    _persist_contacts(manager, contacts)

    return jsonify({"status": "ok", "contact": _serialise_contact(contact), "total": len(contacts)})


@governance_bp.route("/governance/escalations/<string:name>", methods=["PUT"])
def update_escalation_contact(name: str):
    manager = _get_config_manager()
    payload = request.get_json(silent=True) or {}
    contact_payload = payload.get("contact") if isinstance(payload, dict) else None
    if contact_payload is None:
        contact_payload = payload
    if not isinstance(contact_payload, dict):
        return (
            jsonify({"status": "error", "message": "Contact payload must be an object"}),
            400,
        )

    try:
        updated_contact = EscalationContact(**contact_payload)
    except ValidationError as exc:
        return jsonify({"status": "error", "message": exc.errors()}), 400

    contacts = _load_contacts(manager)
    target_key = name.strip().lower()
    replaced = False
    for index, contact in enumerate(contacts):
        if contact.name.strip().lower() == target_key:
            contacts[index] = updated_contact
            replaced = True
            break
    if not replaced:
        return jsonify({"status": "error", "message": f"Escalation contact '{name}' not found"}), 404

    _persist_contacts(manager, contacts)
    return jsonify({"status": "ok", "contact": _serialise_contact(updated_contact)})


@governance_bp.route("/governance/escalations/<string:name>", methods=["DELETE"])
def delete_escalation_contact(name: str):
    manager = _get_config_manager()
    contacts = _load_contacts(manager)
    target_key = name.strip().lower()
    filtered = [contact for contact in contacts if contact.name.strip().lower() != target_key]
    if len(filtered) == len(contacts):
        return jsonify({"status": "error", "message": f"Escalation contact '{name}' not found"}), 404

    _persist_contacts(manager, filtered)
    return jsonify({"status": "ok", "total": len(filtered)})


@governance_bp.route("/governance/escalations/<string:name>/preview", methods=["POST"])
def preview_escalation(name: str):
    manager = _get_config_manager()
    contacts = _load_contacts(manager)
    target = next(
        (contact for contact in contacts if contact.name.strip().lower() == name.strip().lower()),
        None,
    )
    if target is None:
        return jsonify({"status": "error", "message": f"Escalation contact '{name}' not found"}), 404

    payload = request.get_json(silent=True) or {}
    message = payload.get("message") if isinstance(payload, dict) else None
    preview_payload = {
        "channel": target.channel,
        "target": target.target,
        "priority": target.priority,
        "message": message or "Test notification from Memoria",
    }
    return jsonify({"status": "ok", "preview": preview_payload})


@governance_bp.route("/governance/escalations/verification", methods=["GET"])
def escalation_roster_verification():
    manager = _get_config_manager()
    contacts = _load_contacts(manager)
    cadence = _parse_positive_int(request.args.get("cadence"), 60, maximum=1440)
    refresh = _parse_bool(request.args.get("refresh"))

    payload = _load_roster_status()
    if payload is None or refresh or payload.get("cadence_minutes") != cadence:
        payload = verify_escalation_contacts(
            contacts,
            cadence_minutes=cadence,
        )
        _persist_roster_status(payload)

    return jsonify(
        {
            "status": "ok",
            "verification": payload,
            "total_contacts": len(contacts),
        }
    )


@governance_bp.route("/governance/escalations/rotation", methods=["GET"])
def escalation_roster_rotation():
    manager = _get_config_manager()
    contacts = _load_contacts(manager)
    cadence = _parse_positive_int(request.args.get("cadence"), 60, maximum=1440)
    refresh = _parse_bool(request.args.get("refresh"))

    payload = _load_rotation_status()
    if payload is None or refresh or payload.get("cadence_minutes") != cadence:
        reference = datetime.now(timezone.utc)
        updated_contacts: list[EscalationContact] = []
        records: list[dict[str, Any]] = []
        metadata_updates = 0
        overdue_contacts = 0

        for contact in contacts:
            updated_contact, record = apply_rotation_metadata(contact, reference_time=reference)
            updated_contacts.append(updated_contact)
            if record.get("metadata_updated"):
                metadata_updates += 1
            if record.get("overdue_windows"):
                overdue_contacts += 1
            records.append(record)

        if metadata_updates:
            _persist_contacts(manager, updated_contacts)

        payload = {
            "generated_at": _serialize_datetime(reference),
            "next_check_at": _serialize_datetime(reference + timedelta(minutes=cadence)),
            "cadence_minutes": cadence,
            "summary": {
                "total_contacts": len(contacts),
                "metadata_updates": metadata_updates,
                "overdue_contacts": overdue_contacts,
            },
            "contacts": records,
        }
        _persist_rotation_status(payload)

    return jsonify(
        {
            "status": "ok",
            "rotation": payload,
            "total_contacts": len(contacts),
        }
    )


def _serialise_details(details: Any) -> Any:
    if details is None:
        return None
    if isinstance(details, (str, int, float, bool)):
        return details
    if isinstance(details, dict):
        return {key: _serialise_details(value) for key, value in details.items()}
    if isinstance(details, (list, tuple, set)):
        return [_serialise_details(item) for item in details]
    if isinstance(details, datetime):
        return _serialize_datetime(details)
    try:
        import json

        return json.loads(json.dumps(details, default=str))
    except Exception:  # pragma: no cover - defensive
        return str(details)


def _build_audit_payload(row: RetentionPolicyAudit) -> dict[str, Any]:
    return {
        "id": row.id,
        "timestamp": _serialize_datetime(row.created_at),
        "namespace": row.namespace,
        "policy_name": row.policy_name,
        "action": row.action,
        "escalate_to": row.escalate_to,
        "team_id": row.team_id,
        "workspace_id": row.workspace_id,
        "details": _serialise_details(row.details),
    }


def _with_session(func):
    def wrapper(*args, **kwargs):
        manager = _get_database_manager()
        if manager is None or not getattr(manager, "SessionLocal", None):
            return jsonify({"status": "error", "message": "Database unavailable"}), 503
        session: Session = manager.SessionLocal()
        try:
            return func(session, *args, **kwargs)
        finally:
            session.close()

    wrapper.__name__ = func.__name__
    return wrapper


def _parse_positive_int(value: str | None, default: int, maximum: int = 500) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return min(parsed, maximum)


def _build_audit_query(session: Session) -> tuple[Any, dict[str, Any]]:
    query = session.query(RetentionPolicyAudit)
    action = request.args.get("action")
    namespace_glob = request.args.get("namespace")
    escalation = request.args.get("escalation")
    role = request.args.get("role")

    if action and action != "all":
        query = query.filter(RetentionPolicyAudit.action == action)
    if escalation and escalation != "all":
        query = query.filter(RetentionPolicyAudit.escalate_to == escalation)
    if role and role != "all":
        query = query.filter(RetentionPolicyAudit.team_id == role)
    if namespace_glob:
        pattern = namespace_glob.replace("*", "%")
        query = query.filter(RetentionPolicyAudit.namespace.like(pattern))

    return query, {
        "action": action,
        "namespace": namespace_glob,
        "escalation": escalation,
        "role": role,
    }


@governance_bp.route("/governance/audits", methods=["GET"])
@_with_session
def list_audits(session: Session):
    query, _filters = _build_audit_query(session)

    page = _parse_positive_int(request.args.get("page"), 1)
    page_size = _parse_positive_int(request.args.get("page_size"), 50, maximum=200)

    total = query.count()
    rows = (
        query.order_by(RetentionPolicyAudit.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    actions = [value for (value,) in session.query(RetentionPolicyAudit.action).distinct()]
    escalations = [value for (value,) in session.query(RetentionPolicyAudit.escalate_to).distinct() if value]
    roles = [value for (value,) in session.query(RetentionPolicyAudit.team_id).distinct() if value]

    return jsonify(
        {
            "status": "ok",
            "audits": [_build_audit_payload(row) for row in rows],
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": page * page_size < total,
            "has_previous": page > 1,
            "filters": {
                "actions": sorted(actions),
                "escalations": sorted(escalations),
                "roles": sorted(roles),
            },
        }
    )


@governance_bp.route("/governance/audits/export", methods=["GET"])
@_with_session
def export_audits(session: Session):
    query, _ = _build_audit_query(session)
    fmt = (request.args.get("format") or "csv").strip().lower()
    limit = _parse_positive_int(request.args.get("limit"), 1000, maximum=5000)

    query = query.order_by(RetentionPolicyAudit.created_at.desc())
    if limit:
        query = query.limit(limit)

    rows = query.all()
    audits = [_build_audit_payload(row) for row in rows]

    if fmt == "json":
        return jsonify({"status": "ok", "audits": audits, "count": len(audits)})

    output = io.StringIO()
    fieldnames = [
        "id",
        "timestamp",
        "namespace",
        "policy_name",
        "action",
        "escalate_to",
        "team_id",
        "workspace_id",
        "details",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for audit in audits:
        row = dict(audit)
        details = row.get("details")
        row["details"] = json.dumps(details or {}, ensure_ascii=False)
        writer.writerow(row)

    response = Response(output.getvalue(), mimetype="text/csv")
    filename = f"policy_audits_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    response.headers["Cache-Control"] = "no-store"
    return response


@governance_bp.route("/governance/audits/<int:audit_id>", methods=["GET"])
@_with_session
def get_audit(session: Session, audit_id: int):
    row = session.get(RetentionPolicyAudit, audit_id)
    if row is None:
        return jsonify({"status": "error", "message": "Audit not found"}), 404

    payload = _build_audit_payload(row)
    detail = payload.get("details") or {}
    violations = []
    if isinstance(detail, dict):
        violations = detail.get("violations") or []
    payload["violations"] = violations
    return jsonify({"status": "ok", "audit": payload})


@governance_bp.route("/governance/overrides", methods=["GET"])
def list_overrides():
    manager = _get_config_manager()
    overrides = _load_overrides(manager)
    window_days = _parse_positive_int(request.args.get("window"), 14, maximum=180)
    reference = datetime.now(timezone.utc)

    payload = [
        summarise_override_window(
            override,
            reference_time=reference,
            window_days=window_days,
        )
        for override in overrides
    ]
    payload.sort(key=lambda item: (item.get("status"), item.get("expires_at") or ""))

    return jsonify(
        {
            "status": "ok",
            "overrides": payload,
            "total": len(payload),
            "window_days": window_days,
        }
    )


@governance_bp.route("/governance/overrides/automation", methods=["GET"])
def overrides_automation():
    manager = _get_config_manager()
    overrides = _load_overrides(manager)
    cadence = _parse_positive_int(request.args.get("cadence"), 60, maximum=1440)
    refresh = _parse_bool(request.args.get("refresh"))

    payload = _load_override_status()
    if payload is None or refresh or payload.get("cadence_minutes") != cadence:
        reference = datetime.now(timezone.utc)
        updated_overrides: list[PolicyOverride] = []
        records: list[dict[str, Any]] = []
        metadata_updates = 0
        expired_count = 0

        for override in overrides:
            updated_override, summary = apply_override_expiry(override, reference_time=reference)
            updated_overrides.append(updated_override)
            if summary.get("metadata_updated"):
                metadata_updates += 1
            if summary.get("status") == "expired":
                expired_count += 1
            records.append(summary)

        if metadata_updates:
            _persist_overrides(manager, updated_overrides)

        payload = {
            "generated_at": _serialize_datetime(reference),
            "next_check_at": _serialize_datetime(reference + timedelta(minutes=cadence)),
            "cadence_minutes": cadence,
            "summary": {
                "total_overrides": len(overrides),
                "metadata_updates": metadata_updates,
                "expired_overrides": expired_count,
            },
            "overrides": records,
        }
        _persist_override_status(payload)

    return jsonify(
        {
            "status": "ok",
            "automation": payload,
            "total_overrides": len(overrides),
        }
    )


@governance_bp.route("/governance/overrides/<string:name>/expire", methods=["POST"])
def expire_override(name: str):
    manager = _get_config_manager()
    overrides = _load_overrides(manager)
    target_key = name.strip().lower()
    reference = datetime.now(timezone.utc)

    updated: list[PolicyOverride] = []
    selected_summary: dict[str, Any] | None = None
    selected_override: PolicyOverride | None = None
    for override in overrides:
        if override.name.strip().lower() == target_key:
            pending = override.copy(update={"expires_at": reference})
            updated_override, summary = apply_override_expiry(pending, reference_time=reference)
            updated.append(updated_override)
            selected_summary = summary
            selected_override = updated_override
        else:
            updated.append(override)

    if selected_summary is None or selected_override is None:
        return jsonify({"status": "error", "message": f"Override '{name}' not found"}), 404

    _persist_overrides(manager, updated)

    return jsonify({"status": "ok", "override": selected_summary})


__all__ = ["governance_bp"]
