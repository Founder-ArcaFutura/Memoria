"""REST endpoints for managing retention policy rules."""

from __future__ import annotations

from http import HTTPStatus
from typing import Iterable

from flask import Blueprint, current_app, jsonify, request

from memoria.config.manager import ConfigManager
from memoria.config.settings import RetentionPolicyRuleSettings
from memoria.policy.utils import (
    PolicyConfigurationError,
    apply_runtime_policies,
    serialise_policy,
)

policy_bp = Blueprint("policy", __name__)

_ADMIN_ROLES = {"admin", "owner", "superuser"}
_ROLE_HEADERS = ("X-Memoria-Role", "X-Role", "X-User-Role")


def _resolve_role() -> str:
    for header in _ROLE_HEADERS:
        value = request.headers.get(header)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    candidate = request.args.get("role")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip().lower()
    return "member"


def _require_admin(role: str) -> tuple[dict[str, str], int] | None:
    if role in _ADMIN_ROLES:
        return None
    return (
        {"status": "error", "message": "Administrative role required."},
        HTTPStatus.FORBIDDEN,
    )


def _get_config_manager() -> ConfigManager:
    manager = current_app.config.get("config_manager")
    if not isinstance(manager, ConfigManager):
        raise RuntimeError("Config manager unavailable")
    return manager


def _load_policies(manager: ConfigManager) -> list[RetentionPolicyRuleSettings]:
    settings = manager.get_settings()
    raw = getattr(settings.memory, "retention_policy_rules", []) or []
    return list(raw)


def _persist_policies(
    manager: ConfigManager,
    policies: Iterable[RetentionPolicyRuleSettings],
) -> None:
    policy_list = list(policies)
    payload = []
    for policy in policy_list:
        data = policy.dict()
        action = data.get("action")
        if hasattr(action, "value"):
            data["action"] = action.value
        payload.append(data)
    manager.update_setting("memory.retention_policy_rules", payload)

    memoria_instance = current_app.config.get("memoria")
    storage_service = getattr(memoria_instance, "storage_service", None)
    retention_service = getattr(memoria_instance, "retention_service", None)
    apply_runtime_policies(retention_service, storage_service, policy_list)


def _normalise_payload(payload: dict[str, object]) -> RetentionPolicyRuleSettings:
    try:
        return RetentionPolicyRuleSettings(**payload)
    except Exception as exc:  # pragma: no cover - validation error formatting
        raise PolicyConfigurationError(str(exc)) from exc


@policy_bp.route("/policy/rules", methods=["GET"])
def list_policies():
    manager = _get_config_manager()
    role = _resolve_role()
    try:
        policies = _load_policies(manager)
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"status": "error", "message": str(exc)}), HTTPStatus.SERVICE_UNAVAILABLE

    include_sensitive = role in _ADMIN_ROLES
    payload = [
        serialise_policy(policy, include_sensitive=include_sensitive)
        for policy in policies
    ]
    return jsonify({"status": "ok", "policies": payload, "count": len(payload)})


@policy_bp.route("/policy/rules/<string:name>", methods=["GET"])
def get_policy(name: str):
    manager = _get_config_manager()
    role = _resolve_role()
    target = name.strip().lower()
    for policy in _load_policies(manager):
        if policy.name.strip().lower() == target:
            include_sensitive = role in _ADMIN_ROLES
            return jsonify(
                {
                    "status": "ok",
                    "policy": serialise_policy(policy, include_sensitive=include_sensitive),
                }
            )
    return (
        jsonify({"status": "error", "message": f"Policy '{name}' not found"}),
        HTTPStatus.NOT_FOUND,
    )


@policy_bp.route("/policy/rules", methods=["POST"])
def create_policy():
    role = _resolve_role()
    forbidden = _require_admin(role)
    if forbidden is not None:
        return jsonify(forbidden[0]), forbidden[1]

    manager = _get_config_manager()
    payload = request.get_json(silent=True) or {}
    rule_payload = payload.get("rule") if isinstance(payload, dict) else None
    if rule_payload is None:
        rule_payload = payload
    if not isinstance(rule_payload, dict):
        return (
            jsonify({"status": "error", "message": "Policy payload must be an object"}),
            HTTPStatus.BAD_REQUEST,
        )

    try:
        new_rule = _normalise_payload(rule_payload)
    except PolicyConfigurationError as exc:
        return jsonify({"status": "error", "message": str(exc)}), HTTPStatus.BAD_REQUEST

    policies = _load_policies(manager)
    if any(rule.name.strip().lower() == new_rule.name.strip().lower() for rule in policies):
        return (
            jsonify({"status": "error", "message": "Policy with that name already exists"}),
            HTTPStatus.CONFLICT,
        )

    policies.append(new_rule)
    _persist_policies(manager, policies)

    return (
        jsonify(
            {
                "status": "ok",
                "policy": serialise_policy(new_rule, include_sensitive=True),
                "count": len(policies),
            }
        ),
        HTTPStatus.CREATED,
    )


@policy_bp.route("/policy/rules/<string:name>", methods=["PUT"])
def update_policy(name: str):
    role = _resolve_role()
    forbidden = _require_admin(role)
    if forbidden is not None:
        return jsonify(forbidden[0]), forbidden[1]

    manager = _get_config_manager()
    payload = request.get_json(silent=True) or {}
    rule_payload = payload.get("rule") if isinstance(payload, dict) else None
    if rule_payload is None:
        rule_payload = payload
    if not isinstance(rule_payload, dict):
        return (
            jsonify({"status": "error", "message": "Policy payload must be an object"}),
            HTTPStatus.BAD_REQUEST,
        )

    try:
        updated_rule = _normalise_payload(rule_payload)
    except PolicyConfigurationError as exc:
        return jsonify({"status": "error", "message": str(exc)}), HTTPStatus.BAD_REQUEST

    target = name.strip().lower()
    policies = _load_policies(manager)
    replaced = False
    for index, policy in enumerate(policies):
        if policy.name.strip().lower() == target:
            policies[index] = updated_rule
            replaced = True
            break
    if not replaced:
        return (
            jsonify({"status": "error", "message": f"Policy '{name}' not found"}),
            HTTPStatus.NOT_FOUND,
        )

    _persist_policies(manager, policies)

    return jsonify(
        {
            "status": "ok",
            "policy": serialise_policy(updated_rule, include_sensitive=True),
        }
    )


@policy_bp.route("/policy/rules/<string:name>", methods=["DELETE"])
def delete_policy(name: str):
    role = _resolve_role()
    forbidden = _require_admin(role)
    if forbidden is not None:
        return jsonify(forbidden[0]), forbidden[1]

    manager = _get_config_manager()
    policies = _load_policies(manager)
    target = name.strip().lower()
    filtered = [policy for policy in policies if policy.name.strip().lower() != target]
    if len(filtered) == len(policies):
        return (
            jsonify({"status": "error", "message": f"Policy '{name}' not found"}),
            HTTPStatus.NOT_FOUND,
        )

    _persist_policies(manager, filtered)

    return jsonify({"status": "ok", "count": len(filtered)})


__all__ = ["policy_bp"]

