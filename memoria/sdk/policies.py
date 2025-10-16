"""Thin HTTP client for managing policies via the Memoria API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from memoria.config.settings import RetentionPolicyRuleSettings
from memoria.policy.utils import PolicyConfigurationError


class PolicyClientError(RuntimeError):
    """Raised when the remote API returns an error response."""

    status: int
    payload: Mapping[str, Any] | None

    def __init__(
        self, message: str, *, status: int, payload: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


@dataclass(slots=True)
class PolicyClient:
    """Simple helper around :mod:`urllib` for policy CRUD operations."""

    base_url: str
    api_key: str | None = None
    default_role: str = "admin"
    timeout: float = 10.0

    def _build_headers(self, role: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        effective_role = role or self.default_role
        if effective_role:
            headers["X-Memoria-Role"] = effective_role
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        role: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        data: bytes | None = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request_obj = urllib.request.Request(
            url,
            data=data,
            method=method.upper(),
            headers=self._build_headers(role),
        )
        try:
            with urllib.request.urlopen(request_obj, timeout=self.timeout) as response:
                body = response.read()
                if not body:
                    return {}
                return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            payload_data: Mapping[str, Any] | None = None
            if body:
                try:
                    payload_data = json.loads(body)
                except json.JSONDecodeError:
                    payload_data = {"message": body}
            message = (
                payload_data.get("message")
                if isinstance(payload_data, Mapping)
                else body
            )
            raise PolicyClientError(
                str(message or exc), status=exc.code, payload=payload_data
            )

    @staticmethod
    def _prepare_rule_payload(
        rule: Mapping[str, Any] | RetentionPolicyRuleSettings,
    ) -> MutableMapping[str, Any]:
        if isinstance(rule, RetentionPolicyRuleSettings):
            payload = rule.dict()
            action = payload.get("action")
            if hasattr(action, "value"):
                payload["action"] = action.value
            return payload
        payload = dict(rule)
        action = payload.get("action")
        if hasattr(action, "value"):
            payload["action"] = action.value  # pragma: no cover - defensive
        return payload

    def list(self, *, role: str | None = None) -> Sequence[Mapping[str, Any]]:
        response = self._request("GET", "/policy/rules", role=role)
        return response.get("policies", []) if isinstance(response, Mapping) else []

    def get(self, name: str, *, role: str | None = None) -> Mapping[str, Any]:
        response = self._request("GET", f"/policy/rules/{name}", role=role)
        if not isinstance(response, Mapping):
            return {}
        return (
            response.get("policy", {})
            if isinstance(response.get("policy"), Mapping)
            else {}
        )

    def create(
        self,
        rule: Mapping[str, Any] | RetentionPolicyRuleSettings,
        *,
        role: str | None = None,
    ) -> Mapping[str, Any]:
        payload = {"rule": self._prepare_rule_payload(rule)}
        response = self._request("POST", "/policy/rules", payload=payload, role=role)
        return response.get("policy", {}) if isinstance(response, Mapping) else {}

    def update(
        self,
        name: str,
        rule: Mapping[str, Any] | RetentionPolicyRuleSettings,
        *,
        role: str | None = None,
    ) -> Mapping[str, Any]:
        payload = {"rule": self._prepare_rule_payload(rule)}
        response = self._request(
            "PUT", f"/policy/rules/{name}", payload=payload, role=role
        )
        return response.get("policy", {}) if isinstance(response, Mapping) else {}

    def delete(self, name: str, *, role: str | None = None) -> Mapping[str, Any]:
        return self._request("DELETE", f"/policy/rules/{name}", role=role)

    def apply(
        self,
        rules: Sequence[Mapping[str, Any] | RetentionPolicyRuleSettings],
        *,
        role: str | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Replace all policies by deleting existing ones and creating the supplied set."""

        current = self.list(role=role)
        existing_names = [
            entry.get("name") for entry in current if isinstance(entry, Mapping)
        ]
        for name in existing_names:
            if isinstance(name, str):
                try:
                    self.delete(name, role=role)
                except PolicyClientError:
                    continue
        created: list[Mapping[str, Any]] = []
        for rule in rules:
            created.append(self.create(rule, role=role))
        return created

    def validate_local(
        self, rules: Sequence[Mapping[str, Any] | RetentionPolicyRuleSettings]
    ) -> list[RetentionPolicyRuleSettings]:
        """Validate payloads locally using the Pydantic schema."""

        validated: list[RetentionPolicyRuleSettings] = []
        for rule in rules:
            payload = self._prepare_rule_payload(rule)
            try:
                validated.append(RetentionPolicyRuleSettings(**payload))
            except Exception as exc:
                raise PolicyConfigurationError(str(exc)) from exc
        return validated


__all__ = ["PolicyClient", "PolicyClientError"]
