from __future__ import annotations

import pytest

from memoria.config.settings import RetentionPolicyAction, RetentionPolicyRuleSettings


@pytest.fixture()
def policy_client(tmp_path, monkeypatch):
    db_path = tmp_path / "policy.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("MEMORIA_API_KEY", "secret")

    from memoria_server.api.app_factory import create_app

    app = create_app()
    app.config["TESTING"] = True

    manager = app.config["config_manager"]
    manager.reset_to_defaults()

    memoria = app.config["memoria"]
    with app.test_client() as client:
        yield client, memoria, manager


def _headers(role: str) -> dict[str, str]:
    return {"X-API-Key": "secret", "X-Memoria-Role": role}


def _seed_rule(name: str) -> RetentionPolicyRuleSettings:
    return RetentionPolicyRuleSettings(
        name=name,
        namespaces=["ops/*"],
        privacy_ceiling=5.0,
        action=RetentionPolicyAction.ESCALATE,
        escalate_to="secops",
        metadata={"severity": "high"},
    )


def test_list_policies_viewer_redacts_sensitive(policy_client) -> None:
    client, _memoria, manager = policy_client
    manager.update_setting(
        "memory.retention_policy_rules", [_seed_rule("ops-block").dict()]
    )

    response = client.get("/policy/rules", headers=_headers("viewer"))
    assert response.status_code == 200
    data = response.get_json()
    policy = data["policies"][0]
    assert "escalate_to" not in policy
    assert policy.get("has_escalation")
    assert policy.get("metadata_keys") == ["severity"]


def test_list_policies_admin_includes_sensitive(policy_client) -> None:
    client, _memoria, manager = policy_client
    manager.update_setting(
        "memory.retention_policy_rules", [_seed_rule("ops-block").dict()]
    )

    response = client.get("/policy/rules", headers=_headers("admin"))
    assert response.status_code == 200
    policy = response.get_json()["policies"][0]
    assert policy["escalate_to"] == "secops"
    assert policy["metadata"]["severity"] == "high"


def test_create_policy_requires_admin(policy_client) -> None:
    client, _memoria, _manager = policy_client
    payload = {
        "rule": {
            "name": "ops-block",
            "namespaces": ["ops/*"],
            "privacy_ceiling": 4.0,
            "action": "block",
        }
    }

    response = client.post("/policy/rules", headers=_headers("viewer"), json=payload)
    assert response.status_code == 403


def test_create_policy_updates_runtime(policy_client) -> None:
    client, memoria, manager = policy_client
    payload = {
        "rule": {
            "name": "ops-block",
            "namespaces": ["ops/*"],
            "privacy_ceiling": 4.0,
            "action": "block",
        }
    }

    response = client.post("/policy/rules", headers=_headers("admin"), json=payload)
    assert response.status_code == 201
    stored = manager.get_setting("memory.retention_policy_rules", [])
    assert stored and stored[0]["name"] == "ops-block"

    runtime = getattr(memoria.storage_service, "_retention_policies", ())
    assert len(runtime) == 1


def test_update_policy(policy_client) -> None:
    client, memoria, manager = policy_client
    manager.update_setting(
        "memory.retention_policy_rules", [_seed_rule("ops-block").dict()]
    )

    response = client.put(
        "/policy/rules/ops-block",
        headers=_headers("admin"),
        json={
            "rule": {
                "name": "ops-block",
                "namespaces": ["ops/*"],
                "privacy_ceiling": 3.5,
                "action": "block",
            }
        },
    )
    assert response.status_code == 200
    stored = manager.get_setting("memory.retention_policy_rules", [])
    assert stored[0]["privacy_ceiling"] == 3.5

    runtime = getattr(memoria.storage_service, "_retention_policies", ())
    assert runtime[0].privacy_ceiling == 3.5


def test_delete_policy(policy_client) -> None:
    client, memoria, manager = policy_client
    manager.update_setting(
        "memory.retention_policy_rules", [_seed_rule("ops-block").dict()]
    )

    response = client.delete("/policy/rules/ops-block", headers=_headers("admin"))
    assert response.status_code == 200
    stored = manager.get_setting("memory.retention_policy_rules", [])
    assert stored == []

    runtime = getattr(memoria.storage_service, "_retention_policies", ())
    assert runtime == ()
