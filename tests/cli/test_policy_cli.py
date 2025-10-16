from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import memoria.cli as cli_module
from memoria.cli import main
from memoria.config.manager import ConfigManager
from memoria.policy.enforcement import PolicyMetricsSnapshot


@pytest.fixture(autouse=True)
def reset_config() -> None:
    manager = ConfigManager.get_instance()
    manager.reset_to_defaults()
    yield
    manager.reset_to_defaults()


def _write_policy(path: Path) -> None:
    payload = {
        "name": "ops-block",
        "namespaces": ["ops/*"],
        "privacy_ceiling": 5.0,
        "importance_floor": 0.2,
        "action": "escalate",
        "escalate_to": "secops",
    }
    path.write_text(json.dumps([payload]), encoding="utf-8")


def _seed_contacts(manager: ConfigManager) -> None:
    contacts = [
        {
            "name": "primary",
            "channel": "email",
            "target": "primary@example.com",
            "priority": "high",
            "metadata": {
                "namespaces": ["ops/*"],
                "triggers": ["privacy", "urgency"],
                "coverage": "24x7",
                "rotation": [
                    {
                        "date": "2050-01-01T00:00:00Z",
                        "primary": "alice",
                        "secondary": "bob",
                    },
                    {
                        "date": "2050-01-08T00:00:00Z",
                        "primary": "carol",
                        "secondary": "dave",
                    },
                ],
            },
        },
        {
            "name": "backup",
            "channel": "sms",
            "target": "+1000000000",
            "metadata": {
                "namespaces": ["ops/*"],
                "triggers": ["backup"],
                "coverage": "follow-the-sun",
                "rotation": [
                    {
                        "date": "2050-01-04T00:00:00Z",
                        "primary": "erin",
                        "secondary": "mallory",
                    }
                ],
            },
        },
    ]
    manager.update_setting("policy.escalation_contacts", contacts)


def test_policy_lint(tmp_path, capsys) -> None:
    policy_path = tmp_path / "policy.json"
    _write_policy(policy_path)

    exit_code = main(["policy", "lint", str(policy_path)])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Validated 1 policy" in output
    assert "ops-block" in output


def test_policy_lint_invalid_json(tmp_path) -> None:
    policy_path = tmp_path / "broken.json"
    policy_path.write_text("not-json", encoding="utf-8")

    exit_code = main(["policy", "lint", str(policy_path)])
    assert exit_code == 1


def test_policy_apply_updates_configuration(tmp_path) -> None:
    policy_path = tmp_path / "policy.json"
    _write_policy(policy_path)

    manager = ConfigManager.get_instance()
    assert main(["policy", "apply", str(policy_path)]) == 0

    stored = manager.get_setting("memory.retention_policy_rules", [])
    assert isinstance(stored, list)
    assert stored and stored[0]["name"] == "ops-block"


def test_policy_test_reports_triggers(tmp_path, capsys) -> None:
    policy_path = tmp_path / "policy.json"
    _write_policy(policy_path)

    samples = [
        {
            "memory_id": "mem-1",
            "namespace": "ops/team-alpha",
            "y_coord": 9.5,
            "importance_score": 0.1,
            "created_at": "2025-10-15T00:00:00",
            "reference_time": "2025-10-17T00:00:00",
        }
    ]
    sample_path = tmp_path / "samples.json"
    sample_path.write_text(json.dumps(samples), encoding="utf-8")

    exit_code = main(
        [
            "policy",
            "test",
            str(policy_path),
            "--samples",
            str(sample_path),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "mem-1" in captured
    assert "privacy_ceiling" in captured


def test_roster_verify_json(capsys) -> None:
    manager = ConfigManager.get_instance()
    _seed_contacts(manager)

    exit_code = main(["roster", "verify", "--cadence", "30", "--format", "json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["summary"]["total_contacts"] == 2
    assert len(payload["contacts"]) == 2
    first = payload["contacts"][0]
    assert set(first).issuperset({"name", "status", "issues"})
    assert "Roster verification" in captured.err


def test_roster_rotate_csv(capsys) -> None:
    manager = ConfigManager.get_instance()
    _seed_contacts(manager)

    exit_code = main(["roster", "rotate", "--format", "csv"])
    assert exit_code == 0
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert lines and lines[0].startswith("name,channel,target")
    assert "metadata_updated" in lines[0]
    assert "require metadata updates" in captured.err


def test_policy_telemetry_single_snapshot_json_webhook(monkeypatch, capsys) -> None:
    webhook_calls: list[dict[str, Any]] = []

    class _StubCollector:
        def __init__(self) -> None:
            self.capture_invocations: list[int | None] = []

        def register_observer(
            self, observer
        ):  # pragma: no cover - should not be called
            raise AssertionError(
                "register_observer should not be invoked without --follow"
            )

        def capture_snapshot(self, *, limit=None):  # type: ignore[override]
            self.capture_invocations.append(limit)
            return PolicyMetricsSnapshot(
                generated_at=1_700_000_123.0,
                counts={"ingest:allow": 3},
                stage_counts={"ingest": {"allow": 3}},
                policy_actions=[
                    {
                        "policy": "baseline",
                        "action": "allow",
                        "count": 3,
                        "total_duration_ms": 27.0,
                        "average_duration_ms": 9.0,
                        "min_duration_ms": 8.5,
                        "max_duration_ms": 9.5,
                        "stage_counts": {"ingest": 3},
                        "last_triggered_at": 1_700_000_120.0,
                    }
                ],
            )

    collector = _StubCollector()

    def _fake_loader():
        return collector, object()

    monkeypatch.setattr(
        cli_module,
        "_load_policy_metrics_collector_for_cli",
        _fake_loader,
    )

    def _fake_urlopen(
        request, timeout=10.0
    ):  # pragma: no cover - exercised via assertions
        payload = request.data.decode("utf-8") if request.data else ""
        webhook_calls.append(
            {
                "url": request.full_url,
                "content_type": request.get_header("Content-Type")
                or request.get_header("Content-type"),
                "payload": payload,
            }
        )

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Response()

    monkeypatch.setattr(cli_module.urllib.request, "urlopen", _fake_urlopen)

    exit_code = main(
        [
            "policy",
            "telemetry",
            "--format",
            "json",
            "--max-events",
            "1",
            "--webhook",
            "https://example.invalid/webhook",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["counts"] == {"ingest:allow": 3}
    assert payload["policy_actions"][0]["total_duration_ms"] == 27.0
    assert collector.capture_invocations == [None]

    assert len(webhook_calls) == 1
    call = webhook_calls[0]
    assert call["url"] == "https://example.invalid/webhook"
    assert call["content_type"] == "application/json"
    assert call["payload"] == lines[0]


def test_policy_telemetry_follow_json(monkeypatch, capsys) -> None:
    webhook_calls: list[dict[str, Any]] = []

    class _StubCollector:
        def __init__(self) -> None:
            self._observer: Any | None = None
            self._event_sent = False

        def register_observer(self, observer):
            self._observer = observer

            def _undo() -> None:
                self._observer = None

            return _undo

        def capture_snapshot(self, *, limit=None):  # type: ignore[override]
            first_snapshot = PolicyMetricsSnapshot(
                generated_at=1_700_000_000.0,
                counts={"ingest:allow": 1},
                stage_counts={"ingest": {"allow": 1}},
                policy_actions=[
                    {
                        "policy": "baseline",
                        "action": "allow",
                        "count": 1,
                        "total_duration_ms": 12.5,
                        "average_duration_ms": 12.5,
                        "min_duration_ms": 12.5,
                        "max_duration_ms": 12.5,
                        "stage_counts": {"ingest": 1},
                        "last_triggered_at": 1_700_000_000.0,
                    }
                ],
            )

            if self._observer is not None and not self._event_sent:
                self._event_sent = True
                second_snapshot = PolicyMetricsSnapshot(
                    generated_at=1_700_000_300.0,
                    counts={"ingest:allow": 2, "review:escalate": 1},
                    stage_counts={
                        "ingest": {"allow": 2},
                        "review": {"escalate": 1},
                    },
                    policy_actions=[
                        {
                            "policy": "baseline",
                            "action": "allow",
                            "count": 2,
                            "total_duration_ms": 30.0,
                            "average_duration_ms": 15.0,
                            "min_duration_ms": 12.5,
                            "max_duration_ms": 17.5,
                            "stage_counts": {"ingest": 2},
                            "last_triggered_at": 1_700_000_300.0,
                        },
                        {
                            "policy": "ops-block",
                            "action": "escalate",
                            "count": 1,
                            "total_duration_ms": 45.0,
                            "average_duration_ms": 45.0,
                            "min_duration_ms": 45.0,
                            "max_duration_ms": 45.0,
                            "stage_counts": {"review": 1},
                            "last_triggered_at": 1_700_000_280.0,
                        },
                    ],
                )
                self._observer(second_snapshot)

            return first_snapshot

    collector = _StubCollector()

    def _fake_loader():
        return collector, object()

    monkeypatch.setattr(
        cli_module,
        "_load_policy_metrics_collector_for_cli",
        _fake_loader,
    )

    def _fake_urlopen(
        request, timeout=10.0
    ):  # pragma: no cover - exercised via assertions
        data = request.data.decode("utf-8") if request.data else ""
        content_type = request.get_header("Content-Type") or request.get_header(
            "Content-type"
        )
        webhook_calls.append(
            {
                "url": request.full_url,
                "content_type": content_type,
                "payload": data,
            }
        )

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Response()

    monkeypatch.setattr(cli_module.urllib.request, "urlopen", _fake_urlopen)

    exit_code = main(
        [
            "policy",
            "telemetry",
            "--format",
            "json",
            "--follow",
            "--max-events",
            "1",
            "--webhook",
            "https://example.invalid/webhook",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    outputs = [line for line in captured.out.splitlines() if line.strip()]
    assert len(outputs) == 2
    first = json.loads(outputs[0])
    second = json.loads(outputs[1])

    assert first["counts"] == {"ingest:allow": 1}
    assert second["counts"]["review:escalate"] == 1
    assert second["policy_actions"][1]["policy"] == "ops-block"

    assert len(webhook_calls) == 2
    assert webhook_calls[0]["url"] == "https://example.invalid/webhook"
    assert webhook_calls[0]["content_type"] == "application/json"
    assert webhook_calls[0]["payload"] == outputs[0]
    assert webhook_calls[1]["payload"] == outputs[1]
