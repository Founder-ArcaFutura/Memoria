from __future__ import annotations

import json

from memoria.cli_support.policy_tooling import simulate_policies
from memoria.config.settings import RetentionPolicyAction, RetentionPolicyRuleSettings


def test_simulate_policies_honours_zero_values(tmp_path) -> None:
    rule = RetentionPolicyRuleSettings(
        name="ops-safeguard",
        namespaces=["ops/*"],
        privacy_ceiling=5.0,
        importance_floor=0.5,
        action=RetentionPolicyAction.BLOCK,
    )

    samples = [
        {
            "memory_id": "mem-0",
            "namespace": "ops/core",
            "y_coord": 0,
            "privacy": 12,
            "importance_score": 0,
            "importance": 1,
            "reference_time": "2025-01-01T00:00:00Z",
        }
    ]

    sample_path = tmp_path / "samples.json"
    sample_path.write_text(json.dumps(samples), encoding="utf-8")

    reports = simulate_policies([rule], sample_path=sample_path)

    assert len(reports) == 1
    report = reports[0]
    assert report.total_samples == 1
    assert report.hits and report.hits[0].reasons == ("importance_floor",)


def test_simulate_policies_falls_back_to_importance_field(tmp_path) -> None:
    rule = RetentionPolicyRuleSettings(
        name="ops-safeguard",
        namespaces=["ops/*"],
        privacy_ceiling=5.0,
        importance_floor=0.5,
        action=RetentionPolicyAction.BLOCK,
    )

    samples = [
        {
            "memory_id": "mem-1",
            "namespace": "ops/core",
            "privacy": 0,
            "importance": 0.2,
            "reference_time": "2025-01-01T00:00:00Z",
        }
    ]

    sample_path = tmp_path / "samples.json"
    sample_path.write_text(json.dumps(samples), encoding="utf-8")

    reports = simulate_policies([rule], sample_path=sample_path)

    assert len(reports) == 1
    report = reports[0]
    assert report.total_samples == 1
    assert report.hits and report.hits[0].reasons == ("importance_floor",)
