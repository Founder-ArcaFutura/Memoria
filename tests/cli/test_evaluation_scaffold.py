from __future__ import annotations

import yaml

from memoria.cli_support.evaluation_scaffold import (
    SuiteScaffoldConfig,
    parse_retrieval_queries,
    write_suite_scaffold,
)


def test_write_suite_scaffold_generates_yaml(tmp_path) -> None:
    destination = tmp_path / "suite.yaml"
    config = SuiteScaffoldConfig(
        suite_id="demo",
        suite_name="Demo Suite",
        description="Smoke test suite",
        use_cases=("support",),
        workspace="demo-workspace",
        scenario_id="demo-scenario",
        scenario_name="Scenario One",
        scenario_description="Covers the primary retrieval flow.",
        dataset_kind="fixture",
        dataset_path="datasets/demo.jsonl",
        retrieval_queries=("How do I escalate a ticket?",),
    )

    result_path = write_suite_scaffold(destination, config, overwrite=False)
    assert result_path == destination
    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
    suite = payload["suites"][0]
    assert suite["id"] == "demo"
    assert suite["use_cases"] == ["support"]
    scenario = suite["scenarios"][0]
    assert scenario["id"] == "demo-scenario"
    assert scenario["dataset"]["path"] == "datasets/demo.jsonl"
    assert scenario["retrieval_tasks"][0]["query"] == "How do I escalate a ticket?"


def test_parse_retrieval_queries_normalises_inputs() -> None:
    values = parse_retrieval_queries([" How? ", "", "Why?"])
    assert values == ["How?", "Why?"]
