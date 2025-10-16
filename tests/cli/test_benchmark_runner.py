from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from memoria.cli_support.benchmark_runner import (
    BenchmarkRunner,
    BenchmarkSuite,
    write_benchmark_ndjson,
)


class _FakeProviderRegistry:
    def __init__(self) -> None:
        self._routes: dict[str, object] = {}
        self._lock = threading.Lock()

    def get_task_routes(self) -> dict[str, object]:
        with self._lock:
            return dict(self._routes)

    def set_task_routes(self, routes: dict[str, object]) -> None:
        with self._lock:
            self._routes = dict(routes)


class _FakeStorageService:
    def __init__(self) -> None:
        self.namespace = "default"


class _FakeMemoria:
    def __init__(self) -> None:
        self.storage_service = _FakeStorageService()
        self.namespace = "default"
        self.provider_registry = _FakeProviderRegistry()
        self._lock = threading.Lock()
        self._store: dict[str, list[dict[str, str]]] = {}

    def store_memory(
        self,
        *,
        anchor: str,
        text: str,
        tokens: int,
        timestamp=None,
        x_coord=None,
        y=None,
        z=None,
        symbolic_anchors=None,
        metadata=None,
        namespace=None,
        promotion_weights=None,
        return_status=False,
        **_: object,
    ):
        target_namespace = namespace or self.namespace
        with self._lock:
            bucket = self._store.setdefault(target_namespace, [])
            memory_id = f"{target_namespace}:{len(bucket)}"
            bucket.append({"memory_id": memory_id, "text": text, "anchor": anchor})
        if return_status:
            return {"memory_id": memory_id, "status": "promoted", "promoted": True}
        return memory_id

    def search_memories(self, query: str, **params: object):
        target_namespace = params.get("namespace") or self.namespace
        results: list[dict[str, object]] = []
        with self._lock:
            for record in self._store.get(target_namespace, []):
                text = record["text"]
                if query.lower() in text.lower():
                    results.append(
                        {
                            "memory_id": record["memory_id"],
                            "summary": text,
                            "match_score": 1.0,
                        }
                    )
        return {"results": results}


def test_benchmark_runner_produces_metrics():
    suite_definition = {
        "scenarios": [
            {
                "name": "basic",
                "ingest": {
                    "records": [
                        {"text": "Alice likes tea", "anchor": "alice"},
                        {"text": "Bob prefers coffee", "anchor": "bob"},
                    ]
                },
                "queries": [
                    {
                        "query": "Alice",
                        "expectation": {"contains": ["alice"], "min_results": 1},
                    },
                    {
                        "query": "coffee",
                        "expectation": {"contains": ["coffee"], "min_results": 1},
                    },
                ],
                "provider_mixes": [
                    {"name": "baseline"},
                    {
                        "name": "override",
                        "task_routes": {"memory_ingest": {"provider": "openai"}},
                    },
                ],
                "retrieval_policies": [
                    {"name": "default", "parameters": {"limit": 5}},
                ],
                "concurrency": [1, 2],
            }
        ]
    }

    suite = BenchmarkSuite.from_mapping(suite_definition)
    memoria = _FakeMemoria()
    runner = BenchmarkRunner(memoria, suite.scenarios, base_namespace="test-suite")
    report = runner.run().to_dict()

    assert report["summary"]["scenario_count"] == 1
    assert report["summary"]["combination_count"] == 4
    assert report["summary"]["total_queries"] == 8
    assert pytest.approx(report["summary"]["mean_accuracy"], 1e-6) == 1.0

    combinations = report["scenarios"][0]["combinations"]
    assert all(combo["metrics"]["accuracy"] == 1.0 for combo in combinations)
    assert {combo["provider_mix"]["name"] for combo in combinations} == {
        "baseline",
        "override",
    }


def test_benchmark_suite_from_file_json(tmp_path: Path):
    config_path = tmp_path / "suite.json"
    config_path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "name": "demo",
                        "queries": [{"query": "x"}],
                        "ingest": {"records": []},
                    }
                ]
            }
        )
    )

    suite = BenchmarkSuite.from_file(config_path)
    assert len(suite.scenarios) == 1
    assert suite.scenarios[0].name == "demo"


def test_ndjson_writer(tmp_path: Path):
    destination = tmp_path / "report.ndjson"
    report = {
        "started_at": "2024-01-01T00:00:00",
        "finished_at": "2024-01-01T00:00:01",
        "duration_seconds": 1.0,
        "summary": {"scenario_count": 1},
        "scenarios": [
            {
                "name": "s1",
                "description": None,
                "combinations": [
                    {
                        "provider_mix": {"name": "baseline"},
                        "retrieval_policy": {"name": "default", "parameters": {}},
                        "concurrency": 1,
                        "namespace": "ns",
                        "ingest": {},
                        "queries": [],
                        "metrics": {"total_queries": 0, "accuracy": 0.0},
                    }
                ],
            }
        ],
    }

    write_benchmark_ndjson(destination, report)

    lines = destination.read_text().splitlines()
    assert lines[0].startswith("{")
    assert any('"type": "combination"' in line for line in lines[1:])
