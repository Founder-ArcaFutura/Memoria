"""Utilities for scaffolding evaluation suite specifications from the CLI."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SuiteScaffoldConfig:
    """Configuration used to scaffold a new evaluation suite document."""

    suite_id: str
    suite_name: str
    description: str
    use_cases: Sequence[str] = field(default_factory=tuple)
    workspace: str = "workspace"
    scenario_id: str = "scenario"
    scenario_name: str = "New Scenario"
    scenario_description: str = "Describe the workflow being evaluated."
    dataset_kind: str = "fixture"
    dataset_path: str = "datasets/example.jsonl"
    retrieval_queries: Sequence[str] = field(default_factory=tuple)


def _normalise_sequence(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    return [value for value in (str(item).strip() for item in values) if value]


def build_scaffold_payload(config: SuiteScaffoldConfig) -> dict:
    """Return a dictionary suitable for serialising into a suite specification."""

    retrieval_tasks = []
    for query in config.retrieval_queries:
        retrieval_tasks.append(
            {
                "query": query,
                "expected_anchors": [],
                "notes": "Populate expected anchors once ground truth is known.",
            }
        )

    payload = {
        "version": "1",
        "suites": [
            {
                "id": config.suite_id,
                "name": config.suite_name,
                "description": config.description,
                "use_cases": list(config.use_cases) or [config.suite_name],
                "privacy_mix": {
                    "public": 0.5,
                    "private": 0.5,
                },
                "memory_distribution": {
                    "synthetic": 100,
                },
                "scenarios": [
                    {
                        "id": config.scenario_id,
                        "name": config.scenario_name,
                        "description": config.scenario_description,
                        "workspace": config.workspace,
                        "dataset": {
                            "kind": config.dataset_kind,
                            "path": config.dataset_path,
                        },
                        "retrieval_tasks": retrieval_tasks
                        or [
                            {
                                "query": "Describe the primary question tested by this scenario.",
                                "expected_anchors": [],
                                "notes": "Replace with concrete assertions once you have labels.",
                            }
                        ],
                    }
                ],
            }
        ],
        "metadata": {
            "generated_by": "memoria evaluation scaffold",
        },
    }
    return payload


def write_suite_scaffold(
    destination: Path,
    config: SuiteScaffoldConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """Serialise the scaffold payload to ``destination`` as YAML."""

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Destination {destination} already exists. Pass overwrite=True to replace it."
        )

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required to scaffold evaluation suites") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_scaffold_payload(config)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return destination


def parse_retrieval_queries(values: Iterable[str] | None) -> list[str]:
    """Parse user-supplied retrieval query stubs from CLI arguments."""

    return _normalise_sequence(values)
