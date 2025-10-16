"""Utilities for loading evaluation suite specifications."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

try:
    import yaml
except (
    ImportError
) as exc:  # pragma: no cover - handled at runtime where dependency missing
    raise ImportError(
        "PyYAML is required to load evaluation suite specifications"
    ) from exc


@dataclass(slots=True)
class ScenarioSpec:
    """Definition of a single evaluation scenario."""

    id: str
    name: str
    description: str
    workspace: str
    dataset: Mapping[str, Any]
    retrieval_tasks: list[Mapping[str, Any]] = field(default_factory=list)
    ingest_profile: Mapping[str, Any] = field(default_factory=dict)
    privacy_expectations: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationSuite:
    """Collection of scenarios with shared evaluation goals."""

    id: str
    name: str
    description: str
    use_cases: list[str]
    privacy_mix: Mapping[str, float]
    memory_distribution: Mapping[str, Any]
    scenarios: list[ScenarioSpec]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationSpec:
    """Top-level container for all evaluation suites."""

    version: str
    suites: dict[str, EvaluationSuite]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def get_suite(self, suite_id: str) -> EvaluationSuite:
        try:
            return self.suites[suite_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown evaluation suite: {suite_id}") from exc

    def iter_scenarios(self) -> Iterable[ScenarioSpec]:
        for suite in self.suites.values():
            yield from suite.scenarios


def _load_document(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(
            f"Unsupported spec format '{path.suffix}'. Use YAML or JSON files."
        )
    if not isinstance(data, Mapping):
        raise ValueError("Evaluation spec root must be a mapping")
    return data


def _parse_scenario(raw: Mapping[str, Any]) -> ScenarioSpec:
    required = {"id", "name", "description", "workspace", "dataset"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Scenario definition missing keys: {sorted(missing)}")

    dataset = raw["dataset"]
    if not isinstance(dataset, Mapping):
        raise ValueError("Scenario dataset definition must be a mapping")

    retrieval = raw.get("retrieval_tasks", [])
    if not isinstance(retrieval, list):
        raise ValueError("Scenario retrieval_tasks must be a list")

    return ScenarioSpec(
        id=str(raw["id"]),
        name=str(raw["name"]),
        description=str(raw["description"]),
        workspace=str(raw["workspace"]),
        dataset=dataset,
        retrieval_tasks=[task for task in retrieval if isinstance(task, Mapping)],
        ingest_profile=raw.get("ingest_profile", {}),
        privacy_expectations=raw.get("privacy_expectations", {}),
        metadata=raw.get("metadata", {}),
    )


def _parse_suite(raw: Mapping[str, Any]) -> EvaluationSuite:
    required = {
        "id",
        "name",
        "description",
        "use_cases",
        "privacy_mix",
        "memory_distribution",
        "scenarios",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Evaluation suite missing keys: {sorted(missing)}")

    scenarios_raw = raw["scenarios"]
    if not isinstance(scenarios_raw, Iterable):
        raise ValueError("Evaluation suite scenarios must be a list")

    scenarios = [
        _parse_scenario(item) for item in scenarios_raw if isinstance(item, Mapping)
    ]
    if not scenarios:
        raise ValueError("Evaluation suite must contain at least one scenario")

    use_cases = raw.get("use_cases", [])
    if not isinstance(use_cases, list):
        raise ValueError("Evaluation suite use_cases must be a list")

    privacy_mix = raw.get("privacy_mix", {})
    if not isinstance(privacy_mix, Mapping):
        raise ValueError("Evaluation suite privacy_mix must be a mapping")

    memory_distribution = raw.get("memory_distribution", {})
    if not isinstance(memory_distribution, Mapping):
        raise ValueError("Evaluation suite memory_distribution must be a mapping")

    return EvaluationSuite(
        id=str(raw["id"]),
        name=str(raw["name"]),
        description=str(raw["description"]),
        use_cases=[str(item) for item in use_cases],
        privacy_mix=privacy_mix,
        memory_distribution=memory_distribution,
        scenarios=scenarios,
        metadata=raw.get("metadata", {}),
    )


def _parse_spec(data: Mapping[str, Any]) -> EvaluationSpec:
    version = str(data.get("version", "1"))
    suites_raw = data.get("suites", [])
    if not isinstance(suites_raw, Iterable):
        raise ValueError("Evaluation spec suites must be a list")

    suites: dict[str, EvaluationSuite] = {}
    for entry in suites_raw:
        if not isinstance(entry, Mapping):
            continue
        suite = _parse_suite(entry)
        suites[suite.id] = suite

    if not suites:
        raise ValueError("Evaluation spec must contain at least one suite")

    return EvaluationSpec(
        version=version, suites=suites, metadata=data.get("metadata", {})
    )


def load_spec_from_path(path: str | Path) -> EvaluationSpec:
    """Load an evaluation spec from ``path``."""

    resolved = Path(path).resolve()
    data = _load_document(resolved)
    return _parse_spec(data)


def load_default_spec(resource_name: str = "default_suites.yaml") -> EvaluationSpec:
    """Load the default evaluation spec packaged with Memoria."""

    package = "memoria.evaluation.specs"
    with resources.as_file(
        resources.files(package).joinpath(resource_name)
    ) as spec_path:
        return load_spec_from_path(spec_path)
