"""Materialise evaluation scenarios into ephemeral workspaces."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

from .datasets import DATASET_GENERATORS, write_jsonl
from .spec import EvaluationSpec, EvaluationSuite, ScenarioSpec, load_default_spec


class ScenarioLoaderError(RuntimeError):
    """Raised when a scenario cannot be materialised."""


@dataclass(slots=True)
class MaterializedScenario:
    """Filesystem artefacts representing a scenario."""

    spec: ScenarioSpec
    workspace_path: Path
    dataset_path: Path
    manifest_path: Path


@dataclass(slots=True)
class MaterializedSuite:
    """Container returned by :func:`materialize_suite`."""

    suite: EvaluationSuite
    root: Path
    scenarios: list[MaterializedScenario]

    def iter_datasets(self) -> Iterable[Path]:
        for scenario in self.scenarios:
            yield scenario.dataset_path


def _resolve_dataset_resource(path: str) -> Path:
    package = "memoria.evaluation.starter_datasets"
    with resources.as_file(resources.files(package).joinpath(path)) as dataset_path:
        if not dataset_path.exists():
            raise ScenarioLoaderError(f"Unknown dataset resource: {path}")
        return dataset_path


def _copy_fixture(dataset: dict[str, Any], destination: Path) -> Path:
    relative_path = dataset.get("path")
    if not isinstance(relative_path, str):
        raise ScenarioLoaderError("Fixture dataset requires a 'path' entry")
    source = _resolve_dataset_resource(relative_path)
    destination_path = destination / Path(relative_path).name
    shutil.copy2(source, destination_path)
    return destination_path


def _generate_dataset(dataset: dict[str, Any], destination: Path) -> Path:
    generator_name = dataset.get("generator")
    if not isinstance(generator_name, str):
        raise ScenarioLoaderError("Generated dataset requires a 'generator' entry")

    try:
        generator = DATASET_GENERATORS[generator_name]
    except KeyError as exc:
        raise ScenarioLoaderError(
            f"Unknown dataset generator: {generator_name}"
        ) from exc

    parameters = dataset.get("parameters", {})
    records = generator(parameters)
    if not records:
        raise ScenarioLoaderError(
            f"Generator '{generator_name}' did not return any records"
        )

    filename = dataset.get("filename") or f"{generator_name}.jsonl"
    destination_path = destination / filename
    write_jsonl(records, destination_path)
    return destination_path


def _materialize_dataset(dataset: dict[str, Any], destination: Path) -> Path:
    dataset_type = dataset.get("kind", "fixture")
    destination.mkdir(parents=True, exist_ok=True)
    if dataset_type == "fixture":
        return _copy_fixture(dataset, destination)
    if dataset_type == "generated":
        return _generate_dataset(dataset, destination)
    raise ScenarioLoaderError(f"Unsupported dataset kind: {dataset_type}")


def _write_manifest(
    scenario: ScenarioSpec, dataset_path: Path, destination: Path
) -> Path:
    manifest = {
        "scenario_id": scenario.id,
        "name": scenario.name,
        "description": scenario.description,
        "workspace": scenario.workspace,
        "dataset": dataset_path.name,
        "retrieval_tasks": scenario.retrieval_tasks,
        "ingest_profile": scenario.ingest_profile,
        "privacy_expectations": scenario.privacy_expectations,
        "metadata": scenario.metadata,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    manifest_path = destination / "scenario_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def materialize_suite(
    suite_id: str,
    *,
    spec: EvaluationSpec | None = None,
    output_root: str | Path | None = None,
    keep_root: bool = True,
) -> MaterializedSuite:
    """Materialise scenarios for ``suite_id`` into an ephemeral workspace tree."""

    evaluation_spec = spec or load_default_spec()
    suite = evaluation_spec.get_suite(suite_id)

    if output_root is None:
        root_dir = Path(tempfile.mkdtemp(prefix=f"memoria-eval-{suite_id}-"))
        root = root_dir
    else:
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)

    scenarios: list[MaterializedScenario] = []
    for scenario in suite.scenarios:
        workspace_path = root / scenario.workspace
        workspace_path.mkdir(parents=True, exist_ok=True)

        dataset_dir = workspace_path / "datasets"
        dataset_path = _materialize_dataset(dict(scenario.dataset), dataset_dir)
        manifest_path = _write_manifest(scenario, dataset_path, workspace_path)

        scenarios.append(
            MaterializedScenario(
                spec=scenario,
                workspace_path=workspace_path,
                dataset_path=dataset_path,
                manifest_path=manifest_path,
            )
        )

    if not keep_root and output_root is None:
        # When the caller does not request retention we still need to keep the
        # directory alive for the object lifetime.  Deletion should be handled
        # by the caller explicitly to avoid removing the tree too early.
        root.touch(exist_ok=True)

    return MaterializedSuite(suite=suite, root=root, scenarios=scenarios)
