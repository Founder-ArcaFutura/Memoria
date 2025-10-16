"""Benchmark orchestration utilities for Memoria."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from memoria.core.providers import TaskRouteSpec

logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    """Return a filesystem and namespace safe slug for ``value``."""

    slug = re.sub(r"[^0-9a-zA-Z]+", "-", value).strip("-_").lower()
    return slug or "scenario"


def _estimate_tokens(text: str) -> int:
    """Rudimentary token estimate based on whitespace splitting."""

    if not text:
        return 0
    return max(1, len(text.split()))


def _parse_datetime(value: Any) -> datetime | None:
    """Parse ISO 8601-like values into :class:`datetime`."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise TypeError("timestamp must be a string or datetime instance")
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = f"{cleaned[:-1]}+00:00"
    return datetime.fromisoformat(cleaned)


def _normalise_fallback(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(
            segment.strip()
            for segment in value.split(",")
            if segment and segment.strip()
        )
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)


def _extract_result_text(result: Mapping[str, Any]) -> str:
    for key in ("summary", "content", "text", "processed_data"):
        value = result.get(key)
        if isinstance(value, str):
            return value
    return ""


def _extract_result_score(result: Mapping[str, Any]) -> float:
    for key in ("composite_score", "match_score", "search_score", "importance_score"):
        value = result.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


class BenchmarkConfigurationError(ValueError):
    """Raised when benchmark configuration files are invalid."""


@dataclass(slots=True)
class ProviderMix:
    """Description of provider routing overrides for a benchmark run."""

    name: str
    task_routes: dict[str, dict[str, Any]] = field(default_factory=dict)
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ProviderMix:
        if "name" not in payload:
            raise BenchmarkConfigurationError("provider mix requires a 'name'")
        name = str(payload["name"])
        routes_raw = payload.get("task_routes") or {}
        if not isinstance(routes_raw, Mapping):
            raise BenchmarkConfigurationError(
                f"provider mix '{name}' expected 'task_routes' to be a mapping"
            )
        task_routes: dict[str, dict[str, Any]] = {}
        for task, spec in routes_raw.items():
            if not isinstance(spec, Mapping):
                raise BenchmarkConfigurationError(
                    f"task route for '{task}' in provider mix '{name}' must be a mapping"
                )
            provider = spec.get("provider")
            if not provider:
                raise BenchmarkConfigurationError(
                    f"task route '{task}' in provider mix '{name}' missing 'provider'"
                )
            entry = {
                "provider": str(provider),
            }
            if spec.get("model"):
                entry["model"] = str(spec.get("model"))
            fallback = _normalise_fallback(spec.get("fallback"))
            if fallback:
                entry["fallback"] = list(fallback)
            task_routes[str(task)] = entry
        description = payload.get("description")
        metadata = (
            dict(payload.get("metadata") or {})
            if isinstance(payload.get("metadata"), Mapping)
            else {}
        )
        return cls(
            name=name,
            task_routes=task_routes,
            description=description,
            metadata=metadata,
        )

    @classmethod
    def default(cls) -> ProviderMix:
        return cls(name="configured")

    def to_specs(self) -> dict[str, TaskRouteSpec]:
        specs: dict[str, TaskRouteSpec] = {}
        for task, route in self.task_routes.items():
            specs[task] = TaskRouteSpec(
                provider=route["provider"],
                model=route.get("model"),
                fallback=tuple(route.get("fallback", [])),
            )
        return specs

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "task_routes": self.task_routes,
        }
        if self.description:
            payload["description"] = self.description
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(slots=True)
class RetrievalPolicy:
    """Search parameter overrides for a benchmark run."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RetrievalPolicy:
        if "name" not in payload:
            raise BenchmarkConfigurationError("retrieval policy requires a 'name'")
        name = str(payload["name"])
        params_raw = payload.get("parameters") or {}
        if not isinstance(params_raw, Mapping):
            raise BenchmarkConfigurationError(
                f"retrieval policy '{name}' expected 'parameters' to be a mapping"
            )
        parameters = dict(params_raw)
        description = payload.get("description")
        return cls(name=name, parameters=parameters, description=description)

    @classmethod
    def default(cls) -> RetrievalPolicy:
        return cls(name="default")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "parameters": self.parameters}
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(slots=True, eq=True, frozen=True)
class QueryExpectation:
    """Expected outcomes for a retrieval query."""

    contains: tuple[str, ...] = ()
    memory_ids: tuple[str, ...] = ()
    min_score: float | None = None
    min_results: int | None = None
    top_k: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> QueryExpectation:
        contains: tuple[str, ...]
        raw_contains = payload.get("contains")
        if isinstance(raw_contains, str):
            contains = (raw_contains,)
        elif isinstance(raw_contains, Sequence):
            contains = tuple(str(item) for item in raw_contains)
        else:
            contains = ()

        raw_ids = payload.get("memory_ids")
        if isinstance(raw_ids, str):
            memory_ids = (raw_ids,)
        elif isinstance(raw_ids, Sequence):
            memory_ids = tuple(str(item) for item in raw_ids)
        else:
            memory_ids = ()

        min_score = payload.get("min_score")
        if min_score is not None:
            min_score = float(min_score)

        min_results = payload.get("min_results")
        if min_results is not None:
            min_results = int(min_results)

        top_k = payload.get("top_k")
        if top_k is not None:
            top_k = int(top_k)

        return cls(
            contains=contains,
            memory_ids=memory_ids,
            min_score=min_score,
            min_results=min_results,
            top_k=top_k,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.contains:
            payload["contains"] = list(self.contains)
        if self.memory_ids:
            payload["memory_ids"] = list(self.memory_ids)
        if self.min_score is not None:
            payload["min_score"] = self.min_score
        if self.min_results is not None:
            payload["min_results"] = self.min_results
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        return payload

    def evaluate(
        self,
        results: Sequence[Mapping[str, Any]],
    ) -> tuple[bool, list[str]]:
        """Return ``(success, diagnostics)`` for ``results``."""

        diagnostics: list[str] = []
        limited: Sequence[Mapping[str, Any]]
        if self.top_k is not None:
            limited = list(results)[: self.top_k]
        else:
            limited = results

        success = True
        if self.min_results is not None and len(limited) < self.min_results:
            success = False
            diagnostics.append(
                f"expected at least {self.min_results} results, received {len(limited)}"
            )

        if self.contains:
            lowered = [(_extract_result_text(item) or "").lower() for item in limited]
            for needle in self.contains:
                needle_lower = needle.lower()
                if not any(needle_lower in haystack for haystack in lowered):
                    success = False
                    diagnostics.append(f"missing substring '{needle}'")

        if self.memory_ids:
            available = {
                str(item.get("memory_id") or item.get("id"))
                for item in limited
                if item.get("memory_id") or item.get("id")
            }
            for expected_id in self.memory_ids:
                if expected_id not in available:
                    success = False
                    diagnostics.append(f"missing memory_id '{expected_id}'")

        if self.min_score is not None:
            score = max((_extract_result_score(item) for item in limited), default=0.0)
            if score < self.min_score:
                success = False
                diagnostics.append(
                    f"highest score {score:.3f} below minimum {self.min_score:.3f}"
                )

        return success, diagnostics


@dataclass(slots=True, eq=True, frozen=True)
class ScenarioQuery:
    """Single retrieval query within a scenario."""

    prompt: str
    label: str | None = None
    limit: int | None = None
    expectation: QueryExpectation = field(default_factory=QueryExpectation)
    parameters: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ScenarioQuery:
        if "query" not in payload:
            raise BenchmarkConfigurationError("scenario queries require a 'query'")
        prompt = str(payload["query"])
        label = payload.get("name") or payload.get("label")
        limit = payload.get("limit")
        if limit is not None:
            limit = int(limit)
        params_raw = payload.get("parameters") or {}
        params = tuple(sorted((params_raw or {}).items()))
        expectation_payload = payload.get("expected") or payload.get("expectation")
        if isinstance(expectation_payload, Mapping):
            expectation = QueryExpectation.from_mapping(expectation_payload)
        else:
            expectation = QueryExpectation()
        return cls(
            prompt=prompt,
            label=label,
            limit=limit,
            expectation=expectation,
            parameters=params,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.prompt,
            "parameters": dict(self.parameters),
        }
        if self.label:
            payload["label"] = self.label
        if self.limit is not None:
            payload["limit"] = self.limit
        expectation_payload = self.expectation.to_dict()
        if expectation_payload:
            payload["expectation"] = expectation_payload
        return payload


@dataclass(slots=True)
class IngestRecord:
    """Memory payload staged during the ingest phase."""

    text: str
    anchor: str | None = None
    namespace: str | None = None
    tokens: int | None = None
    timestamp: datetime | None = None
    x_coord: float | None = None
    y_coord: float | None = None
    z_coord: float | None = None
    symbolic_anchors: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> IngestRecord:
        if "text" not in payload:
            raise BenchmarkConfigurationError("ingest records require 'text'")
        text = str(payload["text"])
        anchor = payload.get("anchor")
        namespace = payload.get("namespace")
        tokens = payload.get("tokens")
        if tokens is not None:
            tokens = int(tokens)
        timestamp = _parse_datetime(payload.get("timestamp"))
        x_coord = (
            payload.get("x") if payload.get("x") is not None else payload.get("x_coord")
        )
        if x_coord is not None:
            x_coord = float(x_coord)
        y_coord = (
            payload.get("y") if payload.get("y") is not None else payload.get("y_coord")
        )
        if y_coord is not None:
            y_coord = float(y_coord)
        z_coord = (
            payload.get("z") if payload.get("z") is not None else payload.get("z_coord")
        )
        if z_coord is not None:
            z_coord = float(z_coord)
        raw_anchors = payload.get("symbolic_anchors") or payload.get("anchors")
        if isinstance(raw_anchors, str):
            symbolic_anchors = tuple(
                segment.strip() for segment in raw_anchors.split(",") if segment.strip()
            )
        elif isinstance(raw_anchors, Sequence):
            symbolic_anchors = tuple(str(item) for item in raw_anchors)
        else:
            symbolic_anchors = ()
        metadata_raw = payload.get("metadata")
        metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
        return cls(
            text=text,
            anchor=str(anchor) if anchor is not None else None,
            namespace=str(namespace) if namespace is not None else None,
            tokens=tokens,
            timestamp=timestamp,
            x_coord=x_coord,
            y_coord=y_coord,
            z_coord=z_coord,
            symbolic_anchors=symbolic_anchors,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"text": self.text}
        if self.anchor:
            payload["anchor"] = self.anchor
        if self.namespace:
            payload["namespace"] = self.namespace
        if self.tokens is not None:
            payload["tokens"] = self.tokens
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp.isoformat()
        if self.x_coord is not None:
            payload["x_coord"] = self.x_coord
        if self.y_coord is not None:
            payload["y_coord"] = self.y_coord
        if self.z_coord is not None:
            payload["z_coord"] = self.z_coord
        if self.symbolic_anchors:
            payload["symbolic_anchors"] = list(self.symbolic_anchors)
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(slots=True)
class BenchmarkScenario:
    """Benchmark definition combining ingestion and retrieval phases."""

    name: str
    description: str | None
    ingest_records: tuple[IngestRecord, ...]
    queries: tuple[ScenarioQuery, ...]
    provider_mixes: tuple[ProviderMix, ...]
    retrieval_policies: tuple[RetrievalPolicy, ...]
    concurrency_levels: tuple[int, ...]
    namespace: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> BenchmarkScenario:
        if "name" not in payload:
            raise BenchmarkConfigurationError("scenario requires a 'name'")
        name = str(payload["name"])
        description = payload.get("description")
        namespace = payload.get("namespace")

        ingest_section = payload.get("ingest")
        if isinstance(ingest_section, Mapping):
            records_raw = ingest_section.get("records") or []
        else:
            records_raw = payload.get("ingest_records") or []
        records: list[IngestRecord] = []
        for record_payload in records_raw:
            if not isinstance(record_payload, Mapping):
                raise BenchmarkConfigurationError(
                    f"ingest record in scenario '{name}' must be a mapping"
                )
            records.append(IngestRecord.from_mapping(record_payload))

        query_section = payload.get("queries") or []
        if not isinstance(query_section, Sequence):
            raise BenchmarkConfigurationError(
                f"scenario '{name}' expected 'queries' to be a list"
            )
        queries = [ScenarioQuery.from_mapping(item) for item in query_section]
        if not queries:
            raise BenchmarkConfigurationError(
                f"scenario '{name}' must define at least one query"
            )

        provider_mix_section = payload.get("provider_mixes") or []
        provider_mixes: list[ProviderMix]
        if provider_mix_section:
            if not isinstance(provider_mix_section, Sequence):
                raise BenchmarkConfigurationError(
                    f"scenario '{name}' expected 'provider_mixes' to be a list"
                )
            provider_mixes = [
                ProviderMix.from_mapping(item)
                for item in provider_mix_section
                if isinstance(item, Mapping)
            ]
        else:
            provider_mixes = [ProviderMix.default()]

        retrieval_section = payload.get("retrieval_policies") or []
        if retrieval_section:
            if not isinstance(retrieval_section, Sequence):
                raise BenchmarkConfigurationError(
                    f"scenario '{name}' expected 'retrieval_policies' to be a list"
                )
            retrieval_policies = [
                RetrievalPolicy.from_mapping(item)
                for item in retrieval_section
                if isinstance(item, Mapping)
            ]
        else:
            retrieval_policies = [RetrievalPolicy.default()]

        concurrency_section = payload.get("concurrency")
        if concurrency_section is None:
            concurrency_levels = (1,)
        elif isinstance(concurrency_section, Sequence) and not isinstance(
            concurrency_section, (str, bytes)
        ):
            concurrency_levels = tuple(
                max(1, int(value)) for value in concurrency_section
            )
        else:
            concurrency_levels = (max(1, int(concurrency_section)),)

        return cls(
            name=name,
            description=str(description) if description is not None else None,
            ingest_records=tuple(records),
            queries=tuple(queries),
            provider_mixes=tuple(provider_mixes),
            retrieval_policies=tuple(retrieval_policies),
            concurrency_levels=tuple(concurrency_levels),
            namespace=str(namespace) if namespace is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "ingest": {"records": [record.to_dict() for record in self.ingest_records]},
            "queries": [query.to_dict() for query in self.queries],
            "provider_mixes": [mix.to_dict() for mix in self.provider_mixes],
            "retrieval_policies": [
                policy.to_dict() for policy in self.retrieval_policies
            ],
            "concurrency": list(self.concurrency_levels),
        }
        if self.description:
            payload["description"] = self.description
        if self.namespace:
            payload["namespace"] = self.namespace
        return payload

    @property
    def slug(self) -> str:
        return _slugify(self.name)


@dataclass(slots=True)
class QueryRunResult:
    query: ScenarioQuery
    success: bool
    retrieval_time: float
    scoring_time: float
    tokens: int
    diagnostics: list[str]
    raw_response: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "query": self.query.prompt,
            "label": self.query.label,
            "success": self.success,
            "retrieval_time_seconds": self.retrieval_time,
            "scoring_time_seconds": self.scoring_time,
            "token_estimate": self.tokens,
            "diagnostics": list(self.diagnostics),
        }
        preview_results = self.raw_response.get("results", [])
        if isinstance(preview_results, Sequence):
            payload["results_preview"] = preview_results[
                : self.query.expectation.top_k or 3
            ]
        else:
            payload["results_preview"] = []
        return payload


@dataclass(slots=True)
class ScenarioCombinationResult:
    provider_mix: ProviderMix
    retrieval_policy: RetrievalPolicy
    concurrency: int
    namespace: str
    ingest_metadata: dict[str, Any]
    queries: list[QueryRunResult]
    wall_clock_seconds: float

    def to_dict(self) -> dict[str, Any]:
        total_queries = len(self.queries)
        successes = sum(1 for item in self.queries if item.success)
        latency_sum = sum(item.retrieval_time for item in self.queries)
        scoring_sum = sum(item.scoring_time for item in self.queries)
        tokens_sum = sum(item.tokens for item in self.queries)
        latencies = [item.retrieval_time for item in self.queries]
        mean_latency = latency_sum / total_queries if total_queries else 0.0
        max_latency = max(latencies) if latencies else 0.0
        return {
            "provider_mix": self.provider_mix.to_dict(),
            "retrieval_policy": self.retrieval_policy.to_dict(),
            "concurrency": self.concurrency,
            "namespace": self.namespace,
            "ingest": self.ingest_metadata,
            "queries": [item.to_dict() for item in self.queries],
            "metrics": {
                "total_queries": total_queries,
                "successful_queries": successes,
                "accuracy": (successes / total_queries) if total_queries else 0.0,
                "retrieval_time_seconds": latency_sum,
                "scoring_time_seconds": scoring_sum,
                "wall_clock_seconds": self.wall_clock_seconds,
                "mean_latency_seconds": mean_latency,
                "max_latency_seconds": max_latency,
                "token_estimate": tokens_sum,
            },
        }


@dataclass(slots=True)
class ScenarioRunResult:
    scenario: BenchmarkScenario
    combinations: list[ScenarioCombinationResult]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.scenario.name,
            "description": self.scenario.description,
            "combinations": [item.to_dict() for item in self.combinations],
        }
        return payload


@dataclass(slots=True)
class BenchmarkReport:
    started_at: datetime
    finished_at: datetime
    scenarios: list[ScenarioRunResult]

    def to_dict(self) -> dict[str, Any]:
        duration = self.finished_at - self.started_at
        scenario_payloads = [scenario.to_dict() for scenario in self.scenarios]
        combination_count = sum(len(item.combinations) for item in self.scenarios)
        total_queries = sum(
            combination.to_dict()["metrics"]["total_queries"]
            for scenario in self.scenarios
            for combination in scenario.combinations
        )
        accuracy_values = [
            combo.to_dict()["metrics"]["accuracy"]
            for scenario in self.scenarios
            for combo in scenario.combinations
        ]
        mean_accuracy = (
            sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0
        )
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "scenarios": scenario_payloads,
            "summary": {
                "scenario_count": len(self.scenarios),
                "combination_count": combination_count,
                "total_queries": total_queries,
                "mean_accuracy": mean_accuracy,
            },
        }


class BenchmarkRunner:
    """Execute benchmark scenarios against a Memoria instance."""

    def __init__(
        self,
        memoria: Any,
        scenarios: Sequence[BenchmarkScenario],
        *,
        base_namespace: str | None = None,
    ) -> None:
        self.memoria = memoria
        self.scenarios = list(scenarios)
        self.base_namespace = base_namespace
        self._lock = threading.RLock()

    def run(self) -> BenchmarkReport:
        started = datetime.utcnow()
        scenario_results: list[ScenarioRunResult] = []
        for scenario in self.scenarios:
            scenario_results.append(self._run_scenario(scenario))
        finished = datetime.utcnow()
        return BenchmarkReport(
            started_at=started, finished_at=finished, scenarios=scenario_results
        )

    def _run_scenario(self, scenario: BenchmarkScenario) -> ScenarioRunResult:
        combinations: list[ScenarioCombinationResult] = []
        for provider_mix in scenario.provider_mixes:
            for retrieval_policy in scenario.retrieval_policies:
                for concurrency in scenario.concurrency_levels:
                    combinations.append(
                        self._run_combination(
                            scenario,
                            provider_mix,
                            retrieval_policy,
                            concurrency,
                        )
                    )
        return ScenarioRunResult(scenario=scenario, combinations=combinations)

    @contextmanager
    def _override_provider_mix(self, provider_mix: ProviderMix):
        registry = getattr(self.memoria, "provider_registry", None)
        if registry is None or not provider_mix.task_routes:
            yield
            return

        snapshot = None
        if hasattr(registry, "get_task_routes"):
            try:
                snapshot = dict(registry.get_task_routes())
            except Exception:  # pragma: no cover - defensive
                snapshot = None

        specs = provider_mix.to_specs()
        try:
            registry.set_task_routes(specs)
        except Exception as exc:
            logger.warning(
                "Failed to apply provider mix '%s': %s", provider_mix.name, exc
            )
        try:
            yield
        finally:
            if snapshot is not None:
                try:
                    registry.set_task_routes(snapshot)
                except Exception:
                    logger.debug("Failed to restore provider routes after benchmark")

    @contextmanager
    def _temporary_namespace(self, namespace: str):
        memoria = self.memoria
        original_namespace = getattr(memoria, "namespace", None)
        original_storage_namespace = None
        storage_service = getattr(memoria, "storage_service", None)
        if storage_service is not None:
            original_storage_namespace = getattr(storage_service, "namespace", None)
        memoria.namespace = namespace
        if storage_service is not None:
            storage_service.namespace = namespace
        try:
            yield
        finally:
            memoria.namespace = original_namespace
            if storage_service is not None:
                storage_service.namespace = original_storage_namespace

    def _run_combination(
        self,
        scenario: BenchmarkScenario,
        provider_mix: ProviderMix,
        retrieval_policy: RetrievalPolicy,
        concurrency: int,
    ) -> ScenarioCombinationResult:
        namespace = self._build_namespace(scenario, provider_mix, concurrency)
        ingest_metadata = self._ingest_records(scenario, namespace)
        query_results, wall_clock = self._execute_queries(
            scenario,
            namespace,
            retrieval_policy,
            concurrency,
        )
        return ScenarioCombinationResult(
            provider_mix=provider_mix,
            retrieval_policy=retrieval_policy,
            concurrency=concurrency,
            namespace=namespace,
            ingest_metadata=ingest_metadata,
            queries=query_results,
            wall_clock_seconds=wall_clock,
        )

    def _build_namespace(
        self,
        scenario: BenchmarkScenario,
        provider_mix: ProviderMix,
        concurrency: int,
    ) -> str:
        prefix = (
            self.base_namespace or scenario.namespace or f"benchmark-{scenario.slug}"
        )
        mix_slug = _slugify(provider_mix.name)
        return f"{prefix}::{mix_slug}::c{concurrency}"

    def _ingest_records(
        self, scenario: BenchmarkScenario, namespace: str
    ) -> dict[str, Any]:
        if not scenario.ingest_records:
            return {"records": 0, "token_estimate": 0, "duration_seconds": 0.0}

        start = time.perf_counter()
        stored_ids: list[str] = []
        token_total = 0
        with self._temporary_namespace(namespace):
            for index, record in enumerate(scenario.ingest_records):
                anchor = record.anchor or f"{scenario.slug}-record-{index}"
                tokens = (
                    record.tokens
                    if record.tokens is not None
                    else _estimate_tokens(record.text)
                )
                token_total += tokens
                metadata = dict(record.metadata)
                metadata.setdefault("user_priority", 1.0)
                try:
                    response = self.memoria.store_memory(
                        anchor=anchor,
                        text=record.text,
                        tokens=tokens,
                        timestamp=record.timestamp,
                        x_coord=record.x_coord,
                        y=record.y_coord,
                        z=record.z_coord,
                        symbolic_anchors=list(record.symbolic_anchors) or None,
                        metadata=metadata,
                        namespace=record.namespace or namespace,
                        promotion_weights={"threshold": 0.0, "user_priority": 1.0},
                        return_status=True,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to ingest record %s for scenario '%s': %s",
                        index,
                        scenario.name,
                        exc,
                    )
                    continue
                memory_id = (
                    str(response.get("memory_id"))
                    if isinstance(response, Mapping)
                    else str(response)
                )
                stored_ids.append(memory_id)
        duration = time.perf_counter() - start
        return {
            "records": len(stored_ids),
            "token_estimate": token_total,
            "duration_seconds": duration,
            "memory_ids": stored_ids,
        }

    def _execute_queries(
        self,
        scenario: BenchmarkScenario,
        namespace: str,
        retrieval_policy: RetrievalPolicy,
        concurrency: int,
    ) -> tuple[list[QueryRunResult], float]:
        if concurrency < 1:
            concurrency = 1

        parameters = dict(retrieval_policy.parameters)
        results: list[QueryRunResult] = []
        start_wall = time.perf_counter()

        def run_query(query: ScenarioQuery) -> QueryRunResult:
            payload = dict(parameters)
            payload.update(dict(query.parameters))
            if query.limit is not None:
                payload.setdefault("limit", query.limit)
            payload.setdefault("namespace", namespace)

            with self._temporary_namespace(namespace):
                retrieval_start = time.perf_counter()
                try:
                    response = self.memoria.search_memories(query.prompt, **payload)
                except Exception as exc:
                    logger.error(
                        "Query '%s' in scenario '%s' failed: %s",
                        query.label or query.prompt,
                        scenario.name,
                        exc,
                    )
                    retrieval_time = time.perf_counter() - retrieval_start
                    return QueryRunResult(
                        query=query,
                        success=False,
                        retrieval_time=retrieval_time,
                        scoring_time=0.0,
                        tokens=_estimate_tokens(query.prompt),
                        diagnostics=[f"search failed: {exc}"],
                        raw_response={"results": []},
                    )
                retrieval_time = time.perf_counter() - retrieval_start

            response_results = (
                response.get("results") if isinstance(response, Mapping) else []
            )
            if not isinstance(response_results, Sequence):
                response_results = []
            scoring_start = time.perf_counter()
            success, diagnostics = query.expectation.evaluate(response_results)
            scoring_time = time.perf_counter() - scoring_start
            token_estimate = _estimate_tokens(query.prompt)
            if response_results:
                considered = (
                    response_results[: query.expectation.top_k]
                    if query.expectation.top_k
                    else response_results
                )
                token_estimate += sum(
                    _estimate_tokens(_extract_result_text(item)) for item in considered
                )
            return QueryRunResult(
                query=query,
                success=success,
                retrieval_time=retrieval_time,
                scoring_time=scoring_time,
                tokens=token_estimate,
                diagnostics=diagnostics,
                raw_response=response,
            )

        if concurrency == 1:
            for query in scenario.queries:
                results.append(run_query(query))
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency
            ) as executor:
                futures = [
                    executor.submit(run_query, query) for query in scenario.queries
                ]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            # Preserve original order for readability
            order_index = {query: idx for idx, query in enumerate(scenario.queries)}
            results.sort(key=lambda item: order_index.get(item.query, 0))

        wall_clock = time.perf_counter() - start_wall
        return results, wall_clock


@dataclass(slots=True)
class BenchmarkSuite:
    """Collection of benchmark scenarios loaded from configuration."""

    scenarios: tuple[BenchmarkScenario, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> BenchmarkSuite:
        scenarios_raw = payload.get("scenarios")
        if not isinstance(scenarios_raw, Sequence):
            raise BenchmarkConfigurationError(
                "benchmark configuration must include 'scenarios'"
            )
        scenarios = [
            BenchmarkScenario.from_mapping(item)
            for item in scenarios_raw
            if isinstance(item, Mapping)
        ]
        if not scenarios:
            raise BenchmarkConfigurationError(
                "benchmark configuration contained no scenarios"
            )
        return cls(scenarios=tuple(scenarios))

    @classmethod
    def from_file(cls, path: Path) -> BenchmarkSuite:
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        with path.open("r", encoding="utf-8") as handle:
            if suffix in {".yaml", ".yml"}:
                try:
                    import yaml
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise BenchmarkConfigurationError(
                        "PyYAML is required to load YAML benchmark files"
                    ) from exc
                payload = yaml.safe_load(handle) or {}
            else:
                payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise BenchmarkConfigurationError(
                "benchmark configuration root must be a mapping"
            )
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return {"scenarios": [scenario.to_dict() for scenario in self.scenarios]}


def write_benchmark_ndjson(destination: Path, report: Mapping[str, Any]) -> None:
    """Serialise ``report`` to newline-delimited JSON at ``destination``."""

    summary_entry = {
        "type": "summary",
        "started_at": report.get("started_at"),
        "finished_at": report.get("finished_at"),
        "duration_seconds": report.get("duration_seconds"),
        "summary": report.get("summary", {}),
    }

    scenario_entries: list[dict[str, Any]] = []
    for scenario in report.get("scenarios", []):
        scenario_header = {
            "name": scenario.get("name"),
            "description": scenario.get("description"),
        }
        for combination in scenario.get("combinations", []):
            scenario_entries.append(
                {
                    "type": "combination",
                    "scenario": scenario_header,
                    **combination,
                }
            )

    with destination.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary_entry) + "\n")
        for entry in scenario_entries:
            handle.write(json.dumps(entry) + "\n")
