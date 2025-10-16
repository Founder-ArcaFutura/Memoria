"""Core enforcement primitives used to gate sensitive workflows."""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from loguru import logger

from ..utils.exceptions import MemoriaError


class EnforcementStage(Enum):
    """Workflow stages where policy checks can run."""

    INGESTION = "ingestion"
    RETRIEVAL = "retrieval"
    SYNC = "sync"


class PolicyAction(Enum):
    """Supported policy outcomes."""

    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class PolicyDecision:
    """Result returned by a policy hook."""

    action: PolicyAction
    message: str | None = None
    code: str | None = None
    redactions: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    policy_name: str | None = None

    @classmethod
    def allow(cls) -> PolicyDecision:
        return cls(action=PolicyAction.ALLOW)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": self.action.value,
            "code": self.code or "policy_violation",
        }
        if self.message:
            payload["message"] = self.message
        if self.policy_name:
            payload["policy_name"] = self.policy_name
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self.redactions:
            payload["redactions"] = _normalise_redaction_payload(self.redactions)
        return payload


@dataclass(frozen=True)
class PolicyViolationError(MemoriaError):
    """Raised when a policy explicitly blocks a workflow."""

    decision: PolicyDecision
    stage: EnforcementStage | None = None

    def __init__(
        self,
        decision: PolicyDecision,
        *,
        stage: EnforcementStage | None = None,
    ) -> None:
        message = decision.message or "Policy enforcement blocked the request"
        super().__init__(message)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "stage", stage)

    def to_payload(self) -> dict[str, Any]:
        payload = self.decision.to_payload()
        if self.stage is not None:
            payload["stage"] = self.stage.value
        return payload


@dataclass(frozen=True)
class RedactionResult:
    """Represents the edits applied to a payload during redaction."""

    replaced: dict[str, Any]
    removed: set[str]


PolicyHook = Callable[[EnforcementStage, Mapping[str, Any]], PolicyDecision | None]


@dataclass
class _PolicyMetric:
    """Aggregated telemetry for a policy/action pair."""

    count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float | None = None
    max_duration_ms: float | None = None
    stage_counts: Counter[str] = field(default_factory=Counter)
    last_triggered_at: float | None = None

    def record(
        self,
        *,
        stage: EnforcementStage,
        duration_ms: float | None,
        timestamp: float,
    ) -> None:
        self.count += 1
        self.stage_counts[stage.value] += 1
        if duration_ms is not None:
            self.total_duration_ms += duration_ms
            if self.min_duration_ms is None:
                self.min_duration_ms = duration_ms
            else:
                self.min_duration_ms = min(self.min_duration_ms, duration_ms)
            if self.max_duration_ms is None:
                self.max_duration_ms = duration_ms
            else:
                self.max_duration_ms = max(self.max_duration_ms, duration_ms)
        self.last_triggered_at = timestamp


@dataclass(frozen=True)
class PolicyMetricsSnapshot:
    """Immutable snapshot of telemetry emitted by :class:`PolicyMetricsCollector`."""

    generated_at: float
    counts: dict[str, int]
    stage_counts: dict[str, dict[str, int]]
    policy_actions: list[dict[str, object]]

    def to_dict(
        self,
        *,
        limit: int | None = None,
        include_iso: bool = True,
        round_durations: int | None = 3,
    ) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the snapshot."""

        actions: list[dict[str, object]]
        if limit is not None and limit >= 0:
            actions = self.policy_actions[:limit]
        else:
            actions = list(self.policy_actions)

        serialised_actions: list[dict[str, object]] = []
        for action in actions:
            record = dict(action)
            last_triggered = record.get("last_triggered_at")
            if isinstance(last_triggered, (int, float)):
                record["last_triggered_at"] = (
                    datetime.fromtimestamp(last_triggered, tz=timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
            elif last_triggered is not None:
                record["last_triggered_at"] = str(last_triggered)
            else:
                record["last_triggered_at"] = None

            if round_durations is not None:
                for key in (
                    "average_duration_ms",
                    "min_duration_ms",
                    "max_duration_ms",
                    "total_duration_ms",
                ):
                    value = record.get(key)
                    if value is not None:
                        record[key] = round(float(value), round_durations)

            serialised_actions.append(record)

        payload: dict[str, Any] = {
            "generated_at": self.generated_at,
            "counts": {key: int(value) for key, value in self.counts.items()},
            "stage_counts": {
                stage: {action: int(count) for action, count in bucket.items()}
                for stage, bucket in self.stage_counts.items()
            },
            "policy_actions": serialised_actions,
        }

        if include_iso:
            payload["generated_at_iso"] = (
                datetime.fromtimestamp(self.generated_at, tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )

        return payload

    def to_payload(
        self,
        *,
        limit: int | None = None,
        round_durations: int | None = 3,
    ) -> dict[str, Any]:
        """Return a convenience payload for API/CLI consumers."""

        payload = self.to_dict(
            limit=limit, include_iso=True, round_durations=round_durations
        )
        iso_timestamp = payload.pop("generated_at_iso", None)
        policy_actions = payload.get("policy_actions", [])
        packaged: dict[str, Any] = {
            "generated_at": iso_timestamp,
            "generated_at_epoch": payload.get("generated_at"),
            "counts": payload.get("counts", {}),
            "stage_counts": payload.get("stage_counts", {}),
            "policy_actions": policy_actions,
            "total": len(policy_actions),
        }
        return packaged


class PolicyMetricsObserver(Protocol):
    """Interface implemented by sinks that consume policy telemetry snapshots."""

    def publish(
        self, snapshot: PolicyMetricsSnapshot
    ) -> None:  # pragma: no cover - protocol
        ...


ObserverCallback = Callable[[PolicyMetricsSnapshot], None]


class PolicyMetricsCollector:
    """Thread-safe counter for policy decisions and durations with observer support."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._policy_metrics: dict[tuple[str, str], _PolicyMetric] = {}
        self._observers: list[tuple[object, ObserverCallback]] = []

    def increment(
        self,
        stage: EnforcementStage,
        action: PolicyAction,
        *,
        policy: str | None = None,
        duration_ms: float | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Record a policy decision along with optional telemetry."""

        key = f"{stage.value}:{action.value}"
        policy_key = (policy or "unspecified", action.value)
        now = timestamp if timestamp is not None else time.time()

        with self._lock:
            self._counts[key] += 1
            metric = self._policy_metrics.get(policy_key)
            if metric is None:
                metric = _PolicyMetric()
                self._policy_metrics[policy_key] = metric
            metric.record(stage=stage, duration_ms=duration_ms, timestamp=now)
            observers = [callback for _observer, callback in self._observers]

        if observers:
            snapshot = self.capture_snapshot()
            self._notify_observers(snapshot, observers)

    def snapshot(self) -> dict[str, int]:
        """Return flat counts keyed by ``"stage:action"``."""

        with self._lock:
            return dict(self._counts)

    def stage_snapshot(self) -> dict[str, dict[str, int]]:
        """Return counts grouped by enforcement stage."""

        grouped: dict[str, dict[str, int]] = {}
        with self._lock:
            for key, count in self._counts.items():
                try:
                    stage, action = key.split(":", 1)
                except ValueError:
                    stage, action = key, "unknown"
                stage_bucket = grouped.setdefault(stage, {})
                stage_bucket[action] = count
        return grouped

    def policy_action_summary(
        self, *, limit: int | None = None
    ) -> list[dict[str, object]]:
        """Return aggregated metrics for each policy/action pair."""

        with self._lock:
            items = list(self._policy_metrics.items())

        items.sort(key=lambda entry: entry[1].count, reverse=True)
        if limit is not None and limit >= 0:
            items = items[:limit]

        summary: list[dict[str, object]] = []
        for (policy, action), metric in items:
            average = (
                metric.total_duration_ms / metric.count
                if metric.count and metric.total_duration_ms
                else None
            )
            entry: dict[str, object] = {
                "policy": policy,
                "action": action,
                "count": metric.count,
                "total_duration_ms": (
                    metric.total_duration_ms if metric.total_duration_ms else 0.0
                ),
                "average_duration_ms": average,
                "min_duration_ms": metric.min_duration_ms,
                "max_duration_ms": metric.max_duration_ms,
                "stage_counts": dict(metric.stage_counts),
                "last_triggered_at": metric.last_triggered_at,
            }
            summary.append(entry)
        return summary

    def capture_snapshot(self, *, limit: int | None = None) -> PolicyMetricsSnapshot:
        """Capture a consistent snapshot of the current telemetry state."""

        with self._lock:
            counts = dict(self._counts)
            stage_counts: dict[str, dict[str, int]] = {}
            for key, count in self._counts.items():
                try:
                    stage, action = key.split(":", 1)
                except ValueError:
                    stage, action = key, "unknown"
                stage_bucket = stage_counts.setdefault(stage, {})
                stage_bucket[action] = count
            items = list(self._policy_metrics.items())

        items.sort(key=lambda entry: entry[1].count, reverse=True)
        if limit is not None and limit >= 0:
            items = items[:limit]

        summary: list[dict[str, object]] = []
        for (policy, action), metric in items:
            average = (
                metric.total_duration_ms / metric.count
                if metric.count and metric.total_duration_ms
                else None
            )
            entry: dict[str, object] = {
                "policy": policy,
                "action": action,
                "count": metric.count,
                "total_duration_ms": (
                    metric.total_duration_ms if metric.total_duration_ms else 0.0
                ),
                "average_duration_ms": average,
                "min_duration_ms": metric.min_duration_ms,
                "max_duration_ms": metric.max_duration_ms,
                "stage_counts": dict(metric.stage_counts),
                "last_triggered_at": metric.last_triggered_at,
            }
            summary.append(entry)

        return PolicyMetricsSnapshot(
            generated_at=time.time(),
            counts=counts,
            stage_counts=stage_counts,
            policy_actions=summary,
        )

    def register_observer(
        self, observer: PolicyMetricsObserver | ObserverCallback
    ) -> Callable[[], None]:
        """Register an observer and return a callable to detach it."""

        callback = self._coerce_observer(observer)
        with self._lock:
            self._observers.append((observer, callback))

        def _deregister() -> None:
            self.unregister_observer(callback)

        return _deregister

    def unregister_observer(
        self, observer: PolicyMetricsObserver | ObserverCallback
    ) -> None:
        """Remove a previously registered observer if present."""

        with self._lock:
            self._observers = [
                (saved, callback)
                for saved, callback in self._observers
                if saved is not observer and callback is not observer
            ]

    def _notify_observers(
        self, snapshot: PolicyMetricsSnapshot, observers: Sequence[ObserverCallback]
    ) -> None:
        for observer in observers:
            try:
                observer(snapshot)
            except Exception:  # pragma: no cover - defensive observer isolation
                logger.opt(exception=True).warning(
                    "Policy metrics observer %r failed", observer
                )

    @staticmethod
    def _coerce_observer(
        observer: PolicyMetricsObserver | ObserverCallback,
    ) -> ObserverCallback:
        if hasattr(observer, "publish"):
            publish = observer.publish
            if callable(publish):
                return lambda snapshot: publish(snapshot)
        if callable(observer):
            return observer
        raise TypeError("Observer must be callable or implement 'publish'")


@dataclass
class _SpanRecord:
    stage: EnforcementStage
    payload: Mapping[str, Any]
    start_time: float
    decision: PolicyDecision | None = None
    duration_ms: float | None = None


@contextmanager
def _policy_span(stage: EnforcementStage, payload: Mapping[str, Any]):
    record = _SpanRecord(stage=stage, payload=payload, start_time=time.perf_counter())
    logger.debug(
        "policy.enforce.start",
        stage=stage.value,
        payload_keys=sorted(payload.keys()),
    )
    try:
        yield record
    finally:
        duration_ms = (time.perf_counter() - record.start_time) * 1000
        record.duration_ms = duration_ms
        logger.debug(
            "policy.enforce.finish",
            stage=stage.value,
            duration_ms=duration_ms,
            action=(record.decision.action.value if record.decision else None),
            policy=(record.decision.policy_name if record.decision else None),
        )


class PolicyEnforcementEngine:
    """Coordinator that runs registered hooks and tracks metrics."""

    _global_lock = threading.Lock()
    _global_instance: PolicyEnforcementEngine | None = None

    def __init__(
        self,
        *,
        hooks: Sequence[PolicyHook] | None = None,
        metrics: PolicyMetricsCollector | None = None,
    ) -> None:
        self._hooks: list[PolicyHook] = list(hooks or [])
        self._metrics = metrics or PolicyMetricsCollector()

    @classmethod
    def get_global(cls) -> PolicyEnforcementEngine:
        with cls._global_lock:
            if cls._global_instance is None:
                cls._global_instance = cls()
            return cls._global_instance

    def register_hook(self, hook: PolicyHook) -> None:
        self._hooks.append(hook)

    @property
    def metrics(self) -> PolicyMetricsCollector:
        return self._metrics

    def evaluate(
        self, stage: EnforcementStage, payload: Mapping[str, Any] | None
    ) -> PolicyDecision:
        normalized_payload: Mapping[str, Any] = payload or {}
        decision: PolicyDecision = PolicyDecision.allow()

        with _policy_span(stage, normalized_payload) as span:
            for hook in self._hooks:
                try:
                    result = hook(stage, normalized_payload)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.exception(
                        "Policy hook %r failed for stage %s: %s",
                        hook,
                        stage.value,
                        exc,
                    )
                    continue

                if result is None:
                    continue

                if not isinstance(result, PolicyDecision):
                    logger.warning(
                        "Policy hook %r returned unsupported result %r", hook, result
                    )
                    continue

                decision = result
                if decision.action is not PolicyAction.ALLOW:
                    break

            span.decision = decision

        duration_ms = span.duration_ms
        policy_name = (
            decision.policy_name
            or str(normalized_payload.get("policy_name") or "").strip()
        )
        if not policy_name:
            policy_name = None

        self._metrics.increment(
            stage,
            decision.action,
            policy=policy_name,
            duration_ms=duration_ms,
            timestamp=time.time(),
        )
        return decision


def apply_redactions(
    target: MutableMapping[str, Any], decision: PolicyDecision
) -> RedactionResult:
    """Apply redaction instructions to ``target`` and return the applied edits."""

    replacements, removals = _extract_redaction_instructions(decision.redactions)

    for field_name, value in replacements.items():
        target[field_name] = value

    for field_name in removals:
        target.pop(field_name, None)

    return RedactionResult(replaced=replacements, removed=removals)


def _extract_redaction_instructions(
    redactions: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], set[str]]:
    replacements: dict[str, Any] = {}
    removals: set[str] = set()

    if not redactions:
        return replacements, removals

    if not isinstance(redactions, Mapping):
        return replacements, removals

    replace_section = redactions.get("replace")
    remove_section = redactions.get("remove")

    if isinstance(replace_section, Mapping):
        replacements.update(dict(replace_section))
    else:
        # Treat other keys (except "remove") as direct replacements
        for key, value in redactions.items():
            if key == "remove":
                continue
            replacements[key] = value

    if isinstance(remove_section, Mapping):
        for key, flag in remove_section.items():
            if flag:
                removals.add(str(key))
    elif _is_iterable(remove_section):
        removals.update(
            str(value) for value in remove_section if value not in (None, "")
        )

    return replacements, removals


def _normalise_redaction_payload(redactions: Mapping[str, Any]) -> dict[str, Any]:
    replacements, removals = _extract_redaction_instructions(redactions)
    payload: dict[str, Any] = {}
    if replacements:
        payload["replace"] = replacements
    if removals:
        payload["remove"] = sorted(removals)
    return payload


def _is_iterable(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return False
    return isinstance(value, Sequence)


__all__ = [
    "EnforcementStage",
    "PolicyAction",
    "PolicyDecision",
    "PolicyEnforcementEngine",
    "PolicyViolationError",
    "PolicyMetricsObserver",
    "PolicyMetricsCollector",
    "PolicyMetricsSnapshot",
    "RedactionResult",
    "apply_redactions",
]
