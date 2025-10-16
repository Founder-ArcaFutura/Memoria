"""Retention heuristics for reinforcing or decaying memory importance."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from memoria.database.models import (
    Cluster,
    ClusterMember,
    LongTermMemory,
    ShortTermMemory,
)
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager
from memoria.schemas.constants import Y_AXIS, Z_AXIS

_ALLOWED_POLICY_ACTIONS = {"block", "escalate", "log"}


@dataclass(frozen=True)
class RetentionPolicyRule:
    """In-memory representation of a retention policy rule."""

    name: str
    namespaces: Sequence[str] = field(default_factory=lambda: ("*",))
    privacy_ceiling: float | None = None
    importance_floor: float | None = None
    lifecycle_days: float | None = None
    action: str = "block"
    escalate_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalised = tuple(
            value
            for value in (
                self._normalise_namespace(raw) for raw in (self.namespaces or ("*",))
            )
            if value is not None
        )
        if not normalised:
            normalised = ("*",)
        action = (self.action or "block").strip().lower()
        if action not in _ALLOWED_POLICY_ACTIONS:
            action = "block"
        object.__setattr__(self, "namespaces", normalised)
        object.__setattr__(self, "action", action)

    @staticmethod
    def _normalise_namespace(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        return cleaned.lower()

    def matches(self, namespace: str | None) -> bool:
        target = (namespace or "default").strip().lower() or "default"
        for pattern in self.namespaces:
            if pattern == "*":
                return True
            if pattern.endswith("*") and target.startswith(pattern[:-1]):
                return True
            if target == pattern:
                return True
        return False


@dataclass
class RetentionConfig:
    """Configuration values controlling retention heuristics."""

    decay_half_life_hours: float = 72.0
    reinforcement_bonus: float = 0.05
    privacy_shift: float = 0.5
    importance_floor: float = 0.05
    cluster_gravity_lambda: float = 1.0
    policies: tuple[RetentionPolicyRule, ...] = ()


class MemoryRetentionService:
    """Apply decay and reinforcement rules to stored memories."""

    def __init__(
        self,
        *,
        db_manager: SQLAlchemyDatabaseManager,
        namespace: str,
        config: RetentionConfig,
        cluster_enabled: bool,
        audit_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.db_manager = db_manager
        self.namespace = namespace or "default"
        self.config = config
        self.cluster_enabled = cluster_enabled
        self._policies: tuple[RetentionPolicyRule, ...] = tuple(config.policies or ())
        self._audit_callback = audit_callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_cycle(self) -> None:
        """Execute a single decay/reinforcement pass."""

        now = datetime.utcnow()
        with self.db_manager.SessionLocal() as session:
            changed_ids: set[str] = set()
            changed_ids.update(self._update_memory_group(session, LongTermMemory, now))

            if self.db_manager.enable_short_term:
                changed_ids.update(
                    self._update_memory_group(session, ShortTermMemory, now)
                )

            if self.cluster_enabled and changed_ids:
                self._update_clusters(session, changed_ids, now)

            session.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_memory_group(self, session: Session, model, now: datetime) -> set[str]:
        records: Iterable[LongTermMemory | ShortTermMemory] = (
            session.query(model).filter(model.namespace == self.namespace).all()
        )

        updated_ids: set[str] = set()
        half_life = max(self.config.decay_half_life_hours, 1.0)

        for record in records:
            last_reference: datetime | None = getattr(record, "last_accessed", None)
            if last_reference is None:
                last_reference = getattr(record, "created_at", None)
            if last_reference is None:
                last_reference = now

            age_hours = max((now - last_reference).total_seconds() / 3600.0, 0.0)
            importance = record.importance_score or 0.0

            decayed = importance * math.pow(0.5, age_hours / half_life)
            decayed = max(self.config.importance_floor, decayed)

            reinforcement = self._reinforcement_bonus(
                record.access_count or 0, age_hours
            )
            new_importance = max(
                self.config.importance_floor,
                min(1.0, decayed + reinforcement),
            )

            importance_changed = not math.isclose(
                new_importance, importance, rel_tol=1e-3, abs_tol=1e-4
            )

            proposed_privacy = self._compute_privacy_axis(
                record, reinforcement, age_hours
            )

            allow_importance, allow_privacy = self._enforce_policies(
                record,
                importance_before=importance,
                importance_after=new_importance,
                privacy_before=record.y_coord,
                privacy_after=proposed_privacy,
                age_hours=age_hours,
            )

            applied_importance = False
            if importance_changed and allow_importance:
                record.importance_score = new_importance
                applied_importance = True

            privacy_changed = False
            if allow_privacy and proposed_privacy is not None:
                current_privacy = record.y_coord
                if current_privacy is None or not math.isclose(
                    proposed_privacy,
                    current_privacy,
                    rel_tol=1e-3,
                    abs_tol=1e-3,
                ):
                    record.y_coord = proposed_privacy
                    privacy_changed = True

            if applied_importance or privacy_changed:
                updated_ids.add(record.memory_id)

        return updated_ids

    def _reinforcement_bonus(self, access_count: int, age_hours: float) -> float:
        if access_count <= 0 or self.config.reinforcement_bonus <= 0:
            return 0.0

        freshness_factor = math.exp(
            -age_hours / max(self.config.decay_half_life_hours, 1.0)
        )
        return (
            self.config.reinforcement_bonus
            * math.log1p(access_count)
            * freshness_factor
        )

    def _compute_privacy_axis(
        self,
        record: LongTermMemory | ShortTermMemory,
        reinforcement: float,
        age_hours: float,
    ) -> float:
        base = record.y_coord if record.y_coord is not None else 0.0
        decay_scale = math.pow(
            0.5, age_hours / max(self.config.decay_half_life_hours * 2.0, 1.0)
        )
        decayed_value = base * decay_scale
        adjusted = decayed_value + reinforcement * self.config.privacy_shift
        return max(Y_AXIS.min, min(Y_AXIS.max, adjusted))

    def _enforce_policies(
        self,
        record: LongTermMemory | ShortTermMemory,
        *,
        importance_before: float,
        importance_after: float,
        privacy_before: float | None,
        privacy_after: float | None,
        age_hours: float,
    ) -> tuple[bool, bool]:
        if not self._policies:
            return True, True

        allow_importance = True
        allow_privacy = True
        namespace = getattr(record, "namespace", self.namespace)
        age_days = max(age_hours / 24.0, 0.0)

        for rule in self._policies:
            if not rule.matches(namespace):
                continue

            privacy_violation = False
            importance_violation = False
            lifecycle_violation = False
            violations: list[str] = []

            if (
                privacy_before is not None
                and rule.privacy_ceiling is not None
                and privacy_before > rule.privacy_ceiling + 1e-6
            ):
                privacy_violation = True
                violations.append(

                        f"privacy {privacy_after:.3f} exceeds ceiling "
                        f"{rule.privacy_ceiling:.3f}"

                )

            if (
                rule.importance_floor is not None
                and importance_after < rule.importance_floor - 1e-6
            ):
                importance_violation = True
                violations.append(

                        f"importance {importance_after:.3f} below floor "
                        f"{rule.importance_floor:.3f}"

                )

            if (
                rule.lifecycle_days is not None
                and age_days > rule.lifecycle_days + 1e-6
            ):
                lifecycle_violation = True
                violations.append(

                        f"age {age_days:.2f}d exceeds lifecycle "
                        f"{rule.lifecycle_days:.2f}d"

                )

            if not violations:
                continue

            action = rule.action
            self._emit_policy_audit(
                rule=rule,
                record=record,
                action=action,
                violations=violations,
                importance_before=importance_before,
                importance_after=importance_after,
                privacy_before=privacy_before,
                privacy_after=privacy_after,
                age_days=age_days,
            )

            if action != "log":
                if privacy_violation or lifecycle_violation:
                    allow_privacy = False
                if importance_violation or lifecycle_violation:
                    allow_importance = False
                break

        return allow_importance, allow_privacy

    def _emit_policy_audit(
        self,
        *,
        rule: RetentionPolicyRule,
        record: LongTermMemory | ShortTermMemory,
        action: str,
        violations: list[str],
        importance_before: float,
        importance_after: float,
        privacy_before: float | None,
        privacy_after: float | None,
        age_days: float,
    ) -> None:
        if self._audit_callback is None:
            return

        payload = {
            "memory_id": getattr(record, "memory_id", None),
            "namespace": getattr(record, "namespace", self.namespace),
            "policy_name": rule.name,
            "action": action,
            "escalate_to": rule.escalate_to,
            "violations": violations,
            "importance": {
                "before": importance_before,
                "after": importance_after,
            },
            "privacy": {
                "before": privacy_before,
                "after": privacy_after,
            },
            "age_days": age_days,
            "team_id": getattr(record, "team_id", None),
            "workspace_id": getattr(record, "workspace_id", None),
            "metadata": dict(rule.metadata or {}),
        }

        try:
            self._audit_callback(payload)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(f"Retention policy audit callback failed: {exc}")

    def _update_clusters(
        self, session: Session, changed_ids: set[str], now: datetime
    ) -> None:
        if not changed_ids:
            return

        clusters: list[Cluster] = (
            session.query(Cluster)
            .join(ClusterMember)
            .filter(ClusterMember.memory_id.in_(list(changed_ids)))
            .all()
        )

        if not clusters:
            return

        for cluster in clusters:
            member_ids = [member.memory_id for member in cluster.members]
            if not member_ids:
                continue

            memories: list[LongTermMemory] = (
                session.query(LongTermMemory)
                .filter(
                    LongTermMemory.memory_id.in_(member_ids),
                    LongTermMemory.namespace == self.namespace,
                )
                .all()
            )

            if not memories:
                continue

            avg_importance = sum(m.importance_score or 0.0 for m in memories) / len(
                memories
            )

            centroid_x = self._mean([m.x_coord for m in memories])
            centroid_y = self._mean([m.y_coord for m in memories])
            centroid_z = self._mean([m.z_coord for m in memories])

            previous_update = cluster.last_updated or now
            age_seconds = max((now - previous_update).total_seconds(), 0.0)
            weight = avg_importance * len(memories)
            weight *= math.exp(-self.config.cluster_gravity_lambda * age_seconds)

            cluster.avg_importance = avg_importance
            cluster.weight = weight
            cluster.centroid = {
                "x": centroid_x,
                "y": centroid_y,
                "z": centroid_z,
            }
            if centroid_y is not None:
                cluster.y_centroid = max(Y_AXIS.min, min(Y_AXIS.max, centroid_y))
            if centroid_z is not None:
                cluster.z_centroid = max(Z_AXIS.min, min(Z_AXIS.max, centroid_z))
            cluster.last_updated = now
            cluster.update_count = (cluster.update_count or 0) + 1

    @staticmethod
    def _mean(values: Iterable[float | None]) -> float | None:
        cleaned = [v for v in values if v is not None]
        if not cleaned:
            return None
        return sum(cleaned) / len(cleaned)


class MemoryRetentionScheduler:
    """Background scheduler that periodically runs retention cycles."""

    def __init__(
        self,
        service: MemoryRetentionService,
        *,
        interval_seconds: int,
    ) -> None:
        self.service = service
        self.interval_seconds = max(60, interval_seconds)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Run an immediate cycle for freshness
        try:
            self.service.run_cycle()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Initial retention cycle failed: {exc}")

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds)
        self._thread = None

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.service.run_cycle()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(f"Retention cycle failed: {exc}")
