"""Heuristic helpers for staging and promoting manually-created memories."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger


@dataclass
class StagedManualMemory:
    """Container describing a staged manual memory entry."""

    memory_id: str
    chat_id: str
    namespace: str
    anchor: str
    text: str
    tokens: int
    timestamp: datetime
    x_coord: float | None
    y_coord: float | None
    z_coord: float | None
    symbolic_anchors: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    images: list[dict[str, Any]] | None = None


@dataclass
class PromotionDecision:
    """Result of the manual promotion heuristic."""

    should_promote: bool
    score: float
    threshold: float
    weights: dict[str, float] = field(default_factory=dict)


DEFAULT_WEIGHTS: dict[str, float] = {
    "recency": 0.2,
    "anchors": 0.1,
    "spatial": 0.1,
    "cluster_gravity": 0.25,
    "anchor_recurrence": 0.2,
    "relationship_support": 0.05,
    "user_priority": 0.1,
    "threshold": 0.55,
}


def _compute_recency_score(timestamp: datetime | None, x_coord: float | None) -> float:
    """Return a bounded recency score favouring near-term entries."""

    if timestamp is None and x_coord is None:
        return 0.5

    now = datetime.now(timezone.utc)
    if timestamp is not None:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        delta = abs((now - timestamp).total_seconds())
    else:
        # Interpret x_coord as day-offset from present.
        delta = abs(float(x_coord)) * 24 * 3600

    days = delta / (24 * 3600)
    if days <= 1:
        return 1.0
    if days >= 30:
        return 0.0
    return max(0.0, 1.0 - (days / 30.0))


def _compute_anchor_score(anchor: str, symbolic_anchors: list[str]) -> float:
    """Score symbolic anchors by matching the primary anchor."""

    if not symbolic_anchors:
        return 0.0

    normalized = [a.lower() for a in symbolic_anchors if a]
    anchor_lower = anchor.lower()
    if anchor_lower in normalized:
        return 1.0

    # Partial credit if there is at least one anchor present.
    return min(1.0, len(normalized) / 5.0)


def _compute_spatial_score(x: float | None, y: float | None, z: float | None) -> float:
    """Return a completeness score for provided spatial coordinates."""

    coords = [c for c in (x, y, z) if c is not None]
    if not coords:
        return 0.0
    return len(coords) / 3.0


def _resolve_anchor_candidates(staged: StagedManualMemory) -> list[str]:
    """Return a deduplicated list of symbolic anchors to analyse."""

    ordered: list[str] = []
    for candidate in [staged.anchor, *(staged.symbolic_anchors or [])]:
        if not candidate:
            continue
        normalized = candidate.strip()
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return ordered


def _compute_cluster_gravity_score(
    staged: StagedManualMemory,
    storage_service: Any | None,
) -> float:
    """Fetch cluster proximity data and convert it into a bounded score."""

    if storage_service is None:
        return 0.0

    try:
        anchors = _resolve_anchor_candidates(staged)
        return float(
            storage_service.compute_cluster_gravity(
                x_coord=staged.x_coord,
                y_coord=staged.y_coord,
                z_coord=staged.z_coord,
                anchors=anchors,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug(f"Cluster gravity computation failed: {exc}")
        return 0.0


def _compute_anchor_recurrence_score(
    staged: StagedManualMemory,
    storage_service: Any | None,
) -> float:
    """Return a recurrence score derived from symbolic anchor frequencies."""

    if storage_service is None:
        return 0.0

    try:
        anchors = _resolve_anchor_candidates(staged)
        if not anchors:
            return 0.0

        counts = storage_service.count_anchor_occurrences(
            anchors,
            exclude_memory_ids=[staged.memory_id],
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug(f"Anchor recurrence computation failed: {exc}")
        return 0.0

    if not counts:
        return 0.0

    max_count = max(float(value or 0.0) for value in counts.values())
    if max_count <= 0.0:
        return 0.0

    effective_count = max_count + 1.0
    return 1.0 - math.exp(-effective_count)


def _compute_relationship_support_score(
    staged: StagedManualMemory,
) -> float:
    """Return a score based on existing relationship candidates."""

    candidates = staged.metadata.get("relationship_candidates")
    if not candidates:
        return 0.0

    unique_ids = {
        candidate.get("memory_id")
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("memory_id")
    }
    if not unique_ids:
        return 0.0

    return min(1.0, len(unique_ids) / 5.0)


def score_staged_memory(
    staged: StagedManualMemory,
    weights: dict[str, float] | None = None,
    storage_service: Any | None = None,
) -> PromotionDecision:
    """Evaluate a staged manual memory using heuristic weights."""

    resolved_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        resolved_weights.update({k: v for k, v in weights.items() if v is not None})

    recency = _compute_recency_score(staged.timestamp, staged.x_coord)
    anchor_score = _compute_anchor_score(staged.anchor, staged.symbolic_anchors)
    spatial_score = _compute_spatial_score(
        staged.x_coord, staged.y_coord, staged.z_coord
    )
    try:
        user_priority_raw = staged.metadata.get("user_priority", 0.5)
        user_priority = float(user_priority_raw)
    except (TypeError, ValueError):
        user_priority = 0.5
    user_priority = max(0.0, min(1.0, user_priority))
    cluster_gravity = _compute_cluster_gravity_score(staged, storage_service)
    anchor_recurrence = _compute_anchor_recurrence_score(staged, storage_service)
    relationship_support = _compute_relationship_support_score(staged)

    factors = {
        "recency": recency,
        "anchors": anchor_score,
        "spatial": spatial_score,
        "cluster_gravity": max(0.0, min(1.0, cluster_gravity)),
        "anchor_recurrence": max(0.0, min(1.0, anchor_recurrence)),
        "user_priority": user_priority,
        "relationship_support": max(0.0, min(1.0, relationship_support)),
    }

    threshold = float(resolved_weights.get("threshold", DEFAULT_WEIGHTS["threshold"]))
    priority_bias = max(0.0, user_priority - 0.5)
    if priority_bias > 0.0:
        threshold = max(0.4, threshold - (priority_bias * 0.2))

    weight_total = sum(resolved_weights.get(key, 0.0) for key in factors)
    if weight_total <= 0:
        weight_total = 1.0

    aggregate = 0.0
    for key, value in factors.items():
        aggregate += value * resolved_weights.get(key, 0.0)

    score = aggregate / weight_total
    should_promote = score >= threshold

    return PromotionDecision(
        should_promote=should_promote,
        score=score,
        threshold=threshold,
        weights={
            key: resolved_weights.get(key, 0.0)
            for key in (
                "recency",
                "anchors",
                "spatial",
                "cluster_gravity",
                "anchor_recurrence",
                "relationship_support",
                "user_priority",
            )
        },
    )
