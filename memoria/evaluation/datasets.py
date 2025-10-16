"""Synthetic dataset helpers for evaluation scenarios."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def _normalise_privacy_buckets(buckets: dict[str, float]) -> list[tuple[str, float]]:
    total = float(sum(value for value in buckets.values()))
    if total <= 0:
        raise ValueError("Privacy bucket weights must sum to a positive value")
    cumulative = 0.0
    normalised: list[tuple[str, float]] = []
    for key, value in buckets.items():
        cumulative += float(value) / total
        normalised.append((key, min(cumulative, 1.0)))
    if normalised[-1][1] < 1.0:
        normalised[-1] = (normalised[-1][0], 1.0)
    return normalised


def _pick_bucket(weights: list[tuple[str, float]], rng: random.Random) -> str:
    target = rng.random()
    for label, cumulative in weights:
        if target <= cumulative:
            return label
    return weights[-1][0]


_PRIVACY_TO_Y = {
    "public": 12.0,
    "internal": 4.0,
    "restricted": -2.0,
    "private": -12.5,
}

_ANCHOR_AXIS = {
    "knowledge": 9.0,
    "process": 6.0,
    "customer": 3.0,
    "security": 8.0,
    "finance": 7.5,
    "product": 5.5,
}


def _pick_anchor(pool: Iterable[str], rng: random.Random) -> tuple[list[str], float]:
    anchors = list(pool)
    if not anchors:
        anchors = ["general"]
    anchor = rng.choice(anchors)
    secondary = rng.choice(anchors) if len(anchors) > 1 else anchor
    z_axis = _ANCHOR_AXIS.get(anchor, 0.0)
    return sorted({anchor, secondary}), z_axis


def generate_synthetic_privacy_mix(parameters: dict[str, Any]) -> list[Record]:
    """Generate synthetic memories following a privacy distribution."""

    count = int(parameters.get("count", 24))
    seed = int(parameters.get("seed", 1337))
    namespaces = parameters.get("namespaces") or ["synthetic_workspace"]
    anchor_pool = parameters.get("anchor_pool") or ["knowledge", "process"]
    cadence_days = float(parameters.get("cadence_days", 0.5))
    privacy_buckets = parameters.get(
        "privacy_buckets",
        {"public": 0.2, "internal": 0.5, "restricted": 0.2, "private": 0.1},
    )

    if not isinstance(namespaces, Iterable):
        raise ValueError("namespaces parameter must be an iterable")

    rng = random.Random(seed)
    weights = _normalise_privacy_buckets(dict(privacy_buckets))
    start_value = parameters.get("start", "2024-01-01T09:00:00")
    if isinstance(start_value, datetime):
        base_time = start_value
    elif isinstance(start_value, str):
        base_time = datetime.fromisoformat(start_value)
    else:
        raise TypeError("start parameter must be a datetime or ISO-8601 string")

    records: list[Record] = []
    for index in range(count):
        bucket = _pick_bucket(weights, rng)
        anchors, z_axis = _pick_anchor(anchor_pool, rng)
        namespace = rng.choice(list(namespaces))

        y_axis = _PRIVACY_TO_Y.get(bucket, 0.0)
        timestamp = base_time + timedelta(days=index * cadence_days)
        x_axis = -index * cadence_days

        text = parameters.get("template", "{anchor} update {index}").format(
            anchor=anchors[0],
            index=index + 1,
        )

        record: Record = {
            "anchor": f"synthetic_{bucket}_{index}",
            "text": text,
            "tokens": max(8, len(text.split())),
            "x_coord": float(x_axis),
            "y_coord": float(y_axis),
            "z_coord": float(z_axis),
            "symbolic_anchors": anchors,
            "timestamp": timestamp.isoformat() + "Z",
            "importance_score": round(0.4 + (index % 5) * 0.1, 2),
            "retention_type": (
                "conscious" if bucket in {"public", "internal"} else "auto"
            ),
            "namespace": namespace,
            "privacy_bucket": bucket,
        }
        records.append(record)

    return records


def write_jsonl(records: Iterable[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


DATASET_GENERATORS: dict[str, Callable[[dict[str, Any]], list[Record]]] = {
    "synthetic_privacy_mix": generate_synthetic_privacy_mix,
}
