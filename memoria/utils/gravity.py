from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any


class GravityScorer:
    """Score memories using a gravity-inspired formula."""

    def __init__(
        self,
        *,
        anchor_bonus: float = 1.0,
        emotion_weight: float = 0.1,
        epsilon: float = 1e-8,
    ) -> None:
        self.anchor_bonus = anchor_bonus
        self.emotion_weight = emotion_weight
        self.epsilon = epsilon

    @staticmethod
    def _distance(a: Sequence[float], b: Sequence[float]) -> float:
        """Euclidean distance between two embeddings."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))

    def score(
        self,
        query_embedding: Sequence[float],
        memories: Iterable[dict[str, Any]],
        *,
        query_anchors: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return memories sorted by gravitational score."""
        q_anchors = {a.lower() for a in query_anchors or []}
        ranked: list[dict[str, Any]] = []

        for mem in memories:
            mass = float(mem.get("semantic_mass", 1.0))
            mem_embedding = mem.get("embedding") or []
            dist = self._distance(query_embedding, mem_embedding)
            score = mass / (dist + self.epsilon)

            anchors = {str(a).lower() for a in mem.get("anchors", [])}
            if q_anchors and anchors.intersection(q_anchors):
                score += self.anchor_bonus

            emotion = float(mem.get("emotional_intensity", 0.0))
            score += emotion * self.emotion_weight

            ranked.append({**mem, "score": score})

        ranked.sort(key=lambda m: m["score"], reverse=True)
        return ranked


__all__ = ["GravityScorer"]
