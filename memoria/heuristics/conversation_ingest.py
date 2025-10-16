"""Heuristic ingestion helpers for conversational turns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..schemas import canonicalize_symbolic_anchors
from ..storage.service import StorageService
from .manual_promotion import (
    PromotionDecision,
    StagedManualMemory,
    score_staged_memory,
)


@dataclass
class HeuristicConversationResult:
    """Container describing the result of a heuristic ingest."""

    staged: StagedManualMemory
    decision: PromotionDecision
    summary: str
    anchor: str
    symbolic_anchors: list[str]
    emotional_intensity: float | None
    related_candidates: list[dict[str, Any]] = field(default_factory=list)


def _safe_datetime(value: Any) -> datetime | None:
    """Attempt to coerce a timestamp-like value into ``datetime``."""

    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    return None


def _slugify_anchor(*candidates: str) -> str:
    """Generate a fallback anchor from provided text candidates."""

    for candidate in candidates:
        if not candidate:
            continue
        words = re.findall(r"[A-Za-z0-9]+", candidate.lower())
        if words:
            return "_".join(words[:3])
    return "conversation_context"


def _collect_symbolic_anchors(
    metadata: dict[str, Any] | None,
    anchor: str,
    user_input: str,
    ai_output: str,
) -> list[str]:
    """Aggregate symbolic anchors from metadata and lightweight heuristics."""

    anchors: list[str] = []
    if metadata:
        for key in ("symbolic_anchors", "anchors", "tags"):
            if key in metadata:
                anchors.extend(canonicalize_symbolic_anchors(metadata.get(key)) or [])

        topic = metadata.get("topic") if isinstance(metadata, dict) else None
        if isinstance(topic, str) and topic.strip():
            anchors.append(topic.strip())

    for text in (user_input, ai_output):
        if not text:
            continue
        hashtags = re.findall(r"#(\w+)", text)
        for tag in hashtags:
            anchors.append(tag)

    anchors = canonicalize_symbolic_anchors(anchors) or []
    if anchor and anchor not in anchors:
        anchors.append(anchor)
    return anchors


def _derive_anchor(
    metadata: dict[str, Any] | None,
    user_input: str,
    ai_output: str,
) -> tuple[str, bool]:
    """Determine the primary anchor for a conversation turn."""

    if metadata:
        explicit = metadata.get("anchor")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip(), True

        symbolic = canonicalize_symbolic_anchors(metadata.get("symbolic_anchors"))
        if symbolic:
            return symbolic[0], True

        for key in ("topic", "title", "label"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), True

    return _slugify_anchor(user_input, ai_output), False


def _determine_summary(
    user_input: str, ai_output: str, metadata: dict[str, Any] | None
) -> str:
    """Build a compact summary representing the conversational turn."""

    if metadata:
        summary = metadata.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()

    user_part = user_input.strip()
    ai_part = ai_output.strip()

    if user_part and ai_part:
        return f"User: {user_part}\nAssistant: {ai_part}"
    return user_part or ai_part or ""


def _estimate_tokens(summary: str, metadata: dict[str, Any] | None) -> int:
    """Estimate token count for heuristic staging."""

    if metadata and "tokens_used" in metadata:
        try:
            tokens = int(metadata["tokens_used"])
            if tokens > 0:
                return tokens
        except (TypeError, ValueError):
            pass

    words = summary.split()
    return max(1, len(words))


def process_conversation_turn(
    storage_service: StorageService,
    *,
    chat_id: str,
    user_input: str,
    ai_output: str,
    metadata: dict[str, Any] | None = None,
    promotion_weights: dict[str, float] | None = None,
) -> HeuristicConversationResult:
    """Stage a conversational turn and return heuristic scoring details."""

    metadata_copy: dict[str, Any] = dict(metadata or {})

    anchor, anchor_explicit = _derive_anchor(metadata_copy, user_input, ai_output)
    summary = _determine_summary(user_input, ai_output, metadata_copy)
    tokens = _estimate_tokens(summary, metadata_copy)

    timestamp = _safe_datetime(metadata_copy.get("timestamp"))
    x_coord = metadata_copy.get("x_coord")
    y_coord = metadata_copy.get("y_coord")
    z_coord = metadata_copy.get("z_coord")

    try:
        x_coord = float(x_coord) if x_coord is not None else None
    except (TypeError, ValueError):
        x_coord = None

    try:
        y_coord = float(y_coord) if y_coord is not None else None
    except (TypeError, ValueError):
        y_coord = None

    try:
        z_coord = float(z_coord) if z_coord is not None else None
    except (TypeError, ValueError):
        z_coord = None

    emotional_intensity = metadata_copy.get("emotional_intensity")
    try:
        if emotional_intensity is not None:
            emotional_intensity = float(emotional_intensity)
    except (TypeError, ValueError):
        emotional_intensity = None

    symbolic_anchors = _collect_symbolic_anchors(
        metadata_copy, anchor, user_input, ai_output
    )

    weight_overrides = dict(promotion_weights or {})
    if not anchor_explicit:
        weight_overrides.setdefault("anchors", 0.05)
        weight_overrides.setdefault("threshold", 0.85)

    staged = storage_service.stage_manual_memory(
        anchor,
        summary,
        tokens,
        timestamp=timestamp,
        x_coord=x_coord,
        y=y_coord,
        z=z_coord,
        symbolic_anchors=symbolic_anchors,
        emotional_intensity=emotional_intensity,
        chat_id=chat_id,
        metadata=metadata_copy if metadata_copy else None,
    )

    relationship_candidates: list[dict[str, Any]] = []
    candidate_keywords: list[str] = []
    if metadata_copy:
        for key in ("keywords", "entities"):
            value = metadata_copy.get(key)
            if isinstance(value, (list, tuple)):
                candidate_keywords.extend(str(item) for item in value if item)
    candidate_keywords.append(anchor)

    get_candidates = getattr(storage_service, "get_relationship_candidates", None)
    if callable(get_candidates):
        try:
            relationship_candidates = get_candidates(
                symbolic_anchors=staged.symbolic_anchors,
                keywords=candidate_keywords,
                topic=metadata_copy.get("topic") if metadata_copy else None,
                exclude_ids=[staged.memory_id],
                limit=5,
            )
        except Exception:
            relationship_candidates = []

    if relationship_candidates:
        staged.metadata.setdefault("relationship_candidates", relationship_candidates)

    decision = score_staged_memory(
        staged,
        weight_overrides or None,
        storage_service=storage_service,
    )

    return HeuristicConversationResult(
        staged=staged,
        decision=decision,
        summary=summary,
        anchor=anchor,
        symbolic_anchors=staged.symbolic_anchors,
        emotional_intensity=emotional_intensity,
        related_candidates=relationship_candidates,
    )
