"""Helpers for rerunning manual ingestion heuristics on staged memories."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from ..database.models import ShortTermMemory
from ..database.sqlalchemy_manager import SQLAlchemyDatabaseManager
from ..heuristics.manual_promotion import (
    PromotionDecision,
    StagedManualMemory,
    score_staged_memory,
)
from ..schemas import canonicalize_symbolic_anchors
from ..storage.service import StorageService
from ..utils.embeddings import generate_embedding, vector_search_enabled
from ..utils.pydantic_models import (
    MemoryClassification,
    MemoryImportanceLevel,
    ProcessedLongTermMemory,
)


def _coerce_datetime(value: Any, fallback: datetime | None) -> datetime | None:
    """Best-effort conversion of stored payload values into ``datetime``."""

    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            ts = fallback
    elif value is None:
        ts = fallback
    else:
        try:
            ts = datetime.fromtimestamp(float(value))
        except (TypeError, ValueError, OSError):
            ts = fallback

    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def _determine_importance(decision: PromotionDecision) -> MemoryImportanceLevel:
    if decision.score >= 0.75:
        return MemoryImportanceLevel.HIGH
    if decision.score >= decision.threshold:
        return MemoryImportanceLevel.MEDIUM
    return MemoryImportanceLevel.LOW


def _build_staged_payload(row: ShortTermMemory) -> StagedManualMemory:
    """Convert a :class:`ShortTermMemory` row into :class:`StagedManualMemory`."""

    processed = row.processed_data or {}
    text = processed.get("text") or row.summary or ""
    try:
        tokens = int(processed.get("tokens", 0))
    except (TypeError, ValueError):
        tokens = 0
    if tokens <= 0:
        tokens = max(1, len(text.split()))

    timestamp = _coerce_datetime(processed.get("timestamp"), row.created_at)
    x_coord = row.x_coord
    if x_coord is None and timestamp is not None:
        x_coord = float((timestamp.date() - datetime.utcnow().date()).days)

    metadata: dict[str, Any] = {
        "short_term_stored": True,
        "user_priority": row.importance_score,
        "category_primary": row.category_primary,
        "namespace": row.namespace,
    }

    return StagedManualMemory(
        memory_id=row.memory_id,
        chat_id=row.chat_id or row.memory_id,
        namespace=row.namespace,
        anchor=processed.get("anchor") or (row.symbolic_anchors or ["manual"])[0],
        text=text,
        tokens=tokens,
        timestamp=timestamp or datetime.now(timezone.utc),
        x_coord=x_coord,
        y_coord=row.y_coord,
        z_coord=row.z_coord,
        symbolic_anchors=canonicalize_symbolic_anchors(row.symbolic_anchors) or [],
        metadata=metadata,
    )


def _iter_manual_staged(
    storage_service: StorageService,
    db_manager: SQLAlchemyDatabaseManager,
    namespace: str,
) -> Iterable[ShortTermMemory]:
    if not getattr(db_manager, "enable_short_term", False):
        return []

    with db_manager.SessionLocal() as session:
        return list(
            session.query(ShortTermMemory)
            .filter(
                ShortTermMemory.namespace == namespace,
                ShortTermMemory.category_primary == "manual_staged",
            )
            .all()
        )


def run_ingestion_pass(
    *,
    storage_service: StorageService,
    db_manager: SQLAlchemyDatabaseManager,
    namespace: str,
    promotion_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Re-evaluate staged short-term memories and promote those that qualify."""

    rows = _iter_manual_staged(storage_service, db_manager, namespace)
    if not rows:
        return []

    results: list[dict[str, Any]] = []
    vector_enabled = vector_search_enabled()
    for row in rows:
        staged = _build_staged_payload(row)
        decision = score_staged_memory(
            staged,
            promotion_weights,
            storage_service=storage_service,
        )

        long_term_id: str | None = None
        if decision.should_promote:
            processed_memory = ProcessedLongTermMemory(
                content=staged.text,
                summary=staged.text,
                classification=MemoryClassification.CONTEXTUAL,
                importance=_determine_importance(decision),
                topic=None,
                entities=[],
                keywords=[],
                is_user_context=False,
                is_preference=False,
                is_skill_knowledge=False,
                is_current_project=False,
                duplicate_of=None,
                supersedes=[],
                related_memories=[],
                conversation_id=staged.chat_id,
                confidence_score=0.85,
                emotional_intensity=(
                    row.processed_data.get("emotional_intensity")
                    if isinstance(row.processed_data, dict)
                    else None
                ),
                x_coord=staged.x_coord,
                y_coord=staged.y_coord,
                z_coord=staged.z_coord,
                symbolic_anchors=staged.symbolic_anchors,
                classification_reason=(
                    f"Daily ingestion approval (score={decision.score:.2f})"
                ),
                promotion_eligible=True,
            )

            if vector_enabled and not processed_memory.embedding:
                processed_memory.embedding = generate_embedding(staged.text)

            try:
                long_term_id = db_manager.store_long_term_memory_enhanced(
                    processed_memory,
                    staged.chat_id,
                    namespace,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(f"Failed to promote {row.memory_id}: {exc}")
                long_term_id = None
            else:
                try:
                    storage_service.transfer_spatial_metadata(
                        staged.memory_id, long_term_id
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error(
                        f"Failed moving spatial metadata for {long_term_id}: {exc}"
                    )
                try:
                    storage_service.remove_short_term_memory(staged.memory_id)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error(
                        f"Failed removing staged short-term {staged.memory_id}: {exc}"
                    )

        results.append(
            {
                "short_term_id": staged.memory_id,
                "long_term_id": long_term_id,
                "promoted": bool(long_term_id),
                "promotion_score": decision.score,
                "threshold": decision.threshold,
                "weights": decision.weights,
                "anchor": staged.anchor,
                "summary": staged.text,
            }
        )

    return results


class ShortTermIngestionService:
    """Service wrapper that coordinates ingestion passes for a namespace."""

    def __init__(
        self,
        *,
        storage_service: StorageService,
        db_manager: SQLAlchemyDatabaseManager,
        namespace: str,
        on_run: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> None:
        self.storage_service = storage_service
        self.db_manager = db_manager
        self.namespace = namespace
        self._on_run = on_run

    def run_ingestion_pass(
        self, *, promotion_weights: dict[str, float] | None = None
    ) -> list[dict[str, Any]]:
        results = run_ingestion_pass(
            storage_service=self.storage_service,
            db_manager=self.db_manager,
            namespace=self.namespace,
            promotion_weights=promotion_weights,
        )
        if self._on_run is not None:
            try:
                self._on_run(results)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(f"Ingestion callback failed: {exc}")
        return results


class DailyIngestionScheduler:
    """Background scheduler that triggers ingestion passes at a fixed cadence."""

    def __init__(
        self,
        service: ShortTermIngestionService,
        *,
        interval_seconds: int = 24 * 60 * 60,
        promotion_weights: dict[str, float] | None = None,
    ) -> None:
        self.service = service
        self.interval_seconds = max(interval_seconds, 60)
        self.promotion_weights = promotion_weights
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        try:
            self.service.run_ingestion_pass(promotion_weights=self.promotion_weights)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Initial ingestion pass failed: {exc}")

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds)
        self._thread = None

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.service.run_ingestion_pass(
                    promotion_weights=self.promotion_weights
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(f"Scheduled ingestion pass failed: {exc}")
