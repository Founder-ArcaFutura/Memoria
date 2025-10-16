"""Coordinate audit maintenance job."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import or_

from ..database.models import LongTermMemory
from ..storage.service import StorageService


@dataclass(slots=True)
class CoordinateAuditCandidate:
    """A single memory record that requires coordinate verification."""

    memory_id: str
    namespace: str
    text: str
    anchors: Sequence[str]
    x_coord: float | None
    y_coord: float | None
    z_coord: float | None
    timestamp: datetime | None
    created_at: datetime | None
    team_id: str | None
    workspace_id: str | None


class CoordinateAuditResult(BaseModel):
    """Structured response for a single audited memory."""

    memory_id: str = Field(
        description="Identifier of the memory entry that was audited"
    )
    temporal: float | None = Field(
        default=None,
        description="Revised temporal (x axis) coordinate measured in days",
    )
    privacy: float | None = Field(
        default=None,
        description="Revised privacy (y axis) coordinate",
    )
    cognitive: float | None = Field(
        default=None,
        description="Revised cognitive (z axis) coordinate",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score from the auditing model",
    )
    notes: str | None = Field(
        default=None,
        description="Optional free-form commentary from the auditing model",
    )


class CoordinateAuditResponse(BaseModel):
    """Structured payload returned by the auditing model."""

    audits: list[CoordinateAuditResult] = Field(
        default_factory=list,
        description="Audited coordinate results keyed by memory identifier",
    )


class CoordinateAuditJob:
    """Batch coordinate audit that reconciles long-term memory coordinates."""

    METADATA_KEY = "coordinate_audit_last_run"

    def __init__(
        self,
        storage_service: StorageService,
        memory_agent: Any,
        *,
        lookback_days: int = 7,
        batch_size: int = 10,
        completion_event: threading.Event | None = None,
    ) -> None:
        self.storage_service = storage_service
        self.memory_agent = memory_agent
        self.lookback_days = max(1, int(lookback_days))
        self.batch_size = max(1, int(batch_size))
        self._completion_event = completion_event
        self._lock = threading.Lock()
        self._namespace_services: dict[str, StorageService] = {
            getattr(storage_service, "namespace", "default"): storage_service
        }
        self._structured_supported = self._detect_structured_output_support()

    def _detect_structured_output_support(self) -> bool:
        client = getattr(self.memory_agent, "async_client", None)
        if client is None:
            return False
        beta = getattr(client, "beta", None)
        chat = getattr(beta, "chat", None)
        completions = getattr(chat, "completions", None)
        return bool(completions and hasattr(completions, "parse"))

    def _run_async(self, coroutine):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        else:  # pragma: no cover - background schedulers should not share loops
            return loop.run_until_complete(coroutine)

    def _load_last_run(self) -> datetime | None:
        raw_value = None
        try:
            raw_value = self.storage_service.get_service_metadata_value(
                self.METADATA_KEY
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed to load coordinate audit metadata: %s", exc)
            return None

        if not raw_value:
            return None

        try:
            return datetime.fromisoformat(raw_value)
        except ValueError:
            logger.debug("Invalid coordinate audit watermark '%s'", raw_value)
            return None

    def _determine_window_start(self, last_run: datetime | None) -> datetime | None:
        if last_run is None:
            return None
        return datetime.utcnow() - timedelta(days=self.lookback_days)

    def _collect_candidates(
        self, window_start: datetime | None
    ) -> list[CoordinateAuditCandidate]:
        with self.storage_service.db_manager.SessionLocal() as session:
            query = session.query(LongTermMemory)
            if window_start is not None:
                query = query.filter(
                    or_(
                        LongTermMemory.created_at >= window_start,
                        LongTermMemory.timestamp >= window_start,
                    )
                )
            records: Sequence[LongTermMemory] = query.order_by(
                LongTermMemory.created_at.asc()
            ).all()

        candidates: list[CoordinateAuditCandidate] = []
        for record in records:
            anchors: Sequence[str]
            if isinstance(record.symbolic_anchors, Sequence):
                anchors = list(record.symbolic_anchors)
            elif record.symbolic_anchors:
                anchors = [str(record.symbolic_anchors)]
            else:
                anchors = []

            candidates.append(
                CoordinateAuditCandidate(
                    memory_id=record.memory_id,
                    namespace=record.namespace or "default",
                    text=getattr(record, "text", "") or "",
                    anchors=anchors,
                    x_coord=getattr(record, "x_coord", None),
                    y_coord=getattr(record, "y_coord", None),
                    z_coord=getattr(record, "z_coord", None),
                    timestamp=getattr(record, "timestamp", None),
                    created_at=getattr(record, "created_at", None),
                    team_id=getattr(record, "team_id", None),
                    workspace_id=getattr(record, "workspace_id", None),
                )
            )
        return candidates

    def _iter_batches(
        self, candidates: Sequence[CoordinateAuditCandidate]
    ) -> Iterable[Sequence[CoordinateAuditCandidate]]:
        for idx in range(0, len(candidates), self.batch_size):
            yield candidates[idx : idx + self.batch_size]

    def _build_messages(
        self, batch: Sequence[CoordinateAuditCandidate]
    ) -> list[dict[str, str]]:
        details: list[str] = []
        for item in batch:
            created_at = item.created_at.isoformat() if item.created_at else "unknown"
            timestamp = item.timestamp.isoformat() if item.timestamp else "unknown"
            anchors = ", ".join(item.anchors) if item.anchors else "none"
            details.append(
                "\n".join(
                    [
                        f"memory_id: {item.memory_id}",
                        f"namespace: {item.namespace}",
                        f"created_at: {created_at}",
                        f"timestamp: {timestamp}",
                        "text: " + item.text.replace("\n", " "),
                        f"anchors: {anchors}",
                        "prior_coordinates: "
                        f"x={item.x_coord}, y={item.y_coord}, z={item.z_coord}",
                    ]
                )
            )

        user_content = (
            "Audit the following memories and return corrected spatial coordinates "
            "for each entry."
            "\n\n" + "\n\n".join(details)
        )

        system_prompt = (
            "You are a diligent quality auditor tasked with validating Memoria's "
            "spatial coordinates. For every memory you receive you must review the "
            "timestamp, privacy tone, and cognitive framing. Return revised values "
            "only when an adjustment is necessary; otherwise echo the existing "
            "coordinate. Always respond using the provided structured schema."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    async def _audit_batch_async(
        self, batch: Sequence[CoordinateAuditCandidate]
    ) -> list[CoordinateAuditResult]:
        if not self._structured_supported:
            return []

        client = getattr(self.memory_agent, "async_client", None)
        if client is None:
            logger.debug("Coordinate audit skipped: memory agent has no async client")
            return []
        beta = getattr(client, "beta", None)
        chat = getattr(beta, "chat", None)
        completions = getattr(chat, "completions", None)
        if completions is None or not hasattr(completions, "parse"):
            logger.debug("Coordinate audit skipped: structured outputs unavailable")
            self._structured_supported = False
            return []

        messages = self._build_messages(batch)
        model = getattr(self.memory_agent, "model", None) or "gpt-4o-mini"

        try:
            completion = await completions.parse(
                model=model,
                messages=messages,
                response_format=CoordinateAuditResponse,
                temperature=0,
            )
        except Exception as exc:
            logger.warning("Coordinate audit batch failed: %s", exc)
            return []

        choice = completion.choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            return []
        if getattr(message, "refusal", None):
            logger.warning("Coordinate audit refused with message: %s", message.refusal)
            return []

        parsed = getattr(message, "parsed", None)
        if not isinstance(parsed, CoordinateAuditResponse):
            logger.debug("Coordinate audit returned unexpected payload: %s", parsed)
            return []

        return list(parsed.audits)

    def _resolve_service(self, namespace: str) -> StorageService:
        if namespace in self._namespace_services:
            return self._namespace_services[namespace]

        service = StorageService(
            db_manager=self.storage_service.db_manager,
            namespace=namespace,
            search_engine=self.storage_service.search_engine,
            conscious_ingest=self.storage_service.conscious_ingest,
            user_id=getattr(self.storage_service, "_default_user_id", None),
            policy_engine=self.storage_service.policy_engine,
        )
        self._namespace_services[namespace] = service
        return service

    def _apply_results(
        self,
        batch: Sequence[CoordinateAuditCandidate],
        results: Sequence[CoordinateAuditResult],
    ) -> tuple[int, int]:
        updated = 0
        failures = 0
        result_map = {item.memory_id: item for item in results}

        for candidate in batch:
            result = result_map.get(candidate.memory_id)
            if result is None:
                failures += 1
                continue

            updates: dict[str, Any] = {}
            if result.temporal is not None:
                updates["x_coord"] = result.temporal
            if result.privacy is not None:
                updates["y_coord"] = result.privacy
            if result.cognitive is not None:
                updates["z_coord"] = result.cognitive

            if not updates:
                logger.debug(
                    "Coordinate audit produced no changes for %s", candidate.memory_id
                )
                continue

            try:
                service = self._resolve_service(candidate.namespace)
                if service.update_memory(candidate.memory_id, updates):
                    updated += 1
                else:
                    failures += 1
            except Exception as exc:
                failures += 1
                logger.warning(
                    "Failed to apply coordinate audit for %s: %s",
                    candidate.memory_id,
                    exc,
                )

        return updated, failures

    def run(self) -> dict[str, int]:
        run_timestamp = datetime.utcnow()
        if not self._structured_supported:
            logger.debug(
                "Coordinate audit disabled because structured outputs are unavailable"
            )
            self._record_completion(run_timestamp)
            return {"processed": 0, "updated": 0, "failed": 0}

        with self._lock:
            last_run = self._load_last_run()
            window_start = self._determine_window_start(last_run)
            candidates = self._collect_candidates(window_start)
            stats = {"processed": len(candidates), "updated": 0, "failed": 0}

            if not candidates:
                self._record_completion(run_timestamp)
                return stats

            for batch in self._iter_batches(candidates):
                audits: list[CoordinateAuditResult]
                try:
                    audits = self._run_async(self._audit_batch_async(batch))
                except ValidationError as exc:
                    logger.warning(
                        "Coordinate audit response validation failed: %s", exc
                    )
                    stats["failed"] += len(batch)
                    continue

                batch_updated, batch_failures = self._apply_results(batch, audits)
                stats["updated"] += batch_updated
                stats["failed"] += batch_failures

            self._record_completion(run_timestamp)
            return stats

    def _record_completion(self, timestamp: datetime) -> None:
        try:
            self.storage_service.set_service_metadata_value(
                self.METADATA_KEY, timestamp.isoformat()
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed to persist coordinate audit watermark: %s", exc)

        if self._completion_event is not None:
            self._completion_event.set()


class CoordinateAuditScheduler:
    """Thread-based scheduler mirroring other maintenance loops."""

    def __init__(
        self,
        job: CoordinateAuditJob,
        *,
        interval_seconds: int,
    ) -> None:
        self.job = job
        self.interval_seconds = max(60, int(interval_seconds))
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        try:
            self.job.run()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Initial coordinate audit failed: %s", exc)

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds)
        self._thread = None

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.job.run()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Scheduled coordinate audit failed: %s", exc)
