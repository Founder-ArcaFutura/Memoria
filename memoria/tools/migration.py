"""Utilities for exporting and importing Memoria datasets.

The helpers in this module provide a higher level migration workflow that can
be reused by standalone scripts, the :mod:`memoria` CLI, or external
automation.  They wrap the existing :class:`~memoria.storage.service.StorageService`
APIs and database models to provide:

* Rich export payloads that bundle long/short term memories together with
  cluster metadata and relationship graph edges.
* Validation helpers that reuse the storage service's import normalisation so
  previews mirror the real ingest behaviour.
* Dry-run and reconciliation modes that report which records would be
  inserted, skipped, or error before committing to a target database.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from memoria.database.models import (
    Cluster,
    ClusterMember,
    LongTermMemory,
    ShortTermMemory,
)
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager
from memoria.storage.service import StorageService

ISO_SUFFIX = "Z"


@dataclass(slots=True)
class MigrationSnapshot:
    """Container describing the exported dataset."""

    metadata: dict[str, Any]
    payload: dict[str, Any]


@dataclass(slots=True)
class ValidationResult:
    """Normalised import payload and any issues found during validation."""

    normalized_long_term: list[dict[str, Any]]
    skipped: list[dict[str, Any]]
    errors: list[dict[str, Any]]


@dataclass(slots=True)
class MigrationReport:
    """Summary generated after applying or previewing a migration."""

    dry_run: bool
    inserted: dict[str, list[str]]
    skipped: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def _to_datetime(value: Any) -> datetime | None:
    """Parse ISO formatted strings into :class:`~datetime.datetime` objects."""

    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value))
    if isinstance(value, str):
        candidate = value.rstrip(ISO_SUFFIX) if value.endswith(ISO_SUFFIX) else value
        try:
            return datetime.fromisoformat(candidate)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid datetime value: {value!r}") from exc
    raise TypeError(f"Unsupported datetime value: {value!r}")


def _serialize(instance: Any) -> dict[str, Any]:
    """Convert a SQLAlchemy model instance into a JSON-serialisable mapping."""

    mapper = inspect(instance.__class__)
    record: dict[str, Any] = {}
    for column in mapper.columns:
        value = getattr(instance, column.key)
        if isinstance(value, datetime):
            record[column.key] = value.isoformat() + ISO_SUFFIX
        else:
            record[column.key] = value
    return record


def _table_exists(engine: Engine | None, table_name: str) -> bool:
    if engine is None:
        return False
    inspector = inspect(engine)
    try:
        return table_name in set(inspector.get_table_names())
    finally:
        inspector.bind = None  # release references early


def _normalise_namespace(
    value: str | None,
    namespace_map: Mapping[str, str] | None,
    default_namespace: str,
) -> str:
    canonical = (value or default_namespace or "default").strip() or "default"
    if namespace_map:
        return namespace_map.get(canonical, canonical) or "default"
    return canonical


def parse_namespace_mappings(mappings: Iterable[str]) -> dict[str, str]:
    """Parse ``source=target`` strings into a dictionary."""

    parsed: dict[str, str] = {}
    for item in mappings:
        if "=" not in item:
            raise ValueError(
                f"Invalid namespace mapping '{item}'. Expected format 'source=target'."
            )
        source, target = item.split("=", 1)
        source_key = source.strip() or "default"
        target_value = target.strip()
        if not target_value:
            raise ValueError(
                f"Namespace mapping '{item}' is missing a destination namespace."
            )
        parsed[source_key] = target_value
    return parsed


def export_snapshot(
    db_manager: SQLAlchemyDatabaseManager,
    *,
    namespaces: Sequence[str] | None = None,
    include_short_term: bool = True,
    include_clusters: bool = True,
    include_relationships: bool = True,
) -> MigrationSnapshot:
    """Collect memories, clusters, and relationships into a serialisable payload."""

    with db_manager.SessionLocal() as session:
        engine = session.get_bind()
        namespace_filter = list(
            dict.fromkeys(ns.strip() for ns in namespaces or [] if ns)
        )

        def _apply_namespace_filter(query, model) -> Any:
            if namespace_filter and hasattr(model, "namespace"):
                return query.filter(model.namespace.in_(namespace_filter))
            return query

        long_term_query = _apply_namespace_filter(
            session.query(LongTermMemory), LongTermMemory
        )
        long_term_records = [_serialize(row) for row in long_term_query.all()]

        short_term_records: list[dict[str, Any]] = []
        if include_short_term and _table_exists(engine, ShortTermMemory.__tablename__):
            short_term_query = _apply_namespace_filter(
                session.query(ShortTermMemory), ShortTermMemory
            )
            short_term_records = [_serialize(row) for row in short_term_query.all()]

        cluster_records: list[dict[str, Any]] = []
        member_records: list[dict[str, Any]] = []
        if include_clusters and _table_exists(engine, Cluster.__tablename__):
            cluster_records = [_serialize(row) for row in session.query(Cluster).all()]
            member_records = [
                _serialize(row) for row in session.query(ClusterMember).all()
            ]

        relationship_records: list[dict[str, Any]] = []
        if include_relationships and _table_exists(engine, "memory_relationships"):
            rows = session.execute(
                text(
                    "SELECT relationship_id, source_memory_id, target_memory_id, "
                    "relationship_type, strength, reasoning, namespace, created_at "
                    "FROM memory_relationships"
                )
            )
            for row in rows.mappings():
                record = dict(row)
                created_at = record.get("created_at")
                if isinstance(created_at, datetime):
                    record["created_at"] = created_at.isoformat() + ISO_SUFFIX
                relationship_records.append(record)

    namespace_values = {
        entry.get("namespace")
        for entry in long_term_records + short_term_records
        if entry.get("namespace")
    }

    payload = {
        "long_term_memories": long_term_records,
        "short_term_memories": short_term_records,
        "clusters": cluster_records,
        "cluster_members": member_records,
        "relationships": relationship_records,
        "namespaces": sorted(namespace_values),
    }

    metadata = {
        "exported_at": datetime.utcnow().isoformat() + ISO_SUFFIX,
        "database_type": db_manager.database_type,
        "namespaces_filter": namespace_filter,
        "counts": {
            key: len(value) if isinstance(value, list) else 0
            for key, value in payload.items()
        },
    }

    return MigrationSnapshot(metadata=metadata, payload=payload)


def dump_snapshot(snapshot: MigrationSnapshot) -> dict[str, Any]:
    """Render :class:`MigrationSnapshot` instances into a JSON-ready mapping."""

    return {"metadata": snapshot.metadata, "payload": snapshot.payload}


def load_snapshot(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    """Load a snapshot from disk or return a copy of an in-memory mapping."""

    if isinstance(source, Mapping):
        return {
            "metadata": dict(source.get("metadata", {})),
            "payload": dict(source.get("payload", {})),
        }

    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    elif isinstance(source, str):
        candidate_path = Path(source)
        if candidate_path.exists():
            text = candidate_path.read_text(encoding="utf-8")
        else:
            text = source
    else:
        raise TypeError("Unsupported snapshot source")

    import json

    parsed = json.loads(text)
    if not isinstance(parsed, Mapping):
        raise ValueError("Snapshot file must contain a mapping at the top level")

    return {
        "metadata": dict(parsed.get("metadata", {})),
        "payload": dict(parsed.get("payload", {})),
    }


def _prepare_short_term_record(
    record: Mapping[str, Any],
    namespace_map: Mapping[str, str] | None,
    default_namespace: str,
) -> dict[str, Any]:
    prepared = dict(record)
    prepared["namespace"] = _normalise_namespace(
        prepared.get("namespace"), namespace_map, default_namespace
    )
    for field in ("created_at", "expires_at", "last_accessed"):
        try:
            prepared[field] = _to_datetime(prepared.get(field))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field} value for short-term memory") from exc
    return prepared


def _prepare_cluster_record(record: Mapping[str, Any]) -> dict[str, Any]:
    prepared = dict(record)
    if "last_updated" in prepared:
        try:
            prepared["last_updated"] = _to_datetime(prepared.get("last_updated"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid last_updated value for cluster") from exc
    return prepared


def _prepare_cluster_member_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return dict(record)


def _prepare_relationship_record(
    record: Mapping[str, Any],
    namespace_map: Mapping[str, str] | None,
    default_namespace: str,
) -> dict[str, Any]:
    prepared = dict(record)
    prepared["namespace"] = _normalise_namespace(
        prepared.get("namespace"), namespace_map, default_namespace
    )
    try:
        prepared["created_at"] = _to_datetime(prepared.get("created_at"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid created_at value for relationship") from exc
    try:
        strength_value = prepared.get("strength")
        if strength_value is not None:
            prepared["strength"] = float(strength_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid strength value for relationship") from exc
    return prepared


def validate_payload(
    service: StorageService,
    snapshot: Mapping[str, Any],
    *,
    namespace_map: Mapping[str, str] | None = None,
    default_namespace: str | None = None,
    dedupe: bool = True,
) -> ValidationResult:
    """Normalise incoming payloads using the storage service's logic."""

    normalized: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    default_ns = default_namespace or service.namespace or "default"
    namespace_map = dict(namespace_map or {})

    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str, datetime]] = set()

    payload = snapshot.get("payload", {}) if isinstance(snapshot, Mapping) else {}
    memories: Sequence[Mapping[str, Any]] = payload.get("long_term_memories", [])  # type: ignore[assignment]

    for index, record in enumerate(memories or []):
        try:
            normalized_record = service._normalise_import_record(  # type: ignore[attr-defined]
                record,
                namespace_map,
                default_ns,
                index=index,
            )
        except Exception as exc:  # pragma: no cover - defensive
            errors.append({"index": index, "error": str(exc)})
            continue

        mem_id = normalized_record["memory_id"]
        pair = (
            normalized_record["namespace"],
            normalized_record["text"],
            normalized_record["created_at"],
        )

        if dedupe:
            if mem_id in seen_ids:
                skipped.append(
                    {
                        "type": "long_term",
                        "memory_id": mem_id,
                        "reason": "duplicate memory_id in payload",
                    }
                )
                continue
            if pair in seen_pairs:
                skipped.append(
                    {
                        "type": "long_term",
                        "memory_id": mem_id,
                        "reason": "duplicate text/timestamp in payload",
                    }
                )
                continue

        seen_ids.add(mem_id)
        seen_pairs.add(pair)
        normalized.append(normalized_record)

    return ValidationResult(
        normalized_long_term=normalized,
        skipped=skipped,
        errors=errors,
    )


def import_snapshot(
    db_manager: SQLAlchemyDatabaseManager,
    service: StorageService,
    snapshot: Mapping[str, Any],
    *,
    namespace_map: Mapping[str, str] | None = None,
    default_namespace: str | None = None,
    dedupe: bool = True,
    dry_run: bool = False,
    validation: ValidationResult | None = None,
) -> MigrationReport:
    """Import a migration snapshot into the configured database."""

    namespace_map = dict(namespace_map or {})
    default_ns = default_namespace or service.namespace or "default"
    payload: Mapping[str, Any] = (
        snapshot.get("payload", {}) if isinstance(snapshot, Mapping) else {}
    )

    if validation is None:
        validation = validate_payload(
            service,
            snapshot,
            namespace_map=namespace_map,
            default_namespace=default_ns,
            dedupe=dedupe,
        )

    skipped = list(validation.skipped)
    errors = list(validation.errors)
    inserted: dict[str, list[str]] = {
        "long_term": [],
        "short_term": [],
        "clusters": [],
        "cluster_members": [],
        "relationships": [],
    }

    if validation.errors:
        logger.warning("Validation reported %d errors", len(validation.errors))

    # Long-term memories -------------------------------------------------
    long_term_payload = payload.get("long_term_memories", []) or []

    if dry_run:
        with db_manager.SessionLocal() as session:
            for record in validation.normalized_long_term:
                mem_id = record["memory_id"]
                namespace = record["namespace"]

                existing_by_id = (
                    session.query(LongTermMemory.memory_id)
                    .filter(
                        LongTermMemory.memory_id == mem_id,
                        LongTermMemory.namespace == namespace,
                    )
                    .first()
                )
                if existing_by_id is not None and dedupe:
                    skipped.append(
                        {
                            "type": "long_term",
                            "memory_id": mem_id,
                            "reason": "existing memory_id in destination",
                        }
                    )
                    continue

                existing_pair = (
                    session.query(LongTermMemory.memory_id)
                    .filter(
                        LongTermMemory.namespace == namespace,
                        LongTermMemory.searchable_content == record["text"],
                        LongTermMemory.created_at == record["created_at"],
                    )
                    .first()
                )
                if existing_pair is not None and dedupe:
                    skipped.append(
                        {
                            "type": "long_term",
                            "memory_id": mem_id,
                            "reason": "existing text/timestamp in destination",
                        }
                    )
                    continue

                inserted["long_term"].append(mem_id)
    elif long_term_payload:
        result = service.import_memories_bulk(
            long_term_payload,
            namespace_map=namespace_map,
            default_namespace=default_ns,
            skip_existing=dedupe,
        )
        inserted["long_term"].extend(result.get("inserted", []))
        skipped.extend(result.get("skipped", []))
        errors.extend(result.get("errors", []))

    # Short-term memories -----------------------------------------------
    short_term_payload = payload.get("short_term_memories", []) or []
    if short_term_payload:
        with db_manager.SessionLocal() as session:
            engine = session.get_bind()
            if not _table_exists(engine, ShortTermMemory.__tablename__):
                logger.debug(
                    "Short-term table not available; skipping short-term import"
                )
            else:
                for index, record in enumerate(short_term_payload):
                    try:
                        prepared = _prepare_short_term_record(
                            record, namespace_map, default_ns
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        errors.append(
                            {
                                "type": "short_term",
                                "index": index,
                                "error": str(exc),
                            }
                        )
                        continue

                    mem_id = prepared.get("memory_id")
                    namespace = prepared.get("namespace")
                    existing = (
                        session.query(ShortTermMemory.memory_id)
                        .filter(
                            ShortTermMemory.memory_id == mem_id,
                            ShortTermMemory.namespace == namespace,
                        )
                        .first()
                    )
                    if existing is not None and dedupe:
                        skipped.append(
                            {
                                "type": "short_term",
                                "memory_id": mem_id,
                                "reason": "existing short-term memory in destination",
                            }
                        )
                        continue

                    if dry_run:
                        inserted["short_term"].append(str(mem_id))
                        continue

                    session.add(ShortTermMemory(**prepared))
                if not dry_run:
                    session.commit()

    # Clusters ----------------------------------------------------------
    cluster_payload = payload.get("clusters", []) or []
    if cluster_payload:
        with db_manager.SessionLocal() as session:
            if not _table_exists(session.get_bind(), Cluster.__tablename__):
                logger.debug("Cluster table not present; skipping clusters")
            else:
                for index, record in enumerate(cluster_payload):
                    try:
                        prepared = _prepare_cluster_record(record)
                    except Exception as exc:  # pragma: no cover - defensive
                        errors.append(
                            {"type": "clusters", "index": index, "error": str(exc)}
                        )
                        continue

                    cluster_id = prepared.get("id")
                    existing = (
                        session.query(Cluster.id)
                        .filter(Cluster.id == cluster_id)
                        .first()
                    )
                    if existing is not None and dedupe:
                        skipped.append(
                            {
                                "type": "clusters",
                                "id": cluster_id,
                                "reason": "existing cluster id in destination",
                            }
                        )
                        continue

                    if dry_run:
                        inserted["clusters"].append(str(cluster_id))
                        continue

                    session.add(Cluster(**prepared))
                if not dry_run:
                    session.commit()

    member_payload = payload.get("cluster_members", []) or []
    if member_payload:
        with db_manager.SessionLocal() as session:
            if not _table_exists(session.get_bind(), ClusterMember.__tablename__):
                logger.debug("Cluster members table absent; skipping members")
            else:
                for index, record in enumerate(member_payload):
                    try:
                        prepared = _prepare_cluster_member_record(record)
                    except Exception as exc:  # pragma: no cover - defensive
                        errors.append(
                            {
                                "type": "cluster_members",
                                "index": index,
                                "error": str(exc),
                            }
                        )
                        continue

                    member_id = prepared.get("id")
                    if member_id is not None:
                        existing = (
                            session.query(ClusterMember.id)
                            .filter(ClusterMember.id == member_id)
                            .first()
                        )
                        if existing is not None and dedupe:
                            skipped.append(
                                {
                                    "type": "cluster_members",
                                    "id": member_id,
                                    "reason": "existing cluster member id in destination",
                                }
                            )
                            continue

                    if dry_run:
                        inserted["cluster_members"].append(str(member_id))
                        continue

                    session.add(ClusterMember(**prepared))
                if not dry_run:
                    session.commit()

    # Relationships -----------------------------------------------------
    relationship_payload = payload.get("relationships", []) or []
    if relationship_payload:
        with db_manager.SessionLocal() as session:
            if not _table_exists(session.get_bind(), "memory_relationships"):
                logger.debug("Relationships table not found; skipping graph edges")
            else:
                for index, record in enumerate(relationship_payload):
                    try:
                        prepared = _prepare_relationship_record(
                            record, namespace_map, default_ns
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        errors.append(
                            {
                                "type": "relationships",
                                "index": index,
                                "error": str(exc),
                            }
                        )
                        continue

                    relationship_id = prepared.get("relationship_id")
                    existing = session.execute(
                        text(
                            "SELECT 1 FROM memory_relationships "
                            "WHERE relationship_id = :relationship_id"
                        ),
                        {"relationship_id": relationship_id},
                    ).first()
                    if existing is not None and dedupe:
                        skipped.append(
                            {
                                "type": "relationships",
                                "relationship_id": relationship_id,
                                "reason": "existing relationship id in destination",
                            }
                        )
                        continue

                    if dry_run:
                        inserted["relationships"].append(str(relationship_id))
                        continue

                    session.execute(
                        text(
                            "INSERT INTO memory_relationships ("
                            "relationship_id, source_memory_id, target_memory_id,"
                            " relationship_type, strength, reasoning, namespace, created_at"
                            ") VALUES ("
                            " :relationship_id, :source_memory_id, :target_memory_id,"
                            " :relationship_type, :strength, :reasoning, :namespace, :created_at"
                            ")"
                        ),
                        prepared,
                    )
                if not dry_run:
                    session.commit()

    return MigrationReport(
        dry_run=dry_run,
        inserted=inserted,
        skipped=skipped,
        errors=errors,
    )
