"""Helpers for exporting and importing Memoria databases.

The utilities in this module provide JSON and NDJSON serialisation helpers
for the primary Memoria tables. They are intentionally conservative and only
operate on a curated set of ORM models so that migrations remain predictable
across database engines.
"""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .models import (
    ChatHistory,
    Cluster,
    ClusterMember,
    LinkMemoryThread,
    LongTermMemory,
    MemoryAccessEvent,
    ShortTermMemory,
    SpatialMetadata,
    Team,
    ThreadEvent,
    ThreadMessageLink,
    Workspace,
    WorkspaceMember,
)

TEAM_TABLE = "teams"
TEAM_COLUMN = "team_id"
TEAM_FOREIGN_KEYS = {
    "chat_history": "fk_chat_history_team_id",
    "short_term_memory": "fk_short_term_team_id",
    "long_term_memory": "fk_long_term_team_id",
}
TEAM_INDEXES: dict[str, list[tuple[str, tuple[str, ...], bool]]] = {
    TEAM_TABLE: [
        ("idx_team_slug", ("slug",), True),
        ("idx_team_created", ("created_at",), False),
    ],
    "chat_history": [("idx_chat_team", (TEAM_COLUMN,), False)],
    "short_term_memory": [
        ("idx_short_term_team", (TEAM_COLUMN,), False),
        ("idx_short_term_namespace_team", ("namespace", TEAM_COLUMN), False),
    ],
    "long_term_memory": [
        ("idx_long_term_team", (TEAM_COLUMN,), False),
        ("idx_long_term_namespace_team", ("namespace", TEAM_COLUMN), False),
    ],
}


EXPORT_TABLES: OrderedDict[str, Any] = OrderedDict(
    [
        ("teams", Team),
        ("workspaces", Workspace),
        ("workspace_members", WorkspaceMember),
        ("chat_history", ChatHistory),
        ("long_term_memory", LongTermMemory),
        ("short_term_memory", ShortTermMemory),
        ("spatial_metadata", SpatialMetadata),
        ("memory_access_events", MemoryAccessEvent),
        ("link_memory_threads", LinkMemoryThread),
        ("thread_events", ThreadEvent),
        ("thread_message_links", ThreadMessageLink),
        ("clusters", Cluster),
        ("cluster_members", ClusterMember),
    ]
)


DELETE_ORDER: list[str] = list(reversed(list(EXPORT_TABLES.keys())))


@dataclass
class ExportResult:
    """Container returned by :func:`export_database`."""

    format: str
    metadata: dict[str, Any]
    content: str


def _serialize_value(value: Any) -> Any:
    """Convert SQLAlchemy values into JSON-compatible objects."""

    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _collect_rows(
    session: Session, tables: Sequence[str] | None
) -> dict[str, list[dict[str, Any]]]:
    """Collect rows for the requested tables."""

    inspector = inspect(session.bind)
    available_tables = set(inspector.get_table_names())
    selected_tables = list(EXPORT_TABLES.keys()) if tables is None else list(tables)

    collected: dict[str, list[dict[str, Any]]] = OrderedDict()

    for table_name in selected_tables:
        model = EXPORT_TABLES.get(table_name)
        if model is None:
            logger.warning("Skipping unknown table '%s' during export", table_name)
            continue
        if table_name not in available_tables:
            logger.debug("Table '%s' is not present in current database", table_name)
            continue

        mapper = inspect(model)
        rows: list[dict[str, Any]] = []
        for instance in session.query(model).all():
            row: dict[str, Any] = {}
            for column in mapper.columns:
                row[column.key] = _serialize_value(getattr(instance, column.key))
            rows.append(row)
        collected[table_name] = rows

    return collected


def _render_json_payload(
    rows: Mapping[str, list[Mapping[str, Any]]], metadata: Mapping[str, Any]
) -> str:
    payload = {"metadata": metadata, "data": rows}
    return json.dumps(payload, indent=2, sort_keys=True)


def _render_ndjson_payload(
    rows: Mapping[str, list[Mapping[str, Any]]], metadata: Mapping[str, Any]
) -> str:
    lines: list[str] = [json.dumps({"metadata": metadata})]
    for table_name, table_rows in rows.items():
        for row in table_rows:
            lines.append(json.dumps({"table": table_name, "data": row}))
    return "\n".join(lines) + "\n"


def export_database(
    session_factory: Any,
    *,
    format: str = "json",
    destination: Path | str | None = None,
    tables: Sequence[str] | None = None,
) -> ExportResult:
    """Serialise the configured Memoria tables to JSON or NDJSON.

    Parameters
    ----------
    session_factory:
        SQLAlchemy session factory, typically ``SQLAlchemyDatabaseManager.SessionLocal``.
    format:
        Output serialisation format (``"json"`` or ``"ndjson"``).
    destination:
        Optional filesystem location. When provided the payload is written to disk.
    tables:
        Optional subset of table names to export.
    """

    normalised_format = (format or "json").strip().lower()
    if normalised_format not in {"json", "ndjson"}:
        raise ValueError("format must be 'json' or 'ndjson'")

    session: Session = session_factory()
    try:
        bind = session.bind
        rows = _collect_rows(session, tables)
    finally:
        session.close()

    dialect = getattr(getattr(bind, "dialect", None), "name", None) if bind else None

    metadata = {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "format": normalised_format,
        "dialect": dialect,
        "tables": [
            {"name": name, "row_count": len(entries)} for name, entries in rows.items()
        ],
    }

    if normalised_format == "json":
        rendered = _render_json_payload(rows, metadata)
    else:
        rendered = _render_ndjson_payload(rows, metadata)

    if destination is not None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")

    return ExportResult(format=normalised_format, metadata=metadata, content=rendered)


def _parse_json_payload(source: Any) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    if isinstance(source, (str, bytes)):
        text = source.decode("utf-8") if isinstance(source, bytes) else source
        return json.loads(text)
    if isinstance(source, Path):
        return json.loads(Path(source).read_text(encoding="utf-8"))
    raise TypeError("Unsupported JSON payload source")


def _parse_ndjson_payload(source: Any) -> dict[str, Any]:
    if isinstance(source, Path):
        lines = Path(source).read_text(encoding="utf-8").splitlines()
    elif isinstance(source, bytes):
        lines = source.decode("utf-8").splitlines()
    elif isinstance(source, str):
        lines = source.splitlines()
    else:
        raise TypeError("Unsupported NDJSON payload source")

    if not lines:
        return {"metadata": {}, "data": {}}

    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("First line of NDJSON payload must contain metadata") from exc

    metadata = header.get("metadata", header)
    data: MutableMapping[str, list[dict[str, Any]]] = OrderedDict()

    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        table_name = record.get("table")
        row = record.get("data")
        if not table_name or not isinstance(row, Mapping):
            continue
        table_rows = data.setdefault(table_name, [])
        table_rows.append(dict(row))

    return {"metadata": metadata, "data": data}


def _coerce_value(column, value):
    if value is None:
        return None
    try:
        python_type = column.type.python_type
    except NotImplementedError:
        return value
    if python_type is datetime and isinstance(value, str):
        normalised = value.strip()
        if normalised.endswith("Z"):
            normalised = normalised[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(normalised)
        except ValueError:
            logger.warning(
                "Unable to parse datetime value '%s' for column '%s'", value, column.key
            )
            return value
    if python_type in {int, float, bool} and isinstance(value, str):
        try:
            return python_type(value)
        except (TypeError, ValueError):
            return value
    return value


def _truncate_tables(session: Session, table_names: Iterable[str]) -> None:
    engine_inspector = inspect(session.bind) if session.bind is not None else None
    for name in table_names:
        model = EXPORT_TABLES.get(name)
        if model is None:
            continue
        mapper = inspect(model)
        if engine_inspector is not None and not engine_inspector.has_table(
            mapper.local_table.name
        ):
            continue
        try:
            session.execute(mapper.local_table.delete())
        except SQLAlchemyError:
            logger.warning("Failed to truncate table '%s' during import", name)


def import_database(
    session_factory: Any,
    source: Any,
    *,
    format: str | None = None,
    tables: Sequence[str] | None = None,
    truncate: bool = True,
) -> dict[str, Any]:
    """Import data previously generated by :func:`export_database`."""

    inferred_format = (format or "").strip().lower()
    if not inferred_format:
        if isinstance(source, Path) and source.suffix.lower() == ".ndjson":
            inferred_format = "ndjson"
        else:
            inferred_format = "json"
    if inferred_format not in {"json", "ndjson"}:
        raise ValueError("format must be 'json' or 'ndjson'")

    if inferred_format == "json":
        payload = _parse_json_payload(source)
    else:
        payload = _parse_ndjson_payload(source)

    metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
    if isinstance(metadata, dict) and "format" not in metadata:
        metadata["format"] = inferred_format

    raw_rows = payload.get("data", {})
    if tables is not None:
        table_filter = set(tables)
        filtered_rows = OrderedDict()
        for table_name, table_rows in raw_rows.items():
            if table_name in table_filter:
                filtered_rows[table_name] = list(table_rows)
        rows_to_import = filtered_rows
    else:
        rows_to_import = OrderedDict((k, list(v)) for k, v in raw_rows.items())

    session: Session = session_factory()
    dialect = session.bind.dialect.name if session.bind is not None else ""

    fk_disabled = False
    try:
        if dialect == "sqlite":
            session.execute(text("PRAGMA foreign_keys=OFF"))
            fk_disabled = True
        elif dialect == "mysql":
            session.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            fk_disabled = True

        if truncate:
            _truncate_tables(session, DELETE_ORDER)

        for table_name in EXPORT_TABLES.keys():
            if table_name not in rows_to_import:
                continue
            model = EXPORT_TABLES[table_name]
            mapper = inspect(model)
            prepared_rows: list[dict[str, Any]] = []
            for row in rows_to_import[table_name]:
                if not isinstance(row, Mapping):
                    continue
                prepared: dict[str, Any] = {}
                for column in mapper.columns:
                    if column.key in row:
                        prepared[column.key] = _coerce_value(column, row[column.key])
                prepared_rows.append(prepared)
            if prepared_rows:
                session.execute(mapper.local_table.insert(), prepared_rows)

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        if fk_disabled:
            if dialect == "sqlite":
                session.execute(text("PRAGMA foreign_keys=ON"))
            elif dialect == "mysql":
                session.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        session.close()

    return metadata


def ensure_team_support(engine) -> dict[str, list[str]]:
    """Ensure the ``teams`` schema and foreign keys exist for an engine.

    Parameters
    ----------
    engine:
        SQLAlchemy engine bound to the target database.

    Returns
    -------
    dict[str, list[str]]
        Summary of tables, columns, indexes, and foreign keys created.
    """

    if engine is None:  # pragma: no cover - defensive guard
        return {"tables": [], "columns": [], "indexes": [], "foreign_keys": []}

    inspector = inspect(engine)
    dialect = engine.dialect.name
    actions: dict[str, list[str]] = defaultdict(list)

    def _refresh_inspector() -> None:
        nonlocal inspector
        inspector = inspect(engine)

    def _table_exists(table: str) -> bool:
        try:
            return table in inspector.get_table_names()
        except SQLAlchemyError:
            return False

    def _column_exists(table: str, column: str) -> bool:
        try:
            return any(col["name"] == column for col in inspector.get_columns(table))
        except SQLAlchemyError:
            return False

    def _index_exists(table: str, name: str) -> bool:
        try:
            return any(idx.get("name") == name for idx in inspector.get_indexes(table))
        except SQLAlchemyError:
            return False

    def _foreign_key_exists(table: str, name: str) -> bool:
        try:
            return any(
                fk.get("name") == name for fk in inspector.get_foreign_keys(table)
            )
        except SQLAlchemyError:
            return False

    if not _table_exists(TEAM_TABLE):
        if dialect == "sqlite":
            ddl = text(
                """
                CREATE TABLE teams (
                    team_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    slug TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        elif dialect == "postgresql":
            ddl = text(
                """
                CREATE TABLE teams (
                    team_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    slug VARCHAR(255) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        elif dialect in {"mysql", "mariadb"}:
            ddl = text(
                """
                CREATE TABLE teams (
                    team_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    slug VARCHAR(255) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
        else:
            ddl = text(
                """
                CREATE TABLE teams (
                    team_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    slug VARCHAR(255) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

        with engine.begin() as conn:
            conn.execute(ddl)
        actions["tables"].append(TEAM_TABLE)
        _refresh_inspector()

    for index_name, columns, is_unique in TEAM_INDEXES.get(TEAM_TABLE, []):
        if _index_exists(TEAM_TABLE, index_name):
            continue
        prefix = "CREATE UNIQUE INDEX" if is_unique else "CREATE INDEX"
        if dialect in {"sqlite", "postgresql"}:
            prefix = f"{prefix} IF NOT EXISTS"
        column_list = ", ".join(columns)
        statement = text(f"{prefix} {index_name} ON {TEAM_TABLE} ({column_list})")
        with engine.begin() as conn:
            conn.execute(statement)
        actions["indexes"].append(f"{TEAM_TABLE}.{index_name}")
        _refresh_inspector()

    column_type = "TEXT" if dialect == "sqlite" else "VARCHAR(255)"
    fk_clause = f" REFERENCES {TEAM_TABLE}({TEAM_COLUMN}) ON DELETE SET NULL"

    for table_name, fk_name in TEAM_FOREIGN_KEYS.items():
        if not _table_exists(table_name):
            continue

        if not _column_exists(table_name, TEAM_COLUMN):
            if dialect == "sqlite":
                statement = text(
                    f"ALTER TABLE {table_name} ADD COLUMN {TEAM_COLUMN} {column_type}{fk_clause}"
                )
            else:
                statement = text(
                    f"ALTER TABLE {table_name} ADD COLUMN {TEAM_COLUMN} {column_type}"
                )
            with engine.begin() as conn:
                conn.execute(statement)
            actions["columns"].append(f"{table_name}.{TEAM_COLUMN}")
            _refresh_inspector()

        if dialect not in {"sqlite"} and not _foreign_key_exists(table_name, fk_name):
            fk_statement = text(
                f"ALTER TABLE {table_name} ADD CONSTRAINT {fk_name} "
                f"FOREIGN KEY ({TEAM_COLUMN}) REFERENCES {TEAM_TABLE}({TEAM_COLUMN}) ON DELETE SET NULL"
            )
            with engine.begin() as conn:
                conn.execute(fk_statement)
            actions["foreign_keys"].append(f"{table_name}.{fk_name}")
            _refresh_inspector()

        for index_name, columns, is_unique in TEAM_INDEXES.get(table_name, []):
            if _index_exists(table_name, index_name):
                continue
            prefix = "CREATE UNIQUE INDEX" if is_unique else "CREATE INDEX"
            if dialect in {"sqlite", "postgresql"}:
                prefix = f"{prefix} IF NOT EXISTS"
            column_list = ", ".join(columns)
            statement = text(f"{prefix} {index_name} ON {table_name} ({column_list})")
            with engine.begin() as conn:
                conn.execute(statement)
            actions["indexes"].append(f"{table_name}.{index_name}")
            _refresh_inspector()

    return {key: list(values) for key, values in actions.items()}


__all__ = [
    "EXPORT_TABLES",
    "export_database",
    "import_database",
    "ensure_team_support",
    "ExportResult",
]
