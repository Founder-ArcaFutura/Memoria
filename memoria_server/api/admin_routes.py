"""Administrative routes for inspecting Memoria database tables."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable, Sequence

import sqlite3

from flask import Blueprint, Response, current_app, jsonify, request
from loguru import logger
from sqlalchemy import (
    MetaData,
    String,
    Table,
    cast,
    func,
    inspect,
    or_,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError


admin_bp = Blueprint("admin", __name__)


def _get_db_manager():
    memoria = current_app.config.get("memoria")
    return getattr(memoria, "db_manager", None)


def _normalise_tables(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, Sequence):
        candidates = list(raw)
    else:
        raise ValueError("'tables' must be a sequence of strings")

    cleaned: list[str] = []
    for value in candidates:
        if not isinstance(value, str):
            raise ValueError("'tables' must be a sequence of strings")
        name = value.strip()
        if name:
            cleaned.append(name)
    return cleaned or None


def _error_response(message: str, status: int):
    return jsonify({"status": "error", "message": message}), status


@admin_bp.route("/admin/migrations/export", methods=["POST"])
def export_dataset_route():
    """Export core Memoria tables as JSON or NDJSON."""

    manager = _get_db_manager()
    if manager is None or not getattr(manager, "SessionLocal", None):
        return _error_response("Database manager unavailable", 503)

    payload = request.get_json(silent=True) or {}
    fmt = (payload.get("format") or request.args.get("format") or "json").lower()
    try:
        tables = _normalise_tables(payload.get("tables") or request.args.getlist("table"))
    except ValueError as exc:
        return _error_response(str(exc), 400)

    try:
        result = manager.export_dataset(format=fmt, tables=tables)
    except ValueError as exc:
        return _error_response(str(exc), 400)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to export dataset")
        return _error_response(str(exc), 500)

    mimetype = "application/json" if result.format == "json" else "application/x-ndjson"
    suffix = "json" if result.format == "json" else "ndjson"
    response = Response(result.content, mimetype=mimetype)
    response.headers["Content-Disposition"] = f"attachment; filename=memoria-export.{suffix}"
    response.headers["X-Memoria-Exported-At"] = result.metadata.get("exported_at", "")
    return response


@admin_bp.route("/admin/migrations/import", methods=["POST"])
def import_dataset_route():
    """Import a dataset produced by :func:`export_dataset_route`."""

    manager = _get_db_manager()
    if manager is None or not getattr(manager, "SessionLocal", None):
        return _error_response("Database manager unavailable", 503)

    payload = request.get_json(silent=True)
    fmt = request.args.get("format") or request.form.get("format")
    tables: list[str] | None = None
    truncate = True

    if payload and isinstance(payload, dict):
        fmt = payload.get("format") or fmt
        try:
            tables = _normalise_tables(payload.get("tables")) or tables
        except ValueError as exc:
            return _error_response(str(exc), 400)
        if "truncate" in payload:
            truncate = bool(payload.get("truncate", True))
    else:
        try:
            query_tables = _normalise_tables(
                request.args.getlist("table") or request.form.getlist("table")
            )
        except ValueError as exc:
            return _error_response(str(exc), 400)
        if query_tables:
            tables = query_tables
        truncate_param = request.args.get("truncate") or request.form.get("truncate")
        if truncate_param is not None:
            truncate = str(truncate_param).strip().lower() not in {"0", "false", "no"}

    data_source: Any = None
    upload = request.files.get("file")
    if upload is not None:
        data_source = upload.read()
        if fmt is None and upload.filename:
            if upload.filename.lower().endswith(".ndjson"):
                fmt = "ndjson"
    elif payload and isinstance(payload, dict) and "data" in payload:
        data_source = payload["data"]
    else:
        # Fallback to raw request body when no JSON payload is provided
        if not request.files and request.data:
            data_source = request.get_data(as_text=True)

    if data_source is None:
        return _error_response("No dataset provided", 400)

    try:
        metadata = manager.import_dataset(
            data_source,
            format=fmt,
            tables=tables,
            truncate=truncate,
        )
    except ValueError as exc:
        return _error_response(str(exc), 400)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to import dataset")
        return _error_response(str(exc), 500)

    return jsonify({"status": "ok", "metadata": metadata or {}})


def _parse_positive_int(value: str | None, default: int, maximum: int) -> int:
    """Parse a positive integer query parameter within bounds."""

    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return min(parsed, maximum)


def _quote_identifier(identifier: str) -> str:
    """Safely quote SQLite identifiers using double quotes."""

    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _serialize_value(value: Any) -> Any:
    """Convert database values into JSON-serializable objects."""

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return bytes(value).hex()
    return value


def _coerce_sqlalchemy_values(table: Table, values: dict[str, Any]) -> dict[str, Any]:
    """Coerce incoming ISO-8601 strings for SQLAlchemy datetime/date columns."""

    coerced: dict[str, Any] = dict(values)
    for column in table.c:
        name = column.key
        if name not in values:
            continue
        raw_value = values[name]
        if raw_value is None or not isinstance(raw_value, str):
            continue
        try:
            python_type = column.type.python_type
        except NotImplementedError:
            continue
        if python_type not in {datetime, date}:
            continue

        normalized = raw_value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"

        try:
            if python_type is datetime:
                coerced[name] = datetime.fromisoformat(normalized)
            else:  # python_type is date
                try:
                    coerced[name] = date.fromisoformat(normalized)
                except ValueError:
                    coerced[name] = datetime.fromisoformat(normalized).date()
        except ValueError as exc:
            raise ValueError(
                f"Invalid ISO-8601 value for column '{name}'"
            ) from exc

    return coerced


@contextmanager
def _sqlite_connection(db_path: str):
    """Context manager yielding a SQLite connection with row factory."""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _format_sqlalchemy_columns(columns: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for column in columns:
        default = column.get("default")
        formatted.append(
            {
                "name": column.get("name"),
                "type": str(column.get("type")),
                "nullable": bool(column.get("nullable", True)),
                "default": str(default) if default is not None else None,
                "primary_key": bool(column.get("primary_key")),
            }
        )
    return formatted


def _format_sqlite_columns(columns: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for column in columns:
        formatted.append(
            {
                "name": column["name"],
                "type": column["type"] or "",
                "nullable": not bool(column["notnull"]),
                "default": column["dflt_value"],
                "primary_key": bool(column["pk"]),
            }
        )
    return formatted


def _collect_namespaces_engine(conn, table: Table) -> list[str]:
    if "namespace" not in table.c:
        return []
    try:
        stmt = (
            select(table.c.namespace)
            .where(table.c.namespace.isnot(None))
            .where(table.c.namespace != "")
            .distinct()
            .order_by(table.c.namespace)
            .limit(100)
        )
        return [row[0] for row in conn.execute(stmt) if row[0] not in (None, "")]
    except SQLAlchemyError:
        logger.exception("Failed to collect namespaces for %s", table.name)
        return []


def _collect_namespaces_sqlite(conn: sqlite3.Connection, table_name: str) -> list[str]:
    try:
        rows = conn.execute(
            f"""
            SELECT DISTINCT namespace
            FROM {_quote_identifier(table_name)}
            WHERE namespace IS NOT NULL AND namespace <> ''
            ORDER BY namespace
            LIMIT 100
            """
        ).fetchall()
    except sqlite3.Error:
        logger.exception("Failed to collect namespaces for %s", table_name)
        return []
    return [row[0] for row in rows if row[0] not in (None, "")]


def _list_tables_engine(engine: Engine) -> list[dict[str, Any]]:
    inspector = inspect(engine)
    physical_tables = set(inspector.get_table_names())
    view_names = set(inspector.get_view_names())
    all_names = sorted(physical_tables | view_names)
    metadata = MetaData()
    tables: list[dict[str, Any]] = []
    with engine.connect() as conn:
        for name in all_names:
            try:
                table = Table(name, metadata, autoload_with=engine)
            except SQLAlchemyError:
                logger.exception("Unable to reflect table %s", name)
                continue
            try:
                column_info = inspector.get_columns(name)
            except SQLAlchemyError:
                logger.exception("Unable to inspect columns for %s", name)
                column_info = []
            columns = _format_sqlalchemy_columns(column_info)
            namespaces = _collect_namespaces_engine(conn, table)
            row_count: int | None = None
            try:
                count_stmt = select(func.count()).select_from(table)
                result = conn.execute(count_stmt).scalar()
                if result is not None:
                    row_count = int(result)
            except SQLAlchemyError:
                logger.debug("Count query failed for %s", name, exc_info=True)
            tables.append(
                {
                    "name": name,
                    "type": "table" if name in physical_tables else "view",
                    "columns": columns,
                    "row_count": row_count,
                    "namespaces": namespaces,
                }
            )
    return tables


def _list_tables_sqlite(db_path: str) -> list[dict[str, Any]]:
    with _sqlite_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
            ORDER BY name
            """
        ).fetchall()
        tables: list[dict[str, Any]] = []
        for name, table_type in rows:
            try:
                column_rows = conn.execute(
                    f"PRAGMA table_info({_quote_identifier(name)})"
                ).fetchall()
            except sqlite3.Error:
                logger.exception("Unable to inspect columns for %s", name)
                column_rows = []
            columns = _format_sqlite_columns(column_rows)
            namespaces = (
                _collect_namespaces_sqlite(conn, name)
                if any(col["name"] == "namespace" for col in columns)
                else []
            )
            row_count: int | None = None
            try:
                result = conn.execute(
                    f"SELECT COUNT(*) FROM {_quote_identifier(name)}"
                ).fetchone()
                if result is not None:
                    row_count = int(result[0])
            except sqlite3.Error:
                logger.debug("Count query failed for %s", name, exc_info=True)
            tables.append(
                {
                    "name": name,
                    "type": table_type,
                    "columns": columns,
                    "row_count": row_count,
                    "namespaces": namespaces,
                }
            )
    return tables


def _table_exists_sqlite(conn: sqlite3.Connection, table_name: str) -> bool:
    query = """
        SELECT 1 FROM sqlite_master
        WHERE type IN ('table', 'view') AND name = ?
        LIMIT 1
    """
    return conn.execute(query, (table_name,)).fetchone() is not None


def _sqlite_table_metadata(
    conn: sqlite3.Connection, table_name: str
) -> dict[str, Any] | None:
    """Return SQLite table type and column metadata for mutations."""

    try:
        row = conn.execute(
            "SELECT type FROM sqlite_master WHERE name = ? LIMIT 1", (table_name,)
        ).fetchone()
    except sqlite3.Error:
        logger.exception("Unable to inspect table metadata for %s", table_name)
        return None
    if row is None:
        return None

    table_type = row[0]
    try:
        column_rows = conn.execute(
            f"PRAGMA table_info({_quote_identifier(table_name)})"
        ).fetchall()
    except sqlite3.Error:
        logger.exception("Unable to inspect columns for %s", table_name)
        column_rows = []

    pk_columns = [col["name"] for col in column_rows if col["pk"]]
    columns = [col["name"] for col in column_rows]
    return {
        "type": table_type,
        "columns": columns,
        "pk_columns": pk_columns,
        "column_rows": column_rows,
    }


def _engine_table_metadata(
    engine: Engine, table_name: str
) -> dict[str, Any] | None:
    """Return SQLAlchemy table metadata for mutations."""

    inspector = inspect(engine)
    physical_tables = set(inspector.get_table_names())
    view_names = set(inspector.get_view_names())
    if table_name not in physical_tables and table_name not in view_names:
        return None

    metadata = MetaData()
    try:
        table = Table(table_name, metadata, autoload_with=engine)
    except SQLAlchemyError:
        logger.exception("Unable to reflect table %s", table_name)
        return None

    return {
        "table": table,
        "is_view": table_name in view_names,
        "columns": [column.name for column in table.columns],
        "pk_columns": [column.name for column in table.primary_key.columns],
    }


def _parse_primary_key_mapping(
    path_value: str,
    body: dict[str, Any] | None,
    pk_columns: Sequence[str],
) -> dict[str, Any] | None:
    """Deserialize a path parameter or request body into a PK mapping."""

    if not pk_columns:
        return None

    mapping: dict[str, Any] | None = None
    if body and isinstance(body.get("primary_key"), dict):
        mapping = body["primary_key"]
    else:
        try:
            parsed = json.loads(path_value)
        except json.JSONDecodeError:
            parsed = path_value

        if isinstance(parsed, dict):
            mapping = parsed
        elif isinstance(parsed, list):
            if len(parsed) != len(pk_columns):
                return None
            mapping = {name: parsed[idx] for idx, name in enumerate(pk_columns)}
        else:
            if len(pk_columns) != 1:
                return None
            mapping = {pk_columns[0]: parsed}

    if mapping is None:
        return None

    normalized: dict[str, Any] = {}
    for column in pk_columns:
        if column not in mapping:
            return None
        normalized[column] = mapping[column]
    return normalized


def _filter_unknown_columns(values: dict[str, Any], valid: Sequence[str]) -> list[str]:
    """Return a list of columns that are not present in the table definition."""

    valid_set = set(valid)
    return [column for column in values.keys() if column not in valid_set]


def _insert_row_engine(
    engine: Engine, table_name: str, values: dict[str, Any]
) -> dict[str, Any] | None:
    metadata = _engine_table_metadata(engine, table_name)
    if metadata is None:
        return None
    if metadata.get("is_view"):
        raise PermissionError("Cannot modify a database view")

    table: Table = metadata["table"]
    columns = metadata["columns"]
    unknown = _filter_unknown_columns(values, columns)
    if unknown:
        raise ValueError(f"Unknown columns: {', '.join(sorted(unknown))}")
    if not values:
        raise ValueError("No values provided")

    coerced_values = _coerce_sqlalchemy_values(table, values)
    ordered_values = {
        key: coerced_values[key] for key in values.keys() if key in columns
    }
    try:
        with engine.begin() as conn:
            result = conn.execute(table.insert().values(**ordered_values))
    except IntegrityError as exc:
        raise ValueError(str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.exception("Failed to insert row into %s", table_name)
        raise RuntimeError(str(exc)) from exc

    pk_columns = metadata.get("pk_columns", [])
    inserted_pk = list(result.inserted_primary_key or [])
    primary_key: dict[str, Any] = {}
    for idx, column in enumerate(pk_columns):
        if idx < len(inserted_pk) and inserted_pk[idx] is not None:
            primary_key[column] = inserted_pk[idx]
        elif column in ordered_values:
            primary_key[column] = ordered_values[column]
    return {"primary_key": primary_key or None}


def _insert_row_sqlite(
    db_path: str, table_name: str, values: dict[str, Any]
) -> dict[str, Any] | None:
    with _sqlite_connection(db_path) as conn:
        metadata = _sqlite_table_metadata(conn, table_name)
        if metadata is None:
            return None
        if metadata.get("type") != "table":
            raise PermissionError("Cannot modify a database view")
        columns = metadata.get("columns", [])
        unknown = _filter_unknown_columns(values, columns)
        if unknown:
            raise ValueError(f"Unknown columns: {', '.join(sorted(unknown))}")
        if not values:
            raise ValueError("No values provided")

        column_names: list[str] = []
        params: list[Any] = []
        for column, value in values.items():
            if column not in columns:
                continue
            column_names.append(_quote_identifier(column))
            params.append(value)

        placeholders = ", ".join("?" for _ in column_names)
        sql = (
            f"INSERT INTO {_quote_identifier(table_name)} "
            f"({', '.join(column_names)}) VALUES ({placeholders})"
        )

        try:
            cursor = conn.execute(sql, tuple(params))
            conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(str(exc)) from exc
        except sqlite3.Error as exc:
            logger.exception("Failed to insert row into %s", table_name)
            raise RuntimeError(str(exc)) from exc

        pk_columns: list[str] = metadata.get("pk_columns", [])
        primary_key: dict[str, Any] = {}
        if pk_columns:
            for column in pk_columns:
                if column in values and values[column] is not None:
                    primary_key[column] = values[column]
            if len(pk_columns) == 1 and pk_columns[0] not in primary_key:
                primary_key[pk_columns[0]] = cursor.lastrowid
        return {"primary_key": primary_key or None}


def _update_row_engine(
    engine: Engine,
    table_name: str,
    pk_mapping: dict[str, Any],
    values: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> int | None:
    if metadata is None:
        metadata = _engine_table_metadata(engine, table_name)
    if metadata is None:
        return None
    if metadata.get("is_view"):
        raise PermissionError("Cannot modify a database view")

    table: Table = metadata["table"]
    columns = metadata["columns"]
    pk_columns = metadata.get("pk_columns", [])
    unknown = _filter_unknown_columns(values, columns)
    if unknown:
        raise ValueError(f"Unknown columns: {', '.join(sorted(unknown))}")
    if set(values).intersection(pk_columns):
        raise ValueError("Primary key columns cannot be updated")
    if not values:
        raise ValueError("No values provided")

    stmt = table.update()
    for column, value in pk_mapping.items():
        stmt = stmt.where(table.c[column] == value)

    coerced_values = _coerce_sqlalchemy_values(table, values)
    ordered_updates = {
        key: coerced_values[key] for key in values.keys() if key in columns
    }
    try:
        with engine.begin() as conn:
            result = conn.execute(stmt.values(**ordered_updates))
    except IntegrityError as exc:
        raise ValueError(str(exc)) from exc
    except SQLAlchemyError:
        logger.exception("Failed to update row in %s", table_name)
        raise RuntimeError("Update failed")
    return result.rowcount


def _update_row_sqlite(
    db_path: str,
    table_name: str,
    pk_mapping: dict[str, Any],
    values: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> int | None:
    with _sqlite_connection(db_path) as conn:
        if metadata is None:
            metadata = _sqlite_table_metadata(conn, table_name)
        if metadata is None:
            return None
        if metadata.get("type") != "table":
            raise PermissionError("Cannot modify a database view")

        columns = metadata.get("columns", [])
        pk_columns = metadata.get("pk_columns", [])
        unknown = _filter_unknown_columns(values, columns)
        if unknown:
            raise ValueError(f"Unknown columns: {', '.join(sorted(unknown))}")
        if set(values).intersection(pk_columns):
            raise ValueError("Primary key columns cannot be updated")
        if not values:
            raise ValueError("No values provided")

        set_parts: list[str] = []
        params: list[Any] = []
        for column, value in values.items():
            if column not in columns:
                continue
            set_parts.append(f"{_quote_identifier(column)} = ?")
            params.append(value)

        where_parts: list[str] = []
        for column in pk_columns:
            if column not in pk_mapping:
                raise ValueError("Incomplete primary key")
            where_parts.append(f"{_quote_identifier(column)} = ?")
            params.append(pk_mapping[column])

        sql = (
            f"UPDATE {_quote_identifier(table_name)} SET {', '.join(set_parts)} "
            f"WHERE {' AND '.join(where_parts)}"
        )

        try:
            cursor = conn.execute(sql, tuple(params))
            conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(str(exc)) from exc
        except sqlite3.Error:
            logger.exception("Failed to update row in %s", table_name)
            raise RuntimeError("Update failed")
        return cursor.rowcount


def _delete_row_engine(
    engine: Engine,
    table_name: str,
    pk_mapping: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> int | None:
    if metadata is None:
        metadata = _engine_table_metadata(engine, table_name)
    if metadata is None:
        return None
    if metadata.get("is_view"):
        raise PermissionError("Cannot modify a database view")

    table: Table = metadata["table"]
    stmt = table.delete()
    for column, value in pk_mapping.items():
        stmt = stmt.where(table.c[column] == value)

    try:
        with engine.begin() as conn:
            result = conn.execute(stmt)
    except SQLAlchemyError:
        logger.exception("Failed to delete row from %s", table_name)
        raise RuntimeError("Delete failed")
    return result.rowcount


def _delete_row_sqlite(
    db_path: str,
    table_name: str,
    pk_mapping: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> int | None:
    with _sqlite_connection(db_path) as conn:
        if metadata is None:
            metadata = _sqlite_table_metadata(conn, table_name)
        if metadata is None:
            return None
        if metadata.get("type") != "table":
            raise PermissionError("Cannot modify a database view")
        pk_columns = metadata.get("pk_columns", [])

        where_parts: list[str] = []
        params: list[Any] = []
        for column in pk_columns:
            if column not in pk_mapping:
                raise ValueError("Incomplete primary key")
            where_parts.append(f"{_quote_identifier(column)} = ?")
            params.append(pk_mapping[column])

        if not where_parts:
            raise ValueError("Primary key required for delete")

        sql = (
            f"DELETE FROM {_quote_identifier(table_name)} "
            f"WHERE {' AND '.join(where_parts)}"
        )

        try:
            cursor = conn.execute(sql, tuple(params))
            conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to delete row from %s", table_name)
            raise RuntimeError("Delete failed")
        return cursor.rowcount


def _fetch_rows_engine(
    engine: Engine,
    table_name: str,
    page: int,
    page_size: int,
    search_term: str,
    namespaces: list[str],
) -> dict[str, Any] | None:
    inspector = inspect(engine)
    physical_tables = set(inspector.get_table_names())
    view_names = set(inspector.get_view_names())
    if table_name not in physical_tables and table_name not in view_names:
        return None

    metadata = MetaData()
    try:
        table = Table(table_name, metadata, autoload_with=engine)
    except SQLAlchemyError:
        logger.exception("Unable to reflect table %s", table_name)
        return None

    try:
        column_info = inspector.get_columns(table_name)
    except SQLAlchemyError:
        logger.exception("Unable to inspect columns for %s", table_name)
        column_info = []
    columns = _format_sqlalchemy_columns(column_info)

    applied_namespaces: list[str] = []
    filters = []
    if namespaces and "namespace" in table.c:
        applied_namespaces = [ns for ns in namespaces if ns]
        if applied_namespaces:
            filters.append(table.c.namespace.in_(applied_namespaces))
    search_filters = []
    pattern = f"%{search_term}%" if search_term else None
    if pattern:
        for column in table.c:
            try:
                search_filters.append(cast(column, String).ilike(pattern))
            except Exception:  # pragma: no cover - defensive
                continue
        if search_filters:
            filters.append(or_(*search_filters))

    base_query = select(table)
    count_query = select(func.count()).select_from(table)
    for filter_clause in filters:
        base_query = base_query.where(filter_clause)
        count_query = count_query.where(filter_clause)

    pk_columns = list(table.primary_key.columns)
    if pk_columns:
        base_query = base_query.order_by(*pk_columns)
    else:
        first_column = next(iter(table.c), None)
        if first_column is not None:
            base_query = base_query.order_by(first_column)

    offset = (page - 1) * page_size
    base_query = base_query.offset(offset).limit(page_size)

    with engine.connect() as conn:
        total_rows = conn.execute(count_query).scalar() or 0
        try:
            result_rows = conn.execute(base_query).fetchall()
        except SQLAlchemyError:
            logger.exception("Unable to fetch rows for %s", table_name)
            return None
        namespaces_available = _collect_namespaces_engine(conn, table)

    rows = [
        {key: _serialize_value(value) for key, value in row._mapping.items()}
        for row in result_rows
    ]

    return {
        "columns": columns,
        "rows": rows,
        "total_rows": int(total_rows),
        "available_namespaces": namespaces_available,
        "applied_namespaces": applied_namespaces,
        "table_type": "table" if table_name in physical_tables else "view",
    }


def _fetch_rows_sqlite(
    db_path: str,
    table_name: str,
    page: int,
    page_size: int,
    search_term: str,
    namespaces: list[str],
) -> dict[str, Any] | None:
    with _sqlite_connection(db_path) as conn:
        if not _table_exists_sqlite(conn, table_name):
            return None

        try:
            column_rows = conn.execute(
                f"PRAGMA table_info({_quote_identifier(table_name)})"
            ).fetchall()
        except sqlite3.Error:
            logger.exception("Unable to inspect columns for %s", table_name)
            column_rows = []
        columns = _format_sqlite_columns(column_rows)

        filters: list[str] = []
        params: list[Any] = []
        applied_namespaces: list[str] = []
        if namespaces and any(col["name"] == "namespace" for col in columns):
            applied_namespaces = [ns for ns in namespaces if ns]
            if applied_namespaces:
                placeholders = ",".join("?" for _ in applied_namespaces)
                filters.append(f"namespace IN ({placeholders})")
                params.extend(applied_namespaces)

        if search_term:
            like = f"%{search_term}%"
            search_clauses: list[str] = []
            for column in columns:
                identifier = _quote_identifier(column["name"])
                search_clauses.append(f"CAST({identifier} AS TEXT) LIKE ?")
                params.append(like)
            if search_clauses:
                filters.append("(" + " OR ".join(search_clauses) + ")")

        where_clause = " WHERE " + " AND ".join(filters) if filters else ""
        order_clause = ""
        pk_names = [col["name"] for col in columns if col["primary_key"]]
        if pk_names:
            order_clause = " ORDER BY " + ", ".join(
                _quote_identifier(name) for name in pk_names
            )
        elif columns:
            order_clause = f" ORDER BY {_quote_identifier(columns[0]['name'])}"

        offset = (page - 1) * page_size
        base_params = list(params)
        count_query = (
            f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}{where_clause}"
        )
        try:
            total_rows = conn.execute(count_query, tuple(base_params)).fetchone()[0]
        except sqlite3.Error:
            logger.exception("Unable to count rows for %s", table_name)
            return None

        data_query = (
            f"SELECT * FROM {_quote_identifier(table_name)}"
            f"{where_clause}{order_clause} LIMIT ? OFFSET ?"
        )
        data_params = base_params + [page_size, offset]
        try:
            rows = conn.execute(data_query, tuple(data_params)).fetchall()
        except sqlite3.Error:
            logger.exception("Unable to fetch rows for %s", table_name)
            return None

        namespaces_available = (
            _collect_namespaces_sqlite(conn, table_name)
            if any(col["name"] == "namespace" for col in columns)
            else []
        )

    results = []
    for row in rows:
        record = {key: _serialize_value(row[key]) for key in row.keys()}
        results.append(record)

    return {
        "columns": columns,
        "rows": results,
        "total_rows": int(total_rows),
        "available_namespaces": namespaces_available,
        "applied_namespaces": applied_namespaces,
        "table_type": "table",
    }


@admin_bp.route("/session", methods=["POST"])
def create_session():
    """Validate API key submissions for the dashboard."""

    expected = current_app.config.get("EXPECTED_API_KEY")
    payload = request.get_json(silent=True) or {}
    provided = payload.get("api_key") or payload.get("key") or payload.get("token")

    if expected:
        if not provided:
            return jsonify({"status": "error", "message": "API key required"}), 400
        if provided != expected:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
    return jsonify({"status": "ok", "requires_key": bool(expected)})


@admin_bp.route("/admin/tables", methods=["GET"])
def list_tables():
    """Return available database tables along with column metadata."""

    engine = current_app.config.get("ENGINE")
    db_path = current_app.config.get("DB_PATH")

    try:
        if engine is not None:
            tables = _list_tables_engine(engine)
        elif db_path:
            tables = _list_tables_sqlite(db_path)
        else:
            return (
                jsonify({"status": "error", "message": "No database connection"}),
                503,
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to list tables")
        return jsonify({"status": "error", "message": str(exc)}), 500

    return jsonify({"tables": tables})


@admin_bp.route("/admin/tables/<string:table_name>/rows", methods=["GET"])
def stream_rows(table_name: str):
    """Stream paginated rows for a given table with optional filters."""

    page = _parse_positive_int(request.args.get("page"), default=1, maximum=1000)
    page_size = _parse_positive_int(
        request.args.get("page_size"), default=25, maximum=500
    )
    search_term = (request.args.get("search") or "").strip()
    namespaces = request.args.getlist("namespace")

    engine = current_app.config.get("ENGINE")
    db_path = current_app.config.get("DB_PATH")

    try:
        if engine is not None:
            data = _fetch_rows_engine(engine, table_name, page, page_size, search_term, namespaces)
        elif db_path:
            data = _fetch_rows_sqlite(db_path, table_name, page, page_size, search_term, namespaces)
        else:
            return (
                jsonify({"status": "error", "message": "No database connection"}),
                503,
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to fetch rows for %s", table_name)
        return jsonify({"status": "error", "message": str(exc)}), 500

    if data is None:
        return jsonify({"status": "error", "message": "Table not found"}), 404

    total_rows = data["total_rows"]
    offset = (page - 1) * page_size
    has_next = offset + len(data["rows"]) < total_rows

    response = {
        "table": table_name,
        "type": data.get("table_type"),
        "columns": data["columns"],
        "rows": data["rows"],
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "has_next": has_next,
        "has_previous": page > 1,
        "namespaces": data.get("available_namespaces", []),
        "applied_filters": {
            "namespace": data.get("applied_namespaces", []) or None,
            "search": search_term or None,
        },
    }

    return jsonify(response)


@admin_bp.route("/admin/tables/<string:table_name>/rows", methods=["POST"])
def create_row_entry(table_name: str):
    """Insert a new row into a database table."""

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        payload = {}
    values = payload.get("values")
    if not isinstance(values, dict):
        return _error_response("Request body must include a 'values' object", 400)

    engine = current_app.config.get("ENGINE")
    db_path = current_app.config.get("DB_PATH")

    try:
        if engine is not None:
            result = _insert_row_engine(engine, table_name, values)
        elif db_path:
            result = _insert_row_sqlite(db_path, table_name, values)
        else:
            return _error_response("No database connection", 503)
    except PermissionError as exc:
        return _error_response(str(exc), 400)
    except ValueError as exc:
        return _error_response(str(exc), 400)
    except RuntimeError as exc:  # pragma: no cover - defensive
        return _error_response(str(exc), 500)

    if result is None:
        return _error_response("Table not found", 404)

    response = {"status": "ok"}
    response.update(result)
    return jsonify(response)


@admin_bp.route(
    "/admin/tables/<string:table_name>/rows/<string:pk_value>", methods=["PATCH"]
)
def update_row_entry(table_name: str, pk_value: str):
    """Update an existing row identified by its primary key."""

    payload_raw = request.get_json(silent=True) or {}
    payload = payload_raw if isinstance(payload_raw, dict) else {}
    values = payload.get("values")
    if not isinstance(values, dict) or not values:
        return _error_response("Request body must include values to update", 400)

    engine = current_app.config.get("ENGINE")
    db_path = current_app.config.get("DB_PATH")

    try:
        if engine is not None:
            metadata = _engine_table_metadata(engine, table_name)
            if metadata is None:
                return _error_response("Table not found", 404)
            pk_columns = metadata.get("pk_columns", [])
            if not pk_columns:
                return _error_response("Table does not define a primary key", 400)
            pk_mapping = _parse_primary_key_mapping(pk_value, payload, pk_columns)
            if pk_mapping is None:
                return _error_response("Primary key values are required", 400)
            updated = _update_row_engine(
                engine, table_name, pk_mapping, values, metadata
            )
        elif db_path:
            with _sqlite_connection(db_path) as conn:
                metadata = _sqlite_table_metadata(conn, table_name)
            if metadata is None:
                return _error_response("Table not found", 404)
            pk_columns = metadata.get("pk_columns", [])
            if not pk_columns:
                return _error_response("Table does not define a primary key", 400)
            pk_mapping = _parse_primary_key_mapping(pk_value, payload, pk_columns)
            if pk_mapping is None:
                return _error_response("Primary key values are required", 400)
            updated = _update_row_sqlite(
                db_path, table_name, pk_mapping, values, metadata
            )
        else:
            return _error_response("No database connection", 503)
    except PermissionError as exc:
        return _error_response(str(exc), 400)
    except ValueError as exc:
        return _error_response(str(exc), 400)
    except RuntimeError as exc:  # pragma: no cover - defensive
        return _error_response(str(exc), 500)

    if updated is None:
        return _error_response("Table not found", 404)
    if updated == 0:
        return _error_response("Row not found", 404)

    return jsonify({"status": "ok", "updated": int(updated)})


@admin_bp.route(
    "/admin/tables/<string:table_name>/rows/<string:pk_value>", methods=["DELETE"]
)
def delete_row_entry(table_name: str, pk_value: str):
    """Delete a row identified by its primary key."""

    payload_raw = request.get_json(silent=True) or {}
    payload = payload_raw if isinstance(payload_raw, dict) else {}

    engine = current_app.config.get("ENGINE")
    db_path = current_app.config.get("DB_PATH")

    try:
        if engine is not None:
            metadata = _engine_table_metadata(engine, table_name)
            if metadata is None:
                return _error_response("Table not found", 404)
            pk_columns = metadata.get("pk_columns", [])
            if not pk_columns:
                return _error_response("Table does not define a primary key", 400)
            pk_mapping = _parse_primary_key_mapping(pk_value, payload, pk_columns)
            if pk_mapping is None:
                return _error_response("Primary key values are required", 400)
            deleted = _delete_row_engine(
                engine, table_name, pk_mapping, metadata
            )
        elif db_path:
            with _sqlite_connection(db_path) as conn:
                metadata = _sqlite_table_metadata(conn, table_name)
            if metadata is None:
                return _error_response("Table not found", 404)
            pk_columns = metadata.get("pk_columns", [])
            if not pk_columns:
                return _error_response("Table does not define a primary key", 400)
            pk_mapping = _parse_primary_key_mapping(pk_value, payload, pk_columns)
            if pk_mapping is None:
                return _error_response("Primary key values are required", 400)
            deleted = _delete_row_sqlite(
                db_path, table_name, pk_mapping, metadata
            )
        else:
            return _error_response("No database connection", 503)
    except PermissionError as exc:
        return _error_response(str(exc), 400)
    except ValueError as exc:
        return _error_response(str(exc), 400)
    except RuntimeError as exc:  # pragma: no cover - defensive
        return _error_response(str(exc), 500)

    if deleted is None:
        return _error_response("Table not found", 404)
    if deleted == 0:
        return _error_response("Row not found", 404)

    return jsonify({"status": "ok", "deleted": int(deleted)})
