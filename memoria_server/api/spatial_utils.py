from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Sequence

from memoria.database.models import SpatialMetadata


def upsert_spatial_metadata(
    *,
    memory_id: str,
    namespace: str,
    db_path: str | None,
    db_manager: object | None,
    timestamp: datetime | str | None,
    x: float | None,
    y: float | None,
    z: float | None,
    symbolic_anchors: Sequence[str] | None,
    ensure_timestamp: bool = False,
    prefer_db_manager: bool = False,
) -> None:
    """Persist spatial metadata for a memory in SQLite or SQLAlchemy stores."""

    anchors_value: list[str] = list(symbolic_anchors or [])
    session_factory = getattr(db_manager, "SessionLocal", None)

    def _upsert_sqlite(path: str) -> None:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(spatial_metadata)")
        columns = [row[1] for row in cur.fetchall()]

        ts_value: str | None
        raw_timestamp = timestamp
        if isinstance(raw_timestamp, datetime):
            ts_value = raw_timestamp.isoformat()
        else:
            ts_value = raw_timestamp

        if ts_value is None and ensure_timestamp and "timestamp" in columns:
            ts_value = datetime.now(timezone.utc).isoformat()

        anchors_json = json.dumps(anchors_value)

        if "timestamp" in columns:
            cur.execute(
                """
                INSERT OR REPLACE INTO spatial_metadata
                (memory_id, namespace, timestamp, x, y, z, symbolic_anchors)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    namespace,
                    ts_value,
                    x,
                    y,
                    z,
                    anchors_json,
                ),
            )
        else:
            cur.execute(
                """
                INSERT OR REPLACE INTO spatial_metadata
                (memory_id, namespace, x, y, z, symbolic_anchors)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    namespace,
                    x,
                    y,
                    z,
                    anchors_json,
                ),
            )

        conn.commit()
        conn.close()

    def _upsert_sqlalchemy() -> None:
        if session_factory is None:
            return

        with session_factory() as session:
            spatial_entry = (
                session.query(SpatialMetadata)
                .filter_by(memory_id=memory_id, namespace=namespace)
                .one_or_none()
            )

            if spatial_entry is None:
                spatial_entry = SpatialMetadata(
                    memory_id=memory_id,
                    namespace=namespace,
                )
                session.add(spatial_entry)

            if timestamp is not None:
                spatial_entry.timestamp = timestamp
            elif ensure_timestamp and spatial_entry.timestamp is None:
                spatial_entry.timestamp = datetime.now(timezone.utc)

            spatial_entry.x = x
            spatial_entry.y = y
            spatial_entry.z = z
            spatial_entry.symbolic_anchors = anchors_value

            session.commit()

    if prefer_db_manager and session_factory is not None:
        _upsert_sqlalchemy()
        return

    if db_path:
        _upsert_sqlite(db_path)
        return

    if session_factory is not None:
        _upsert_sqlalchemy()
