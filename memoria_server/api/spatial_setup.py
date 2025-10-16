"""Spatial metadata bootstrap helpers for Memoria's API."""

from __future__ import annotations

import json
import sqlite3

from flask import Flask
from sqlalchemy import text


def init_spatial_db(app: Flask) -> None:
    """Ensure auxiliary tables exist and migrate anchor data."""

    db_path = app.config.get("DB_PATH")
    engine = app.config.get("ENGINE")

    if db_path:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS spatial_metadata (
                memory_id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL DEFAULT 'default',
                x REAL,
                y REAL,
                z REAL,
                symbolic_anchors TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS service_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        cur.execute("PRAGMA table_info(spatial_metadata)")
        columns = [row[1] for row in cur.fetchall()]
        if "symbolic_anchors" not in columns and "symbolic_anchor" in columns:
            cur.execute(
                "ALTER TABLE spatial_metadata RENAME COLUMN symbolic_anchor TO symbolic_anchors"
            )
        if "namespace" not in columns:
            cur.execute(
                "ALTER TABLE spatial_metadata ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
            )
        cur.execute(
            "UPDATE spatial_metadata SET namespace = 'default' WHERE namespace IS NULL OR namespace = ''"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_spatial_namespace ON spatial_metadata(namespace)")
        cur.execute("SELECT memory_id, symbolic_anchors FROM spatial_metadata")
        rows = cur.fetchall()
        for memory_id, anchors in rows:
            if not anchors:
                continue
            try:
                json.loads(anchors)
            except json.JSONDecodeError:
                anchor_list = [a.strip() for a in anchors.split(",") if a.strip()]
                cur.execute(
                    "UPDATE spatial_metadata SET symbolic_anchors = ? WHERE memory_id = ?",
                    (json.dumps(anchor_list), memory_id),
                )
        conn.commit()
        conn.close()
        return

    if engine is None:
        return

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS spatial_metadata (
                    memory_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    x REAL,
                    y REAL,
                    z REAL,
                    symbolic_anchors TEXT
                )
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS service_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
        )

        try:
            conn.execute(
                text(
                    "ALTER TABLE spatial_metadata RENAME COLUMN symbolic_anchor TO symbolic_anchors"
                )
            )
        except Exception:
            pass

        try:
            conn.execute(
                text(
                    "ALTER TABLE spatial_metadata ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
                )
            )
        except Exception:
            pass

        try:
            conn.execute(
                text(
                    "UPDATE spatial_metadata SET namespace = 'default' "
                    "WHERE namespace IS NULL OR namespace = ''"
                )
            )
        except Exception:
            pass

        try:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_spatial_namespace ON spatial_metadata(namespace)"
                )
            )
        except Exception:
            try:
                conn.execute(
                    text(
                        "CREATE INDEX idx_spatial_namespace ON spatial_metadata(namespace)"
                    )
                )
            except Exception:
                pass

        result = conn.execute(text("SELECT memory_id, symbolic_anchors FROM spatial_metadata"))
        rows = result.fetchall()
        for memory_id, anchors in rows:
            if not anchors:
                continue
            try:
                json.loads(anchors)
            except json.JSONDecodeError:
                anchor_list = [a.strip() for a in anchors.split(",") if a.strip()]
                conn.execute(
                    text(
                        "UPDATE spatial_metadata SET symbolic_anchors = :anchors WHERE memory_id = :id"
                    ),
                    {"anchors": json.dumps(anchor_list), "id": memory_id},
                )
