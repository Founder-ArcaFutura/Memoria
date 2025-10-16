#!/usr/bin/env python
"""Cluster database texts using simple heuristics."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import List

from loguru import logger
from sqlalchemy import create_engine, text

from memoria.config.manager import ConfigManager


def _sqlite_url_from_path(db_path: str) -> str:
    """Convert a filesystem path to a SQLAlchemy SQLite URL."""

    path = str(db_path)
    return path if path.startswith("sqlite://") else f"sqlite:///{path}"


def resolve_database_url(connection_string: str | None = None) -> str:
    """Resolve the database connection string at call time."""

    if connection_string:
        return connection_string

    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    try:
        from flask import current_app, has_app_context  # type: ignore

        try:
            if has_app_context():
                url = current_app.config.get("DATABASE_URL")
                if url:
                    return url
                db_path = current_app.config.get("DB_PATH")
                if db_path:
                    return _sqlite_url_from_path(db_path)
        except RuntimeError:
            # No active Flask application context
            pass
    except Exception:
        # Flask is optional for this module
        pass

    config_manager = ConfigManager.get_instance()
    try:
        settings = config_manager.get_settings()
    except Exception:
        try:
            config_manager.auto_load()
            settings = config_manager.get_settings()
        except Exception:
            return "sqlite:///memoria.db"

    if hasattr(settings, "get_database_url"):
        try:
            url = settings.get_database_url()
            if url:
                return url
        except Exception:
            pass

    database = getattr(settings, "database", None)
    connection = getattr(database, "connection_string", None)
    return connection or "sqlite:///memoria.db"


def _load_lambda() -> float:
    """Return decay parameter λ from environment or configuration."""

    config_manager = ConfigManager.get_instance()
    try:  # best-effort configuration load
        config_manager.auto_load()
    except Exception:
        pass

    raw_value = os.getenv(
        "WEIGHT_DECAY_LAMBDA",
        config_manager.get_setting(
            "custom_settings.weight_decay_lambda", DEFAULT_LAMBDA
        ),
    )

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid decay λ value {} – falling back to default {}",
            raw_value,
            DEFAULT_LAMBDA,
        )
        return float(DEFAULT_LAMBDA)


# Limit the number of clusters logged for brevity
MAX_LOG_CLUSTERS = 5
DEFAULT_LAMBDA = 1.0 / (7 * 24 * 60 * 60)

# Scoring parameters
ALPHA = 1 / 3
BETA = 1 / 3
GAMMA = 1 / 3

TOP_N_SEEDS = 10
TEMPORAL_THRESHOLD = 1.0

try:
    CENTROID_SMOOTHING = float(os.getenv("CLUSTER_CENTROID_SMOOTHING", "1.0"))
except ValueError:
    CENTROID_SMOOTHING = 1.0
if CENTROID_SMOOTHING <= 0:
    CENTROID_SMOOTHING = 1.0
CENTROID_SMOOTHING = min(CENTROID_SMOOTHING, 1.0)


def _blend_centroid(previous: float | None, target: float) -> float:
    if previous is None or CENTROID_SMOOTHING >= 1.0:
        return target
    return previous + CENTROID_SMOOTHING * (target - previous)


def fetch_texts(connection_string: str | None = None) -> list[dict]:
    """Fetch memory rows from the database.

    Returns a list of dictionaries containing memory metadata including
    spatial coordinates.
    """
    engine = create_engine(resolve_database_url(connection_string))
    rows: list[dict] = []

    try:
        with engine.connect() as conn:
            for table in ("long_term_memory", "short_term_memory"):
                try:
                    result = conn.execute(
                        text(
                            f"SELECT memory_id, summary, symbolic_anchors, x_coord, y_coord, z_coord FROM {table}"
                        )
                    )
                    for (
                        memory_id,
                        summary,
                        anchors,
                        x_coord,
                        y_coord,
                        z_coord,
                    ) in result:
                        if not summary:
                            continue
                        anchor_list: list[str] = []
                        if anchors:
                            if isinstance(anchors, str):
                                try:
                                    anchor_list = json.loads(anchors)
                                except Exception:
                                    anchor_list = [
                                        a.strip()
                                        for a in anchors.split(",")
                                        if a.strip()
                                    ]
                            else:
                                anchor_list = list(anchors)
                        rows.append(
                            {
                                "memory_id": memory_id,
                                "anchors": anchor_list,
                                "summary": summary,
                                "x": float(x_coord or 0.0),
                                "y": float(y_coord) if y_coord is not None else None,
                                "z": float(z_coord) if z_coord is not None else None,
                            }
                        )
                except Exception:  # pragma: no cover - table may not exist
                    logger.exception("Failed to fetch texts from %s", table)
                    continue
    except Exception:
        logger.exception("Database connection failed")
    return rows


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_heuristic_clusters(
    connection_string: str | None = None, verbose: bool = False
) -> tuple[list[dict], str]:
    """Build clusters using meta_score and temporal/anchor heuristics.

    Returns a tuple of ``(clusters, summary)`` where ``summary`` describes the
    outcome.  Returning the cluster list alongside a textual summary makes it
    easier for callers to surface partial results when an error occurs.
    """
    sink_id = logger.add(sys.stderr, level="INFO" if verbose else "WARNING")

    try:

        try:
            rows = fetch_texts(connection_string)
            logger.info("Fetched {} texts", len(rows))
        except Exception as e:
            logger.exception("Failed to fetch texts")
            return [], f"Failed to fetch texts: {e}"

        if not rows:
            return [], "No texts fetched"

        lam = _load_lambda()
        logger.info("Using decay λ={}", lam)

        # Compute token counts and anchor frequencies
        anchor_counts: Counter[str] = Counter()
        max_tokens = 1
        for row in rows:
            tokens = _tokenize(row["summary"])
            row["token_count"] = len(tokens)
            max_tokens = max(max_tokens, row["token_count"])
            if row["anchors"]:
                anchor_counts[row["anchors"][0]] += 1

        max_anchor_freq = max(anchor_counts.values(), default=1)

        # Calculate meta scores
        for row in rows:
            anchor = row["anchors"][0] if row["anchors"] else ""
            token_weight = row["token_count"] / max_tokens
            recency_weight = math.exp(-lam * abs(row["x"]))
            anchor_weight = (
                anchor_counts.get(anchor, 0) / max_anchor_freq if anchor else 0.0
            )
            row["meta_score"] = (
                ALPHA * token_weight + BETA * recency_weight + GAMMA * anchor_weight
            )

        # Sort by meta score and select seeds
        sorted_rows = sorted(rows, key=lambda r: r["meta_score"], reverse=True)
        seeds = sorted_rows[:TOP_N_SEEDS]

        clusters: List[dict] = []
        assigned = set()

        for seed in seeds:
            if seed["memory_id"] in assigned:

                continue
            cluster_members = [seed]
            assigned.add(seed["memory_id"])
            seed_anchor_set = set(seed["anchors"])
            for row in sorted_rows:
                if row["memory_id"] in assigned:
                    continue
                if (
                    seed_anchor_set & set(row["anchors"])
                    and abs(row["x"] - seed["x"]) <= TEMPORAL_THRESHOLD
                ):
                    cluster_members.append(row)
                    assigned.add(row["memory_id"])
            clusters.append(
                {
                    "id": len(clusters),
                    "anchor": seed["anchors"][0] if seed["anchors"] else "",
                    "members": cluster_members,
                }
            )

        for cluster in clusters:
            ys = [m.get("y") for m in cluster["members"] if m.get("y") is not None]
            zs = [m.get("z") for m in cluster["members"] if m.get("z") is not None]
            if ys:
                avg_y = sum(ys) / len(ys)
                cluster["y_centroid"] = _blend_centroid(
                    cluster.get("y_centroid"), avg_y
                )
            else:
                cluster["y_centroid"] = None
            if zs:
                avg_z = sum(zs) / len(zs)
                cluster["z_centroid"] = _blend_centroid(
                    cluster.get("z_centroid"), avg_z
                )
            else:
                cluster["z_centroid"] = None
            cluster["size"] = len(cluster["members"])
            cluster["avg_meta_score"] = (
                sum(m["meta_score"] for m in cluster["members"]) / cluster["size"]
                if cluster["size"]
                else 0.0
            )
        summary = f"Built {len(clusters)} clusters"
        logger.info(summary)
        return clusters, summary
    finally:
        logger.remove(sink_id)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Build heuristic clusters")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    clusters, summary = build_heuristic_clusters(verbose=args.verbose)
    print(summary)
    for c in clusters[:MAX_LOG_CLUSTERS]:
        print(c)
