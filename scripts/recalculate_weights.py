#!/usr/bin/env python
"""Recalculate importance weights with exponential decay.

This lightweight script updates the ``importance_score`` for all memories
in the database without rebuilding the cluster index.  The decay follows
an exponential curve ``score * exp(-λ * age_days)`` where ``λ`` can be
configured via the ``WEIGHT_DECAY_LAMBDA`` environment variable or the
``custom_settings.weight_decay_lambda`` configuration setting.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
import logging

from sqlalchemy import create_engine, text

from memoria.config.manager import ConfigManager
from memoria.database.backup_guard import backup_guard

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "recalculate_weights.log"

# Logging configuration
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///memoria.db")
DEFAULT_LAMBDA = 0.05


def _load_lambda() -> float:
    """Return decay parameter λ from environment or configuration."""
    config = ConfigManager()
    try:  # best-effort; config may not exist
        config.auto_load()
    except Exception:
        pass
    return float(
        os.getenv(
            "WEIGHT_DECAY_LAMBDA",
            config.get_setting("custom_settings.weight_decay_lambda", DEFAULT_LAMBDA),
        )
    )


def _recalculate_table(conn, table: str, lam: float) -> None:
    """Apply exponential decay to ``importance_score`` for a table."""
    now = datetime.now(timezone.utc)
    rows = conn.execute(
        text(f"SELECT memory_id, created_at, importance_score FROM {table}")
    ).fetchall()
    for memory_id, created_at, importance in rows:
        if not created_at:
            continue
        try:
            if isinstance(created_at, str):
                normalized = created_at.replace("Z", "+00:00")
                created_dt = datetime.fromisoformat(normalized)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                else:
                    created_dt = created_dt.astimezone(timezone.utc)
            else:
                if getattr(created_at, "tzinfo", None) is None:
                    created_dt = created_at.replace(tzinfo=timezone.utc)
                else:
                    created_dt = created_at.astimezone(timezone.utc)
        except Exception as exc:
            logger.warning(
                "Skipping memory %s in %s due to invalid timestamp %r: %s",
                memory_id,
                table,
                created_at,
                exc,
            )
            continue
        age_days = (now - created_dt).total_seconds() / 86400.0
        new_score = float(importance) * math.exp(-lam * age_days)
        conn.execute(
            text(
                f"UPDATE {table} SET importance_score = :score WHERE memory_id = :mid"
            ),
            {"score": new_score, "mid": memory_id},
        )


def recalculate_weights() -> None:
    lam = _load_lambda()
    logger.info("Using decay λ=%s", lam)
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        for table in ("short_term_memory", "long_term_memory"):
            _recalculate_table(conn, table, lam)
    logger.info("Completed weight recalculation")


def main() -> None:
    with backup_guard(DATABASE_URL):
        recalculate_weights()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # pragma: no cover - top-level logging
        logger.exception("Weight recalculation failed")
        raise
