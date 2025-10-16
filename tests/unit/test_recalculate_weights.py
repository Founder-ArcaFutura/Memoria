import math
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

import scripts.recalculate_weights as recalc


def _freeze_datetime(monkeypatch: pytest.MonkeyPatch, fixed_now: datetime) -> None:
    real_datetime = recalc.datetime

    class _FrozenDateTime:
        @classmethod
        def now(cls, tz=None):
            return fixed_now

        @classmethod
        def fromisoformat(cls, value):
            return real_datetime.fromisoformat(value)

    monkeypatch.setattr(recalc, "datetime", _FrozenDateTime)


def _create_memory_table(engine, rows):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE memory (
                    memory_id INTEGER PRIMARY KEY,
                    created_at TEXT,
                    importance_score REAL
                )
                """
            )
        )
        conn.execute(
            text(
                "INSERT INTO memory (memory_id, created_at, importance_score)"
                " VALUES (:mid, :created_at, :importance)"
            ),
            rows,
        )


def test_recalculate_table_parses_naive_and_utc_strings(
    monkeypatch: pytest.MonkeyPatch,
):
    engine = create_engine("sqlite:///:memory:")
    _create_memory_table(
        engine,
        [
            {"mid": 1, "created_at": "2024-01-01T00:00:00", "importance": 1.0},
            {"mid": 2, "created_at": "2024-01-02T00:00:00+00:00", "importance": 2.0},
            {"mid": 3, "created_at": "2024-01-02T12:00:00Z", "importance": 3.0},
        ],
    )

    fixed_now = datetime(2024, 1, 3, tzinfo=timezone.utc)
    _freeze_datetime(monkeypatch, fixed_now)

    with engine.begin() as conn:
        recalc._recalculate_table(conn, "memory", lam=0.1)
        rows = (
            conn.execute(
                text(
                    "SELECT memory_id, importance_score FROM memory ORDER BY memory_id"
                )
            )
            .mappings()
            .all()
        )

    assert rows[0]["importance_score"] == pytest.approx(math.exp(-0.2))
    assert rows[1]["importance_score"] == pytest.approx(2.0 * math.exp(-0.1))
    assert rows[2]["importance_score"] == pytest.approx(3.0 * math.exp(-0.1 / 2))


def test_recalculate_table_logs_warning_for_bad_timestamp(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    engine = create_engine("sqlite:///:memory:")
    _create_memory_table(
        engine,
        [
            {"mid": 1, "created_at": "2024-01-01T00:00:00", "importance": 1.0},
            {"mid": 2, "created_at": "not-a-date", "importance": 4.0},
        ],
    )

    fixed_now = datetime(2024, 1, 2, tzinfo=timezone.utc)
    _freeze_datetime(monkeypatch, fixed_now)
    caplog.set_level("WARNING", logger=recalc.logger.name)

    with engine.begin() as conn:
        recalc._recalculate_table(conn, "memory", lam=0.2)
        rows = (
            conn.execute(
                text(
                    "SELECT memory_id, importance_score FROM memory ORDER BY memory_id"
                )
            )
            .mappings()
            .all()
        )

    assert rows[0]["importance_score"] == pytest.approx(math.exp(-0.2))
    assert rows[1]["importance_score"] == 4.0
    assert "Skipping memory" in caplog.text
