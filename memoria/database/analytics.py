"""Analytics helpers for aggregating memory statistics.

The helpers operate on SQLAlchemy sessions so they can be reused by the
REST API, Streamlit dashboards, or ad-hoc scripts without coupling them to
any specific Flask application state.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from .models import LongTermMemory, ShortTermMemory


@dataclass
class CategoryCount:
    """Simple data structure representing category tallies."""

    category: str
    count: int


@dataclass
class RetentionSeries:
    """Time-series data for a specific retention type."""

    retention_type: str
    daily_counts: dict[str, int]


@dataclass
class UsageRecord:
    """Represents the usage metrics for a specific memory record."""

    memory_id: str
    summary: str
    category: str | None
    access_count: int
    last_accessed: datetime | None


def _normalise_category(value: str | None) -> str:
    cleaned = (value or "").strip()
    return cleaned or "uncategorised"


def _date_range(days: int, end: datetime | None = None) -> list[str]:
    if days <= 0:
        days = 1
    end = end or datetime.utcnow()
    start = end - timedelta(days=days - 1)
    return [
        (start + timedelta(days=offset)).date().isoformat() for offset in range(days)
    ]


def get_category_counts(
    session: Session,
    *,
    namespace: str | None = None,
    include_short_term: bool = True,
) -> dict[str, list[CategoryCount]]:
    """Return category tallies for long- and short-term memories."""

    results: dict[str, list[CategoryCount]] = {"long_term": [], "short_term": []}

    category_col = LongTermMemory.category_primary
    count_col = func.count(LongTermMemory.memory_id).label("count")

    long_query = session.query(category_col, count_col)
    if namespace:
        long_query = long_query.filter(LongTermMemory.namespace == namespace)
    long_query = long_query.group_by(category_col).order_by(count_col.desc())

    results["long_term"] = [
        CategoryCount(_normalise_category(category), int(count))
        for category, count in long_query.all()
    ]

    if include_short_term:
        st_category_col = ShortTermMemory.category_primary
        st_count_col = func.count(ShortTermMemory.memory_id).label("count")

        short_query = session.query(st_category_col, st_count_col)
        if namespace:
            short_query = short_query.filter(ShortTermMemory.namespace == namespace)
        short_query = short_query.group_by(st_category_col).order_by(
            st_count_col.desc()
        )

        results["short_term"] = [
            CategoryCount(_normalise_category(category), int(count))
            for category, count in short_query.all()
        ]

    return results


def get_retention_trends(
    session: Session,
    *,
    days: int = 30,
    namespace: str | None = None,
    include_short_term: bool = True,
) -> dict[str, dict[str, list[RetentionSeries]]]:
    """Return time-series counts of created memories per retention type."""

    buckets = _date_range(days)
    bucket_set = set(buckets)
    start_date = datetime.fromisoformat(buckets[0])
    end_date = datetime.fromisoformat(buckets[-1]) + timedelta(days=1)

    def _series_for(rows: Iterable[tuple[str | None, datetime | None]]):
        grouped: dict[str, dict[str, int]] = defaultdict(
            lambda: dict.fromkeys(buckets, 0)
        )
        for retention_type, created_at in rows:
            if not created_at:
                continue
            day = created_at.date().isoformat()
            if day not in bucket_set:
                continue
            key = (retention_type or "unknown").strip() or "unknown"
            grouped[key][day] += 1
        return [
            RetentionSeries(retention_type=key, daily_counts=dict(counts))
            for key, counts in grouped.items()
        ]

    long_query = session.query(LongTermMemory.retention_type, LongTermMemory.created_at)
    long_query = long_query.filter(LongTermMemory.created_at >= start_date)
    long_query = long_query.filter(LongTermMemory.created_at < end_date)
    if namespace:
        long_query = long_query.filter(LongTermMemory.namespace == namespace)

    result: dict[str, dict[str, list[RetentionSeries]]] = {
        "range": {"start": buckets[0], "end": buckets[-1]},
        "long_term": {"series": _series_for(long_query.all())},
    }

    if include_short_term:
        short_query = session.query(
            ShortTermMemory.retention_type, ShortTermMemory.created_at
        )
        short_query = short_query.filter(ShortTermMemory.created_at >= start_date)
        short_query = short_query.filter(ShortTermMemory.created_at < end_date)
        if namespace:
            short_query = short_query.filter(ShortTermMemory.namespace == namespace)
        result["short_term"] = {"series": _series_for(short_query.all())}

    return result


def get_usage_frequency(
    session: Session,
    *,
    namespace: str | None = None,
    include_short_term: bool = True,
    top_n: int = 10,
) -> dict[str, Any]:
    """Return usage frequency statistics for memories."""

    top_n = max(1, min(top_n, 100))

    def _usage_query(model):
        query = session.query(
            model.memory_id,
            model.summary,
            model.category_primary,
            model.access_count,
            model.last_accessed,
        )
        if namespace:
            query = query.filter(model.namespace == namespace)
        query = query.filter(model.access_count.isnot(None))
        query = query.order_by(model.access_count.desc())
        return query

    def _aggregate(model):
        count_query = session.query(func.count(model.memory_id))
        sum_query = session.query(func.coalesce(func.sum(model.access_count), 0))
        if namespace:
            count_query = count_query.filter(model.namespace == namespace)
            sum_query = sum_query.filter(model.namespace == namespace)

        total_records = int(count_query.scalar() or 0)
        total_accesses = int(sum_query.scalar() or 0)

        records = [
            UsageRecord(
                memory_id=memory_id,
                summary=(summary or "").strip(),
                category=(category or None),
                access_count=int(access_count or 0),
                last_accessed=last_accessed,
            )
            for memory_id, summary, category, access_count, last_accessed in _usage_query(
                model
            ).limit(
                top_n
            )
        ]

        average = float(total_accesses) / total_records if total_records else 0.0

        return {
            "total_records": total_records,
            "total_accesses": total_accesses,
            "average_accesses": average,
            "top_records": records,
        }

    payload: dict[str, Any] = {"long_term": _aggregate(LongTermMemory)}

    if include_short_term:
        payload["short_term"] = _aggregate(ShortTermMemory)

    return payload


def get_analytics_summary(
    session: Session,
    *,
    namespace: str | None = None,
    include_short_term: bool = True,
    days: int = 30,
    top_n: int = 10,
) -> dict[str, Any]:
    """Return a combined analytics payload used by the dashboard and API."""

    return {
        "namespace": namespace or "all",
        "include_short_term": bool(include_short_term),
        "category_counts": get_category_counts(
            session,
            namespace=namespace,
            include_short_term=include_short_term,
        ),
        "retention_trends": get_retention_trends(
            session,
            days=days,
            namespace=namespace,
            include_short_term=include_short_term,
        ),
        "usage_frequency": get_usage_frequency(
            session,
            namespace=namespace,
            include_short_term=include_short_term,
            top_n=top_n,
        ),
    }


__all__ = [
    "CategoryCount",
    "RetentionSeries",
    "UsageRecord",
    "get_category_counts",
    "get_retention_trends",
    "get_usage_frequency",
    "get_analytics_summary",
]
