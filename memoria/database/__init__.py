"""Database components for Memoria"""

from .analytics import (
    CategoryCount,
    RetentionSeries,
    UsageRecord,
    get_analytics_summary,
    get_category_counts,
    get_retention_trends,
    get_usage_frequency,
)
from .connectors import MySQLConnector, PostgreSQLConnector, SQLiteConnector

try:  # Lazy import to avoid circular dependency during initialization
    from .sqlalchemy_manager import SQLAlchemyDatabaseManager
except Exception:  # pragma: no cover - fallback when config not ready
    SQLAlchemyDatabaseManager = None

__all__ = [
    "SQLiteConnector",
    "PostgreSQLConnector",
    "MySQLConnector",
    "SQLAlchemyDatabaseManager",
    "CategoryCount",
    "RetentionSeries",
    "UsageRecord",
    "get_category_counts",
    "get_retention_trends",
    "get_usage_frequency",
    "get_analytics_summary",
]
