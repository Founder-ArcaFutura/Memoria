"""
Database adapters for different database backends
Provides database-specific implementations with proper security measures
"""

from .mysql_adapter import MySQLSearchAdapter
from .postgresql_adapter import PostgreSQLSearchAdapter
from .sqlite_adapter import SQLiteSearchAdapter

__all__ = ["SQLiteSearchAdapter", "PostgreSQLSearchAdapter", "MySQLSearchAdapter"]
