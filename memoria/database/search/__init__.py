"""
Search adapters for different database backends
"""

from .mysql_search_adapter import MySQLSearchAdapter
from .sqlite_search_adapter import SQLiteSearchAdapter

__all__ = ["SQLiteSearchAdapter", "MySQLSearchAdapter"]
