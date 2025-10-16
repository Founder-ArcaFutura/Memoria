"""Simple dashboard utilities for Memoria."""

from __future__ import annotations

from ..config.manager import ConfigManager
from ..database import SQLAlchemyDatabaseManager


def show_dashboard(namespace: str = "default") -> dict[str, object]:
    """Display memory counts and category breakdown for a namespace.

    This helper uses :class:`SQLAlchemyDatabaseManager` to collect basic
    statistics about stored memories and prints a human-friendly summary.

    Args:
        namespace: Target namespace to summarize.

    Returns:
        The statistics dictionary returned by
        :meth:`SQLAlchemyDatabaseManager.get_memory_stats`.
    """
    settings = ConfigManager.get_instance().get_settings()
    manager = SQLAlchemyDatabaseManager(settings.database.connection_string)

    try:
        stats = manager.get_memory_stats(namespace)
    finally:
        manager.close()

    print(f"\n📊 Memory dashboard for namespace '{namespace}':")
    print(f"  Chat history entries: {stats.get('chat_history_count', 0)}")
    print(f"  Short-term memories: {stats.get('short_term_count', 0)}")
    print(f"  Long-term memories: {stats.get('long_term_count', 0)}")
    print("  Memories by category:")
    for category, count in stats.get("memories_by_category", {}).items():
        print(f"    {category}: {count}")
    print(f"  Average importance: {stats.get('average_importance', 0.0):.2f}\n")

    return stats
