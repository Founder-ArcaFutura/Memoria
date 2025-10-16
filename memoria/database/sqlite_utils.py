"""Utility helpers for SQLite-specific behavior."""

from __future__ import annotations

__all__ = ["is_sqlite_json_error"]


def is_sqlite_json_error(exc: Exception) -> bool:
    """Return ``True`` if *exc* indicates missing SQLite JSON1 support."""
    message = str(getattr(exc, "orig", exc)).lower()
    markers = (
        "json_each",
        "json_extract",
        "json_insert",
        "json_remove",
        "json_replace",
        "json_set",
        "json_group_array",
        "json_array",
        "json_valid",
        "json1",
        "no such function: json",
    )
    return any(marker in message for marker in markers)
