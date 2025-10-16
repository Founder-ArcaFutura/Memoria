"""Shared constants for conscious context management."""

# Primary category assigned to memories promoted into short-term storage by the
# conscious ingest workflows. Using an ``essential_`` prefix keeps the values
# compatible with existing essential-memory queries (e.g.,
# ``get_essential_conversations``) while distinguishing the conscious source.
CONSCIOUS_CONTEXT_CATEGORY = "essential_conscious"

__all__ = ["CONSCIOUS_CONTEXT_CATEGORY"]
