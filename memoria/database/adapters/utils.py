"""Utility helpers for database adapters."""

from __future__ import annotations

import re

_INDEX_NAME_PATTERN = re.compile(
    r"\b(?:INDEX|FULLTEXT)\b\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:(?P<quote>[`\"'])(?P<quoted>[^`\"']+)(?P=quote)|(?P<unquoted>[A-Za-z_][\w$]*))",
    flags=re.IGNORECASE,
)


def extract_index_name(ddl: str) -> str | None:
    """Return the index identifier from a DDL statement.

    The function searches for the token immediately following ``INDEX`` or
    ``FULLTEXT`` in a CREATE/ALTER DDL statement and returns the identifier
    without any surrounding quotes. If no identifier is found, ``None`` is
    returned.
    """

    if not ddl:
        return None

    match = _INDEX_NAME_PATTERN.search(ddl)
    if not match:
        return None

    name = match.group("quoted") or match.group("unquoted")
    return name.strip() if name else None


__all__ = ["extract_index_name"]
