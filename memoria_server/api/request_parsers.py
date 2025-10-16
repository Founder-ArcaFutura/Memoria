from __future__ import annotations

"""Helpers for parsing request payload fields."""

from datetime import datetime, timedelta, timezone
from typing import Any


class CoordinateParsingError(ValueError):
    """Raised when a coordinate value cannot be converted to ``float``."""

    def __init__(self, field_name: str, value: Any):
        message = f"{field_name} must be a numeric value"
        super().__init__(message)
        self.field_name = field_name
        self.value = value


def parse_optional_coordinate(value: Any, field_name: str) -> float | None:
    """Parse an optional coordinate value from a request payload.

    Parameters
    ----------
    value:
        The raw value from the request payload.
    field_name:
        Name of the field being parsed (used for error messaging).

    Returns
    -------
    float | None
        ``None`` when the value is empty, otherwise the converted float.

    Raises
    ------
    CoordinateParsingError
        If the value cannot be converted to ``float``.
    """

    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - TypeError defensive
        raise CoordinateParsingError(field_name, value) from exc


def timestamp_from_x(x_coord: float | None) -> str | None:
    """Derive an ISO formatted timestamp from an ``x`` coordinate."""

    if x_coord is None:
        return None
    ts = datetime.now(timezone.utc) + timedelta(days=float(x_coord))
    return ts.isoformat()

