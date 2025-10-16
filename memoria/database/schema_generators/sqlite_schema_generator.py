"""SQLite schema generator that loads the bundled SQL template."""

from __future__ import annotations

from pathlib import Path

from ...utils.exceptions import DatabaseError


class SQLiteSchemaGenerator:
    """Generate the SQLite schema from the shipped SQL template."""

    def __init__(self, template_path: Path | None = None) -> None:
        self._template_path = (
            Path(template_path)
            if template_path is not None
            else Path(__file__).resolve().parent.parent
            / "templates"
            / "schemas"
            / "basic.sql"
        )

    @property
    def template_path(self) -> Path:
        """Return the resolved template path."""
        return self._template_path

    def generate_full_schema(self) -> str:
        """Return the full SQLite schema as a string.

        Raises:
            DatabaseError: If the schema template cannot be read or is empty.
        """

        try:
            schema_sql = self.template_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise DatabaseError(
                f"SQLite schema template not found at {self.template_path}"
            ) from exc
        except OSError as exc:  # pragma: no cover - platform-specific errors
            raise DatabaseError(
                f"Unable to read SQLite schema template at {self.template_path}: {exc}"
            ) from exc

        schema_sql = schema_sql.strip()
        if not schema_sql:
            raise DatabaseError(
                f"SQLite schema template at {self.template_path} is empty"
            )

        return schema_sql
