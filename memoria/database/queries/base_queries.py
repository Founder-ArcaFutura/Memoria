"""Base database queries and schema operations."""

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "templates" / "schemas" / "basic.sql"
)

_TABLE_PATTERN = re.compile(
    r"^CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
    re.IGNORECASE,
)
_INDEX_PATTERN = re.compile(
    r"^CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
    re.IGNORECASE,
)
_TRIGGER_PATTERN = re.compile(
    r"^CREATE\s+TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)", re.IGNORECASE
)


def _normalize_statement(statement: str) -> str:
    """Return a consistently formatted SQL statement."""

    lines = [line.rstrip() for line in statement.splitlines()]
    return "\n".join(lines).strip()


def _split_sql_statements(schema_path: Path) -> list[str]:
    """Split the schema file into individual SQL statements."""

    statements = []
    current_lines = []

    with schema_path.open("r", encoding="utf-8") as sql_file:
        for raw_line in sql_file:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("--"):
                if not stripped and current_lines:
                    current_lines.append("")
                continue

            if "--" in raw_line:
                raw_line = raw_line.split("--", 1)[0]

            line = raw_line.rstrip()
            if not line:
                if current_lines:
                    current_lines.append("")
                continue

            while ";" in line:
                before, line = line.split(";", 1)
                before = before.rstrip()
                if before:
                    current_lines.append(before)

                statement = "\n".join(current_lines).strip()
                if statement:
                    statements.append(statement)
                current_lines = []
                line = line.lstrip()

            if line:
                current_lines.append(line.rstrip())

    return statements


def load_schema_from_sql(schema_path: Path = SCHEMA_PATH) -> dict[str, dict[str, str]]:
    """Load schema components from the canonical SQL definition file."""

    tables: dict[str, str] = {}
    indexes: dict[str, str] = {}
    triggers: dict[str, str] = {}

    for statement in _split_sql_statements(schema_path):
        normalized = _normalize_statement(statement)

        if match := _TABLE_PATTERN.match(normalized):
            table_name = match.group(1)
            if table_name in tables:
                raise ValueError(f"Duplicate table definition for {table_name}")
            tables[table_name] = normalized
        elif match := _INDEX_PATTERN.match(normalized):
            index_name = match.group(1)
            if index_name in indexes:
                raise ValueError(f"Duplicate index definition for {index_name}")
            indexes[index_name] = normalized
        elif match := _TRIGGER_PATTERN.match(normalized):
            trigger_name = match.group(1)
            if trigger_name in triggers:
                raise ValueError(f"Duplicate trigger definition for {trigger_name}")
            triggers[trigger_name] = normalized

    return {"tables": tables, "indexes": indexes, "triggers": triggers}


_SCHEMA_COMPONENTS = load_schema_from_sql()


class BaseQueries(ABC):
    """Abstract base class for database queries."""

    @abstractmethod
    def get_table_creation_queries(self) -> dict[str, str]:
        """Return dictionary of table creation SQL statements."""

    @abstractmethod
    def get_index_creation_queries(self) -> dict[str, str]:
        """Return dictionary of index creation SQL statements."""

    @abstractmethod
    def get_trigger_creation_queries(self) -> dict[str, str]:
        """Return dictionary of trigger creation SQL statements."""


class SchemaQueries:
    """Schema management queries sourced from the canonical SQL file."""

    SCHEMA_PATH: Path = SCHEMA_PATH
    TABLE_CREATION: Mapping[str, str] = MappingProxyType(_SCHEMA_COMPONENTS["tables"])
    INDEX_CREATION: Mapping[str, str] = MappingProxyType(_SCHEMA_COMPONENTS["indexes"])
    TRIGGER_CREATION: Mapping[str, str] = MappingProxyType(
        _SCHEMA_COMPONENTS["triggers"]
    )
