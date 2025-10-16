#!/usr/bin/env python3
"""Export a migration snapshot containing memories, clusters, and relationships."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memoria.config.manager import ConfigManager
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager
from memoria.tools import migration


def export_memories(
    output: Path,
    *,
    namespaces: Sequence[str] | None = None,
    include_short_term: bool = True,
    include_clusters: bool = True,
    include_relationships: bool = True,
) -> dict[str, object]:
    """Export the configured database to ``output`` and return the payload."""

    config = ConfigManager.get_instance()
    settings = config.get_settings()
    manager = SQLAlchemyDatabaseManager(settings.database.connection_string)

    snapshot = migration.export_snapshot(
        manager,
        namespaces=namespaces,
        include_short_term=include_short_term,
        include_clusters=include_clusters,
        include_relationships=include_relationships,
    )

    payload = migration.dump_snapshot(snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    counts = snapshot.metadata.get("counts", {})
    logger.info(
        "Exported %s long-term, %s short-term, %s clusters, %s relationships",  # noqa: TRY003
        counts.get("long_term_memories", 0),
        counts.get("short_term_memories", 0),
        counts.get("clusters", 0),
        counts.get("relationships", 0),
    )
    logger.debug("Snapshot metadata: %s", snapshot.metadata)

    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export memories, clusters, and relationships to JSON",
    )
    parser.add_argument(
        "--namespace",
        dest="namespaces",
        action="append",
        default=[],
        help="Limit the export to the provided namespace (repeatable)",
    )
    parser.add_argument(
        "--no-short-term",
        dest="include_short_term",
        action="store_false",
        help="Exclude short-term memories from the export",
    )
    parser.add_argument(
        "--no-clusters",
        dest="include_clusters",
        action="store_false",
        help="Exclude cluster metadata from the export",
    )
    parser.add_argument(
        "--no-relationships",
        dest="include_relationships",
        action="store_false",
        help="Exclude relationship graph entries from the export",
    )
    parser.add_argument(
        "--output",
        help="Output file path (defaults to export/memories_<timestamp>.json)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    export_dir = Path(__file__).resolve().parents[1] / "export"
    export_dir.mkdir(exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = args.output or f"memories_{timestamp}.json"
    output_path = (Path(filename) if args.output else export_dir / filename).resolve()

    export_memories(
        output_path,
        namespaces=args.namespaces,
        include_short_term=args.include_short_term,
        include_clusters=args.include_clusters,
        include_relationships=args.include_relationships,
    )

    logger.info("Snapshot written to %s", output_path)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()

