"""Capture a policy telemetry snapshot for CI reviewers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from memoria.cli import (
    _load_policy_metrics_collector_for_cli,
    _snapshot_to_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "telemetry-artifacts"


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the telemetry snapshot artifacts will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional limit for the number of policy action rows to include in the"
            " snapshot. Use -1 to include all rows."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = args.limit if args.limit is not None else None
    if limit is not None and limit < 0:
        limit = None

    collector, app = _load_policy_metrics_collector_for_cli()
    _ = app  # Keep reference alive for any background tasks.

    snapshot = collector.capture_snapshot(limit=limit)
    payload = snapshot.to_payload(limit=limit, round_durations=3)

    json_path = output_dir / "policy_telemetry.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_text = _snapshot_to_csv(payload, include_header=True)
    if not csv_text.endswith("\n"):
        csv_text += "\n"
    csv_path = output_dir / "policy_telemetry.csv"
    csv_path.write_text(csv_text, encoding="utf-8")

    print(f"Wrote policy telemetry snapshot to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
