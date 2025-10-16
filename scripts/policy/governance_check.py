#!/usr/bin/env python3
"""Capture governance telemetry snapshots via the `memoria policy telemetry` CLI."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect policy telemetry snapshots for governance reviews and "
            "generate an artifact that can be attached to change requests."
        )
    )
    parser.add_argument(
        "--vendor",
        help="Optional substring used to highlight policy actions for a vendor.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format produced by the telemetry command (default: json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of policy action records per snapshot.",
    )
    parser.add_argument(
        "--output",
        help="Write the report to this path (default: stdout only).",
    )
    parser.add_argument(
        "--memoria-bin",
        default="memoria",
        help="Executable used to invoke the CLI (default: memoria).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tag the report as a dry-run (no configuration changes).",
    )
    return parser


def _format_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _extract_vendor_matches(payload: dict[str, Any], vendor: str) -> list[dict[str, Any]]:
    vendor_lower = vendor.lower()
    matches: list[dict[str, Any]] = []
    for action in payload.get("policy_actions") or []:
        serialised = json.dumps(action, sort_keys=True)
        if vendor_lower in serialised.lower():
            matches.append(action)
    return matches


def _render_json_report(
    command: Sequence[str],
    snapshots: list[dict[str, Any]],
    vendor: str | None,
    dry_run: bool,
    stderr_output: str,
) -> str:
    lines: list[str] = []
    title = "Governance Telemetry Dry-Run" if dry_run else "Governance Telemetry Snapshot"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Command: `{_format_command(command)}`")
    if vendor:
        lines.append(f"Vendor filter: `{vendor}`")
    else:
        lines.append("Vendor filter: _not specified_")
    lines.append("")

    for index, snapshot in enumerate(snapshots, start=1):
        lines.append(f"## Snapshot {index}")
        matches: list[dict[str, Any]] = []
        if vendor:
            matches = _extract_vendor_matches(snapshot, vendor)
            if matches:
                lines.append(
                    f"{len(matches)} policy action(s) matched the vendor filter `{vendor}`."
                )
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(matches, indent=2, sort_keys=True))
                lines.append("```")
                lines.append("")
            else:
                lines.append(f"No policy actions matched the vendor filter `{vendor}`.")
                lines.append("")

        lines.append("```json")
        lines.append(json.dumps(snapshot, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

    if stderr_output.strip():
        lines.append("## stderr")
        lines.append("```")
        lines.extend(stderr_output.rstrip().splitlines())
        lines.append("```")
        lines.append("")

    lines.append(f"Total snapshots: {len(snapshots)}")
    lines.append("")
    return "\n".join(lines)


def _render_csv_report(
    command: Sequence[str],
    raw_output: str,
    dry_run: bool,
    stderr_output: str,
) -> str:
    title = "Governance Telemetry Dry-Run" if dry_run else "Governance Telemetry Snapshot"
    lines = [f"# {title}", "", f"Command: `{_format_command(command)}`", ""]
    lines.append("```csv")
    lines.extend(raw_output.rstrip().splitlines())
    lines.append("```")
    lines.append("")

    if stderr_output.strip():
        lines.append("## stderr")
        lines.append("```")
        lines.extend(stderr_output.rstrip().splitlines())
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    command = [args.memoria_bin, "policy", "telemetry", "--format", args.format]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on environment
        parser.error(f"Failed to invoke `{args.memoria_bin}`: {exc}")
        return 2

    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode

    report: str
    if args.format == "json":
        snapshots: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            try:
                snapshots.append(json.loads(line))
            except json.JSONDecodeError:
                # Include malformed JSON as plain text for troubleshooting.
                snapshots.append({"raw": line})
        report = _render_json_report(command, snapshots, args.vendor, args.dry_run, result.stderr)
    else:
        report = _render_csv_report(command, result.stdout, args.dry_run, result.stderr)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Report written to {output_path}", file=sys.stderr)

    print(report)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation entrypoint
    sys.exit(main())
