#!/usr/bin/env python3
"""Wrapper for running `memoria policy test` with artifact output.

This helper mirrors the governance runbook instructions that call for a
"pager" dry-run to validate escalation policies. It forwards arguments to the
`memoria policy test` CLI command and captures the output in a Markdown report
suitable for attaching to change requests.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate pager escalation policies via `memoria policy test` and "
            "capture the results in an attachment-friendly format."
        )
    )
    parser.add_argument(
        "policy",
        help="Path to the JSON/YAML policy file (forwarded to `memoria policy test`).",
    )
    parser.add_argument(
        "--samples",
        help="Optional JSON or NDJSON dataset used for the simulation.",
    )
    parser.add_argument(
        "--output",
        help="Write the formatted report to this path (default: stdout only).",
    )
    parser.add_argument(
        "--memoria-bin",
        default="memoria",
        help="Executable used to invoke the CLI (default: memoria).",
    )
    return parser


def _format_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _render_report(command: Sequence[str], result: subprocess.CompletedProcess[str]) -> str:
    lines: list[str] = []
    lines.append("# Pager Dry-Run Report")
    lines.append("")
    lines.append(f"Command: `{_format_command(command)}`")
    lines.append("")

    if result.stdout.strip():
        lines.append("## Simulation Output")
        lines.append("```")
        lines.extend(result.stdout.rstrip().splitlines())
        lines.append("```")
        lines.append("")
    else:
        lines.append("_No output was produced by `memoria policy test`._")
        lines.append("")

    if result.stderr.strip():
        lines.append("## stderr")
        lines.append("```")
        lines.extend(result.stderr.rstrip().splitlines())
        lines.append("```")
        lines.append("")

    lines.append(f"Exit status: {result.returncode}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    command = [args.memoria_bin, "policy", "test", args.policy]
    if args.samples:
        command.extend(["--samples", args.samples])

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

    report = _render_report(command, result)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Report written to {output_path}", file=sys.stderr)

    print(report)
    return result.returncode


if __name__ == "__main__":  # pragma: no cover - manual invocation entrypoint
    sys.exit(main())
