#!/usr/bin/env python3
"""Validate pytest JUnit results against an allowed failure threshold."""

from __future__ import annotations

import argparse
import pathlib
import sys
import xml.etree.ElementTree as ET


def _coerce_int(value: str | None) -> int:
    """Convert the given attribute value to an integer."""
    if value is None:
        return 0
    value = value.strip()
    if not value:
        return 0
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unable to parse integer value from '{value}'.") from exc


def _collect_suites(root: ET.Element) -> list[ET.Element]:
    """Return a flat list of all <testsuite> elements without duplicates."""
    if root.tag == "testsuite":
        return [root]
    if root.tag == "testsuites":
        return list(root.findall("testsuite"))
    # Fallback: collect recursively but ensure tag name matches to avoid double counting.
    return [element for element in root.iter() if element.tag == "testsuite"]


def evaluate(report_path: pathlib.Path, threshold: int) -> int:
    """Evaluate the report and return the combined failures + errors count."""
    tree = ET.parse(report_path)
    root = tree.getroot()
    failures = 0
    errors = 0
    for suite in _collect_suites(root):
        failures += _coerce_int(suite.attrib.get("failures"))
        errors += _coerce_int(suite.attrib.get("errors"))
    total = failures + errors
    print(
        f"Detected {failures} failures and {errors} errors across all suites. "
        f"Allowed threshold: {threshold}."
    )
    if total > threshold:
        print(
            "::error title=Pytest threshold::The number of failing tests exceeds the allowed threshold."
        )
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=pathlib.Path, help="Path to a pytest JUnit XML report")
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Allowed number of combined failures and errors before exiting with a non-zero status.",
    )
    args = parser.parse_args()

    if not args.report.exists():
        raise SystemExit(f"Report not found: {args.report}")

    total = evaluate(args.report, args.threshold)
    return 0 if total <= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
