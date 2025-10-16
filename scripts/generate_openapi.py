#!/usr/bin/env python3
"""Synchronise the committed OpenAPI artifacts.

The project maintains a hand-authored ``openapi.json`` file that contains the
fully typed specification used by API consumers.  This helper script keeps the
JSON and YAML exports in sync and enforces a consistent formatting style.

Run without arguments to normalise ``openapi.json`` and regenerate
``openapi_1.yaml``.  Use ``--check`` in CI to verify that the committed
artifacts are up to date.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = ROOT / "openapi.json"
OUTPUT_YAML = ROOT / "openapi_1.yaml"
ENCODING = "utf-8"


def _load_json_spec() -> dict:
    """Load the canonical JSON specification from disk."""

    if not OUTPUT_JSON.exists():
        raise FileNotFoundError(
            "openapi.json is missing. Update the canonical specification before running this script."
        )
    with OUTPUT_JSON.open("r", encoding=ENCODING) as fp:
        return json.load(fp)


def _serialise_json(spec: dict) -> str:
    """Return the canonical JSON serialisation for the specification."""

    return json.dumps(spec, indent=2) + "\n"


def _serialise_yaml(spec: dict) -> str:
    """Return the canonical YAML serialisation for the specification."""

    return yaml.safe_dump(spec, sort_keys=False)


def _write_files(spec: dict) -> None:
    """Write the normalised JSON and YAML artifacts to disk."""

    OUTPUT_JSON.write_text(_serialise_json(spec), encoding=ENCODING)
    OUTPUT_YAML.write_text(_serialise_yaml(spec), encoding=ENCODING)


def _check_files(spec: dict) -> int:
    """Validate that the committed artifacts match the canonical serialisation."""

    current_json = OUTPUT_JSON.read_text(encoding=ENCODING) if OUTPUT_JSON.exists() else ""
    expected_json = _serialise_json(spec)
    current_yaml = OUTPUT_YAML.read_text(encoding=ENCODING) if OUTPUT_YAML.exists() else ""
    expected_yaml = _serialise_yaml(spec)
    if current_json != expected_json or current_yaml != expected_yaml:
        print(
            "OpenAPI spec artifacts are out of date. Run scripts/generate_openapi.py to regenerate."
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate the committed spec")
    args = parser.parse_args()

    spec = _load_json_spec()
    if args.check:
        return _check_files(spec)

    _write_files(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
