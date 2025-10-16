"""Utility script for running the documentation quickstart scenario."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from memoria import Memoria


def _estimate_tokens(text: str) -> int:
    """Return a lightweight token estimate for manual memories."""
    word_count = max(1, len(text.split()))
    return max(8, math.ceil(word_count * 1.2))


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "memory"


def load_scenario(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    namespace = scenario.get("namespace", "quickstart")
    memoria = Memoria(
        database_connect=scenario.get(
            "database", f"sqlite:///{namespace}_evaluation.db"
        ),
        conscious_ingest=scenario.get("conscious_ingest", True),
        auto_ingest=scenario.get("auto_ingest", True),
        namespace=namespace,
        verbose=scenario.get("verbose", False),
    )
    memoria.enable()

    inserted = []
    for idx, fact in enumerate(scenario.get("facts", []), start=1):
        text = fact["text"].strip()
        anchor = fact.get("anchor") or f"fact-{idx}-{_slugify(text[:24])}"
        tokens = fact.get("tokens") or _estimate_tokens(text)
        memoria.store_memory(
            anchor=anchor,
            text=text,
            tokens=tokens,
            x_coord=fact.get("x"),
            y=fact.get("y"),
            z=fact.get("z"),
            symbolic_anchors=fact.get("symbolic_anchors"),
            namespace=namespace,
        )
        inserted.append(anchor)

    results = []
    for probe in scenario.get("probes", []):
        query = probe["query"]
        prompt = memoria.get_auto_ingest_system_prompt(query)
        expected = probe.get("expects", [])
        missing = [needle for needle in expected if needle.lower() not in prompt.lower()]
        results.append(
            {
                "label": probe.get("label", query),
                "query": query,
                "prompt": prompt,
                "expects": expected,
                "missing": missing,
                "passed": not missing,
            }
        )

    stats = memoria.db_manager.get_memory_stats(namespace)
    memoria.disable()

    summary = {
        "scenario": {
            "name": scenario.get("name", "quickstart"),
            "description": scenario.get("description", ""),
            "namespace": namespace,
        },
        "inserted": inserted,
        "probes": results,
        "stats": stats,
    }
    return summary


def print_report(report: dict[str, Any]) -> None:
    scenario = report["scenario"]
    print(f"Scenario: {scenario['name']}")
    if scenario["description"]:
        print(f"Description: {scenario['description']}")
    print(f"Namespace: {scenario['namespace']}")
    print()
    print("Memories staged:")
    for anchor in report["inserted"]:
        print(f"  • {anchor}")
    if not report["inserted"]:
        print("  (none)")

    print()
    print("Probe results:")
    for probe in report["probes"]:
        status = "PASS" if probe["passed"] else "FAIL"
        print(f"- {probe['label']}: {status}")
        print(f"  query   : {probe['query']}")
        print(f"  expects : {probe['expects']}")
        if probe["missing"]:
            print(f"  missing : {probe['missing']}")
        print(f"  prompt  : {probe['prompt'].strip() or '(no context)'}")
    if not report["probes"]:
        print("  (no probes configured)")

    print()
    print("Memory stats:")
    for key, value in sorted(report["stats"].items()):
        print(f"  {key}: {value}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Memoria evaluation quickstart scenario.")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("examples/evaluation_quickstart.json"),
        help="Path to a JSON scenario file.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path to write the JSON report.",
    )
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    report = run_scenario(scenario)
    print_report(report)

    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
