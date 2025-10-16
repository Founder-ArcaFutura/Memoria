"""CI helper to simulate retention policies against high-risk sample payloads."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO_ROOT = Path(__file__).resolve().parents[2]

if TYPE_CHECKING:  # pragma: no cover
    from memoria.cli_support.policy_tooling import SimulationReport
    from memoria.config.settings import RetentionPolicyRuleSettings

DEFAULT_SAMPLE_PATH = REPO_ROOT / "tests/fixtures/policy_dry_run_samples.ndjson"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "policy-dry-run-artifacts"


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
        help="Path to the NDJSON samples used for the dry-run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory where the Markdown and JSON artifacts will be written",
    )
    return parser.parse_args(argv)


def _serialise_rule(rule: RetentionPolicyRuleSettings) -> dict[str, Any]:
    action = getattr(rule.action, "value", rule.action)
    payload: dict[str, Any] = {
        "name": rule.name,
        "namespaces": list(rule.namespaces or ("*",)),
        "action": str(action),
    }
    if rule.privacy_ceiling is not None:
        payload["privacy_ceiling"] = rule.privacy_ceiling
    if rule.importance_floor is not None:
        payload["importance_floor"] = rule.importance_floor
    if rule.lifecycle_days is not None:
        payload["lifecycle_days"] = rule.lifecycle_days
    if rule.escalate_to:
        payload["escalate_to"] = rule.escalate_to
    if rule.metadata:
        payload["metadata"] = dict(rule.metadata)
    return payload


def _serialise_report(report: SimulationReport) -> dict[str, Any]:
    return {
        "rule": _serialise_rule(report.rule),
        "total_samples": report.total_samples,
        "trigger_count": report.trigger_count,
        "hits": [
            {
                "sample_index": hit.sample_index,
                "memory_id": hit.memory_id,
                "reasons": list(hit.reasons),
            }
            for hit in report.hits
        ],
    }


def _render_markdown(reports: list[SimulationReport], total_samples: int) -> str:
    lines: list[str] = ["# Policy dry-run summary", ""]
    if total_samples == 0 or not reports:
        lines.append(
            "No retention policy rules were simulated or the sample dataset was empty."
        )
        return "\n".join(lines) + "\n"

    lines.append(f"Total sample records: **{total_samples}**")
    lines.append("")
    for report in reports:
        rule = report.rule
        lines.append(f"## {rule.name}")
        lines.append("")
        lines.append(
            "| Trigger count | Total samples | Namespaces | Action | Reasons |"
        )
        lines.append("| --- | --- | --- | --- | --- |")
        reason_sets = {
            ", ".join(sorted(hit.reasons)) or "(unspecified)" for hit in report.hits
        }
        reasons_text = ", ".join(sorted(reason_sets)) or "None"
        lines.append(
            "| {trigger} | {total} | {namespaces} | {action} | {reasons} |".format(
                trigger=report.trigger_count,
                total=report.total_samples,
                namespaces=", ".join(rule.namespaces or ("*",)),
                action=getattr(rule.action, "value", rule.action),
                reasons=reasons_text,
            )
        )
        if report.hits:
            lines.append("")
            lines.append("### Triggered samples")
            lines.append("")
            lines.append("| Sample index | Memory ID | Reasons |")
            lines.append("| --- | --- | --- |")
            for hit in report.hits:
                lines.append(
                    "| {index} | {memory} | {reasons} |".format(
                        index=hit.sample_index,
                        memory=hit.memory_id or "(n/a)",
                        reasons=", ".join(hit.reasons) or "(unspecified)",
                    )
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    from memoria.cli_support.policy_tooling import simulate_policies
    from memoria.config.manager import ConfigManager

    args = _parse_args(argv)
    samples_path = args.samples.resolve()
    artifact_dir = args.output.resolve()

    if not samples_path.exists():
        raise FileNotFoundError(f"Sample file not found: {samples_path}")

    artifact_dir.mkdir(parents=True, exist_ok=True)

    manager = ConfigManager.get_instance()
    manager.auto_load()
    settings = manager.get_settings()
    policies = list(getattr(settings.memory, "retention_policy_rules", []) or [])

    reports = simulate_policies(policies, sample_path=samples_path)

    serialised = [_serialise_report(report) for report in reports]
    total_samples = serialised[0]["total_samples"] if serialised else 0

    json_path = artifact_dir / "policy_dry_run_results.json"
    json_path.write_text(
        json.dumps({"reports": serialised, "total_samples": total_samples}, indent=2),
        encoding="utf-8",
    )

    markdown_path = artifact_dir / "policy_dry_run_summary.md"
    markdown_path.write_text(
        _render_markdown(reports, total_samples), encoding="utf-8"
    )

    print(f"Wrote policy dry-run artifacts to {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
