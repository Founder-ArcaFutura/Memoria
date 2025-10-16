"""Generate phased release guides for Memoria.

This helper bundles the automation and analytics upgrades called out in the
roadmap into checklists that can be attached to governance and evaluation
change requests. Use it to keep beta, release candidate, and GA launches
aligned with the documented rollout expectations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


GA_LAUNCH_ANNOUNCEMENT_PATH = Path("docs/advanced_usage/ga-launch-announcement.md")
RELEASE_COMMUNICATIONS: List[Path] = [GA_LAUNCH_ANNOUNCEMENT_PATH]


@dataclass(frozen=True)
class ReleasePhase:
    """Structured representation of a release phase."""

    key: str
    title: str
    description: str
    automation_focus: List[str]
    analytics_focus: List[str]
    checkpoints: List[str]

    def to_markdown(self) -> str:
        """Render the phase as Markdown."""

        lines: List[str] = [f"## {self.title}", "", self.description, ""]

        def section(header: str, items: Iterable[str]) -> None:
            lines.append(f"### {header}")
            lines.append("")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")

        section(
            "Automation Focus",
            self.automation_focus,
        )
        section(
            "Analytics Focus",
            self.analytics_focus,
        )
        section(
            "Release Checkpoints",
            self.checkpoints,
        )

        return "\n".join(lines).rstrip()


PHASES: List[ReleasePhase] = [
    ReleasePhase(
        key="beta",
        title="Phase 1 – Beta Hardening",
        description=(
            "Stabilise the governance surface and ensure automation jobs"
            " behave predictably while the community scales early access deployments."
        ),
        automation_focus=[
            "Automate policy dry-runs for high-risk namespaces and capture trace IDs for review",
            "Schedule escalation roster verification jobs and document rotation outcomes",
            "Validate bootstrap scripts (e.g. `memoria bootstrap`) on clean environments",
        ],
        analytics_focus=[
            "Enable policy enforcement telemetry exports for dashboard aggregation",
            "Baseline evaluation suite latency and cost metrics across providers",
            "Publish beta adoption stats into the shared analytics workspace",
        ],
        checkpoints=[
            "Governance dry-run output attached to change requests",
            "Evaluation suite summaries uploaded for retrieval-affecting changes",
            "Telemetry schema updates reviewed and merged",
            "Incident response contacts verified via automation job output",
        ],
    ),
    ReleasePhase(
        key="rc",
        title="Phase 2 – Release Candidate",
        description=(
            "Bundle automation guardrails with regression monitoring so the release"
            " candidate mirrors GA operations, including escalation readiness."
        ),
        automation_focus=[
            "Promote scheduled guardrails (override expiry, roster rotation) to production timers",
            "Document CLI workflows that mirror the dashboard governance controls",
            "Expand CI evaluation workflows with regression thresholds and PR annotations",
        ],
        analytics_focus=[
            "Wire evaluation outputs into governance dashboards for combined reporting",
            "Track latency percentiles and provider mix changes across RC builds",
            "Enable alerting on telemetry anomalies surfaced during RC burn-in",
        ],
        checkpoints=[
            "Release guide attached to change request with sign-off timestamps",
            "Automation playbooks linked in Additional Notes of PRs",
            "Analytics dashboards updated with RC baseline views",
            "Roll-forward and rollback commands tested via CI pipelines (captured in the `Run release command checks` CI step)",
        ],
    ),
    ReleasePhase(
        key="ga",
        title="Phase 3 – General Availability",
        description=(
            "Finalize documentation, analytics, and automation so self-managed"
            " and hosted deployments share the same operational posture."
        ),
        automation_focus=[
            "Harden backup/restore scripts and document recovery point objectives",
            "Enable telemetry exports feeding external observability tooling",
            "Codify governance certification packs (SOX/GDPR) in the release artifact",
            "Bundle compliance evidence templates (`docs/compliance/*.md` and `.json`) with GA binaries",
        ],
        analytics_focus=[
            "Publish GA dashboards with comparative trends across beta and RC",
            "Store evaluation outputs in long-term analytics backends (Parquet/SQL)",
            "Update public roadmap trackers with automation and analytics status",
        ],
        checkpoints=[
            "[GA launch announcement](ga-launch-announcement.md) drafted with upgrade and rollback instructions",
            "GA launch announcement attached to hosted and self-managed release bundles",
            "Phased release checklist archived with links to automation logs",
            "Analytics backfills completed for legacy deployments",
            "Post-GA incident review template circulated to maintainers",
            "GA artifact bundle validated to include SOX/GDPR evidence packs",
        ],
    ),
]


def build_markdown(phases: Iterable[ReleasePhase]) -> str:
    """Return Markdown containing the release guide for the provided phases."""

    rendered = [
        "# Memoria Phased Release Guide",
        "",
        (
            "Generated via `scripts/releases/generate_release_guides.py`. Attach the"
            " relevant sections to governance and evaluation change requests to"
            " demonstrate alignment with the release roadmap."
        ),
        "",
        (
            "Refer to the [GA launch announcement](ga-launch-announcement.md) for"
            " upgrade sequencing, rollback commands, and operator hand-offs that"
            " must accompany the GA release package."
        ),
        "",
    ]

    for phase in phases:
        rendered.append(phase.to_markdown())
        rendered.append("")

    return "\n".join(rendered).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate release guide markdown")
    parser.add_argument(
        "--phase",
        choices=[phase.key for phase in PHASES],
        help="Render only the specified phase",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to write the generated markdown",
    )
    parser.add_argument(
        "--list-communications",
        action="store_true",
        help=(
            "Print newline-separated documentation files that must be bundled"
            " with release artifacts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phases: Iterable[ReleasePhase]

    if args.phase:
        phase_lookup = {phase.key: phase for phase in PHASES}
        phases = [phase_lookup[args.phase]]
    else:
        phases = PHASES

    markdown = build_markdown(phases)

    if args.list_communications:
        for path in RELEASE_COMMUNICATIONS:
            print(path.as_posix())
        return

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(markdown)
    else:
        print(markdown)


if __name__ == "__main__":
    main()

