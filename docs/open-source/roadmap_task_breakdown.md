# Roadmap Implementation Tasks

This document breaks the remaining roadmap themes into actionable engineering tasks. Each workstream is scoped so independent contributors can pick up issues incrementally while maintaining alignment with the published roadmap in `features.md`.

## Status Update

- ✅ **Governance foundation delivered**: Declarative rule schemas, runtime enforcement hooks, CLI workflows, persistence layers, and audit trails now ship in the core runtime, unlocking policy-driven lifecycle management across ingestion, retrieval, and sync paths.【F:memoria/config/settings.py†L62-L170】【F:memoria/policy/enforcement.py†L19-L213】【F:memoria/cli.py†L895-L933】【F:memoria/storage/service.py†L196-L565】【F:memoria/database/models.py†L426-L480】
- ✅ **Evaluation harness available**: Packaged scenario specs, dataset generators, materialisation helpers, and benchmarking utilities provide repeatable load/quality assessments with policy-compliance and cost metrics built in.【F:memoria/evaluation/spec.py†L19-L178】【F:memoria/evaluation/loader.py†L1-L157】【F:memoria/evaluation/datasets.py†L1-L160】【F:memoria/tools/benchmark.py†L1-L220】【F:memoria/evaluation/specs/default_suites.yaml†L1-L128】
- ✅ **Admin governance console shipped**: The dashboard now exposes namespace segmentation, escalation roster management, and audit exploration directly in the browser with dry-run tooling aligned to the CLI workflows.【F:docs/admin/policy_console_guide.md†L1-L86】
- ✅ **CLI runbook published**: A dedicated runbook maps every governance console action to scripted `memoria policy` commands so operators can automate rollouts and incident response.【F:docs/admin/policy_console_runbook.md†L1-L90】【F:memoria/cli.py†L895-L1942】
- ✅ **Evaluation CI automation live**: Pull requests trigger the evaluation workflow to materialise smoke suites, generate benchmark summaries, and upload artifacts for review, matching the documentation and CI helper script.【F:docs/configuration/evaluation-suites.md†L80-L144】【F:.github/workflows/evaluation.yml†L1-L69】【F:scripts/ci/run_evaluation_suites.py†L1-L160】
- ✅ **Automation guardrails wired up**: Scheduled roster rotations and override sweeps keep escalation contacts and temporary exceptions within their SLAs while surfacing status cards in the governance console.【F:memoria_server/api/scheduler.py†L214-L357】【F:memoria_server/api/governance_routes.py†L235-L357】【F:memoria_server/dashboard/app.js†L3336-L3441】【F:memoria_server/dashboard/app.js†L4058-L4136】
- ✅ **Enforcement telemetry surfaced end-to-end**: The policy metrics collector now emits per-action counters with observer streaming, the governance API serves JSON and NDJSON snapshots, the CLI streams CSV/JSON/webhook payloads, and the dashboard panel renders live telemetry for operators.【F:memoria/policy/enforcement.py†L210-L360】【F:memoria_server/api/governance_routes.py†L607-L656】【F:memoria/cli.py†L1193-L1942】【F:memoria_server/dashboard/app.js†L4070-L4160】【F:docs/admin/policy_console_guide.md†L83-L106】
- ✅ **Escalation automation live**: Background schedulers now verify roster coverage, rotate on-call metadata, and expire overrides while the governance API and runbooks expose refresh endpoints and persisted status payloads.【F:memoria_server/api/scheduler.py†L336-L558】【F:memoria_server/api/governance_routes.py†L833-L1159】【F:docs/admin/policy_console_runbook.md†L40-L57】
- ✅ **Evaluation guardrails & tooling delivered**: CI enforces precision/recall/cost thresholds, posts PR summaries, and captures rich metadata while the CLI scaffolds suite specs and the docs outline end-to-end workflows.【F:.github/workflows/evaluation.yml†L87-L145】【F:scripts/ci/run_evaluation_suites.py†L120-L420】【F:memoria/cli.py†L1549-L1619】【F:memoria/cli.py†L667-L695】【F:memoria/cli_support/evaluation_scaffold.py†L10-L112】【F:docs/configuration/evaluation-suites.md†L100-L239】
- ✅ **Certification evidence packs published**: SOX and GDPR templates map `policy_artifacts` records to runtime audit outputs for release packaging, enabling GA bundles to ship with compliance-ready archives.【F:docs/compliance/sox_evidence_pack_template.md†L1-L89】【F:docs/compliance/sox_evidence_pack_template.json†L1-L56】【F:docs/compliance/gdpr_evidence_pack_template.md†L1-L78】【F:docs/compliance/gdpr_evidence_pack_template.json†L1-L64】

> **Release readiness:** Public documentation, examples, and governance trackers now reference [https://github.com/Founder-ArcaFutura/Memoria](https://github.com/Founder-ArcaFutura/Memoria). Remaining roadmap tasks focus on sustaining the 0.9 alpha baseline—hardening optional infrastructure extras, broadening validation coverage, and shaping the backlog for upcoming pre-1.0 feature drops.

## Governance Next Steps

### Validation & Certification
- Expand integration tests that ingest across privacy bands to assert enforcement decisions, audit persistence, and policy metrics stay consistent release-to-release.【F:memoria/heuristics/retention.py†L289-L349】【F:memoria/storage/service.py†L283-L512】
- Create governance-focused contribution templates capturing scope, acceptance criteria, and rollout considerations to streamline review cycles.【F:CONTRIBUTING.md†L1-L120】

## Evaluation Harness Next Steps

All evaluation guardrails, authoring helpers, and reporting upgrades from the previous milestone have shipped. New harness initiatives will be documented here as the next release train is scoped.

## Cross-cutting Tasks
- Continue removing residual Arca Futura infrastructure references from scripts, docs, and configuration while preserving attribution for contributed features.【F:AGENTS.md†L64-L101】
- Update issue templates and contribution guidelines to reflect the new governance and evaluation workflows, ensuring reviewers can request benchmark runs or policy dry-runs consistently.【F:CONTRIBUTING.md†L1-L226】
- Plan phased releases (beta/GA) that bundle UI, automation, and evaluation improvements with explicit adoption guides for self-managed and hosted deployments.【F:README.md†L1-L120】【F:docs/advanced_usage/phased-guides.md†L1-L68】【F:scripts/releases/generate_release_guides.py†L1-L171】
