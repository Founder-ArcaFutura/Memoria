# Memoria Phased Release Guide

Generated via `scripts/releases/generate_release_guides.py`. Attach the relevant sections to governance and evaluation change requests to demonstrate alignment with the release roadmap.

Refer to the [GA launch announcement](ga-launch-announcement.md) for upgrade sequencing, rollback commands, and operator hand-offs that must accompany the GA release package.

## Phase 1 – Beta Hardening

Stabilise the governance surface and ensure automation jobs behave predictably while the community scales early access deployments.

### Automation Focus

- Automate policy dry-runs for high-risk namespaces and capture trace IDs for review
- Schedule escalation roster verification jobs and document rotation outcomes
- Validate bootstrap scripts (e.g. `memoria bootstrap`) on clean environments

### Analytics Focus

- Enable policy enforcement telemetry exports for dashboard aggregation
- Baseline evaluation suite latency and cost metrics across providers
- Publish beta adoption stats into the shared analytics workspace

### Release Checkpoints

- Governance dry-run output attached to change requests
- Evaluation suite summaries uploaded for retrieval-affecting changes
- Telemetry schema updates reviewed and merged
- Incident response contacts verified via automation job output

## Phase 2 – Release Candidate

Bundle automation guardrails with regression monitoring so the release candidate mirrors GA operations, including escalation readiness.

### Automation Focus

- Promote scheduled guardrails (override expiry, roster rotation) to production timers
- Document CLI workflows that mirror the dashboard governance controls
- Expand CI evaluation workflows with regression thresholds and PR annotations

### Analytics Focus

- Wire evaluation outputs into governance dashboards for combined reporting
- Track latency percentiles and provider mix changes across RC builds
- Enable alerting on telemetry anomalies surfaced during RC burn-in

### Release Checkpoints

- Release guide attached to change request with sign-off timestamps
- Automation playbooks linked in Additional Notes of PRs
- Analytics dashboards updated with RC baseline views
- Roll-forward and rollback commands tested via CI pipelines (captured in the `Run release command checks` CI step)

## Phase 3 – General Availability

Finalize documentation, analytics, and automation so self-managed and hosted deployments share the same operational posture.

### Automation Focus

- Harden backup/restore scripts and document recovery point objectives
- Enable telemetry exports feeding external observability tooling
- Codify governance certification packs (SOX/GDPR) in the release artifact
- Bundle compliance evidence templates (`docs/compliance/*.md` and `.json`) with GA binaries

### Analytics Focus

- Publish GA dashboards with comparative trends across beta and RC
- Store evaluation outputs in long-term analytics backends (Parquet/SQL)
- Update public roadmap trackers with automation and analytics status

### Release Checkpoints

- [GA launch announcement](ga-launch-announcement.md) drafted with upgrade and rollback instructions
- GA launch announcement attached to hosted and self-managed release bundles
- Phased release checklist archived with links to automation logs
- Analytics backfills completed for legacy deployments
- Post-GA incident review template circulated to maintainers ([docs/admin/post_ga_incident_template.md](../../admin/post_ga_incident_template.md))
- GA artifact bundle validated to include SOX/GDPR evidence packs
