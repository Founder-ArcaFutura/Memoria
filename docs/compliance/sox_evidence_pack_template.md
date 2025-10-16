# SOX Evidence Pack Template

This template documents the artefacts required to demonstrate Sarbanes-Oxley (SOX) readiness for a Memoria workspace. It maps
the persisted governance configuration in the `policy_artifacts` table to supporting audit outputs so reviewers can reconcile
controls, change history, and runtime signals.

## Workspace Overview

- **Workspace ID:** `{{workspace_id}}`
- **Reporting Period:** `{{iso8601_period_start}}` → `{{iso8601_period_end}}`
- **Primary Contact:** `{{name}} <{{email}}>`
- **Change Request Link:** `{{governance_change_request_url}}`

## Policy Artifact Inventory

Capture every SOX-relevant policy as stored in `policy_artifacts`.

| Artifact Name | Namespace | Artifact Type | Schema Version | Payload Digest (SHA256) | Storage Reference | Maintainer | Last Reviewed |
| ------------- | --------- | ------------- | -------------- | ----------------------- | ----------------- | ---------- | ------------- |
| `{{name}}` | `{{namespace}}` | `{{artifact_type}}` | `{{schema_version}}` | `{{payload_sha256}}` | `s3://.../policy_artifacts/{{id}}.json` | `{{maintainer}}` | `{{iso8601_timestamp}}` |

> **Traceability note:** For each row include the `id` from `policy_artifacts` (if stored separately), cross-referencing the
> linked payload and the change request or pull request that introduced the revision.

## Control Execution Evidence

List automation and manual control runs that validate the policies.

### Scheduled Automation Jobs

- **Job Identifier:** `{{job_name}}`
- **Run Timestamp:** `{{iso8601_timestamp}}`
- **Policy Scope:** `{{namespace}}`
- **Output Reference:** `{{log_or_artifact_url}}`
- **Result Summary:** `{{pass/fail + notes}}`

Include outputs from:

- `retention_policy_audits` (blocked/deleted events).
- `governance dry-run` CLI executions (`memoria policy dry-run ...`).
- Escalation roster verification reports.

### Manual Review Checklist

- [ ] Policy payload reviewed against business requirements.
- [ ] Segregation-of-duties exceptions documented with approval references.
- [ ] Override expirations validated against `policy_artifacts.payload.overrides` (if present).
- [ ] Evidence exported to the compliance archive (`{{archive_location}}`).

## Audit Log Excerpts

Summarise high-value events sourced from runtime audit tables.

| Source Table | Record Identifier | Action | Actor / Escalation Target | Timestamp | Evidence Reference |
| ------------ | ---------------- | ------ | ------------------------- | --------- | ------------------ |
| `retention_policy_audits` | `{{record_id}}` | `{{action}}` | `{{escalate_to}}` | `{{iso8601_timestamp}}` | `{{ndjson_or_csv_url}}` |
| `policy_artifacts` | `{{artifact_id}}` | `update` | `{{created_by}}` | `{{updated_at}}` | `{{pr_or_cr_link}}` |

> **Retention of evidence:** store referenced exports alongside the generated `policy_artifacts` bundle to simplify the GA
> release packaging workflow.

## Sign-off

- **Prepared By:** `{{name}}` on `{{iso8601_timestamp}}`
- **Reviewed By (Engineering):** `{{name}}` on `{{iso8601_timestamp}}`
- **Reviewed By (Compliance):** `{{name}}` on `{{iso8601_timestamp}}`
- **Executive Approval:** `{{name}}` on `{{iso8601_timestamp}}`
