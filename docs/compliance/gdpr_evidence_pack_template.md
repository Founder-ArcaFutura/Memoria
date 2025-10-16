# GDPR Evidence Pack Template

Use this template to assemble GDPR accountability evidence before publishing a GA build. It links configured governance controls
(`policy_artifacts`) with the audit outputs that demonstrate data subject protection, lawful processing, and incident handling.

## Processing Context

- **Workspace ID:** `{{workspace_id}}`
- **Controller / Processor Role:** `{{controller_or_processor}}`
- **Data Protection Officer:** `{{dpo_name}} <{{email}}>`
- **Assessment Window:** `{{iso8601_period_start}}` → `{{iso8601_period_end}}`
- **Relevant Change Requests:** `{{links_to_policy_crs}}`

## Policy-to-Control Mapping

Document GDPR-relevant policy artefacts and link them to their operational enforcement.

| Policy Name | Namespace | Article Coverage | Artifact Type | Schema Version | Payload Digest (SHA256) | Enforcement Mechanism |
| ----------- | --------- | ---------------- | ------------- | -------------- | ----------------------- | --------------------- |
| `{{name}}` | `{{namespace}}` | `{{gdpr_article_reference}}` | `{{artifact_type}}` | `{{schema_version}}` | `{{payload_sha256}}` | `{{automation_job_or_cli_workflow}}` |

> **Source:** `policy_artifacts` – include the numeric `id` and `created_by` in the appendix for traceability.

## Audit Evidence Catalogue

### Retention & Deletion Events

Summarise enforced retention actions and data subject requests.

| Memory ID | Namespace | Policy Name | Action | Escalation Target | Timestamp | Export Reference |
| --------- | --------- | ----------- | ------ | ----------------- | --------- | ---------------- |
| `{{memory_id}}` | `{{namespace}}` | `{{policy_name}}` | `{{action}}` | `{{escalate_to}}` | `{{iso8601_timestamp}}` | `{{ndjson_or_csv_url}}` |

Populate from `retention_policy_audits` exports to show automated deletion/hold outcomes.

### Subject Request Tracking

- **Request Identifier:** `{{request_id}}`
- **Origin:** `portal|email|api`
- **Received At:** `{{iso8601_timestamp}}`
- **Fulfillment Timestamp:** `{{iso8601_timestamp}}`
- **Linked Policy Artifact:** `{{policy_artifact_name}}`
- **Evidence Reference:** `{{ticketing_or_archive_link}}`

### Incident & Breach Reporting

- [ ] Incident response plan attached (link to `policy_artifacts` payload or repo path).
- [ ] Breach notification workflow executed (attach automation job output if triggered).
- [ ] Data transfer impact assessments archived.

## Data Flow Validation

Document reviews validating data minimisation and access controls.

1. **Collection Review:** `{{notes}}`
2. **Processing Review:** `{{notes}}`
3. **Storage & Retention Review:** `{{notes}}`
4. **Access Review:** `{{notes}}`

Reference relevant policy payload sections (e.g., `policy_artifacts.payload.retention_rules`) when capturing findings.

## Sign-off

- **Prepared By:** `{{name}}` on `{{iso8601_timestamp}}`
- **DPO Approval:** `{{name}}` on `{{iso8601_timestamp}}`
- **Security Approval:** `{{name}}` on `{{iso8601_timestamp}}`
- **Executive Approval:** `{{name}}` on `{{iso8601_timestamp}}`
