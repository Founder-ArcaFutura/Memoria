# Post-GA Incident Review Template

Use this template for every governance-related incident that occurs after the GA launch. Each section should be filled before the review can be considered complete. Attach the rendered document to the incident ticket and store it with the corresponding release evidence bundle.

## Incident summary
- **Incident ID:**
- **Date opened:**
- **Date closed:**
- **Primary owners:**
- **Impacted services / namespaces:**
- **Severity:**
- **Detection channel:** (alert, audit trail, manual report, etc.)

## Timeline
Document the sequence of events from detection through resolution. Include timestamps with time zones and reference linked evidence (telemetry snapshots, audit exports, commits) inline.

| Timestamp | Actor | Event | Evidence |
| --- | --- | --- | --- |
| | | | |
| | | | |

## Telemetry & audit exports
List every telemetry or audit artifact gathered during the investigation. Attach the files produced by the governance console, CLI commands (for example `memoria policy telemetry --output ...`), or NDJSON exports. Provide a brief description for each attachment and note any gaps or missing data.

- `attachment-name.ext` – context on why it is relevant.
- `...`

## Policy impacts
Describe how policies were affected or updated. Reference the specific policy files, overrides, or escalation contact changes. Summarize dry-run results, policy telemetry deviations, and any manual adjustments taken during the incident.

## Follow-up actions
Capture remediation work, owners, and target due dates. Link to tickets or pull requests that track the follow-ups.

| Action item | Owner | Target date | Status |
| --- | --- | --- | --- |
| | | | |
| | | | |

## Distribution & sign-off
Record who reviewed and approved the incident report, along with the date when the template was circulated to maintainers.

- **Reviewers:**
- **Approval date:**
- **Distribution list:**
