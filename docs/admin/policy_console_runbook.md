# Policy Console Runbook

This runbook maps the new dashboard governance surfaces to their command-line
counterparts so operations teams can execute the same workflows through GitOps or
remote automation. The CLI references use the `memoria policy` command group
implemented in `memoria/cli.py`. If you prefer Python-based automation, see the
[PolicyClient SDK guide](../sdk/policy-client.md) for end-to-end examples that mirror
these governance tasks through the HTTP API.

## Namespace segmentation & dry-runs

| UI action | CLI equivalent |
| --- | --- |
| Refresh namespace summary | `memoria policy lint <file>` to validate policy files before applying updates. |
| Run dry-run for selected namespace | `memoria policy test <file> --samples sample.json` – produces the same violation summary returned by the UI modal. |
| Apply confirmed rules | `memoria policy apply <file>` writes the policies into the active configuration via the `ConfigManager`. |

### Escalation of issues found in dry-runs

1. Export the dry-run report from the UI or run the CLI with `--json` redirects.
2. File a ticket referencing the policy names and violation reasons.
3. Update the policy file, re-run `memoria policy lint`, and repeat the dry-run.

### CI artifacts for pull request reviews

Pull requests that modify `memoria/policy/**` automatically execute
`scripts/ci/run_policy_dry_runs.py`, which loads the active configuration and runs
the simulator against the high-risk sample dataset. The GitHub Action uploads a
`policy-dry-run` artifact containing:

- `policy_dry_run_summary.md` – reviewer-friendly Markdown with per-rule totals.
- `policy_dry_run_results.json` – structured data that mirrors the CLI output.

To include the dry-run evidence in change requests:

1. Open the PR’s **Checks** tab and select the `CI` workflow run.
2. Scroll to the **Artifacts** section, download `policy-dry-run.zip`, and extract
   it locally.
3. Attach `policy_dry_run_summary.md` (or the JSON file for automation tickets) to
   the review thread or incident record so the governance team can audit the
   simulated violations alongside the policy diff.

Policy pull requests now also run `scripts/ci/run_policy_telemetry_snapshot.py`
from the `CI` workflow (see the **Capture policy telemetry snapshot** step). The
script writes `telemetry-artifacts/policy_telemetry.json` and
`telemetry-artifacts/policy_telemetry.csv`, and the workflow uploads them as the
`policy-telemetry` artifact for reviewers. Reference these files in the
[Post-GA incident review template](post_ga_incident_template.md) when compiling
evidence for incident follow-ups.【F:.github/workflows/ci.yml†L53-L86】【F:scripts/ci/run_policy_telemetry_snapshot.py†L1-L63】

To add the telemetry evidence to a change request:

1. Open the PR’s **Checks** tab and select the latest `CI` workflow run. If the
   run is still queued, wait for the `Capture policy telemetry snapshot` step to
   finish.
2. Scroll to **Artifacts**, download `policy-telemetry.zip`, and extract it
   locally so you have `policy_telemetry.json` and `policy_telemetry.csv`.
3. Attach `policy_telemetry.json` (plus the CSV when spreadsheets help) to the
   change-request ticket or PR review so governance can diff the baseline policy
   enforcement counts alongside the proposed configuration changes. For
   incidents, drop the artifacts into the **Telemetry & audit exports** section
   of the [Post-GA incident review template](post_ga_incident_template.md).

## Escalation roster updates

Escalation contacts are stored under `policy.escalation_contacts` in the
Memoria configuration. Use the REST endpoints when scripting changes, or follow
these CLI-oriented steps:

1. Export the current configuration: `memoria settings export --include-sensitive > memoria.json`.
2. Edit the `policy.escalation_contacts` array, ensuring each entry matches the
   `EscalationContact` schema (name, channel, target, optional metadata).
3. Re-import the file with `memoria settings import memoria.json` or use the new
   `POST /governance/escalations` endpoint for incremental changes.
4. Validate integration hooks by sending a preview notification from the UI or by
   calling `POST /governance/escalations/<name>/preview` in automation.

### Automation guardrails

Two background jobs keep the roster metadata and overrides aligned with their
service-level agreements:

- `GET /governance/escalations/rotation` exposes the scheduler that rewrites the
  active/on-deck rotation windows, marks overdue hand-offs, and persists the
  sanitized roster back into `policy.escalation_contacts` for audit trails. Use
  `memoria policy export` followed by a `jq '.policy.escalation_contacts'` check
  to diff automation output against Git-managed manifests.【F:memoria_server/api/scheduler.py†L214-L307】【F:memoria_server/api/governance_routes.py†L235-L288】
- `GET /governance/overrides/automation` returns the periodic sweep that stamps
  expired overrides with `override_status`, `expired_at`, and `expired_by` before
  committing the metadata changes through the `ConfigManager`. Pair the API with
  `memoria policy overrides list --json` to capture the automation payload in
  change requests.【F:memoria/policy/schemas.py†L154-L217】【F:memoria_server/api/governance_routes.py†L300-L357】

Both endpoints accept `?refresh=1` when you need to force a recalculation after
making manual edits through Git or the CLI. The dashboard surfaces the same
timelines so operations teams can verify that automation ran before they close
an incident or compliance ticket.【F:memoria_server/dashboard/app.js†L3367-L3441】【F:memoria_server/dashboard/app.js†L4094-L4136】

### Capturing roster evidence from the CLI

The `memoria roster` command mirrors the verification and rotation evidence that
the governance console produces so on-call engineers can attach machine-readable
artifacts to tickets:

```bash
# Export JSON telemetry summarising roster health
memoria roster verify --cadence 60 --format json > roster-verification.json

# Produce a CSV table with the active/next rotation slots for each contact
memoria roster rotate --format csv > roster-rotation.csv
```

Use `--persist` with `memoria roster rotate` when you want to apply the rotation
metadata updates that the scheduler would normally commit. Without this flag the
command emits audit records without changing the underlying configuration, which
is useful for evidence gathering during incident reviews.

## Policy audit investigations

When an audit event requires deeper investigation:

1. Use the dashboard filters to identify the relevant event IDs.
2. Fetch the full payload with `GET /governance/audits/<id>` for automation, or
   export the JSON directly from the UI.
3. Attach the exported JSON to incident tickets and, if needed, run
   `memoria policy test` with the offending payloads to reproduce the violation.
   Archive the JSON under **Telemetry & audit exports** in the
   [Post-GA incident review template](post_ga_incident_template.md) so the
   follow-up record stays complete.
4. If escalation metadata needs adjustment, update the roster as described above
   and re-run the audit search to confirm the new path.

## Telemetry streaming & exports

The CLI now surfaces the same policy telemetry used by the governance console.
Run `memoria policy telemetry` to capture a one-off snapshot of enforcement
counts, stage distributions, and per-policy duration metrics. Key options:

- `--format {json,csv}` writes either the JSON payload returned by
  `/governance/telemetry` or a CSV table per policy/action pair.
- `--follow` keeps the process running and streams snapshots whenever the
  enforcement engine records a decision. Combine with `--max-events` to emit a
  fixed number of updates.
- `--webhook <url>` posts each snapshot to an HTTP endpoint. Pair with
  `--format json --follow` to build lightweight alerts or collectors.
- `--output <path>` persists snapshots to disk instead of stdout, making it easy
  to append telemetry to incident timelines.

For long-running scrapes, prefer the NDJSON endpoint `GET
/governance/telemetry/export` which streams the same snapshots the CLI prints.
This keeps dashboards and automations aligned with the CLI output without
scraping the UI.

## Summary checklist

- [ ] Validate policy changes with `memoria policy lint`.
- [ ] Simulate enforcement using either the UI dry-run modal or `memoria policy test`.
- [ ] Persist approved rules via `memoria policy apply`.
- [ ] Synchronise escalation contacts by editing `policy.escalation_contacts` or
      calling the new governance endpoints.
- [ ] Capture telemetry baselines with `memoria policy telemetry` or subscribe to
      the NDJSON export for continuous monitoring.
- [ ] Export audit events for post-incident review and attach them to follow-up tickets.
