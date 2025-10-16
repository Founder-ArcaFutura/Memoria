# Policy Console Guide

The Memoria dashboard now ships with a governance console that lets administrators
inspect namespace coverage, stage escalations, and review audit activity without
leaving the browser. This guide walks through each surface and highlights the
backing API endpoints so you can automate or extend the experience. If you need a
plain-language refresher before diving into the UI, start with the
[Governance Framework guide](../getting-started/governance-plain-language.md).

## Overview

The console introduces three primary views aligned with the admin mockups:

1. **Namespace segmentation** – summarises policy density, privacy floors, and
   linked escalation contacts for each namespace pattern.
2. **Escalation rosters** – manages on-call queues, integration metadata, and
   preview notifications across PagerDuty, Slack, and custom channels.
3. **Policy audit explorer** – filters enforcement events and renders payload
   diffs so compliance and operations teams can triage incidents quickly.

All views require an authenticated dashboard session. Each action returns useful
error messages when the Memoria API key expires or when the runtime is
misconfigured.

## Namespace segmentation

The segmentation card calls `GET /governance/namespaces` to retrieve a compact
summary of each namespace pattern. Filters for team, workspace, privacy band,
and lifecycle band are passed as query parameters, e.g.

```
GET /governance/namespaces?team=secops&privacy=strict
```

Selecting a namespace triggers `GET /governance/namespaces/<namespace>` which
returns the full policy list, privacy floors, and associated escalation contacts.
The **Test policies with sample dataset** button opens the dry-run modal which
POSTs sample payloads to `/governance/policies/dry-run`. The simulation reports
mirror the CLI output produced by `memoria policy test` but run entirely in the
browser.

### Dry-run workflow

1. Collect one or more JSON documents representing the memories or payloads you
   plan to ingest.
2. Paste the array (or single object) into the modal and submit.
3. Review violation counts per policy and drill into the sample indices that
   would trigger enforcement.
4. Export results or follow up with the CLI if you need to apply the updated
   ruleset.

## Escalation roster management

The roster view queries `GET /governance/escalations` to list the stored
`policy.escalation_contacts`. Selecting a queue reveals metadata, rotation
schedules, and integration status derived from the contact's `metadata` field.

- **Add queue** uses `POST /governance/escalations` with a payload validated by
  the `EscalationContact` schema.
- **Edit queue** issues `PUT /governance/escalations/<name>` so operators can
  update coverage windows, rotation assignments, or additional channels.
- **Remove queue** calls `DELETE /governance/escalations/<name>` and refreshes the
  list automatically.
- **Preview notification** triggers `POST /governance/escalations/<name>/preview`
  and returns a synthesised payload that confirms the channel, endpoint, and
  message that would be dispatched.

All roster mutations are persisted via the `ConfigManager`, so changes survive
restarts alongside other Memoria settings.

## Policy audit explorer

Compliance analysts can inspect enforcement events via the audit explorer.
`GET /governance/audits` accepts filters for `action`, `namespace` glob,
`escalation` queue, and `role` (team identifier). Results are paginated and
sorted by most recent timestamp.

Selecting a row calls `GET /governance/audits/<id>` to fetch full payload
details. The detail card surfaces the violation reasons, escalation trail, and a
JSON diff extracted from `RetentionPolicyAudit.details`. Buttons allow exporting
the event as JSON or raising a follow-up ticket directly from the UI.

## Telemetry export and automation

The governance console now surfaces a live telemetry drawer that polls
`GET /governance/telemetry` for aggregated counts, stage distributions, and the
latest policy/action durations. The endpoint returns rounded metrics alongside a
UTC timestamp (`generated_at`) and `generated_at_epoch` for downstream
aggregation. Operations teams can subscribe to the streaming variant,
`GET /governance/telemetry/export`, which emits newline-delimited JSON (NDJSON)
snapshots every time a policy decision is recorded. The export endpoint relies
on the same observer interface used by the CLI and can drive dashboards or alert
pipelines without scraping the UI.

## Backing API summary

| Endpoint | Description |
| --- | --- |
| `GET /governance/namespaces` | Summaries of namespaces with rule density, privacy and lifecycle bands |
| `GET /governance/namespaces/<namespace>` | Detailed policy data, floors, and escalation bindings |
| `POST /governance/policies/dry-run` | Simulate policy enforcement against uploaded samples |
| `GET /governance/escalations` | List configured escalation contacts |
| `POST /governance/escalations` | Create a new escalation contact |
| `PUT /governance/escalations/<name>` | Update an existing escalation contact |
| `DELETE /governance/escalations/<name>` | Remove an escalation contact |
| `POST /governance/escalations/<name>/preview` | Generate a preview notification payload |
| `GET /governance/telemetry` | Aggregate policy enforcement metrics for the console cards |
| `GET /governance/telemetry/export` | Stream NDJSON telemetry snapshots for automation |
| `GET /governance/audits` | Filterable list of retention policy audit events |
| `GET /governance/audits/<id>` | Retrieve detailed payload and violation data |

Pair these endpoints with the CLI runbook for command-line parity and with your
existing automation to keep governance state in sync across environments.
