# Admin Policy Console Mockups

The following low-fidelity mockups outline the proposed admin experience for managing Memoria governance policies. Each view focuses on transparency for namespace segmentation, escalation oversight, and compliance auditing.

> **Updated:** The mockups are now implemented in the dashboard. See
> [`policy_console_guide.md`](policy_console_guide.md) for usage details and
> [`policy_console_runbook.md`](policy_console_runbook.md) for CLI parity.

## 1. Namespace Segmentation Overview

```
+--------------------------------------------------------------+
| Namespace Segmentation                                      |
+----------------------+---------------+-------------+--------+
| Namespace            | Active Rules  | Escalations | Owners |
+----------------------+---------------+-------------+--------+
| default              | ████████ 8    | —           | Core   |
| research/*           | ████ 4        | SOC Queue   | R&D    |
| partners/*           | █████ 5       | Compliance  | Alli.  |
| ops/critical         | ██ 2          | SecOps      | Ops    |
| personal/*           | █ 1           | —           | HR     |
+----------------------+---------------+-------------+--------+
| Filters: [Team] [Workspace] [Lifecycle] [Privacy]            |
| Legend: █ Rule density (darker = more rules)                 |
+--------------------------------------------------------------+
| Namespace drill-down (on select)                            |
| ----------------------------------------------------------  |
| • Active policies list with quick toggle                    |
| • Privacy ceilings visualised as stacked thresholds         |
| • Button: “Test policies with sample dataset”               |
+--------------------------------------------------------------+
```

### Interaction Notes
- Hovering a namespace reveals the rule names and enforcement stages.
- The “Test policies” action opens the CLI-equivalent dry run modal, allowing admins to upload sample payloads and preview triggers before applying changes.

## 2. Escalation Roster Management

```
+--------------------------------------------------------------+
| Escalation Rosters                                          |
+-------------------+-------------------+---------------------+
| Queue             | Coverage          | On-Call Rotation    |
+-------------------+-------------------+---------------------+
| SecOps (primary)  | Week 42 · 24/7    | Alice → Ben → Priya |
| Compliance Review | Week 42 · 9–5     | Dana → Lee          |
| Trust & Safety    | Week 42 · 24/5    | Omar → Quinn → Rui  |
+-------------------+-------------------+---------------------+
| [Add Queue] [Sync with PagerDuty] [Download CSV]            |
+--------------------------------------------------------------+
| Selected Queue: SecOps                                      |
| ----------------------------------------------------------- |
| • Bound namespaces: ops/*, research/red, partners/high      |
| • Trigger sources: privacy_ceiling, lifecycle breach        |
| • Escalation channels: Slack #sec-urgent, PagerDuty service |
| • Upcoming coverage                                         |
|    ┌──────────────┬───────────┬────────────┐                |
|    | Date         | Primary   | Secondary  |                |
|    +──────────────+───────────+────────────+                |
|    | Thu 17 Oct   | Alice     | Ben        |                |
|    | Fri 18 Oct   | Ben       | Priya      |                |
|    | Sat 19 Oct   | Priya     | Alice      |                |
|    └──────────────┴───────────┴────────────┘                |
+--------------------------------------------------------------+
```

### Interaction Notes
- Rosters can be synced with external incident tooling; sync status is displayed per queue.
- Drag-and-drop reordering adjusts escalation precedence.
- A “Preview notification” button lets admins send a test alert to confirm integrations.

## 3. Policy Audit Log Explorer

```
+----------------------------------------------------------------+
| Policy Audit Explorer                                          |
+------------+------------+------------+--------------+---------+
| Timestamp  | Namespace  | Policy     | Action       | Outcome |
+------------+------------+------------+--------------+---------+
| 2025-10-17 | ops/alpha  | ops-block  | block        | ✅       |
| 2025-10-17 | research/1 | privacy-7  | escalate     | ↗ Sent  |
| 2025-10-16 | partners/x | gdpr-hold  | log          | 🛈 Note |
| 2025-10-16 | default    | baseline   | redact       | ✂️ Done |
+------------+------------+------------+--------------+---------+
| Filters: [Action] [Namespace glob] [Escalation queue] [Role]   |
| Saved views: {SOC Daily} {Compliance Weekly}                  |
+----------------------------------------------------------------+
| Detail Panel (right side)                                     |
| ------------------------------------------------------------  |
| • Full payload diff (pre/post redaction)                      |
| • Privacy / importance before & after                         |
| • Escalation trail with acknowledgement timestamps            |
| • Buttons: “Export JSON”, “Raise follow-up ticket”            |
+----------------------------------------------------------------+
```

### Interaction Notes
- Audit rows support quick filters and export to CSV/JSON for compliance reviews.
- Selecting an escalate event exposes the roster contact timeline and allows inline acknowledgement.
- The view shares components with the namespace dashboard, reinforcing mental models between segmentation and enforcement outcomes.

---

These mockups are intentionally schematic so teams can iterate on styling while preserving the information architecture required for effective policy governance.

