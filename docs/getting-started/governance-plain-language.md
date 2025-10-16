# Governance Framework (Plain-Language Guide)

Memoria ships with a governance framework that acts like a safety net and a checklist for your memory data. It helps you describe the rules your deployment should follow, keeps track of who must respond when something slips, and records what actually happened so you can prove it later. Think of it as a traffic system: policies set the speed limits, escalations name the traffic wardens, and telemetry lights up the dashboard when something breaks the rules.

## Why you might use it

- **Stay compliant without guessing.** Policies make sure retention limits, privacy flags, and approval steps are enforced the same way every time.
- **Know who is on call.** Escalation rosters list the people (or teams) who get alerted when a rule is broken or an override expires.
- **See problems early.** Telemetry and audits show which rules are firing, who acknowledged them, and whether automation is keeping up.
- **Share clear evidence.** When auditors or stakeholders ask, you can export the same policy state, audit trail, and telemetry that the system uses internally.

## What happens behind the scenes

1. **You define policies.** A policy is a small JSON or YAML document that says what should happen to certain memories or actions (keep, expire, require approval, notify someone, etc.).
2. **The runtime enforces them.** Every time the system processes a memory or workflow, it checks the policies. If something violates a rule, the governance engine blocks it or marks it for review.
3. **Escalations keep humans in the loop.** Governance stores on-call queues and notification methods (email, webhook, chat). When an incident occurs, the correct contact list is pinged automatically.
4. **Telemetry and audits get recorded.** Each enforcement, override, or change request becomes an audit row. Live telemetry counters show trends so you can spot noisy rules or slow responders.
5. **Dashboards and CLI stay in sync.** The web console, CLI commands, and SDK all read from the same governance database so you can manage policies however you prefer.

## A quick story

> Atia runs Memoria for a regulated customer support team. She uses the governance console to create a "30-day retention" policy for sensitive transcripts and adds an escalation queue that pings compliance if the rule fires three times in an hour. When a transcript breaches the rule, Memoria flags it, notifies the compliance inbox, and records the event. During an audit, Atia exports the telemetry report and the policy definition to show exactly how the rule behaves and who responded.

## How to try it

- **From configuration:** Set `governance.enabled = true` in your settings file and provide the database connection so policies can be stored and enforced.
- **From the CLI:**
  ```bash
  # Load or update a policy bundle
  memoria policy apply policies/retention.json

  # Preview the impact before committing
  memoria policy dry-run samples/retention_check.json

  # Review recent enforcement telemetry
  memoria policy telemetry --limit 20
  ```
- **From the dashboard:** Log in and open the **Governance** section to browse policies, view the escalation roster, acknowledge incidents, and download evidence bundles.
- **From code:** Use the `PolicyClient` in `memoria.sdk.policies` to automate policy CRUD operations or fetch audit entries as part of your CI/CD checks.

## Where to learn more

- Step-by-step UI guidance: [Governance Console Guide](../admin/policy_console_guide.md).
- Command-line workflows: [Governance Console Runbook](../admin/policy_console_runbook.md).
- Automating from Python: [Policy Client SDK](../sdk/policy-client.md).
- Reference architecture and API routes: [Governance features overview](../open-source/features.md#governance).
