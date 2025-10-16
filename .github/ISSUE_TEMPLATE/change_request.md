---
name: Change Request
title: "[Change] <title>"
labels: ["governance"]
about: Propose updates to governance policies, evaluation suites, or automation.
---

## Summary

Provide a concise description of the requested change.

## Phase & Scope

- **Target release phase** (Beta Hardening / Release Candidate / GA):
- **Governance & automation impact** (affected policies, schedulers, or CLI tooling):
- **Analytics & telemetry impact** (dashboards, metrics, or exports to update):

## Acceptance Criteria

List the measurable outcomes that must be satisfied before the change can ship.

## Verification Checklist

- [ ] Governance policy dry-run results attached (required for policy updates)
- [ ] Evaluation suite artifacts linked or attached (required for evaluation changes)
- [ ] `scripts/releases/generate_release_guides.py` checklist attached for the declared phase
- [ ] Telemetry/analytics rollout notes documented or linked
- [ ] Stakeholder sign-off or review notes included

## Additional Context

Add any relevant references, dashboards, screenshots, or supporting material.
