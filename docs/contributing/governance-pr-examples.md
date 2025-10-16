# Sample PR Descriptions for Governance and Evaluation Changes

Use the following templates as references when preparing pull requests that modify governance policies, compliance automation, or evaluation suites. Each example demonstrates the evidence reviewers expect to see in the new checklist sections.

## Governance Policy Update

```
## Summary
- Align incident response escalation matrix with the 2025 compliance framework.
- Replace manual reviewer ping with automated pager integration.

## Testing
- `python scripts/policy/pager_dry_run.py policies/pager-escalations.yaml --samples fixtures/pager-scenarios.ndjson --output artifacts/pager-dry-run-2025-02-04.md`.
- Verified Slack webhooks on staging workspace.

## Evidence Checklist
- [x] Governance policy dry-run evidence attached  
  - Attached `artifacts/pager-dry-run-2025-02-04.md` summarizing simulated escalations and reviewer acknowledgements.  
  - Linked to the workflow run: https://github.com/ORG/REPO/actions/runs/123456789.
- [x] Evaluation suite results linked or attached (required for evaluation or benchmark updates)  
  - Not applicable — no evaluation changes.  
- [x] Automation or tooling updates documented  
  - Added rollout doc in `docs/admin/pager-integration.md`.

## Additional Notes
- Rollout scheduled for the next compliance window (Mar 2025).
- Security approved change request SEC-2025-117.
```

## Evaluation Benchmark Adjustment

```
## Summary
- Refresh retrieval benchmark with 200 new governance scenarios covering multilingual policies.
- Update ranking thresholds to account for new high-risk categories.

## Testing
- `make eval-suite` to regenerate summary artefacts.
- `pytest tests/evaluations/test_risk_thresholds.py`.

## Evidence Checklist
- [ ] Governance policy dry-run evidence attached (required for governance or policy changes)  
  - Not applicable — no policy updates.
- [x] Evaluation suite results linked or attached (required for evaluation or benchmark updates)  
  - Uploaded `artifacts/evaluation-suite-summary-2025-02-04.md`.  
  - Included diff of `reports/risk-thresholds.json` with new baselines.
- [x] Automation or tooling updates documented  
  - Workflow dispatch link: https://github.com/ORG/REPO/actions/runs/987654321.

## Additional Notes
- Coordinate with localization team before merging to ensure translation coverage.
```

## Mixed Governance and Evaluation Change

```
## Summary
- Introduce automated governance checks for new vendor onboarding.
- Expand evaluation suite with vendor risk scenarios and integrate with the compliance dashboard.

## Testing
- `python scripts/policy/governance_check.py --dry-run --vendor sample_inc --output artifacts/vendor-governance-telemetry.md`.
- `make eval-suite`.
- Reviewed dashboard sync logs in `logs/dashboard-sync.log`.

## Evidence Checklist
- [x] Governance policy dry-run evidence attached (required for governance or policy changes)  
  - Uploaded `artifacts/vendor-governance-telemetry.md` with trace IDs and reviewer acknowledgements.
- [x] Evaluation suite results linked or attached (required for evaluation or benchmark updates)  
  - Added `artifacts/vendor-eval-suite-summary.md` and included screenshot of dashboard diff.
- [x] Automation or tooling updates documented  
  - Documented new cron configuration in `docs/admin/vendor-automation.md`.

## Additional Notes
- Requires coordination with Ops — see ticket OPS-2045.
- Merge only after the `vendor-onboarding` feature flag is enabled.
```
