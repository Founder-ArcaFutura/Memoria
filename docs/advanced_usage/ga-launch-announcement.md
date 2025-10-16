# Memoria GA Launch Announcement

This bulletin aligns self-managed operators, hosted service teams, and community maintainers on the final steps for reaching General Availability. Share the announcement alongside the release artifact so every deployment receives the same operational guidance.

## Pre-flight Checklist

Before starting the upgrade, confirm the following have been completed:

- **Support coverage locked in** – on-call schedules and escalation paths are confirmed for the upgrade window.
- **Release assets staged** – container images, signed artifacts, and evidence packs are uploaded to the distribution registry.
- **Communication templates prepared** – customer-facing messaging, internal change approvals, and incident response notes reference this announcement.
- **Rollback snapshot targets validated** – verify storage capacity for pre-upgrade exports in both hosted and self-managed environments.

## Upgrade Steps

### Hosted control plane (Arca Futura managed environments)
1. **Schedule the maintenance window** – confirm the regional window in Statuspage and notify customers via the governance mailing list.
2. **Apply the GA container images** – update Helm values or Terraform modules to reference the `ga` release tag (`registry.arca.run/memoria/server:<ga-version>` and `registry.arca.run/memoria/dashboard:<ga-version>`).
3. **Run the post-deploy verifications**
   - `python -m memoria.cli migrate list --database-url $CONTROL_PLANE_DB`
   - `python -m memoria.cli migrate run --database-url $CONTROL_PLANE_DB --all`
   - Trigger the `Run release command checks` GitHub workflow to validate export/import parity.
4. **Re-enable ingestion workers** – remove any maintenance mode flags and confirm queue depth normalises within 10 minutes.
5. **Publish the hosted changelog** – update the managed environments status page with GA feature highlights and migration references.

### Self-managed operators
1. **Back up the deployment database**
   ```bash
   python -m memoria.cli export-data backups/pre-ga.json --database-url sqlite:///memoria.db
   ```
2. **Upgrade application services**
   - Update `requirements.txt`/`pyproject.toml` to reference the GA tag.
   - Rebuild containers or restart services after pulling the GA release artifact.
3. **Run schema migrations**
   ```bash
   python -m memoria.cli migrate run --database-url sqlite:///memoria.db --all
   ```
4. **Refresh configuration** – adopt new GA defaults from `config.example.yaml` and reapply custom overrides.
5. **Validate ingestion** – execute a sample `memoria bootstrap` command and confirm telemetry dashboards receive events.

## Rollback Commands

Use the pre-upgrade export to revert either hosted or self-managed environments.

```bash
# Restore database state
python -m memoria.cli import-data backups/pre-ga.json --database-url <database-url>

# Re-apply previous migrations if required
python -m memoria.cli migrate run <migration_name> --database-url <database-url>

# Capture evidence after rollback
python -m memoria.cli export-data backups/post-rollback.json --database-url <database-url>
```

For containerised deployments, roll back Helm/Terraform modules to the previous image tag (`<version-prior-to-ga>`) and restart services. Capture audit notes in the governance runbook before closing the incident.

## Key Links and Artifacts

- [Phased release guide](phased-guides.md) – validates GA prerequisites and automation checkpoints.
- [Compliance evidence packs](../../compliance/) – bundle SOX/GDPR artifacts with the GA release package.
- [Release command CI workflow](../../../scripts/ci/run_release_command_checks.py) – automated verification of export/import and migration parity.
- [Configuration baseline](../../../config.example.yaml) – reference defaults when preparing the upgrade announcement.
- [Support escalation playbooks](../../governance/) – ensure the GA window has on-call coverage and contact paths.

## Distribution Checklist

- Attach this announcement to the GA GitHub release and any mirrored artifact repositories.
- Include the Markdown file when running `scripts/releases/generate_release_guides.py --list-communications` so automation bundles it with other governance materials.
- For hosted environments, reference the document in maintenance-mode Statuspage updates and the governance mailing list archive.
- For self-managed operators, copy the Markdown into the `/docs/releases/` directory (or equivalent) within their deployment repository.

Share feedback or incident reports in the `#memoria-release` channel so they can be incorporated into follow-up patches.
