# Secure Credentials on Self-Hosted CI Runners

Self-hosted runners unlock faster, hardware-aware test suites, but they also widen the surface area for credential exposure. Follow these practices whenever a workflow (such as the core-memory nightly suite) needs to reach out to model providers, vector stores, or any third-party service.

## Use Ephemeral Credentials

- Prefer **short-lived tokens** that are minted just-in-time through the provider's OAuth/OIDC flow.
- Where the provider supports GitHub's OpenID Connect integration, configure a trust policy so the runner exchanges the GitHub-issued identity token for a scoped credential.
- Avoid baking long-lived API keys into runner images or AMIs. Rotate any fallback credentials on a strict cadence and scope them to read-only operations.

## Store Secrets in GitHub, Not on the Runner

- Use **GitHub Actions secrets** for provider tokens. Reference them in workflows via `${{ secrets.PROVIDER_TOKEN }}` so the value is injected only at runtime.
- For runner-level overrides, use **encrypted environment files** or a **secret manager** (AWS Secrets Manager, HashiCorp Vault, etc.) that the runner fetches dynamically. Do not persist credentials on disk between jobs.
- Lock down repository and organization secrets with role-based access control so only trusted maintainers can update them.

## Restrict Runner Access

- Run self-hosted agents on an isolated network segment with strict outbound allow-lists for provider endpoints. Block all inbound traffic except what is required for the GitHub runner service.
- Configure the runner service account with the minimum IAM privileges needed to fetch secrets, pull containers, or write artifacts.
- Enable audit logging on the underlying platform (AWS CloudTrail, Azure Monitor, etc.) to track every credential fetch and workflow execution.

## Mask and Scrub Output

- Keep `ACTIONS_STEP_DEBUG` disabled in production workflows so token values are not echoed.
- Use the `::add-mask::` command for any dynamic tokens printed from scripts.
- Ensure log retention policies purge sensitive output after the compliance-required window.

## Validate and Rotate

- Add automated checks that verify provider keys remain valid and expire soon enough to limit blast radius.
- Rotate credentials immediately if the runner VM/container is recycled outside of the expected automation path or if compromise is suspected.
- Document runbooks for revoking provider access, including where to invalidate refresh tokens and how to audit affected workflows.

Adhering to these controls keeps Memoria's integration tests fast while ensuring external provider credentials remain protected even when workflows execute on infrastructure we manage.
