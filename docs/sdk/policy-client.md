# PolicyClient SDK guide

The `PolicyClient` helper in `memoria.sdk.policies` provides a minimal HTTP wrapper for automating governance policies through the Memoria API. This guide walks through authenticating, listing, creating, updating, deleting, replacing, and validating policy rules from Python code.

## Prerequisites

- Memoria deployment with the governance API enabled.
- Python environment with the `memoria` package available (the client ships with the project).
- API base URL (for example `https://memoria.example.com/governance`).
- Optional: API key and elevated role header if required by your deployment.

Install the project locally so the SDK is available in your environment:

```bash
pip install -e .
```

## Authenticating and constructing the client

Instantiate the client with the API base URL and, if needed, the API key or default role that should be applied to outgoing requests.

```python
from memoria.sdk.policies import PolicyClient

client = PolicyClient(
    base_url="https://memoria.example.com",  # root URL to your Memoria API
    api_key="memoria-ops-token",             # optional API key header
    default_role="governance-admin",         # optional default role for headers
    timeout=15.0,                             # optional request timeout in seconds
)
```

The client automatically injects the `X-API-Key` and `X-Memoria-Role` headers when present. Override the role per call with the `role=` argument on individual operations if you need to assume different personas.

## Listing and fetching policies

Retrieve all configured policies or fetch a single rule by name:

```python
# List all rules
table = client.list()
for policy in table:
    print(policy["name"], policy.get("description"))

# Fetch a specific rule by name
data = client.get("default_retention")
print(data)
```

The `list()` method returns a sequence of mappings that mirror the payload from `GET /policy/rules`, while `get(name)` returns the nested `policy` document from `GET /policy/rules/<name>`.

## Creating policies

Submit a new rule either as a dictionary or as a `RetentionPolicyRuleSettings` instance. The helper serializes enum actions automatically.

```python
rule_definition = {
    "name": "support-inbox-retention",
    "namespace": "support",
    "filters": {"tags": ["support"]},
    "action": "delete",
    "after_days": 30,
}

created = client.create(rule_definition)
print("Created:", created["name"])
```

If the API rejects the payload, a `PolicyClientError` is raised with the HTTP status and response payload for troubleshooting.

## Updating policies

Update an existing rule by passing the new payload to `update(name, rule)`:

```python
updated = client.update(
    "support-inbox-retention",
    {"after_days": 45, "action": "archive"},
)
print("Updated action:", updated["action"])
```

Provide only the keys you wish to change if the API supports partial updates, or send the full rule definition for a full replacement depending on your deployment’s expectations.

## Deleting policies

Remove a rule by name:

```python
client.delete("support-inbox-retention")
```

Successful calls return the API response mapping. If the rule does not exist the client raises `PolicyClientError` with the remote status code, allowing you to handle `404` gracefully.

## Replacing all policies

Use `apply()` when you need to replace the entire policy set with a new sequence. The helper deletes existing rules and recreates the supplied definitions in order.

```python
rules = [
    {"name": "default", "action": "retain", "after_days": 0},
    {"name": "archive-legacy", "action": "archive", "after_days": 180},
]

client.apply(rules)
```

This is useful for GitOps pipelines that store policy definitions in version control and want to ensure the API matches the checked-in state.

## Validating policies locally

Before hitting the API you can validate payloads locally using the same Pydantic schema as the server. The client returns normalized `RetentionPolicyRuleSettings` objects or raises `PolicyConfigurationError`.

```python
from memoria.config.settings import RetentionPolicyRuleSettings
from memoria.policy.utils import PolicyConfigurationError

try:
    validated = client.validate_local([
        RetentionPolicyRuleSettings(
            name="default",
            namespace="global",
            action="retain",
            after_days=0,
        )
    ])
    print("Validated", len(validated), "rule(s)")
except PolicyConfigurationError as exc:
    print("Validation failed:", exc)
```

Local validation is helpful in CI pipelines to catch schema regressions before submitting requests.

## Error handling tips

- Wrap API calls in `try`/`except PolicyClientError` to capture the HTTP status (`exc.status`) and the returned payload (`exc.payload`).
- Adjust the `timeout` parameter for long-running operations or high-latency deployments.
- Use the `role` keyword argument on each method (for example `client.list(role="auditor")`) when you need per-call impersonation.

With these helpers, operators can automate the same governance workflows exposed in the Policy Console and CLI while integrating with custom tooling.
