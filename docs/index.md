# Memoria documentation

Welcome to the documentation for the Memoria 0.9 alpha release line. These pages cover the spatial memory model, operational guardrails, and tooling that ship with the open-source distribution maintained by Arca Futura.

## Audience & scope

Memoria is built for teams who need a transparent memory layer for LLM-powered systems. Whether you orchestrate autonomous agents, augment human workflows, or run evaluation suites against retrieval heuristics, the documentation is organised so you can:

- Stand up the API service, SDK, or CLI in a reproducible way.
- Understand how spatial coordinates, symbolic anchors, and promotion heuristics govern what is stored.
- Extend the platform with new providers, sync targets, or custom ingestion policies without guessing at hidden behaviour.
- Audit and debug memory decisions before they ship to production.

## Highlights

- **Spatial model.** Memories carry `(x, y, z)` coordinates (temporal offset, privacy, cognitive axis) plus optional symbolic anchors. Queries can target precise coordinates or fall back to semantic anchors when distance thresholds miss.
- **Deterministic ingestion.** Heuristic scoring promotes staged memories into long-term storage only when privacy rules are satisfied. Manual overrides remain auditable through metadata.
- **Flexible deployment.** Choose between SQLite for local development or PostgreSQL/MySQL for production. Optional sync and clustering extras activate cross-instance replication and vector indexing.
- **Composable interface.** The REST API, Python SDK, CLI, and automation scripts expose the same behaviours, making it easy to embed Memoria wherever your agents run.

## How the docs are organised

| Section | Use it to… |
| --- | --- |
| [Getting started](getting-started/) | Install the SDK, bootstrap the API stack, and run evaluation scenarios. |
| [Core concepts](core-concepts/) | Learn the spatial coordinate system, promotion heuristics, and data model. |
| [Configuration](configuration/) | Reference every environment variable and configuration flag. |
| [Integrations](integrations/) | Plug Memoria into external providers, frameworks, and observability stacks. |
| [Open source roadmap](open-source/) | Track stabilisation work, governance policies, and release cadence. |
| [Contributing](contributing.md) | Follow contribution guidelines, coding standards, and evaluation expectations. |

## Quick links

- [Installation guide](getting-started/installation.md) – covers PyPI/source installs, optional extras, and Docker bootstrap flows.
- [Quick-start scenario](getting-started/quick-start.md) – run the evaluation harness to validate ingestion changes.
- [Basic usage examples](getting-started/basic-usage.md) – manipulate memories programmatically using the SDK.
- [API reference](../openapi.json) – OpenAPI schema for the REST service.

> **Release readiness:** All installation and support links point to the public [Memoria 0.9 alpha](https://github.com/Founder-ArcaFutura/Memoria) repository. Patch planning, migration notes, and governance updates are issued as 0.9 alpha maintenance releases ship, so please surface blockers or upgrade feedback through the issue tracker.

## Release cadence

The 0.9 alpha branch is the supported public baseline. Expect:

- Frequent documentation updates that mirror new defaults or CLI switches.
- Feature toggles exposed behind configuration flags so you can opt-in safely.
- Archived migrations and upgrade notes to help legacy deployments converge on the current schema.

Join the discussion in GitHub issues and pull requests to influence stabilisation priorities and share operational feedback.

---

*Made for operators and builders who need auditable, high-signal memories for their AI systems.*
