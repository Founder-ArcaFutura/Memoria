# Command-Line Interface (CLI)

Memoria includes a powerful command-line interface (`memoria`) to help you manage your environment, database, and agents.

## Command Overview
## Automated upgrades and capability checks

Every command that touches the configured database now runs the bundled
structural migrations (the scripts in `scripts/migrations/`) automatically when
`database.migration_auto` is enabled. The CLI prints a concise summary after
`bootstrap`, `init-db`, or when loading a `Memoria` instance for other
subcommands. The dashboard mirrors the latest status in the System Settings
card so operators can confirm that upgrades completed after deployment.

The CLI also inspects optional provider capabilities at startup—OpenAI (core),
Gemini, and sync backends. When the corresponding Python extras are missing,
you will see guidance on the command line and within the dashboard for how to
resolve the dependency (for example `pip install -e ".[integrations]"` for
Gemini support, or `pip install -e ".[sync]"` for Redis/PostgreSQL sync).

## migrate

Inspect and execute the bundled schema migration scripts without invoking them
through `python scripts/...` directly.

### List available migrations

```bash
memoria migrate list [--include-archived]
```

* Prints each migration discovered under `scripts/migrations` with the docstring
  summary.
* Pass `--include-archived` to surface the legacy helpers stored beneath
  `scripts/migrations/archive/`.

### Run a migration

```bash
memoria migrate run NAME [--database-url URL] [--dry-run] [--include-archived]
```

* `NAME` – Migration filename or stem (e.g. `add_team_support`). The command
  wraps the equivalent `python scripts/migrations/add_team_support.py` call.
* `--database-url` – Optional SQLAlchemy URL override. When omitted the command
  uses the connection configured in `memoria.json` / `.env`.
* `--dry-run` – Print the resolved script path and target database without
  executing it. Useful when validating backups or change windows.
* `--include-archived` – Allow selecting migrations from the archive folder.

All migrations execute in-process, so take a backup before running destructive
updates. Combine `--dry-run` with your normal backup workflow to double-check
the target database URL.

## assign-task-model
## Command overview

| Command | Summary |
| --- | --- |
| [`memoria bootstrap`](#bootstrap) | Run the interactive setup wizard to create `.env` and configuration files. |
| [`memoria runserver`](#runserver) | Start the bundled Flask development API server for local testing. |
| [`memoria init-db`](#init-db) | Create or upgrade the configured Memoria database schema. |
| [`memoria migrate`](#migrate) | Inspect and run schema migration scripts. |
| [`memoria build-clusters`](#build-clusters) | Generate heuristic or vector retrieval clusters to accelerate search. |
| [`memoria export-data`](#export-data) | Dump selected tables to JSON or NDJSON for backups or migrations. |
| [`memoria import-data`](#import-data) | Load exported datasets into the configured database. |
| [`memoria benchmark`](#benchmark) | Execute scripted evaluation scenarios against your deployment. |
| [`memoria evaluation-scaffold`](#evaluation-scaffold) | Generate a starter evaluation suite YAML file. |
| [`memoria assign-task-model`](#assign-task-model) | Override LLM provider/model routing for specific logical tasks. |
| [`memoria teams`](#teams) | Inspect and manage collaborative team spaces. |
| [`memoria workspaces`](#workspaces) | Workspace-focused aliases for team management workflows. |
| [`memoria policy`](#policy) | Validate, simulate, and apply retention policies. |

---

## `bootstrap`

Run an interactive wizard that bootstraps a Memoria deployment by writing an `.env` file, generating a `memoria.json` configuration, and validating provider credentials.

### Arguments
- `--config <path>`: Path to the configuration file to write (default: `memoria.json`).
- `--env-file <path>`: Path to the environment file to write (default: `.env`).
- `--force`: Overwrite existing files without prompting.

### Example
```bash
memoria bootstrap --force
```

---
After the wizard finishes the CLI executes the structural migrations for the
configured database and highlights any optional provider extras that are
missing. Accept the suggested `pip install ...` command or adjust your
configuration to disable the capability.

## `runserver`

Launch the bundled Flask development server for local API testing.

### Arguments
- `--host <ip>`: Hostname or IP address to bind (default: `0.0.0.0`).
- `--port <port>`: Port to listen on (default: `8080`).
- `--reload`: Enable Flask's debug reloader for hot-reloading during development.

### Example
```bash
memoria runserver --port 5000 --reload
```

---

## `init-db`

Initialize or upgrade the configured database schema. This prepares the spatial tables and metadata used by Memoria.

### Example
```bash
memoria init-db
```

---
On completion the command reports the structural migration summary and optional
dependency health so you can verify that the runtime matches your expectations.

## `build-clusters`

## `migrate`

Inspect and run bundled schema migration scripts.

### `memoria migrate list`
List available migration scripts.

- `--include-archived`: Include legacy migrations from the `archive/` folder.

### `memoria migrate run <name>`
Execute a migration script.

- `<name>`: The name of the migration to run (e.g., `add_team_support`).
- `--database-url <url>`: Override the database URL from your configuration.
- `--dry-run`: Preview the migration without executing it.
- `--include-archived`: Allow running migrations from the archive folder.

### Example
```bash
memoria migrate run add_cluster_token_columns --dry-run
```

---

## `build-clusters`

Generate memory clusters to accelerate retrieval.

### Arguments
- `--mode <mode>`: Clustering strategy (`heuristic` or `vector`). Default: `heuristic`.
- `--database-url <url>`: Database URL override for heuristic clustering.
- `--source <source>`: Limit clustering to a specific source (can be repeated).
- `--verbose`: Enable verbose logging for heuristic clustering.

### Example
```bash
memoria build-clusters --mode vector --source my_data
```

---

## `export-data`

Export Memoria tables to JSON or NDJSON.

### Arguments
- `<destination>`: Output file path or `-` for stdout.
- `--database-url <url>`: Override the configured database connection.
- `--format <format>`: Output format (`json` or `ndjson`). Default: `json`.
- `--table <name>`: Limit the export to a specific table (can be repeated).

### Example
```bash
memoria export-data backup.json
```

---

## `import-data`

Load a previously exported dataset into the database.

### Arguments
- `<source>`: Path to the dataset file or `-` for stdin.
- `--database-url <url>`: Override the configured database connection.
- `--format <format>`: Input format (`json` or `ndjson`).
- `--table <name>`: Restrict import to a specific table (can be repeated).
- `--no-truncate`: Append data without clearing existing rows first.

### Example
```bash
memoria import-data backup.json --no-truncate
```

---

## `benchmark`

Run scripted evaluation suites from a configuration file.

### Arguments
- `<config>`: Path to the benchmark YAML/JSON file.
- `--namespace <ns>`: Base namespace for temporary benchmark data.
- `--team-id <id>`: Optional team context for the run.
- `--output <path>`: File to write results to instead of stdout.
- `--format <format>`: Output format for results (`json` or `ndjson`).

---

## `evaluation-scaffold`

Generate a starter YAML file for an evaluation suite.

### Arguments
- `<output>`: Destination path for the generated YAML file.
- `--suite-id <id>`: A unique identifier for the suite.
- `--force`: Overwrite the destination file if it exists.

---

## `assign-task-model`

Override the LLM provider and model for a specific task (e.g., ingestion, search).

### Arguments
- `<task>`: The logical task name (e.g., `memory_ingest`).
- `--provider <name>`: The provider name to use for the task.
- `--model <model>`: Optional model override.
- `--fallback <provider>`: Fallback provider (can be repeated).
- `--clear-fallback`: Remove existing fallbacks.

---

## `teams`

Inspect and manage collaborative team spaces.

### Subcommands
- `list`: List configured teams.
- `create <team_id>`: Create a new team space.
- `activate <team_id>`: Set a team as active for the current session.
- `clear`: Clear the currently active team.

---

## `workspaces`

Manage shared workspaces (an alias for team spaces).

### Subcommands
- `list`: List configured workspaces.
- `create <workspace_id>`: Create a new workspace.
- `switch <workspace_id>`: Activate a workspace for the current session.

---

## `policy`

Validate, simulate, and apply retention policies.

### Subcommands
- `lint <source>`: Validate a policy file.
- `test <source>`: Simulate policies against sample data.
- `apply <source>`: Persist policy rules to the configuration.
- `telemetry`: Export retention policy decision telemetry for audit trails or live monitoring.

### `memoria policy telemetry`

Capture decision telemetry emitted by the policy engine so you can archive a
snapshot or stream events into an external system. By default the command reads
from the configured database and prints JSON to stdout.

#### Options

- `--format <format>`: Output encoding for the export. Supports `json`,
  `ndjson`, and `table` (pretty console table). Defaults to `json`.
- `--output <path>`: File path to write the export to. Use `-` (default) to
  stream to stdout.
- `--webhook <url>`: Deliver events to an HTTPS endpoint instead of (or in
  addition to) writing to disk. Each event is POSTed as JSON.
- `--follow`: Keep the command running and stream new telemetry as policies are
  evaluated. Without this flag the command exits after emitting the current
  snapshot.
- `--max-events <count>`: Limit the total number of events to emit during the
  session. Useful for sampling when following a high-volume deployment.
- `--namespace <name>` / `--workspace <name>`: Restrict the export to a specific
  namespace or workspace (optional but handy for multi-tenant installs).
- `--since <timestamp>`: Only emit events created after the provided ISO 8601
  timestamp. Combine with `--follow` to resume from a known checkpoint.

#### Examples

One-off snapshot written to disk:

```bash
memoria policy telemetry --format ndjson --output policy-telemetry.ndjson
```

Continuous stream to a webhook while tailing the console:

```bash
memoria policy telemetry --follow --webhook https://ops.example.com/hooks/policy --format json
```
