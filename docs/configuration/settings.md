# Configuration

Comprehensive guide to configuring Memoria for your needs.

## Bootstrap wizard

The quickest path to a working `.env` and `memoria.json` is the interactive
wizard bundled with the CLI:

```bash
memoria bootstrap && docker compose up
```

During bootstrap you can enter provider credentials, pick either a local SQLite
file or the Dockerised Postgres service from `docker-compose.yml`, and validate
connectivity through `SQLAlchemyDatabaseManager`. Re-run the wizard at any time
with `memoria bootstrap --force` to rotate credentials or switch storage
backends; it will overwrite existing configuration files once you confirm the
prompt (or immediately when `--force` is present).

Heuristic clustering is **enabled by default** (`"enable_heuristic_clustering": true`) while
vector embedding clustering is **disabled by default** (`"enable_vector_clustering": false`).
`enable_vector_clustering` requires additional dependencies – install them with
`pip install -e ".[cluster]"` or `pip install -r requirements-cluster.txt` before
turning the feature on.

## Runtime settings API

Authenticated callers can inspect and adjust a safe subset of these options at runtime via
the Memoria API:

* `GET /settings` – returns a sanitized snapshot containing clustering toggles, agent
  ingestion flags, high-level memory controls, and integration switches. Sensitive
  credentials such as `agents.openai_api_key` or the database connection string are never
  included.
* `PATCH /settings` – accepts a JSON object whose keys are dot-paths. The following fields
  can be updated without a restart:
  - `enable_cluster_indexing`
  - `enable_heuristic_clustering`
  - `enable_vector_clustering`
  - `use_db_clusters`
  - `cluster_gravity_lambda`
  - `agents.conscious_ingest`
  - `memory.sovereign_ingest`
  - `memory.context_injection`
  - `integrations.litellm_enabled`
  - `integrations.openai_wrapper_enabled`
  - `integrations.anthropic_wrapper_enabled`
  - `integrations.auto_enable_on_import`
  - `integrations.callback_timeout`
  - `sync.enabled`
  - `sync.backend`
  - `sync.connection_url`

Example request toggling vector clustering and conscious ingest:

```bash
curl -X PATCH https://memoria.example.com/settings \
  -H "X-API-Key: $MEMORIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "enable_vector_clustering": true,
    "agents.conscious_ingest": true
  }'
```

To enable automatic context injection at runtime, send a boolean for the
`memory.context_injection` key:

```bash
curl -X PATCH https://memoria.example.com/settings \
  -H "X-API-Key: $MEMORIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "memory.context_injection": true
  }'
```

The response includes the applied values, the runtime diff reported by the backend helper,
and the sanitized settings payload that UI clients can cache.

Toggle these behaviors by setting `"enable_heuristic_clustering"` or
`"enable_vector_clustering"` in your configuration or exporting the environment
variables `MEMORIA_ENABLE_HEURISTIC_CLUSTERING` / `MEMORIA_ENABLE_VECTOR_CLUSTERING`.
When the `/clusters` endpoint is called without a `mode`, it will run heuristic
clustering if enabled, otherwise vector clustering when that is allowed. If both
features are disabled, the endpoint simply returns a disabled message. The
`enable_vector_clustering` flag also controls how cluster summaries are generated:
when disabled, summaries are derived heuristically from the first words and
common keywords instead of using an OpenAI model.

You can also force a clustering mode per request:

```bash
# run heuristic clustering
curl -X POST "http://localhost:8080/clusters?mode=heuristic"

# run vector clustering (requires extras and enable_vector_clustering=true)
curl -X POST "http://localhost:8080/clusters?mode=vector"
```


### Team and workspace configuration

Collaborative deployments can fine-tune team behaviour through the memory block.
The booleans exposed for earlier releases remain valid (`true` ⇒ optional,
`false` ⇒ disabled), but modern setups should prefer the explicit modes
documented in
[Team-aware deployments](../open-source/features.md#team-and-workspace-modes).
At a glance:

* `memory.team_mode` / `memory.workspace_mode` accept `"disabled"`, `"optional"`,
  or `"required"` (`TeamMode` enum). `"required"` forces a team/workspace to be
  selected before ingestion or retrieval.
* `memory.team_default_id` / `memory.workspace_default_id` set the active
  collaborative space automatically for eligible callers and implicitly enable
  team memory even when the mode flag is omitted.
* `memory.team_share_by_default` controls whether new shared spaces default to
  sharing stored memories when callers do not supply a preference; Memoria resets
  the value to `false` whenever team features are disabled.

All of these settings synchronise with `memory.team_memory_enabled`: any mode or
default that resolves to `TeamMode.DISABLED` clears defaults and switches the
legacy boolean off, while `"optional"`, `"required"`, or a provided default
automatically turn collaborative memory on.


### Coordinate audit scheduler

Memoria periodically re-validates the spatial coordinates attached to stored
memories. Operators can control the cadence through the new configuration keys:

* `memory.coordinate_audit_enabled` – Master switch for the background job.
  Enabled by default so fresh deployments retroactively correct legacy data.
* `memory.coordinate_audit_cadence` – Currently supports `"weekly"` and
  `"disabled"`. Weekly runs immediately once at startup and then every seven
  days.
* `memory.coordinate_audit_lookback_days` – When a previous run has completed,
  the scheduler only reprocesses memories created within the supplied window
  (defaults to seven days).

The job records its last watermark in the `service_metadata` table under the
`coordinate_audit_last_run` key. Inspect that value or the service logs to track
success/failure counts and to confirm the next scheduled run.


The `cluster_index_path` setting must point to a writable location. On
startup, Memoria verifies that the parent directory exists and can be
written to, creating it if necessary. If the directory cannot be created
or lacks write permissions, startup halts with an error.

Cluster weights decay over time according to the exponential
`weight = meta_score * exp(-λ * age_seconds)`.  The decay rate `λ` is
configured via `cluster_gravity_lambda` (environment variable
`MEMORIA_CLUSTER_GRAVITY_LAMBDA`) and defaults to a seven‑day time
constant.

## Cross-instance sync

Memoria can broadcast storage changes so caches on other application nodes stay
fresh. The feature is controlled by the `sync` block in `MemoriaSettings` and can be
flipped at runtime through `/settings`.

* `sync.enabled` – Turns the publisher/subscriber bridge on or off without
  restarting the service.
* `sync.backend` – Selects the backend adapter. The open-source build ships with
  the in-memory pub/sub driver (great for tests and single-process deployments),
  `RedisSyncBackend`, and `PostgresSyncBackend`. When `enabled=true` and no
  backend is specified, Memoria defaults to the in-memory transport.
* `sync.connection_url` – Connection string for external brokers (for example a
  Redis URI). `RedisSyncBackend` requires the optional sync extra
  (install with `pip install -e ".[sync]"` or `pip install -r
  requirements-sync.txt`). The configuration shape matches the
  enterprise adapters, so you can switch to other brokers later without changing
  code.

To enable Redis-backed broadcasts, install the sync extra and set either
environment variables or config fields, for example:

```bash
pip install -e ".[sync]"
# or
pip install -r requirements-sync.txt
export MEMORIA_SYNC__ENABLED=true
export MEMORIA_SYNC__BACKEND=redis
export MEMORIA_SYNC__CONNECTION_URL="redis://localhost:6379/0"
export MEMORIA_SYNC__CHANNEL="memoria-sync"
```

For PostgreSQL-backed broadcasts, install a psycopg driver (`psycopg[binary]`
or `psycopg2-binary`) and provide a DSN. The backend stores a JSONB event log in
`sync.table` by default, which you can inspect for auditing or truncate
periodically if you only rely on live notifications:

```bash
pip install -e .
pip install "psycopg[binary]"
export MEMORIA_SYNC__ENABLED=true
export MEMORIA_SYNC__BACKEND=postgres
export MEMORIA_SYNC__CONNECTION_URL="postgresql://memoria:secret@localhost:5432/memoria"
export MEMORIA_SYNC__CHANNEL="memoria_sync"
export MEMORIA_SYNC__TABLE="memoria_sync_events"
```

PostgreSQL `LISTEN/NOTIFY` delivers payloads to all subscribed Memoria
processes. If the listener connection is interrupted (for example during a
database failover), the backend automatically reconnects on the next publish or
subscription. Archived rows in `sync.table` let you replay missed events or
perform manual cache invalidations after maintenance windows.

### Real-time replication

Set `sync.realtime_replication` to `true` when you want each node to apply
incoming events directly to its local database instead of only refreshing
caches. In this mode the storage layer publishes full memory payloads on
`memory.created`, `memory.updated`, and `memory.deleted`, and remote nodes
upsert or remove the matching records.

Use the privacy bounds to control which memories are broadcast. The Y axis in
Memoria ranges from -15 (private) to +15 (public). Provide `sync.privacy_floor`
and/or `sync.privacy_ceiling` to restrict event emission to a subset of that
range:

```bash
export MEMORIA_SYNC__ENABLED=true
export MEMORIA_SYNC__BACKEND=redis
export MEMORIA_SYNC__REALTIME_REPLICATION=true
export MEMORIA_SYNC__PRIVACY_FLOOR=0   # only share public / semi-public memories
```

Leave the bounds unset to replicate every memory regardless of its privacy
score. The filters apply to both outbound and inbound events, so a node that
opts out of sharing sensitive data also ignores matching events from other
publishers.

See [Synchronisation Layer](../open-source/architecture.md#8-synchronisation-layer)
for a code sample that wires `RedisSyncBackend` into a `SyncCoordinator`.

### Operational guidance

* **Scaling** – Use the bundled in-memory backend only when all workers live in
  the same Python process (for instance, during unit tests). For multi-instance
  deployments connect a real message bus (Redis, Postgres `NOTIFY`, NATS, etc.)
  via the `sync.connection_url` setting. Each Memoria process establishes a
  lightweight listener thread that pushes targeted cache refreshes instead of
  full table reloads.
* **Failure handling** – If the backend drops events or becomes unavailable,
  Memoria continues to serve requests directly from the database. Cached snapshots
  may become stale until the next explicit read with `refresh=True`, so monitor
  the broker and fall back to manual refreshes when necessary. All publish/
  subscribe failures are logged but do not abort the underlying write.
* **Rolling restarts** – Because change propagation is event based, new nodes can
  join and subscribe to the same channel without flushing the database or
  restarting existing instances.

## Configuration Methods

### 1. Direct Configuration
```python
from memoria import Memoria

memoria = Memoria(
    database_connect="sqlite:///memoria.db",
    conscious_ingest=True,
    auto_ingest=True,
    openai_api_key="sk-..."
)
```

### 2. Configuration File
Create `memoria.json`:
```json
{
  "enable_heuristic_clustering": true,
  "enable_vector_clustering": false,
  "database": {
    "connection_string": "sqlite:///memoria_example.db",
    "database_type": "sqlite",
    "pool_size": 5,
    "echo_sql": false,
    "migration_auto": true,
    "backup_enabled": false,
    "backup_interval_hours": 24
  },
  "agents": {
    "openai_api_key": "sk-your-openai-key-here",
    "anthropic_api_key": "anthropic-your-key",
    "anthropic_model": "claude-3-haiku",
    "anthropic_base_url": "https://api.anthropic.com",
    "gemini_api_key": "your-gemini-key",
    "gemini_model": "gemini-1.5-pro",
    "default_model": "gpt-4o-mini",
    "fallback_model": "gpt-3.5-turbo",
    "conscious_ingest": true,
    "max_tokens": 2000,
    "temperature": 0.1,
    "timeout_seconds": 30,
    "retry_attempts": 3
  },
  "memory": {
    "namespace": "example_project",
    "shared_memory": false,
    "sovereign_ingest": true,
    "retention_policy": "30_days",
    "auto_cleanup": true,
    "importance_threshold": 0.3,
    "context_limit": 3,
    "context_injection": true,
    "max_short_term_memories": 1000
  },
  "logging": {
    "level": "INFO",
    "log_to_file": false,
    "structured_logging": false
  },
  "integrations": {
    "litellm_enabled": true,
    "auto_enable_on_import": false
  }
}
```

The sample above enables conscious and auto ingestion explicitly; if you omit
those keys the defaults remain `false`.

Include the Anthropic or Gemini credentials only when those providers should be
available for ingestion pipelines. Whitespace is trimmed automatically; remove
the fields entirely when rotating keys through environment variables instead.
The equivalent environment overrides are:

* `MEMORIA_AGENTS__ANTHROPIC_API_KEY`
* `MEMORIA_AGENTS__ANTHROPIC_MODEL`
* `MEMORIA_AGENTS__ANTHROPIC_BASE_URL`
* `MEMORIA_AGENTS__GEMINI_API_KEY`
* `MEMORIA_AGENTS__GEMINI_MODEL`

Vector clustering remains disabled by default. After installing the cluster
extras, change `"enable_vector_clustering"` to `true` to activate it.

Set `"sovereign_ingest": true` (or export
`MEMORIA_MEMORY__SOVEREIGN_INGEST=true`) when you want Memoria to persist chat
history without registering LiteLLM/OpenAI callbacks.

### Database exports and imports

Automated SQLite file copies remain available via the
`database.backup_enabled` and `database.backup_interval_hours` settings, but
operators often need portable exports that can be restored into PostgreSQL or
MySQL. Memoria exposes this workflow through the CLI and the admin API.

To export all supported tables (`chat_history`, `short_term_memory`,
`long_term_memory`, `spatial_metadata`, and related metadata tables) into a
structured JSON file, run:

```bash
memoria export-data backups/memoria-backup.json --database-url sqlite:////data/my_memory.db
```

Swap the `--format` flag to `ndjson` when you prefer newline-delimited JSON
for incremental processing:

```bash
memoria export-data backups/memoria-backup.ndjson --format ndjson
```

Restoring into any other database is symmetrical. Point the CLI at the target
connection string and feed it the dataset that was previously exported:

```bash
memoria import-data backups/memoria-backup.ndjson --database-url postgresql://user:pass@host/db
```

By default the importer clears the destination tables before inserting rows.
Pass `--no-truncate` if you need to append manually curated data and have
ensured primary keys will not collide. The same functionality is exposed via
`POST /admin/migrations/export` and `POST /admin/migrations/import` for
automation pipelines.

### Retention heuristics

The reinforcement/decay scheduler keeps sovereign memories fresh without manual
cleanup.  Tune its behavior through these configuration keys (environment
overrides in parentheses):

* `memory.retention_update_interval_minutes`
  (`MEMORIA_MEMORY__RETENTION_UPDATE_INTERVAL_MINUTES`) – how frequently the
  retention service runs.  The scheduler is namespace-aware, so isolated
  Memoria instances do not interfere with each other.
* `memory.retention_decay_half_life_hours`
  (`MEMORIA_MEMORY__RETENTION_DECAY_HALF_LIFE_HOURS`) – controls exponential
  decay; higher values make importance scores fade more slowly.
* `memory.retention_reinforcement_bonus`
  (`MEMORIA_MEMORY__RETENTION_REINFORCEMENT_BONUS`) – multiplier applied when
  memories are accessed frequently.  Set to zero to disable reinforcement.
* `memory.retention_privacy_shift`
  (`MEMORIA_MEMORY__RETENTION_PRIVACY_SHIFT`) – nudges the privacy (Y) axis
  toward public/private based on reinforcement energy.
* `memory.retention_importance_floor`
  (`MEMORIA_MEMORY__RETENTION_IMPORTANCE_FLOOR`) – minimum importance score a
  record can reach after decay.
* `memory.retention_policy_rules`
  (`MEMORIA_MEMORY__RETENTION_POLICY_RULES`) – optional list of governance
  policies that cap privacy ceilings, enforce per-namespace lifecycle limits, or
  trigger escalations instead of silent retention updates. Each rule accepts a
  `name`, a set of `namespaces` (exact strings or glob patterns), constraint
  fields such as `privacy_ceiling`, `importance_floor`, or `lifecycle_days`, and
  an `action` of `block`, `escalate`, or `log`.
* `cluster_gravity_lambda` (`MEMORIA_CLUSTER_GRAVITY_LAMBDA`) – optional decay
  rate applied when updating cluster centroids and weights after reinforcement
  cycles.

```json
{
  "memory": {
    "retention_policy_rules": [
      {
        "name": "executive-privacy",
        "namespaces": ["executive"],
        "privacy_ceiling": 0.0,
        "action": "block"
      },
      {
        "name": "compliance-review",
        "namespaces": ["hr", "legal*"],
        "lifecycle_days": 30,
        "action": "escalate",
        "escalate_to": "governance@corp",
        "metadata": {"priority": "high"}
      }
    ]
  }
}
```

When a retention cycle or write operation would violate a rule, Memoria records
an audit entry describing the matched policy, the attempted change, and any
escalation target. Blocked policies prevent the write entirely; escalate rules
allow the write or decay to continue but flag the event for administrators.

```python
from memoria import ConfigManager, Memoria

# Recommended approach using ConfigManager
config = ConfigManager()
config.auto_load()  # Loads memoria.json automatically

memoria = Memoria()  # Uses loaded configuration
memoria.enable()
```

### 3. Provider Configuration with Azure/Custom Endpoints
For Azure OpenAI, custom endpoints, or advanced provider configurations:

```python
from memoria import Memoria
from memoria.core.providers import ProviderConfig

# Azure OpenAI Configuration
azure_provider = ProviderConfig.from_azure(
    api_key="your-azure-openai-api-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="gpt-4o-mini",
    api_version="2024-12-01-preview",
    model="gpt-4o-mini"
)

memoria = Memoria(
    database_connect="sqlite:///azure_memory.db",
    provider_config=azure_provider,
    conscious_ingest=True,
    namespace="azure_project"
)
memoria.enable()
```

### 4. Environment Variables
```bash
export MEMORIA_AGENTS__OPENAI_API_KEY="sk-..."
export MEMORIA_DATABASE__CONNECTION_STRING="postgresql://..."
export MEMORIA_MEMORY__NAMESPACE="production"
export MEMORIA_LOGGING__LEVEL="INFO"
export MEMORIA_DATABASE__POOL_SIZE="20"
export MEMORIA_MEMORY__AUTO_CLEANUP="true"
# defaults
export MEMORIA_ENABLE_VECTOR_CLUSTERING="false"
export MEMORIA_ENABLE_HEURISTIC_CLUSTERING="true"
# enable vector clustering only after installing extras
# export MEMORIA_ENABLE_VECTOR_CLUSTERING="true"
```

```python
from memoria import ConfigManager, Memoria

config = ConfigManager()
config.load_from_env()

memoria = Memoria()
memoria.enable()
```

## Configuration Sections

### Database Settings

```python
database = {
    "connection_string": "sqlite:///memoria.db",
    "database_type": "sqlite",  # sqlite, postgresql, mysql
    "template": "basic",
    "pool_size": 10,
    "echo_sql": False,
    "migration_auto": True,
    "backup_enabled": False,
    "backup_interval_hours": 24
}
```

#### Connection Strings
```python
# SQLite (recommended for development)
"sqlite:///path/to/database.db"

# PostgreSQL (recommended for production)
"postgresql://user:password@localhost:5432/memoria"

# PostgreSQL with driver-specific schemes and legacy aliases
"postgresql+psycopg2://user:password@localhost:5432/memoria"
"postgres://user:password@localhost:5432/memoria"

# MySQL
"mysql://user:password@localhost:3306/memoria"
"mysql+pymysql://user:password@localhost:3306/memoria"

# Cloud Databases
"postgresql://user:pass@neon-host:5432/memoria"  # Neon
"postgresql://user:pass@supabase-host:5432/memoria"  # Supabase
```

Memoria accepts SQLAlchemy-style connection URLs whose base scheme starts with
`sqlite`, `postgres`, or `mysql`, so driver qualifiers such as `+psycopg2` or
`+pymysql` are supported out of the box.

#### Backups

Enable automated backups by setting `backup_enabled` and `backup_interval_hours` in
your database configuration. Automatic scheduling depends on
[`APScheduler`](https://apscheduler.readthedocs.io/en/stable/) being installed.
If the library is missing, Memoria logs a warning and skips scheduling (backups
remain disabled until APScheduler is available or `backup_enabled` is set to
`False`). To create a manual backup, use the provided script:

```bash
python scripts/backup_database.py /path/to/backup.sql
```

### Agent Settings

```python
agents = {
    "openai_api_key": "sk-...",
    "default_model": "gpt-4o-mini",  # Updated default model
    "fallback_model": "gpt-3.5-turbo", 
    "max_tokens": 2000,
    "temperature": 0.1,
    "timeout_seconds": 30,
    "retry_attempts": 3,
    "conscious_ingest": True  # Opt-in (disabled by default)
}
```

Both `conscious_ingest` and `context_injection` default to `false` so that new
deployments avoid unintentional LLM calls. Flip them on explicitly when you
need proactive short-term memory analysis or automatic context injection.

#### Supported Models
- **OpenAI**: `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`
- **Azure OpenAI**: Any deployed model name
- **LiteLLM**: Any supported model from 100+ providers
- **Custom endpoints**: Any compatible model

#### Per-task model routing

Use `agents.task_model_routes` to direct specific workflows to dedicated
providers or models. The setting is a mapping whose keys are logical task
names (`memory_ingest`, `search_planning`, or any additional task that you
register) and whose values follow the shape below:

```json
{
  "provider": "<alias>",
  "model": "<optional-model>",
  "fallback": ["<alias>", "<alias>:<model>"]
}
```

* `provider` – Required alias of the preferred provider. The default
  installation exposes the primary OpenAI-compatible client as both
  `primary` and `openai`, while optional integrations register aliases such as
  `anthropic` and `gemini` when their API keys are present.
* `model` – Optional model override passed to the provider when the task is
  routed. Omit it to use the provider’s configured default.
* `fallback` – Optional ordered list describing backup providers. Each entry
  can be supplied as a bare alias (`"openai"`) or as `provider:model` when the
  fallback needs a specific model (`"anthropic:claude-3-haiku"`). Internally
  the list is normalised from comma-separated strings, arrays, or mapping
  syntax, so both JSON and environment variable overrides stay ergonomic.

Out of the box Memoria ships with two routes:

```json
{
  "agents": {
    "task_model_routes": {
      "memory_ingest": {
        "provider": "primary",
        "fallback": ["openai"]
      },
      "search_planning": {
        "provider": "primary",
        "fallback": ["anthropic", "openai"]
      }
    }
  }
}
```

Override the defaults through configuration files, environment variables, or
the CLI:

```toml
[agents.task_model_routes.search_planning]
provider = "anthropic"
model = "claude-3-haiku"
fallback = ["primary", "openai:gpt-4o-mini"]
```

```bash
export MEMORIA_AGENTS__TASK_MODEL_ROUTES__memory_ingest__provider="primary"
export MEMORIA_AGENTS__TASK_MODEL_ROUTES__memory_ingest__fallback="anthropic,openai"

memoria assign-task-model search_planning \
  --provider anthropic \
  --model claude-3-haiku \
  --fallback primary \
  --fallback openai:gpt-4o-mini
```

The CLI command persists the same structure via
`agents.task_model_routes.<task>` so runtime updates immediately influence the
provider registry that powers ingestion and search planners.

### Memory Settings

```python
memory = {
    "namespace": "default",
    "shared_memory": False,
    "retention_policy": "30_days",  # 7_days, 30_days, 90_days, permanent
    "auto_cleanup": True,
    "cleanup_interval_hours": 24,
    "importance_threshold": 0.3,
    "max_short_term_memories": 1000,
    "max_long_term_memories": 10000,
    "context_injection": True,  # Opt-in (disabled by default)
    "context_limit": 3,
    "retention_update_interval_minutes": 30,
    "retention_decay_half_life_hours": 72.0,
    "retention_reinforcement_bonus": 0.05,
    "retention_privacy_shift": 0.5,
    "retention_importance_floor": 0.05,
    "conscious_analysis_interval_seconds": 21600,
}
```

#### Memory Features
- **Conscious Ingest**: Intelligent filtering of memory-worthy content
- **Auto Ingest**: Automatic memory recording for all conversations
- **Namespace Isolation**: Separate memory spaces for different projects
- **Retention Policies**: Automatic cleanup based on time or importance
- **Context Injection**: Relevant memories injected into conversations
- **Background Conscious Refresh**: Adjustable interval (default six hours) for the
  conscious agent to re-run analysis without manual triggers
- **Reinforcement Scheduler**: Periodic decay/boost of importance scores with configurable half-life and reinforcement bonus

### Logging Settings

```python
logging = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "<green>{time}</green> | <level>{level}</level> | {message}",
    "log_to_file": False,
    "log_file_path": "logs/memoria.log",
    "log_rotation": "10 MB",
    "log_retention": "30 days",
    "structured_logging": False
}
```

### Integration Settings

```python
integrations = {
    "litellm_enabled": True,
    "openai_wrapper_enabled": False,
    "anthropic_wrapper_enabled": False,
    "auto_enable_on_import": False,
    "callback_timeout": 5
}
```

#### Supported Integrations
- **LiteLLM**: Universal LLM wrapper (100+ providers)
- **OpenAI**: Direct OpenAI API integration
- **Anthropic**: Claude models integration
- **Azure AI Foundry**: Enterprise Azure AI platform
- **Custom**: Any OpenAI-compatible endpoint

## Configuration Manager

### Auto-Loading
```python
from memoria import ConfigManager

config = ConfigManager()
config.auto_load()  # Loads from multiple sources in priority order
```

Priority (highest to lowest):
1. Environment variables (`MEMORIA_*`)
2. `MEMORIA_CONFIG_PATH` environment variable
3. `memoria.json` in current directory
4. `memoria.yaml`/`memoria.yml` in current directory
5. `config/memoria.json`
6. `config/memoria.yaml`/`config/memoria.yml`
7. `~/.memoria/config.json`
8. `~/.memoria/config.yaml`
9. `/etc/memoria/config.json` (Linux/macOS)
10. Default settings

### Manual Configuration
```python
# Load from specific file
config.load_from_file("custom_config.json")

# Load from environment
config.load_from_env()

# Update specific setting
config.update_setting("database.pool_size", 20)

# Get setting value
db_url = config.get_setting("database.connection_string")

# Save configuration
config.save_to_file("memoria_backup.json")

# Get configuration info
info = config.get_config_info()
print(f"Sources: {', '.join(info['sources'])}")
print(f"Debug mode: {info['debug_mode']}")
print(f"Production: {info['is_production']}")
print(
    f"Environment overrides: {', '.join(info['env_overrides']) or 'none'}"
)

# Validate configuration
is_valid = config.validate_configuration()
```

## Production Configuration

## Production Configuration

### Development
```json
{
  "database": {
    "connection_string": "sqlite:///dev_memoria.db",
    "echo_sql": true
  },
  "agents": {
    "default_model": "gpt-4o-mini",
    "conscious_ingest": true
  },
  "memory": {
    "namespace": "development",
    "auto_cleanup": false
  },
  "logging": {
    "level": "DEBUG",
    "log_to_file": false
  },
  "debug": true,
  "verbose": true
}
```

### Production
```json
{
  "database": {
    "connection_string": "postgresql://user:pass@prod-db:5432/memoria",
    "pool_size": 20,
    "backup_enabled": true,
    "backup_interval_hours": 24,
    "migration_auto": true
  },
  "agents": {
    "default_model": "gpt-4o-mini",
    "retry_attempts": 5,
    "timeout_seconds": 60
  },
  "memory": {
    "namespace": "production",
    "auto_cleanup": true,
    "importance_threshold": 0.4,
    "max_short_term_memories": 5000
  },
  "logging": {
    "level": "INFO",
    "log_to_file": true,
    "log_file_path": "/var/log/memoria/memoria.log",
    "structured_logging": true
  },
  "integrations": {
    "litellm_enabled": true
  },
  "debug": false,
  "verbose": false
}
```

### Docker Environment
```dockerfile
# Basic configuration
ENV MEMORIA_DATABASE__CONNECTION_STRING="postgresql://user:pass@db:5432/memoria"
ENV MEMORIA_AGENTS__OPENAI_API_KEY="sk-..."
ENV MEMORIA_MEMORY__NAMESPACE="production"
ENV MEMORIA_LOGGING__LEVEL="INFO"
ENV MEMORIA_LOGGING__LOG_TO_FILE="true"

# Performance tuning
ENV MEMORIA_DATABASE__POOL_SIZE="50"
ENV MEMORIA_MEMORY__IMPORTANCE_THRESHOLD="0.4"
ENV MEMORIA_AGENTS__RETRY_ATTEMPTS="5"
```

### Azure Container Apps
```yaml
# Environment variables for Azure deployment
- name: MEMORIA_DATABASE__CONNECTION_STRING
  value: "postgresql://user:pass@azure-postgres:5432/memoria"
- name: MEMORIA_AGENTS__OPENAI_API_KEY
  secretRef: openai-api-key
- name: MEMORIA_MEMORY__NAMESPACE
  value: "azure-production"
- name: MEMORIA_INTEGRATIONS__LITELLM_ENABLED
  value: "true"
```

## Configuration Examples

## Configuration Examples

### Multi-Project Setup
```python
from memoria import ConfigManager, Memoria

# Project A
config_a = ConfigManager()
config_a.update_setting("memory.namespace", "project_a")
config_a.update_setting("database.connection_string", "sqlite:///project_a.db")
memoria_a = Memoria()
memoria_a.enable()

# Project B  
config_b = ConfigManager()
config_b.update_setting("memory.namespace", "project_b")
config_b.update_setting("database.connection_string", "sqlite:///project_b.db")
memoria_b = Memoria()
memoria_b.enable()
```

### Azure AI Foundry Integration
```python
from memoria import Memoria
from memoria.core.providers import ProviderConfig

# Azure provider configuration
azure_provider = ProviderConfig.from_azure(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_version="2024-12-01-preview",
    model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
)

memory_system = Memoria(
    database_connect="sqlite:///azure_ai_foundry_memory.db",
    conscious_ingest=True,
    auto_ingest=True,
    provider_config=azure_provider,
    namespace="azure_ai_foundry_example"
)
memory_system.enable()
```

### High-Performance Setup
```json
{
  "database": {
    "connection_string": "postgresql://user:pass@high-perf-db:5432/memoria",
    "pool_size": 50,
    "migration_auto": true
  },
  "memory": {
    "importance_threshold": 0.5,
    "max_short_term_memories": 5000,
    "max_long_term_memories": 50000,
    "context_limit": 5,
    "auto_cleanup": true
  },
  "agents": {
    "default_model": "gpt-4o-mini",
    "max_tokens": 4000,
    "retry_attempts": 2,
    "timeout_seconds": 60
  },
  "integrations": {
    "litellm_enabled": true,
    "callback_timeout": 10
  }
}
```

### Cloud-Native Setup (Neon + Vercel)
```json
{
  "database": {
    "connection_string": "postgresql://user:pass@neon-serverless:5432/memoria",
    "pool_size": 1,
    "migration_auto": true
  },
  "memory": {
    "namespace": "vercel-production",
    "retention_policy": "90_days",
    "auto_cleanup": true
  },
  "logging": {
    "level": "INFO",
    "structured_logging": true
  }
}
```

### Memory Tools Integration
```python
from memoria import Memoria, create_memory_tool

# Initialize Memoria with tool support
memoria = Memoria(
    database_connect="sqlite:///tool_memory.db",
    conscious_ingest=True,
    auto_ingest=True,
    namespace="tool_integration"
)
memoria.enable()

# Create memory tool for AI agents/frameworks
memory_tool = create_memory_tool(memoria)

# Use with function calling frameworks
def search_memory(query: str) -> str:
    """Search agent's memory for past conversations"""
    result = memory_tool.execute(query=query)
    return str(result) if result else "No relevant memories found"
```

Record operations use ``settings.agents.default_model`` by default. Provide a
``model`` parameter to override the configured default.

### Security-Focused Setup
```json
{
  "database": {
    "connection_string": "postgresql://user:pass@secure-db:5432/memoria?sslmode=require"
  },
  "logging": {
    "level": "WARNING",
    "structured_logging": true
  },
  "agents": {
    "timeout_seconds": 10
  }
}
```

## Environment Variables Reference

All configuration can be set via environment variables with the prefix `MEMORIA_`:

```bash
# Database Configuration
export MEMORIA_DATABASE__CONNECTION_STRING="sqlite:///memoria.db"
export MEMORIA_DATABASE__DATABASE_TYPE="sqlite"
export MEMORIA_DATABASE__POOL_SIZE="10"
export MEMORIA_DATABASE__ECHO_SQL="false"
export MEMORIA_DATABASE__MIGRATION_AUTO="true"
export MEMORIA_DATABASE__BACKUP_ENABLED="false"

# Agent Configuration
export MEMORIA_AGENTS__OPENAI_API_KEY="sk-..."
export MEMORIA_AGENTS__DEFAULT_MODEL="gpt-4o-mini"
export MEMORIA_AGENTS__FALLBACK_MODEL="gpt-3.5-turbo"
export MEMORIA_AGENTS__CONSCIOUS_INGEST="true"
export MEMORIA_AGENTS__MAX_TOKENS="2000"
export MEMORIA_AGENTS__TEMPERATURE="0.1"
export MEMORIA_AGENTS__TIMEOUT_SECONDS="30"
export MEMORIA_AGENTS__RETRY_ATTEMPTS="3"

# Memory Configuration
export MEMORIA_MEMORY__NAMESPACE="production"
export MEMORIA_MEMORY__SHARED_MEMORY="false"
export MEMORIA_MEMORY__RETENTION_POLICY="30_days"
export MEMORIA_MEMORY__AUTO_CLEANUP="true"
export MEMORIA_MEMORY__IMPORTANCE_THRESHOLD="0.3"
export MEMORIA_MEMORY__CONTEXT_LIMIT="3"
export MEMORIA_MEMORY__CONTEXT_INJECTION="true"

# Logging Configuration
export MEMORIA_LOGGING__LEVEL="INFO"
export MEMORIA_LOGGING__LOG_TO_FILE="false"
export MEMORIA_LOGGING__STRUCTURED_LOGGING="false"

# Integration Configuration
export MEMORIA_INTEGRATIONS__LITELLM_ENABLED="true"
export MEMORIA_INTEGRATIONS__AUTO_ENABLE_ON_IMPORT="false"

# Global Settings
export MEMORIA_DEBUG="false"
export MEMORIA_VERBOSE="false"
export MEMORIA_VERSION="1.0.0"
export MEMORIA_ENABLE_CLUSTER_INDEXING="true"
export MEMORIA_ENABLE_VECTOR_CLUSTERING="false"
export MEMORIA_ENABLE_HEURISTIC_CLUSTERING="true"
```

### Compatibility Environment Aliases

For legacy deployments, Memoria recognises a small set of historical environment
variables and maps them onto the structured settings shown above. These
aliases help older scripts continue to function while new projects adopt the
`MEMORIA_SECTION__FIELD` pattern:

- `MEMORIA_DB_URL` or `MEMORIA_DATABASE_URL` → `MEMORIA_DATABASE__CONNECTION_STRING`
- `MEMORIA_DEFAULT_MODEL` → `MEMORIA_AGENTS__DEFAULT_MODEL`
- `MEMORIA_DEFAULT_NAMESPACE` → `MEMORIA_MEMORY__NAMESPACE`

When these aliases are used, they show up in
`ConfigManager().get_config_info()["env_overrides"]` so operators can see that a
non-nested variable supplied the value.

## Configuration Schema

The complete configuration schema with all available options:

```json
{
  "version": "1.0.0",
  "debug": false,
  "verbose": false,
  "enable_cluster_indexing": true,
  "enable_heuristic_clustering": true,
  "enable_vector_clustering": false,
  "database": {
    "connection_string": "sqlite:///memoria.db",
    "database_type": "sqlite",
    "template": "basic",
    "pool_size": 10,
    "echo_sql": false,
    "migration_auto": true,
    "backup_enabled": false,
    "backup_interval_hours": 24
  },
  "agents": {
    "openai_api_key": null,
    "default_model": "gpt-4o-mini",
    "fallback_model": "gpt-3.5-turbo",
    "max_tokens": 2000,
    "temperature": 0.1,
    "timeout_seconds": 30,
    "retry_attempts": 3,
    "conscious_ingest": true
  },
  "memory": {
    "namespace": "default",
    "shared_memory": false,
    "retention_policy": "30_days",
    "auto_cleanup": true,
    "cleanup_interval_hours": 24,
    "importance_threshold": 0.3,
    "max_short_term_memories": 1000,
    "max_long_term_memories": 10000,
    "context_injection": true,
    "context_limit": 3
  },
  "logging": {
    "level": "INFO",
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
    "log_to_file": false,
    "log_file_path": "logs/memoria.log",
    "log_rotation": "10 MB",
    "log_retention": "30 days",
    "log_compression": "gz",
    "structured_logging": false
  },
  "integrations": {
    "litellm_enabled": true,
    "openai_wrapper_enabled": false,
    "anthropic_wrapper_enabled": false,
    "auto_enable_on_import": false,
    "callback_timeout": 5
  },
  "custom_settings": {}
}
```