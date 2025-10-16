# Single Configuration File

Memoria ships with a single configuration object, `MemoriaSettings`, that knows how to
load, validate, and persist every tunable option.  This page walks through building one
configuration document that covers the database, memory behaviour, providers, and UI
controls.

## Why a Single File Works

The runtime uses `ConfigManager` to collect settings from multiple sources in priority
order.  When a JSON or YAML file is present (for example, `memoria.json` or
`memoria.yaml`), the manager deserializes it into `MemoriaSettings` and keeps the resolved
instance in memory for the rest of the process.  Because `MemoriaSettings` nests the
sub-models for databases, agents, memory, logging, and integrations, one document can
safely express every option.

```python
from memoria import ConfigManager

config = ConfigManager()
config.auto_load()  # Automatically finds memoria.json / memoria.yaml / MEMORIA_CONFIG_PATH

settings = config.get_settings()  # Access the unified MemoriaSettings object
```

Environment variables that follow the `MEMORIA_*` naming convention still override the
file when present, so teams can ship a single file while keeping secrets in the runtime
environment.

## Example `memoria.json`

Create `memoria.json` next to your application entrypoint (or point the
`MEMORIA_CONFIG_PATH` environment variable at it):

```json
{
  "database": {
    "connection_string": "sqlite:///memoria.db",
    "pool_size": 8
  },
  "agents": {
    "openai_api_key": "sk-your-key",
    "anthropic_api_key": "anthropic-your-key",
    "anthropic_model": "claude-3-sonnet",
    "gemini_api_key": "your-gemini-key",
    "gemini_model": "gemini-1.5-pro",
    "default_model": "gpt-4o-mini",
    "conscious_ingest": true,
    "auto_ingest": false
  },
  "memory": {
    "namespace": "production",
    "context_injection": true,
    "sovereign_ingest": true,
    "conscious_analysis_interval_seconds": 21600
  },
  "logging": {
    "level": "INFO",
    "structured_logging": true
  },
  "integrations": {
    "litellm_enabled": true,
    "auto_enable_on_import": false
  }
}
```

Both ingestion toggles default to `false` in the base configuration; include
them only when the deployment should opt in to proactive memory analysis.

Save the file and restart the Memoria API or any scripts that instantiate `Memoria`.  The
Flask app factory calls `ConfigManager().auto_load()` on startup, so dropping the file
into the repository root is enough for both the backend defaults and the dashboard
settings view to pick it up.

## YAML Variant

Prefer YAML?  The same structure works:

```yaml
database:
  connection_string: postgresql://user:pass@db/memoria
  pool_size: 20
agents:
  openai_api_key: sk-your-key
  anthropic_api_key: anthropic-your-key
  anthropic_model: claude-3-sonnet
  gemini_api_key: your-gemini-key
  gemini_model: gemini-1.5-pro
  default_model: gpt-4o-mini
memory:
  namespace: production
  context_injection: false
  sovereign_ingest: true
  conscious_analysis_interval_seconds: 21600
logging:
  level: DEBUG
integrations:
  litellm_enabled: false
```

Rename the file to `memoria.yaml` (or set `MEMORIA_CONFIG_PATH=/path/to/settings.yaml`).
The same loader handles either format.

Set `conscious_analysis_interval_seconds` (default six hours) to control how often the
conscious agent refreshes short-term context in the background. Lower values increase
responsiveness at the cost of more frequent database reads and agent calls.

## Persisting Changes Back to Disk

When you need to write the current settings back to disk, use the helper methods on
`MemoriaSettings`:

```python
from memoria import ConfigManager

manager = ConfigManager()
manager.auto_load()
settings = manager.get_settings()
settings.to_file("./memoria.json")
```

This serializes the in-memory configuration, including any runtime overrides from the
settings API, into a single document that can be checked into version control or shipped
with deployments.
