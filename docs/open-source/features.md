# Features

## Dual Memory System

### Conscious Ingest Mode
- **One-shot Working Memory**: Essential memories injected once per session
- **Background Analysis**: Automatic analysis of conversation patterns every 6 hours  
- **Essential Memory Promotion**: Key personal facts promoted for instant access
- **Human-like Memory**: Like short-term memory for immediate access to important info
- **Performance Optimized**: Minimal token usage, fast response times

### Auto Ingest Mode
- **Dynamic Context Search**: Analyzes each query for relevant memories
- **Full Database Search**: Searches entire memory database intelligently
- **Context-Aware Injection**: 3-5 most relevant memories per LLM call
- **Retrieval Agent**: AI-powered search strategy and ranking
- **Rich Context**: Higher token usage for maximum context awareness

### Combined Mode
```python
# Best of both worlds
memoria = Memoria(
    conscious_ingest=True,  # Essential working memory
    auto_ingest=True,       # Dynamic context search
    database_connect="postgresql://..."
)
```

### Three-Layer Intelligence
```mermaid
graph TD
    A[Retrieval Agent] --> B[Dynamic Search & Ranking]
    C[Conscious Agent] --> D[Essential Memory Promotion]
    E[Memory Agent] --> F[Structured Processing]
    
    B --> G[Auto-Ingest Context]
    D --> G[Conscious Context]
    F --> H[Categorized Storage]
    G --> I[Intelligent Context Injection]
```

## Memory Types & Categories

### Automatic Categorization
| Category | Description | Examples |
|----------|-------------|----------|
| **Facts** | Objective information | "I use PostgreSQL for databases" |
| **Preferences** | Personal choices | "I prefer clean, readable code" |
| **Skills** | Abilities & expertise | "Experienced with FastAPI" |
| **Context** | Project information | "Working on e-commerce platform" |
| **Rules** | Guidelines & constraints | "Always write tests first" |

### Retention Policies
- **Short-term**: Recent activities, temporary information (7 days)
- **Long-term**: Important information, learned skills, preferences
- **Permanent**: Critical rules, core preferences, essential facts

## Analytics & Observability

### Memory Analytics API
- **Category distribution**: `/analytics/categories` and `/analytics/summary` expose long- and short-term memory counts per category with namespace-aware filters.
- **Retention trends**: `/analytics/retention` provides per-day counts by retention type for the last 7–365 days, ideal for spotting spikes in ingestion cadence.
- **Usage frequency**: `/analytics/usage` surfaces the most-accessed memories, total access counts, and rolling averages so you can identify “hot” knowledge.
- **JSON-first design**: All analytics endpoints return structured payloads that are easy to stream into notebooks or BI pipelines.

### Dashboard Insights
- **Interactive controls**: The dashboard’s Analytics panel lets you change the trend window (7/30/90 days), adjust the “Top N” usage table, and manually refresh without reloading the app.
- **Category tables**: Quickly review the balance of categories across long- and short-term stores to ensure the taxonomy stays healthy.
- **Retention heat**: Totals and most-recent counts highlight whether working memory promotion or pruning policies need attention.
- **Usage leaderboard**: See which facts, rules, or preferences are powering the agent most often, complete with last-accessed timestamps.

### Requirements
- **SQLAlchemy session factory**: The analytics routes expect `memoria.db_manager.SessionLocal` to be configured; self-hosted deployments using only the lightweight SQLite wrapper should enable the SQLAlchemy manager (enabled by default in `Memoria` instances).
- **Authenticated session**: The HTTP routes respect the existing API key/session guard—sign in to the dashboard or attach `X-API-Key` headers when calling the endpoints programmatically.
- **Populated data**: Metrics appear once memories exist; seed a few conversations or import a dataset to populate the charts.

## Multi-Region Deployments

Memoria's sync layer and scheduler services can be combined to run the platform in multiple regions with privacy-aware replication. See the [Multi-Region Deployment Guide](deployments/multi-region.md) and the new [Sync & Failover Playbook](architecture/sync-failover-playbook.md) for reference architectures, configuration commands, and operational drills.

## Universal Integration

### Works with ANY LLM Library via LiteLLM

#### LiteLLM (Recommended)

```python
from litellm import completion
from memoria import Memoria

memoria = Memoria(
    conscious_ingest=True,
    auto_ingest=True
)
memoria.enable()

# Automatic context injection with dual modes
response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Help me code"}]
)
```

#### OpenAI Direct

```python
import openai
from memoria import Memoria

memoria = Memoria(conscious_ingest=True)
memoria.enable()

client = openai.OpenAI()
# All conversations automatically recorded
response = client.chat.completions.create(...)
```

#### Azure OpenAI

```python
from loguru import logger

from memoria import Memoria
from memoria.core.providers import ProviderConfig

azure_provider = ProviderConfig.from_azure(
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="gpt-4o"
)

memoria = Memoria(
    provider_config=azure_provider,
    conscious_ingest=True
)
memoria.enable()
```

#### Anthropic
    
```python
import anthropic
from memoria import Memoria

memoria = Memoria(conscious_ingest=True)
memoria.enable()

client = anthropic.Anthropic()
# All conversations automatically recorded
response = client.messages.create(...)
```

#### Custom/Ollama
```python
from memoria import Memoria
from memoria.core.providers import ProviderConfig

ollama_provider = ProviderConfig.from_custom(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="llama3.2:3b"
)

memoria = Memoria(
    provider_config=ollama_provider,
    conscious_ingest=True
)
```

## Team-aware deployments

Team features let you create collaborative workspaces that share memories while
preserving namespace-level isolation. Roll-out follows three stages:

1. **Run the schema migration.** Existing databases need the `teams` table plus
   nullable `team_id` columns on memory tables. Use the CLI wrapper around the
   bundled migration script:

   ```bash
   # Default SQLite deployment
   memoria migrate run add_team_support

   # Managed Postgres/MySQL instance
   memoria migrate run add_team_support \
     --database-url postgresql://user:pass@host:5432/memoria
   ```

   The command executes `scripts/migrations/add_team_support.py`, creating
   foreign keys and indexes when required. Add `--dry-run` during change windows
   to confirm the target database before applying updates.

2. **Switch on team memory in configuration.** Set the flags in your config
   file or environment so Memoria loads team policies on startup:

   ```toml
   [memory]
   team_memory_enabled = true
   team_namespace_prefix = "team"
   team_enforce_membership = true
   ```

   Leaving `team_memory_enabled` false keeps the runtime in namespace-only mode
   and ignores any `team_id` values supplied through the API or SDK.

   <a id="team-and-workspace-modes"></a>

   #### Team and workspace modes

   Memoria exposes a unified set of options for collaborative contexts. These
   keys mirror the logic in `MemoriaSettings` (see `memoria/config/settings.py`)
   and accept either legacy booleans or the canonical lower-case strings shown
   below. Boolean `true` maps to `"optional"` and `false` maps to
   `"disabled"`.

   | Setting | Allowed values | Purpose |
   |---------|----------------|---------|
   | `memory.team_mode` | `"disabled"`, `"optional"`, `"required"` (`TeamMode`) | Controls whether the runtime expects a team to be selected. `"required"` blocks ingest/retrieval until a team is supplied, while `"optional"` keeps personal namespaces available. |
   | `memory.team_default_id` | Team slug or UUID, `null` | Automatically activates a team when the caller has access. Supplying any default implicitly enables optional team mode even when you omit `team_mode`. |
   | `memory.team_share_by_default` | `true`/`false` | Governs whether new team spaces default to sharing stored memories without callers opting in each time. Reset to `false` when team mode is disabled. |
   | `memory.workspace_mode` | Same values as `team_mode` | Alias for deployments that prefer “workspace” terminology. Mirrors all validation and coercion rules from `team_mode`. |
   | `memory.workspace_default_id` | Workspace slug or UUID, `null` | Optional default workspace that follows the same activation semantics as `team_default_id`. |

   `team_memory_enabled` stays synchronised with these fields. Any combination
   that resolves to `TeamMode.DISABLED` forces
   `team_memory_enabled=false`, clears defaults, and disables automatic sharing
   to preserve single-tenant behaviour. Conversely, setting
   `team_mode`/`workspace_mode` to `"optional"` or `"required"`, or providing a
   default identifier, automatically flips `team_memory_enabled` to `true` even
   if the legacy boolean is omitted.

   ```toml
   # Require a team to be selected before storing or retrieving memories
   [memory]
   team_mode = "required"
   team_default_id = "product"
   ```

   ```python
   from memoria.config.settings import MemoriaSettings, TeamMode

   settings = MemoriaSettings(memory={
       "team_mode": TeamMode.REQUIRED,
       "team_default_id": "product",
   })
   ```

   In both snippets Memoria activates the `product` team automatically for
   eligible callers and rejects ingestion attempts that lack an explicit team
   once the default is cleared.

   ```toml
   # Workspace terminology with automatic defaults
   [memory]
   workspace_mode = "optional"
   workspace_default_id = "west-coast"
   team_share_by_default = true
   ```

   ```python
   settings = MemoriaSettings(memory={
       "workspace_mode": TeamMode.OPTIONAL,
       "workspace_default_id": "west-coast",
       "team_share_by_default": True,
   })
   ```

   Here the server enables collaborative workspaces, shares new memories with
   the `west-coast` workspace unless a request overrides the behaviour, and keeps
   `team_memory_enabled=true` without explicitly setting the legacy flag. Users
   can still switch to other workspaces manually.

3. **Bootstrap teams programmatically.** Once enabled you can register teams,
   seed membership, and opt memories into the shared context:

   ```python
   from memoria import Memoria

   memoria = Memoria(team_memory_enabled=True)
   memoria.enable()

   memoria.register_team_space(
       "product",
       display_name="Product Guild",
       members=["ava", "samir"],
       admins=["riley"],
       share_by_default=True,
   )

   memoria.set_active_team("product")

   memoria.store_memory(
       anchor="weekly_sync",
       text="Outlined launch blockers for Q3 roadmap",
       tokens=28,
       share_with_team=True,
   )
   ```

### API and CLI updates

- `/memory` ingestion endpoints accept optional `team_id` and
  `share_with_team` fields. When teams are disabled they reply with HTTP 503 so
  automation can fall back to namespace-only flows.
- Dedicated endpoints like `POST /memory/teams`,
  `POST /memory/teams/<team_id>/members`, and
  `POST /memory/teams/<team_id>/activate` expose CRUD operations for team
  workspaces without direct database access.
- The `memoria` CLI ships a `teams` command group for local administration.
  Use `memoria teams list` to inspect configured spaces, `memoria teams create`
  to register namespaces and manage membership, and `memoria teams activate`
  / `memoria teams clear` to switch the active workspace for follow-up
  commands:

  ```bash
  memoria teams list --include-members
  memoria teams create product --member ava --member samir --activate
  memoria teams activate product
  memoria teams clear
  ```
- [`memoria export-data`](../../memoria/cli.py) / `memoria import-data` include the
  new team columns automatically. Use the standalone scripts
  [`scripts/export_memories.py`](../../scripts/export_memories.py) and
  [`scripts/import_memories.py`](../../scripts/import_memories.py) when you need
  namespace filters or remapping (for example `--namespace team:<slug>` or
  `--namespace-map team:legacy=team:platform`).
- Existing automation that only passes namespaces continues to work. When team
  support is off Memoria stores data exactly as previous releases did.
- When `team_enforce_membership=true`, HTTP clients must include a user
  identifier so the backend can verify membership before serving a team
  namespace. By default the server reads the `X-Memoria-User` header (falling
  back to `X-User-Id`). Set the `MEMORIA_USER_HEADER` environment variable and
  optionally `MEMORIA_USER_HEADER_FALLBACKS` (comma-separated) to match your
  gateway conventions. Requests that hit team-aware routes without a resolved
  user now return HTTP 400 with guidance on supplying the identity hint.

## Production Architecture

### Modular Design
```
memoria/
├── core/              # Main Memoria class, providers, memory modes
├── agents/            # AI-powered memory processing agents
├── database/          # Multi-database support with cloud options
├── integrations/      # LLM provider integrations (LiteLLM native)
├── config/            # Configuration management with Pydantic
├── utils/             # Helpers, validation, logging
└── tools/             # Memory search and function calling tools
```

### Database Support
- **SQLite**: Perfect for development and small applications
- **PostgreSQL**: Production-ready with full-text search and JSON support
- **MySQL**: Enterprise database support with modern features
- **Cloud Databases**: Neon, Supabase, and community managed hosting partners
- **Connection Pooling**: Optimized performance with connection management

### Configuration Management
```python
from memoria import ConfigManager

# Auto-load from multiple sources
config = ConfigManager()
config.auto_load()

# Loads from (in priority order):
# 1. Environment variables
# 2. memoria.json/yaml files
# 3. Default Pydantic settings

memoria = Memoria()  # Uses loaded configuration
```

### Real-time Synchronisation
- **Pluggable Backends**: Choose the lightweight in-memory bus for tests or connect to Redis for multi-instance deployments.
- **One-line Install**: Add Redis support with `pip install -e ".[sync]"` (or `pip install -r requirements-sync.txt`).
- **Structured Events**: Storage updates emit typed events (`memory.created`, `memory.updated`, `memory.deleted`, `thread.updated`) and, when enabled, include full payloads so peer nodes can upsert records in near real time.
- **Automatic Cache Refresh**: Each Memoria process listens in the background and refreshes memory/thread caches when changes arrive.
- **Configuration Driven**: Enable with `sync.enabled = true`, select a backend (`memory` or `redis`), and provide connection details via `sync.connection_url` and `sync.channel`.
- **Privacy-aware Replication**: Guard sensitive data by configuring `sync.privacy_floor` / `sync.privacy_ceiling`; only memories within the allowed Y-axis range are broadcast or applied.
- **Real-time Toggle**: Set `sync.realtime_replication = true` to switch from cache invalidations to full database replication across nodes.
- **Safe Defaults**: When disabled the synchronisation layer gracefully falls back to a no-op backend so single-node installs incur no overhead.

## Performance Features

### Dual Mode Token Optimization
```
Conscious Mode (Working Memory):
- Essential memories: 150-200 tokens
- One-shot injection per session
- Minimal overhead, maximum relevance

Auto Mode (Dynamic Search):
- Relevant context: 200-300 tokens  
- Per-query intelligent search
- Rich context, performance optimized

Traditional Context Injection:
- Full conversation history: 2000+ tokens
- No intelligence, maximum waste
```

### Efficiency Metrics
- **LiteLLM Native Integration**: No monkey-patching overhead
- **Async Background Processing**: Analysis doesn't block conversations
- **Intelligent Caching**: Smart caching of search results and promotions
- **Provider Optimization**: Efficient client management and connection reuse

### Memory Mode Comparison
| Feature | Conscious Mode | Auto Mode | Combined |
|---------|---------------|-----------|----------|
| **Token Usage** | ~150 tokens | ~250 tokens | ~300 tokens |
| **Response Time** | Fastest | Fast | Medium |
| **Context Richness** | Essential only | Query-specific | Best of both |
| **Use Case** | Quick access | Deep context | Production apps |

## Security & Reliability

### Data Protection
- **Input Sanitization**: Protection against injection attacks
- **Credential Safety**: Secure handling of API keys and secrets
- **Error Context**: Detailed logging without exposing sensitive data
- **Graceful Degradation**: Continues operation when components fail

### Governance Controls
> Need the non-technical version? Read the [plain-language governance guide](../getting-started/governance-plain-language.md).
- **Retention Policies**: Configure `memory.retention_policy_rules` to cap
  privacy ceilings, enforce namespace-specific lifecycles, or escalate records
  that breach governance rules before their importance scores are mutated.
- **Automated Auditing**: Every blocked or escalated update writes a
  `retention_policy_audits` record capturing the policy name, action (`block`,
  `escalate`, or `log`), and contextual metadata so administrators can trace
  enforcement decisions.
- **Write-time Enforcement**: The storage service evaluates policies during
  inserts and updates, preventing high-sensitivity memories from being promoted
  accidentally while still allowing escalation workflows when desired.

### Enforcement Telemetry & Automation
- **Streaming policy metrics**: The `PolicyMetricsCollector` records per-stage
  decisions, exposes observer hooks, and powers the `/governance/telemetry`
  JSON/NDJSON endpoints plus the `memoria policy telemetry` CLI for CSV/JSON
  streaming and webhook dispatches.【F:memoria/policy/enforcement.py†L210-L360】【F:memoria_server/api/governance_routes.py†L607-L656】【F:memoria/cli.py†L1193-L1942】
- **Dashboard insights**: The governance console renders live telemetry tables,
  stage totals, and refresh controls backed by the same API surface documented
  in the policy console guide.【F:memoria_server/dashboard/app.js†L4070-L4160】【F:docs/admin/policy_console_guide.md†L83-L106】
- **Roster & override schedulers**: Background jobs verify escalation coverage,
  rotate contact metadata, and expire overrides while persisting status through
  governance routes and the operations runbook.【F:memoria_server/api/scheduler.py†L336-L558】【F:memoria_server/api/governance_routes.py†L833-L1159】【F:docs/admin/policy_console_runbook.md†L40-L57】

### Production Ready
- **Connection Pooling**: Automatic database connection management
- **Resource Cleanup**: Proper cleanup of resources and connections
- **Error Handling**: Comprehensive exception handling with context
- **Monitoring**: Built-in logging and performance metrics

## Developer Experience

### Simple Setup
```python
# One line to enable dual memory modes
memoria = Memoria(
    conscious_ingest=True,
    auto_ingest=True
)
memoria.enable()

# No more repeating context!
```

### Advanced Configuration
```python
# Production configuration with provider support
from memoria.core.providers import ProviderConfig

azure_provider = ProviderConfig.from_azure(
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="gpt-4o"
)

memoria = Memoria(
    database_connect="postgresql://user:pass@localhost/memoria",
    provider_config=azure_provider,
    conscious_ingest=True,
    auto_ingest=True,
    namespace="production_app"
)
```

### Memory Tools & Function Calling
```python
from memoria import create_memory_tool

# Create memory search tool
memory_tool = create_memory_tool(memoria)

# Use with AI agents and function calling
def search_memory(query: str) -> str:
    """Search agent's memory for past conversations"""
    result = memory_tool.execute(query=query)
    return str(result) if result else "No relevant memories found"

# Function calling integration
tools = [{
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": "Search memory for relevant information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
}]

completion(model="gpt-4o", messages=[...], tools=tools)
```

### Evaluation Guardrails & Tooling
- **CI regression fences**: `scripts/ci/run_evaluation_suites.py` enforces
  precision/recall/cost thresholds, persists Markdown/JSON summaries, and the
  GitHub workflow posts the generated PR comment alongside uploaded artifacts
  for reviewers.【F:scripts/ci/run_evaluation_suites.py†L120-L420】【F:.github/workflows/evaluation.yml†L87-L145】【F:docs/configuration/evaluation-suites.md†L159-L239】
- **Suite scaffolding**: `memoria evaluation-scaffold` bootstraps new YAML
  suites with configurable queries, workspaces, and descriptions, matching the
  authoring guidance in the evaluation handbook.【F:memoria/cli.py†L1549-L1619】【F:memoria/cli.py†L667-L695】【F:memoria/cli_support/evaluation_scaffold.py†L10-L112】【F:docs/configuration/evaluation-suites.md†L100-L144】
- **Rich benchmark exports**: Benchmark utilities compute provider mix tables,
  cost curves, latency percentiles, and can write JSONL or Parquet outputs for
  downstream analysis.【F:memoria/tools/benchmark.py†L300-L423】

## Memory Analytics

### Real-time Statistics
```python
# Get comprehensive memory insights
stats = memoria.get_memory_stats()
print(f"Total conversations: {stats.get('chat_history_count', 0)}")
print(f"Short-term memories: {stats.get('short_term_count', 0)}")
print(f"Long-term memories: {stats.get('long_term_count', 0)}")

# Get essential conversations (conscious mode)
essential = memoria.get_essential_conversations()

# Trigger manual analysis
memoria.trigger_conscious_analysis()

# Background analysis runs automatically every six hours by default. Override the
# cadence through configuration when you need faster or slower refreshes:
#
# ```python
# from memoria import ConfigManager
#
# manager = ConfigManager()
# settings = manager.get_settings()
# settings.memory.conscious_analysis_interval_seconds = 900  # 15 minutes
# ```
#
# Or set `MEMORIA_MEMORY__CONSCIOUS_ANALYSIS_INTERVAL_SECONDS=900` in the
# environment / configuration file before Memoria starts.

# Search by category
skills = memoria.search_memories_by_category("skill")
preferences = memoria.search_memories_by_category("preference")
```

### Memory Mode Monitoring
```python
# Check which modes are enabled
print(f"Conscious mode: {memoria.conscious_ingest}")
print(f"Auto mode: {memoria.auto_ingest}")

# Monitor performance
config_info = memoria.memory_manager.get_config_info()
print(f"Provider: {memoria.provider_config.api_type if memoria.provider_config else 'default'}")
```

### Debug Mode
```python
# See what's happening behind the scenes
memoria = Memoria(
    conscious_ingest=True,
    auto_ingest=True,
    verbose=True  # Shows agent activity and mode switching
)
```

## Extensibility

### Custom Memory Processing
- Create specialized agents for specific domains
- Extend memory processing with custom logic
- Domain-specific categorization and entity extraction
- Custom retention policies and importance scoring

### Provider System
```python
# Extend provider support
class CustomProviderConfig(ProviderConfig):
    @classmethod
    def from_custom_service(cls, endpoint, auth):
        return cls(base_url=endpoint, api_key=auth, ...)
```

### Memory Tools Extensions
- Custom search functions for specific use cases
- Domain-specific memory tools
- Integration with AI agent frameworks
- Function calling extensions for complex workflows

### Plugin Architecture
Memoria ships with a lightweight plugin system so teams can extend the runtime
without forking the core codebase. Plugins are regular Python classes that
implement the `initialize(memoria)` hook (plus optional `on_memoria_ready` and
`shutdown` callbacks) and are registered declaratively through the central
configuration system.

```python
from memoria import Memoria


class AuditTrailPlugin:
    """Store every committed memory summary in an external system."""

    def __init__(self, destination: str) -> None:
        self.destination = destination
        self._memoria = None

    def initialize(self, memoria: Memoria) -> None:
        self._memoria = memoria
        # Full access to storage_service, memory_manager, etc.

    def on_memoria_ready(self, memoria: Memoria) -> None:
        logger.info("Audit trail plugin ready")

    def shutdown(self) -> None:
        print(f"Disconnecting from {self.destination}")


memoria = Memoria()
print([plugin for plugin in memoria.plugins])
```

Configuration uses the new `plugins` section and supports both JSON/YAML files
and environment variables:

```json
{
  "database": { "connection_string": "sqlite:///memoria.db" },
  "plugins": [
    {
      "name": "audit",
      "import_path": "my_project.audit:AuditTrailPlugin",
      "options": {"destination": "s3://memoria-audit"}
    },
    {
      "name": "analytics",
      "import_path": "company.analytics:MetricsPlugin",
      "enabled": false
    }
  ]
}
```

```bash
export MEMORIA_PLUGINS='[
  {"name": "audit", "import_path": "my_project.audit:AuditTrailPlugin"},
  {"import_path": "tools.indexing:VectorPlugin", "options": {"backend": "pinecone"}}
]'
```

At runtime you can inspect loaded extensions via `memoria.plugins` and review any
misconfigurations with `memoria.plugin_failures` to keep deployments healthy.

## Scalability

### Enterprise Features
- **Multi-tenant Support**: Separate memory spaces with namespaces
- **Horizontal Scaling**: Distributed database support and load balancing
- **Provider Flexibility**: Support for Azure, AWS, custom endpoints
- **Configuration Management**: Centralized config with environment-specific settings
- **Monitoring**: Comprehensive observability for production deployments

### Performance Optimization
- **Indexed Search**: Full-text search with proper indexing and ranking
- **Memory Compression**: Intelligent consolidation over time
- **Adaptive Analysis**: Dynamic frequency based on usage patterns
- **Connection Pooling**: Optimized database connections for high throughput
- **Provider Caching**: Smart caching for frequently accessed memories

### Cloud-Native Support
```python
# Serverless database integration
memoria = Memoria(
    database_connect="postgresql://user:pass@neon-serverless:5432/memoria",
    provider_config=azure_provider,
    conscious_ingest=True,
    auto_ingest=True
)

# Environment-based configuration
# MEMORIA_DATABASE__CONNECTION_STRING=postgresql://...
# MEMORIA_AGENTS__OPENAI_API_KEY=sk-...
config = ConfigManager()
config.auto_load()
```

## Roadmap

### Current Capabilities
- **Vector Search & Semantic Ranking**: Ship-ready embedding support powers clustering, reranking, and semantic discovery across the memory store.
- **Provider Integrations**: Native connectors for OpenAI, Azure OpenAI, Anthropic, LiteLLM, and custom/Ollama backends are production-tested.
- **Relationship Graph**: Entities and connections are automatically mapped so agents understand how people, projects, and facts relate.
- **Analytics Dashboard & API**: The observability suite provides category trends, usage frequency, and retention analysis in both the UI and JSON endpoints.
- **Sync Backends**: Pluggable in-memory and Redis transports keep multi-instance deployments consistent with privacy-aware replication controls.
- **Team Memory Workspaces**: Delivered schema migrations, configuration flags, CLI tooling, and guarded REST endpoints let squads collaborate in shared spaces while preserving namespace isolation and privacy controls.
- **Memory Migration Toolkit**: The `memoria migrate` CLI and export/import scripts ship JSON snapshots—including namespaces, clusters, and relationship graphs—for reliable environment-to-environment transfers.
- **Adaptive Context Orchestration**: A configuration-driven engine sizes retrieval windows dynamically, balancing query complexity, token budgets, privacy floors, and boosted memories from recent analytics.
- **Multi-model Workflows**: The provider registry orchestrates OpenAI, Azure, Anthropic, Gemini, and custom adapters simultaneously so task routes can target the ideal model per job.
- **Federated Deployment Playbooks**: The [Sync & Failover Playbook](architecture/sync-failover-playbook.md) walks through multi-region Redis bridges, PostgreSQL replication, observability checklists, and routine failover drills promised on the roadmap.
- **Policy-driven Governance Controls**: Declarative policy schemas, runtime enforcement hooks, CLI management commands, and audit pipelines now govern ingestion, retrieval, and sync operations with privacy ceilings, lifecycle caps, and escalation routing baked in.【F:memoria/config/settings.py†L62-L170】【F:memoria/policy/enforcement.py†L19-L213】【F:memoria/cli.py†L895-L933】【F:memoria/storage/service.py†L196-L565】
- **Governance Admin Console**: The production dashboard ships namespace segmentation, escalation roster, and audit explorer views with inline dry-run simulations, complemented by an operator runbook that mirrors every UI workflow in the `memoria policy` CLI for GitOps parity.【F:docs/admin/policy_console_guide.md†L1-L88】【F:docs/admin/policy_console_runbook.md†L1-L46】
- **Evaluation Harness**: Scenario specs, materialised datasets, and benchmarking utilities ship with the distribution so teams can rehearse workloads, score retrieval quality, track policy compliance, and export longitudinal reports before production rollouts.【F:memoria/evaluation/spec.py†L19-L178】【F:memoria/evaluation/loader.py†L1-L157】【F:memoria/tools/benchmark.py†L1-L220】【F:memoria/evaluation/specs/default_suites.yaml†L1-L128】
- **Automated Benchmark Operations**: A reusable CI workflow, dataset override flow, and `run_evaluation_suites` helper orchestrate smoke suites on pull requests, publish markdown/JSON artifacts, and document configuration levers for nightly runs or ad-hoc overrides.【F:.github/workflows/evaluation.yml†L1-L72】【F:scripts/ci/run_evaluation_suites.py†L1-L158】【F:docs/configuration/evaluation-suites.md†L1-L144】
- **Long-term Benchmark Storage**: Evaluation runs can stream structured results into Parquet files or a SQL warehouse by passing `--analytics-format` and `--analytics-dsn` to the CI helper, which reuses the benchmark serializer to hydrate durable tables for longitudinal analysis.【F:scripts/ci/run_evaluation_suites.py†L238-L330】【F:memoria/tools/benchmark.py†L299-L370】

### Upcoming Focus
- **Governance Automation & Telemetry Enhancements**: Layer scheduled roster verifications, richer enforcement dashboards, and certification checklists on top of the new console so policy teams can trace overrides, audits, and compliance evidence end-to-end.【F:docs/open-source/roadmap_task_breakdown.md†L20-L37】
- **Benchmark Authoring & Analytics Upgrades**: Expand the evaluation toolkit with suite scaffolding, troubleshooting guides, and deeper metrics (cost curves, latency, provider mix) while aligning reports with governance telemetry for regression tracking.【F:docs/open-source/roadmap_task_breakdown.md†L34-L42】

> 💡 Want to help? Review the [contribution guidelines](../contributing.md) and open an issue or pull request for any roadmap item that excites you.
