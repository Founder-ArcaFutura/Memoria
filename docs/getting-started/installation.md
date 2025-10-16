# Installation

This guide walks through installing the Memoria SDK, bringing up the reference API stack, and preparing optional extras such as clustering and cross-instance sync. Follow the sections that match your environment.

## Requirements

- Python 3.10 or newer
- Git (if installing from source)
- Optional: API keys for OpenAI, Anthropic, Gemini, or any OpenAI-compatible endpoint when you want LLM-backed ingestion

## Install from source

There is no published PyPI package yet. Install Memoria directly from the repository:

```bash
git clone https://github.com/Founder-ArcaFutura/memoria_staging.git
cd memoria_staging
pip install -e .
```

### Development installation

Editable installs include tooling for linting, testing, and documentation:

```bash
git clone https://github.com/Founder-ArcaFutura/memoria_staging.git
cd memoria_staging
pip install -e ".[dev]"
```

## Optional extras

Install extras on demand so minimal deployments stay lightweight.

| Extra | Command | Enables |
| --- | --- | --- |
| Clustering | `pip install -e ".[cluster]"` | Vector clustering helpers for `/clusters` endpoints. |
| Integrations | `pip install -e ".[integrations]"` | Google Generative AI SDK and provider helpers. |
| Sync backends | `pip install -e ".[sync]"` | Redis or PostgreSQL LISTEN/NOTIFY replication. |

After installing the relevant extras, set the environment variables or configuration flags described in [`docs/configuration/settings.md`](../configuration/settings.md) to activate each feature.

## Bootstrap the runtime

1. **Generate configuration** with the interactive wizard:

   ```bash
   memoria bootstrap
   ```

   The wizard writes `.env` credentials, generates API keys, and prepares `memoria.json`. Re-run it with `--force` to rotate secrets or switch between SQLite and PostgreSQL.

2. **Start the reference stack** with Docker Compose:

   ```bash
   docker compose up
   ```

   The API listens on `http://localhost:8080`. Combine `--build` when you want to rebuild images.

3. **Use the helper script (mirrors CI)**:

   ```bash
   python scripts/run_local.py bootstrap  # or: make bootstrap
   make up                                 # build and start API + Postgres containers
   make down                               # stop services (volumes preserved)
   make reset                              # destroy and regenerate everything
   make smoke                              # CI-aligned smoke test for docker compose
   ```

   `scripts/run_local.py` appends new keys to `.env` without overwriting local changes and ensures database directories exist before Docker starts.

## Verify the installation

Confirm imports succeed after activating your virtual environment:

```bash
python - <<'PY'
from memoria import Memoria
Memoria().enable()
print("Memoria ready")
PY
```

If you installed optional providers, export the corresponding API keys first.

## Database options

### SQLite (default)

No additional setup is required. SQLite files are created automatically. The Docker stack stores them under `/workspace/sqlite/memoria.db` by default.

### PostgreSQL

Install a driver compatible with SQLAlchemy:

```bash
pip install psycopg2-binary
```

Update `DATABASE_URL` (or the equivalent setting in `memoria.json`) to point at your PostgreSQL instance before running migrations.

### MySQL

Memoria ships a connector wrapper backed by the official MySQL driver:

```bash
pip install mysql-connector-python
```

Other DB-API compatible drivers (for example `mysqlclient` or `PyMySQL`) work manually, but the packaged extras and examples assume `mysql-connector-python`.

## Enable optional features

### Clustering

Enable heuristic or vector clustering before calling `/clusters`:

```bash
export MEMORIA_ENABLE_VECTOR_CLUSTERING=true      # vector-based clustering
export MEMORIA_ENABLE_HEURISTIC_CLUSTERING=true   # heuristic clustering
```

At least one flag must be enabled for cluster endpoints to return results. Vector clustering also requires the `cluster` extra.

### Real-time sync

Redis- or PostgreSQL-backed replication lives behind the `sync` extra:

```bash
pip install -e ".[sync]"
```

Then set the flags documented in [`MemoriaSettings`](../configuration/settings.md#cross-instance-sync), for example:

```bash
export MEMORIA_SYNC__ENABLED=true
export MEMORIA_SYNC__BACKEND=redis
export MEMORIA_SYNC__CONNECTION_URL=redis://localhost:6379/0
export MEMORIA_SYNC__CHANNEL=memoria
```

## API key configuration

Choose whichever approach fits your deployment:

- **Environment variable**
  ```bash
  export OPENAI_API_KEY="sk-your-openai-key-here"
  ```
- **`.env` file**
  ```
  OPENAI_API_KEY=sk-your-openai-key-here
  ```
- **Direct configuration**
  ```python
  from memoria import Memoria

  memoria = Memoria(api_key="sk-your-openai-key-here")
  ```

The configuration manager automatically loads values from `memoria.json`, environment variables, and `.env` files.

## Upgrading from older versions

Earlier releases stored `emotional_intensity` as a dedicated column. The current version keeps this value in the `processed_data` JSON field, so no standalone column is required. Drop legacy columns via the CLI:

```bash
memoria migrate run remove_emotional_intensity --include-archived
```

If your database predates cluster token/character totals, run:

```bash
memoria migrate run add_cluster_token_columns
```

Combine migrations with `--database-url` to target remote databases or `--dry-run` when validating maintenance windows. Review `memoria migrate list --include-archived` if you need to audit available scripts before executing them.
