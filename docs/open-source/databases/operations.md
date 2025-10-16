# Database Operations

Memoria bundles a set of pragmatic workflows for moving data between database
backends. The goal is to make it easy to promote a local SQLite instance to a
managed PostgreSQL or MySQL deployment without bespoke ETL scripts.

## Exporting data

Use the CLI to serialise the primary tables (`chat_history`, `short_term_memory`,
`long_term_memory`, `spatial_metadata`, and auxiliary metadata such as clusters
and thread links) into a portable dataset:

```bash
# emit indented JSON
memoria export-data backups/memoria-backup.json --database-url sqlite:////data/memoria.db

# emit newline-delimited JSON for streaming pipelines
memoria export-data backups/memoria-backup.ndjson --format ndjson
```

Pass `--table` multiple times to limit the export to a subset of tables. When
`destination` is set to `-`, the CLI streams the payload to STDOUT so you can
pipe directly into `gzip` or upload tools.

## Importing data

Imports default to destructive mode: the helper truncates the destination
tables before inserting rows to avoid stale duplicates. Override this behaviour
with `--no-truncate` when you have carefully curated namespaces that should be
merged manually:

```bash
memoria import-data backups/memoria-backup.ndjson \
  --database-url postgresql://memoria:secret@localhost/memoria \
  --format ndjson

# append without truncating (ensure primary keys will not collide!)
memoria import-data - --database-url mysql://user:pass@localhost/memoria \
  --format ndjson --no-truncate < backups/memoria-backup.ndjson
```

For zero-downtime rotations you can pre-warm a new database by importing into a
staging URL while the original instance continues to serve traffic. Once the
import completes, update the runtime configuration and call
`POST /admin/settings/refresh` (or restart the API) to swap connections.

## Running schema migrations

Schema changes continue to ship as standalone Python scripts under
`scripts/migrations/`. The CLI exposes them through a unified command so you can
discover descriptions, preview targets, and run the scripts without leaving the
Memoria toolchain:

```bash
# inspect available migrations (add --include-archived for legacy helpers)
memoria migrate list

# run a migration against the configured database
memoria migrate run add_cluster_member_counts

# target a remote database and verify during change windows
memoria migrate run add_team_support \
  --database-url postgresql://user:pass@host:5432/memoria \
  --dry-run
```

Each invocation executes the corresponding file in `scripts/migrations/` and
sets `DATABASE_URL` for you. Take a backup before destructive migrations and
use `--dry-run` to confirm the resolved script path plus target connection.

## Admin API endpoints

Operators who prefer HTTP workflows can hit the admin endpoints directly. Both
routes require the standard `X-API-Key` header when the dashboard security key
is configured.

```bash
# download a JSON payload
curl -X POST https://memoria.example.com/admin/migrations/export \
  -H "X-API-Key: $MEMORIA_API_KEY" \
  -o memoria-backup.json

# restore into the active database
curl -X POST https://memoria.example.com/admin/migrations/import \
  -H "X-API-Key: $MEMORIA_API_KEY" \
  -F "file=@memoria-backup.json" \
  -F "format=json"
```

For JSON requests you can embed the payload directly instead of uploading a
file:

```bash
curl -X POST https://memoria.example.com/admin/migrations/import \
  -H "X-API-Key: $MEMORIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "format": "json",
        "truncate": true,
        "data": {"metadata": {...}, "data": {...}}
      }'
```

The endpoints respond with metadata describing the import/export operation,
including timestamps and row counts per table. These helpers wrap the same
logic that powers the CLI, ensuring consistent behaviour across automation and
human-operated workflows.

## Backups and recovery objectives

Memoria targets an aggressive recovery point objective (RPO) of "the last
successful operation" for maintenance tasks that run behind a
`backup_guard`. The guard takes a physical backup before yielding control and
compares table row counts when the operation completes. If the database size or
table counts change unexpectedly, the guard restores from the freshly created
backup so that no writes performed inside the guarded block escape into the
live system.

```python
from memoria.database.backup_guard import backup_guard

with backup_guard(settings.database.connection_string):
    run_maintenance_job()
```

By default, any mismatch triggers an immediate restore. Workflows that mutate
counts intentionally (for example, rebuilding vector clusters) can opt out in
two ways:

* Provide `ignore_tables` with the names of tables whose counts are expected to
  diverge. The guard will ignore those tables when deciding whether to restore.
* Disable the automatic restore logic entirely by passing
  `auto_restore_on_mismatch=False`. This is useful for pipelines that expect to
  replace entire table populations and have their own verification steps.

```python
with backup_guard(
    connection_string,
    ignore_tables={"clusters", "cluster_members"},
    auto_restore_on_mismatch=False,
):
    rebuild_cluster_indexes()
```

Remember that disabling the automatic restore widens the RPO to whatever
external backups you keep. When in doubt, prefer the defaults and only opt out
after documenting why the workflow is safe without the automatic rollback.
