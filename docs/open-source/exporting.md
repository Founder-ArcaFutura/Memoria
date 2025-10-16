# Exporting and Importing Memories

Memoria bundles a complete migration workflow for long-term memories, short-term context, clusters, and relationship graph edges. Use the [`memoria export-data`](../../memoria/cli.py) / `memoria import-data` CLI commands for straightforward table dumps, and fall back to the purpose-built scripts in `scripts/` when you need namespace filters or namespace mapping.

## Exporting

### Using the CLI

```bash
memoria export-data export/snapshot.json
```

By default the snapshot contains the exported tables exactly as stored in the database:

* Long-term and short-term memories (unless you limit tables with `--table`)
* Cluster summaries and cluster members
* Relationship graph edges (`memory_relationships`)
* A namespace manifest so downstream imports can remap environments

Key options:

| Option | Description |
| ------ | ----------- |
| `--database-url <DSN>` | Override the configured connection string |
| `--format {json,ndjson}` | Select JSON or newline-delimited JSON output |
| `--table <name>` | Repeatable filter limiting the export to specific tables |
| `-` as destination | Stream the payload to stdout for piping |

Run `memoria export-data --help` for the authoritative list of flags. The CLI prints a quick summary when the export completes.

### Using the standalone script

```bash
python scripts/export_memories.py --namespace research --output export/research.json
```

The script accepts the migration-specific switches required for filtered exports and always writes JSON. Output files are placed in `export/` unless an absolute path is provided.

| Option | Description |
| ------ | ----------- |
| `--namespace <name>` | Repeatable filter limiting the export to specific namespaces |
| `--no-short-term` | Skip short-term memories |
| `--no-clusters` | Skip cluster metadata |
| `--no-relationships` | Skip relationship graph edges |
| `--output <path>` | Optional output file override (defaults to `export/memories_<timestamp>.json`) |

See `python scripts/export_memories.py --help` for the latest arguments.

## Importing

### Using the CLI

```bash
memoria import-data export/snapshot.json
```

Important options:

| Option | Description |
| ------ | ----------- |
| `--database-url <DSN>` | Override the configured connection string |
| `--format {json,ndjson}` | Force the expected input format when the filename lacks an extension |
| `--table <name>` | Repeatable filter limiting the import to specific tables |
| `--no-truncate` | Append data without clearing existing rows first |

The CLI honours stdout/stdin (`-`) for piping data. Use `memoria import-data --help` to inspect all supported arguments.

### Using the standalone script

```bash
python scripts/import_memories.py export/research.json --namespace prod --dry-run
```

The script retains the namespace and validation controls formerly offered by the migration CLI, making it ideal for environment-to-environment transfers.

| Option | Description |
| ------ | ----------- |
| `--namespace <name>` | Destination namespace (defaults to `default`) |
| `--namespace-map source=target` | Remap namespaces embedded in the snapshot |
| `--allow-duplicates` | Disable duplicate detection (inserts may still fail on PK collisions) |
| `--dry-run` | Preview the import without committing changes |
| `--no-validate` | Skip the preflight validation step |

Refer to `python scripts/import_memories.py --help` for the current signature.

### Conflict resolution workflow

1. **Preview first** – run the import with `--dry-run`. Review the summary for skipped items and ensure namespace mappings are correct.
2. **Adjust mappings** – update the script arguments (`--namespace`, `--namespace-map`) to land the data in the right environment.
3. **Resolve collisions** – if the skipped rows are intentional duplicates rerun with `--allow-duplicates`. Otherwise edit the snapshot to remove them.
4. **Commit** – rerun without `--dry-run` and verify the counts reported by the CLI or script.

> Tip: snapshots are plain JSON. You can inspect the `payload` section directly to audit namespaces, clusters, or graph edges before importing.

## Full Database Backup

For a full database snapshot (schema + data) continue to use `scripts/backup_database.py`:

```bash
python scripts/backup_database.py export/memories_backup.sql
```

Running a structured snapshot alongside a full SQL backup ensures you have both an easily inspectable JSON payload and a disaster-recovery friendly dump.
