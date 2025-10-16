# Sync & Failover Playbook

Memoria's sync layer can operate across multiple regions with privacy-aware
replication, deterministic failover, and health instrumentation that matches the
roadmap commitments. This playbook documents the canonical reference
architectures, the exact configuration surfaces (`memoria.toml` and environment
variables), and the drills you can run to validate disaster readiness.

## Multi-Region Topologies

The table below highlights when to select each topology. Detailed wiring
instructions follow in the subsequent sections.

| Topology | Recommended For | Sync Backend | Failover Mode |
| --- | --- | --- | --- |
| Redis Bridge (Active/Active) | Low-latency read/write in two regions with shared tenants | Redis pub/sub with privacy floor and namespace routing | Manual or orchestrated DNS/Anycast swap |
| PostgreSQL Logical Replication (Active/Passive) | Strong consistency, high write volume, centralized audit history | PostgreSQL advisory channel + logical replication slots | Promote read replica to writer |

### Redis-to-Redis Bridge (Active/Active)

```
Region A App  ─┐        ┌─ Redis Primary (memoria-sync-a)
               ├─ Sync ─┤
Region B App  ─┘        └─ Redis Replica/Cluster (memoria-sync-b)
```

*Both regions publish and consume on the same logical channel. Privacy floors
and namespace routing stop tenant data from crossing regional boundaries.*

**Environment variables** (Region A):

```bash
export MEMORIA_MEMORY__NAMESPACE="prod-shared"
export MEMORIA_SYNC__ENABLED=true
export MEMORIA_SYNC__BACKEND=redis
export MEMORIA_SYNC__CONNECTION_URL="redis://redis.use1.example.com:6379/0"
export MEMORIA_SYNC__CHANNEL="memoria-sync"
export MEMORIA_SYNC__REALTIME_REPLICATION=true
export MEMORIA_SYNC__PRIVACY_FLOOR=0          # share only public/semi-public events
export MEMORIA_SYNC__PRIVACY_CEILING=12        # keep strictly private entries local
export MEMORIA_SYNC__OPTIONS__ssl=True         # forwarded to redis.from_url
```

**Environment variables** (Region B):

```bash
export MEMORIA_MEMORY__NAMESPACE="prod-shared"
export MEMORIA_SYNC__ENABLED=true
export MEMORIA_SYNC__BACKEND=redis
export MEMORIA_SYNC__CONNECTION_URL="redis://redis.euw1.example.com:6379/0"
export MEMORIA_SYNC__CHANNEL="memoria-sync"
export MEMORIA_SYNC__REALTIME_REPLICATION=true
export MEMORIA_SYNC__PRIVACY_FLOOR=0
export MEMORIA_SYNC__PRIVACY_CEILING=12
export MEMORIA_SYNC__OPTIONS__ssl=True
```

**`memoria.toml` (shared template):**

```toml
[memory]
namespace = "prod-shared"  # shared namespace for the tenant across regions
team_namespace_prefix = "team"  # generates team-alpha, team-bravo, etc.
privacy_floor = -15

[sync]
enabled = true
backend = "redis"
connection_url = "redis://redis.use1.example.com:6379/0"
channel = "memoria-sync"
realtime_replication = true
privacy_floor = 0
privacy_ceiling = 12
[sync.options]
ssl = true
socket_timeout = 3
retry_on_timeout = true

```

Override `memoria.toml` per region by injecting
`MEMORIA_SYNC__CONNECTION_URL` and Redis client options. The sync coordinator
only publishes and consumes events whose `namespace` matches the local value, so
run one Memoria deployment per tenant namespace (e.g. `prod-shared`) and let
team spaces inherit prefixes such as `team-alpha` or `team-bravo`. That routing
keeps tenant data separated even when both regions stay active.

**Operational notes**

- Use DNS or Anycast to direct clients to the nearest region.
- Redis' built-in replication or a managed service (AWS Elasticache Global Data
  Store, GCP MemoryStore) keeps channels aligned. Memoria only needs a reachable
  endpoint per region.
- Because `privacy_floor` and `privacy_ceiling` map to the Y-axis of the privacy
  coordinate, private workspaces (`y <= -10`) never exit the origin region.

### PostgreSQL Logical Replication (Active/Passive)

```
Region A App ─┐            ┌─ PostgreSQL Primary (writer)
              ├─ Sync Bus ─┤
Region B App ─┘            └─ PostgreSQL Replica (logical slot + read endpoint)
```

*All sync events are appended to a dedicated table in Region A. Region B reads
via logical replication and replays changes when promoted.*

**Environment variables** (Region A / current writer):

```bash
export MEMORIA_MEMORY__NAMESPACE="prod-shared"
export MEMORIA_SYNC__ENABLED=true
export MEMORIA_SYNC__BACKEND=postgres
export MEMORIA_SYNC__CONNECTION_URL="postgresql://memoria:secret@use1-writer.example.com:5432/memoria"
export MEMORIA_SYNC__CHANNEL="memoria_sync"
export MEMORIA_SYNC__TABLE="infra.memoria_sync_events"
export MEMORIA_SYNC__PRIVACY_FLOOR=-5         # allow semi-private research to replicate
export MEMORIA_SYNC__PRIVACY_CEILING=15
```

**Environment variables** (Region B / hot standby):

```bash
export MEMORIA_MEMORY__NAMESPACE="prod-shared"
export MEMORIA_SYNC__ENABLED=true
export MEMORIA_SYNC__BACKEND=postgres
export MEMORIA_SYNC__CONNECTION_URL="postgresql://memoria:secret@euw1-reader.example.com:5432/memoria"
export MEMORIA_SYNC__CHANNEL="memoria_sync"
export MEMORIA_SYNC__TABLE="infra.memoria_sync_events"
export MEMORIA_SYNC__PRIVACY_FLOOR=-5
export MEMORIA_SYNC__PRIVACY_CEILING=15
export MEMORIA_SYNC__CONNECT_KWARGS__sslmode=require
```

**`memoria.toml` (writer):**

```toml
[memory]
namespace = "prod-shared"

[sync]
enabled = true
backend = "postgres"
connection_url = "postgresql://memoria:secret@use1-writer.example.com:5432/memoria"
channel = "memoria_sync"
table = "infra.memoria_sync_events"
realtime_replication = true
privacy_floor = -5
privacy_ceiling = 15
[sync.connect_kwargs]
sslmode = "require"
application_name = "memoria-sync-writer"
```

Because both regions share the namespace (`prod-shared`), tenant isolation is
driven by the privacy floor. Set `privacy_floor` to `0` when the standby should
only ingest public context, or to `-5` when regional regulators allow
confidential (but not private) material to cross.

## Failover Drills

Run the following exercises quarterly (or during every change window) to prove
that the multi-region architecture still behaves as designed.

### Redis Bridge Drill

1. **Warm-up validation** – Run `tests/utils/sync_health_smoketest.py --topology redis`.
2. **Cut primary network** – Block outbound traffic from Region A to Redis for
   five minutes (security group toggle or firewall rule).
3. **Observe propagation** – Confirm Region B continues to service requests and
   writes are broadcast through its Redis endpoint.
4. **Promote Region B** – Switch DNS (or traffic manager weight) to Region B.
5. **Reinstate Region A** – Re-enable connectivity and verify the health script
   reports both Redis endpoints green.
6. **Post-drill review** – Export metrics dashboards and audit the sync logs to
   confirm privacy floors were respected (no private events transferred).

### PostgreSQL Logical Replication Drill

1. **Health check** – `tests/utils/sync_health_smoketest.py --topology postgres`.
2. **Simulate outage** – Stop the writer instance or revoke its security group.
3. **Promote standby** – Execute your managed database failover command (e.g.
   `aws rds failover-db-cluster --db-cluster-identifier memoria`).
4. **Flip application** – Update the `MEMORIA_SYNC__CONNECTION_URL` and database
   URL to the new writer; reload the Memoria API.
5. **Replay backlog** – Run `memoria sync catch-up --since <timestamp>` (CLI
   plugin) to apply any ledger rows generated during failover.
6. **Downgrade privacy** – Temporarily tighten `privacy_floor` to `0` while the
   promoted region is authoritative; relax once regulatory review completes.

## Observability Checklist

To keep the playbook actionable, wire these probes into your dashboards and
incident runbooks.

### Metrics & Alerts

- `sync_events_published_total{backend="redis"}` – Expect roughly equal counts
  between regions in Active/Active mode.
- `sync_subscription_latency_seconds{region="*"}` – Should stay < 1s for Redis
  and < 5s for Postgres logical replication.
- `privacy_filtered_events_total` – Alert if the number spikes, indicating
  tenants are attempting to cross privacy floors.
- `namespace_mismatch_drops_total` – Derive from Memoria logs (`SyncCoordinator`)
  to ensure namespace routing still aligns with expected tenants.

### Logs

- Enable `MEMORIA_LOGGING__LEVEL=DEBUG` during drills to capture
  `SyncCoordinator` publish/consume entries.
- Ship Redis `monitor` output or Postgres `pg_stat_replication` states to your
  SIEM for correlation.

### Health Endpoints

- `GET /health/sync` (Memoria server) – returns per-backend status, including
  namespace and privacy floor metadata.
- Redis `INFO replication` – monitor `master_link_status` or cluster shard
  health.
- Postgres `pg_is_in_recovery()` – detect when standby promotion completes.

### Dashboards

- **Topology Overview** – Graph per-region sync rates, namespace counts, and
  privacy floor thresholds.
- **Failover Timeline** – During drills, annotate when DNS flips, when sync
  resumes, and when audit checks finish.

## Privacy Floors & Namespace Routing

Memoria encodes privacy on the Y-axis (−15 private → +15 public) and always
stamps sync events with the originating namespace. Combine both controls:

1. Assign each tenant/team to a namespace (e.g. `team-alpha`, `team-bravo`).
2. Map namespaces to regions by running a dedicated Memoria deployment per namespace or overriding `MEMORIA_MEMORY__NAMESPACE` during launch.
3. Set `privacy_floor` and `privacy_ceiling` so only approved content crosses
   regions.

During failover, temporarily raise the privacy floor if regulators require
additional controls. Because the receiving region will only apply events with a
matching namespace, tenants never bleed into the wrong geography even while the
floor is relaxed.

---

**Next steps**: Pair this playbook with the [Multi-Region Deployment Guide](../deployments/multi-region.md) for infrastructure-as-code
examples and cloud-specific notes.
