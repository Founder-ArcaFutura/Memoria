# Multi-Region Deployment Guide

This guide documents a practical approach to running Memoria across multiple geographic regions with synchronized storage, coordinated background jobs, and resilient failover paths. It references the Redis and PostgreSQL sync backends (`memoria/sync/redis.py`, `memoria/sync/postgres.py`) as well as the scheduler helpers from `memoria_server/api/scheduler.py`.

## 1. Topology Overview

A minimum multi-region deployment typically has:

1. **Primary Memoria API nodes** that own the write path for ingestion and clustering.
2. **Secondary Memoria API nodes** that mirror the state for local reads and burst traffic.
3. **Regional data services** (Redis or PostgreSQL) that transport sync events.
4. **Coordinated background jobs** (daily temporal decrement, cluster refresh) that must not double-execute.

```mermaid
graph LR
    subgraph RegionA[Region A]
        A1[Memoria API
        autoscaling group]
        A2[(PostgreSQL primary)]
        A3[(Redis cache)]
    end

    subgraph RegionB[Region B]
        B1[Memoria API
        read replica set]
        B2[(PostgreSQL read replica)]
        B3[(Redis replica)]
    end

    A1 -- Sync events --> A3
    A1 -- LISTEN/NOTIFY --> A2
    B1 -- Sync events --> B3
    B1 -- LISTEN/NOTIFY --> B2

    A3 <-- Cross-region replication --> B3
    A2 <-- Logical replication --> B2

    classDef infra fill:#1f2933,stroke:#d1d5db,color:#f9fafb;
    classDef compute fill:#111827,stroke:#6b7280,color:#f9fafb;
    class A1,B1 compute;
    class A2,B2,A3,B3 infra;
```

The diagram shows how each region can maintain a local broker while still emitting sync events to peers. You may choose **Redis** for low-latency pub/sub or **PostgreSQL** when you want durable event logs—both drivers speak the same [`SyncEvent`](https://github.com/Founder-ArcaFutura/Memoria/blob/main/memoria/sync/base.py) contract.

## 2. Choosing a Sync Backend

`RedisSyncBackend` (`memoria/sync/redis.py`) is optimized for ephemeral, high-throughput pub/sub. Redis streams each JSON payload to subscribers via a background listener thread. The backend automatically:

- Sets `decode_responses=True` to work with string payloads.
- Spawns a daemon thread per subscription (`_RedisSubscription`) that forwards decoded `SyncEvent`s.
- Handles reconnection when a publish call raises an exception.

`PostgresSyncBackend` (`memoria/sync/postgres.py`) uses `LISTEN/NOTIFY` with optional JSONB persistence. Highlights include:

- Automatic creation of the `memoria_sync_events` table when `table` is defined, allowing replay during failovers.
- Driver flexibility (`psycopg` v3 or `psycopg2`).
- Reconnection safeguards for the publisher and per-subscriber listener threads.

**Recommendation.** Run Redis in every region and enable cross-region replication (or use a globally distributed Redis service). Configure PostgreSQL logical replication if you need audit history or want replayable sync streams after outages. You can also run both: Redis for live fan-out and PostgreSQL for durable recovery.

## 3. Runtime Configuration

All switches live under the `sync` block of [`MemoriaSettings`](../../configuration/settings.md#cross-instance-sync). You can toggle them via environment variables or the `/settings` API.

### Environment variables

```bash
# Redis-backed fan-out with privacy floor
export MEMORIA_SYNC__ENABLED=true
export MEMORIA_SYNC__BACKEND=redis
export MEMORIA_SYNC__CONNECTION_URL="redis://redis.region-a.internal:6379/0"
export MEMORIA_SYNC__CHANNEL="memoria-sync"
export MEMORIA_SYNC__PRIVACY_FLOOR=0         # only broadcast public/semi-public memories
export MEMORIA_SYNC__REALTIME_REPLICATION=true

# Secondary region pulls from a replicated Redis endpoint
export MEMORIA_SYNC__CONNECTION_URL="redis://redis.region-b.internal:6379/0"
```

```bash
# PostgreSQL LISTEN/NOTIFY with durable journal
export MEMORIA_SYNC__ENABLED=true
export MEMORIA_SYNC__BACKEND=postgres
export MEMORIA_SYNC__CONNECTION_URL="postgresql://memoria:secret@writer.region-a.internal:5432/memoria"
export MEMORIA_SYNC__CHANNEL="memoria_sync"
export MEMORIA_SYNC__TABLE="infra.memoria_sync_events"
export MEMORIA_SYNC__PRIVACY_FLOOR=-5        # share semi-private memories too
export MEMORIA_SYNC__PRIVACY_CEILING=15
```

Use the `/settings` endpoint to change these without restarts:

```bash
curl -X PATCH "https://memoria.region-a.example.com/settings" \
  -H "X-API-Key: $MEMORIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "sync.enabled": true,
        "sync.backend": "redis",
        "sync.realtime_replication": true,
        "sync.privacy_floor": 0,
        "sync.privacy_ceiling": 15
      }'
```

### Privacy floors and replication policies

`sync.privacy_floor` and `sync.privacy_ceiling` map directly to the spatial Y axis (privacy) described in the [Settings](../../configuration/settings.md#cross-instance-sync) reference. Values outside the range `-15` to `15` are ignored. Combine them with `sync.realtime_replication=true` to push full payloads only when the target node is allowed to store them.

When a node has stricter privacy settings than the publisher, the inbound handler discards events that fall outside its allowed range. This prevents secondary regions with different data residency requirements from accidentally storing restricted memories.

## 4. Coordinating Background Schedulers

`memoria_server/api/scheduler.py` manages the **daily temporal decrement** job. In multi-region clusters you must ensure only one region triggers the job at midnight per time zone, otherwise `x_coord` values would decrement multiple times.

1. Designate a “scheduler leader” region (usually the write-primary).
2. Set `DECREMENT_TZ` in that region’s environment (defaults to `UTC`).
3. In standby regions, disable the timer by clearing the timer after app start:

```python
# in your region-b bootstrap script
from memoria_server.api.scheduler import cancel_daily_decrement_timer

cancel_daily_decrement_timer(app)
```

Alternatively, gate the scheduler behind a feature flag and toggle it through `/settings` only in the leader region. Because `_start_cancellable_timer` records timers on the Flask app config, the command above prevents drift when failover occurs.

During failover, promote a secondary region by:

```bash
curl -X POST "https://memoria.region-b.example.com/admin/promote" \
  -H "X-API-Key: $MEMORIA_API_KEY"
# then reschedule the decrement timer
python - <<'PY'
from memoria_server.api.scheduler import restart_daily_decrement_schedule
from memoria_server.api.app_factory import create_app

app = create_app()
restart_daily_decrement_schedule(app)
PY
```

## 5. Deployment Examples

### Docker Compose overlay

The following overlay composes two regions on a shared Docker network. Each region mounts its own configuration file but points to the same Postgres cluster with logical replication enabled. The `sync` block injects Redis URIs for the local brokers.

```yaml
# docker-compose.multi-region.yml
services:
  memoria-region-a:
    extends:
      file: docker-compose.yml
      service: memoria
    environment:
      MEMORIA_SYNC__ENABLED: "true"
      MEMORIA_SYNC__BACKEND: "redis"
      MEMORIA_SYNC__CONNECTION_URL: "redis://redis-a:6379/0"
      MEMORIA_SYNC__CHANNEL: "memoria-sync"
      MEMORIA_SYNC__PRIVACY_FLOOR: "0"
      MEMORIA_SYNC__REALTIME_REPLICATION: "true"
      DECREMENT_TZ: "America/New_York"
    depends_on:
      - redis-a
      - postgres-primary

  memoria-region-b:
    extends:
      file: docker-compose.yml
      service: memoria
    environment:
      MEMORIA_SYNC__ENABLED: "true"
      MEMORIA_SYNC__BACKEND: "redis"
      MEMORIA_SYNC__CONNECTION_URL: "redis://redis-b:6379/0"
      MEMORIA_SYNC__CHANNEL: "memoria-sync"
      MEMORIA_SYNC__PRIVACY_FLOOR: "0"
      MEMORIA_SYNC__PRIVACY_CEILING: "10"
      MEMORIA_SYNC__REALTIME_REPLICATION: "true"
      DISABLE_SCHEDULERS: "true"  # custom flag consumed at bootstrap
    command: ["/bin/bash", "-c", "python -m memoria_api --disable-schedulers"]
    depends_on:
      - redis-b
      - postgres-replica

  redis-a:
    image: redis:7
    command: ["redis-server", "--appendonly", "yes"]

  redis-b:
    image: redis:7
    command: ["redis-server", "--replicaof", "redis-a", "6379"]

  postgres-primary:
    image: postgres:15
    environment:
      POSTGRES_USER: memoria
      POSTGRES_PASSWORD: memoria
      POSTGRES_DB: memoria

  postgres-replica:
    image: postgres:15
    command: ["postgres", "-c", "primary_conninfo=host=postgres-primary user=replicator"]
```

Deploy with:

```bash
docker compose -f docker-compose.yml -f docker-compose.multi-region.yml up -d
```

### Terraform snippet (AWS example)

```hcl
module "memoria_region_a" {
  source = "./modules/memoria"  # wraps ECS/Fargate service
  name   = "memoria-a"
  env = {
    MEMORIA_SYNC__ENABLED             = "true"
    MEMORIA_SYNC__BACKEND             = "postgres"
    MEMORIA_SYNC__CONNECTION_URL      = aws_rds_cluster.memoria.writer
    MEMORIA_SYNC__CHANNEL             = "memoria_sync"
    MEMORIA_SYNC__TABLE               = "infra.memoria_sync_events"
    MEMORIA_SYNC__PRIVACY_FLOOR       = "0"
    MEMORIA_SYNC__REALTIME_REPLICATION = "true"
    DECREMENT_TZ                      = "UTC"
  }
}

module "memoria_region_b" {
  source = "./modules/memoria"
  name   = "memoria-b"
  env = {
    MEMORIA_SYNC__ENABLED        = "true"
    MEMORIA_SYNC__BACKEND        = "postgres"
    MEMORIA_SYNC__CONNECTION_URL = aws_rds_cluster.memoria.reader
    MEMORIA_SYNC__CHANNEL        = "memoria_sync"
    MEMORIA_SYNC__PRIVACY_FLOOR  = "0"
    MEMORIA_SYNC__PRIVACY_CEILING = "10"
  }
}

resource "aws_route53_health_check" "memoria_api" {
  fqdn              = "api.memoria.example.com"
  type              = "HTTPS"
  resource_path     = "/healthz"
  failure_threshold = 3
}

resource "aws_route53_record" "memoria_failover" {
  zone_id = data.aws_route53_zone.primary.id
  name    = "api.memoria.example.com"
  type    = "A"

  set_identifier = "primary"
  failover_routing_policy {
    type = "PRIMARY"
  }
  alias {
    name                   = module.memoria_region_a.lb_dns_name
    zone_id                = module.memoria_region_a.lb_zone_id
    evaluate_target_health = true
  }

  depends_on = [aws_route53_health_check.memoria_api]
}

resource "aws_route53_record" "memoria_failover_secondary" {
  zone_id = data.aws_route53_zone.primary.id
  name    = "api.memoria.example.com"
  type    = "A"

  set_identifier = "secondary"
  failover_routing_policy {
    type = "SECONDARY"
  }
  alias {
    name                   = module.memoria_region_b.lb_dns_name
    zone_id                = module.memoria_region_b.lb_zone_id
    evaluate_target_health = true
  }

  health_check_id = aws_route53_health_check.memoria_api.id
}
```

This example wires Route 53 failover records so that the secondary region takes over when the primary health check fails. Because both modules share the same RDS cluster (with writer/reader endpoints), replication continues during failover.

## 6. Validation Checklist

1. **Sync heartbeat** – Trigger a memory write in Region A and confirm Region B receives the corresponding `memoria_sync_events` notification.
2. **Privacy filters** – Create a memory with `y_coord=-12`. Verify it is not replicated when `privacy_floor=0`.
3. **Scheduler control** – Ensure only the leader region decrements `x_coord` by querying the `service_metadata` table across regions.
4. **Failover drill** – Simulate a shutdown of Region A. Confirm traffic fails over to Region B and that Redis/PostgreSQL listeners reconnect automatically.

By combining the sync backends with disciplined scheduler control, you can run Memoria in multiple regions without duplicating background tasks or leaking sensitive memories across residency boundaries.
