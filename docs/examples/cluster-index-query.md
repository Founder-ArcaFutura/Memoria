# Cluster Index Query

Install optional clustering dependencies:

```bash
pip install -e ".[cluster]"
# or
pip install -r requirements-cluster.txt
```

The clustering pipeline groups similar memories and stores them in
`/data/index/latest_clusters.json`. Use the helper
`query_cluster_index` to retrieve clusters by keyword, sentiment, size,
average importance, recency-weight, or time since last update. These summaries can be
injected into LLM-based agents for broader context, and by default exclude
raw member texts so only aggregate metrics are returned. Results can also
be sorted by any numeric field such as ``weight`` or ``update_count``.

During index builds, the first few clusters are logged with their ID, size,
average importance, token totals, a centroid vector snippet, and weight so
you can quickly verify metrics without scanning the full output. Only the
first five clusters are shown to keep logs concise.

> **Note:** Vector clustering is disabled by default. Enable it by setting
> `MEMORIA_ENABLE_VECTOR_CLUSTERING=true` or adding
> `"enable_vector_clustering": true` to your `memoria.json` configuration.
> Heuristic clustering is enabled by default; disable it with
> `MEMORIA_ENABLE_HEURISTIC_CLUSTERING=false` if needed. When no `mode` is
> supplied to the `/clusters` endpoint, heuristic clustering runs if
> enabled, otherwise vector clustering when allowed. If both toggles are
> off, the endpoint reports that clustering is disabled.

## Usage

```python
from memoria.utils import query_cluster_index

# Find clusters mentioning "travel" with positive sentiment
clusters = query_cluster_index(
    keyword="travel",
    emotion_range=(0.2, 1.0),
    size_range=(3, 50),
    importance_range=(0.4, 1.0),
    weight_range=(0.5, 10.0),
    time_since_update_range=(0, 1),  # updated within the last day
    sort_by="weight",
)

for c in clusters:
    print(c["summary"], c["emotions"]["polarity"])
```

### Filtering, Sorting, and Weights

```python
# Retrieve recent high-weight clusters sorted by weight
clusters = query_cluster_index(
    weight_range=(0.5, 10.0),       # filter by decayed activity weight
    time_since_update_range=(0, 7),  # clusters updated within the last week
    sort_by="weight",               # sort descending by weight
)
```

### Cluster Ranking Metrics

Each cluster maintains several fields that influence how results are ranked:

- `update_count` – number of times the cluster has received new memories.
- `last_updated` – ISO timestamp of the most recent update.
- `weight` – a decayed score computed as `update_count * exp(-age_days / DECAY_SECONDS)`.

Higher weights indicate clusters that are both active and recent. Sorting by
`weight` or using the helper below surfaces the most relevant topics.

### Heaviest Clusters

```python
from memoria.utils import get_heaviest_clusters

top = get_heaviest_clusters(top_n=3)
for c in top:
    print(c["summary"], c["weight"])
```


## Triggering Clusters via API

You can also build clusters on demand over HTTP and immediately read them
back. This is helpful for LLM agents that need fresh context before
planning or reflection.

```bash
# create clusters from the latest memories
curl -X POST http://localhost:8000/clusters

# retrieve clusters filtered by keyword
curl http://localhost:8000/clusters?keyword=travel
```

The API responds with a JSON payload:

```json
{
  "clusters": [
    {
      "summary": "Trips to Japan and Europe",

      "members": [
        {"memory_id": "memory-id-1", "anchor": "travel", "summary": "Japan trip"},
        {"memory_id": "memory-id-2", "anchor": "travel", "summary": "Europe vacation"}
      ],


      "size": 2,
      "avg_importance": 0.8,
      "emotions": { "polarity": 0.6 },
      "update_count": 3,
      "last_updated": "2024-01-01T00:00:00Z",
      "weight": 2.7
    }
  ]
}

## Triggering an Index Rebuild

Agents can ask the server to refresh the cluster index on demand. A simple
`POST /clusters` request rebuilds the index and returns the new clusters:

```bash
curl -X POST https://memoria-in-verity.fly.dev/clusters
```

The response mirrors the GET endpoint:

```json
{"clusters": [{"id": 0, "summary": "..."}]}

```

<button id="rebuild-clusters">Rebuild clusters</button>
<div id="cluster-status"></div>

## Feeding Into an LLM

```python
from memoria.utils import query_cluster_index
from openai import OpenAI

client = OpenAI()
clusters = query_cluster_index(keyword="project x")
context = "\n".join(c["summary"] for c in clusters)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Use the memory clusters to answer."},
        {"role": "user", "content": f"Context:\n{context}\nWhat should we do next?"},
    ],
)
print(response.choices[0].message.content)
```

## Scheduling the Indexer

Run the clustering script daily at 3 AM with cron:

```
0 3 * * * /usr/bin/python /path/to/scripts/index_clusters.py >> /path/to/logs/index_clusters.log 2>&1
```

Add a lightweight job to refresh importance weights without reclustering:

```
0 * * * * /usr/bin/python /path/to/scripts/recalculate_weights.py >> /path/to/logs/recalculate_weights.log 2>&1
```

The decay parameter λ defaults to 0.05 and can be overridden via the `WEIGHT_DECAY_LAMBDA` environment variable or `custom_settings.weight_decay_lambda` configuration option.

## Memory Dashboard

For a bird's-eye summary of recent memories grouped by category and
importance, call the dashboard helper:

```python
from memoria import Memoria

m = Memoria()
m.enable()
summary = m.get_memory_dashboard(limit=50)
print(summary)
```

The same data is available over HTTP:

```bash
curl http://localhost:8000/memory/dashboard?limit=50
```

which responds with JSON such as:

```json
{
  "total_memories": 5,
  "by_category": {"work": 3, "personal": 2},
  "by_importance_score": {"0.2": 1, "0.9": 2}
}
```

### Deep-dive analytics

Need richer telemetry? The `/analytics/summary` route aggregates category counts,
retention trends, and usage leaderboards in a single payload:

```bash
curl "http://localhost:8000/analytics/summary?days=30&top=10"
```

Each metric is returned as structured JSON, so you can pump it directly into
Grafana, Pandas, or the revamped dashboard Analytics panel. Filter by namespace
with `?namespace=project-x` to compare tenants or teams.
