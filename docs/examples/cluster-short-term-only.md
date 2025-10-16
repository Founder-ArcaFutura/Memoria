# Cluster Short-Term Memory Only

Vector clustering now lets you choose which memory tables are read during an index
rebuild. By default `build_index()` looks at both `LongTermMemory` and
`ShortTermMemory`, but you can run the pipeline against short-term entries only to
avoid touching the long-term archive. This is useful when experimenting with new
cluster settings or when you want to prune transient memories without rewriting
archival context.

## API Request

Send a `POST /clusters` request with the desired sources. Setting
`mode=vector` forces the vector pipeline and the `sources` payload limits the
SQL queries to short-term memory.

```bash
curl -X POST "http://localhost:8080/clusters?mode=vector" \
  -H "Content-Type: application/json" \
  -d '{"sources": ["ShortTermMemory"]}'
```

You can also pass `source=ShortTermMemory` as a repeated query parameter. Any
unknown table names are ignored and, if every source is skipped, the request
fails with `"No texts available for clustering"`.

## Python Usage

To rebuild clusters directly from Python without touching the long-term table,
call `build_index()` with an explicit `sources` argument:

```python
from scripts.index_clusters import build_index

clusters = build_index(sources=["ShortTermMemory"])
print(len(clusters), "clusters updated")
```

The order of the `sources` sequence controls the query order. Duplicate names
are ignored and the function skips tables that do not exist in the current
schema, logging a warning for each unsupported source.
