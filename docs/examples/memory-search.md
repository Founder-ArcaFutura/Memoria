# Memory search API

The `/memory/search` endpoint lets you query memories with multiple filters.
Provide any combination of textual, categorical, temporal, spatial and
importance parameters to target exactly the memories you need.

## Query parameters

- `query`/`q` – search text
- `limit` – maximum number of results
- `keyword` – require specific term (may repeat)
- `category` – filter by category label (may repeat)
- `anchor` – match symbolic anchors (may repeat)
- `start_timestamp` / `end_timestamp` – ISO 8601 range
- `min_importance` – minimum importance score
- `x`, `y`, `z`, `max_distance` – spatial coordinates and search radius

## Example

```bash
curl "http://localhost:8000/memory/search?query=project&category=work&anchor=planning&start_timestamp=2024-01-01T00:00:00Z&end_timestamp=2024-02-01T00:00:00Z&min_importance=0.5&x=0&y=0&z=0&max_distance=5"
```

Response:

```json
{
  "memories": [...],
  "applied_filters": {
    "query": "project",
    "category": ["work"],
    "anchors": ["planning"],
    "start_timestamp": "2024-01-01T00:00:00+00:00",
    "end_timestamp": "2024-02-01T00:00:00+00:00",
    "min_importance": 0.5,
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "max_distance": 5.0
  }
}
```

You can mix and match multiple categories or anchors by repeating the
parameter:

```bash
curl "http://localhost:8000/memory/search?keyword=python&category=skill&category=project&anchor=reflection&min_importance=0.4"
```
