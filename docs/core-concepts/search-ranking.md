# Search Ranking

Memoria ranks search results using a weighted formula that balances text relevance,
importance, recency and symbolic anchor matches.

The composite score for a result is calculated as:

```
composite = (search_score * w_search) +
            (importance_score * w_importance) +
            (recency_score * w_recency) +
            (anchor_match ? w_anchor : 0)
```

Default weights:

- `w_search` = 0.5
- `w_importance` = 0.3
- `w_recency` = 0.2
- `w_anchor` = 0.1 (bonus when the query matches a memory's `symbolic_anchors`)

You can override these coefficients globally or per search:

```python
from memoria.database.search_service import SearchService

service = SearchService(session, "sqlite", rank_weights={"search": 0.4, "anchor": 0.2})

# Per-query override using the rank_weights parameter
results = service.search_memories("project-x", rank_weights={"anchor": 0.5})
```

Tuning these weights lets you emphasise the factors that matter most for your
application's relevance model. You can also supply the weight dictionary from
configuration when constructing the service to apply it globally.
