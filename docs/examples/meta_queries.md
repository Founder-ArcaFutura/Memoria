# Meta queries


Use boolean anchor expressions and privacy ranges to explore memory metadata.

## Building a query

```python
from memoria.utils.query_builder import QueryBuilder, DatabaseDialect

qb = QueryBuilder(DatabaseDialect.SQLITE)
sql, params = qb.build_search_query(
    tables=["long_term_memory"],
    search_columns=["searchable_content"],
    query_text="",
    namespace="default",
    anchor_expression="ritual AND (confession OR wound) AND NOT secret",
    privacy_range=(-5.0, 5.0),
)
```

## Aggregating results

```python
from memoria.database.search_service import SearchService

service = SearchService(session, "sqlite")
counts = service.meta_query(["ritual", "confession"], (-5.0, 5.0))
# {'ritual': 3, 'confession': 1}
=====
`SearchService.meta_query()` aggregates memory counts using boolean anchor
expressions and a privacy (`y`) range. Provide one or more anchor expressions
and a `(min, max)` tuple for the privacy range; the service returns the total
number of matching memories and counts for each expression.

## Example

```python
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memoria.database.models import Base, ShortTermMemory
from memoria.database.search_service import SearchService

engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

session.add(
    ShortTermMemory(
        memory_id="m1",
        processed_data={"text": "alpha beta"},
        importance_score=0.5,
        category_primary="demo",
        namespace="default",
        searchable_content="alpha beta",
        summary="",
        created_at=datetime.utcnow(),
        y_coord=0.0,
        symbolic_anchors=["alpha", "beta"],
    )
)
session.commit()

service = SearchService(session, "sqlite")
summary = service.meta_query(["alpha AND beta", "alpha"], (-5.0, 5.0))
print(summary)
```

Example output:

```json
{
  "total_memories": 1,
  "by_anchor": {
    "alpha AND beta": 1,
    "alpha": 1
  }
}

```
