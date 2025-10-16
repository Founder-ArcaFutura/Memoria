# Supported Databases

Memoria supports multiple relational databases for persistent memory storage. Below is a table of supported databases.

## Supported Database Systems

| Database | Website | Example Link |
|----------|---------|--------------|
| **SQLite** | [https://www.sqlite.org/](https://www.sqlite.org/) | [SQLite Example](https://github.com/Founder-ArcaFutura/Memoria/tree/main/examples/databases/sqlite_demo.py) |
| **PostgreSQL** | [https://www.postgresql.org/](https://www.postgresql.org/) | [PostgreSQL Example](https://github.com/Founder-ArcaFutura/Memoria/tree/main/examples/databases/postgres_demo.py) |
| **MySQL** | [https://www.mysql.com/](https://www.mysql.com/) | [MySQL Example](https://github.com/Founder-ArcaFutura/Memoria/tree/main/examples/databases/mysql_demo.py) |
| **Neon** | [https://neon.com/](https://neon.com/) | PostgreSQL-compatible serverless database |
| **Supabase** | [https://supabase.com/](https://supabase.com/) | PostgreSQL-compatible with real-time features |
| **Community Managed** | [https://memoria.dev/](https://memoria.dev/) | Directory of community-backed hosting partners |

## Quick Start Examples

### SQLite (Recommended for Development)
```python
from memoria import Memoria

# Simple file-based database
memoria = Memoria(
    database_connect="sqlite:///memoria.db",
    conscious_ingest=True,
    auto_ingest=True
)
```

### PostgreSQL
```python
from memoria import Memoria

# PostgreSQL connection
memoria = Memoria(
    database_connect="postgresql+psycopg2://user:password@localhost:5432/memoria_db",
    conscious_ingest=True,
    auto_ingest=True
)
```

### MySQL
```python
from memoria import Memoria

# MySQL connection
memoria = Memoria(
    database_connect="mysql+pymysql://user:password@localhost:3306/memoria_db",
    conscious_ingest=True,
    auto_ingest=True
)
```

## Team Memory Configuration

Team collaboration requires explicit opt-in so that personal and shared
workspaces remain isolated by default. Enable the feature and customise the
namespace prefix through the `memory` section of your configuration:

```toml
[memory]
team_memory_enabled = true
team_namespace_prefix = "team"
team_enforce_membership = true
```

With team memory enabled you can manage shared spaces via the server API:

* `POST /memory/teams` – create or update a team namespace and seed membership.
* `POST /memory/teams/<team_id>/members` – add members or admins.
* `DELETE /memory/teams/<team_id>/members/<user_id>` – remove access for a user.
* `POST /memory/teams/<team_id>/activate` – switch the active namespace for the
  running Memoria instance.

All team data is stored under namespaces prefixed with `team_namespace_prefix`,
keeping collaborative knowledge distinct from individual memory archives.