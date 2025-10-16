# Basic usage

Learn how to work with Memoria programmatically using the Python SDK. The examples below mirror the behaviour exposed by the REST API and CLI so you can embed the memory engine in agents, automations, or evaluation harnesses.

## Understand the spatial model

Long-term memories live in a spatial index. Each entry includes:

| Field | Purpose | Typical range |
| --- | --- | --- |
| `x_coord` | Temporal offset in days relative to “now”. Negative values are in the past, positive values point to the future. | `-∞` → `+∞` |
| `y_coord` | Privacy spectrum. `-15` is highly private, `+15` is public. Policies prevent accidental promotion across sensitive boundaries. | `-15` → `+15` |
| `z_coord` | Cognitive vs. physical emphasis. Negative values lean sensory/emotional, positive values lean abstract/strategic. | `-15` → `+15` |
| `symbolic_anchors` | Optional semantic tags (for example `"migration"`, `"ritual"`). Used when spatial queries miss but anchors align. | Any list of canonical labels |

Heuristic scoring combines these coordinates with promotion metadata to decide when staged items graduate to long-term storage.

## Initialise the SDK

```python
from memoria import Memoria

memoria = Memoria(
    database_connect="sqlite:///memoria.db",
    conscious_ingest=True,    # inject essential context at session start
    auto_ingest=True,         # opt into per-request context retrieval
    namespace="default",
)
memoria.enable()  # registers OpenAI/LiteLLM hooks when they are available
```

`Memoria` automatically loads additional configuration from `memoria.json`, environment variables, and `.env` files created by the bootstrap wizard.

## Record conversations automatically

When `enable()` is called, Memoria attaches instrumentation to supported providers (OpenAI, Anthropic, LiteLLM). Manual recording is also available:

```python
chat_id = memoria.record_conversation(
    user_input="I'm migrating analytics workloads to Delta Lake.",
    ai_output={"content": "Great! Let's capture milestones in the memory graph."},
)
```

The conversation is written to chat history, staged for promotion, and evaluated by heuristic scoring. Use `memoria.trigger_conscious_analysis()` when you want to force promotion immediately instead of waiting for the scheduler.

## Stage manual memories

Provide explicit coordinates and anchors when you already know how a fact should be classified:

```python
note = "Riley Chen owns the analytics migration for Q4 2025."
memoria.store_memory(
    anchor="analytics-migration",
    text=note,
    tokens=len(note.split()),
    x_coord=-90.0,  # roughly three months ago
    y=-3.0,         # moderately private
    z=9.5,          # strategic initiative
    symbolic_anchors=["analytics", "migration", "roadmap"],
)
```

`store_memory` stages the entry, applies promotion heuristics, and—when approved—copies coordinates into long-term storage. Attach `metadata` or `user_priority` to influence promotion decisions.

## Retrieve context for agents

Use retrieval helpers to fetch high-signal context:

```python
context = memoria.retrieve_context(
    "Who is leading the analytics migration?",
    limit=3,
)
for item in context:
    print(item["content"], item.get("importance"))
```

Spatial lookups let you target specific areas of the coordinate system:

```python
nearby = memoria.retrieve_memories_near(
    x=0.0,
    y=0.0,
    z=9.0,
    max_distance=6.0,
    anchor=["analytics", "roadmap"],
)
```

Memoria falls back to anchors when no memories meet the distance threshold, ensuring deterministic behaviour even when vector search is disabled.

## Namespaces, teams & workspaces

Namespaces isolate memory graphs per project or customer. Team and workspace features add collaborative sharing when enabled:

```python
team_memoria = Memoria(
    namespace="engineering",
    team_mode="optional",
    default_team_id="platform",
    team_share_by_default=True,
)
team_memoria.enable()

team_memoria.record_conversation(
    user_input="Share the auth rollout plan with the platform team.",
    ai_output="Uploaded milestones and key contacts to the shared namespace.",
    metadata={"workspace": "q4-rollout"},
)
```

CLI helpers and the REST API expose the same namespace, team, and workspace semantics so auditors can trace which operator or team promoted each memory.

## Inspect memory health

Surface ingestion health before rolling out changes:

```python
stats = memoria.get_memory_stats()
print(f"Conversations: {stats['chat_history_count']}")
print(f"Long-term memories: {stats['long_term_count']}")

if stats["short_term_count"] > 1000:
    memoria.trigger_conscious_analysis()
```

Pair stats with the evaluation harness (`examples/evaluation_quickstart.py`) to confirm retrieval quality after modifying heuristics or datasets.

## Expose a tool for agents

Wrap the retrieval helpers inside a tool or function-call interface to plug Memoria into other agent frameworks:

```python
from memoria import create_memory_tool

memory_tool = create_memory_tool(memoria)

def search_memory(query: str) -> str:
    result = memory_tool.execute(query=query)
    return "\n".join(item.content for item in result.memories)
```

The same helper powers the REST `/tools` endpoint, keeping agent integrations consistent across runtimes.

## Troubleshooting checklist

| Symptom | Likely cause | Suggested fix |
| --- | --- | --- |
| `MemoriaError: Memoria is not enabled` | Forgot to call `enable()` before recording conversations. | Call `memoria.enable()` after instantiation. |
| Memories are staged but not promoted | Promotion heuristics rejected the entry. | Review `staged.metadata`, adjust coordinates or `user_priority`, or inspect policy logs in the API service. |
| Retrieval returns an empty list | Namespace mismatch or overly strict filters. | Pass `namespace=...` explicitly or widen `limit`/`max_distance`. |
| Cluster endpoints return `disabled` | No clustering flag enabled. | Set `MEMORIA_ENABLE_HEURISTIC_CLUSTERING` or `MEMORIA_ENABLE_VECTOR_CLUSTERING` and rebuild indexes. |

Use the logging hooks (`MEMORIA_LOG_LEVEL=DEBUG`) when you need verbose traces from promotion pipelines or sync backends.
