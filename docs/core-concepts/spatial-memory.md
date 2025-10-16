# Spatial & Symbolic Memory

At the heart of Memoria is a unique spatial-symbolic memory model. Instead of relying solely on vector similarity, Memoria represents each memory in a 3D conceptual space, augmented with symbolic "anchors." This approach provides a transparent and auditable way to reason about how an AI agent stores and retrieves information.

## The 3D Coordinate System

Each memory is assigned a coordinate `(x, y, z)` that represents its position in a conceptual space. This allows for powerful queries based on the *context* of a memory, not just its content.

-   **`x` - Temporal Axis**: Represents the time offset in days from the present moment.
    -   `x = -7.0` means one week ago.
    -   `x = 0.0` means now.
    -   `x = 1.5` means a day and a half in the future.
    -   This value is automatically calculated from a memory's timestamp if not provided explicitly.

-   **`y` - Privacy Axis**: Represents the sensitivity of the information, ranging from -15 to +15.
    -   `y <= -10`: Deeply private (e.g., confessions, secrets). Should never be shared in multi-agent contexts.
    -   `-10 < y < 0`: Moderately sensitive (e.g., personal reflections). Should be shared with caution.
    -   `y >= 10`: Public information, safe to share freely.

-   **`z` - Cognitive/Physical Axis**: Represents the nature of the memory, ranging from -15 to +15.
    -   `z < -5`: Sensory, bodily, or emotional memories (e.g., the feeling of a cold breeze).
    -   `z > 5`: Abstract, strategic, or intellectual memories (e.g., a business plan, a philosophical reflection).

This spatial model allows for nuanced queries, such as "retrieve all moderately private, abstract thoughts from the last week."

## Symbolic Anchors

In addition to its spatial coordinates, a memory can be tagged with one or more **symbolic anchors**. These are simple string labels that represent archetypal or conceptual categories.

Examples: `"decision"`, `"ritual"`, `"goal"`, `"insight"`, `"wound"`.

Symbolic anchors provide a powerful, human-readable way to filter and retrieve memories. You can query for memories near a certain spatial point *and* tagged with a specific anchor, creating a hybrid retrieval strategy that combines the best of spatial and categorical search.

## How Spatial Queries Work

The `StorageService` in Memoria provides methods like `retrieve_memories_near()` that execute spatial queries directly in the database.

The query calculates the Euclidean distance between a target point `(x, y, z)` and the coordinates of all stored memories. It then returns the memories that fall within a specified `max_distance`.

**Example SQLite Distance Formula:**

```sql
SELECT *
FROM long_term_memory
WHERE ((x_coord - ?) * (x_coord - ?) +
       (y_coord - ?) * (y_coord - ?) +
       (z_coord - ?) * (z_coord - ?)) < ? * ?;
```

This hybrid spatial-symbolic system allows for a rich and intuitive way to manage an AI's memory, moving beyond simple keyword or vector search to a more context-aware retrieval process.