Overview

This document defines the operational guidelines for agents interacting with the Memoria in Verity memory system. Agents are expected to:

Respect the spatial-symbolic schema for memory storage and retrieval.

Use curated knowledge for identity, role, and core preferences.

Retrieve from the long-term SQL memory only what is relevant, avoiding context bloat.

Apply privacy, temporal, and semantic logic consistently.

🧠 Memory Spatial-Symbolic Schema

Each memory is represented in a 3D conceptual space with optional symbolic anchors:

Axis	Dimension	Range	Meaning
x	Temporal	float ±∞	Time offset in days relative to the present
y	Privacy	-15 → +15	-15 = private; +15 = public
z	Cognitive/Physical	-15 → +15	-15 = sensory/physical; +15 = abstract/intellectual
Coordinate Details

Temporal (x)

Represents days offset from the current moment.

Automatically computed from timestamp if not set.

Example:

x = -3.0 → 3 days ago

x = +1.5 → 1.5 days in the future

Privacy (y)

Estimated from tone, content, or recipient context.

Recommended usage:

y <= -10 → private; never auto-inject in multi-agent contexts

-10 < y < 0 → moderately sensitive; inject cautiously

y >= 10 → public; safe for full injection

Cognitive/Physical (z)

Indicates content type:

z < -5 → sensory, bodily, emotional

z > 5 → abstract, strategic, contemplative

🔖 Symbolic Anchors

Field: symbolic_anchors

String or array of strings representing conceptual/archetypal labels.

Examples: "eros", "ritual", "confession", "decision", "wound".

Usage Guidelines:

Maintain canonicalized labels to avoid semantic fragmentation.

Combine anchors with spatial coordinates when querying for contextually relevant memories.

Symbolic anchors are stored in symbolic_anchors column (plural) in SQL.

🔍 Retrieval Guidelines

Agents should follow a hierarchical retrieval strategy:

Proximity-based retrieval

Query memories near (x,y,z) coordinates using a distance threshold.

Example SQLite formula for 3D distance:

SELECT * 
FROM long_term_memory
WHERE ((x_coord - ?) * (x_coord - ?) +
       (y_coord - ?) * (y_coord - ?) +
       (z_coord - ?) * (z_coord - ?)) < ? * ?;


Symbolic anchor fallback

If no nearby memories exist, filter by symbolic_anchors only.

Example:

memoria.retrieve_memories_near(
    x=0.0, y=0.0, z=0.0, max_distance=5.0,
    anchor=["reflection", "clarity"]
)


Recency / importance fallback

Pull the top N memories by importance_score or timestamp if previous filters yield nothing.

Privacy filtering

Respect the y axis.

Private entries (y <= -10) should not be injected into multi-agent or public prompts.

📋 Memory Entry Schema

A memory object should include the following fields:

{
  "anchor": "reflections_august",
  "text": "I realized how much solitude helps me think clearly.",
  "tokens": 13,
  "x_coord": -12.5,
  "y_coord": -8.0,
  "z_coord": 9.0,
  "symbolic_anchors": ["solitude", "reflection", "clarity"],
  "timestamp": "2025-08-30T14:30:00Z",
  "importance_score": 0.87,
  "retention_type": "conscious",
  "namespace": "personal_journal"
}


Field Descriptions:

anchor: canonical label for semantic classification

text: memory content

tokens: token count (optional)

x_coord, y_coord, z_coord: spatial coordinates

symbolic_anchors: array of conceptual labels

timestamp: ISO-8601 timestamp

importance_score: float [0.0,1.0] indicating priority

retention_type: "conscious" vs "auto" to control injection

namespace: logical grouping for context separation

🛠 Agent Operational Rules

Curate Knowledge

Keep identity, role, and core rules in knowledge layer, not in token-heavy memory.

Inject knowledge only once at session start or on explicit update.

Insert New Memory

Store new memory in SQL with proper (x,y,z) coordinates and anchors.

Assign importance_score and retention_type according to relevance.

Query Memory

Always filter by privacy and optionally by anchor or distance.

Do not auto-inject entire tables into context windows.

Update / Promote Memory

Long-term memory can be promoted to short-term (conscious) if frequently accessed or highly relevant.

Automatically decrement x_coord daily to maintain temporal offset.

⚡ Example Workflow

Add memory:

memoria.insert_memory({
    "text": "Completed meditation session.",
    "x_coord": -0.1,
    "y_coord": -5,
    "z_coord": 10,
    "symbolic_anchors": ["ritual","mindfulness"],
    "importance_score": 0.9,
    "retention_type": "conscious",
    "namespace": "daily_reflections"
})


Retrieve relevant memories:

memories = memoria.retrieve_memories_near(
    x=0.0, y=0.0, z=0.0,
    max_distance=5.0,
    anchor=["mindfulness"]
)


Inject into session:

Select top 3–5 memories by importance_score.

Ensure y axis rules are respected.

Pass only these memories into the context window.

✅ Key Takeaways

Knowledge: identity and core rules; minimal, static.

Memory: long-term, spatial-symbolic, queryable.


----

Agents: query selectively, respect privacy, inject only what’s needed.

Spatial-symbolic coordinates: guide semantic retrieval without bloating context.

Fallbacks: anchors → importance → recency.
