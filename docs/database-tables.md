# Database Table Reference

This document summarizes the purpose, behavior, and integration points for each table that powers Memoria's memory system.

## chat_history
- **Purpose:** Persist every conversational exchange with associated metadata such as user input, AI response, model, timestamps, namespace, token counts, and auxiliary metadata.
- **Interactions:** Short-term memories can reference their originating chat via the `chat_id` column. Service-level APIs expose helpers that enumerate chats by namespace or session for UI and integration use.

## cluster_members
- **Purpose:** Associate individual memories with their containing cluster, preserving each member's anchor, summary, token and character counts alongside the owning `cluster_id` and memory identifier.
- **Interactions:** Cluster management utilities rebuild membership lists and support filtering clusters by member content during query operations. Retention logic traverses cluster memberships to recompute weights and centroids whenever a member changes.

## clusters
- **Purpose:** Store high-level cluster metadata including the aggregate summary, spatial centroids, sentiment scores, membership count, average importance, update counters, weights, and token/character totals.
- **Interactions:** Cluster administration and retention routines keep these aggregates in sync with member updates and surface them to reporting and search tools.

## link_memory_threads
- **Purpose:** Maintain directed edges between memories so the system can represent traversable conversation graphs; records include source, target, relationship label, and creation time.
- **Interactions:** The API ensures the table exists prior to use. Thread ingestion rebuilds adjacency edges on every upload, and traversal endpoints walk these links to render graph visualizations.

## long_term_memory
- **Purpose:** Hold durable processed memories enriched with importance scores, categorical labels, classification metadata, conscious-context flags, duplicate tracking, confidence metrics, and spatial coordinates or anchors.
- **Interactions:** Long-term rows participate in retrieval unions and full-text searches alongside short-term and spatial metadata, enabling downstream services to surface the right memories for prompts or dashboards.

## memory_access_events
- **Purpose:** Audit every memory access, capturing the memory identifier, namespace, timestamp, access type, source, and supplemental metadata.
- **Interactions:** Whenever a memory is read, `record_memory_touches` increments per-memory counters and appends a corresponding access-event row so the audit log matches usage statistics.

## memory_search_fts
- **Purpose:** Provide an SQLite FTS5 virtual table that indexes memory text, summaries, categories, anchors, namespaces, and memory types to enable phrase and keyword search across storage tiers.
- **Interactions:** Database triggers keep the index synchronized with inserts and deletes from the short- and long-term tables. The SQLite search adapter queries the index, then joins to the underlying memory tables to return ranked results.

## memory_search_fts_config, memory_search_fts_docsize, memory_search_fts_idx
- **Purpose:** Shadow tables created automatically by SQLite FTS5 to persist tokenizer configuration, per-document statistics, and the inverted index that backs `memory_search_fts`.
- **Interactions:** Maintained internally by SQLite. Applications should not modify them directly.

## memory_threads
- **Purpose:** Not part of the canonical schema. Any table bearing this name in a deployment is typically legacy or derived from the primary thread tables listed below.
- **Interactions:** Thread functionality is exposed through `link_memory_threads`, `thread_events`, and `thread_message_links`.

## service_metadata
- **Purpose:** Act as a key/value store for service-level state, such as schema migration checkpoints or scheduler markers.
- **Interactions:** The temporal-coordinate scheduler records its last run date here so daily decrements can resume safely without double-processing.

## short_term_memory
- **Purpose:** Record recently extracted memories including processed payloads, importance scores, category labels, retention type, creation/expiry timestamps, search summaries, permanence flag, and optional spatial coordinates or anchors. Entries may reference the originating chat via `chat_id`.
- **Interactions:** Short-term rows merge with long-term and spatial metadata during retrieval and search pipelines, ensuring fresh context is represented in spatial and full-text lookups.

## spatial_metadata
- **Purpose:** Store per-memory spatial coordinates and canonicalized anchor lists, keyed by memory and namespace, to support spatial reasoning independent of the primary memory tables.
- **Interactions:** Application startup guarantees the table exists and normalizes historical columns. Storage services write new coordinates and include the table in spatial unions during retrieval.

## thread_events
- **Purpose:** Capture per-thread metadata such as shared anchors, ritual descriptors, centroid coordinates, and timestamps for each ingested conversational thread within a namespace.
- **Interactions:** Thread ingestion upserts these rows, and thread reconstruction combines them with message links so APIs can surface thread-level context and rituals.

## thread_message_links
- **Purpose:** Map stored memories into ordered thread sequences, recording the thread identifier, memory ID, namespace, sequence index, speaker role, anchor, and optional timestamp for each linked message.
- **Interactions:** Thread ingestion rebuilds the link set whenever a thread payload is processed. Query helpers use these links (plus associated thread events) to expose per-memory thread participation.

