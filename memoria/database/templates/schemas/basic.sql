-- Memoria 0.9 alpha (0.9.0a0) Streamlined Database Schema
-- Simplified schema with only essential tables for production use

CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Workspace Table
-- Provides fine-grained scoping for conversations and memories
CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    description TEXT,
    owner_id TEXT,
    team_id TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL
);

-- Workspace Membership Table
-- Tracks which users are associated with a workspace
CREATE TABLE IF NOT EXISTS workspace_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    is_admin BOOLEAN NOT NULL DEFAULT 0,
    joined_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workspace_id) REFERENCES workspaces (workspace_id) ON DELETE CASCADE,
    UNIQUE(workspace_id, user_id)
);

-- Chat History Table
-- Stores all conversations between users and AI systems
CREATE TABLE IF NOT EXISTS chat_history (
    chat_id TEXT PRIMARY KEY,
    user_input TEXT NOT NULL,
    ai_output TEXT NOT NULL,
    model TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    session_id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    tokens_used INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    team_id TEXT,
    workspace_id TEXT,
    FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces (workspace_id) ON DELETE SET NULL
);

-- Short-term Memory Table (with full ProcessedMemory structure)
-- Stores temporary memories with expiration (auto-expires after ~7 days)
-- Also stores permanent user context when expires_at is NULL
CREATE TABLE IF NOT EXISTS short_term_memory (
    memory_id TEXT PRIMARY KEY,
    chat_id TEXT,
    processed_data TEXT NOT NULL,  -- Full ProcessedMemory JSON
    importance_score REAL NOT NULL DEFAULT 0.5,
    category_primary TEXT NOT NULL,  -- Extracted for indexing
    retention_type TEXT NOT NULL DEFAULT 'short_term',
    namespace TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP,  -- NULL = permanent storage (for user context)
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    searchable_content TEXT NOT NULL,  -- Optimized for search
    summary TEXT NOT NULL,  -- Concise summary
    is_permanent_context BOOLEAN DEFAULT 0,  -- Marks permanent user context
    x_coord REAL,
    y_coord REAL,
    z_coord REAL,
    symbolic_anchors JSON,
    embedding JSON,
    team_id TEXT,
    workspace_id TEXT,
    FOREIGN KEY (chat_id) REFERENCES chat_history (chat_id),
    FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces (workspace_id) ON DELETE SET NULL
);

-- Long-term Memory Table (Enhanced with Classification and Conscious Context)
-- Stores persistent memories with intelligent classification and deduplication
CREATE TABLE IF NOT EXISTS long_term_memory (
    memory_id TEXT PRIMARY KEY,
    original_chat_id TEXT,
    processed_data TEXT NOT NULL,  -- Full ProcessedLongTermMemory JSON
    importance_score REAL NOT NULL DEFAULT 0.5,
    category_primary TEXT NOT NULL,  -- Extracted for indexing
    retention_type TEXT NOT NULL DEFAULT 'long_term',
    namespace TEXT NOT NULL DEFAULT 'default',
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    searchable_content TEXT NOT NULL,  -- Optimized for search
    summary TEXT NOT NULL,  -- Concise summary
    novelty_score REAL DEFAULT 0.5,
    relevance_score REAL DEFAULT 0.5,
    actionability_score REAL DEFAULT 0.5,

    -- Enhanced Classification Fields
    classification TEXT NOT NULL DEFAULT 'conversational',  -- essential, contextual, conversational, reference, personal, conscious-info
    memory_importance TEXT NOT NULL DEFAULT 'medium',  -- critical, high, medium, low
    topic TEXT,  -- Main topic/subject
    entities_json TEXT DEFAULT '[]',  -- JSON array of extracted entities
    keywords_json TEXT DEFAULT '[]',  -- JSON array of keywords for search
    
    -- Conscious Context Flags
    is_user_context BOOLEAN DEFAULT 0,  -- Contains user personal info
    is_preference BOOLEAN DEFAULT 0,    -- User preference/opinion
    is_skill_knowledge BOOLEAN DEFAULT 0,  -- User abilities/expertise
    is_current_project BOOLEAN DEFAULT 0,  -- Current work context
    promotion_eligible BOOLEAN DEFAULT 0,  -- Should be promoted to short-term

    -- Memory Management
    duplicate_of TEXT,  -- Links to original if duplicate
    supersedes_json TEXT DEFAULT '[]',  -- JSON array of memory IDs this replaces
    related_memories_json TEXT DEFAULT '[]',  -- JSON array of connected memory IDs
    
    -- Technical Metadata
    confidence_score REAL DEFAULT 0.8,  -- AI confidence in extraction
    extraction_timestamp TIMESTAMP NOT NULL,
    classification_reason TEXT,  -- Why this classification was chosen
    
    -- Processing Status
    processed_for_duplicates BOOLEAN DEFAULT 0,  -- Processed for duplicate detection
    conscious_processed BOOLEAN DEFAULT 0,  -- Processed for conscious context extraction
    x_coord REAL,
    y_coord REAL,
    z_coord REAL,
    symbolic_anchors JSON,
    embedding JSON,
    documents_json JSON,
    team_id TEXT,
    workspace_id TEXT,
    FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces (workspace_id) ON DELETE SET NULL
);

-- Rules Memory Table (legacy rule storage)
CREATE TABLE IF NOT EXISTS rules_memory (
    rule_id TEXT PRIMARY KEY,
    rule_text TEXT NOT NULL,
    rule_type TEXT NOT NULL,
    priority INTEGER DEFAULT 5,
    active BOOLEAN DEFAULT 1,
    context_conditions TEXT,
    namespace TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    processed_data TEXT,
    metadata TEXT DEFAULT '{}'
);

-- Extracted Entity Table
CREATE TABLE IF NOT EXISTS memory_entities (
    entity_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_value TEXT NOT NULL,
    relevance_score REAL NOT NULL DEFAULT 0.5,
    entity_context TEXT,
    namespace TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMP NOT NULL
);

-- Memory Relationship Table
CREATE TABLE IF NOT EXISTS memory_relationships (
    relationship_id TEXT PRIMARY KEY,
    source_memory_id TEXT NOT NULL,
    target_memory_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.5,
    reasoning TEXT,
    namespace TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMP NOT NULL
);

-- Memory Access Audit Table
CREATE TABLE IF NOT EXISTS memory_access_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    team_id TEXT,
    workspace_id TEXT,
    accessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    access_type TEXT NOT NULL DEFAULT 'retrieval',
    source TEXT,
    metadata JSON,
    FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces (workspace_id) ON DELETE SET NULL
);

-- Cluster Tables
CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    summary TEXT NOT NULL,
    centroid JSON,
    polarity REAL,
    subjectivity REAL,
    size INTEGER,
    avg_importance REAL,
    update_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP,
    weight REAL DEFAULT 0.0,
    total_tokens INTEGER DEFAULT 0,
    total_chars INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cluster_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    memory_id TEXT NOT NULL,
    anchor TEXT,
    summary TEXT NOT NULL,
    tokens INTEGER DEFAULT 0,
    chars INTEGER DEFAULT 0,
    FOREIGN KEY(cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);

-- Performance Indexes

-- Team Indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_team_slug ON teams(slug);
CREATE INDEX IF NOT EXISTS idx_team_created ON teams(created_at);

-- Workspace Indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_workspace_slug ON workspaces(slug);
CREATE INDEX IF NOT EXISTS idx_workspace_owner ON workspaces(owner_id);
CREATE INDEX IF NOT EXISTS idx_workspace_team ON workspaces(team_id);
CREATE INDEX IF NOT EXISTS idx_workspace_created ON workspaces(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS uq_workspace_member ON workspace_members(workspace_id, user_id);
CREATE INDEX IF NOT EXISTS idx_workspace_member_workspace ON workspace_members(workspace_id);
CREATE INDEX IF NOT EXISTS idx_workspace_member_user ON workspace_members(user_id);

-- Chat History Indexes
CREATE INDEX IF NOT EXISTS idx_chat_namespace_session ON chat_history(namespace, session_id);
CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_chat_model ON chat_history(model);
CREATE INDEX IF NOT EXISTS idx_chat_team ON chat_history(team_id);
CREATE INDEX IF NOT EXISTS idx_chat_workspace ON chat_history(workspace_id);
CREATE INDEX IF NOT EXISTS idx_chat_workspace_namespace ON chat_history(workspace_id, namespace);

-- Short-term Memory Indexes
CREATE INDEX IF NOT EXISTS idx_short_term_namespace ON short_term_memory(namespace);
CREATE INDEX IF NOT EXISTS idx_short_term_category ON short_term_memory(category_primary);
CREATE INDEX IF NOT EXISTS idx_short_term_importance ON short_term_memory(importance_score);
CREATE INDEX IF NOT EXISTS idx_short_term_expires ON short_term_memory(expires_at);
CREATE INDEX IF NOT EXISTS idx_short_term_created ON short_term_memory(created_at);
CREATE INDEX IF NOT EXISTS idx_short_term_searchable ON short_term_memory(searchable_content);
CREATE INDEX IF NOT EXISTS idx_short_term_access ON short_term_memory(access_count, last_accessed);
CREATE INDEX IF NOT EXISTS idx_short_term_permanent ON short_term_memory(is_permanent_context);
CREATE INDEX IF NOT EXISTS idx_short_term_team ON short_term_memory(team_id);
CREATE INDEX IF NOT EXISTS idx_short_term_namespace_team ON short_term_memory(namespace, team_id);
CREATE INDEX IF NOT EXISTS idx_short_term_workspace ON short_term_memory(workspace_id);
CREATE INDEX IF NOT EXISTS idx_short_term_namespace_workspace ON short_term_memory(namespace, workspace_id);

-- Long-term Memory Indexes  
CREATE INDEX IF NOT EXISTS idx_long_term_namespace ON long_term_memory(namespace);
CREATE INDEX IF NOT EXISTS idx_long_term_category ON long_term_memory(category_primary);
CREATE INDEX IF NOT EXISTS idx_long_term_importance ON long_term_memory(importance_score);
CREATE INDEX IF NOT EXISTS idx_long_term_created ON long_term_memory(created_at);
CREATE INDEX IF NOT EXISTS idx_long_term_timestamp ON long_term_memory(timestamp);
CREATE INDEX IF NOT EXISTS idx_long_term_searchable ON long_term_memory(searchable_content);
CREATE INDEX IF NOT EXISTS idx_long_term_access ON long_term_memory(access_count, last_accessed);
CREATE INDEX IF NOT EXISTS idx_long_term_scores ON long_term_memory(novelty_score, relevance_score, actionability_score);
CREATE INDEX IF NOT EXISTS idx_long_term_team ON long_term_memory(team_id);
CREATE INDEX IF NOT EXISTS idx_long_term_namespace_team ON long_term_memory(namespace, team_id);
CREATE INDEX IF NOT EXISTS idx_long_term_workspace ON long_term_memory(workspace_id);
CREATE INDEX IF NOT EXISTS idx_long_term_namespace_workspace ON long_term_memory(namespace, workspace_id);

-- Enhanced Classification Indexes
CREATE INDEX IF NOT EXISTS idx_long_term_classification ON long_term_memory(classification);
CREATE INDEX IF NOT EXISTS idx_long_term_memory_importance ON long_term_memory(memory_importance);
CREATE INDEX IF NOT EXISTS idx_long_term_topic ON long_term_memory(topic);
CREATE INDEX IF NOT EXISTS idx_long_term_conscious_flags ON long_term_memory(is_user_context, is_preference, is_skill_knowledge, promotion_eligible);
CREATE INDEX IF NOT EXISTS idx_long_term_conscious_processed ON long_term_memory(conscious_processed);
CREATE INDEX IF NOT EXISTS idx_long_term_duplicates ON long_term_memory(processed_for_duplicates);
CREATE INDEX IF NOT EXISTS idx_long_term_confidence ON long_term_memory(confidence_score);

-- Rules Memory Indexes
CREATE INDEX IF NOT EXISTS idx_rules_namespace ON rules_memory(namespace);
CREATE INDEX IF NOT EXISTS idx_rules_active ON rules_memory(active);
CREATE INDEX IF NOT EXISTS idx_rules_priority ON rules_memory(priority);
CREATE INDEX IF NOT EXISTS idx_rules_type ON rules_memory(rule_type);
CREATE INDEX IF NOT EXISTS idx_rules_updated ON rules_memory(updated_at);

-- Memory Entity Indexes
CREATE INDEX IF NOT EXISTS idx_entities_namespace ON memory_entities(namespace);
CREATE INDEX IF NOT EXISTS idx_entities_type ON memory_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_value ON memory_entities(entity_value);
CREATE INDEX IF NOT EXISTS idx_entities_memory ON memory_entities(memory_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_entities_relevance ON memory_entities(relevance_score);
CREATE INDEX IF NOT EXISTS idx_entities_value_type ON memory_entities(entity_value, entity_type);

-- Relationship Indexes
CREATE INDEX IF NOT EXISTS idx_relationships_source ON memory_relationships(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON memory_relationships(target_memory_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON memory_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationships_strength ON memory_relationships(strength);

-- Memory Access Indexes
CREATE INDEX IF NOT EXISTS idx_access_events_memory ON memory_access_events(memory_id);
CREATE INDEX IF NOT EXISTS idx_access_events_namespace ON memory_access_events(namespace, accessed_at);
CREATE INDEX IF NOT EXISTS idx_access_events_team_namespace ON memory_access_events(team_id, namespace, accessed_at);
CREATE INDEX IF NOT EXISTS idx_access_events_workspace_namespace ON memory_access_events(workspace_id, namespace, accessed_at);

-- Cluster Indexes
CREATE INDEX IF NOT EXISTS idx_clusters_weight ON clusters(weight);
CREATE INDEX IF NOT EXISTS idx_clusters_last_updated ON clusters(last_updated);
CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster ON cluster_members(cluster_id);

-- Full-Text Search Support (SQLite FTS5)
-- Enables advanced text search capabilities
CREATE VIRTUAL TABLE IF NOT EXISTS memory_search_fts USING fts5(
    memory_id,
    memory_type,
    namespace,
    searchable_content,
    summary,
    category_primary,
    symbolic_anchors,
    content='',
    contentless_delete=1
);

-- Triggers to maintain FTS index
CREATE TRIGGER IF NOT EXISTS short_term_memory_fts_insert AFTER INSERT ON short_term_memory
BEGIN
    INSERT INTO memory_search_fts(memory_id, memory_type, namespace, searchable_content, summary, category_primary, symbolic_anchors)
    VALUES (NEW.memory_id, 'short_term', NEW.namespace, NEW.searchable_content, NEW.summary, NEW.category_primary, NEW.symbolic_anchors);
END;

CREATE TRIGGER IF NOT EXISTS long_term_memory_fts_insert AFTER INSERT ON long_term_memory
BEGIN
    INSERT INTO memory_search_fts(memory_id, memory_type, namespace, searchable_content, summary, category_primary, symbolic_anchors)
    VALUES (NEW.memory_id, 'long_term', NEW.namespace, NEW.searchable_content, NEW.summary, NEW.category_primary, NEW.symbolic_anchors);
END;

CREATE TRIGGER IF NOT EXISTS short_term_memory_fts_delete AFTER DELETE ON short_term_memory
BEGIN
    DELETE FROM memory_search_fts WHERE memory_id = OLD.memory_id AND memory_type = 'short_term';
END;

CREATE TRIGGER IF NOT EXISTS long_term_memory_fts_delete AFTER DELETE ON long_term_memory
BEGIN
    DELETE FROM memory_search_fts WHERE memory_id = OLD.memory_id AND memory_type = 'long_term';
END;
