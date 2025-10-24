"""
MySQL schema generator for the Memoria 0.9 alpha (0.9.0a0) release.
Converts SQLite schema to MySQL-compatible schema with FULLTEXT search.
"""

from ..connectors.base_connector import BaseSchemaGenerator, DatabaseType


class MySQLSchemaGenerator(BaseSchemaGenerator):
    """MySQL-specific schema generator"""

    def __init__(self):
        super().__init__(DatabaseType.MYSQL)

    def get_data_type_mappings(self) -> dict[str, str]:
        """Get MySQL-specific data type mappings from SQLite"""
        return {
            "TEXT": "TEXT",
            "INTEGER": "INT",
            "REAL": "DECIMAL(10,2)",
            "BOOLEAN": "BOOLEAN",
            "TIMESTAMP": "TIMESTAMP",
            "AUTOINCREMENT": "AUTO_INCREMENT",
        }

    def generate_core_schema(self) -> str:
        """Generate core tables schema for MySQL"""
        return """
-- Team Table
CREATE TABLE IF NOT EXISTS teams (
    team_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Workspace Table
CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id VARCHAR(255),
    team_id VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Workspace Membership Table
CREATE TABLE IF NOT EXISTS workspace_members (
    id INT AUTO_INCREMENT PRIMARY KEY,
    workspace_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'member',
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    is_agent BOOLEAN NOT NULL DEFAULT FALSE,
    preferred_model VARCHAR(255),
    last_edited_by_model VARCHAR(255),
    joined_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (workspace_id) REFERENCES workspaces (workspace_id) ON DELETE CASCADE,
    UNIQUE KEY uq_workspace_member (workspace_id, user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Chat History Table
CREATE TABLE IF NOT EXISTS chat_history (
    chat_id VARCHAR(255) PRIMARY KEY,
    user_input TEXT NOT NULL,
    ai_output TEXT NOT NULL,
    model VARCHAR(255) NOT NULL,
    last_edited_by_model VARCHAR(255),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(255) NOT NULL,
    namespace VARCHAR(255) NOT NULL DEFAULT 'default',
    tokens_used INT DEFAULT 0,
    metadata JSON,
    team_id VARCHAR(255),
    workspace_id VARCHAR(255),
    FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces (workspace_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Short-term Memory Table
CREATE TABLE IF NOT EXISTS short_term_memory (
    memory_id VARCHAR(255) PRIMARY KEY,
    chat_id VARCHAR(255),
    processed_data JSON NOT NULL,
    importance_score DECIMAL(3,2) NOT NULL DEFAULT 0.5,
    category_primary VARCHAR(255) NOT NULL,
    retention_type VARCHAR(50) NOT NULL DEFAULT 'short_term',
    namespace VARCHAR(255) NOT NULL DEFAULT 'default',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NULL,
    access_count INT DEFAULT 0,
    last_accessed TIMESTAMP NULL,
    searchable_content TEXT NOT NULL,
    summary TEXT NOT NULL,
    is_permanent_context BOOLEAN DEFAULT FALSE,
    x_coord DECIMAL(10,2),
    y_coord DECIMAL(10,2),
    z_coord DECIMAL(10,2),
    symbolic_anchors JSON,
    embedding JSON,
    last_edited_by_model VARCHAR(255),
    team_id VARCHAR(255),
    workspace_id VARCHAR(255),
    INDEX idx_chat_id (chat_id),
    FOREIGN KEY (chat_id) REFERENCES chat_history (chat_id) ON DELETE SET NULL,
    FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces (workspace_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Long-term Memory Table
CREATE TABLE IF NOT EXISTS long_term_memory (
    memory_id VARCHAR(255) PRIMARY KEY,
    original_chat_id VARCHAR(255),
    processed_data JSON NOT NULL,
    importance_score DECIMAL(3,2) NOT NULL DEFAULT 0.5,
    category_primary VARCHAR(255) NOT NULL,
    retention_type VARCHAR(50) NOT NULL DEFAULT 'long_term',
    namespace VARCHAR(255) NOT NULL DEFAULT 'default',
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    access_count INT DEFAULT 0,
    last_accessed TIMESTAMP NULL,
    searchable_content TEXT NOT NULL,
    summary TEXT NOT NULL,
    novelty_score DECIMAL(3,2) DEFAULT 0.5,
    relevance_score DECIMAL(3,2) DEFAULT 0.5,
    actionability_score DECIMAL(3,2) DEFAULT 0.5,
    last_edited_by_model VARCHAR(255),

    -- Enhanced Classification Fields
    classification VARCHAR(50) NOT NULL DEFAULT 'conversational',
    memory_importance VARCHAR(20) NOT NULL DEFAULT 'medium',
    topic VARCHAR(255),
    entities_json JSON DEFAULT (JSON_ARRAY()),
    keywords_json JSON DEFAULT (JSON_ARRAY()),

    -- Conscious Context Flags
    is_user_context BOOLEAN DEFAULT FALSE,
    is_preference BOOLEAN DEFAULT FALSE,
    is_skill_knowledge BOOLEAN DEFAULT FALSE,
    is_current_project BOOLEAN DEFAULT FALSE,
    promotion_eligible BOOLEAN DEFAULT FALSE,

    -- Memory Management
    duplicate_of VARCHAR(255),
    supersedes_json JSON DEFAULT (JSON_ARRAY()),
    related_memories_json JSON DEFAULT (JSON_ARRAY()),

    -- Technical Metadata
    confidence_score DECIMAL(3,2) DEFAULT 0.8,
    extraction_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    classification_reason TEXT,

    -- Processing Status
    processed_for_duplicates BOOLEAN DEFAULT FALSE,
    conscious_processed BOOLEAN DEFAULT FALSE,
    x_coord DECIMAL(10,2),
    y_coord DECIMAL(10,2),
    z_coord DECIMAL(10,2),
    symbolic_anchors JSON,
    embedding JSON,
    documents_json JSON,
    team_id VARCHAR(255),
    workspace_id VARCHAR(255),
    FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces (workspace_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS memory_access_events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    memory_id VARCHAR(255) NOT NULL,
    namespace VARCHAR(255) NOT NULL DEFAULT 'default',
    team_id VARCHAR(255),
    workspace_id VARCHAR(255),
    accessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    access_type VARCHAR(50) NOT NULL DEFAULT 'retrieval',
    source VARCHAR(50),
    metadata JSON,
    INDEX idx_access_events_memory (memory_id),
    INDEX idx_access_events_namespace (namespace, accessed_at),
    INDEX idx_access_events_team_namespace (team_id, namespace, accessed_at),
    INDEX idx_access_events_workspace_namespace (workspace_id, namespace, accessed_at),
    FOREIGN KEY (team_id) REFERENCES teams (team_id) ON DELETE SET NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces (workspace_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Cluster Tables
CREATE TABLE IF NOT EXISTS clusters (
    id INT AUTO_INCREMENT PRIMARY KEY,
    summary TEXT NOT NULL,
    centroid JSON,
    polarity DECIMAL(3,2),
    subjectivity DECIMAL(3,2),
    size INT,
    avg_importance DECIMAL(3,2),
    update_count INT DEFAULT 0,
    last_updated TIMESTAMP NULL,
    weight DECIMAL(10,2) DEFAULT 0.0,
    total_tokens INT DEFAULT 0,
    total_chars INT DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS cluster_members (
    id INT AUTO_INCREMENT PRIMARY KEY,
    cluster_id INT NOT NULL,
    memory_id VARCHAR(255) NOT NULL,
    anchor VARCHAR(255),
    summary TEXT NOT NULL,
    tokens INT DEFAULT 0,
    chars INT DEFAULT 0,
    INDEX idx_cluster_id (cluster_id),
    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

    def generate_indexes(self) -> str:
        """Generate MySQL-specific indexes"""
        return """
-- Team Indexes
CREATE UNIQUE INDEX idx_team_slug ON teams(slug);
CREATE INDEX idx_team_created ON teams(created_at);

-- Workspace Indexes
CREATE UNIQUE INDEX idx_workspace_slug ON workspaces(slug);
CREATE INDEX idx_workspace_owner ON workspaces(owner_id);
CREATE INDEX idx_workspace_team ON workspaces(team_id);
CREATE INDEX idx_workspace_created ON workspaces(created_at);
CREATE UNIQUE INDEX uq_workspace_member ON workspace_members(workspace_id, user_id);
CREATE INDEX idx_workspace_member_workspace ON workspace_members(workspace_id);
CREATE INDEX idx_workspace_member_user ON workspace_members(user_id);

-- Chat History Indexes
CREATE INDEX idx_chat_namespace_session ON chat_history(namespace, session_id);
CREATE INDEX idx_chat_timestamp ON chat_history(timestamp);
CREATE INDEX idx_chat_model ON chat_history(model);
CREATE INDEX idx_chat_team ON chat_history(team_id);
CREATE INDEX idx_chat_workspace ON chat_history(workspace_id);
CREATE INDEX idx_chat_workspace_namespace ON chat_history(workspace_id, namespace);

-- Short-term Memory Indexes
CREATE INDEX idx_short_term_namespace ON short_term_memory(namespace);
CREATE INDEX idx_short_term_category ON short_term_memory(category_primary);
CREATE INDEX idx_short_term_importance ON short_term_memory(importance_score);
CREATE INDEX idx_short_term_expires ON short_term_memory(expires_at);
CREATE INDEX idx_short_term_created ON short_term_memory(created_at);
CREATE INDEX idx_short_term_access ON short_term_memory(access_count, last_accessed);
CREATE INDEX idx_short_term_permanent ON short_term_memory(is_permanent_context);
CREATE INDEX idx_short_term_team ON short_term_memory(team_id);
CREATE INDEX idx_short_term_namespace_team ON short_term_memory(namespace, team_id);
CREATE INDEX idx_short_term_workspace ON short_term_memory(workspace_id);
CREATE INDEX idx_short_term_namespace_workspace ON short_term_memory(namespace, workspace_id);

-- Long-term Memory Indexes
CREATE INDEX idx_long_term_namespace ON long_term_memory(namespace);
CREATE INDEX idx_long_term_category ON long_term_memory(category_primary);
CREATE INDEX idx_long_term_importance ON long_term_memory(importance_score);
CREATE INDEX idx_long_term_created ON long_term_memory(created_at);
CREATE INDEX idx_long_term_timestamp ON long_term_memory(timestamp);
CREATE INDEX idx_long_term_access ON long_term_memory(access_count, last_accessed);
CREATE INDEX idx_long_term_scores ON long_term_memory(novelty_score, relevance_score, actionability_score);
CREATE INDEX idx_long_term_team ON long_term_memory(team_id);
CREATE INDEX idx_long_term_namespace_team ON long_term_memory(namespace, team_id);
CREATE INDEX idx_long_term_workspace ON long_term_memory(workspace_id);
CREATE INDEX idx_long_term_namespace_workspace ON long_term_memory(namespace, workspace_id);

-- Enhanced Classification Indexes
CREATE INDEX idx_long_term_classification ON long_term_memory(classification);
CREATE INDEX idx_long_term_memory_importance ON long_term_memory(memory_importance);
CREATE INDEX idx_long_term_topic ON long_term_memory(topic);
CREATE INDEX idx_long_term_conscious_flags ON long_term_memory(is_user_context, is_preference, is_skill_knowledge, promotion_eligible);
CREATE INDEX idx_long_term_conscious_processed ON long_term_memory(conscious_processed);
CREATE INDEX idx_long_term_duplicates ON long_term_memory(processed_for_duplicates);
CREATE INDEX idx_long_term_confidence ON long_term_memory(confidence_score);

-- Cluster Indexes
CREATE INDEX idx_clusters_weight ON clusters(weight);
CREATE INDEX idx_clusters_last_updated ON clusters(last_updated);

-- Composite indexes for search optimization
CREATE INDEX idx_short_term_namespace_category_importance ON short_term_memory(namespace, category_primary, importance_score);
CREATE INDEX idx_long_term_namespace_category_importance ON long_term_memory(namespace, category_primary, importance_score);
"""

    def generate_search_setup(self) -> str:
        """Generate MySQL FULLTEXT search setup"""
        return """
-- MySQL FULLTEXT Search Indexes
-- These replace SQLite's FTS5 virtual table with MySQL's native FULLTEXT indexes

-- FULLTEXT index for short-term memory

-- FULLTEXT index for long-term memory
ALTER TABLE short_term_memory ADD FULLTEXT INDEX ft_short_term_search (searchable_content, summary, symbolic_anchors);
ALTER TABLE long_term_memory ADD FULLTEXT INDEX ft_long_term_search (searchable_content, summary, symbolic_anchors);

-- Additional FULLTEXT indexes for enhanced search capabilities
ALTER TABLE long_term_memory ADD FULLTEXT INDEX ft_long_term_topic (topic);

-- Note: MySQL FULLTEXT indexes are maintained automatically
-- No triggers needed like SQLite FTS5
"""

    def generate_mysql_specific_optimizations(self) -> str:
        """Generate MySQL-specific optimizations"""
        return """
-- MySQL-specific optimizations

-- Set InnoDB buffer pool size (if you have permission)
-- SET GLOBAL innodb_buffer_pool_size = 268435456;  -- 256MB

-- Optimize FULLTEXT search settings
-- SET GLOBAL ft_min_word_len = 3;
-- SET GLOBAL ft_boolean_syntax = ' +-><()~*:""&|';

-- Enable query cache (MySQL 5.7 and below)
-- SET GLOBAL query_cache_type = ON;
-- SET GLOBAL query_cache_size = 67108864;  -- 64MB
"""

    def generate_full_schema(self) -> str:
        """Generate complete MySQL schema"""
        schema_parts = [
            "-- Memoria 0.9 alpha (0.9.0a0) MySQL Schema",
            "-- Generated schema for cross-database compatibility",
            "",
            "-- Set MySQL session variables for optimal performance",
            "SET SESSION sql_mode = 'STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO';",
            "SET SESSION innodb_lock_wait_timeout = 30;",
            "",
            self.generate_core_schema(),
            "",
            self.generate_indexes(),
            "",
            self.generate_search_setup(),
            "",
            "-- Schema generation completed",
        ]
        return "\n".join(schema_parts)

    def get_migration_queries(self) -> list[str]:
        """Get queries to migrate from SQLite to MySQL"""
        return [
            "-- Memoria ships JSON/NDJSON exporters instead of raw SQL migration scripts.",
            "-- Use `memoria export-data backup.json --database-url <sqlite_url>`",
            "-- followed by `memoria import-data backup.json --database-url <mysql_url>`.",
            "-- See memoria.database.migration for details on supported tables.",
        ]
