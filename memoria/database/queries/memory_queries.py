"""
Memory-related database queries
"""

from .base_queries import BaseQueries


class MemoryQueries(BaseQueries):
    """Centralized memory-related SQL queries"""

    def get_table_creation_queries(self) -> dict[str, str]:
        """Memory table creation queries"""
        from .base_queries import SchemaQueries

        return {
            "short_term_memory": SchemaQueries.TABLE_CREATION["short_term_memory"],
            "long_term_memory": SchemaQueries.TABLE_CREATION["long_term_memory"],
            "rules_memory": SchemaQueries.TABLE_CREATION["rules_memory"],
        }

    def get_index_creation_queries(self) -> dict[str, str]:
        """Memory index creation queries"""
        from .base_queries import SchemaQueries

        return {
            k: v
            for k, v in SchemaQueries.INDEX_CREATION.items()
            if any(table in k for table in ["short_term", "long_term", "rules"])
        }

    def get_trigger_creation_queries(self) -> dict[str, str]:
        """Memory trigger creation queries"""
        from .base_queries import SchemaQueries

        return SchemaQueries.TRIGGER_CREATION

    # INSERT Queries
    INSERT_SHORT_TERM_MEMORY = """
        INSERT INTO short_term_memory (
            memory_id, chat_id, processed_data, importance_score, category_primary,
            retention_type, namespace, created_at, expires_at, searchable_content, summary,
            x_coord, y_coord, z_coord, symbolic_anchors
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    INSERT_LONG_TERM_MEMORY = """
        INSERT INTO long_term_memory (
            memory_id, original_chat_id, processed_data, importance_score, category_primary,
            retention_type, namespace, timestamp, created_at, searchable_content, summary,
            novelty_score, relevance_score, actionability_score, x_coord, y_coord, z_coord, symbolic_anchors,
            classification, memory_importance, topic, entities_json, keywords_json,
            is_user_context, is_preference, is_skill_knowledge, is_current_project, promotion_eligible,
            duplicate_of, supersedes_json, related_memories_json,
            confidence_score, extraction_timestamp, classification_reason,
            processed_for_duplicates, conscious_processed, documents_json,
            image_assets_json, includes_image
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    INSERT_RULES_MEMORY = """
        INSERT INTO rules_memory (
            rule_id, rule_text, rule_type, priority, active, context_conditions,
            namespace, created_at, updated_at, processed_data, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    # SELECT Queries
    SELECT_MEMORIES_BY_NAMESPACE = """
        SELECT memory_id, processed_data, importance_score, category_primary, created_at, summary, x_coord, y_coord, z_coord, symbolic_anchors
        FROM {table}
        WHERE namespace = ?
        ORDER BY importance_score DESC, created_at DESC
        LIMIT ?
    """

    SELECT_MEMORIES_BY_CATEGORY = """
        SELECT memory_id, processed_data, importance_score, created_at, summary, x_coord, y_coord, z_coord, symbolic_anchors
        FROM {table}
        WHERE namespace = ? AND category_primary = ?
        ORDER BY importance_score DESC, created_at DESC
        LIMIT ?
    """

    SELECT_MEMORIES_BY_IMPORTANCE = """
        SELECT memory_id, processed_data, importance_score, created_at, summary, x_coord, y_coord, z_coord, symbolic_anchors
        FROM {table}
        WHERE namespace = ? AND importance_score >= ?
        ORDER BY importance_score DESC, created_at DESC
        LIMIT ?
    """

    SELECT_EXPIRED_MEMORIES = """
        SELECT memory_id, processed_data
        FROM short_term_memory
        WHERE namespace = ? AND expires_at <= ?
    """

    SELECT_MEMORY_BY_ID = """
        SELECT * FROM {table} WHERE memory_id = ? AND namespace = ?
    """

    # UPDATE Queries
    UPDATE_MEMORY_ACCESS = """
        UPDATE {table}
        SET access_count = access_count + 1, last_accessed = ?
        WHERE memory_id = ? AND namespace = ?
    """

    UPDATE_MEMORY_IMPORTANCE = """
        UPDATE {table}
        SET importance_score = ?
        WHERE memory_id = ? AND namespace = ?
    """

    UPDATE_RULE_STATUS = """
        UPDATE rules_memory
        SET active = ?, updated_at = ?
        WHERE rule_id = ? AND namespace = ?
    """

    # DELETE Queries
    DELETE_MEMORY = """
        DELETE FROM {table} WHERE memory_id = ? AND namespace = ?
    """

    DELETE_EXPIRED_MEMORIES = """
        DELETE FROM short_term_memory
        WHERE namespace = ? AND expires_at <= ?
    """

    DELETE_MEMORIES_BY_CATEGORY = """
        DELETE FROM {table}
        WHERE namespace = ? AND category_primary = ?
    """

    # SEARCH Queries
    SEARCH_MEMORIES_FTS = """
        SELECT m.memory_id, m.memory_type, m.namespace, m.searchable_content, m.summary, m.category_primary
        FROM memory_search_fts m
        WHERE m.searchable_content MATCH ? AND m.namespace = ?
        ORDER BY rank
        LIMIT ?
    """

    SEARCH_MEMORIES_SEMANTIC = """
        SELECT memory_id, processed_data, importance_score, searchable_content, summary, x_coord, y_coord, z_coord, symbolic_anchors
        FROM {table}
        WHERE namespace = ? AND (
            searchable_content LIKE ? OR
            summary LIKE ? OR
            category_primary = ?
        )
        ORDER BY importance_score DESC, created_at DESC
        LIMIT ?
    """

    # ANALYTICS Queries
    COUNT_MEMORIES_BY_CATEGORY = """
        SELECT category_primary, COUNT(*) as count
        FROM {table}
        WHERE namespace = ?
        GROUP BY category_primary
        ORDER BY count DESC
    """

    GET_MEMORY_STATISTICS = """
        SELECT
            COUNT(*) as total_memories,
            AVG(importance_score) as avg_importance,
            MAX(importance_score) as max_importance,
            MIN(importance_score) as min_importance,
            COUNT(DISTINCT category_primary) as unique_categories
        FROM {table}
        WHERE namespace = ?
    """

    GET_RECENT_MEMORIES = """
        SELECT memory_id, summary, importance_score, created_at
        FROM {table}
        WHERE namespace = ? AND created_at >= ?
        ORDER BY created_at DESC
        LIMIT ?
    """

    # Conscious Context Queries
    SELECT_CONSCIOUS_MEMORIES = """
        SELECT memory_id, processed_data, summary, classification, importance_score,
               is_user_context, is_preference, is_skill_knowledge, is_current_project,
               promotion_eligible, created_at
        FROM long_term_memory
        WHERE namespace = ?
        AND (
            classification = 'conscious-info'
            OR promotion_eligible = ?
            OR is_user_context = ?
        )
        ORDER BY importance_score DESC, extraction_timestamp DESC
    """

    SELECT_UNPROCESSED_CONSCIOUS = """
        SELECT memory_id, processed_data, classification, is_user_context, promotion_eligible
        FROM long_term_memory
        WHERE namespace = ? AND conscious_processed = ?
        AND (classification = 'conscious-info' OR promotion_eligible = ? OR is_user_context = ?)
    """

    SELECT_USER_CONTEXT_PROFILE = """
        SELECT processed_data FROM short_term_memory
        WHERE namespace = ? AND is_permanent_context = ?
        AND category_primary = 'user_context'
    """

    INSERT_USER_CONTEXT_PROFILE = """
        INSERT OR REPLACE INTO short_term_memory (
            memory_id, processed_data, importance_score, category_primary,
            retention_type, namespace, created_at, expires_at,
            searchable_content, summary, is_permanent_context
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    MARK_CONSCIOUS_PROCESSED = """
        UPDATE long_term_memory
        SET conscious_processed = ?
        WHERE memory_id = ? AND namespace = ?
    """

    # Classification and Filtering Queries
    SELECT_MEMORIES_BY_CLASSIFICATION = """
        SELECT memory_id, processed_data, importance_score, classification, created_at, summary
        FROM long_term_memory
        WHERE namespace = ? AND classification = ?
        ORDER BY importance_score DESC, created_at DESC
        LIMIT ?
    """

    SELECT_MEMORIES_FOR_DEDUPLICATION = """
        SELECT memory_id, summary, searchable_content, classification, created_at
        FROM long_term_memory
        WHERE namespace = :namespace
          AND processed_for_duplicates = :processed_for_duplicates
          AND (:workspace_id IS NULL OR workspace_id = :workspace_id)
        ORDER BY created_at DESC
        LIMIT :limit
    """

    UPDATE_DUPLICATE_STATUS = """
        UPDATE long_term_memory
        SET duplicate_of = ?, processed_for_duplicates = ?
        WHERE memory_id = ? AND namespace = ?
    """

    SELECT_PROMOTION_ELIGIBLE_MEMORIES = """
        SELECT memory_id, processed_data, summary, classification
        FROM long_term_memory
        WHERE namespace = ? AND promotion_eligible = ?
        AND conscious_processed = ?
    """

    # Performance Queries
    SELECT_MEMORIES_WITH_CONTEXT_FLAGS = """
        SELECT memory_id, processed_data, classification,
               is_user_context, is_preference, is_skill_knowledge, is_current_project,
               confidence_score, created_at
        FROM long_term_memory
        WHERE namespace = ?
        AND (is_user_context = ? OR is_preference = ? OR is_skill_knowledge = ? OR is_current_project = ?)
        ORDER BY importance_score DESC, created_at DESC
        LIMIT ?
    """
