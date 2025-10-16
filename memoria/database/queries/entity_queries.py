"""
Entity and relationship database queries
"""

from .base_queries import BaseQueries


class EntityQueries(BaseQueries):
    """Centralized entity and relationship SQL queries"""

    def get_table_creation_queries(self) -> dict[str, str]:
        """Entity table creation queries"""
        from .base_queries import SchemaQueries

        return {
            "memory_entities": SchemaQueries.TABLE_CREATION["memory_entities"],
            "memory_relationships": SchemaQueries.TABLE_CREATION[
                "memory_relationships"
            ],
        }

    def get_index_creation_queries(self) -> dict[str, str]:
        """Entity index creation queries"""
        from .base_queries import SchemaQueries

        return {
            k: v
            for k, v in SchemaQueries.INDEX_CREATION.items()
            if any(word in k for word in ["entities", "relationships"])
        }

    def get_trigger_creation_queries(self) -> dict[str, str]:
        """Entity trigger creation queries"""
        return {}  # No triggers for entities currently

    # ENTITY INSERT Queries
    INSERT_MEMORY_ENTITY = """
        INSERT INTO memory_entities (
            entity_id, memory_id, memory_type, entity_type, entity_value,
            relevance_score, entity_context, namespace, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    BATCH_INSERT_ENTITIES = """
        INSERT INTO memory_entities (
            entity_id, memory_id, memory_type, entity_type, entity_value,
            relevance_score, entity_context, namespace, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    # RELATIONSHIP INSERT Queries
    INSERT_MEMORY_RELATIONSHIP = """
        INSERT INTO memory_relationships (
            relationship_id, source_memory_id, target_memory_id, relationship_type,
            strength, reasoning, namespace, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    # ENTITY SELECT Queries
    SELECT_ENTITIES_BY_MEMORY = """
        SELECT entity_id, entity_type, entity_value, relevance_score, entity_context
        FROM memory_entities
        WHERE memory_id = ? AND namespace = ?
        ORDER BY relevance_score DESC
    """

    SELECT_ENTITIES_BY_TYPE = """
        SELECT entity_id, memory_id, memory_type, entity_value, relevance_score, entity_context
        FROM memory_entities
        WHERE entity_type = ? AND namespace = ?
        ORDER BY relevance_score DESC
        LIMIT ?
    """

    SELECT_ENTITIES_BY_VALUE = """
        SELECT entity_id, memory_id, memory_type, entity_type, relevance_score, entity_context
        FROM memory_entities
        WHERE entity_value = ? AND namespace = ?
        ORDER BY relevance_score DESC
        LIMIT ?
    """

    SELECT_SIMILAR_ENTITIES = """
        SELECT entity_id, memory_id, memory_type, entity_type, entity_value, relevance_score
        FROM memory_entities
        WHERE entity_value LIKE ? AND entity_type = ? AND namespace = ?
        ORDER BY relevance_score DESC
        LIMIT ?
    """

    SELECT_ENTITIES_BY_MEMORY_TYPE = """
        SELECT entity_id, entity_type, entity_value, relevance_score, memory_id
        FROM memory_entities
        WHERE memory_type = ? AND namespace = ?
        ORDER BY relevance_score DESC
        LIMIT ?
    """

    # RELATIONSHIP SELECT Queries
    SELECT_RELATIONSHIPS_BY_MEMORY = """
        SELECT relationship_id, target_memory_id, relationship_type, strength, reasoning
        FROM memory_relationships
        WHERE source_memory_id = ? AND namespace = ?
        ORDER BY strength DESC
    """

    SELECT_RELATED_MEMORIES = """
        SELECT DISTINCT target_memory_id as memory_id, relationship_type, strength, reasoning
        FROM memory_relationships
        WHERE source_memory_id = ? AND namespace = ?
        UNION
        SELECT DISTINCT source_memory_id as memory_id, relationship_type, strength, reasoning
        FROM memory_relationships
        WHERE target_memory_id = ? AND namespace = ?
        ORDER BY strength DESC
        LIMIT ?
    """

    SELECT_RELATIONSHIPS_BY_TYPE = """
        SELECT relationship_id, source_memory_id, target_memory_id, strength, reasoning
        FROM memory_relationships
        WHERE relationship_type = ? AND namespace = ?
        ORDER BY strength DESC
        LIMIT ?
    """

    # UPDATE Queries
    UPDATE_ENTITY_RELEVANCE = """
        UPDATE memory_entities
        SET relevance_score = ?
        WHERE entity_id = ? AND namespace = ?
    """

    UPDATE_RELATIONSHIP_STRENGTH = """
        UPDATE memory_relationships
        SET strength = ?, reasoning = ?
        WHERE relationship_id = ? AND namespace = ?
    """

    # DELETE Queries
    DELETE_ENTITIES_BY_MEMORY = """
        DELETE FROM memory_entities
        WHERE memory_id = ? AND namespace = ?
    """

    DELETE_RELATIONSHIPS_BY_MEMORY = """
        DELETE FROM memory_relationships
        WHERE (source_memory_id = ? OR target_memory_id = ?) AND namespace = ?
    """

    DELETE_ENTITY = """
        DELETE FROM memory_entities
        WHERE entity_id = ? AND namespace = ?
    """

    DELETE_RELATIONSHIP = """
        DELETE FROM memory_relationships
        WHERE relationship_id = ? AND namespace = ?
    """

    # ANALYTICS Queries
    GET_ENTITY_STATISTICS = """
        SELECT
            entity_type,
            COUNT(*) as count,
            AVG(relevance_score) as avg_relevance,
            MAX(relevance_score) as max_relevance
        FROM memory_entities
        WHERE namespace = ?
        GROUP BY entity_type
        ORDER BY count DESC
    """

    GET_TOP_ENTITIES = """
        SELECT entity_value, entity_type, COUNT(*) as frequency, AVG(relevance_score) as avg_relevance
        FROM memory_entities
        WHERE namespace = ? AND entity_type = ?
        GROUP BY entity_value, entity_type
        ORDER BY frequency DESC, avg_relevance DESC
        LIMIT ?
    """

    GET_RELATIONSHIP_STATISTICS = """
        SELECT
            relationship_type,
            COUNT(*) as count,
            AVG(strength) as avg_strength,
            MAX(strength) as max_strength
        FROM memory_relationships
        WHERE namespace = ?
        GROUP BY relationship_type
        ORDER BY count DESC
    """

    GET_MEMORY_CONNECTIVITY = """
        SELECT
            memory_id,
            memory_type,
            COUNT(*) as connection_count
        FROM (
            SELECT source_memory_id as memory_id, 'outgoing' as memory_type
            FROM memory_relationships
            WHERE namespace = ?
            UNION ALL
            SELECT target_memory_id as memory_id, 'incoming' as memory_type
            FROM memory_relationships
            WHERE namespace = ?
        ) as connections
        GROUP BY memory_id
        ORDER BY connection_count DESC
        LIMIT ?
    """

    # SEARCH Queries
    SEARCH_ENTITIES = """
        SELECT entity_id, memory_id, memory_type, entity_type, entity_value, relevance_score
        FROM memory_entities
        WHERE namespace = ? AND (
            entity_value LIKE ? OR
            entity_context LIKE ?
        )
        ORDER BY relevance_score DESC
        LIMIT ?
    """

    FIND_ENTITY_CLUSTERS = """
        SELECT entity_value, entity_type, COUNT(DISTINCT memory_id) as memory_count
        FROM memory_entities
        WHERE namespace = ? AND entity_type = ?
        GROUP BY entity_value, entity_type
        HAVING memory_count > ?
        ORDER BY memory_count DESC
        LIMIT ?
    """
