"""
Unified query builder with database-agnostic parameter binding
Provides consistent parameter handling across SQLite, PostgreSQL, and MySQL
"""

import re
from enum import Enum
from typing import Any

from loguru import logger

from .exceptions import DatabaseError, ValidationError
from .input_validator import InputValidator


class DatabaseDialect(str, Enum):
    """Supported database dialects"""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class QueryBuilder:
    """Database-agnostic query builder with proper parameter binding"""

    # Parameter styles for different databases
    PARAM_STYLES = {
        DatabaseDialect.SQLITE: "?",
        DatabaseDialect.POSTGRESQL: "%s",
        DatabaseDialect.MYSQL: "%s",
    }

    # SQL keywords that need special handling per database
    LIMIT_SYNTAX = {
        DatabaseDialect.SQLITE: "LIMIT ?",
        DatabaseDialect.POSTGRESQL: "LIMIT %s",
        DatabaseDialect.MYSQL: "LIMIT %s",
    }

    def __init__(self, dialect: DatabaseDialect):
        self.dialect = dialect
        self.param_placeholder = self.PARAM_STYLES[dialect]

    def _parse_anchor_expression(self, expression: str) -> tuple[str, list[Any]]:
        """Parse boolean anchor expression into SQL condition and parameters."""
        tokens = re.findall(
            r"\(|\)|AND|OR|NOT|[^\s()]+", expression, flags=re.IGNORECASE
        )
        if not tokens:
            raise ValidationError("Empty anchor expression")

        precedence = {"OR": 1, "AND": 2, "NOT": 3}
        output: list[Any] = []
        operators: list[str] = []

        for token in tokens:
            upper = token.upper()
            if upper in precedence:
                while (
                    operators
                    and operators[-1] != "("
                    and precedence.get(operators[-1], 0) >= precedence[upper]
                ):
                    output.append(operators.pop())
                operators.append(upper)
            elif token == "(":
                operators.append(token)
            elif token == ")":
                while operators and operators[-1] != "(":
                    output.append(operators.pop())
                if not operators:
                    raise ValidationError("Mismatched parentheses in anchor expression")
                operators.pop()
            else:
                anchor = InputValidator.validate_and_sanitize_query(token)
                output.append({"ANCHOR": anchor})

        while operators:
            op = operators.pop()
            if op in ("(", ")"):
                raise ValidationError("Mismatched parentheses in anchor expression")
            output.append(op)

        stack: list[tuple[str, list[Any]]] = []
        for item in output:
            if isinstance(item, dict):
                anchor = item["ANCHOR"]
                stack.append(
                    (f"symbolic_anchors LIKE {self.param_placeholder}", [f"%{anchor}%"])
                )
            else:
                if item == "NOT":
                    if not stack:
                        raise ValidationError(
                            "Invalid NOT placement in anchor expression"
                        )
                    sql, params = stack.pop()
                    stack.append((f"NOT ({sql})", params))
                else:
                    if len(stack) < 2:
                        raise ValidationError("Invalid boolean expression")
                    right_sql, right_params = stack.pop()
                    left_sql, left_params = stack.pop()
                    if item == "AND":
                        combined_sql = f"({left_sql} AND {right_sql})"
                    else:
                        combined_sql = f"({left_sql} OR {right_sql})"
                    stack.append((combined_sql, left_params + right_params))

        if len(stack) != 1:
            raise ValidationError("Invalid anchor expression")
        return stack[0]

    def build_search_query(
        self,
        tables: list[str],
        search_columns: list[str],
        query_text: str,
        namespace: str,
        category_filter: list[str] | None = None,
        limit: int = 10,
        use_fts: bool = False,
        time_range: tuple[Any, Any] | None = None,
        importance_min: float | None = None,
        coordinates: dict[str, tuple[float, float]] | None = None,
        anchor_expression: str | None = None,
        privacy_range: tuple[float, float] | None = None,
    ) -> tuple[str, list[Any]]:
        """Build a database-specific search query with proper parameter binding"""

        try:
            # Validate inputs using our validator
            query_text = InputValidator.validate_and_sanitize_query(query_text)
            namespace = InputValidator.validate_namespace(namespace)
            category_filter = InputValidator.validate_category_filter(category_filter)
            limit = InputValidator.validate_limit(limit)

            # Validate table and column names
            for table in tables:
                InputValidator.sanitize_sql_identifier(table)
            for column in search_columns:
                InputValidator.sanitize_sql_identifier(column)

        except ValidationError as e:
            raise DatabaseError(f"Invalid query parameters: {e}")

        params = []
        where_conditions = []

        # Build FTS-specific or LIKE-based search conditions
        if use_fts and self.dialect == DatabaseDialect.SQLITE:
            # SQLite FTS5 syntax
            fts_condition = f"memory_search_fts MATCH {self.param_placeholder}"
            params.append(f'"{query_text}"' if query_text else "*")
            where_conditions.append(fts_condition)
        elif use_fts and self.dialect == DatabaseDialect.POSTGRESQL:
            # PostgreSQL full-text search
            search_conditions = []
            for column in search_columns:
                search_conditions.append(
                    f"to_tsvector('english', {column}) @@ plainto_tsquery('english', {self.param_placeholder})"
                )
                params.append(query_text)
            where_conditions.append(f"({' OR '.join(search_conditions)})")
        elif use_fts and self.dialect == DatabaseDialect.MYSQL:
            # MySQL FULLTEXT search
            search_conditions = []
            for column in search_columns:
                search_conditions.append(
                    f"MATCH({column}) AGAINST ({self.param_placeholder} IN BOOLEAN MODE)"
                )
                params.append(query_text)
            where_conditions.append(f"({' OR '.join(search_conditions)})")
        else:
            # Fallback LIKE search for all databases
            like_conditions = []
            for column in search_columns:
                like_conditions.append(f"{column} LIKE {self.param_placeholder}")
                params.append(f"%{query_text}%")
            where_conditions.append(f"({' OR '.join(like_conditions)})")

        # Add namespace condition
        where_conditions.append(f"namespace = {self.param_placeholder}")
        params.append(namespace)

        # Add category filter if provided
        if category_filter:
            placeholders = ",".join([self.param_placeholder] * len(category_filter))
            where_conditions.append(f"category_primary IN ({placeholders})")
            params.extend(category_filter)

        # Add time range filter if provided
        if time_range:
            start, end = time_range
            where_conditions.append(
                f"created_at BETWEEN {self.param_placeholder} AND {self.param_placeholder}"
            )
            params.extend([start, end])

        # Add minimum importance filter if provided
        if importance_min is not None:
            where_conditions.append(f"importance_score >= {self.param_placeholder}")
            params.append(importance_min)

        # Add boolean anchor expression if provided
        anchor_sql: str | None = None
        anchor_params: list[Any] = []
        if anchor_expression:
            try:
                anchor_sql, anchor_params = self._parse_anchor_expression(
                    anchor_expression
                )
            except ValidationError as e:
                raise DatabaseError(f"Invalid anchor expression: {e}")

        if anchor_sql:
            where_conditions.append(anchor_sql)
            params.extend(anchor_params)

        # Add privacy range filter if provided
        if privacy_range:
            privacy_min, privacy_max = privacy_range
            where_conditions.append(
                f"y_coord BETWEEN {self.param_placeholder} AND {self.param_placeholder}"
            )
            params.extend([privacy_min, privacy_max])

        # Add coordinate range filters if provided
        if coordinates:
            axis_map = {"x": "x_coord", "y": "y_coord", "z": "z_coord"}
            for axis, bounds in coordinates.items():
                column = axis_map.get(axis, axis)
                try:
                    InputValidator.sanitize_sql_identifier(column)
                except ValidationError:
                    continue
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    where_conditions.append(
                        f"{column} BETWEEN {self.param_placeholder} AND {self.param_placeholder}"
                    )
                    params.extend([bounds[0], bounds[1]])

        # Build the complete query
        if len(tables) == 1:
            # Single table query
            query = f"""
                SELECT *, '{tables[0]}' as memory_type
                FROM {tables[0]}
                WHERE {' AND '.join(where_conditions)}
                ORDER BY importance_score DESC, created_at DESC
                {self.LIMIT_SYNTAX[self.dialect]}
            """
        else:
            # Multi-table UNION query
            union_parts = []
            for table in tables:
                table_query = f"""
                    SELECT *, '{table}' as memory_type
                    FROM {table}
                    WHERE {' AND '.join(where_conditions)}
                """
                union_parts.append(table_query)

            query = f"""
                SELECT * FROM (
                    {' UNION ALL '.join(union_parts)}
                ) combined
                ORDER BY importance_score DESC, created_at DESC
                {self.LIMIT_SYNTAX[self.dialect]}
            """

        params.append(limit)

        return query, params

    def build_insert_query(
        self, table: str, data: dict[str, Any], on_conflict: str = "REPLACE"
    ) -> tuple[str, list[Any]]:
        """Build database-specific insert query with proper parameter binding"""

        try:
            # Validate table name
            table = InputValidator.sanitize_sql_identifier(table)

            # Validate column names and data
            validated_data = {}
            for key, value in data.items():
                validated_key = InputValidator.sanitize_sql_identifier(key)
                validated_data[validated_key] = value

        except ValidationError as e:
            raise DatabaseError(f"Invalid insert parameters: {e}")

        columns = list(validated_data.keys())
        values = list(validated_data.values())

        # Build column list and placeholders
        columns_str = ", ".join(columns)
        placeholders = ", ".join([self.param_placeholder] * len(values))

        # Handle different conflict resolution strategies per database
        if on_conflict == "REPLACE":
            if self.dialect == DatabaseDialect.SQLITE:
                query = f"INSERT OR REPLACE INTO {table} ({columns_str}) VALUES ({placeholders})"
            elif self.dialect == DatabaseDialect.POSTGRESQL:
                # PostgreSQL uses ON CONFLICT clause
                primary_key = self._get_primary_key_column(columns)
                if primary_key:
                    update_clause = ", ".join(
                        [
                            f"{col} = EXCLUDED.{col}"
                            for col in columns
                            if col != primary_key
                        ]
                    )
                    query = f"""
                        INSERT INTO {table} ({columns_str}) VALUES ({placeholders})
                        ON CONFLICT ({primary_key}) DO UPDATE SET {update_clause}
                    """
                else:
                    # Fallback to simple insert if no primary key detected
                    query = (
                        f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
                    )
            elif self.dialect == DatabaseDialect.MYSQL:
                # MySQL uses ON DUPLICATE KEY UPDATE
                update_clause = ", ".join([f"{col} = VALUES({col})" for col in columns])
                query = f"""
                    INSERT INTO {table} ({columns_str}) VALUES ({placeholders})
                    ON DUPLICATE KEY UPDATE {update_clause}
                """
        else:
            # Simple insert for all databases
            query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"

        return query, values

    def build_update_query(
        self, table: str, data: dict[str, Any], where_conditions: dict[str, Any]
    ) -> tuple[str, list[Any]]:
        """Build database-specific update query"""

        try:
            # Validate table name
            table = InputValidator.sanitize_sql_identifier(table)

            # Validate all column names
            for key in list(data.keys()) + list(where_conditions.keys()):
                InputValidator.sanitize_sql_identifier(key)

        except ValidationError as e:
            raise DatabaseError(f"Invalid update parameters: {e}")

        # Build SET clause
        set_conditions = []
        params = []

        for column, value in data.items():
            set_conditions.append(f"{column} = {self.param_placeholder}")
            params.append(value)

        # Build WHERE clause
        where_parts = []
        for column, value in where_conditions.items():
            where_parts.append(f"{column} = {self.param_placeholder}")
            params.append(value)

        query = f"""
            UPDATE {table}
            SET {', '.join(set_conditions)}
            WHERE {' AND '.join(where_parts)}
        """

        return query, params

    def build_delete_query(
        self, table: str, where_conditions: dict[str, Any]
    ) -> tuple[str, list[Any]]:
        """Build database-specific delete query"""

        try:
            # Validate table name and columns
            table = InputValidator.sanitize_sql_identifier(table)
            for key in where_conditions.keys():
                InputValidator.sanitize_sql_identifier(key)

        except ValidationError as e:
            raise DatabaseError(f"Invalid delete parameters: {e}")

        where_parts = []
        params = []

        for column, value in where_conditions.items():
            where_parts.append(f"{column} = {self.param_placeholder}")
            params.append(value)

        query = f"DELETE FROM {table} WHERE {' AND '.join(where_parts)}"

        return query, params

    def build_fts_query(
        self,
        query_text: str,
        namespace: str,
        category_filter: list[str] | None = None,
        limit: int = 10,
    ) -> tuple[str, list[Any]]:
        """Build database-specific full-text search query"""

        try:
            query_text = InputValidator.validate_and_sanitize_query(query_text)
            namespace = InputValidator.validate_namespace(namespace)
            category_filter = InputValidator.validate_category_filter(category_filter)
            limit = InputValidator.validate_limit(limit)
        except ValidationError as e:
            raise DatabaseError(f"Invalid FTS parameters: {e}")

        params = []
        where_conditions = []

        if self.dialect == DatabaseDialect.SQLITE:
            # SQLite FTS5
            where_conditions.append(f"memory_search_fts MATCH {self.param_placeholder}")
            params.append(f'"{query_text}"' if query_text else "*")

            where_conditions.append(f"fts.namespace = {self.param_placeholder}")
            params.append(namespace)

            if category_filter:
                placeholders = ",".join([self.param_placeholder] * len(category_filter))
                where_conditions.append(f"fts.category_primary IN ({placeholders})")
                params.extend(category_filter)

            query = f"""
                SELECT
                    fts.memory_id, fts.memory_type, fts.category_primary,
                    CASE
                        WHEN fts.memory_type = 'short_term' THEN st.processed_data
                        WHEN fts.memory_type = 'long_term' THEN lt.processed_data
                    END as processed_data,
                    CASE
                        WHEN fts.memory_type = 'short_term' THEN st.importance_score
                        WHEN fts.memory_type = 'long_term' THEN lt.importance_score
                        ELSE 0.5
                    END as importance_score,
                    CASE
                        WHEN fts.memory_type = 'short_term' THEN st.created_at
                        WHEN fts.memory_type = 'long_term' THEN lt.created_at
                    END as created_at,
                    fts.summary,
                    rank
                FROM memory_search_fts fts
                LEFT JOIN short_term_memory st ON fts.memory_id = st.memory_id AND fts.memory_type = 'short_term'
                LEFT JOIN long_term_memory lt ON fts.memory_id = lt.memory_id AND fts.memory_type = 'long_term'
                WHERE {' AND '.join(where_conditions)}
                ORDER BY rank, importance_score DESC
                {self.LIMIT_SYNTAX[self.dialect]}
            """

        elif self.dialect == DatabaseDialect.POSTGRESQL:
            # PostgreSQL full-text search using tsvector
            where_conditions.append(
                "(to_tsvector('english', st.searchable_content) @@ plainto_tsquery('english', %s) OR to_tsvector('english', lt.searchable_content) @@ plainto_tsquery('english', %s))"
            )
            params.extend([query_text, query_text])

            where_conditions.append("(st.namespace = %s OR lt.namespace = %s)")
            params.extend([namespace, namespace])

            if category_filter:
                placeholders = ",".join(["%s"] * len(category_filter))
                where_conditions.append(
                    f"(st.category_primary IN ({placeholders}) OR lt.category_primary IN ({placeholders}))"
                )
                params.extend(category_filter * 2)  # For both tables

            query = f"""
                SELECT DISTINCT
                    COALESCE(st.memory_id, lt.memory_id) as memory_id,
                    CASE WHEN st.memory_id IS NOT NULL THEN 'short_term' ELSE 'long_term' END as memory_type,
                    COALESCE(st.category_primary, lt.category_primary) as category_primary,
                    COALESCE(st.processed_data, lt.processed_data) as processed_data,
                    COALESCE(st.importance_score, lt.importance_score) as importance_score,
                    COALESCE(st.created_at, lt.created_at) as created_at,
                    COALESCE(st.summary, lt.summary) as summary,
                    ts_rank(COALESCE(to_tsvector('english', st.searchable_content), to_tsvector('english', lt.searchable_content)), plainto_tsquery('english', %s)) as rank
                FROM short_term_memory st
                FULL OUTER JOIN long_term_memory lt ON FALSE  -- Force separate processing
                WHERE {' AND '.join(where_conditions)}
                ORDER BY rank DESC, importance_score DESC
                {self.LIMIT_SYNTAX[self.dialect]}
            """
            params.append(query_text)  # For ts_rank

        elif self.dialect == DatabaseDialect.MYSQL:
            # MySQL FULLTEXT search
            where_conditions.append(
                "(MATCH(st.searchable_content) AGAINST(%s IN BOOLEAN MODE) OR MATCH(lt.searchable_content) AGAINST(%s IN BOOLEAN MODE))"
            )
            params.extend([query_text, query_text])

            where_conditions.append("(st.namespace = %s OR lt.namespace = %s)")
            params.extend([namespace, namespace])

            if category_filter:
                placeholders = ",".join(["%s"] * len(category_filter))
                where_conditions.append(
                    f"(st.category_primary IN ({placeholders}) OR lt.category_primary IN ({placeholders}))"
                )
                params.extend(category_filter * 2)

            query = f"""
                SELECT
                    COALESCE(st.memory_id, lt.memory_id) as memory_id,
                    CASE WHEN st.memory_id IS NOT NULL THEN 'short_term' ELSE 'long_term' END as memory_type,
                    COALESCE(st.category_primary, lt.category_primary) as category_primary,
                    COALESCE(st.processed_data, lt.processed_data) as processed_data,
                    COALESCE(st.importance_score, lt.importance_score) as importance_score,
                    COALESCE(st.created_at, lt.created_at) as created_at,
                    COALESCE(st.summary, lt.summary) as summary,
                    GREATEST(
                        COALESCE(MATCH(st.searchable_content) AGAINST(%s IN BOOLEAN MODE), 0),
                        COALESCE(MATCH(lt.searchable_content) AGAINST(%s IN BOOLEAN MODE), 0)
                    ) as rank
                FROM short_term_memory st
                LEFT JOIN long_term_memory lt ON FALSE  -- Force UNION behavior
                UNION ALL
                SELECT
                    lt.memory_id,
                    'long_term' as memory_type,
                    lt.category_primary,
                    lt.processed_data,
                    lt.importance_score,
                    lt.created_at,
                    lt.summary,
                    MATCH(lt.searchable_content) AGAINST(%s IN BOOLEAN MODE) as rank
                FROM long_term_memory lt
                WHERE {' AND '.join(where_conditions)}
                ORDER BY rank DESC, importance_score DESC
                {self.LIMIT_SYNTAX[self.dialect]}
            """
            params.extend(
                [query_text, query_text, query_text]
            )  # For MATCH calculations

        params.append(limit)
        return query, params

    def _get_primary_key_column(self, columns: list[str]) -> str | None:
        """Detect likely primary key column from column names"""
        pk_candidates = [
            "id",
            "memory_id",
            "chat_id",
            "entity_id",
            "relationship_id",
            "rule_id",
        ]

        for candidate in pk_candidates:
            if candidate in columns:
                return candidate

        # If no standard primary key found, use the first column ending with '_id'
        for column in columns:
            if column.endswith("_id"):
                return column

        return None


class DatabaseQueryExecutor:
    """Execute queries with proper error handling and transaction management"""

    def __init__(self, connector, dialect: DatabaseDialect):
        self.connector = connector
        self.query_builder = QueryBuilder(dialect)

    def execute_search(
        self,
        query_text: str,
        namespace: str = "default",
        category_filter: list[str] | None = None,
        limit: int = 10,
        use_fts: bool = True,
    ) -> list[dict[str, Any]]:
        """Execute search with proper error handling"""
        try:
            if use_fts:
                # Try FTS first
                try:
                    sql_query, params = self.query_builder.build_fts_query(
                        query_text, namespace, category_filter, limit
                    )
                    results = self.connector.execute_query(sql_query, params)
                    if results:
                        return results
                except Exception as e:
                    logger.debug(f"FTS search failed, falling back to LIKE: {e}")

            # Fallback to LIKE search
            tables = ["short_term_memory", "long_term_memory"]
            columns = ["searchable_content", "summary"]

            sql_query, params = self.query_builder.build_search_query(
                tables,
                columns,
                query_text,
                namespace,
                category_filter,
                limit,
                use_fts=False,
            )

            return self.connector.execute_query(sql_query, params)

        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return []

    def execute_safe_insert(
        self, table: str, data: dict[str, Any], on_conflict: str = "REPLACE"
    ) -> str | None:
        """Execute insert with proper error handling"""
        try:
            sql_query, params = self.query_builder.build_insert_query(
                table, data, on_conflict
            )
            return self.connector.execute_insert(sql_query, params)
        except Exception as e:
            logger.error(f"Insert execution failed: {e}")
            raise DatabaseError(f"Failed to insert into {table}: {e}")

    def execute_safe_update(
        self, table: str, data: dict[str, Any], where_conditions: dict[str, Any]
    ) -> int:
        """Execute update with proper error handling"""
        try:
            sql_query, params = self.query_builder.build_update_query(
                table, data, where_conditions
            )
            return self.connector.execute_update(sql_query, params)
        except Exception as e:
            logger.error(f"Update execution failed: {e}")
            raise DatabaseError(f"Failed to update {table}: {e}")

    def execute_safe_delete(self, table: str, where_conditions: dict[str, Any]) -> int:
        """Execute delete with proper error handling"""
        try:
            sql_query, params = self.query_builder.build_delete_query(
                table, where_conditions
            )
            return self.connector.execute_delete(sql_query, params)
        except Exception as e:
            logger.error(f"Delete execution failed: {e}")
            raise DatabaseError(f"Failed to delete from {table}: {e}")
