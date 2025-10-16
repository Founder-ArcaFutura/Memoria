"""
SQLAlchemy-based search service for the Memoria 0.9 alpha (0.9.0a0) release.
Provides cross-database full-text search capabilities.
"""

import json
import math
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from loguru import logger
from sqlalchemy import (
    JSON,
    String,
    bindparam,
    cast,
    desc,
    func,
    literal,
    not_,
    or_,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - fallback when rapidfuzz isn't installed
    from difflib import SequenceMatcher

    class _FallbackFuzz:
        @staticmethod
        def partial_ratio(a: str, b: str) -> int:
            return int(SequenceMatcher(None, a, b).ratio() * 100)

        @staticmethod
        def ratio(a: str, b: str) -> int:
            return int(SequenceMatcher(None, a, b).ratio() * 100)

    fuzz = _FallbackFuzz()

from memoria.utils.embeddings import normalize_embedding, vector_similarity
from memoria.schemas import canonicalize_symbolic_anchors

from .models import LinkMemoryThread, LongTermMemory, ShortTermMemory
from .sqlite_utils import is_sqlite_json_error

DEFAULT_RANK_WEIGHTS = {
    "search": 0.5,
    "importance": 0.3,
    "recency": 0.2,
    "anchor": 0.1,
    "vector": 0.3,
}


class SearchService:
    """Cross-database search service using SQLAlchemy"""

    _ANCHOR_DIRECTIVE_PATTERN = re.compile(
        r"""
        (?P<full>
            (?P<prefix>\b(?:anchor|anchors|symbolic_anchor|symbolic_anchors|tag)\b)
            \s*[:=]\s*
            (?P<value>
                \[[^\]]*\]
                |
                "[^"]*"
                |
                '[^']*'
                |
                [^\s]+)
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(
        self,
        session: Session,
        database_type: str,
        rank_weights: dict[str, float] | None = None,
        *,
        vector_search_enabled: bool = False,
        team_id: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        self.session = session
        self.database_type = database_type
        self.rank_weights = DEFAULT_RANK_WEIGHTS.copy()
        if rank_weights:
            self.rank_weights.update(rank_weights)
        self._sqlite_json_anchor_supported: bool | None = None
        self._sqlite_anchor_warning_logged = False
        self.vector_search_enabled = bool(vector_search_enabled)
        self.team_id = team_id
        self.workspace_id = workspace_id

    def _normalize_symbolic_anchors(self, anchors: Any) -> list[str]:
        """Normalize anchor representations into a list of strings."""
        if anchors is None:
            return []

        if isinstance(anchors, str):
            stripped = anchors.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
            except Exception:
                return [stripped]
            else:
                return self._normalize_symbolic_anchors(parsed)

        if isinstance(anchors, (list, tuple, set)):
            normalized: list[str] = []
            for anchor in anchors:
                normalized.extend(self._normalize_symbolic_anchors(anchor))
            return normalized

        if isinstance(anchors, dict):
            normalized: list[str] = []
            for value in anchors.values():
                normalized.extend(self._normalize_symbolic_anchors(value))
            return normalized

        text = str(anchors).strip()
        return [text] if text else []

    def _normalize_related_ids(self, value: Any) -> list[str]:
        """Normalize serialized related memory identifiers into a list."""

        if value is None:
            return []

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
            except Exception:
                return [stripped]
            return self._normalize_related_ids(parsed)

        if isinstance(value, (list, tuple, set)):
            normalized: list[str] = []
            for item in value:
                normalized.extend(self._normalize_related_ids(item))
            return normalized

        if isinstance(value, dict):
            normalized: list[str] = []
            for item in value.values():
                normalized.extend(self._normalize_related_ids(item))
            return normalized

        text = str(value).strip()
        return [text] if text else []

    def _anchors_from_processed_data(self, processed_data: Any) -> Any:
        """Extract symbolic anchors from serialized processed data payloads."""
        if isinstance(processed_data, dict):
            return processed_data.get("symbolic_anchors")
        if isinstance(processed_data, str):
            stripped = processed_data.strip()
            if not stripped:
                return None
            try:
                parsed = json.loads(stripped)
            except Exception:
                return None
            if isinstance(parsed, dict):
                return parsed.get("symbolic_anchors")
        return None

    @staticmethod
    def _sqlite_anchor_substring_conditions(column, anchors: list[str], prefix: str):
        return [
            func.instr(
                func.lower(cast(column, String)),
                bindparam(f"{prefix}_{idx}", f'"{anchor.lower()}"'),
            )
            > literal(0)
            for idx, anchor in enumerate(anchors)
        ]

    @classmethod
    def _extract_anchor_directives(cls, query: str) -> tuple[str, list[str]]:
        """Parse anchor directives from the query string."""

        anchors: list[str] = []

        def _normalize(raw: str) -> list[str]:
            value = raw.strip()
            if not value:
                return []
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            parts = [part.strip() for part in re.split(r"[,;]", value) if part.strip()]
            return parts or [value]

        def _replace(match: re.Match[str]) -> str:
            raw_value = match.group("value") or ""
            anchors.extend(_normalize(raw_value))
            return " "

        cleaned = cls._ANCHOR_DIRECTIVE_PATTERN.sub(_replace, query or "")
        return cleaned.strip(), anchors

    def _build_anchor_conditions(
        self,
        column,
        anchors,
        *,
        sqlite_fallback: bool = False,
        prefix: str = "anchor",
    ) -> list[Any]:
        """Return backend-appropriate symbolic anchor conditions for a column."""

        normalized: list[str] = []
        if anchors:
            for anchor in anchors:
                if anchor is None:
                    continue
                value = str(anchor)
                if not value:
                    continue
                normalized.append(value)

        if not normalized:
            return []

        if self.database_type == "sqlite":
            fallback_required = (
                sqlite_fallback or self._sqlite_json_anchor_supported is False
            )
            if fallback_required:
                return self._sqlite_anchor_substring_conditions(
                    column, normalized, prefix
                )
            return [column.contains(anchor) for anchor in normalized]

        if self.database_type == "postgresql":
            anchor_param = bindparam(
                f"{prefix}_anchors", normalized, type_=ARRAY(String)
            )
            return [func.jsonb_exists_any(cast(column, JSONB), anchor_param)]

        if self.database_type == "mysql":
            conditions: list[Any] = []
            for idx, anchor in enumerate(normalized):
                anchor_param = bindparam(f"{prefix}_{idx}", anchor)
                quoted_anchor = func.json_quote(anchor_param)
                conditions.append(func.json_contains(cast(column, JSON), quoted_anchor))
            return conditions

        return [cast(column, ARRAY(String)).contains([anchor]) for anchor in normalized]

    def _extract_symbolic_anchors(
        self, direct_value: Any, processed_data: Any = None
    ) -> list[str]:
        """Return normalized anchors, falling back to processed_data if needed."""
        anchors = self._normalize_symbolic_anchors(direct_value)
        if anchors:
            return anchors
        fallback = self._anchors_from_processed_data(processed_data)
        return self._normalize_symbolic_anchors(fallback)

    @staticmethod
    def _apply_privacy_constraint(query, model, min_privacy: float | None):
        """Apply a privacy floor to queries when ``y_coord`` is available."""

        if min_privacy is None:
            return query

        y_column = getattr(model, "y_coord", None)
        if y_column is None:
            return query

        return query.filter(or_(y_column.is_(None), y_column >= min_privacy))

    @staticmethod
    def _combine_or(conditions):
        if not conditions:
            raise ValueError("At least one condition is required for OR combination")
        if len(conditions) == 1:
            return conditions[0]
        return or_(*conditions)

    def _meta_anchor_clauses(
        self, anchor: str, idx: int, *, force_sqlite_substring: bool = False
    ) -> tuple[Any, Any, list[str]]:
        anchor_value = str(anchor)
        anchor_list = [anchor_value]

        sqlite_fallback = force_sqlite_substring

        short_conditions = self._build_anchor_conditions(
            ShortTermMemory.symbolic_anchors,
            anchor_list,
            sqlite_fallback=sqlite_fallback,
            prefix=f"meta_short_{idx}",
        )
        long_conditions = self._build_anchor_conditions(
            LongTermMemory.symbolic_anchors,
            anchor_list,
            sqlite_fallback=sqlite_fallback,
            prefix=f"meta_long_{idx}",
        )

        if not short_conditions or not long_conditions:
            raise ValueError("Anchor conditions could not be constructed")

        short_clause = self._combine_or(short_conditions)
        long_clause = self._combine_or(long_conditions)
        return short_clause, long_clause, anchor_list

    def _count_anchor_matches(
        self, model, condition, y_min: float, y_max: float
    ) -> int:
        query = (
            self.session.query(model)
            .filter(model.y_coord.between(y_min, y_max))
            .filter(condition)
        )
        return int(query.count())

    def _log_sqlite_anchor_fallback(self, anchors: list[str], exc: Exception) -> None:
        if not self._sqlite_anchor_warning_logged:
            logger.warning(
                "SQLite JSON1 functions unavailable; falling back to substring anchor search (anchors=%s): %s",
                anchors,
                exc,
            )
            self._sqlite_anchor_warning_logged = True

    def _apply_sqlite_anchor_filter(self, base_query, column, anchors: list[str]):
        if not anchors:
            return base_query

        anchor_values = [str(anchor) for anchor in anchors]
        fallback_conditions = self._build_anchor_conditions(
            column,
            anchor_values,
            sqlite_fallback=True,
            prefix="fallback_anchor",
        )
        fallback_clause = self._combine_or(fallback_conditions)

        def apply_fallback():
            return base_query.filter(fallback_clause)

        if self._sqlite_json_anchor_supported is False:
            return apply_fallback()

        json_conditions = self._build_anchor_conditions(
            column,
            anchor_values,
            sqlite_fallback=False,
            prefix="fallback_anchor",
        )
        json_clause = self._combine_or(json_conditions)
        json_query = base_query.filter(json_clause)

        if self._sqlite_json_anchor_supported is None:
            test_query = json_query.limit(0)
            try:
                test_query.all()
            except OperationalError as exc:
                session = getattr(test_query, "session", None) or self.session
                if session is not None:
                    session.rollback()
                if not is_sqlite_json_error(exc):
                    raise
                self._sqlite_json_anchor_supported = False
                self._log_sqlite_anchor_fallback(anchor_values, exc)
                return apply_fallback()
            else:
                self._sqlite_json_anchor_supported = True

        return json_query

    def _apply_distance_metadata(
        self,
        data: list[dict[str, Any]],
        x: float | None,
        y: float | None,
        z: float | None,
    ) -> list[dict[str, Any]]:
        """Annotate search results with spatial distance when coordinates are provided."""

        if not data or x is None or y is None or z is None:
            return data

        for item in data:
            mx, my, mz = item.get("x"), item.get("y"), item.get("z")
            if mx is None or my is None or mz is None:
                continue
            item["distance"] = math.sqrt((mx - x) ** 2 + (my - y) ** 2 + (mz - z) ** 2)
        return data

    def _search_by_anchor(
        self,
        query: str | Sequence[str],
        namespace: str,
        category_filter: list[str] | None,
        limit: int,
        search_short_term: bool,
        search_long_term: bool,
        start_timestamp: datetime | None,
        end_timestamp: datetime | None,
        min_importance: float | None,
        max_importance: float | None,
        anchors: list[str] | None,
        x: float | None,
        y: float | None,
        z: float | None,
        max_distance: float | None,
        rank_weights: dict[str, float] | None,
        *,
        query_embedding: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Run anchor-based search and return ranked results."""

        if isinstance(query, Sequence) and not isinstance(query, (str, bytes)):
            anchor_terms: list[str] = []
            for value in query:
                text = str(value).strip()
                if text and text not in anchor_terms:
                    anchor_terms.append(text)
            combined_results: list[dict[str, Any]] = []
            for term in anchor_terms:
                anchor_results = self._search_symbolic_anchor(
                    term,
                    namespace,
                    category_filter,
                    limit,
                    search_short_term,
                    search_long_term,
                    start_timestamp,
                    end_timestamp,
                    min_importance,
                    max_importance,
                    anchors,
                    x,
                    y,
                    z,
                    max_distance,
                )
                if not anchor_results:
                    continue
                combined_results = self._merge_results(combined_results, anchor_results)
                if len(combined_results) >= limit:
                    break

            if not combined_results:
                return []

            enriched = self._apply_distance_metadata(combined_results, x, y, z)
            return self._rank_and_limit_results(
                enriched,
                limit,
                " ".join(anchor_terms),
                rank_weights,
                query_embedding=query_embedding,
            )

        anchor_value = str(query).strip()
        if not anchor_value:
            return []

        anchor_results = self._search_symbolic_anchor(
            anchor_value,
            namespace,
            category_filter,
            limit,
            search_short_term,
            search_long_term,
            start_timestamp,
            end_timestamp,
            min_importance,
            max_importance,
            anchors,
            x,
            y,
            z,
            max_distance,
        )

        if not anchor_results:
            return []

        enriched = self._apply_distance_metadata(anchor_results, x, y, z)
        return self._rank_and_limit_results(
            enriched,
            limit,
            anchor_value,
            rank_weights,
            query_embedding=query_embedding,
        )

    def _search_by_keywords(
        self,
        keywords: list[str],
        namespace: str,
        category_filter: list[str] | None,
        limit: int,
        search_short_term: bool,
        search_long_term: bool,
        start_timestamp: datetime | None,
        end_timestamp: datetime | None,
        min_importance: float | None,
        max_importance: float | None,
        anchors: list[str] | None,
        x: float | None,
        y: float | None,
        z: float | None,
        max_distance: float | None,
        rank_weights: dict[str, float] | None,
        *,
        query_embedding: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Run keyword-based search across memories."""

        keyword_results = self._search_keyword_terms(
            keywords,
            namespace,
            category_filter,
            limit,
            search_short_term,
            search_long_term,
            start_timestamp,
            end_timestamp,
            min_importance,
            max_importance,
            anchors,
            x,
            y,
            z,
            max_distance,
        )

        enriched = self._apply_distance_metadata(keyword_results, x, y, z)
        return self._rank_and_limit_results(
            enriched,
            limit,
            " ".join(keywords),
            rank_weights,
            query_embedding=query_embedding,
        )

    def _run_fulltext_pipeline(
        self,
        query: str,
        namespace: str,
        category_filter: list[str] | None,
        limit: int,
        search_short_term: bool,
        search_long_term: bool,
        use_fuzzy: bool,
        fuzzy_max_results: int,
        adjusted_min_similarity: int,
        start_timestamp: datetime | None,
        end_timestamp: datetime | None,
        min_importance: float | None,
        max_importance: float | None,
        anchors: list[str] | None,
        x: float | None,
        y: float | None,
        z: float | None,
        max_distance: float | None,
        advanced_filters: bool,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Execute the full-text, fuzzy, and LIKE search pipeline."""

        results: list[dict[str, Any]] = []
        attempted: list[str] = []

        def run_fuzzy_like(existing: list[dict[str, Any]]):
            stage_attempts: list[str] = []
            remaining_limit = limit - len(existing)
            if (
                remaining_limit > 0
                and (use_fuzzy or len(existing) < limit)
                and fuzzy_max_results > 0
            ):
                stage_attempts.append("fuzzy")
                fuzzy_results = self._search_fuzzy(
                    query,
                    namespace,
                    category_filter,
                    min(fuzzy_max_results, remaining_limit),
                    search_short_term,
                    search_long_term,
                    adjusted_min_similarity,
                    start_timestamp,
                    end_timestamp,
                    min_importance,
                    max_importance,
                    anchors,
                    x,
                    y,
                    z,
                    max_distance,
                )
                enriched_fuzzy = self._apply_distance_metadata(fuzzy_results, x, y, z)
                existing = self._merge_results(existing, enriched_fuzzy)

            remaining_limit = limit - len(existing)
            if remaining_limit > 0:
                stage_attempts.append("like")
                like_results = self._search_like_fallback(
                    query,
                    namespace,
                    category_filter,
                    remaining_limit,
                    search_short_term,
                    search_long_term,
                    start_timestamp,
                    end_timestamp,
                    min_importance,
                    max_importance,
                    anchors,
                    x,
                    y,
                    z,
                    max_distance,
                )
                enriched_like = self._apply_distance_metadata(like_results, x, y, z)
                existing = self._merge_results(existing, enriched_like)

            return existing, stage_attempts

        try:
            if limit > 0 and not advanced_filters:
                attempted.append("fts")
                if self.database_type == "sqlite":
                    fts_results = self._search_sqlite_fts(
                        query,
                        namespace,
                        category_filter,
                        limit,
                        search_short_term,
                        search_long_term,
                    )
                elif self.database_type == "mysql":
                    fts_results = self._search_mysql_fulltext(
                        query,
                        namespace,
                        category_filter,
                        limit,
                        search_short_term,
                        search_long_term,
                    )
                elif self.database_type == "postgresql":
                    fts_results = self._search_postgresql_fts(
                        query,
                        namespace,
                        category_filter,
                        limit,
                        search_short_term,
                        search_long_term,
                    )
                else:
                    fts_results = []

                enriched_fts = self._apply_distance_metadata(fts_results, x, y, z)
                results = self._merge_results(results, enriched_fts)

            results, stage_attempts = run_fuzzy_like(results)
            attempted.extend(stage_attempts)
        except Exception as exc:
            logger.warning(
                f"Full-text search failed: {exc}, falling back to fuzzy/LIKE search"
            )
            results, stage_attempts = run_fuzzy_like(results)
            attempted.extend(stage_attempts)

        return results, attempted

    def search_memories(
        self,
        query: str,
        namespace: str = "default",
        category_filter: list[str] | None = None,
        limit: int = 10,
        memory_types: list[str] | None = None,
        use_anchor: bool = True,
        use_fuzzy: bool = False,
        fuzzy_min_similarity: int = 50,
        fuzzy_max_results: int = 5,
        keywords: list[str] | None = None,
        adaptive_min_similarity: bool = True,
        rank_weights: dict[str, float] | None = None,
        *,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        max_distance: float | None = None,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
        min_importance: float | None = None,
        max_importance: float | None = None,
        anchors: list[str] | None = None,
        query_embedding: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """
        Search memories across different database backends

        Args:
            query: Search query string
            namespace: Memory namespace
            category_filter: list of categories to filter by
            limit: Maximum number of results
            memory_types: Types of memory to search ('short_term', 'long_term', or both)
            use_anchor: Whether to search symbolic_anchors first
            use_fuzzy: Force fuzzy search even if full-text search returns results
            fuzzy_min_similarity: Minimum RapidFuzz score (0-100) for fuzzy matches
            fuzzy_max_results: Maximum number of fuzzy matches to include
            keywords: list of keywords to require in results (overrides query)

            adaptive_min_similarity: Adaptively lower fuzzy_min_similarity based on query length
            rank_weights: Override ranking coefficients for this search
            start_timestamp: Include memories created at or after this time
            end_timestamp: Include memories created before or at this time
            min_importance: Minimum importance score
            max_importance: Maximum importance score
            anchors: list of symbolic anchors any result must contain

        Returns:
            dict with keys:
                "results": list of memory dictionaries with search metadata
                "attempted": list of search stages attempted
                "hint": Optional hint when results are empty or fallbacks used
        """

        query = query or ""
        query, directive_anchors = self._extract_anchor_directives(query)
        combined_anchors = canonicalize_symbolic_anchors(
            (anchors or []) + directive_anchors
        ) or []

        if combined_anchors:
            seen: set[str] = set()
            unique: list[str] = []
            for anchor in combined_anchors:
                if anchor not in seen:
                    seen.add(anchor)
                    unique.append(anchor)
            combined_anchors = unique

        anchors = combined_anchors or None

        anchor_queries: list[str] = []
        if combined_anchors:
            anchor_queries = combined_anchors
        elif self._looks_like_anchor(query):
            trimmed = query.strip()
            if trimmed:
                anchor_queries = [trimmed]

        effective_query = query.strip() or " ".join(anchor_queries)

        if (not effective_query) and not keywords:
            recent = self._get_recent_memories(
                namespace, category_filter, limit, memory_types
            )
            recent = self._apply_distance_metadata(recent, x, y, z)
            ranked = self._rank_and_limit_results(
                recent,
                limit,
                effective_query,
                rank_weights,
                query_embedding=query_embedding,
            )
            return {"results": ranked, "attempted": []}

        remaining_limit = limit

        # Determine which memory types to search
        search_short_term = not memory_types or "short_term" in memory_types
        search_long_term = not memory_types or "long_term" in memory_types

        if use_anchor and self.session is not None and anchor_queries:
            anchor_argument: str | Sequence[str]
            if len(anchor_queries) == 1:
                anchor_argument = anchor_queries[0]
            else:
                anchor_argument = anchor_queries
            anchor_ranked = self._search_by_anchor(
                anchor_argument,
                namespace,
                category_filter,
                remaining_limit,
                search_short_term,
                search_long_term,
                start_timestamp,
                end_timestamp,
                min_importance,
                max_importance,
                anchors,
                x,
                y,
                z,
                max_distance,
                rank_weights,
                query_embedding=query_embedding,
            )
            if anchor_ranked:
                return {"results": anchor_ranked, "attempted": ["anchor"]}

        if keywords:
            keyword_ranked = self._search_by_keywords(
                keywords,
                namespace,
                category_filter,
                remaining_limit,
                search_short_term,
                search_long_term,
                start_timestamp,
                end_timestamp,
                min_importance,
                max_importance,
                anchors,
                x,
                y,
                z,
                max_distance,
                rank_weights,
                query_embedding=query_embedding,
            )
            return {"results": keyword_ranked, "attempted": ["keyword"]}

        adjusted_min_similarity = (
            self._adjust_min_similarity(query, fuzzy_min_similarity)
            if adaptive_min_similarity
            else fuzzy_min_similarity
        )

        advanced_filters = any(
            [
                start_timestamp,
                end_timestamp,
                anchors,
                min_importance is not None,
                max_importance is not None,
                (
                    x is not None
                    and y is not None
                    and z is not None
                    and max_distance is not None
                ),
            ]
        )

        search_query = effective_query or ""

        pipeline_results, attempted = self._run_fulltext_pipeline(
            search_query,
            namespace,
            category_filter,
            limit,
            search_short_term,
            search_long_term,
            use_fuzzy,
            fuzzy_max_results,
            adjusted_min_similarity,
            start_timestamp,
            end_timestamp,
            min_importance,
            max_importance,
            anchors,
            x,
            y,
            z,
            max_distance,
            advanced_filters,
        )

        pipeline_results = self._apply_distance_metadata(pipeline_results, x, y, z)
        ranked = self._rank_and_limit_results(
            pipeline_results,
            limit,
            search_query,
            rank_weights,
            query_embedding=query_embedding,
        )
        response: dict[str, Any] = {"results": ranked, "attempted": attempted}
        if not ranked or any(stage in ("fuzzy", "like") for stage in attempted):
            response["hint"] = "Use shorter keywords or anchor:<label>"
        return response

    def _adjust_min_similarity(self, query: str, base: int) -> int:
        """Scale minimum similarity down for longer queries."""
        tokens = query.strip().split()
        adjusted = base - max(0, len(tokens) - 1) * 2
        return max(0, adjusted)

    def _merge_results(
        self, existing: list[dict[str, Any]], new: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        existing_map = {r["memory_id"]: r for r in existing}
        for item in new:
            existing_item = existing_map.get(item["memory_id"])
            if existing_item:
                for key, value in item.items():
                    if key not in existing_item or existing_item.get(key) is None:
                        existing_item[key] = value
            else:
                existing.append(item)
                existing_map[item["memory_id"]] = item
        return existing

    def _apply_filters(
        self,
        query,
        model,
        namespace: str | None,
        category_filter: list[str] | None,
        start_timestamp: datetime | None,
        end_timestamp: datetime | None,
        min_importance: float | None,
        max_importance: float | None,
        anchors: list[str] | None,
        x: float | None,
        y: float | None,
        z: float | None,
        max_distance: float | None,
        *,
        sqlite_anchor_fallback: bool = False,
        enforce_namespace: bool = True,
    ):
        q = query
        if self.team_id is not None and hasattr(model, "team_id"):
            q = q.filter(model.team_id == self.team_id)
        if self.workspace_id is not None and hasattr(model, "workspace_id"):
            q = q.filter(model.workspace_id == self.workspace_id)
        if enforce_namespace and namespace:
            q = q.filter(model.namespace == namespace)
        if category_filter:
            q = q.filter(model.category_primary.in_(category_filter))
        time_col = getattr(model, "timestamp", None) or model.created_at
        if start_timestamp:
            q = q.filter(time_col >= start_timestamp)
        if end_timestamp:
            q = q.filter(time_col <= end_timestamp)
        if min_importance is not None:
            q = q.filter(model.importance_score >= min_importance)
        if max_importance is not None:
            q = q.filter(model.importance_score <= max_importance)
        if anchors:
            anchor_conditions = self._build_anchor_conditions(
                model.symbolic_anchors,
                anchors,
                sqlite_fallback=sqlite_anchor_fallback,
                prefix=f"{model.__tablename__}_filters",
            )
            if anchor_conditions:
                q = q.filter(self._combine_or(anchor_conditions))
        if (
            x is not None
            and y is not None
            and z is not None
            and max_distance is not None
        ):
            dist_expr = (
                (model.x_coord - x) * (model.x_coord - x)
                + (model.y_coord - y) * (model.y_coord - y)
                + (model.z_coord - z) * (model.z_coord - z)
            )
            q = q.filter(dist_expr <= max_distance * max_distance)
        return q

    def get_neighbors(self, memory_id: str, radius: float) -> list[dict[str, Any]]:
        """Return memories within a spatial radius of the given memory."""

        if not self.session or not memory_id or radius is None or radius <= 0:
            return []

        source = (
            self.session.query(ShortTermMemory)
            .filter(ShortTermMemory.memory_id == memory_id)
            .first()
        )
        if source is None:
            source = (
                self.session.query(LongTermMemory)
                .filter(LongTermMemory.memory_id == memory_id)
                .first()
            )

        if source is None:
            return []

        x, y, z = source.x_coord, source.y_coord, source.z_coord
        if x is None or y is None or z is None:
            return []

        radius_sq = radius * radius
        neighbors: list[dict[str, Any]] = []

        def _collect(model, mem_type: str) -> None:
            dist_expr = (
                (model.x_coord - x) * (model.x_coord - x)
                + (model.y_coord - y) * (model.y_coord - y)
                + (model.z_coord - z) * (model.z_coord - z)
            )

            query = (
                self.session.query(
                    model.memory_id,
                    model.x_coord,
                    model.y_coord,
                    model.z_coord,
                    dist_expr.label("distance_sq"),
                )
                .filter(model.memory_id != memory_id)
                .filter(dist_expr <= radius_sq)
            )

            for neighbor_id, nx, ny, nz, distance_sq in query.all():
                if (
                    neighbor_id is None
                    or nx is None
                    or ny is None
                    or nz is None
                    or distance_sq is None
                ):
                    continue
                neighbors.append(
                    {
                        "memory_id": neighbor_id,
                        "memory_type": mem_type,
                        "x_coord": nx,
                        "y_coord": ny,
                        "z_coord": nz,
                        "distance": math.sqrt(float(distance_sq)),
                    }
                )

        _collect(ShortTermMemory, "short_term")
        _collect(LongTermMemory, "long_term")

        neighbors.sort(key=lambda item: item["distance"])
        return neighbors

    @staticmethod
    def _looks_like_anchor(query: str) -> bool:
        return bool(
            query and " " not in query.strip() and re.match(r"^[A-Za-z0-9_]+$", query)
        )

    def _run_sqlite_anchor_safe(self, build_query, execute, context: str):
        query = build_query(False)
        try:
            return execute(query)
        except OperationalError as exc:
            if self.database_type == "sqlite" and is_sqlite_json_error(exc):
                self.session.rollback()
                logger.warning(
                    "SQLite JSON1 functions unavailable; falling back to substring anchor search (%s): %s",
                    context,
                    exc,
                )
                fallback_query = build_query(True)
                return execute(fallback_query)
            raise

    def _search_symbolic_anchor(
        self,
        anchor: str,
        namespace: str,
        category_filter: list[str] | None,
        limit: int,
        search_short_term: bool,
        search_long_term: bool,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
        min_importance: float | None = None,
        max_importance: float | None = None,
        anchors: list[str] | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        max_distance: float | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        results: list[dict[str, Any]] = []

        normalized_anchor = str(anchor)
        remaining = limit
        if search_short_term and remaining > 0:

            def build_short_query(use_fallback: bool):
                short_query_local = self.session.query(ShortTermMemory)
                short_query_local = self._apply_filters(
                    short_query_local,
                    ShortTermMemory,
                    namespace,
                    category_filter,
                    start_timestamp,
                    end_timestamp,
                    min_importance,
                    max_importance,
                    anchors,
                    x,
                    y,
                    z,
                    max_distance,
                    sqlite_anchor_fallback=use_fallback,
                    enforce_namespace=False,
                )
                conditions = self._build_anchor_conditions(
                    ShortTermMemory.symbolic_anchors,
                    [normalized_anchor],
                    sqlite_fallback=use_fallback,
                    prefix="short_anchor_search",
                )
                if conditions:
                    short_query_local = short_query_local.filter(
                        self._combine_or(conditions)
                    )
                return short_query_local

            def execute_short(query_obj):
                return (
                    query_obj.order_by(
                        desc(ShortTermMemory.importance_score),
                        desc(ShortTermMemory.created_at),
                    )
                    .limit(remaining)
                    .all()
                )

            short_results = self._run_sqlite_anchor_safe(
                build_short_query,
                execute_short,
                context=f"short-term anchor search '{normalized_anchor}'",
            )
            for row in short_results:
                emotion = (
                    row.processed_data.get("emotional_intensity")
                    if isinstance(row.processed_data, dict)
                    else None
                )
                anchors_list = self._extract_symbolic_anchors(
                    getattr(row, "symbolic_anchors", None), row.processed_data
                )
                results.append(
                    {
                        "memory_id": row.memory_id,
                        "memory_type": "short_term",
                        "processed_data": row.processed_data,
                        "importance_score": row.importance_score,
                        "created_at": row.created_at,
                        "x": row.x_coord,
                        "y": row.y_coord,
                        "z": row.z_coord,
                        "summary": row.summary,
                        "category_primary": row.category_primary,
                        "emotional_intensity": emotion,
                        "symbolic_anchors": anchors_list,
                        "search_score": 1.0,
                        "search_strategy": "symbolic_anchor",
                    }
                )
            remaining = limit - len(results)

        if search_long_term and remaining > 0:

            def build_long_query(use_fallback: bool):
                long_query_local = self.session.query(LongTermMemory)
                long_query_local = self._apply_filters(
                    long_query_local,
                    LongTermMemory,
                    namespace,
                    category_filter,
                    start_timestamp,
                    end_timestamp,
                    min_importance,
                    max_importance,
                    anchors,
                    x,
                    y,
                    z,
                    max_distance,
                    sqlite_anchor_fallback=use_fallback,
                    enforce_namespace=False,
                )
                conditions = self._build_anchor_conditions(
                    LongTermMemory.symbolic_anchors,
                    [normalized_anchor],
                    sqlite_fallback=use_fallback,
                    prefix="long_anchor_search",
                )
                if conditions:
                    long_query_local = long_query_local.filter(
                        self._combine_or(conditions)
                    )
                return long_query_local

            def execute_long(query_obj):
                return (
                    query_obj.order_by(
                        desc(LongTermMemory.importance_score),
                        desc(LongTermMemory.created_at),
                    )
                    .limit(remaining)
                    .all()
                )

            long_results = self._run_sqlite_anchor_safe(
                build_long_query,
                execute_long,
                context=f"long-term anchor search '{normalized_anchor}'",
            )
            for row in long_results:
                emotion = (
                    row.processed_data.get("emotional_intensity")
                    if isinstance(row.processed_data, dict)
                    else None
                )
                anchors_list = self._extract_symbolic_anchors(
                    getattr(row, "symbolic_anchors", None), row.processed_data
                )
                results.append(
                    {
                        "memory_id": row.memory_id,
                        "memory_type": "long_term",
                        "processed_data": row.processed_data,
                        "importance_score": row.importance_score,
                        "created_at": row.created_at,
                        "x": row.x_coord,
                        "y": row.y_coord,
                        "z": row.z_coord,
                        "summary": row.summary,
                        "category_primary": row.category_primary,
                        "emotional_intensity": emotion,
                        "symbolic_anchors": anchors_list,
                        "search_score": 1.0,
                        "search_strategy": "symbolic_anchor",
                    }
                )

        return results

    def _search_fuzzy(
        self,
        query: str,
        namespace: str,
        category_filter: list[str] | None,
        limit: int,
        search_short_term: bool,
        search_long_term: bool,
        min_similarity: int,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
        min_importance: float | None = None,
        max_importance: float | None = None,
        anchors: list[str] | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        max_distance: float | None = None,
    ) -> list[dict[str, Any]]:
        """Fuzzy search using RapidFuzz"""
        if limit <= 0:
            return []

        results: list[dict[str, Any]] = []
        query_text = query.lower()

        if search_short_term:

            def build_short_query(use_fallback: bool):
                short_query_local = self.session.query(ShortTermMemory)
                return self._apply_filters(
                    short_query_local,
                    ShortTermMemory,
                    namespace,
                    category_filter,
                    start_timestamp,
                    end_timestamp,
                    min_importance,
                    max_importance,
                    anchors,
                    x,
                    y,
                    z,
                    max_distance,
                    sqlite_anchor_fallback=use_fallback,
                )

            short_rows = self._run_sqlite_anchor_safe(
                build_short_query,
                lambda q: q.all(),
                context="short-term fuzzy filters",
            )

            for row in short_rows:
                row_anchors = self._extract_symbolic_anchors(
                    getattr(row, "symbolic_anchors", None), row.processed_data
                )
                best_anchor_score = 0
                for anchor in row_anchors:
                    anchor_score = fuzz.ratio(query_text, str(anchor).lower())
                    best_anchor_score = max(best_anchor_score, anchor_score)
                emotion = (
                    row.processed_data.get("emotional_intensity")
                    if isinstance(row.processed_data, dict)
                    else None
                )
                if best_anchor_score >= min_similarity:
                    results.append(
                        {
                            "memory_id": row.memory_id,
                            "memory_type": "short_term",
                            "processed_data": row.processed_data,
                            "importance_score": row.importance_score,
                            "created_at": row.created_at,
                            "x": row.x_coord,
                            "y": row.y_coord,
                            "z": row.z_coord,
                            "summary": row.summary,
                            "category_primary": row.category_primary,
                            "emotional_intensity": emotion,
                            "symbolic_anchors": row_anchors,
                            "search_score": best_anchor_score / 100,
                            "search_strategy": f"{self.database_type}_anchor",
                        }
                    )
                    continue

                text = (
                    f"{row.searchable_content} {row.summary} {' '.join(map(str, row_anchors))}"
                ).lower()
                if len(query_text) > len(text):
                    score = fuzz.ratio(query_text, text)
                else:
                    score = fuzz.partial_ratio(query_text, text)
                if score >= min_similarity:
                    results.append(
                        {
                            "memory_id": row.memory_id,
                            "memory_type": "short_term",
                            "processed_data": row.processed_data,
                            "importance_score": row.importance_score,
                            "created_at": row.created_at,
                            "x": row.x_coord,
                            "y": row.y_coord,
                            "z": row.z_coord,
                            "summary": row.summary,
                            "category_primary": row.category_primary,
                            "emotional_intensity": emotion,
                            "symbolic_anchors": row_anchors,
                            "search_score": score / 100,
                            "search_strategy": f"{self.database_type}_fuzzy",
                        }
                    )

        if search_long_term:

            def build_long_query(use_fallback: bool):
                long_query_local = self.session.query(LongTermMemory)
                return self._apply_filters(
                    long_query_local,
                    LongTermMemory,
                    namespace,
                    category_filter,
                    start_timestamp,
                    end_timestamp,
                    min_importance,
                    max_importance,
                    anchors,
                    x,
                    y,
                    z,
                    max_distance,
                    sqlite_anchor_fallback=use_fallback,
                )

            long_rows = self._run_sqlite_anchor_safe(
                build_long_query,
                lambda q: q.all(),
                context="long-term fuzzy filters",
            )

            for row in long_rows:
                row_anchors = self._extract_symbolic_anchors(
                    getattr(row, "symbolic_anchors", None), row.processed_data
                )
                best_anchor_score = 0
                for anchor in row_anchors:
                    anchor_score = fuzz.ratio(query_text, str(anchor).lower())
                    best_anchor_score = max(best_anchor_score, anchor_score)
                emotion = (
                    row.processed_data.get("emotional_intensity")
                    if isinstance(row.processed_data, dict)
                    else None
                )
                if best_anchor_score >= min_similarity:
                    results.append(
                        {
                            "memory_id": row.memory_id,
                            "memory_type": "long_term",
                            "processed_data": row.processed_data,
                            "importance_score": row.importance_score,
                            "created_at": row.created_at,
                            "x": row.x_coord,
                            "y": row.y_coord,
                            "z": row.z_coord,
                            "summary": row.summary,
                            "category_primary": row.category_primary,
                            "emotional_intensity": emotion,
                            "symbolic_anchors": row_anchors,
                            "search_score": best_anchor_score / 100,
                            "search_strategy": f"{self.database_type}_anchor",
                        }
                    )
                    continue

                text = (
                    f"{row.searchable_content} {row.summary} {' '.join(map(str, row_anchors))}"
                ).lower()
                if len(query_text) > len(text):
                    score = fuzz.ratio(query_text, text)
                else:
                    score = fuzz.partial_ratio(query_text, text)
                if score >= min_similarity:
                    results.append(
                        {
                            "memory_id": row.memory_id,
                            "memory_type": "long_term",
                            "processed_data": row.processed_data,
                            "importance_score": row.importance_score,
                            "created_at": row.created_at,
                            "x": row.x_coord,
                            "y": row.y_coord,
                            "z": row.z_coord,
                            "summary": row.summary,
                            "category_primary": row.category_primary,
                            "emotional_intensity": emotion,
                            "symbolic_anchors": row_anchors,
                            "search_score": score / 100,
                            "search_strategy": f"{self.database_type}_fuzzy",
                        }
                    )

        results.sort(key=lambda x: x["search_score"], reverse=True)
        return results[:limit]

    def _search_keyword_terms(
        self,
        keywords: list[str],
        namespace: str,
        category_filter: list[str] | None,
        limit: int,
        search_short_term: bool,
        search_long_term: bool,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
        min_importance: float | None = None,
        max_importance: float | None = None,
        anchors: list[str] | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        max_distance: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search using a list of keywords ensuring all terms are present."""
        if not keywords or limit <= 0:
            return []

        if self.database_type == "sqlite":
            results: list[dict[str, Any]] = []
            if search_short_term:

                def build_short_query(use_fallback: bool):
                    short_query_local = self.session.query(ShortTermMemory)
                    return self._apply_filters(
                        short_query_local,
                        ShortTermMemory,
                        namespace,
                        category_filter,
                        start_timestamp,
                        end_timestamp,
                        min_importance,
                        max_importance,
                        anchors,
                        x,
                        y,
                        z,
                        max_distance,
                        sqlite_anchor_fallback=use_fallback,
                    )

                def execute_short(query_obj):
                    q_local = query_obj
                    for kw in keywords:
                        pattern = f"%{kw}%"
                        q_local = q_local.filter(
                            or_(
                                ShortTermMemory.searchable_content.like(pattern),
                                ShortTermMemory.summary.like(pattern),
                            )
                        )
                    return (
                        q_local.order_by(
                            desc(ShortTermMemory.importance_score),
                            desc(ShortTermMemory.created_at),
                        )
                        .limit(limit)
                        .all()
                    )

                short_results = self._run_sqlite_anchor_safe(
                    build_short_query,
                    execute_short,
                    context="short-term keyword filters",
                )
                for row in short_results:
                    emotion = (
                        row.processed_data.get("emotional_intensity")
                        if isinstance(row.processed_data, dict)
                        else None
                    )
                    anchors_list = self._extract_symbolic_anchors(
                        getattr(row, "symbolic_anchors", None), row.processed_data
                    )
                    results.append(
                        {
                            "memory_id": row.memory_id,
                            "memory_type": "short_term",
                            "processed_data": row.processed_data,
                            "importance_score": row.importance_score,
                            "created_at": row.created_at,
                            "x": row.x_coord,
                            "y": row.y_coord,
                            "z": row.z_coord,
                            "summary": row.summary,
                            "category_primary": row.category_primary,
                            "emotional_intensity": emotion,
                            "symbolic_anchors": anchors_list,
                            "search_score": 0.4,
                            "search_strategy": f"{self.database_type}_like_keywords",
                        }
                    )
            if search_long_term:

                def build_long_query(use_fallback: bool):
                    long_query_local = self.session.query(LongTermMemory)
                    return self._apply_filters(
                        long_query_local,
                        LongTermMemory,
                        namespace,
                        category_filter,
                        start_timestamp,
                        end_timestamp,
                        min_importance,
                        max_importance,
                        anchors,
                        x,
                        y,
                        z,
                        max_distance,
                        sqlite_anchor_fallback=use_fallback,
                    )

                def execute_long(query_obj):
                    q_local = query_obj
                    for kw in keywords:
                        pattern = f"%{kw}%"
                        q_local = q_local.filter(
                            or_(
                                LongTermMemory.searchable_content.like(pattern),
                                LongTermMemory.summary.like(pattern),
                            )
                        )
                    return (
                        q_local.order_by(
                            desc(LongTermMemory.importance_score),
                            desc(LongTermMemory.created_at),
                        )
                        .limit(limit)
                        .all()
                    )

                long_results = self._run_sqlite_anchor_safe(
                    build_long_query,
                    execute_long,
                    context="long-term keyword filters",
                )
                for row in long_results:
                    emotion = (
                        row.processed_data.get("emotional_intensity")
                        if isinstance(row.processed_data, dict)
                        else None
                    )
                    anchors_list = self._extract_symbolic_anchors(
                        getattr(row, "symbolic_anchors", None), row.processed_data
                    )
                    results.append(
                        {
                            "memory_id": row.memory_id,
                            "memory_type": "long_term",
                            "processed_data": row.processed_data,
                            "importance_score": row.importance_score,
                            "created_at": row.created_at,
                            "x": row.x_coord,
                            "y": row.y_coord,
                            "z": row.z_coord,
                            "summary": row.summary,
                            "category_primary": row.category_primary,
                            "emotional_intensity": emotion,
                            "symbolic_anchors": anchors_list,
                            "search_score": 0.4,
                            "search_strategy": f"{self.database_type}_like_keywords",
                        }
                    )
            return results[:limit]
        elif self.database_type == "postgresql":
            joined = " ".join(keywords)
            return self._search_postgresql_fts(
                joined,
                namespace,
                category_filter,
                limit,
                search_short_term,
                search_long_term,
            )
        elif self.database_type == "mysql":
            results: list[dict[str, Any]] = []
            try:
                bool_query = " ".join(f"+{k}" for k in keywords)
                if search_short_term:
                    short_query = self.session.query(ShortTermMemory).filter(
                        ShortTermMemory.namespace == namespace
                    )
                    fulltext_condition = text(
                        "MATCH(searchable_content, summary) AGAINST(:query IN BOOLEAN MODE)"
                    ).params(query=bool_query)
                    short_query = short_query.filter(fulltext_condition)
                    if category_filter:
                        short_query = short_query.filter(
                            ShortTermMemory.category_primary.in_(category_filter)
                        )
                    short_results = self.session.execute(
                        short_query.statement.add_columns(
                            text(
                                "MATCH(searchable_content, summary) AGAINST(:query IN BOOLEAN MODE) as search_score"
                            ).params(query=bool_query),
                            text("'short_term' as memory_type"),
                            text("'mysql_fulltext' as search_strategy"),
                            ShortTermMemory.x_coord,
                            ShortTermMemory.y_coord,
                            ShortTermMemory.z_coord,
                            ShortTermMemory.symbolic_anchors.label("symbolic_anchors"),
                        )
                    ).fetchall()
                    for row in short_results:
                        row_dict = dict(row)
                        row_dict["x"] = row_dict.pop("x_coord", None)
                        row_dict["y"] = row_dict.pop("y_coord", None)
                        row_dict["z"] = row_dict.pop("z_coord", None)
                        row_dict["symbolic_anchors"] = self._extract_symbolic_anchors(
                            row_dict.get("symbolic_anchors"),
                            row_dict.get("processed_data"),
                        )
                        results.append(row_dict)

                if search_long_term:
                    long_query = self.session.query(LongTermMemory).filter(
                        LongTermMemory.namespace == namespace
                    )
                    fulltext_condition = text(
                        "MATCH(searchable_content, summary) AGAINST(:query IN BOOLEAN MODE)"
                    ).params(query=bool_query)
                    long_query = long_query.filter(fulltext_condition)
                    if category_filter:
                        long_query = long_query.filter(
                            LongTermMemory.category_primary.in_(category_filter)
                        )
                    long_results = self.session.execute(
                        long_query.statement.add_columns(
                            text(
                                "MATCH(searchable_content, summary) AGAINST(:query IN BOOLEAN MODE) as search_score"
                            ).params(query=bool_query),
                            text("'long_term' as memory_type"),
                            text("'mysql_fulltext' as search_strategy"),
                            LongTermMemory.x_coord,
                            LongTermMemory.y_coord,
                            LongTermMemory.z_coord,
                            LongTermMemory.symbolic_anchors.label("symbolic_anchors"),
                        )
                    ).fetchall()
                    for row in long_results:
                        row_dict = dict(row)
                        row_dict["x"] = row_dict.pop("x_coord", None)
                        row_dict["y"] = row_dict.pop("y_coord", None)
                        row_dict["z"] = row_dict.pop("z_coord", None)
                        row_dict["symbolic_anchors"] = self._extract_symbolic_anchors(
                            row_dict.get("symbolic_anchors"),
                            row_dict.get("processed_data"),
                        )
                        results.append(row_dict)
                return results[:limit]
            except Exception as e:
                logger.debug(f"MySQL keyword search failed: {e}")
                self.session.rollback()
                return []
        else:
            return []

    def _search_sqlite_fts(
        self,
        query: str,
        namespace: str,
        category_filter: list[str] | None,
        limit: int,
        search_short_term: bool,
        search_long_term: bool,
    ) -> list[dict[str, Any]]:
        """Search using SQLite FTS5"""
        try:
            # Build FTS query - allow boolean expressions
            trimmed = query.strip()
            if re.search(r"\b(AND|OR|NOT)\b", trimmed, re.IGNORECASE):
                fts_query = trimmed
            else:
                fts_query = f'"{trimmed}"'

            # Build category filter
            category_clause = ""
            params = {"fts_query": fts_query, "namespace": namespace}

            if category_filter:
                category_placeholders = ",".join(
                    [f":cat_{i}" for i in range(len(category_filter))]
                )
                category_clause = (
                    f"AND fts.category_primary IN ({category_placeholders})"
                )
                for i, cat in enumerate(category_filter):
                    params[f"cat_{i}"] = cat

            # SQLite FTS5 search query
            sql_query = f"""
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
                    CASE
                        WHEN fts.memory_type = 'short_term' THEN st.x_coord
                        WHEN fts.memory_type = 'long_term' THEN lt.x_coord
                    END as x_coord,
                    CASE
                        WHEN fts.memory_type = 'short_term' THEN st.y_coord
                        WHEN fts.memory_type = 'long_term' THEN lt.y_coord
                    END as y_coord,
                    CASE
                        WHEN fts.memory_type = 'short_term' THEN st.z_coord
                        WHEN fts.memory_type = 'long_term' THEN lt.z_coord
                    END as z_coord,
                    CASE
                        WHEN fts.memory_type = 'short_term' THEN st.symbolic_anchors
                        WHEN fts.memory_type = 'long_term' THEN lt.symbolic_anchors
                    END as symbolic_anchors,
                    fts.summary,
                    rank as search_score,
                    'sqlite_fts5' as search_strategy
                FROM memory_search_fts fts
                LEFT JOIN short_term_memory st ON fts.memory_id = st.memory_id AND fts.memory_type = 'short_term'
                LEFT JOIN long_term_memory lt ON fts.memory_id = lt.memory_id AND fts.memory_type = 'long_term'
                WHERE memory_search_fts MATCH :fts_query AND fts.namespace = :namespace
                {category_clause}
                ORDER BY rank, importance_score DESC
                LIMIT {limit}
            """

            result = self.session.execute(text(sql_query), params)
            rows = []
            for row in result:
                data = row.processed_data
                if not isinstance(data, dict):
                    try:
                        data = json.loads(data)
                    except Exception:
                        data = {}
                emotion = data.get("emotional_intensity")
                anchors_list = self._extract_symbolic_anchors(
                    getattr(row, "symbolic_anchors", None), data
                )
                rows.append(
                    {
                        "memory_id": row.memory_id,
                        "memory_type": row.memory_type,
                        "category_primary": row.category_primary,
                        "processed_data": data,
                        "importance_score": row.importance_score,
                        "created_at": row.created_at,
                        "x": row.x_coord,
                        "y": row.y_coord,
                        "z": row.z_coord,
                        "summary": row.summary,
                        "search_score": row.search_score,
                        "search_strategy": row.search_strategy,
                        "emotional_intensity": emotion,
                        "symbolic_anchors": anchors_list,
                    }
                )
            return rows

        except Exception as e:
            logger.debug(f"SQLite FTS5 search failed: {e}")
            # Roll back the transaction to recover from error state
            self.session.rollback()
            return []

    def _run_mysql_fulltext_query(
        self,
        model: type[ShortTermMemory] | type[LongTermMemory],
        namespace: str,
        category_filter: list[str] | None,
        query: str,
        memory_type: str,
    ) -> list[dict[str, Any]]:
        base_query = self.session.query(model).filter(model.namespace == namespace)

        fulltext_condition = text(
            "MATCH(searchable_content, summary) AGAINST(:query IN NATURAL LANGUAGE MODE)"
        ).params(query=query)
        base_query = base_query.filter(fulltext_condition)

        if category_filter:
            base_query = base_query.filter(model.category_primary.in_(category_filter))

        result_rows = self.session.execute(
            base_query.statement.add_columns(
                text(
                    "MATCH(searchable_content, summary) AGAINST(:query IN NATURAL LANGUAGE MODE) as search_score"
                ).params(query=query),
                literal(memory_type).label("memory_type"),
                literal("mysql_fulltext").label("search_strategy"),
                model.x_coord,
                model.y_coord,
                model.z_coord,
                model.symbolic_anchors.label("symbolic_anchors"),
            )
        ).fetchall()

        processed_rows: list[dict[str, Any]] = []
        for row in result_rows:
            row_dict = dict(row)
            row_dict["x"] = row_dict.pop("x_coord", None)
            row_dict["y"] = row_dict.pop("y_coord", None)
            row_dict["z"] = row_dict.pop("z_coord", None)
            row_dict["symbolic_anchors"] = self._extract_symbolic_anchors(
                row_dict.get("symbolic_anchors"),
                row_dict.get("processed_data"),
            )
            processed_rows.append(row_dict)

        return processed_rows

    def _search_mysql_fulltext(
        self,
        query: str,
        namespace: str,
        category_filter: list[str] | None,
        limit: int,
        search_short_term: bool,
        search_long_term: bool,
    ) -> list[dict[str, Any]]:
        """Search using MySQL FULLTEXT"""
        results = []

        try:
            search_targets: list[
                tuple[
                    type[ShortTermMemory] | type[LongTermMemory],
                    str,
                ]
            ] = []

            if search_short_term:
                search_targets.append((ShortTermMemory, "short_term"))

            if search_long_term:
                search_targets.append((LongTermMemory, "long_term"))

            for model, memory_label in search_targets:
                results.extend(
                    self._run_mysql_fulltext_query(
                        model,
                        namespace,
                        category_filter,
                        query,
                        memory_label,
                    )
                )

            return results

        except Exception as e:
            logger.debug(f"MySQL FULLTEXT search failed: {e}")
            # Roll back the transaction to recover from error state
            self.session.rollback()
            return []

    def _search_postgresql_fts(
        self,
        query: str,
        namespace: str,
        category_filter: list[str] | None,
        limit: int,
        search_short_term: bool,
        search_long_term: bool,
    ) -> list[dict[str, Any]]:
        """Search using PostgreSQL tsvector"""
        results = []

        try:
            # Prepare query for tsquery - handle spaces and special characters
            # Convert simple query to tsquery format (join words with &)
            tsquery_text = " & ".join(query.split())

            # Search short-term memory if requested
            if search_short_term:
                short_query = self.session.query(ShortTermMemory).filter(
                    ShortTermMemory.namespace == namespace
                )

                # Add tsvector search
                ts_query = text(
                    "search_vector @@ to_tsquery('english', :query)"
                ).params(query=tsquery_text)
                short_query = short_query.filter(ts_query)

                # Add category filter
                if category_filter:
                    short_query = short_query.filter(
                        ShortTermMemory.category_primary.in_(category_filter)
                    )

                # Add relevance score
                short_results = self.session.execute(
                    short_query.statement.add_columns(
                        text(
                            "ts_rank(search_vector, to_tsquery('english', :query)) as search_score"
                        ).params(query=tsquery_text),
                        text("'short_term' as memory_type"),
                        text("'postgresql_fts' as search_strategy"),
                        ShortTermMemory.x_coord,
                        ShortTermMemory.y_coord,
                        ShortTermMemory.z_coord,
                        ShortTermMemory.symbolic_anchors.label("symbolic_anchors"),
                    ).order_by(text("search_score DESC"))
                ).fetchall()

                for row in short_results:
                    row_dict = dict(row)
                    row_dict["x"] = row_dict.pop("x_coord", None)
                    row_dict["y"] = row_dict.pop("y_coord", None)
                    row_dict["z"] = row_dict.pop("z_coord", None)
                    row_dict["symbolic_anchors"] = self._extract_symbolic_anchors(
                        row_dict.get("symbolic_anchors"),
                        row_dict.get("processed_data"),
                    )
                    results.append(row_dict)

            # Search long-term memory if requested
            if search_long_term:
                long_query = self.session.query(LongTermMemory).filter(
                    LongTermMemory.namespace == namespace
                )

                # Add tsvector search
                ts_query = text(
                    "search_vector @@ to_tsquery('english', :query)"
                ).params(query=tsquery_text)
                long_query = long_query.filter(ts_query)

                # Add category filter
                if category_filter:
                    long_query = long_query.filter(
                        LongTermMemory.category_primary.in_(category_filter)
                    )

                # Add relevance score
                long_results = self.session.execute(
                    long_query.statement.add_columns(
                        text(
                            "ts_rank(search_vector, to_tsquery('english', :query)) as search_score"
                        ).params(query=tsquery_text),
                        text("'long_term' as memory_type"),
                        text("'postgresql_fts' as search_strategy"),
                        LongTermMemory.x_coord,
                        LongTermMemory.y_coord,
                        LongTermMemory.z_coord,
                        LongTermMemory.symbolic_anchors.label("symbolic_anchors"),
                    ).order_by(text("search_score DESC"))
                ).fetchall()

                for row in long_results:
                    row_dict = dict(row)
                    row_dict["x"] = row_dict.pop("x_coord", None)
                    row_dict["y"] = row_dict.pop("y_coord", None)
                    row_dict["z"] = row_dict.pop("z_coord", None)
                    row_dict["symbolic_anchors"] = self._extract_symbolic_anchors(
                        row_dict.get("symbolic_anchors"),
                        row_dict.get("processed_data"),
                    )
                    results.append(row_dict)

            return results

        except Exception as e:
            logger.debug(f"PostgreSQL FTS search failed: {e}")
            # Roll back the transaction to recover from error state
            self.session.rollback()
            return []

    def _search_like_fallback(
        self,
        query: str,
        namespace: str,
        category_filter: list[str] | None,
        limit: int,
        search_short_term: bool,
        search_long_term: bool,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
        min_importance: float | None = None,
        max_importance: float | None = None,
        anchors: list[str] | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        max_distance: float | None = None,
    ) -> list[dict[str, Any]]:
        """Fallback LIKE-based search"""
        results = []
        search_pattern = f"%{query}%"

        # Search short-term memory
        if search_short_term:

            def build_short_query(use_fallback: bool):
                short_query_local = self.session.query(ShortTermMemory)
                return self._apply_filters(
                    short_query_local,
                    ShortTermMemory,
                    namespace,
                    category_filter,
                    start_timestamp,
                    end_timestamp,
                    min_importance,
                    max_importance,
                    anchors,
                    x,
                    y,
                    z,
                    max_distance,
                    sqlite_anchor_fallback=use_fallback,
                )

            def execute_short(query_obj):
                q_local = query_obj.filter(
                    or_(
                        ShortTermMemory.searchable_content.like(search_pattern),
                        ShortTermMemory.summary.like(search_pattern),
                    )
                )
                return (
                    q_local.order_by(
                        desc(ShortTermMemory.importance_score),
                        desc(ShortTermMemory.created_at),
                    )
                    .limit(limit)
                    .all()
                )

            short_results = self._run_sqlite_anchor_safe(
                build_short_query,
                execute_short,
                context="short-term like fallback",
            )

            for result in short_results:
                emotion = (
                    result.processed_data.get("emotional_intensity")
                    if isinstance(result.processed_data, dict)
                    else None
                )
                anchors_list = self._extract_symbolic_anchors(
                    getattr(result, "symbolic_anchors", None), result.processed_data
                )
                memory_dict = {
                    "memory_id": result.memory_id,
                    "memory_type": "short_term",
                    "processed_data": result.processed_data,
                    "importance_score": result.importance_score,
                    "created_at": result.created_at,
                    "x": result.x_coord,
                    "y": result.y_coord,
                    "z": result.z_coord,
                    "summary": result.summary,
                    "category_primary": result.category_primary,
                    "emotional_intensity": emotion,
                    "symbolic_anchors": anchors_list,
                    "search_score": 0.4,  # Fixed score for LIKE search
                    "search_strategy": f"{self.database_type}_like_fallback",
                }
                results.append(memory_dict)

        # Search long-term memory
        if search_long_term:

            def build_long_query(use_fallback: bool):
                long_query_local = self.session.query(LongTermMemory)
                return self._apply_filters(
                    long_query_local,
                    LongTermMemory,
                    namespace,
                    category_filter,
                    start_timestamp,
                    end_timestamp,
                    min_importance,
                    max_importance,
                    anchors,
                    x,
                    y,
                    z,
                    max_distance,
                    sqlite_anchor_fallback=use_fallback,
                )

            def execute_long(query_obj):
                q_local = query_obj.filter(
                    or_(
                        LongTermMemory.searchable_content.like(search_pattern),
                        LongTermMemory.summary.like(search_pattern),
                    )
                )
                return (
                    q_local.order_by(
                        desc(LongTermMemory.importance_score),
                        desc(LongTermMemory.created_at),
                    )
                    .limit(limit)
                    .all()
                )

            long_results = self._run_sqlite_anchor_safe(
                build_long_query,
                execute_long,
                context="long-term like fallback",
            )

            for result in long_results:
                emotion = (
                    result.processed_data.get("emotional_intensity")
                    if isinstance(result.processed_data, dict)
                    else None
                )
                anchors_list = self._extract_symbolic_anchors(
                    getattr(result, "symbolic_anchors", None), result.processed_data
                )
                memory_dict = {
                    "memory_id": result.memory_id,
                    "memory_type": "long_term",
                    "processed_data": result.processed_data,
                    "importance_score": result.importance_score,
                    "created_at": result.created_at,
                    "x": result.x_coord,
                    "y": result.y_coord,
                    "z": result.z_coord,
                    "summary": result.summary,
                    "category_primary": result.category_primary,
                    "emotional_intensity": emotion,
                    "symbolic_anchors": anchors_list,
                    "search_score": 0.4,  # Fixed score for LIKE search
                    "search_strategy": f"{self.database_type}_like_fallback",
                }
                results.append(memory_dict)

        return results

    def _get_recent_memories(
        self,
        namespace: str,
        category_filter: list[str] | None,
        limit: int,
        memory_types: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Get recent memories when no search query is provided"""
        results = []

        search_short_term = not memory_types or "short_term" in memory_types
        search_long_term = not memory_types or "long_term" in memory_types

        # Get recent short-term memories
        if search_short_term:
            short_query = self.session.query(ShortTermMemory).filter(
                ShortTermMemory.namespace == namespace
            )

            if self.team_id is not None:
                short_query = short_query.filter(
                    ShortTermMemory.team_id == self.team_id
                )
            if self.workspace_id is not None:
                short_query = short_query.filter(
                    ShortTermMemory.workspace_id == self.workspace_id
                )

            if category_filter:
                short_query = short_query.filter(
                    ShortTermMemory.category_primary.in_(category_filter)
                )

            short_results = (
                short_query.order_by(desc(ShortTermMemory.created_at))
                .limit(limit // 2)
                .all()
            )

            for result in short_results:
                emotion = (
                    result.processed_data.get("emotional_intensity")
                    if isinstance(result.processed_data, dict)
                    else None
                )
                anchors_list = self._extract_symbolic_anchors(
                    getattr(result, "symbolic_anchors", None), result.processed_data
                )
                memory_dict = {
                    "memory_id": result.memory_id,
                    "memory_type": "short_term",
                    "processed_data": result.processed_data,
                    "importance_score": result.importance_score,
                    "created_at": result.created_at,
                    "x": result.x_coord,
                    "y": result.y_coord,
                    "z": result.z_coord,
                    "summary": result.summary,
                    "category_primary": result.category_primary,
                    "emotional_intensity": emotion,
                    "symbolic_anchors": anchors_list,
                    "search_score": 1.0,
                    "search_strategy": "recent_memories",
                }
                results.append(memory_dict)

        # Get recent long-term memories
        if search_long_term:
            long_query = self.session.query(LongTermMemory).filter(
                LongTermMemory.namespace == namespace
            )

            if self.team_id is not None:
                long_query = long_query.filter(LongTermMemory.team_id == self.team_id)
            if self.workspace_id is not None:
                long_query = long_query.filter(
                    LongTermMemory.workspace_id == self.workspace_id
                )

            if category_filter:
                long_query = long_query.filter(
                    LongTermMemory.category_primary.in_(category_filter)
                )

            long_results = (
                long_query.order_by(desc(LongTermMemory.created_at))
                .limit(limit // 2)
                .all()
            )

            for result in long_results:
                emotion = (
                    result.processed_data.get("emotional_intensity")
                    if isinstance(result.processed_data, dict)
                    else None
                )
                anchors_list = self._extract_symbolic_anchors(
                    getattr(result, "symbolic_anchors", None), result.processed_data
                )
                memory_dict = {
                    "memory_id": result.memory_id,
                    "memory_type": "long_term",
                    "processed_data": result.processed_data,
                    "importance_score": result.importance_score,
                    "created_at": result.created_at,
                    "x": result.x_coord,
                    "y": result.y_coord,
                    "z": result.z_coord,
                    "summary": result.summary,
                    "category_primary": result.category_primary,
                    "emotional_intensity": emotion,
                    "symbolic_anchors": anchors_list,
                    "search_score": 1.0,
                    "search_strategy": "recent_memories",
                }
                results.append(memory_dict)

        return results

    def find_related_memory_candidates(
        self,
        *,
        namespace: str,
        symbolic_anchors: Sequence[str] | None = None,
        keywords: Sequence[str] | None = None,
        topic: str | None = None,
        exclude_ids: Sequence[str] | None = None,
        limit: int = 5,
        min_privacy: float | None = -10.0,
    ) -> list[dict[str, Any]]:
        """Return candidate memories that likely relate to provided metadata."""

        if not self.session:
            return []

        anchors = self._normalize_symbolic_anchors(symbolic_anchors or [])
        terms: list[str] = []
        if keywords:
            for keyword in keywords:
                if keyword:
                    terms.append(str(keyword))
        if topic:
            terms.append(str(topic))

        if not anchors and not terms:
            return []

        query = self.session.query(LongTermMemory).filter(
            LongTermMemory.namespace == namespace
        )
        if self.team_id is not None:
            query = query.filter(LongTermMemory.team_id == self.team_id)
        if self.workspace_id is not None:
            query = query.filter(LongTermMemory.workspace_id == self.workspace_id)
        query = self._apply_privacy_constraint(query, LongTermMemory, min_privacy)

        if anchors:
            anchor_conditions = self._build_anchor_conditions(
                LongTermMemory.symbolic_anchors,
                anchors,
                sqlite_fallback=True,
                prefix="rel_anchor",
            )
            if anchor_conditions:
                query = query.filter(or_(*anchor_conditions))

        text_conditions: list[Any] = []
        for idx, term in enumerate(terms):
            normalized = term.strip().lower()
            if not normalized:
                continue
            keyword_param = bindparam(f"rel_kw_{idx}", f"%{normalized}%")
            text_conditions.append(
                func.lower(LongTermMemory.summary).like(keyword_param)
            )
            text_conditions.append(
                func.lower(LongTermMemory.searchable_content).like(keyword_param)
            )

        if text_conditions:
            query = query.filter(or_(*text_conditions))

        exclusions = [value for value in (exclude_ids or []) if value]
        if exclusions:
            query = query.filter(not_(LongTermMemory.memory_id.in_(exclusions)))

        fetch_limit = max(limit * 3, limit)
        candidate_rows = (
            query.order_by(
                desc(LongTermMemory.importance_score),
                desc(LongTermMemory.created_at),
            )
            .limit(fetch_limit)
            .all()
        )

        anchor_reference = {anchor.lower() for anchor in anchors if anchor}
        keyword_reference = {
            term.strip().lower() for term in terms if term and term.strip()
        }

        candidates: list[dict[str, Any]] = []
        for row in candidate_rows:
            anchor_values = self._extract_symbolic_anchors(
                getattr(row, "symbolic_anchors", None), row.processed_data
            )
            row_anchor_set = {value.lower() for value in anchor_values if value}
            anchor_overlap = (
                len(anchor_reference & row_anchor_set) / len(anchor_reference)
                if anchor_reference
                else 0.0
            )

            searchable_values = " ".join(
                filter(
                    None,
                    [
                        str(getattr(row, "summary", "") or ""),
                        str(getattr(row, "searchable_content", "") or ""),
                    ],
                )
            ).lower()
            keyword_hits = sum(
                1
                for keyword in keyword_reference
                if keyword and keyword in searchable_values
            )
            text_score = (
                keyword_hits / len(keyword_reference) if keyword_reference else 0.0
            )

            importance_score = float(getattr(row, "importance_score", 0.0) or 0.0)
            match_score = max(
                0.0,
                min(
                    1.0,
                    anchor_overlap * 0.5 + text_score * 0.3 + importance_score * 0.2,
                ),
            )

            reason_parts: list[str] = []
            if anchor_overlap > 0:
                reason_parts.append(
                    f"{len(anchor_reference & row_anchor_set)} anchor overlap"
                )
            if text_score > 0:
                reason_parts.append("keyword similarity")
            relationship_reason = ", ".join(reason_parts) or "metadata similarity"

            candidates.append(
                {
                    "memory_id": row.memory_id,
                    "summary": row.summary,
                    "symbolic_anchors": anchor_values,
                    "importance_score": importance_score,
                    "created_at": row.created_at,
                    "match_score": match_score,
                    "relationship_source": "candidate",
                    "relationship_reason": relationship_reason,
                }
            )

        candidates.sort(
            key=lambda item: (
                item.get("match_score", 0.0),
                item.get("importance_score", 0.0),
            ),
            reverse=True,
        )

        return candidates[:limit]

    def get_related_memories(
        self,
        memory_id: str,
        namespace: str,
        *,
        limit: int = 10,
        min_privacy: float | None = -10.0,
        relation_types: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return related memories using explicit graph and embedded metadata."""

        if not memory_id or not namespace:
            return []

        base_query = self.session.query(LongTermMemory).filter(
            LongTermMemory.memory_id == memory_id,
            LongTermMemory.namespace == namespace,
        )
        if self.team_id is not None:
            base_query = base_query.filter(LongTermMemory.team_id == self.team_id)
        if self.workspace_id is not None:
            base_query = base_query.filter(
                LongTermMemory.workspace_id == self.workspace_id
            )
        base_record = base_query.one_or_none()

        if base_record is None:
            return []

        relation_filter = [value for value in (relation_types or []) if value]
        base_anchor_values = self._extract_symbolic_anchors(
            getattr(base_record, "symbolic_anchors", None), base_record.processed_data
        )
        base_anchor_set = {
            anchor.lower() for anchor in base_anchor_values if isinstance(anchor, str)
        }

        candidate_map: dict[str, dict[str, Any]] = {}
        for order, related_id in enumerate(
            self._normalize_related_ids(base_record.related_memories_json)
        ):
            if not related_id or related_id == memory_id:
                continue
            entry = candidate_map.setdefault(
                related_id,
                {"source": "json", "relation": "related", "order": order},
            )
            entry.setdefault("order", order)

        outgoing_query = (
            self.session.query(LinkMemoryThread)
            .join(
                LongTermMemory,
                LinkMemoryThread.target_memory_id == LongTermMemory.memory_id,
            )
            .filter(LinkMemoryThread.source_memory_id == memory_id)
            .filter(LongTermMemory.namespace == namespace)
        )
        if self.team_id is not None:
            outgoing_query = outgoing_query.filter(
                LongTermMemory.team_id == self.team_id
            )
        if self.workspace_id is not None:
            outgoing_query = outgoing_query.filter(
                LongTermMemory.workspace_id == self.workspace_id
            )
        if relation_filter:
            outgoing_query = outgoing_query.filter(
                LinkMemoryThread.relation.in_(relation_filter)
            )
        outgoing_links = outgoing_query.all()

        incoming_query = (
            self.session.query(LinkMemoryThread)
            .join(
                LongTermMemory,
                LinkMemoryThread.source_memory_id == LongTermMemory.memory_id,
            )
            .filter(LinkMemoryThread.target_memory_id == memory_id)
            .filter(LongTermMemory.namespace == namespace)
        )
        if self.team_id is not None:
            incoming_query = incoming_query.filter(
                LongTermMemory.team_id == self.team_id
            )
        if self.workspace_id is not None:
            incoming_query = incoming_query.filter(
                LongTermMemory.workspace_id == self.workspace_id
            )
        if relation_filter:
            incoming_query = incoming_query.filter(
                LinkMemoryThread.relation.in_(relation_filter)
            )
        incoming_links = incoming_query.all()

        for link in outgoing_links:
            target_id = getattr(link, "target_memory_id", None)
            if not target_id or target_id == memory_id:
                continue
            entry = candidate_map.setdefault(
                target_id,
                {"source": "graph", "relation": link.relation, "order": 0},
            )
            entry["source"] = "graph"
            entry.setdefault("relation", link.relation)
            directions = entry.setdefault("directions", set())
            directions.add("outgoing")

        for link in incoming_links:
            source_id = getattr(link, "source_memory_id", None)
            if not source_id or source_id == memory_id:
                continue
            entry = candidate_map.setdefault(
                source_id,
                {"source": "graph", "relation": link.relation, "order": 0},
            )
            entry["source"] = "graph"
            entry.setdefault("relation", link.relation)
            directions = entry.setdefault("directions", set())
            directions.add("incoming")

        candidate_ids = [
            key for key in candidate_map.keys() if key and key != memory_id
        ]
        if not candidate_ids:
            return []

        related_query = (
            self.session.query(LongTermMemory)
            .filter(LongTermMemory.memory_id.in_(candidate_ids))
            .filter(LongTermMemory.namespace == namespace)
        )
        if self.team_id is not None:
            related_query = related_query.filter(LongTermMemory.team_id == self.team_id)
        if self.workspace_id is not None:
            related_query = related_query.filter(
                LongTermMemory.workspace_id == self.workspace_id
            )
        related_query = self._apply_privacy_constraint(
            related_query, LongTermMemory, min_privacy
        )
        related_records = related_query.all()

        results: list[dict[str, Any]] = []
        for record in related_records:
            meta = candidate_map.get(record.memory_id, {})
            anchor_values = self._extract_symbolic_anchors(
                getattr(record, "symbolic_anchors", None), record.processed_data
            )
            anchor_set = {value.lower() for value in anchor_values if value}
            overlap = len(base_anchor_set & anchor_set)
            overlap_score = overlap / len(base_anchor_set) if base_anchor_set else 0.0
            importance_score = float(getattr(record, "importance_score", 0.0) or 0.0)
            match_score = max(
                0.0, min(1.0, overlap_score * 0.6 + importance_score * 0.4)
            )

            directions = meta.get("directions")
            if isinstance(directions, set):
                direction_list = sorted(directions)
            elif isinstance(directions, list):
                direction_list = directions
            else:
                direction_list = None

            relation_label = meta.get("relation", "related")
            reason_parts: list[str] = []
            if relation_label:
                reason_parts.append(relation_label)
            if overlap:
                reason_parts.append(f"{overlap} shared anchors")
            relationship_reason = ", ".join(reason_parts) or relation_label

            results.append(
                {
                    "memory_id": record.memory_id,
                    "summary": record.summary,
                    "importance_score": importance_score,
                    "created_at": record.created_at,
                    "symbolic_anchors": anchor_values,
                    "relationship_type": relation_label,
                    "relationship_direction": direction_list,
                    "relationship_source": meta.get("source", "graph"),
                    "relationship_reason": relationship_reason,
                    "match_score": match_score,
                    "x": record.x_coord,
                    "y": record.y_coord,
                    "z": record.z_coord,
                }
            )

        results.sort(
            key=lambda item: (
                item.get("match_score", 0.0),
                item.get("importance_score", 0.0),
                item.get("created_at") or datetime.min,
            ),
            reverse=True,
        )

        return results[:limit]

    def meta_query(
        self, anchors: list[str], y_range: tuple[float, float]
    ) -> dict[str, Any]:
        """Return aggregated counts of memories per anchor within a privacy range.

        Supports SQLite (native JSON or substring fallback), PostgreSQL (ARRAY containment),
        and MySQL (JSON_CONTAINS-based matching) backends.
        """

        if not anchors or not self.session:
            return {"total_memories": 0, "by_anchor": {}}

        y_min, y_max = y_range
        anchor_counts: dict[str, int] = {}
        short_conditions = []
        long_conditions = []

        for idx, expr in enumerate(anchors):
            short_cond, long_cond, anchor_values = self._meta_anchor_clauses(expr, idx)

            try:
                count_short = self._count_anchor_matches(
                    ShortTermMemory, short_cond, y_min, y_max
                )
                count_long = self._count_anchor_matches(
                    LongTermMemory, long_cond, y_min, y_max
                )
            except OperationalError as exc:
                if self.database_type != "sqlite":
                    raise
                self.session.rollback()
                if not is_sqlite_json_error(exc):
                    raise
                self._sqlite_json_anchor_supported = False
                self._log_sqlite_anchor_fallback(anchor_values, exc)
                short_cond, long_cond, _ = self._meta_anchor_clauses(
                    expr, idx, force_sqlite_substring=True
                )
                count_short = self._count_anchor_matches(
                    ShortTermMemory, short_cond, y_min, y_max
                )
                count_long = self._count_anchor_matches(
                    LongTermMemory, long_cond, y_min, y_max
                )
            else:
                if (
                    self.database_type == "sqlite"
                    and self._sqlite_json_anchor_supported is None
                ):
                    self._sqlite_json_anchor_supported = True

            anchor_counts[expr] = count_short + count_long
            short_conditions.append(short_cond)
            long_conditions.append(long_cond)

        total_short = self._count_anchor_matches(
            ShortTermMemory, self._combine_or(short_conditions), y_min, y_max
        )
        total_long = self._count_anchor_matches(
            LongTermMemory, self._combine_or(long_conditions), y_min, y_max
        )

        return {
            "total_memories": int(total_short + total_long),
            "by_anchor": anchor_counts,
        }

    def _rank_and_limit_results(
        self,
        results: list[dict[str, Any]],
        limit: int,
        query: str,
        rank_weights: dict[str, float] | None = None,
        *,
        query_embedding: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Rank and limit search results"""

        weights = self.rank_weights.copy()
        if rank_weights:
            weights.update(rank_weights)

        query_terms = [t.lower() for t in re.split(r"\W+", query) if t]
        anchor_weight = weights.get("anchor", 0.0)
        vector_weight = weights.get("vector", 0.0)

        vector_active = bool(self.vector_search_enabled and query_embedding)
        if vector_active:
            self._populate_embeddings(results)

        # Calculate composite score
        for result in results:
            search_score = result.get("search_score", 0.4)
            importance_score = result.get("importance_score", 0.5)
            recency_score = self._calculate_recency_score(result.get("created_at"))

            composite = (
                search_score * weights.get("search", 0.5)
                + importance_score * weights.get("importance", 0.3)
                + recency_score * weights.get("recency", 0.2)
            )

            matched_anchor = False
            if query_terms:
                processed = result.get("processed_data")
                if processed:
                    if isinstance(processed, str):
                        try:
                            processed = json.loads(processed)
                        except Exception:
                            processed = {}
                    if isinstance(processed, dict):
                        anchors = processed.get("symbolic_anchors") or processed.get(
                            "symbolic_anchor"
                        )
                        if isinstance(anchors, str):
                            anchors = [anchors]
                        if isinstance(anchors, list):
                            anchors_lower = [str(a).lower() for a in anchors]
                            if any(term in anchors_lower for term in query_terms):
                                matched_anchor = True
                                composite += anchor_weight

            result["matched_via_anchor"] = matched_anchor

            vector_score = 0.0
            if vector_active:
                embedding_value = normalize_embedding(result.get("embedding"))
                if embedding_value:
                    vector_score = vector_similarity(query_embedding, embedding_value)
                    composite += vector_score * vector_weight
            result["vector_score"] = vector_score
            result["composite_score"] = composite

        # Sort by composite score and limit
        results.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        return results[:limit]

    def _populate_embeddings(self, results: list[dict[str, Any]]) -> None:
        """Ensure result dictionaries include concrete embedding vectors."""

        if self.session is None:
            return

        missing: dict[str, set[str]] = {"short_term": set(), "long_term": set()}
        for item in results:
            if not isinstance(item, dict):
                continue
            if normalize_embedding(item.get("embedding")):
                continue
            memory_id = item.get("memory_id")
            if not memory_id:
                continue
            mem_type = str(item.get("memory_type") or "long_term").lower()
            if mem_type not in missing:
                mem_type = "long_term"
            missing[mem_type].add(memory_id)

        if missing["short_term"]:
            rows = (
                self.session.query(ShortTermMemory.memory_id, ShortTermMemory.embedding)
                .filter(ShortTermMemory.memory_id.in_(missing["short_term"]))
                .all()
            )
            short_map = {
                row.memory_id: normalize_embedding(row.embedding) for row in rows
            }
        else:
            short_map = {}

        if missing["long_term"]:
            rows = (
                self.session.query(LongTermMemory.memory_id, LongTermMemory.embedding)
                .filter(LongTermMemory.memory_id.in_(missing["long_term"]))
                .all()
            )
            long_map = {
                row.memory_id: normalize_embedding(row.embedding) for row in rows
            }
        else:
            long_map = {}

        for item in results:
            if not isinstance(item, dict):
                continue
            mem_id = item.get("memory_id")
            if not mem_id:
                continue
            if mem_id in short_map and short_map[mem_id]:
                item["embedding"] = short_map[mem_id]
            elif mem_id in long_map and long_map[mem_id]:
                item["embedding"] = long_map[mem_id]

    def _calculate_recency_score(self, created_at) -> float:
        """Calculate recency score (0-1, newer = higher)"""
        try:
            if not created_at:
                return 0.0

            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

            days_old = (datetime.now() - created_at).days
            return max(0, 1 - (days_old / 30))  # Full score for recent, 0 after 30 days
        except:
            return 0.0
