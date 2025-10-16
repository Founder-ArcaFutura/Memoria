"""Search execution helpers for MemorySearchEngine.

When a standard search returns no results, this module automatically retries
with a fuzzy search and, if still empty, suggests an alternative query.
"""

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from ..utils.embeddings import (
    generate_embedding,
    normalize_embedding,
    vector_search_enabled,
    vector_similarity,
)
from ..utils.pydantic_models import AgentPermissions, MemorySearchQuery


class SearchExecutor:
    """Execute search strategies against the memory database.

    The executor performs a general search, falls back to fuzzy search when
    necessary, and provides a suggested query if no memories match.
    """

    def __init__(self, permissions: AgentPermissions | None = None):
        self.permissions = permissions or AgentPermissions()
        self.vector_search_enabled = vector_search_enabled()

    def execute_search(
        self,
        query: str,
        search_plan: MemorySearchQuery,
        db_manager,
        namespace: str = "default",
        limit: int = 10,
        rank_weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute search based on a prepared search plan.

        If the general search finds no results, a fuzzy search is attempted. When
        both searches are empty, a suggestion is returned via ``suggested_query``.
        """
        all_results: list[dict[str, Any]] = []
        seen_memory_ids = set()
        query_embedding = (
            self._embed_text(query) if self.vector_search_enabled else None
        )
        query_terms = [t.lower() for t in re.findall(r"\w+", query)]
        weights = rank_weights or getattr(self.permissions, "search_weights", {}) or {}

        time_filters: dict[str, Any] = {}

        coordinate_bounds: dict[str, float] = {}
        allowed_memory_ids: set[str] | None = None

        time_expression = self._extract_time_range_expression(search_plan)
        if time_expression:
            parsed_bounds = self._parse_time_range(time_expression)
            if parsed_bounds:
                start_timestamp = parsed_bounds.get("start_timestamp")
                end_timestamp = parsed_bounds.get("end_timestamp")
                if start_timestamp or end_timestamp:
                    time_filters = {
                        key: value
                        for key, value in (
                            ("start_timestamp", start_timestamp),
                            ("end_timestamp", end_timestamp),
                        )
                        if value is not None
                    }

                coordinate_bounds = {
                    key: parsed_bounds[key]
                    for key in ("start_x", "end_x")
                    if parsed_bounds.get(key) is not None
                }

                if coordinate_bounds:
                    retrieval_fn = getattr(
                        db_manager, "retrieve_memories_by_time_range", None
                    )
                    if callable(retrieval_fn):
                        try:
                            retrieval_kwargs = {
                                "start_timestamp": time_filters.get("start_timestamp"),
                                "end_timestamp": time_filters.get("end_timestamp"),
                                "start_x": coordinate_bounds.get("start_x"),
                                "end_x": coordinate_bounds.get("end_x"),
                            }
                            retrieval_results = retrieval_fn(**retrieval_kwargs) or []
                            allowed_memory_ids = {
                                item.get("memory_id")
                                for item in retrieval_results
                                if isinstance(item, dict) and item.get("memory_id")
                            }
                        except Exception as exc:
                            logger.warning(
                                "Time-range retrieval failed using coordinate bounds: {}",
                                exc,
                            )
                            allowed_memory_ids = None
                    else:
                        allowed_memory_ids = None

        if (
            search_plan.entity_filters
            or "keyword_search" in search_plan.search_strategy
        ):
            keyword_results = self._execute_keyword_search(
                search_plan,
                db_manager,
                namespace,
                limit,
                time_filters=time_filters,
                allowed_memory_ids=allowed_memory_ids,
                coordinate_bounds=coordinate_bounds,
                query_embedding=query_embedding,
            )
            for result in keyword_results:
                if (
                    isinstance(result, dict)
                    and result.get("memory_id") not in seen_memory_ids
                ):
                    seen_memory_ids.add(result["memory_id"])
                    result["search_strategy"] = "keyword_search"
                    result["search_reasoning"] = (
                        f"Keyword match for: {', '.join(search_plan.entity_filters)}"
                    )
                    self._normalize_result(result)
                    all_results.append(result)

        if (
            search_plan.category_filters
            or "category_filter" in search_plan.search_strategy
        ):
            category_results = self._execute_category_search(
                search_plan,
                db_manager,
                namespace,
                limit - len(all_results),
                time_filters=time_filters,
                allowed_memory_ids=allowed_memory_ids,
                coordinate_bounds=coordinate_bounds,
                query_embedding=query_embedding,
            )
            for result in category_results:
                if (
                    isinstance(result, dict)
                    and result.get("memory_id") not in seen_memory_ids
                ):
                    seen_memory_ids.add(result["memory_id"])
                    result["search_strategy"] = "category_filter"
                    result["search_reasoning"] = (
                        f"Category match: {', '.join([c.value for c in search_plan.category_filters])}"
                    )
                    self._normalize_result(result)
                    all_results.append(result)

        if (
            search_plan.min_importance > 0.0
            or "importance_filter" in search_plan.search_strategy
        ):
            importance_results = self._execute_importance_search(
                search_plan,
                db_manager,
                namespace,
                limit - len(all_results),
                time_filters=time_filters,
                allowed_memory_ids=allowed_memory_ids,
                coordinate_bounds=coordinate_bounds,
                query_embedding=query_embedding,
            )
            for result in importance_results:
                if (
                    isinstance(result, dict)
                    and result.get("memory_id") not in seen_memory_ids
                ):
                    seen_memory_ids.add(result["memory_id"])
                    result["search_strategy"] = "importance_filter"
                    result["search_reasoning"] = (
                        f"High importance (≥{search_plan.min_importance})"
                    )
                    self._normalize_result(result)
                    all_results.append(result)

        if not all_results:
            general_results = db_manager.search_memories(
                query=search_plan.query_text,
                namespace=namespace,
                limit=limit,
                query_embedding=query_embedding,
                **time_filters,
            )

            if isinstance(general_results, dict):
                general_results = general_results.get("results", [])
            elif general_results is None:
                general_results = []

            for result in general_results:
                if isinstance(result, dict):
                    self._normalize_result(result)
                    if not self._should_include_result(
                        result, allowed_memory_ids, coordinate_bounds
                    ):
                        continue
                    result["search_strategy"] = "general_search"
                    result["search_reasoning"] = "General content search"
                    all_results.append(result)

            if not all_results:
                fuzzy_results = db_manager.search_memories(
                    query=search_plan.query_text,
                    namespace=namespace,
                    limit=limit,
                    use_fuzzy=True,
                    query_embedding=query_embedding,
                    **time_filters,
                )
                if isinstance(fuzzy_results, dict):
                    fuzzy_results = fuzzy_results.get("results", [])
                elif fuzzy_results is None:
                    fuzzy_results = []
                for result in fuzzy_results:
                    if (
                        isinstance(result, dict)
                        and result.get("memory_id") not in seen_memory_ids
                    ):
                        self._normalize_result(result)
                        if not self._should_include_result(
                            result, allowed_memory_ids, coordinate_bounds
                        ):
                            continue
                        seen_memory_ids.add(result["memory_id"])
                        result["search_strategy"] = "fuzzy_search"
                        result["search_reasoning"] = "Fuzzy content search"
                        all_results.append(result)

                if not all_results:
                    all_results.append(
                        {"suggested_query": self._suggest_query(query_terms)}
                    )

        valid_results = []
        for result in all_results:
            if isinstance(result, dict):
                valid_results.append(result)
            else:
                logger.warning(f"Filtering out non-dict search result: {type(result)}")
        all_results = valid_results

        relationship_budget = max(0, limit - len(all_results))
        if relationship_budget > 0:
            expansion = self._expand_with_relationships(
                all_results,
                db_manager,
                namespace,
                relationship_budget,
                seen_memory_ids,
            )
            if expansion:
                all_results.extend(expansion)
        if all_results:

            def safe_created_at_parse(created_at_value):
                try:
                    if created_at_value is None:
                        return datetime.fromisoformat("2000-01-01")
                    if isinstance(created_at_value, str):
                        return datetime.fromisoformat(created_at_value)
                    if hasattr(created_at_value, "isoformat"):
                        return created_at_value
                    return datetime.fromisoformat("2000-01-01")
                except (ValueError, TypeError):
                    return datetime.fromisoformat("2000-01-01")

            all_results.sort(
                key=lambda x: (
                    x.get("importance_score", 0) * weights.get("importance", 0.7)
                    - (
                        (
                            datetime.now().replace(tzinfo=None)
                            - safe_created_at_parse(x.get("created_at")).replace(
                                tzinfo=None
                            )
                        ).total_seconds()
                        / 86400
                    )
                    * weights.get("recency", 0.001)
                ),
                reverse=True,
            )

            for result in all_results:
                result["gravity_score"] = self._gravity_score(
                    result, query_embedding, query_terms, weights
                )
                result["search_metadata"] = {
                    "original_query": query,
                    "interpreted_intent": search_plan.intent,
                    "search_timestamp": datetime.now().isoformat(),
                    "query_embedding": query_embedding,
                }

            all_results.sort(key=lambda x: x.get("gravity_score", 0), reverse=True)

        logger.debug(f"Search executed for '{query}': {len(all_results)} results found")
        return all_results[:limit]

    def _extract_time_range_expression(
        self, search_plan: MemorySearchQuery
    ) -> str | None:
        if search_plan.time_range:
            return search_plan.time_range
        if "temporal_filter" in (search_plan.search_strategy or []):
            match = re.search(
                r"x\s*:\s*-?\d+(?:\.\d+)?\.\.-?\d+(?:\.\d+)?",
                search_plan.query_text or "",
            )
            if match:
                return match.group(0)
        return None

    def _parse_time_range(self, expression: str) -> dict[str, Any]:
        bounds: dict[str, Any] = {}
        if not expression:
            return bounds

        expr = expression.strip()
        if not expr:
            return bounds

        expr_lower = expr.lower()
        now = datetime.utcnow()

        named_ranges = {
            "last_day": timedelta(days=1),
            "past_day": timedelta(days=1),
            "last_week": timedelta(days=7),
            "past_week": timedelta(days=7),
            "last_month": timedelta(days=30),
            "past_month": timedelta(days=30),
            "last_year": timedelta(days=365),
            "past_year": timedelta(days=365),
            "last_24_hours": timedelta(hours=24),
            "past_24_hours": timedelta(hours=24),
        }

        if expr_lower in named_ranges:
            delta = named_ranges[expr_lower]
            bounds["start_timestamp"] = now - delta
            bounds["end_timestamp"] = now
            return bounds

        if expr_lower == "today":
            start = datetime(now.year, now.month, now.day)
            bounds["start_timestamp"] = start
            bounds["end_timestamp"] = start + timedelta(days=1)
            return bounds

        if expr_lower == "yesterday":
            start = datetime(now.year, now.month, now.day) - timedelta(days=1)
            bounds["start_timestamp"] = start
            bounds["end_timestamp"] = start + timedelta(days=1)
            return bounds

        days_match = re.match(r"^(?:last|past)_(\d+)_days$", expr_lower)
        if days_match:
            delta = timedelta(days=int(days_match.group(1)))

            bounds["start_timestamp"] = now - delta
            bounds["end_timestamp"] = now
            return bounds

        hours_match = re.match(r"^(?:last|past)_(\d+)_hours$", expr_lower)
        if hours_match:
            delta = timedelta(hours=int(hours_match.group(1)))
            bounds["start_timestamp"] = now - delta
            bounds["end_timestamp"] = now
            return bounds

        weeks_match = re.match(r"^(?:last|past)_(\d+)_weeks$", expr_lower)
        if weeks_match:
            delta = timedelta(weeks=int(weeks_match.group(1)))
            bounds["start_timestamp"] = now - delta
            bounds["end_timestamp"] = now
            return bounds

        months_match = re.match(r"^(?:last|past)_(\d+)_months$", expr_lower)
        if months_match:
            days = int(months_match.group(1)) * 30
            delta = timedelta(days=days)
            bounds["start_timestamp"] = now - delta
            bounds["end_timestamp"] = now
            return bounds

        x_match = re.match(
            r"^x\s*:\s*(-?\d+(?:\.\d+)?)\.\.(-?\d+(?:\.\d+)?)$",
            expr_lower,
        )
        if x_match:
            start_x = float(x_match.group(1))
            end_x = float(x_match.group(2))
            if start_x > end_x:
                start_x, end_x = end_x, start_x
            bounds["start_x"] = start_x
            bounds["end_x"] = end_x
            return bounds

        range_match = re.match(r"^(?P<start>[^.]+)\.\.(?P<end>.+)$", expr)
        if range_match:
            start_expr = range_match.group("start").strip()
            end_expr = range_match.group("end").strip()
            start_dt = self._parse_time_component(start_expr, is_end=False)
            end_dt = self._parse_time_component(end_expr, is_end=True)
            if start_dt and end_dt:
                bounds["start_timestamp"] = start_dt
                bounds["end_timestamp"] = end_dt
        return bounds

    def _parse_time_component(self, value: str, *, is_end: bool) -> datetime | None:
        candidate = value.strip()
        if not candidate:
            return None

        normalized = candidate
        if candidate.endswith("Z") or candidate.endswith("z"):
            normalized = candidate[:-1] + "+00:00"

        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            if len(candidate) == 10 and is_end:
                parsed = parsed.replace(
                    hour=23, minute=59, second=59, microsecond=999999
                )
            return parsed
        except ValueError:
            pass

        try:
            parsed = datetime.strptime(candidate, "%Y-%m-%d")
            if is_end:
                parsed = parsed + timedelta(days=1) - timedelta(microseconds=1)
            return parsed
        except ValueError:
            return None

    def _should_include_result(
        self,
        result: dict[str, Any],
        allowed_memory_ids: set[str] | None,
        coordinate_bounds: dict[str, float] | None,
    ) -> bool:
        if allowed_memory_ids is not None:
            memory_id = result.get("memory_id")
            if memory_id is None or memory_id not in allowed_memory_ids:
                return False

        if coordinate_bounds:
            start_x = coordinate_bounds.get("start_x")
            end_x = coordinate_bounds.get("end_x")
            x_value = result.get("x")
            if x_value is None:
                x_value = result.get("x_coord")
            if x_value is None:
                return False
            try:
                x_float = float(x_value)
            except (TypeError, ValueError):
                return False
            if start_x is not None and x_float < start_x:
                return False
            if end_x is not None and x_float > end_x:
                return False

        return True

    def _expand_with_relationships(
        self,
        primary_results: list[dict[str, Any]],
        db_manager,
        namespace: str,
        budget: int,
        seen_memory_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Expand search context with related memories from the relationship graph."""

        if budget <= 0 or not primary_results or not db_manager:
            return []

        getter = getattr(db_manager, "get_related_memories", None)
        if not callable(getter):
            return []

        expansion: list[dict[str, Any]] = []
        per_result_limit = max(1, budget // max(1, len(primary_results)))

        for result in primary_results:
            memory_id = result.get("memory_id") if isinstance(result, dict) else None
            if not memory_id:
                continue
            try:
                related = getter(
                    memory_id=memory_id,
                    namespace=namespace,
                    limit=min(per_result_limit, budget),
                    privacy_floor=-10.0,
                )
            except Exception as exc:
                logger.debug(f"Relationship lookup failed for {memory_id}: {exc}")
                continue

            related_ids: list[str] = []
            for related_item in related or []:
                if not isinstance(related_item, dict):
                    continue
                related_id = related_item.get("memory_id")
                if not related_id or related_id in seen_memory_ids:
                    continue
                self._normalize_result(related_item)
                related_item.setdefault("search_strategy", "relationship_graph")
                related_item.setdefault(
                    "search_reasoning",
                    related_item.get("relationship_reason")
                    or f"Related to {memory_id}",
                )
                expansion.append(related_item)
                seen_memory_ids.add(related_id)
                related_ids.append(related_id)
                budget -= 1
                if budget <= 0:
                    break

            if related_ids:
                existing = result.get("related_memories")
                if isinstance(existing, list):
                    merged = existing + [
                        rid for rid in related_ids if rid not in existing
                    ]
                    result["related_memories"] = merged
                else:
                    result["related_memories"] = related_ids

            if budget <= 0:
                break

        return expansion

    def _embed_text(self, text: str) -> list[float]:
        """Compute embedding for text using shared embedding utilities."""
        return generate_embedding(text)

    def _calculate_recency_score(self, created_at) -> float:
        """Calculate recency score (0-1, newer = higher)."""
        try:
            if not created_at:
                return 0.0
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            days_old = (datetime.now() - created_at).days
            return max(0.0, 1 - (days_old / 30))
        except Exception:
            return 0.0

    def _normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Preserve spatial coordinates when normalizing search results."""
        for axis in ("x", "y", "z"):
            coord_key = f"{axis}_coord"
            if coord_key in result and axis not in result:
                result[axis] = result[coord_key]
        return result

    def _suggest_query(self, query_terms: list[str]) -> str:
        """Generate a simple suggestion to refine search queries."""
        if query_terms:
            return f"Try: anchor:{query_terms[0]}"
        return "Use simpler keywords"

    def _gravity_score(
        self,
        result: dict[str, Any],
        query_embedding: list[float],
        query_terms: list[str],
        weights: dict[str, float],
    ) -> float:
        """Calculate gravity score for ranking search results."""
        importance = result.get("importance_score", 0.0)

        emotion = (
            result.get("emotion_intensity")
            or result.get("emotional_intensity")
            or result.get("emotion_score")
        )
        processed = result.get("processed_data")
        if emotion is None and processed:
            if isinstance(processed, str):
                try:
                    processed = json.loads(processed)
                except Exception:
                    processed = {}
            if isinstance(processed, dict):
                emotion = processed.get("emotional_intensity") or processed.get(
                    "emotion_intensity"
                )
        emotion = emotion or 0.0

        anchor_score = 0.0
        anchors = []
        if processed and isinstance(processed, dict):
            anchors = (
                processed.get("symbolic_anchors")
                or processed.get("symbolic_anchor")
                or []
            )
        if isinstance(anchors, str):
            anchors = [anchors]
        anchors_lower = [str(a).lower() for a in anchors]
        if query_terms and anchors_lower:
            if any(term in anchors_lower for term in query_terms):
                anchor_score = 1.0

        recency = self._calculate_recency_score(result.get("created_at"))

        text = (
            result.get("summary") or result.get("text") or result.get("content") or ""
        )
        mem_embedding = normalize_embedding(result.get("embedding"))
        if self.vector_search_enabled and text and not mem_embedding:
            mem_embedding = generate_embedding(text)
            result["embedding"] = mem_embedding
        if self.vector_search_enabled and query_embedding and mem_embedding:
            semantic = vector_similarity(query_embedding, mem_embedding)
        else:
            semantic = 0.0

        return (
            importance * weights.get("importance", 0.0)
            + anchor_score * weights.get("anchor", 0.0)
            + emotion * weights.get("emotion", 0.0)
            + recency * weights.get("recency", 0.0)
            + semantic
        )

    def _execute_keyword_search(
        self,
        search_plan: MemorySearchQuery,
        db_manager,
        namespace: str,
        limit: int,
        time_filters: dict[str, Any] | None = None,
        allowed_memory_ids: set[str] | None = None,
        coordinate_bounds: dict[str, float] | None = None,
        *,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        keywords = search_plan.entity_filters
        if not keywords:
            keywords = [
                word.strip()
                for word in search_plan.query_text.split()
                if len(word.strip()) > 2
            ]
        search_terms = " ".join(keywords)
        try:
            resp = db_manager.search_memories(
                query=search_terms,
                namespace=namespace,
                limit=limit,
                query_embedding=query_embedding,
                **(time_filters or {}),
            )
            results = resp.get("results", []) if isinstance(resp, dict) else resp

            if not isinstance(results, list):
                logger.warning(f"Search returned non-list result: {type(results)}")
                return []
            valid_results = []
            for result in results:
                if isinstance(result, dict):
                    self._normalize_result(result)
                    if not self._should_include_result(
                        result, allowed_memory_ids, coordinate_bounds or {}
                    ):
                        continue
                    valid_results.append(result)
                else:
                    logger.warning(f"Search returned non-dict item: {type(result)}")
            if not valid_results:
                fuzzy_resp = db_manager.search_memories(
                    query=search_terms,
                    namespace=namespace,
                    limit=limit,
                    use_fuzzy=True,
                    query_embedding=query_embedding,
                    **(time_filters or {}),
                )
                fuzzy_results = (
                    fuzzy_resp.get("results", [])
                    if isinstance(fuzzy_resp, dict)
                    else fuzzy_resp
                )

                if isinstance(fuzzy_results, list):
                    for result in fuzzy_results:
                        if isinstance(result, dict):
                            self._normalize_result(result)
                            if not self._should_include_result(
                                result, allowed_memory_ids, coordinate_bounds or {}
                            ):
                                continue
                            valid_results.append(result)
                        else:
                            logger.warning(
                                f"Fuzzy search returned non-dict item: {type(result)}"
                            )
            return valid_results
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def _execute_category_search(
        self,
        search_plan: MemorySearchQuery,
        db_manager,
        namespace: str,
        limit: int,
        time_filters: dict[str, Any] | None = None,
        allowed_memory_ids: set[str] | None = None,
        coordinate_bounds: dict[str, float] | None = None,
        *,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        categories = (
            [cat.value for cat in search_plan.category_filters]
            if search_plan.category_filters
            else []
        )
        if not categories:
            return []

        resp = db_manager.search_memories(
            query="",
            namespace=namespace,
            limit=limit * 3,
            query_embedding=query_embedding,
            **(time_filters or {}),
        )
        results = resp.get("results", []) if isinstance(resp, dict) else resp or []
        if not isinstance(results, list):
            logger.warning(f"Category search returned non-list result: {type(results)}")
            return []

        filtered_results = []
        for result in results:
            try:
                if "processed_data" in result:
                    processed_data = result["processed_data"]
                    if isinstance(processed_data, str):
                        processed_data = json.loads(processed_data)
                    elif not isinstance(processed_data, dict):
                        continue
                    memory_category = processed_data.get("category", {}).get(
                        "primary_category", ""
                    )
                    if memory_category in categories:
                        self._normalize_result(result)
                        if not self._should_include_result(
                            result, allowed_memory_ids, coordinate_bounds or {}
                        ):
                            continue
                        filtered_results.append(result)
                elif result.get("category") in categories:
                    self._normalize_result(result)
                    if not self._should_include_result(
                        result, allowed_memory_ids, coordinate_bounds or {}
                    ):
                        continue
                    filtered_results.append(result)
            except (json.JSONDecodeError, KeyError, AttributeError):
                continue
        return filtered_results[:limit]

    def _execute_importance_search(
        self,
        search_plan: MemorySearchQuery,
        db_manager,
        namespace: str,
        limit: int,
        time_filters: dict[str, Any] | None = None,
        allowed_memory_ids: set[str] | None = None,
        coordinate_bounds: dict[str, float] | None = None,
        *,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        min_importance = max(search_plan.min_importance, 0.7)
        resp = db_manager.search_memories(
            query="",
            namespace=namespace,
            limit=limit * 2,
            query_embedding=query_embedding,
            **(time_filters or {}),
        )

        results = resp.get("results", []) if isinstance(resp, dict) else resp or []
        if not isinstance(results, list):
            logger.warning(
                f"Importance search returned non-list result: {type(results)}"
            )
            return []

        high_importance_results = []
        for result in results:
            if result.get("importance_score", 0) >= min_importance:
                self._normalize_result(result)
                if not self._should_include_result(
                    result, allowed_memory_ids, coordinate_bounds or {}
                ):
                    continue
                high_importance_results.append(result)
        return high_importance_results[:limit]
