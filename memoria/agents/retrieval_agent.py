"""
Memory Search Engine - Intelligent memory retrieval using Pydantic models
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from functools import partial
from typing import Any

import openai
from loguru import logger

from ..core.providers import (
    ProviderConfig,
    ProviderRegistry,
    ProviderSelection,
    ProviderUnavailableError,
    TaskRouteSpec,
)
from ..utils import get_cluster_activity
from ..utils.pydantic_models import AgentPermissions, MemorySearchQuery
from .search_executor import SearchExecutor
from .search_planner import SearchPlanner


class MemorySearchEngine:
    """
    Pydantic-based search engine for intelligent memory retrieval.
    Uses OpenAI Structured Outputs to understand queries and plan searches.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        provider_config: ProviderConfig | None = None,
        permissions: AgentPermissions | None = None,
        provider_registry: ProviderRegistry | None = None,
        task_routes: Mapping[str, TaskRouteSpec] | None = None,
        planner_factory: type[SearchPlanner] | None = None,
    ):
        """
        Initialize Memory Search Engine with LLM provider configuration

        Args:
            api_key: API key (deprecated, use provider_config)
            model: Model to use for query understanding (defaults to 'gpt-4o-mini' if not specified)
            provider_config: Provider configuration for LLM client
        """
        if provider_config:
            self.client = provider_config.create_client()
            self.model = model or provider_config.model or "gpt-4o-mini"
            logger.debug(f"Search engine initialized with model: {self.model}")
            self.provider_config = provider_config
        else:
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model or "gpt-4o-mini"
            self.provider_config = None

        self.permissions = permissions or AgentPermissions()
        self.provider_registry = provider_registry
        self.task_routes = dict(task_routes or {})
        self.conversation_manager: Any | None = None
        self._planner_factory = planner_factory or SearchPlanner
        self._planner_cache: dict[str, SearchPlanner] = {}
        self._last_route: ProviderSelection | None = None

        # Core components
        self.search_planner = self._planner_factory(
            self.client, self.model, self.provider_config, permissions=self.permissions
        )
        self.search_executor = SearchExecutor(permissions=self.permissions)

        # Background processing
        self._background_executor = None
        self._task_name = "search_planning"

    def set_conversation_manager(self, conversation_manager: Any) -> None:
        """Attach a conversation manager for routing telemetry."""

        self.conversation_manager = conversation_manager

    @property
    def last_route(self) -> ProviderSelection | None:
        """Return the most recent provider selection for search planning."""

        return self._last_route

    def _preferred_identifier_from_session(self, session_id: str | None) -> str | None:
        if session_id is None or self.conversation_manager is None:
            return None
        getter = getattr(self.conversation_manager, "get_last_model_route", None)
        if not callable(getter):
            return None
        record = getter(session_id, self._task_name)
        if isinstance(record, Mapping) and record.get("success"):
            provider = record.get("provider")
            model = record.get("model")
            if provider:
                return f"{provider}:{model}" if model else str(provider)
        return None

    def _record_route_usage(
        self,
        session_id: str | None,
        selection: ProviderSelection | None,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        if self.conversation_manager is None:
            return
        recorder = getattr(self.conversation_manager, "record_model_route", None)
        if not callable(recorder):
            return

        provider_name = selection.provider_name if selection else "default"
        model_name = (selection.model if selection else self.model) or self.model
        fallback_used = bool(selection.fallback_used) if selection else False

        try:
            recorder(
                session_id or "",
                self._task_name,
                provider_name,
                model_name,
                fallback_used,
                success=success,
                error=error,
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to record search planner routing", exc_info=True)

    def _select_planner_instance(
        self,
        *,
        session_id: str | None,
        attempted: set[str] | None = None,
    ) -> tuple[ProviderSelection | None, SearchPlanner]:
        if not self.provider_registry:
            self._last_route = None
            return None, self.search_planner

        preferred_identifier = self._preferred_identifier_from_session(session_id)
        selection = self.provider_registry.select(
            self._task_name,
            require_structured=False,
            preferred_provider=preferred_identifier,
            exclude_providers=attempted or (),
        )

        if selection is None:
            self._last_route = None
            return None, self.search_planner

        planner = self._planner_cache.get(selection.provider_name)
        if planner is None:
            try:
                client = selection.get_client()
            except ProviderUnavailableError as exc:
                if attempted is not None:
                    attempted.add(selection.provider_name.lower())
                if self.provider_registry:
                    self.provider_registry.mark_unavailable(selection.provider_name)
                self._record_route_usage(
                    session_id,
                    selection,
                    success=False,
                    error=str(exc),
                )
                raise

            planner = self._planner_factory(
                client,
                selection.model or self.model,
                self.provider_config,
                permissions=self.permissions,
            )
            if hasattr(planner, "set_provider_type"):
                try:
                    planner.set_provider_type(selection.provider_type)
                except Exception:  # pragma: no cover - defensive
                    pass
            elif hasattr(planner, "provider_type"):
                try:
                    planner.provider_type = selection.provider_type
                except Exception:  # pragma: no cover - defensive
                    pass
            self._planner_cache[selection.provider_name] = planner
        else:
            if hasattr(planner, "provider_type"):
                try:
                    planner.provider_type = selection.provider_type
                except Exception:  # pragma: no cover - defensive
                    pass
            if selection.model:
                planner.model = selection.model

        self._last_route = selection
        return selection, planner

    def plan_search(
        self,
        query: str,
        context: str | None = None,
        *,
        session_id: str | None = None,
    ) -> MemorySearchQuery:
        """Delegate search planning to the planner component with routing."""

        attempted: set[str] = set()
        last_error: str | None = None

        while True:
            try:
                selection, planner = self._select_planner_instance(
                    session_id=session_id, attempted=attempted
                )
            except ProviderUnavailableError:
                # Provider already recorded and marked; try next candidate
                continue

            try:
                search_query = planner.plan_search(query, context)
                self._last_route = selection
                self._record_route_usage(session_id, selection, success=True)
                return search_query
            except ProviderUnavailableError as exc:
                last_error = str(exc)
                if selection is None:
                    self._record_route_usage(
                        session_id, selection, success=False, error=last_error
                    )
                    break

                attempted.add(selection.provider_name.lower())
                if self.provider_registry:
                    self.provider_registry.mark_unavailable(selection.provider_name)
                self._record_route_usage(
                    session_id, selection, success=False, error=last_error
                )
                continue
            except Exception as e:
                last_error = str(e)
                self._record_route_usage(
                    session_id, selection, success=False, error=last_error
                )
                raise

            if selection is None:
                break

        # Fallback to default planner when routing fails
        search_query = self.search_planner.plan_search(query, context)
        self._last_route = None
        self._record_route_usage(session_id, None, success=True, error=last_error)
        return search_query

    def execute_search(
        self,
        query: str,
        db_manager,
        namespace: str = "default",
        limit: int = 10,
        rank_weights: dict[str, float] | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute intelligent search using planned strategies.

        Returns a structured dictionary containing:

        - ``results``: list of memory dictionaries including search metadata
          and any spatial coordinates (``x``, ``y``, ``z``) if available.
        - ``hint``: optional hint derived from the failure mode to help the
          caller recover.
        - ``error``: error message if the search failed.
        """
        try:
            search_plan = self.plan_search(query, session_id=session_id)
            logger.debug(
                f"Search plan for '{query}': strategies={search_plan.search_strategy}, entities={search_plan.entity_filters}"
            )
            results = self.search_executor.execute_search(
                query,
                search_plan,
                db_manager,
                namespace,
                limit,
                rank_weights=rank_weights,
            )
            return {"results": results, "hint": None, "error": None}
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            hint = self._derive_hint_from_exception(e)
            return {"results": [], "hint": hint, "error": str(e)}

    async def execute_search_async(
        self,
        query: str,
        db_manager,
        namespace: str = "default",
        limit: int = 10,
        rank_weights: dict[str, float] | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Async wrapper around :meth:`execute_search`."""
        try:
            loop = asyncio.get_event_loop()
            func = partial(
                self.execute_search,
                query,
                db_manager,
                namespace,
                limit,
                rank_weights,
                session_id=session_id,
            )
            return await loop.run_in_executor(
                self._background_executor,
                func,
            )
        except Exception as e:
            logger.error(f"Async search execution failed: {e}")
            hint = self._derive_hint_from_exception(e)
            return {"results": [], "hint": hint, "error": str(e)}

    def execute_search_background(
        self,
        query: str,
        db_manager,
        namespace: str = "default",
        limit: int = 10,
        callback=None,
        rank_weights: dict[str, float] | None = None,
        *,
        session_id: str | None = None,
    ):
        """
        Execute search in background thread for non-blocking operation

        Args:
            query: Search query
            db_manager: Database manager
            namespace: Memory namespace
            limit: Max results
            callback: Optional callback function to handle results
        """

        def _background_search():
            try:
                result = self.execute_search(
                    query,
                    db_manager,
                    namespace,
                    limit,
                    rank_weights,
                    session_id=session_id,
                )
                if callback:
                    callback(result)
                return result
            except Exception as e:
                logger.error(f"Background search failed: {e}")
                error_result = {
                    "results": [],
                    "hint": self._derive_hint_from_exception(e),
                    "error": str(e),
                }
                if callback:
                    callback(error_result)
                return error_result

        # Start background thread
        thread = threading.Thread(target=_background_search, daemon=True)
        thread.start()
        return thread

    def suggest_related_memories(
        self,
        recent_context: str,
        db_manager,
        namespace: str = "default",
        limit: int = 3,
        callback=None,
        rank_weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Suggest related memories based on recent conversation context.

        Designed to run after each user/assistant turn, this method surfaces
        potentially useful memories that could bridge the current discussion
        with past context. The hosting application may provide a callback to
        receive the suggestions or use the returned list to attach metadata to
        messages.

        Args:
            recent_context: Latest conversation text to base the search on.
            db_manager: Database manager instance.
            namespace: Memory namespace to search in.
            limit: Maximum number of suggested memories.
            callback: Optional function that receives the suggestions list.
            rank_weights: Optional ranking weights for the search executor.

        Returns:
            list of memory dictionaries. Empty list on failure.
        """
        if not recent_context:
            if callback:
                try:
                    callback([])
                except Exception as cb_exc:
                    logger.error(f"Suggestion callback failed: {cb_exc}")
            return []

        try:
            response = self.execute_search(
                recent_context,
                db_manager,
                namespace=namespace,
                limit=limit,
                rank_weights=rank_weights,
            )
            suggestions = response.get("results", [])
            if callback:
                try:
                    callback(suggestions)
                except Exception as cb_exc:
                    logger.error(f"Suggestion callback failed: {cb_exc}")
            return suggestions
        except Exception as e:
            logger.error(f"Related memory suggestion failed: {e}")
            if callback:
                try:
                    callback([])
                except Exception as cb_exc:
                    logger.error(f"Suggestion callback failed: {cb_exc}")
            return []

    def get_cluster_activity(
        self, top_n: int = 5, fading_threshold: float = 0.3
    ) -> dict[str, list[dict[str, Any]]]:
        """Return active and fading clusters from prebuilt index."""
        try:
            return get_cluster_activity(top_n=top_n, fading_threshold=fading_threshold)
        except Exception as e:
            logger.error(f"Cluster activity retrieval failed: {e}")
            return {"active": [], "fading": []}

    def search_memories(
        self, query: str, max_results: int = 5, namespace: str = "default"
    ) -> list[dict[str, Any]]:
        """
        Simple search interface for compatibility with memory tools

        Args:
            query: Search query
            max_results: Maximum number of results
            namespace: Memory namespace

        Returns:
            list of memory search results
        """
        # This is a compatibility method that uses the database manager directly
        # We'll need the database manager to be injected or passed
        # For now, return empty list and log the issue
        logger.warning(f"search_memories called without database manager: {query}")
        return []

    @staticmethod
    def _derive_hint_from_exception(exc: Exception) -> str | None:
        """Return a user-friendly hint based on the exception type."""
        if isinstance(exc, ValueError):
            return "Try keyword search: 'keyword:<term>'"
        if isinstance(exc, SyntaxError):
            return "Syntax: anchor:<label>"
        return None


def create_retrieval_agent(
    memoria_instance=None, api_key: str = None, model: str = "gpt-4o-mini"
) -> MemorySearchEngine:
    """
    Create a retrieval agent instance

    Args:
        memoria_instance: Optional Memoria instance for direct database access
        api_key: OpenAI API key
        model: Model to use for query planning

    Returns:
        MemorySearchEngine instance
    """
    agent = MemorySearchEngine(api_key=api_key, model=model)
    if memoria_instance:
        agent._memoria_instance = memoria_instance
    return agent


def smart_memory_search(query: str, memoria_instance, limit: int = 5) -> str:
    """
    Direct string-based memory search function that uses intelligent retrieval

    Args:
        query: Search query string
        memoria_instance: Memoria instance with database access
        limit: Maximum number of results

    Returns:
        Formatted string with search results
    """
    try:
        # Create search engine
        search_engine = MemorySearchEngine()

        # Execute intelligent search
        search_response = search_engine.execute_search(
            query=query,
            db_manager=memoria_instance.db_manager,
            namespace=memoria_instance.namespace,
            limit=limit,
        )

        if search_response.get("error"):
            hint = search_response.get("hint")
            msg = f"Error in smart memory search: {search_response['error']}"
            if hint:
                msg += f" Hint: {hint}"
            return msg

        results = search_response.get("results", [])
        if not results:
            return f"No relevant memories found for query: '{query}'"

        # Format results as a readable string
        output = f"🔍 Smart Memory Search Results for: '{query}'\n\n"

        for i, result in enumerate(results, 1):
            try:
                # Try to parse processed data for better formatting
                if "processed_data" in result:
                    import json

                    processed_data = result["processed_data"]
                    # Handle both dict and JSON string formats
                    if isinstance(processed_data, str):
                        processed_data = json.loads(processed_data)
                    elif isinstance(processed_data, dict):
                        pass  # Already a dict, use as-is
                    else:
                        # Fallback to basic result fields
                        summary = result.get(
                            "summary",
                            result.get("searchable_content", "")[:100] + "...",
                        )
                        category = result.get("category_primary", "unknown")
                        continue

                    summary = processed_data.get("summary", "")
                    category = processed_data.get("category", {}).get(
                        "primary_category", ""
                    )
                else:
                    summary = result.get(
                        "summary", result.get("searchable_content", "")[:100] + "..."
                    )
                    category = result.get("category_primary", "unknown")

                importance = result.get("importance_score", 0.0)
                created_at = result.get("created_at", "")
                search_strategy = result.get("search_strategy", "unknown")
                search_reasoning = result.get("search_reasoning", "")

                output += f"{i}. [{category.upper()}] {summary}\n"
                output += f"   📊 Importance: {importance:.2f} | 📅 {created_at}\n"
                output += f"   🔍 Strategy: {search_strategy}\n"

                if search_reasoning:
                    output += f"   🎯 {search_reasoning}\n"

                output += "\n"

            except Exception:
                # Fallback formatting
                content = result.get("searchable_content", "Memory content available")[
                    :100
                ]
                output += f"{i}. {content}...\n\n"

        return output.strip()

    except Exception as e:
        logger.error(f"Smart memory search failed: {e}")
        return f"Error in smart memory search: {str(e)}"
