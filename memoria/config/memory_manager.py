"""
MemoryManager - Modular memory management system for Memoria

This is a working implementation that coordinates interceptors and provides
a clean interface for memory management operations.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..config.settings import IngestMode
from ..utils.pydantic_models import AgentPermissions
from .manager import ConfigManager

if TYPE_CHECKING:
    from ..schemas import PersonalMemoryDocument, PersonalMemoryEntry

# Interceptor system removed - using LiteLLM native callbacks only


class MemoryManager:
    """
    Modular memory management system that coordinates interceptors,
    memory processing, and context injection.

    This class provides a clean interface for memory operations while
    maintaining backward compatibility with the existing Memoria system.
    """

    def __init__(
        self,
        database_connect: str = "sqlite:///memoria.db",
        template: str = "basic",
        mem_prompt: str | None = None,
        conscious_ingest: bool = False,
        auto_ingest: bool = False,
        namespace: str | None = None,
        shared_memory: bool = False,
        memory_filters: Mapping[str, Any] | None = None,
        user_id: str | None = None,
        verbose: bool = False,
        provider_config: Any | None = None,
        # Additional parameters for compatibility
        openai_api_key: str | None = None,
        api_key: str | None = None,
        api_type: str | None = None,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        azure_ad_token: str | None = None,
        organization: str | None = None,
        sovereign_ingest: bool = False,
        **kwargs,
    ):
        """
        Initialize the MemoryManager.

        Args:
            database_connect: Database connection string
            template: Memory template to use
            mem_prompt: Optional memory prompt
            conscious_ingest: Enable conscious memory ingestion
            auto_ingest: Enable automatic memory ingestion
            namespace: Optional namespace for memory isolation
            shared_memory: Enable shared memory across agents
            memory_filters: Optional mapping of memory filters
            user_id: Optional user identifier
            verbose: Enable verbose logging
            provider_config: Provider configuration
            **kwargs: Additional parameters for forward compatibility
        """
        self.database_connect = database_connect
        self.template = template
        self.mem_prompt = mem_prompt
        self.conscious_ingest = conscious_ingest
        self.auto_ingest = auto_ingest
        self.namespace = namespace
        self.shared_memory = shared_memory
        if memory_filters is None:
            self.memory_filters: dict[str, Any] = {}
        else:
            try:
                self.memory_filters = dict(memory_filters)
            except Exception:
                logger.warning(
                    "memory_filters should be a mapping; received %s. Ignoring values.",
                    type(memory_filters).__name__,
                )
                self.memory_filters = {}
        self.user_id = user_id
        self.verbose = verbose
        self.provider_config = provider_config

        # Store additional configuration
        self.openai_api_key = openai_api_key
        self.api_key = api_key
        self.api_type = api_type
        self.base_url = base_url
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.api_version = api_version
        self.azure_ad_token = azure_ad_token
        self.organization = organization
        self.kwargs = kwargs
        self.sovereign_ingest = sovereign_ingest

        self._session_id = str(uuid.uuid4())
        self._enabled = False

        # Runtime bindings
        self.memoria_instance = None
        self.storage_service = None
        self.db_manager = None

        # LiteLLM native callback manager
        self.litellm_callback_manager = None

        logger.info(f"MemoryManager initialized with session: {self._session_id}")

    def set_memoria_instance(self, memoria_instance):
        """Set the parent Memoria instance for memory management."""
        self.memoria_instance = memoria_instance
        self.storage_service = getattr(memoria_instance, "storage_service", None)
        self.db_manager = getattr(memoria_instance, "db_manager", self.db_manager)

        # Respect sovereign ingest flag by skipping integration setup
        self.sovereign_ingest = getattr(
            memoria_instance, "sovereign_ingest", self.sovereign_ingest
        )

        if self.sovereign_ingest:
            self.litellm_callback_manager = None
            logger.info(
                "Sovereign ingest active; skipping LiteLLM callback initialization"
            )
            return

        # Initialize LiteLLM callback manager only if integration is enabled
        settings = ConfigManager().get_settings()
        if settings.integrations.litellm_enabled:
            try:
                from ..integrations.litellm_integration import (
                    setup_litellm_callbacks,
                )

                self.litellm_callback_manager = setup_litellm_callbacks(
                    memoria_instance
                )
                if self.litellm_callback_manager:
                    logger.debug("LiteLLM callback manager initialized")
                else:
                    logger.warning("Failed to initialize LiteLLM callback manager")
            except ImportError as e:
                logger.warning(f"Could not initialize LiteLLM callback manager: {e}")
        else:
            self.litellm_callback_manager = None
            logger.info("LiteLLM integration disabled; skipping callback setup")

        logger.debug("MemoryManager configured with Memoria instance")

    def get_config_info(self) -> dict[str, Any]:
        """Return configuration metadata for the current session."""

        base_info: dict[str, Any] = {
            "loaded": False,
            "sources": [],
            "version": None,
            "debug_mode": False,
            "is_production": False,
        }

        config_info: Mapping[str, Any] | None = None

        try:
            config_info = ConfigManager().get_config_info()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(f"ConfigManager.get_config_info failed: {exc}")

        if not isinstance(config_info, Mapping) and self.memoria_instance is not None:
            memoria_instance = self.memoria_instance
            memoria_info: Any | None = None

            getter = getattr(memoria_instance, "get_config_info", None)
            if callable(getter):
                try:
                    memoria_info = getter()
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug(
                        f"Memoria.get_config_info retrieval failed; falling back to attribute: {exc}"
                    )

            if memoria_info is None:
                memoria_info = getattr(memoria_instance, "_config_info", None)

            if isinstance(memoria_info, Mapping):
                config_info = memoria_info

        if isinstance(config_info, Mapping):
            result = base_info.copy()
            for key, value in config_info.items():
                if (
                    key == "sources"
                    and isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes))
                ):
                    result[key] = list(value)
                else:
                    result[key] = value
            return result

        return base_info.copy()

    def enable(self, interceptors: list[str] | None = None) -> dict[str, Any]:
        """
        Enable memory recording using LiteLLM native callbacks.

        Args:
            interceptors: Legacy parameter (ignored, using LiteLLM callbacks)

        Returns:
            dict containing enablement results
        """
        if self._enabled:
            return {
                "success": True,
                "message": "Already enabled",
                "enabled_interceptors": ["litellm_native"],
            }

        if self.sovereign_ingest:
            self._enabled = True
            logger.info(
                "MemoryManager running in sovereign ingest mode; callbacks not registered"
            )
            return {
                "success": True,
                "message": "Sovereign ingest active; LiteLLM callbacks bypassed.",
                "enabled_interceptors": [],
            }

        if interceptors is None:
            interceptors = ["litellm_native"]  # Only LiteLLM native callbacks supported

        try:
            # Enable LiteLLM native callback system
            if (
                self.litellm_callback_manager
                and not self.litellm_callback_manager.is_registered
            ):
                success = self.litellm_callback_manager.register_callbacks()
                if not success:
                    return {
                        "success": False,
                        "message": "Failed to register LiteLLM callbacks",
                    }
            else:
                logger.info("LiteLLM integration disabled; no callbacks registered")
                self._enabled = True
                return {
                    "success": True,
                    "message": "LiteLLM integration disabled",
                    "enabled_interceptors": [],
                }

            self._enabled = True

            logger.info("MemoryManager enabled with LiteLLM native callbacks")

            return {
                "success": True,
                "message": "Enabled LiteLLM native callback system",
                "enabled_interceptors": ["litellm_native"],
            }
        except Exception as e:
            logger.error(f"Failed to enable MemoryManager: {e}")
            return {"success": False, "message": str(e)}

    def disable(self) -> dict[str, Any]:
        """
        Disable memory recording using LiteLLM native callbacks.

        Returns:
            dict containing disable results
        """
        if not self._enabled:
            return {"success": True, "message": "Already disabled"}

        if self.sovereign_ingest:
            self._enabled = False
            logger.info(
                "MemoryManager sovereign ingest mode disabled; no callbacks were registered"
            )
            return {
                "success": True,
                "message": "Sovereign ingest deactivated.",
            }

        try:
            # Disable LiteLLM native callback system
            if (
                self.litellm_callback_manager
                and self.litellm_callback_manager.is_registered
            ):
                success = self.litellm_callback_manager.unregister_callbacks()
                if not success:
                    logger.warning("Failed to unregister LiteLLM callbacks")

            self._enabled = False

            logger.info("MemoryManager disabled")

            return {
                "success": True,
                "message": "MemoryManager disabled successfully (LiteLLM native callbacks)",
            }
        except Exception as e:
            logger.error(f"Failed to disable MemoryManager: {e}")
            return {"success": False, "message": str(e)}

    def get_status(self) -> dict[str, dict[str, Any]]:
        """
        Get status of memory recording system.

        Returns:
            dict containing memory system status information
        """
        callback_status = "inactive"
        if self.litellm_callback_manager:
            if self.litellm_callback_manager.is_registered:
                callback_status = "active"
            else:
                callback_status = "available_but_not_registered"
        else:
            callback_status = "unavailable"

        return {
            "litellm_native": {
                "enabled": self._enabled,
                "status": callback_status,
                "method": "litellm_callbacks",
                "session_id": self._session_id,
                "callback_manager": self.litellm_callback_manager is not None,
            }
        }

    def get_health(self) -> dict[str, Any]:
        """
        Get health check of the memory management system.

        Returns:
            dict containing health information
        """
        return {
            "session_id": self._session_id,
            "enabled": self._enabled,
            "namespace": self.namespace,
            "user_id": self.user_id,
            "litellm_callback_manager": self.litellm_callback_manager is not None,
            "litellm_callbacks_registered": (
                self.litellm_callback_manager.is_registered
                if self.litellm_callback_manager
                else False
            ),
            "memory_filters": {
                "configured": bool(self.memory_filters),
                "count": len(self.memory_filters),
                "values": dict(self.memory_filters),
            },
            "conscious_ingest": self.conscious_ingest,
            "auto_ingest": self.auto_ingest,
            "database_connect": self.database_connect,
            "template": self.template,
        }

    # === STORAGE SERVICE ADAPTER METHODS ===
    def store_memory(
        self,
        anchor: str,
        text: str,
        tokens: int,
        timestamp: datetime | None = None,
        x_coord: float | None = None,
        *,
        y: float | None = None,
        z: float | None = None,
        symbolic_anchors: list[str] | None = None,
        permissions: AgentPermissions | None = None,
        emotional_intensity: float | None = None,
        chat_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        documents: Sequence[Mapping[str, Any] | PersonalMemoryDocument] | None = None,
        namespace: str | None = None,
        team_id: str | None = None,
        workspace_id: str | None = None,
        share_with_team: bool | None = None,
        return_status: bool = False,
        ingest_mode: IngestMode | str | None = None,
        last_edited_by_model: str | None = None,
    ) -> str | dict[str, Any]:
        perms = permissions or AgentPermissions()
        if not perms.can_write:
            raise PermissionError("Agent lacks write permission")
        if self.memoria_instance is not None:
            metadata_payload = dict(metadata) if metadata is not None else None
            documents_payload = list(documents) if documents is not None else None
            return self.memoria_instance.store_memory(
                anchor,
                text,
                tokens,
                timestamp=timestamp,
                x_coord=x_coord,
                y=y,
                z=z,
                symbolic_anchors=symbolic_anchors,
                emotional_intensity=emotional_intensity,
                chat_id=chat_id,
                metadata=metadata_payload,
                documents=documents_payload,
                namespace=namespace,
                team_id=team_id,
                share_with_team=share_with_team,
                workspace_id=workspace_id,
                return_status=return_status,
                ingest_mode=ingest_mode,
                last_edited_by_model=last_edited_by_model,
            )

        if not self.storage_service:
            raise RuntimeError("Storage service not configured")
        if ingest_mode is not None:
            try:
                mode_value = (
                    ingest_mode
                    if isinstance(ingest_mode, IngestMode)
                    else IngestMode(str(ingest_mode).strip().lower())
                )
            except ValueError:
                mode_value = None
            if mode_value == IngestMode.PERSONAL:
                raise RuntimeError(
                    "Personal ingest mode requires an attached Memoria instance"
                )
        return self.storage_service.store_memory(
            anchor,
            text,
            tokens,
            timestamp=timestamp,
            x_coord=x_coord,
            y=y,
            z=z,
            symbolic_anchors=symbolic_anchors,
            emotional_intensity=emotional_intensity,
            chat_id=chat_id,
            last_edited_by_model=last_edited_by_model,
        )

    def store_personal_memory(
        self,
        entry: PersonalMemoryEntry | Mapping[str, Any],
        *,
        namespace: str | None = None,
        team_id: str | None = None,
        workspace_id: str | None = None,
        user_id: str | None = None,
        return_status: bool = False,
    ) -> str | dict[str, Any]:
        """Persist a personal memory entry via the configured backend."""

        if self.memoria_instance is not None:
            return self.memoria_instance.store_personal_memory(
                entry,
                namespace=namespace,
                team_id=team_id,
                workspace_id=workspace_id,
                user_id=user_id or self.user_id,
                return_status=return_status,
            )

        if not self.storage_service:
            raise RuntimeError("Storage service not configured")

        result = self.storage_service.store_personal_memory(
            entry,
            namespace=namespace,
            team_id=team_id,
            workspace_id=workspace_id,
            user_id=user_id or self.user_id,
        )

        if return_status:
            payload = dict(result)
            payload.setdefault("status", payload.get("status", "stored"))
            return payload

        memory_id = result.get("memory_id")
        if not memory_id:
            raise RuntimeError("Personal memory storage did not return a memory_id")
        return memory_id

    def delete_memory(
        self, memory_id: str, *, permissions: AgentPermissions | None = None
    ) -> bool:
        perms = permissions or AgentPermissions()
        if not perms.can_edit:
            raise PermissionError("Agent lacks edit permission")
        if not self.storage_service:
            raise RuntimeError("Storage service not configured")
        return self.storage_service.delete_memory(memory_id)

    def retrieve_context(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not self.storage_service:
            return []
        return self.storage_service.retrieve_context(query, limit)

    def retrieve_memories_near(
        self,
        x: float,
        y: float,
        z: float,
        max_distance: float = 5.0,
        anchor: str | list[str] | None = None,
        limit: int = 10,
        dimensions: str = "3d",
    ) -> list[dict[str, Any]]:
        if not self.storage_service:
            return []
        return self.storage_service.retrieve_memories_near(
            x,
            y,
            z,
            max_distance=max_distance,
            anchor=anchor,
            limit=limit,
            dimensions=dimensions,
        )

    def retrieve_memories_by_anchor(self, anchors: list[str]) -> list[dict[str, Any]]:
        if not self.storage_service:
            return []
        return self.storage_service.retrieve_memories_by_anchor(anchors)

    # === BACKWARD COMPATIBILITY PROPERTIES ===

    @property
    def session_id(self) -> str:
        """Get session ID for backward compatibility."""
        return self._session_id

    @property
    def enabled(self) -> bool:
        """Check if enabled for backward compatibility."""
        return self._enabled

    # === PLACEHOLDER METHODS FOR FUTURE MODULAR COMPONENTS ===

    def record_conversation(
        self,
        user_input: str,
        ai_output: str,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Record a conversation.

        If a Memoria instance has been bound to this manager, the call will be
        delegated to that instance so any additional processing pipelines are
        preserved. Otherwise, the conversation is written directly via the
        configured database manager.

        Raises:
            RuntimeError: If neither a Memoria instance nor a database manager
                is available for persistence.
        """
        settings = ConfigManager().get_settings()
        resolved_model = model or settings.agents.default_model
        resolved_metadata: dict[str, Any] = dict(metadata or {})

        memoria_instance = getattr(self, "memoria_instance", None)
        if memoria_instance and hasattr(memoria_instance, "record_conversation"):
            logger.debug("Delegating conversation recording to Memoria instance")
            return memoria_instance.record_conversation(
                user_input=user_input,
                ai_output=ai_output,
                model=resolved_model,
                metadata=resolved_metadata,
            )

        db_manager = getattr(self, "db_manager", None)
        if db_manager and hasattr(db_manager, "store_chat_history"):
            logger.debug("Recording conversation via database manager fallback")
            chat_id = str(uuid.uuid4())
            timestamp = datetime.now()
            ai_text = "" if ai_output is None else str(ai_output)
            namespace = self.namespace or "default"

            db_manager.store_chat_history(
                chat_id=chat_id,
                user_input=user_input,
                ai_output=ai_text,
                timestamp=timestamp,
                session_id=self._session_id,
                model=resolved_model,
                namespace=namespace,
                metadata=resolved_metadata,
                last_edited_by_model=resolved_model,
            )
            return chat_id

        raise RuntimeError(
            "MemoryManager is not configured with a Memoria instance or database manager for persistence"
        )

    def search_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: list[str] | None = None,
        categories: list[str] | None = None,
        min_importance: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search memories using the configured storage backend.
        """
        namespace = (
            self.namespace
            or getattr(getattr(self, "memoria_instance", None), "namespace", None)
            or "default"
        )

        search_kwargs: dict[str, Any] = {
            "query": query,
            "limit": limit,
        }

        if memory_types is not None:
            search_kwargs["memory_types"] = memory_types
        if categories is not None:
            search_kwargs["category_filter"] = categories
        if min_importance is not None:
            search_kwargs["min_importance"] = min_importance

        backend_callable = None

        if self.storage_service is not None:
            storage_namespace = getattr(self.storage_service, "namespace", None)
            search_kwargs["namespace"] = storage_namespace or namespace

            storage_search = getattr(self.storage_service, "search_memories", None)
            if callable(storage_search):
                backend_callable = storage_search
            else:
                storage_db_manager = getattr(self.storage_service, "db_manager", None)
                if storage_db_manager is not None and hasattr(
                    storage_db_manager, "search_memories"
                ):
                    backend_callable = storage_db_manager.search_memories
        else:
            search_kwargs["namespace"] = namespace

        if backend_callable is None:
            db_manager = getattr(
                getattr(self, "memoria_instance", None), "db_manager", None
            )
            if db_manager is None:
                db_manager = getattr(self, "db_manager", None)

            if db_manager is not None and hasattr(db_manager, "search_memories"):
                backend_callable = db_manager.search_memories

        if backend_callable is None:
            logger.warning(
                "search_memories called without a configured search backend: %s",
                query,
            )
            return []

        try:
            raw_results = backend_callable(**search_kwargs)
        except Exception as exc:
            logger.error(f"search_memories failed for query '{query}': {exc}")
            return []

        def _normalize_payload(payload: Any) -> list[dict[str, Any]]:
            if payload is None:
                return []

            if isinstance(payload, dict):
                items = payload.get("results")
                if items is None:
                    items = [payload]
            elif isinstance(payload, Sequence) and not isinstance(
                payload, (str, bytes)
            ):
                items = list(payload)
            else:
                items = [payload]

            normalized: list[dict[str, Any]] = []
            for item in items:
                if item is None:
                    continue
                if isinstance(item, Mapping):
                    normalized.append(dict(item))
                    continue
                dict_method = getattr(item, "dict", None)
                if callable(dict_method):
                    normalized.append(dict_method())
                    continue
                asdict_method = getattr(item, "_asdict", None)
                if callable(asdict_method):
                    normalized.append(dict(asdict_method()))
                    continue
                logger.debug(
                    "Skipping non-mapping search result item for query '%s': %r",
                    query,
                    item,
                )
            return normalized

        return _normalize_payload(raw_results)

    def cleanup(self):
        """Cleanup resources."""
        try:
            if self._enabled:
                self.disable()

            # Clean up callback manager
            if self.litellm_callback_manager:
                self.litellm_callback_manager.unregister_callbacks()
                self.litellm_callback_manager = None

            logger.info("MemoryManager cleanup completed")
        except Exception as e:
            logger.error(f"Error during MemoryManager cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __del__(self):
        """Destructor - ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
