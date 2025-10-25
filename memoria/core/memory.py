"""
Main Memoria class - Pydantic-based memory interface v1.0
"""

from __future__ import annotations

import asyncio
import copy
import os
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    cast,
    Protocol,
)

from loguru import logger
from pydantic import ValidationError

try:
    import litellm  # noqa: F401
    from litellm import success_callback  # noqa: F401

    LITELLM_AVAILABLE = True
except Exception as e:  # pragma: no cover - defensive import guard
    LITELLM_AVAILABLE = False
    logger.warning(
        f"LiteLLM not available - native callback system disabled ({e})"
    )

from ..agents.conscious_agent import ConsciousAgent
from ..config import ConfigManager
from ..config.memory_manager import MemoryManager
from ..conscious.manager import ConsciousManager
from ..database.sqlalchemy_manager import SQLAlchemyDatabaseManager as DatabaseManager
from ..database.models import ChatHistory
from ..heuristics.conversation_ingest import (
    HeuristicConversationResult,
    process_conversation_turn,
)
from ..heuristics.manual_promotion import PromotionDecision, score_staged_memory
from ..heuristics.ingestion import (
    DailyIngestionScheduler,
    ShortTermIngestionService,
)
from ..heuristics.retention import (
    MemoryRetentionScheduler,
    MemoryRetentionService,
    RetentionConfig,
    RetentionPolicyRule,
)
from ..schemas import (
    MemoryImageAsset,
    PersonalMemoryDocument,
    PersonalMemoryEntry,
    ThreadIngestion,
    canonicalize_symbolic_anchors,
)
from ..policy import EnforcementStage, PolicyDecision, RedactionResult, apply_redactions
from ..storage.service import StorageService
from ..utils.exceptions import MemoriaError
from ..utils.pydantic_compat import model_dump, model_validate
from ..utils.pydantic_models import (
    ConversationContext,
    MemoryClassification,
    MemoryImportanceLevel,
    ProcessedLongTermMemory,
)
from ..sync import SyncCoordinator, SyncEvent, create_sync_backend
from ..config.settings import CoordinateAuditCadence, IngestMode, TeamMode
from .context_injection import (
    inject_anthropic_context,
    inject_gemini_context,
    inject_litellm_context,
    inject_openai_context,
)
from .context_orchestration import ContextOrchestrator, ContextOrchestrationConfig, ContextPlan
from .conversation import ConversationManager
from .initialization import setup_database, setup_logging
from .interceptors import (
    disable_interceptor as disable_interceptor_impl,
)
from .interceptors import (
    enable_interceptor as enable_interceptor_impl,
)
from ..plugins import PluginRegistry, load_plugins
from .provider_setup import ProviderSetupInputs, ProviderSetupResult, setup_provider_components
from ..rituals.coordinate_audit import CoordinateAuditJob, CoordinateAuditScheduler

if TYPE_CHECKING:
    from asyncio import Task

    from ..agents.memory_agent import MemoryAgent
    from ..agents.retrieval_agent import MemorySearchEngine
    from ..config.settings import PluginSettings
    from ..plugins import BasePlugin
    from ..config.settings import SyncSettings
    from ..sync import SyncBackend


class MemoryManagerProtocol(Protocol):
    _enabled: bool

    def enable(self, interceptors: list[str] | None = None) -> dict[str, Any]:
        ...

    def disable(self) -> dict[str, Any]:
        ...

    def get_status(self) -> dict[str, dict[str, Any]]:
        ...

    def get_health(self) -> dict[str, Any]:
        ...

    def set_memoria_instance(self, memoria: "Memoria") -> None:
        ...


class ConversationManagerProtocol(Protocol):
    def get_session_stats(self) -> dict[str, Any]:
        ...

    def clear_session(self, session_id: str) -> None:
        ...

    def clear_all_sessions(self) -> None:
        ...


InjectContextFn = Callable[["Memoria", dict[str, Any]], dict[str, Any]]
_inject_openai_context: InjectContextFn = inject_openai_context
_inject_anthropic_context: InjectContextFn = inject_anthropic_context
_inject_gemini_context: InjectContextFn = inject_gemini_context
_inject_litellm_context: Callable[["Memoria", dict[str, Any], str], dict[str, Any]] = (
    inject_litellm_context
)

EnableInterceptorFn = Callable[["Memoria", str | None], bool]
_enable_interceptor_impl: EnableInterceptorFn = enable_interceptor_impl
_disable_interceptor_impl: EnableInterceptorFn = disable_interceptor_impl


def build_provider_options(settings: Any | None) -> dict[str, object | None]:
    """Collect provider-related options from settings and environment variables."""

    memory_settings = getattr(settings, "memory", None) if settings is not None else None
    agent_settings = getattr(settings, "agents", None) if settings is not None else None

    conscious_ingest = bool(
        getattr(agent_settings, "conscious_ingest", False) if agent_settings else False
    )
    context_injection = bool(
        getattr(memory_settings, "context_injection", False) if memory_settings else False
    )
    sovereign_ingest = bool(
        getattr(memory_settings, "sovereign_ingest", False) if memory_settings else False
    )
    namespace = getattr(memory_settings, "namespace", None) if memory_settings else None
    shared_memory = bool(
        getattr(memory_settings, "shared_memory", False) if memory_settings else False
    )
    team_memory_enabled = bool(
        getattr(memory_settings, "team_memory_enabled", False)
        if memory_settings
        else False
    )
    team_namespace_prefix = (
        getattr(memory_settings, "team_namespace_prefix", "team")
        if memory_settings
        else "team"
    )
    team_enforce_membership = bool(
        getattr(memory_settings, "team_enforce_membership", True)
        if memory_settings
        else True
    )
    team_mode = (
        getattr(memory_settings, "team_mode", TeamMode.DISABLED)
        if memory_settings
        else TeamMode.DISABLED
    )
    default_team_id = (
        getattr(memory_settings, "team_default_id", None)
        if memory_settings
        else None
    )
    team_share_by_default = bool(
        getattr(memory_settings, "team_share_by_default", False)
        if memory_settings
        else False
    )

    configured_openai_key = (
        getattr(agent_settings, "openai_api_key", None) if agent_settings else None
    )
    openai_api_key = configured_openai_key or os.getenv("OPENAI_API_KEY")
    provider_api_key = os.getenv("AZURE_OPENAI_API_KEY") or openai_api_key
    api_type = os.getenv("OPENAI_API_TYPE")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv(
        "AZURE_OPENAI_DEPLOYMENT"
    )
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_ad_token = os.getenv("AZURE_OPENAI_AD_TOKEN")
    organization = os.getenv("OPENAI_ORGANIZATION")
    project = os.getenv("OPENAI_PROJECT")
    model_name = getattr(agent_settings, "default_model", None) if agent_settings else None

    anthropic_api_key = getattr(agent_settings, "anthropic_api_key", None)
    if not anthropic_api_key:
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_base_url = (
        getattr(agent_settings, "anthropic_base_url", None) if agent_settings else None
    )
    if not anthropic_base_url:
        anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL") or os.getenv(
            "ANTHROPIC_API_URL"
        )
    anthropic_model = getattr(agent_settings, "anthropic_model", None) if agent_settings else None
    if not anthropic_model:
        anthropic_model = os.getenv("ANTHROPIC_MODEL") or os.getenv("CLAUDE_MODEL")

    gemini_api_key = getattr(agent_settings, "gemini_api_key", None)
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    gemini_model = getattr(agent_settings, "gemini_model", None) if agent_settings else None
    if not gemini_model:
        gemini_model = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_GEMINI_MODEL")

    plugin_settings = getattr(settings, "plugins", None) if settings is not None else None
    sync_settings_cfg = getattr(settings, "sync", None) if settings is not None else None

    return {
        "conscious_ingest": conscious_ingest,
        "auto_ingest": context_injection,
        "namespace": namespace,
        "shared_memory": shared_memory,
        "team_memory_enabled": team_memory_enabled,
        "team_namespace_prefix": team_namespace_prefix,
        "team_enforce_membership": team_enforce_membership,
        "team_mode": team_mode,
        "default_team_id": default_team_id,
        "team_share_by_default": team_share_by_default,
        "team_id": default_team_id,
        "openai_api_key": openai_api_key,
        "api_key": provider_api_key,
        "api_type": api_type,
        "base_url": base_url,
        "azure_endpoint": azure_endpoint,
        "azure_deployment": azure_deployment,
        "api_version": azure_api_version,
        "azure_ad_token": azure_ad_token,
        "organization": organization,
        "project": project,
        "model": model_name,
        "anthropic_api_key": anthropic_api_key,
        "anthropic_base_url": anthropic_base_url,
        "anthropic_model": anthropic_model,
        "gemini_api_key": gemini_api_key,
        "gemini_model": gemini_model,
        "sovereign_ingest": sovereign_ingest,
        "plugin_settings": plugin_settings,
        "sync_settings": sync_settings_cfg,
    }


class Memoria:
    """
    The main Memoria memory layer for AI agents.

    Provides persistent memory storage, categorization, and retrieval
    for AI conversations and agent interactions.
    """

    @staticmethod
    def _clean_team_identifier(team_id: str | None) -> str | None:
        if team_id is None or not isinstance(team_id, str):
            return None
        cleaned = team_id.strip()
        return cleaned or None

    @staticmethod
    def _normalize_team_mode(
        value: Any, *, default: TeamMode = TeamMode.DISABLED
    ) -> TeamMode:
        if isinstance(value, TeamMode):
            return value
        if value in (None, "", b""):
            return default
        if isinstance(value, bool):
            return TeamMode.OPTIONAL if value else TeamMode.DISABLED
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return default
            if normalized in {"true", "1", "enabled", "on"}:
                return TeamMode.OPTIONAL
            if normalized in {"false", "0", "disabled", "off"}:
                return TeamMode.DISABLED
            try:
                return TeamMode(normalized)
            except ValueError:  # pragma: no cover - defensive
                return default
        return default

    @staticmethod
    def _normalise_coordinate_cadence(
        value: Any,
        *,
        default: CoordinateAuditCadence = CoordinateAuditCadence.WEEKLY,
    ) -> CoordinateAuditCadence:
        if isinstance(value, CoordinateAuditCadence):
            return value
        if value in (None, "", b""):
            return default
        if isinstance(value, bool):
            return (
                CoordinateAuditCadence.WEEKLY if value else CoordinateAuditCadence.DISABLED
            )
        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return default
            if text in {"disabled", "off", "0"}:
                return CoordinateAuditCadence.DISABLED
            if text in {"weekly", "week", "1w"}:
                return CoordinateAuditCadence.WEEKLY
            try:
                return CoordinateAuditCadence(text)
            except ValueError:  # pragma: no cover - defensive
                return default
        return default

    def __init__(
        self,
        database_connect: str = "sqlite:///memoria.db",
        template: str = "basic",
        mem_prompt: str | None = None,
        conscious_ingest: bool | None = None,
        auto_ingest: bool | None = None,
        use_lightweight_conscious_ingest: bool = False,
        enable_short_term: bool = True,
        namespace: str | None = None,
        shared_memory: bool | None = None,
        team_memory_enabled: bool | None = None,
        team_namespace_prefix: str | None = None,
        team_enforce_membership: bool | None = None,
        team_mode: TeamMode | str | bool | None = None,
        default_team_id: str | None = None,
        team_share_by_default: bool | None = None,
        team_id: str | None = None,
        memory_filters: dict[str, Any] | None = None,
        openai_api_key: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        verbose: bool = False,
        sovereign_ingest: bool | None = None,
        # New provider configuration parameters
        api_key: str | None = None,
        api_type: str | None = None,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        azure_ad_token: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        model: str | None = None,  # Allow custom model selection
        provider_config: Any | None = None,  # ProviderConfig when available
        anthropic_config: Any | None = None,
        anthropic_api_key: str | None = None,
        anthropic_base_url: str | None = None,
        anthropic_model: str | None = None,
        gemini_config: Any | None = None,
        gemini_api_key: str | None = None,
        gemini_model: str | None = None,
        schema_init: bool = True,  # Initialize database schema and create tables
        database_prefix: str | None = None,  # Database name prefix
        database_suffix: str | None = None,  # Database name suffix
        plugin_settings: Sequence["PluginSettings"] | None = None,
        sync_settings: "SyncSettings" | None = None,
        sync_backend: "SyncBackend" | None = None,
        coordinate_audit_enabled: bool | None = None,
        coordinate_audit_lookback_days: int | None = None,
        coordinate_audit_cadence: CoordinateAuditCadence | str | None = None,
    ):
        """
        Initialize Memoria memory system v1.0.

        Args:
            database_connect: Database connection string
            template: Memory template to use ('basic')
            mem_prompt: Optional prompt to guide memory recording
            conscious_ingest: Enable one-shot short-term memory context injection at conversation start
                (defaults to configuration value)
            auto_ingest: Enable automatic memory injection on every LLM call (defaults to configuration value)
            use_lightweight_conscious_ingest: Apply lightweight heuristics before copying conscious memories
            enable_short_term: Enable short-term memory table and operations
            namespace: Optional namespace for memory isolation (defaults to configuration value)
            shared_memory: Enable shared memory across agents (defaults to configuration value)
            team_memory_enabled: Enable collaborative team namespaces
            team_namespace_prefix: Override prefix used when creating team namespaces
            team_enforce_membership: Require users to belong to a team before sharing
            team_mode: Override collaborative mode (disabled, optional, or required)
            default_team_id: Team identifier to activate automatically when available
            team_share_by_default: Override default sharing preference for new team spaces
            team_id: Team identifier to set as active during initialization
            memory_filters: Filters for memory ingestion
            openai_api_key: OpenAI API key for memory agent (deprecated, use api_key)
            user_id: Optional user identifier
            verbose: Enable verbose logging (loguru only)
            api_key: API key for the LLM provider
            api_type: Provider type ('openai', 'azure', 'custom')
            base_url: Base URL for custom OpenAI-compatible endpoints
            azure_endpoint: Azure OpenAI endpoint URL
            azure_deployment: Azure deployment name
            api_version: API version for Azure
            azure_ad_token: Azure AD token for authentication
            organization: OpenAI organization ID
            project: OpenAI project ID
            model: Model to use (defaults to configured model if not specified)
            provider_config: Complete provider configuration (overrides individual params)
            anthropic_config: Optional Anthropic provider configuration dataclass
            anthropic_api_key: API key for Anthropic when not providing a config
            anthropic_base_url: Override base URL for Anthropic API requests
            anthropic_model: Default Anthropic model used for context heuristics
            gemini_config: Optional Google Gemini provider configuration dataclass
            gemini_api_key: API key for Gemini when not providing a config
            gemini_model: Default Gemini model to assume for context heuristics
            enable_auto_creation: Enable automatic database creation if database doesn't exist
            database_prefix: Optional prefix for database name (for multi-tenant setups)
            database_suffix: Optional suffix for database name (e.g., 'dev', 'prod', 'test')
            sync_settings: Optional override for cross-instance sync configuration
            sync_backend: Pre-configured backend shared across Memoria instances
        """
        self._config_info: dict[str, Any] = {
            "loaded": False,
            "sources": [],
            "version": None,
            "debug_mode": False,
            "is_production": False,
        }

        settings, effective_sync_settings = self._load_config_info(sync_settings)

        self.database_connect = database_connect
        self.template = template
        self.mem_prompt = mem_prompt
        memory_settings = getattr(settings, "memory", None)
        agent_settings = getattr(settings, "agents", None)
        image_storage_path = (
            getattr(memory_settings, "image_storage_path", None)
            if memory_settings
            else None
        )

        config_ingest_mode = getattr(
            memory_settings, "ingest_mode", IngestMode.STANDARD
        )
        if isinstance(config_ingest_mode, str):
            try:
                config_ingest_mode = IngestMode(config_ingest_mode)
            except ValueError:
                config_ingest_mode = IngestMode.STANDARD
        personal_docs_flag = getattr(
            memory_settings, "personal_documents_enabled", None
        )
        if personal_docs_flag is None:
            personal_docs_flag = config_ingest_mode == IngestMode.PERSONAL

        self.ingest_mode = config_ingest_mode
        self.personal_mode_enabled = config_ingest_mode == IngestMode.PERSONAL
        self.personal_documents_enabled = bool(personal_docs_flag)
        if not self.personal_mode_enabled:
            self.personal_documents_enabled = False

        config_namespace = getattr(memory_settings, "namespace", "default")
        config_shared_memory = getattr(memory_settings, "shared_memory", False)
        config_context_injection = getattr(memory_settings, "context_injection", False)
        config_conscious = getattr(agent_settings, "conscious_ingest", False)
        config_team_enabled = getattr(memory_settings, "team_memory_enabled", False)
        config_team_mode = getattr(memory_settings, "team_mode", TeamMode.DISABLED)
        if team_mode is not None:
            config_team_mode = team_mode
        config_coordinate_enabled = getattr(
            memory_settings, "coordinate_audit_enabled", True
        )
        if coordinate_audit_enabled is not None:
            config_coordinate_enabled = bool(coordinate_audit_enabled)
        config_coordinate_lookback = getattr(
            memory_settings, "coordinate_audit_lookback_days", 7
        )
        if coordinate_audit_lookback_days is not None:
            config_coordinate_lookback = max(1, int(coordinate_audit_lookback_days))
        configured_cadence = getattr(
            memory_settings,
            "coordinate_audit_cadence",
            CoordinateAuditCadence.WEEKLY,
        )
        config_coordinate_cadence = self._normalise_coordinate_cadence(
            configured_cadence
        )
        if coordinate_audit_cadence is not None:
            config_coordinate_cadence = self._normalise_coordinate_cadence(
                coordinate_audit_cadence,
                default=config_coordinate_cadence,
            )
        resolved_team_mode = self._normalize_team_mode(
            config_team_mode,
            default=TeamMode.OPTIONAL if config_team_enabled else TeamMode.DISABLED,
        )
        if team_memory_enabled is not None:
            if team_memory_enabled and resolved_team_mode == TeamMode.DISABLED:
                resolved_team_mode = TeamMode.OPTIONAL
            if not team_memory_enabled:
                resolved_team_mode = TeamMode.DISABLED
        config_team_prefix = getattr(memory_settings, "team_namespace_prefix", "team")
        if team_namespace_prefix:
            config_team_prefix = team_namespace_prefix
        config_team_enforce = getattr(memory_settings, "team_enforce_membership", True)
        if team_enforce_membership is not None:
            config_team_enforce = bool(team_enforce_membership)
        config_share_default = getattr(memory_settings, "team_share_by_default", False)
        if team_share_by_default is not None:
            config_share_default = bool(team_share_by_default)
        config_default_team = getattr(memory_settings, "team_default_id", None)
        if default_team_id is not None:
            config_default_team = default_team_id
        config_workspace_mode = getattr(
            memory_settings, "workspace_mode", TeamMode.DISABLED
        )
        config_workspace_default = getattr(
            memory_settings, "workspace_default_id", None
        )
        requested_team_id = self._clean_team_identifier(team_id)
        initial_team_candidate = requested_team_id

        self.enable_short_term = enable_short_term
        self.use_lightweight_conscious_ingest = use_lightweight_conscious_ingest
        self.context_limit = getattr(memory_settings, "context_limit", 3)
        self._context_orchestration_config = ContextOrchestrationConfig.from_settings(
            memory_settings
        )
        self.context_orchestrator: ContextOrchestrator | None = None

        resolved_conscious = (
            config_conscious if conscious_ingest is None else bool(conscious_ingest)
        )
        resolved_auto_ingest = (
            config_context_injection if auto_ingest is None else bool(auto_ingest)
        )

        self.conscious_ingest = bool(resolved_conscious) and self.enable_short_term
        self.auto_ingest = bool(resolved_auto_ingest) and self.enable_short_term
        self.namespace = namespace or config_namespace
        self.shared_memory = (
            config_shared_memory if shared_memory is None else bool(shared_memory)
        )
        self.coordinate_audit_enabled = bool(config_coordinate_enabled)
        self.coordinate_audit_lookback_days = max(1, int(config_coordinate_lookback))
        self.coordinate_audit_cadence = config_coordinate_cadence
        resolved_workspace_mode = self._normalize_team_mode(
            config_workspace_mode, default=TeamMode.DISABLED
        )
        if resolved_team_mode == TeamMode.DISABLED:
            resolved_workspace_mode = TeamMode.DISABLED
        cleaned_default_workspace = self._clean_team_identifier(
            config_workspace_default
        )

        self.team_mode = resolved_team_mode
        self.team_memory_enabled = resolved_team_mode != TeamMode.DISABLED
        self.team_namespace_prefix = (
            str(config_team_prefix).strip() or "team"
            if isinstance(config_team_prefix, str)
            else "team"
        )
        self.team_enforce_membership = bool(config_team_enforce)
        self.team_share_by_default = (
            bool(config_share_default) if self.team_memory_enabled else False
        )
        self.default_team_id = (
            self._clean_team_identifier(config_default_team)
            if self.team_memory_enabled
            else None
        )
        self.workspace_mode = (
            resolved_workspace_mode if self.team_memory_enabled else TeamMode.DISABLED
        )
        self.workspace_memory_enabled = (
            self.workspace_mode != TeamMode.DISABLED
        )
        self.default_workspace_id = (
            cleaned_default_workspace if self.workspace_memory_enabled else None
        )
        if self.team_memory_enabled and not initial_team_candidate and self.default_team_id:
            initial_team_candidate = self.default_team_id
        self.memory_filters = memory_filters or {}
        self.user_id = user_id
        self.agent_id = agent_id
        self.agent_profile: dict[str, Any] | None = None
        self.verbose = verbose
        self.schema_init = schema_init
        self.database_prefix = database_prefix
        self.database_suffix = database_suffix
        self.personal_namespace = self.namespace
        self._active_team_id: str | None = None
        self._last_ingestion_report: list[dict[str, Any]] = []
        self.ingestion_service: ShortTermIngestionService | None = None
        self._ingestion_scheduler: DailyIngestionScheduler | None = None
        self._coordinate_audit_job: CoordinateAuditJob | None = None
        self._coordinate_audit_scheduler: CoordinateAuditScheduler | None = None
        self._scheduled_job_completion_flags: dict[str, threading.Event] = {}

        self.db_manager: DatabaseManager
        self.storage_service: StorageService

        # Initialize database manager early to inspect agent metadata
        self.db_manager = DatabaseManager(
            database_connect, template, schema_init, enable_short_term
        )

        explicit_model_provided = model is not None
        preferred_model_override: str | None = None
        candidate_agent_ids: list[str] = []
        if agent_id:
            candidate_agent_ids.append(agent_id)
        if user_id and user_id not in candidate_agent_ids:
            candidate_agent_ids.append(user_id)
        for candidate in candidate_agent_ids:
            profile = self.db_manager.get_agent(candidate)
            if profile:
                self.agent_profile = profile
                self.agent_id = profile.get("agent_id", candidate)
                preferred_model_override = profile.get("preferred_model") or preferred_model_override
                break

        # Load default model from central configuration
        self.default_model = settings.agents.default_model
        config_sovereign_ingest = getattr(memory_settings, "sovereign_ingest", False)
        self.sovereign_ingest = (
            config_sovereign_ingest
            if sovereign_ingest is None
            else bool(sovereign_ingest)
        )
        if preferred_model_override and not explicit_model_provided:
            self.default_model = preferred_model_override
            model = preferred_model_override
        self.conscious_analysis_interval_seconds = getattr(
            settings.memory,
            "conscious_analysis_interval_seconds",
            6 * 60 * 60,
        )

        if hasattr(agent_settings, "export_task_routes"):
            task_routes = agent_settings.export_task_routes()
        else:
            task_routes = {}

        provider_inputs = ProviderSetupInputs(
            default_model=self.default_model,
            provider_config=provider_config,
            api_type=api_type,
            base_url=base_url,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            openai_api_key=openai_api_key,
            azure_deployment=azure_deployment,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            organization=organization,
            project=project,
            model=model,
            sovereign_ingest=self.sovereign_ingest,
            conscious_ingest=self.conscious_ingest,
            auto_ingest=self.auto_ingest,
            enable_short_term=self.enable_short_term,
            use_lightweight_conscious_ingest=self.use_lightweight_conscious_ingest,
            anthropic_config=anthropic_config,
            anthropic_api_key=anthropic_api_key,
            anthropic_base_url=anthropic_base_url,
            anthropic_model=anthropic_model,
            gemini_config=gemini_config,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            task_routes=task_routes,
        )
        provider_result = setup_provider_components(provider_inputs)
        self._apply_provider_setup(provider_result)

        if preferred_model_override and not explicit_model_provided:
            self.default_model = preferred_model_override
            self.model = preferred_model_override
            if hasattr(self.provider_config, "model"):
                setattr(self.provider_config, "model", preferred_model_override)

        # Setup logging based on verbose mode
        self._setup_logging()

        # Initialize supporting managers dependent on provider state
        self.conscious_manager = ConsciousManager(self)
        self._retention_scheduler: MemoryRetentionScheduler | None = None
        self.retention_service: MemoryRetentionService | None = None

        # State tracking
        self._enabled = False
        self._session_id = str(uuid.uuid4())
        self._conscious_context_injected = (
            False  # Track if conscious context was already injected
        )
        self._in_context_retrieval = False  # Recursion guard for context retrieval
        self._sovereign_manual_reminder_logged = False
        self._memory_tasks: Set[asyncio.Task[None]] = set()
        self.plugin_registry: PluginRegistry = PluginRegistry()

        # Initialize conversation manager for stateless LLM integration
        self.conversation_manager = cast(
            ConversationManagerProtocol,
            ConversationManager(
                max_sessions=100,
                session_timeout_minutes=60,
                max_history_per_session=20,
            ),
        )

        if self.memory_agent and hasattr(self.memory_agent, "set_conversation_manager"):
            self.memory_agent.set_conversation_manager(self.conversation_manager)
        if self.search_engine and hasattr(self.search_engine, "set_conversation_manager"):
            self.search_engine.set_conversation_manager(self.conversation_manager)

        # User context for memory processing
        self._user_context: dict[str, list[str]] = {
            "current_projects": [],
            "relevant_skills": [],
            "user_preferences": [],
        }

        # Initialize database
        self._setup_database()

        # Initialize storage service for persistence operations
        self.storage_service = StorageService(
            db_manager=self.db_manager,
            namespace=self.namespace,
            search_engine=self.search_engine,
            conscious_ingest=self.conscious_ingest,
            user_id=self.user_id,
            agent_id=self.agent_id,
            image_storage_root=image_storage_path,
        )
        if self.agent_id and not self.agent_profile:
            self.agent_profile = self.storage_service.get_agent(self.agent_id)
        self.policy_engine = self.storage_service.policy_engine
        self.storage_service.configure_team_policy(
            namespace_prefix=self.team_namespace_prefix,
            enforce_membership=self.team_enforce_membership and self.team_memory_enabled,
            share_by_default=self.team_share_by_default if self.team_memory_enabled else False,
        )
        self._apply_default_team(
            initial_team_candidate,
            enforce_membership=self.team_enforce_membership and self.team_memory_enabled,
        )
        if (
            self.workspace_memory_enabled
            and requested_team_id is None
            and not self.get_active_workspace()
            and self.default_workspace_id
        ):
            self._apply_default_workspace(
                self.default_workspace_id,
                enforce_membership=self.team_enforce_membership
                and self.team_memory_enabled,
            )
        self._configure_sync(
            sync_settings=effective_sync_settings,
            backend_override=sync_backend,
        )

        # Initialize the new modular memory manager
        self.memory_manager = cast(
            MemoryManagerProtocol,
            MemoryManager(
                database_connect=database_connect,
                template=template,
                mem_prompt=mem_prompt,
                conscious_ingest=self.conscious_ingest,
                auto_ingest=auto_ingest,
                namespace=namespace,
                shared_memory=shared_memory,
                memory_filters=memory_filters,
                user_id=user_id,
                verbose=verbose,
                provider_config=self.provider_config,
                sovereign_ingest=self.sovereign_ingest,
            ),
        )
        # Set this Memoria instance for memory management
        self.memory_manager.set_memoria_instance(self)

        configured_plugins: Sequence["PluginSettings"] | None
        if plugin_settings is not None:
            configured_plugins = plugin_settings
        else:
            configured_plugins = getattr(settings, "plugins", [])

        self.plugin_registry = self._configure_plugins(configured_plugins)

        # Run conscious agent initialization if enabled
        if self.conscious_ingest and self.conscious_agent:
            self.conscious_manager.start()

        self._init_retention_services(settings)

        self._init_ingestion_scheduler()
        self._init_coordinate_audit_scheduler()

        self._start_schedulers()

        logger.info(
            f"Memoria v1.0 initialized with template: {template}, namespace: {namespace}"
        )

    def _load_config_info(
        self,
        sync_settings_override: "SyncSettings" | Mapping[str, Any] | None,
    ) -> tuple[Any, "SyncSettings" | Mapping[str, Any] | None]:
        """Populate configuration metadata and resolve sync settings."""

        config_manager = ConfigManager.get_instance()
        try:
            config_info = config_manager.get_config_info()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(f"Unable to fetch configuration metadata: {exc}")
            config_info = None

        if isinstance(config_info, Mapping):
            normalized_info: dict[str, Any] = dict(config_info)
            sources = normalized_info.get("sources")
            if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
                normalized_info["sources"] = list(sources)
            elif sources is None:
                normalized_info["sources"] = []

            self._config_info.update(normalized_info)

        settings = config_manager.get_settings()
        config_sync_settings = getattr(settings, "sync", None)
        if sync_settings_override is not None:
            effective_sync_settings = sync_settings_override
        else:
            effective_sync_settings = config_sync_settings

        if isinstance(effective_sync_settings, dict):
            try:
                from ..config.settings import SyncSettings as _SyncSettings

                effective_sync_settings = _SyncSettings(**effective_sync_settings)
            except Exception:  # pragma: no cover - fallback to raw mapping
                pass

        return settings, effective_sync_settings

    def get_config_info(self) -> dict[str, Any]:
        """Return configuration metadata associated with this instance."""

        info = dict(self._config_info)
        sources = info.get("sources")
        if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
            info["sources"] = list(sources)
        elif sources is None:
            info["sources"] = []

        info.setdefault("loaded", False)
        info.setdefault("version", None)
        info.setdefault("debug_mode", False)
        info.setdefault("is_production", False)
        return info

    def _setup_logging(self) -> None:
        """Setup logging configuration based on verbose mode"""
        setup_logging(self)

    def _init_sync(
        self,
        sync_settings: "SyncSettings" | Mapping[str, Any] | None,
        backend_override: "SyncBackend" | None,
    ) -> None:
        """Deprecated helper retained for compatibility."""

        self._configure_sync(sync_settings, backend_override)

    def _configure_sync(
        self,
        sync_settings: "SyncSettings" | Mapping[str, Any] | None,
        backend_override: "SyncBackend" | None,
    ) -> None:
        """Initialize sync bookkeeping and configure the backend."""

        self._sync_origin = f"memoria-{uuid.uuid4()}"
        self._sync_coordinator: SyncCoordinator | None = None
        self._sync_backend: "SyncBackend" | None = None
        self._sync_backend_owned = False
        self._sync_settings: "SyncSettings" | None = None
        self._sync_settings_snapshot: dict[str, Any] | None = None
        self._sync_enabled = False
        self.configure_sync(sync_settings, backend_override=backend_override)

    def _apply_provider_setup(self, result: ProviderSetupResult) -> None:
        """Apply provider setup results to this instance."""

        self.provider_config = result.provider_config
        self.openai_api_key = result.api_key
        self.model = result.model
        self.anthropic_config = result.anthropic_config
        self.anthropic_client = result.anthropic_client
        self.anthropic_async_client = result.anthropic_async_client
        self.gemini_config = result.gemini_config
        self.gemini_client = result.gemini_client
        self.memory_agent = result.memory_agent
        self.search_engine = result.search_engine
        self.conscious_agent = result.conscious_agent
        self.conscious_ingest = result.conscious_ingest
        self.auto_ingest = result.auto_ingest
        self.provider_registry = result.provider_registry
        self.task_model_routes = result.task_routes

    def _configure_plugins(
        self, configured_plugins: Sequence["PluginSettings"] | None
    ) -> PluginRegistry:
        """Load plugins into a fresh registry instance."""

        registry = getattr(self, "plugin_registry", PluginRegistry())
        return load_plugins(
            self,
            list(configured_plugins) if configured_plugins else [],
            registry=registry,
        )

    def _setup_database(self) -> None:
        """Setup database if missing or tables are absent"""
        # Avoid unnecessary work if schema initialization is disabled
        if not self.schema_init:
            return

        from pathlib import Path
        from sqlalchemy import inspect
        from ..database.models import Base

        setup_needed = False

        try:
            inspector = inspect(self.db_manager.engine)
            existing_tables = set(inspector.get_table_names())
            required_tables = set(Base.metadata.tables.keys())

            if not required_tables.issubset(existing_tables):
                setup_needed = True

            if self.db_manager.database_type == "sqlite":
                database_name = self.db_manager.engine.url.database
                if database_name:
                    db_path = Path(database_name)
                    if not db_path.exists():
                        setup_needed = True

        except Exception as e:  # pragma: no cover - inspection failures
            logger.debug(f"Database inspection failed: {e}; running setup")
            setup_needed = True

        if setup_needed:
            setup_database(self)


        self._run_bootstrap_migrations()

    def _init_retention_services(self, settings: Any) -> None:
        """Configure and start memory retention services and scheduler."""

        raw_policy_rules = getattr(settings.memory, "retention_policy_rules", []) or []
        policy_rules: tuple[RetentionPolicyRule, ...] = ()
        if raw_policy_rules:
            converted: list[RetentionPolicyRule] = []
            for rule in raw_policy_rules:
                action_value = getattr(rule.action, "value", rule.action)
                converted.append(
                    RetentionPolicyRule(
                        name=rule.name,
                        namespaces=tuple(rule.namespaces or ["*"]),
                        privacy_ceiling=rule.privacy_ceiling,
                        importance_floor=rule.importance_floor,
                        lifecycle_days=rule.lifecycle_days,
                        action=str(action_value),
                        escalate_to=rule.escalate_to,
                        metadata=dict(rule.metadata or {}),
                    )
                )
            policy_rules = tuple(converted)

        if hasattr(self, "storage_service") and self.storage_service is not None:
            self.storage_service.configure_retention_policies(policy_rules)

        retention_config = RetentionConfig(
            decay_half_life_hours=settings.memory.retention_decay_half_life_hours,
            reinforcement_bonus=settings.memory.retention_reinforcement_bonus,
            privacy_shift=settings.memory.retention_privacy_shift,
            importance_floor=settings.memory.retention_importance_floor,
            cluster_gravity_lambda=settings.cluster_gravity_lambda,
            policies=policy_rules,
        )
        self.retention_service = MemoryRetentionService(
            db_manager=self.db_manager,
            namespace=self.namespace,
            config=retention_config,
            cluster_enabled=(
                settings.enable_cluster_indexing and settings.use_db_clusters
            ),
            audit_callback=self.storage_service.record_retention_audit
            if hasattr(self, "storage_service")
            else None,
        )
        self._scheduled_job_completion_flags.setdefault(
            "retention", threading.Event()
        )
        self._retention_scheduler = MemoryRetentionScheduler(
            self.retention_service,
            interval_seconds=int(
                settings.memory.retention_update_interval_minutes * 60
            ),
        )

    def _init_ingestion_scheduler(self) -> None:
        """Configure and start the daily ingestion scheduler if enabled."""

        if not (
            self.enable_short_term
            and getattr(self.db_manager, "enable_short_term", False)
        ):
            return

        self._scheduled_job_completion_flags.setdefault(
            "ingestion", threading.Event()
        )
        self.ingestion_service = ShortTermIngestionService(
            storage_service=self.storage_service,
            db_manager=self.db_manager,
            namespace=self.namespace,
            on_run=self._record_ingestion_results,
        )
        self._ingestion_scheduler = DailyIngestionScheduler(
            self.ingestion_service,
            interval_seconds=24 * 60 * 60,
        )

    def _init_coordinate_audit_scheduler(self) -> None:
        """Initialise the coordinate audit scheduler when configured."""

        self._coordinate_audit_job = None
        self._coordinate_audit_scheduler = None

        if not self.coordinate_audit_enabled:
            return
        if self.coordinate_audit_cadence is CoordinateAuditCadence.DISABLED:
            return
        if not self.memory_agent:
            logger.debug(
                "Skipping coordinate audit scheduler because no memory agent is available"
            )
            return

        completion_event = self._scheduled_job_completion_flags.setdefault(
            "coordinate_audit", threading.Event()
        )
        job = CoordinateAuditJob(
            storage_service=self.storage_service,
            memory_agent=self.memory_agent,
            lookback_days=self.coordinate_audit_lookback_days,
            completion_event=completion_event,
        )

        cadence = self.coordinate_audit_cadence
        if cadence is CoordinateAuditCadence.WEEKLY:
            interval_seconds = 7 * 24 * 60 * 60
        else:
            interval_seconds = 7 * 24 * 60 * 60

        self._coordinate_audit_job = job
        self._coordinate_audit_scheduler = CoordinateAuditScheduler(
            job,
            interval_seconds=interval_seconds,
        )

    def _start_retention_scheduler(self) -> None:
        if self._retention_scheduler:
            self._retention_scheduler.start()

    def _stop_retention_scheduler(self) -> None:
        if self._retention_scheduler:
            self._retention_scheduler.stop()

    def _start_ingestion_scheduler(self) -> None:
        if self._ingestion_scheduler:
            self._ingestion_scheduler.start()

    def _start_coordinate_audit_scheduler(self) -> None:
        if hasattr(self, "_coordinate_audit_scheduler") and self._coordinate_audit_scheduler:
            self._coordinate_audit_scheduler.start()

    def _stop_ingestion_scheduler(self) -> None:
        if self._ingestion_scheduler:
            self._ingestion_scheduler.stop()

    def _stop_coordinate_audit_scheduler(self) -> None:
        if hasattr(self, "_coordinate_audit_scheduler") and self._coordinate_audit_scheduler:
            self._coordinate_audit_scheduler.stop()

    def _start_schedulers(self) -> None:
        self._start_retention_scheduler()
        self._start_ingestion_scheduler()
        self._start_coordinate_audit_scheduler()

    def _init_context_orchestrator(self) -> None:
        """Initialise the adaptive context planner if enabled."""

        config = getattr(self, "_context_orchestration_config", None)
        if not isinstance(config, ContextOrchestrationConfig):
            self.context_orchestrator = None
            return
        if not config.enabled:
            self.context_orchestrator = None
            return

        from ..database.analytics import get_analytics_summary

        def _fetch_usage_metrics() -> dict[str, Any]:
            try:
                with self.db_manager.SessionLocal() as session:
                    return get_analytics_summary(
                        session,
                        namespace=None if self.shared_memory else self.namespace,
                        include_short_term=self.enable_short_term,
                        days=config.analytics_window_days,
                        top_n=config.analytics_top_n,
                    )
            except Exception as exc:  # pragma: no cover - telemetry best effort
                logger.debug("Context analytics fetch failed: %s", exc)
                return {}

        self.context_orchestrator = ContextOrchestrator(
            config,
            analytics_fetcher=_fetch_usage_metrics,
        )

    def _stop_schedulers(self) -> None:
        self._stop_retention_scheduler()
        self._stop_ingestion_scheduler()
        self._stop_coordinate_audit_scheduler()

    def _run_bootstrap_migrations(self) -> None:
        """Ensure bootstrap migrations run after tables exist."""

        engine = getattr(self.db_manager, "engine", None)
        if engine is None:  # pragma: no cover - defensive guard
            return

        from sqlalchemy import inspect, text

        from ..database.models import Base, ClusterMember

        # Ensure the target table exists before attempting to alter it.
        Base.metadata.create_all(engine, tables=[ClusterMember.__table__])

        inspector = inspect(engine)
        try:
            existing_columns = {
                column["name"] for column in inspector.get_columns("cluster_members")
            }
        except Exception as exc:  # pragma: no cover - inspection backend specific
            logger.debug(
                "Skipping cluster member token migration; inspection failed: {}",
                exc,
            )
            return

        column_definitions = {
            "tokens": "INT",
            "chars": "INT",
        }

        for column, column_type in column_definitions.items():
            if column in existing_columns:
                continue

            sql_type = "INTEGER" if engine.dialect.name == "sqlite" else column_type

            try:
                with engine.begin() as conn:
                    conn.execute(
                        text(
                            f"ALTER TABLE cluster_members ADD COLUMN {column} "
                            f"{sql_type} DEFAULT 0"
                        )
                    )
                existing_columns.add(column)
                logger.info(
                    "Added {} column to cluster_members table during bootstrap", column
                )
            except Exception as exc:  # pragma: no cover - backend specific failures
                logger.warning(
                    "Failed to add {} column to cluster_members table: {}", column, exc
                )



    def _check_deferred_initialization(self) -> None:
        """Delegate deferred initialization check to conscious manager."""
        self.conscious_manager.check_deferred_initialization()

    def enable(self, interceptors: list[str] | None = None) -> dict[str, Any]:
        """
        Enable universal memory recording using LiteLLM's native callback system.

        This automatically sets up recording for LiteLLM completion calls and enables
        automatic interception of OpenAI calls when using the standard OpenAI client.

        Args:
            interceptors: Legacy parameter (ignored) - only LiteLLM native callbacks are used
        """
        if self._enabled:
            logger.warning("Memoria is already enabled.")
            return {"success": True, "message": "Already enabled"}

        self._session_id = str(uuid.uuid4())

        results: dict[str, Any] = {"enabled_interceptors": []}

        if not self.sovereign_ingest:
            # Register for automatic OpenAI interception
            try:
                from ..integrations.openai_integration import register_memoria_instance

                register_memoria_instance(self)
            except ImportError:
                logger.debug(
                    "OpenAI integration not available for automatic interception"
                )

            # Use LiteLLM native callback system only
            if interceptors is None:
                # Only LiteLLM native callbacks supported
                interceptors = ["litellm_native"]

            # Use the memory manager for enablement
            results = self.memory_manager.enable(interceptors)
        else:
            # Sovereign ingest keeps Memoria enabled but avoids automatic integrations
            self.memory_manager._enabled = True
            results.update(
                {
                    "success": True,
                    "message": "Sovereign ingest active; automatic integrations skipped.",
                }
            )

        if not results.get("success", False):
            self._enabled = False
            logger.error(
                "Failed to enable Memoria: {}",
                results.get("message", "Unknown error during memory manager enablement"),
            )
            return results

        self._enabled = True

        # Extract enabled interceptors from results
        enabled_interceptors = results.get("enabled_interceptors", [])

        # Start background conscious agent if available
        if self.conscious_ingest and self.conscious_agent:
            self.conscious_manager.start()

        if results.get("success", True):
            self._start_schedulers()

        # Report status
        status_info = [
            f"Memoria enabled for session: {results.get('session_id', self._session_id)}",
            f"Active interceptors: {', '.join(enabled_interceptors) if enabled_interceptors else 'None'}",
        ]

        if results.get("message"):
            status_info.append(results["message"])

        status_info.extend(
            [
                f"Background analysis: {'Active' if self.conscious_manager.is_running() else 'Disabled'}",
                "Usage: Simply use any LLM client normally - conversations will be auto-recorded!",
                "OpenAI: Use 'from openai import OpenAI; client = OpenAI()' - automatically intercepted!",
            ]
        )

        logger.info("\n".join(status_info))

        return results

    def disable(self) -> None:
        """
        Disable memory recording by unregistering LiteLLM callbacks and OpenAI interception.
        """
        if not self._enabled:
            if getattr(self, "plugin_registry", None):
                self.plugin_registry.shutdown_all()
            return

        if not self.sovereign_ingest:
            # Unregister from automatic OpenAI interception
            try:
                from ..integrations.openai_integration import unregister_memoria_instance

                unregister_memoria_instance(self)
            except ImportError:
                logger.debug(
                    "OpenAI integration not available for automatic interception"
                )

            # Use memory manager for clean disable
            results = self.memory_manager.disable()
        else:
            self.memory_manager._enabled = False
            results = {
                "success": True,
                "message": "Sovereign ingest disabled; no integrations to tear down.",
            }

        # Stop background analysis task
        self.conscious_manager.stop()

        self._stop_schedulers()

        self._enabled = False

        if getattr(self, "plugin_registry", None):
            self.plugin_registry.shutdown_all()

        # Report status based on memory manager results
        if results.get("success"):
            status_message = f"Memoria disabled. {results.get('message', 'All interceptors disabled successfully')}"
        else:
            status_message = (
                f"Memoria disable failed: {results.get('message', 'Unknown error')}"
            )

        logger.info(status_message)

    # Memory system status and control methods

    def get_interceptor_status(self) -> dict[str, dict[str, Any]]:
        """Get status of memory recording system"""
        return self.memory_manager.get_status()

    def get_interceptor_health(self) -> dict[str, Any]:
        """Get health check of interceptor system"""
        return self.memory_manager.get_health()

    def enable_interceptor(self, interceptor_name: str | None = None) -> bool:
        """Enable memory recording (legacy method)"""
        return _enable_interceptor_impl(self, interceptor_name)

    def disable_interceptor(self, interceptor_name: str | None = None) -> bool:
        """Disable memory recording (legacy method)"""
        return _disable_interceptor_impl(self, interceptor_name)

    def get_plugin(self, name: str) -> "BasePlugin" | None:
        """Return a loaded plugin by name, if available."""

        if not getattr(self, "plugin_registry", None):
            return None
        return self.plugin_registry.get(name)

    @property
    def plugins(self) -> list["BasePlugin"]:
        """Return all loaded plugin instances."""

        if not getattr(self, "plugin_registry", None):
            return []
        return list(self.plugin_registry.iter_plugins())

    @property
    def plugin_failures(self) -> dict[str, str]:
        """Return information about plugins that failed to load."""

        if not getattr(self, "plugin_registry", None):
            return {}
        return dict(self.plugin_registry.failures)

    def _inject_openai_context(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Inject context for OpenAI calls based on ingest mode using ConversationManager"""
        return _inject_openai_context(self, kwargs)

    def _inject_anthropic_context(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Inject context for Anthropic calls based on ingest mode"""
        return _inject_anthropic_context(self, kwargs)

    def _inject_gemini_context(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Inject context for Google Gemini calls based on ingest mode"""
        return _inject_gemini_context(self, kwargs)

    def _inject_litellm_context(
        self, params: dict[str, Any], mode: str = "auto"
    ) -> dict[str, Any]:
        """Inject context for LiteLLM calls based on mode"""
        return _inject_litellm_context(self, params, mode)

    def _process_litellm_response(
        self,
        kwargs: dict[str, Any],
        response: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Process and record LiteLLM response"""
        try:
            # Extract user input from messages
            messages = kwargs.get("messages", [])
            user_input = ""

            for message in reversed(messages):
                if message.get("role") == "user":
                    user_input = message.get("content", "")
                    break

            # Extract AI output from response
            ai_output = ""
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    ai_output = choice.message.content or ""
                elif hasattr(choice, "text"):
                    ai_output = choice.text or ""

            # Extract model
            model = kwargs.get("model", "litellm-unknown")

            # Calculate timing (convert to seconds for JSON serialization)
            duration_seconds = (end_time - start_time) if start_time and end_time else 0
            if hasattr(duration_seconds, "total_seconds"):
                duration_seconds = duration_seconds.total_seconds()

            # Prepare metadata
            metadata = {
                "integration": "litellm",
                "auto_recorded": True,
                "duration": float(duration_seconds),
                "timestamp": time.time(),
            }

            # Add token usage if available
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                metadata.update(
                    {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage, "completion_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0),
                    }
                )

            # Record the conversation
            if user_input and ai_output:
                self.record_conversation(
                    user_input=user_input,
                    ai_output=ai_output,
                    model=model,
                    metadata=metadata,
                )

        except Exception as e:
            logger.error(f"Failed to process LiteLLM response: {e}")

    # LiteLLM callback is now handled by the LiteLLMCallbackManager
    # in memoria.integrations.litellm_integration

    def _process_memory_sync(
        self, chat_id: str, user_input: str, ai_output: str, model: str = "unknown"
    ) -> None:
        """Synchronous memory processing fallback"""
        if not self.memory_agent:
            logger.warning("Memory agent not available, skipping memory ingestion")
            return

        try:
            # Run async processing in new event loop
            import threading

            def run_memory_processing() -> None:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(
                        self._process_memory_async(
                            chat_id, user_input, ai_output, model
                        )
                    )
                except Exception as e:
                    logger.error(f"Synchronous memory processing failed: {e}")
                finally:
                    new_loop.close()

            # Run in background thread to avoid blocking
            thread = threading.Thread(target=run_memory_processing, daemon=True)
            thread.start()
            logger.debug(
                f"Memory processing started in background thread for {chat_id}"
            )

        except Exception as e:
            logger.error(f"Failed to start synchronous memory processing: {e}")

    def _parse_llm_response(self, response: Any) -> tuple[str, str]:
        """Extract text and model from various LLM response formats."""
        if response is None:
            return "", "unknown"

        # String response
        if isinstance(response, str):
            return response, "unknown"

        # Anthropic response
        if hasattr(response, "content"):
            text = ""
            if isinstance(response.content, list):
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
            else:
                text = str(response.content)
            model_name = getattr(response, "model", "unknown")
            return text, str(model_name)

        # OpenAI response
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            text = (
                getattr(choice.message, "content", "")
                if hasattr(choice, "message")
                else getattr(choice, "text", "")
            )
            model_name = getattr(response, "model", "unknown")
            return (text or "", str(model_name))

        # dict response
        if isinstance(response, dict):
            content_value = response.get("content")
            if content_value is None:
                content_value = response.get("text", response)
            model_value = response.get("model", "unknown")
            return str(content_value), str(model_value)

        # Fallback
        return str(response), "unknown"

    def _resolve_last_editor(self, model_name: str | None) -> str:
        """Determine the identifier recorded for provenance tracking."""

        if model_name and str(model_name).strip():
            return str(model_name).strip()
        if self.agent_profile:
            preferred = self.agent_profile.get("preferred_model")
            if preferred:
                return str(preferred)
        if self.agent_id:
            return str(self.agent_id)
        if self.user_id:
            return str(self.user_id)
        return "human"

    def record_conversation(
        self,
        user_input: str,
        ai_output: Any | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Record a conversation.

        Args:
            user_input: User's message
            ai_output: AI response (any format)
            model: Optional model name override
            metadata: Optional metadata

        Returns:
            chat_id: Unique conversation ID
        """
        if not self._enabled:
            raise MemoriaError("Memoria is not enabled. Call enable() first.")

        # Parse response
        response_text, detected_model = self._parse_llm_response(ai_output)
        response_model = model or detected_model

        # Generate ID and timestamp
        chat_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Store conversation
        self.db_manager.store_chat_history(
            chat_id=chat_id,
            user_input=user_input,
            ai_output=response_text,
            timestamp=timestamp,
            session_id=self._session_id,
            model=response_model,
            namespace=self.namespace,
            metadata=metadata or {},
            edited_by_model=self._resolve_last_editor(response_model),
        )

        # Always process into long-term memory when memory agent is available
        if self.sovereign_ingest:
            if not self._sovereign_manual_reminder_logged:
                logger.info(
                    "Sovereign ingest mode active: memories must be processed manually."
                )
                self._sovereign_manual_reminder_logged = True

            self.process_recorded_conversation_heuristic(
                chat_id,
                user_input,
                response_text,
                response_model,
                metadata=metadata,
            )
        elif self.memory_agent:
            self._schedule_memory_processing(
                chat_id, user_input, response_text, response_model
            )
        else:
            self.process_recorded_conversation_heuristic(
                chat_id,
                user_input,
                response_text,
                response_model,
                metadata=metadata,
            )

        logger.debug(f"Recorded conversation: {chat_id}")
        return chat_id

    def _schedule_memory_processing(
        self, chat_id: str, user_input: str, ai_output: str, model: str
    ) -> None:
        """Schedule memory processing (async if possible, sync fallback)."""
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._process_memory_async(chat_id, user_input, ai_output, model)
            )

            # Prevent garbage collection
            self._memory_tasks.add(task)
            task.add_done_callback(self._memory_tasks.discard)
        except RuntimeError:
            # No event loop, use sync fallback
            logger.debug("No event loop, using synchronous memory processing")
            self._process_memory_sync(chat_id, user_input, ai_output, model)

    async def _process_memory_async(
        self, chat_id: str, user_input: str, ai_output: str, model: str = "unknown"
    ) -> None:
        """Process conversation with enhanced async memory categorization"""

        if self.sovereign_ingest or not self.memory_agent:
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                await asyncio.to_thread(
                    self.process_recorded_conversation_heuristic,
                    chat_id,
                    user_input,
                    ai_output,
                    model,
                    None,
                )
            else:
                self.process_recorded_conversation_heuristic(
                    chat_id,
                    user_input,
                    ai_output,
                    model,
                    None,
                )
            return

        try:
            # Create conversation context
            context = ConversationContext(
                user_id=self.user_id,
                session_id=self._session_id,
                conversation_id=chat_id,
                model_used=model,
                user_preferences=self._user_context.get("user_preferences", []),
                current_projects=self._user_context.get("current_projects", []),
                relevant_skills=self._user_context.get("relevant_skills", []),
            )

            # Get recent memories for deduplication
            existing_memories = await self._get_recent_memories_for_dedup()

            # Process conversation using async Pydantic-based memory agent
            processed_memory = await self.memory_agent.process_conversation_async(
                chat_id=chat_id,
                user_input=user_input,
                ai_output=ai_output,
                context=context,
                existing_memories=(
                    [mem.summary for mem in existing_memories[:10]]
                    if existing_memories
                    else []
                ),
                relationship_context=
                    {
                        "db_manager": self.db_manager,
                        "namespace": self.namespace,
                        "privacy_floor": -10.0,
                    }
                    if self.db_manager
                    else None,
            )

            # Check for duplicates
            duplicate_id = await self.memory_agent.detect_duplicates(
                processed_memory, existing_memories
            )

            if duplicate_id:
                processed_memory.duplicate_of = duplicate_id
                logger.info(f"Memory marked as duplicate of {duplicate_id}")

            # Apply filters
            if self.memory_agent.should_filter_memory(
                processed_memory, self.memory_filters
            ):
                logger.debug(f"Memory filtered out for chat {chat_id}")
                return

            # Determine if this memory should also be persisted to short-term storage
            short_term_id = None
            should_store_short_term = False
            if self.db_manager.enable_short_term:
                should_store_short_term = any(
                    [
                        processed_memory.promotion_eligible,
                        processed_memory.classification
                        in (
                            MemoryClassification.ESSENTIAL,
                            MemoryClassification.CONSCIOUS_INFO,
                        ),
                        processed_memory.is_user_context,
                        processed_memory.is_preference,
                        processed_memory.is_skill_knowledge,
                        processed_memory.is_current_project,
                    ]
                )

            if should_store_short_term:
                try:
                    short_term_id = self.db_manager.store_short_term_memory(
                        processed_memory,
                        chat_id,
                        self.namespace,
                        workspace_id=self.storage_service.workspace_id,
                        edited_by_model=self._resolve_last_editor(model),
                    )
                    if short_term_id:
                        logger.debug(
                            f"Stored short-term memory {short_term_id} for chat {chat_id}"
                        )
                except Exception as short_err:
                    logger.error(
                        f"Failed to store short-term memory for chat {chat_id}: {short_err}"
                    )

            redaction_state: dict[str, Any] = {
                "summary": processed_memory.summary,
                "content": processed_memory.content,
                "symbolic_anchors": list(processed_memory.symbolic_anchors or []),
                "y_coord": processed_memory.y_coord,
                "metadata": {
                    "classification": processed_memory.classification.value,
                    "importance": getattr(getattr(processed_memory, "importance", None), "value", None),
                },
            }
            removed_fields: set[str] = set()

            def _apply_redaction(decision: "PolicyDecision") -> None:
                result: RedactionResult = apply_redactions(redaction_state, decision)
                removed_fields.update(result.removed)

            policy_payload = {
                "namespace": self.namespace,
                "chat_id": chat_id,
                "source": "memory_agent",
                "classification": processed_memory.classification.value,
                "importance": getattr(getattr(processed_memory, "importance", None), "value", None),
                "privacy": processed_memory.y_coord,
                "anchors": list(processed_memory.symbolic_anchors or []),
            }

            self.storage_service.enforce_policy(
                EnforcementStage.INGESTION,
                policy_payload,
                allow_redaction=True,
                on_redact=_apply_redaction,
            )

            if "summary" in removed_fields and "summary" not in redaction_state:
                redaction_state["summary"] = ""
            if "content" in removed_fields and "content" not in redaction_state:
                redaction_state["content"] = ""
            if "symbolic_anchors" in removed_fields and "symbolic_anchors" not in redaction_state:
                redaction_state["symbolic_anchors"] = []
            if "y_coord" in removed_fields and "y_coord" not in redaction_state:
                redaction_state["y_coord"] = None

            processed_memory.summary = redaction_state.get("summary", processed_memory.summary)
            processed_memory.content = redaction_state.get("content", processed_memory.content)
            processed_memory.symbolic_anchors = redaction_state.get(
                "symbolic_anchors", processed_memory.symbolic_anchors
            )
            processed_memory.y_coord = redaction_state.get("y_coord", processed_memory.y_coord)

            # Store processed memory with new schema
            memory_id = self.db_manager.store_long_term_memory_enhanced(
                processed_memory,
                chat_id,
                self.namespace,
                workspace_id=self.storage_service.workspace_id,
                edited_by_model=self._resolve_last_editor(model),
            )

            if memory_id:
                logger.debug(f"Stored processed memory {memory_id} for chat {chat_id}")

                if processed_memory.related_memories:
                    try:
                        self.memory_agent.persist_relationship_links(
                            memory_id,
                            processed_memory.related_memories,
                            self.db_manager,
                        )
                    except Exception as exc:
                        logger.debug(
                            "Failed to persist relationship links for %s: %s",
                            memory_id,
                            exc,
                        )

                # Check for conscious context updates if promotion eligible and conscious_ingest enabled
                if (
                    processed_memory.promotion_eligible
                    and self.conscious_agent
                    and self.conscious_ingest
                ):
                    await self.conscious_agent.check_for_context_updates(
                        self.db_manager, self.namespace
                    )
            else:
                logger.warning(f"Failed to store memory for chat {chat_id}")

        except Exception as e:
            logger.error(f"Memory ingestion failed for {chat_id}: {e}")

    def process_recorded_conversation_heuristic(
        self,
        chat_id: str,
        user_input: str,
        ai_output: str,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
        promotion_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Process a conversation turn using heuristic staging and promotion."""

        try:
            metadata_payload = dict(metadata or {})
            if model:
                metadata_payload.setdefault("model", model)

            result: HeuristicConversationResult = process_conversation_turn(
                self.storage_service,
                chat_id=chat_id,
                user_input=user_input,
                ai_output=ai_output,
                metadata=metadata_payload,
                promotion_weights=promotion_weights,
            )
        except Exception as exc:
            logger.error(f"Heuristic staging failed for chat {chat_id}: {exc}")
            return {
                "status": "error",
                "error": str(exc),
                "memory_id": None,
                "short_term_id": None,
                "long_term_id": None,
            }

        decision = result.decision
        staged = result.staged
        long_term_id: str | None = None

        if decision.should_promote:
            if decision.score >= 0.75:
                importance_level = MemoryImportanceLevel.HIGH
            elif decision.score >= decision.threshold:
                importance_level = MemoryImportanceLevel.MEDIUM
            else:
                importance_level = MemoryImportanceLevel.LOW

            processed_memory = ProcessedLongTermMemory(
                content=result.summary,
                summary=result.summary,
                classification=MemoryClassification.CONTEXTUAL,
                importance=importance_level,
                topic=None,
                entities=[],
                keywords=[],
                is_user_context=False,
                is_preference=False,
                is_skill_knowledge=False,
                is_current_project=False,
                duplicate_of=None,
                supersedes=[],
                related_memories=[],
                conversation_id=chat_id,
                confidence_score=0.75,
                emotional_intensity=result.emotional_intensity,
                x_coord=staged.x_coord,
                y_coord=staged.y_coord,
                z_coord=staged.z_coord,
                symbolic_anchors=result.symbolic_anchors,
                classification_reason=(
                    f"Heuristic ingest approval (score={decision.score:.2f})"
                ),
                promotion_eligible=True,
            )

            heuristic_state: dict[str, Any] = {
                "summary": processed_memory.summary,
                "content": processed_memory.content,
                "symbolic_anchors": list(processed_memory.symbolic_anchors or []),
                "y_coord": processed_memory.y_coord,
                "metadata": {
                    "classification": processed_memory.classification.value,
                    "importance": importance_level.value,
                },
            }
            heuristic_removed: set[str] = set()

            def _apply_heuristic_redaction(decision: PolicyDecision) -> None:
                result: RedactionResult = apply_redactions(heuristic_state, decision)
                heuristic_removed.update(result.removed)

            heuristic_payload = {
                "namespace": self.namespace,
                "chat_id": chat_id,
                "source": "heuristic_ingest",
                "classification": processed_memory.classification.value,
                "importance": importance_level.value,
                "privacy": processed_memory.y_coord,
                "anchors": list(processed_memory.symbolic_anchors or []),
            }

            self.storage_service.enforce_policy(
                EnforcementStage.INGESTION,
                heuristic_payload,
                allow_redaction=True,
                on_redact=_apply_heuristic_redaction,
            )

            if "summary" in heuristic_removed and "summary" not in heuristic_state:
                heuristic_state["summary"] = ""
            if "content" in heuristic_removed and "content" not in heuristic_state:
                heuristic_state["content"] = ""
            if "symbolic_anchors" in heuristic_removed and "symbolic_anchors" not in heuristic_state:
                heuristic_state["symbolic_anchors"] = []
            if "y_coord" in heuristic_removed and "y_coord" not in heuristic_state:
                heuristic_state["y_coord"] = None

            processed_memory.summary = heuristic_state.get(
                "summary", processed_memory.summary
            )
            processed_memory.content = heuristic_state.get(
                "content", processed_memory.content
            )
            processed_memory.symbolic_anchors = heuristic_state.get(
                "symbolic_anchors", processed_memory.symbolic_anchors
            )
            processed_memory.y_coord = heuristic_state.get(
                "y_coord", processed_memory.y_coord
            )

            try:
                long_term_id = self.db_manager.store_long_term_memory_enhanced(
                    processed_memory,
                    chat_id,
                    self.namespace,
                    workspace_id=self.storage_service.workspace_id,
                    edited_by_model=self._resolve_last_editor(model),
                )
            except Exception as exc:
                logger.error(
                    f"Failed to persist heuristic memory for chat {chat_id}: {exc}"
                )
            else:
                if long_term_id:
                    try:
                        self.storage_service.transfer_spatial_metadata(
                            staged.memory_id,
                            long_term_id,
                            workspace_id=self.storage_service.workspace_id,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.error(
                            f"Failed to transfer spatial metadata for {long_term_id}: {exc}"
                        )

                    if staged.metadata.get("short_term_stored", False):
                        try:
                            self.storage_service.remove_short_term_memory(
                                staged.memory_id
                            )
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.error(
                                f"Failed to remove short-term memory {staged.memory_id}: {exc}"
                            )

                    try:
                        anchor_candidates: list[str] = list(result.symbolic_anchors or [])
                        if result.anchor and result.anchor not in anchor_candidates:
                            anchor_candidates.append(result.anchor)
                        related_ids = (
                            self.storage_service.find_related_memory_ids_by_anchors(
                                anchor_candidates, exclude_memory_ids=[long_term_id]
                            )
                            if anchor_candidates
                            else []
                        )
                        if related_ids:
                            self.db_manager.refresh_memory_last_access(
                                self.namespace,
                                related_ids,
                                workspace_id=self.get_active_workspace(),
                            )
                    except Exception as exc:  # pragma: no cover - defensive logging
                        logger.debug(
                            "Failed to refresh related memory access timestamps: %s", exc
                        )

        response: dict[str, Any] = {
            "status": "promoted" if long_term_id else "staged",
            "memory_id": long_term_id or staged.memory_id,
            "short_term_id": staged.memory_id,
            "long_term_id": long_term_id,
            "promoted": bool(long_term_id),
            "promotion_score": decision.score,
            "threshold": decision.threshold,
            "weights": decision.weights,
        }

        return response

    async def _get_recent_memories_for_dedup(
        self,
    ) -> list[ProcessedLongTermMemory]:
        """Get recent memories for deduplication check"""
        return await self.storage_service.get_recent_memories_for_dedup()

    def _resolve_team_arguments(
        self,
        *,
        namespace: str | None,
        team_id: str | None,
        share_with_team: bool | None,
        require_team: bool,
    ) -> tuple[str | None, str | None, bool | None]:
        cleaned_namespace = (namespace or "").strip() or None
        resolved_team = self._clean_team_identifier(team_id)
        if resolved_team is None:
            resolved_team = self._active_team_id
        if (
            resolved_team is None
            and self.team_memory_enabled
            and self.default_team_id
        ):
            resolved_team = self.default_team_id
        if (
            resolved_team is None
            and self.team_memory_enabled
            and self.workspace_memory_enabled
            and self.default_workspace_id
        ):
            resolved_team = self.default_workspace_id
        if not self.team_memory_enabled:
            if resolved_team:
                raise MemoriaError(
                    "Team memory support is disabled for this Memoria instance"
                )
            return cleaned_namespace, None, share_with_team
        if require_team and resolved_team is None:
            raise MemoriaError("A team context is required for this operation")
        resolved_share = share_with_team
        if resolved_share is not None:
            resolved_share = bool(resolved_share)
        return cleaned_namespace, resolved_team, resolved_share

    def _ensure_chat_history_entry(
        self,
        *,
        chat_id: str | None,
        namespace: str,
        team_id: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        """Ensure a chat history record exists for the supplied identifiers."""

        if not chat_id:
            return

        with self.db_manager.SessionLocal() as session:
            query = session.query(ChatHistory.chat_id).filter(
                ChatHistory.chat_id == chat_id,
                ChatHistory.namespace == namespace,
            )
            if team_id is not None:
                query = query.filter(ChatHistory.team_id == team_id)
            if workspace_id is not None:
                query = query.filter(ChatHistory.workspace_id == workspace_id)

            if query.first() is not None:
                return

        self.db_manager.store_chat_history(
            chat_id=chat_id,
            user_input=StorageService.AUTOGEN_USER_PLACEHOLDER,
            ai_output=StorageService.AUTOGEN_AI_PLACEHOLDER,
            timestamp=datetime.utcnow(),
            session_id=chat_id,
            model="auto-generated",
            namespace=namespace,
            tokens_used=0,
            metadata={"auto_generated": True},
            team_id=team_id,
            workspace_id=workspace_id,
            edited_by_model=self._resolve_last_editor("auto-generated"),
        )

    def _prepare_team_context(
        self,
        *,
        namespace: str | None = None,
        team_id: str | None = None,
        share_with_team: bool | None = None,
        require_team: bool = False,
    ) -> tuple[str, str | None, bool | None]:
        namespace_hint, resolved_team, resolved_share = self._resolve_team_arguments(
            namespace=namespace,
            team_id=team_id,
            share_with_team=share_with_team,
            require_team=require_team,
        )
        if not self.team_memory_enabled:
            target_namespace = namespace_hint or self.namespace
            return target_namespace, None, resolved_share
        target_namespace = self.storage_service.resolve_target_namespace(
            namespace=namespace_hint,
            team_id=resolved_team,
            user_id=self.user_id,
            share_with_team=resolved_share,
        )
        return target_namespace, resolved_team, resolved_share

    def _apply_default_team(
        self, team_id: str | None, *, enforce_membership: bool
    ) -> None:
        if not self.team_memory_enabled:
            self._active_team_id = None
            return
        cleaned = self._clean_team_identifier(team_id)
        if not cleaned:
            return
        try:
            self.set_active_team(cleaned, enforce_membership=enforce_membership)
        except MemoriaError as exc:
            logger.warning(
                "Unable to activate team '%s' during initialisation: %s",
                cleaned,
                exc,
            )

    def _apply_default_workspace(
        self, workspace_id: str | None, *, enforce_membership: bool
    ) -> None:
        if not (self.team_memory_enabled and self.workspace_memory_enabled):
            return
        cleaned = self._clean_team_identifier(workspace_id)
        if not cleaned:
            return
        try:
            self.set_active_workspace(cleaned, enforce_membership=enforce_membership)
        except MemoriaError as exc:
            logger.warning(
                "Unable to activate workspace '%s' during initialisation: %s",
                cleaned,
                exc,
            )

    def retrieve_context(
        self,
        query: str,
        limit: int = 5,
        *,
        namespace: str | None = None,
        team_id: str | None = None,
        share_with_team: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant context for a query with priority on essential facts."""

        target_namespace, resolved_team, resolved_share = self._prepare_team_context(
            namespace=namespace,
            team_id=team_id,
            share_with_team=share_with_team,
            require_team=self.team_mode == TeamMode.REQUIRED,
        )
        return self.storage_service.retrieve_context(
            query,
            limit,
            namespace=target_namespace,
            team_id=resolved_team,
            user_id=self.user_id,
            share_with_team=resolved_share,
        )

    def retrieve_memories_near(
        self,
        x: float,
        y: float,
        z: float,
        max_distance: float = 5.0,
        anchor: str | list[str] | None = None,
        limit: int = 10,
        dimensions: str = "3d",
        *,
        namespace: str | None = None,
        team_id: str | None = None,
        share_with_team: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories near a given spatial coordinate within a threshold distance."""
        target_namespace, _, _ = self._prepare_team_context(
            namespace=namespace,
            team_id=team_id,
            share_with_team=share_with_team,
            require_team=self.team_mode == TeamMode.REQUIRED,
        )
        return self.storage_service.retrieve_memories_near(
            x,
            y,
            z,
            max_distance=max_distance,
            anchor=anchor,
            limit=limit,
            dimensions=dimensions,
            namespace=target_namespace,
        )

    def retrieve_memories_near_2d(
        self,
        y: float,
        z: float,
        max_distance: float = 5.0,
        anchor: str | list[str] | None = None,
        limit: int = 10,
        *,
        namespace: str | None = None,
        team_id: str | None = None,
        share_with_team: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories near a Y/Z coordinate while ignoring the temporal axis."""
        target_namespace, _, _ = self._prepare_team_context(
            namespace=namespace,
            team_id=team_id,
            share_with_team=share_with_team,
            require_team=self.team_mode == TeamMode.REQUIRED,
        )
        return self.storage_service.retrieve_memories_near_2d(
            y,
            z,
            max_distance=max_distance,
            anchor=anchor,
            limit=limit,
            namespace=target_namespace,
        )

    def retrieve_memories_by_anchor(
        self,
        anchors: list[str],
        *,
        namespace: str | None = None,
        team_id: str | None = None,
        share_with_team: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories that match symbolic anchors."""
        # This search is intentionally cross-namespace, so we pass `None`
        # to the storage service to avoid filtering by the current namespace.
        # The `namespace`, `team_id`, and `share_with_team` parameters are
        # kept for API consistency but are not used in this implementation.
        return self.storage_service.retrieve_memories_by_anchor(anchors, namespace=None)

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Fetch persisted thread metadata and ordered messages."""
        return self.storage_service.get_thread(thread_id)

    def get_threads_for_memory(self, memory_id: str) -> list[dict[str, Any]]:
        """Return thread memberships for a stored memory."""
        return self.storage_service.get_threads_for_memory(memory_id)

    def retrieve_memories_by_time_range(
        self,
        *,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
        start_x: float | None = None,
        end_x: float | None = None,
        namespace: str | None = None,
        team_id: str | None = None,
        share_with_team: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories within a time or x-coordinate range."""
        target_namespace, _, _ = self._prepare_team_context(
            namespace=namespace,
            team_id=team_id,
            share_with_team=share_with_team,
            require_team=self.team_mode == TeamMode.REQUIRED,
        )
        return self.storage_service.retrieve_memories_by_time_range(
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            start_x=start_x,
            end_x=end_x,
            namespace=target_namespace,
        )

    def _ensure_team_memory_enabled(self) -> None:
        if not self.team_memory_enabled:
            raise MemoriaError(
                "Team memory support is disabled for this Memoria instance"
            )

    def _team_space_to_dict(
        self, space: Any, *, include_members: bool = True
    ) -> dict[str, Any]:
        if space is None:
            raise MemoriaError("Unknown team space")
        to_dict_fn = getattr(space, "to_dict", None)
        if callable(to_dict_fn):
            return cast(dict[str, Any], to_dict_fn(include_members=include_members))

        payload: dict[str, Any] = {
            "team_id": getattr(space, "team_id", None),
            "namespace": getattr(space, "namespace", None),
            "display_name": getattr(space, "display_name", None),
            "share_by_default": getattr(space, "share_by_default", False),
            "metadata": copy.deepcopy(getattr(space, "metadata", {}) or {}),
        }
        if include_members:
            members = list(getattr(space, "members", []) or [])
            admins = list(getattr(space, "admins", []) or [])
            payload["members"] = sorted(members)
            payload["admins"] = sorted(admins)
        return payload

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
        """Store a personal memory entry, falling back to manual storage when disabled."""

        if entry is None:
            raise MemoriaError("Personal memory entry payload is required")

        try:
            personal_entry = (
                entry
                if isinstance(entry, PersonalMemoryEntry)
                else model_validate(PersonalMemoryEntry, entry)
            )
        except (ValidationError, TypeError) as exc:
            raise MemoriaError("Invalid personal memory entry payload") from exc

        if not self.personal_mode_enabled:
            fallback = self.store_memory(
                personal_entry.anchor,
                personal_entry.text,
                personal_entry.tokens,
                timestamp=personal_entry.timestamp,
                x_coord=personal_entry.x_coord,
                y=personal_entry.y_coord,
                z=personal_entry.z_coord,
                symbolic_anchors=personal_entry.symbolic_anchors,
                chat_id=personal_entry.chat_id,
                metadata=personal_entry.metadata,
                namespace=namespace,
                team_id=team_id,
                return_status=return_status,
            )
            return fallback

        storage_result = self.storage_service.store_personal_memory(
            personal_entry,
            namespace=namespace,
            team_id=team_id,
            workspace_id=workspace_id,
            user_id=user_id or self.user_id,
        )

        memory_id = storage_result.get("memory_id")
        if not memory_id:
            raise MemoriaError("Failed to store personal memory entry")

        resolved_namespace = (
            storage_result.get("namespace")
            or namespace
            or self.storage_service.namespace
            or self.namespace
            or "default"
        )
        resolved_team = storage_result.get("team_id")
        resolved_workspace = storage_result.get("workspace_id")

        self._ensure_chat_history_entry(
            chat_id=personal_entry.chat_id,
            namespace=resolved_namespace,
            team_id=resolved_team,
            workspace_id=resolved_workspace,
        )

        if return_status:
            payload = dict(storage_result)
            payload.setdefault("status", "stored")
            payload.setdefault("chat_id", personal_entry.chat_id)
            return payload

        return memory_id

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
        emotional_intensity: float | None = None,
        chat_id: str | None = None,
        promotion_weights: dict[str, float] | None = None,
        return_status: bool = False,
        user_priority: float | None = None,
        metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
        team_id: str | None = None,
        share_with_team: bool | None = None,
        documents: Sequence[PersonalMemoryDocument | Mapping[str, Any]] | None = None,
        images: Sequence[MemoryImageAsset | Mapping[str, Any]] | None = None,
        ingest_mode: IngestMode | str | None = None,
        workspace_id: str | None = None,
    ) -> str | dict[str, Any]:
        """Stage a manual memory and promote it when heuristics approve."""

        staging_metadata = dict(metadata or {})
        if user_priority is not None:
            staging_metadata["user_priority"] = user_priority
        staging_metadata.setdefault("manual_submission", True)

        resolved_ingest_mode: IngestMode | None
        if ingest_mode is None:
            resolved_ingest_mode = self.ingest_mode
        elif isinstance(ingest_mode, IngestMode):
            resolved_ingest_mode = ingest_mode
        else:
            try:
                candidate_mode = str(ingest_mode).strip().lower()
                resolved_ingest_mode = IngestMode(candidate_mode)
            except ValueError:
                resolved_ingest_mode = self.ingest_mode

        namespace_arg, resolved_team, resolved_share = self._resolve_team_arguments(
            namespace=namespace,
            team_id=team_id,
            share_with_team=share_with_team,
            require_team=self.team_mode == TeamMode.REQUIRED,
        )

        active_workspace = self.get_active_workspace()
        target_workspace = workspace_id or active_workspace

        if resolved_ingest_mode == IngestMode.PERSONAL:
            personal_payload: dict[str, Any] = {
                "anchor": anchor,
                "text": text,
                "tokens": tokens,
                "timestamp": timestamp,
                "x_coord": x_coord,
                "y_coord": y,
                "z_coord": z,
                "symbolic_anchors": symbolic_anchors,
                "chat_id": chat_id or str(uuid.uuid4()),
                "metadata": staging_metadata if staging_metadata else None,
            }
            if documents is not None:
                personal_payload["documents"] = list(documents)
            if images is not None:
                personal_payload["images"] = list(images)

            return self.store_personal_memory(
                personal_payload,
                namespace=namespace_arg,
                team_id=resolved_team,
                workspace_id=target_workspace,
                user_id=self.user_id,
                return_status=return_status,
            )

        staged = self.storage_service.stage_manual_memory(
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
            metadata=staging_metadata if staging_metadata else None,
            namespace=namespace_arg,
            team_id=resolved_team,
            user_id=self.user_id,
            share_with_team=resolved_share,
            workspace_id=target_workspace,
            images=images,
        )

        decision = score_staged_memory(
            staged,
            promotion_weights,
            storage_service=self.storage_service,
        )
        if staging_metadata.get("manual_submission") and not decision.should_promote:
            decision = PromotionDecision(
                should_promote=True,
                score=decision.score,
                threshold=decision.threshold,
                weights=decision.weights,
            )
        if not staged.metadata.get("short_term_stored", False) and not decision.should_promote:
            decision = PromotionDecision(
                should_promote=True,
                score=decision.score,
                threshold=decision.threshold,
                weights=decision.weights,
            )

        long_term_id: str | None = None
        if decision.should_promote:
            if decision.score >= 0.75:
                importance_level = MemoryImportanceLevel.HIGH
            elif decision.score >= 0.55:
                importance_level = MemoryImportanceLevel.MEDIUM
            else:
                importance_level = MemoryImportanceLevel.LOW

            processed_memory = ProcessedLongTermMemory(
                content=staged.text,
                summary=staged.text,
                classification=MemoryClassification.CONTEXTUAL,
                importance=importance_level,
                topic=None,
                entities=[],
                keywords=[],
                is_user_context=False,
                is_preference=False,
                is_skill_knowledge=False,
                is_current_project=False,
                duplicate_of=None,
                supersedes=[],
                related_memories=[],
                conversation_id=staged.chat_id,
                confidence_score=0.9,
                emotional_intensity=emotional_intensity,
                x_coord=staged.x_coord,
                y_coord=staged.y_coord,
                z_coord=staged.z_coord,
                symbolic_anchors=staged.symbolic_anchors,
                classification_reason=(
                    f"Manual staging heuristics approval (score={decision.score:.2f})"
                ),
                promotion_eligible=True,
                extraction_timestamp=staged.timestamp,
            )

            stored_images, _, manual_includes = self.storage_service.prepare_image_assets(
                getattr(staged, "images", None),
                namespace=staged.namespace,
            )
            if stored_images is not None:
                processed_memory.images = stored_images
            if manual_includes:
                processed_memory.includes_image = True

            long_term_id = self.db_manager.store_long_term_memory_enhanced(
                processed_memory,
                staged.chat_id,
                staged.namespace,
                workspace_id=target_workspace,
                edited_by_model=self._resolve_last_editor(
                    staged.metadata.get("model")
                    if hasattr(staged, "metadata") and isinstance(staged.metadata, Mapping)
                    else None
                ),
            )

            if long_term_id:
                self._ensure_chat_history_entry(
                    chat_id=staged.chat_id,
                    namespace=staged.namespace,
                    team_id=resolved_team,
                    workspace_id=target_workspace,
                )
                self.storage_service.transfer_spatial_metadata(
                    staged.memory_id,
                    long_term_id,
                    workspace_id=target_workspace,
                )
                if staged.metadata.get("short_term_stored", False):
                    self.storage_service.remove_short_term_memory(staged.memory_id)

                try:
                    anchor_candidates: list[str] = list(staged.symbolic_anchors or [])
                    if staged.anchor and staged.anchor not in anchor_candidates:
                        anchor_candidates.append(staged.anchor)
                    related_ids = (
                        self.storage_service.find_related_memory_ids_by_anchors(
                            anchor_candidates, exclude_memory_ids=[long_term_id]
                        )
                        if anchor_candidates
                        else []
                    )
                    if related_ids:
                        self.db_manager.refresh_memory_last_access(
                            self.namespace,
                            related_ids,
                            workspace_id=target_workspace,
                        )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Failed to refresh related memory access timestamps: %s", exc
                    )

        result = {
            "short_term_id": staged.memory_id,
            "long_term_id": long_term_id,
            "promoted": bool(long_term_id),
            "promotion_score": decision.score,
            "threshold": decision.threshold,
            "weights": decision.weights,
            "short_term_stored": staged.metadata.get("short_term_stored", False),
            "namespace": staged.namespace,
        }
        if resolved_team:
            result["team_id"] = resolved_team

        if result["promoted"]:
            result.update({"status": "promoted", "memory_id": long_term_id})
        else:
            result.update({"status": "staged", "memory_id": staged.memory_id})

        return result if return_status else result["memory_id"]

    @staticmethod
    def _team_payload_to_workspace(
        payload: dict[str, Any], *, include_members: bool = True
    ) -> dict[str, Any]:
        """Convert a team payload to the workspace representation."""

        workspace_payload: dict[str, Any] = dict(payload)
        workspace_payload["workspace_id"] = workspace_payload.pop("team_id", None)
        if not include_members:
            workspace_payload.pop("members", None)
            workspace_payload.pop("admins", None)
        return workspace_payload

    def register_team_space(
        self,
        team_id: str,
        *,
        namespace: str | None = None,
        display_name: str | None = None,
        members: Sequence[str] | None = None,
        admins: Sequence[str] | None = None,
        share_by_default: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
        include_members: bool = True,
    ) -> dict[str, Any]:
        self._ensure_team_memory_enabled()
        space = self.storage_service.register_team_space(
            team_id,
            namespace=namespace,
            display_name=display_name,
            members=members,
            admins=admins,
            share_by_default=share_by_default,
            metadata=metadata,
        )
        return self._team_space_to_dict(space, include_members=include_members)

    # ------------------------------------------------------------------
    # Workspace management wrappers (aliases for team collaboration)
    # ------------------------------------------------------------------

    def register_workspace(
        self,
        workspace_id: str,
        *,
        namespace: str | None = None,
        display_name: str | None = None,
        members: Sequence[str] | None = None,
        admins: Sequence[str] | None = None,
        share_by_default: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
        include_members: bool = True,
    ) -> dict[str, Any]:
        """Alias of :meth:`register_team_space` with workspace semantics."""

        team_payload = self.register_team_space(
            workspace_id,
            namespace=namespace,
            display_name=display_name,
            members=members,
            admins=admins,
            share_by_default=share_by_default,
            metadata=metadata,
            include_members=include_members,
        )
        return self._team_payload_to_workspace(
            team_payload, include_members=include_members
        )

    def list_workspaces(self, *, include_members: bool = False) -> list[dict[str, Any]]:
        """Return workspaces using the underlying team definitions."""

        teams = self.list_team_spaces(include_members=include_members)
        return [
            self._team_payload_to_workspace(team, include_members=include_members)
            for team in teams
        ]

    def get_workspace(
        self, workspace_id: str, *, include_members: bool = True
    ) -> dict[str, Any]:
        """Retrieve a workspace definition (team alias)."""

        team_payload = self.get_team_space(
            workspace_id, include_members=include_members
        )
        return self._team_payload_to_workspace(
            team_payload, include_members=include_members
        )

    def set_workspace_members(
        self,
        workspace_id: str,
        *,
        members: Sequence[str] | None = None,
        admins: Sequence[str] | None = None,
        include_members: bool = True,
    ) -> dict[str, Any]:
        """Replace workspace member/admin rosters (team alias)."""

        team_payload = self.set_team_members(
            workspace_id,
            members=members,
            admins=admins,
            include_members=include_members,
        )
        return self._team_payload_to_workspace(
            team_payload, include_members=include_members
        )

    def add_workspace_members(
        self,
        workspace_id: str,
        members: Sequence[str],
        *,
        as_admin: bool = False,
        include_members: bool = True,
    ) -> dict[str, Any]:
        """Add members to a workspace (team alias)."""

        team_payload = self.add_team_members(
            workspace_id,
            members,
            as_admin=as_admin,
            include_members=include_members,
        )
        return self._team_payload_to_workspace(
            team_payload, include_members=include_members
        )

    def add_workspace_member(
        self,
        workspace_id: str,
        user_id: str,
        *,
        as_admin: bool = False,
        include_members: bool = True,
    ) -> dict[str, Any]:
        """Add a single member to a workspace (team alias)."""

        return self.add_workspace_members(
            workspace_id,
            [user_id],
            as_admin=as_admin,
            include_members=include_members,
        )

    def remove_workspace_member(
        self, workspace_id: str, user_id: str, *, include_members: bool = True
    ) -> dict[str, Any]:
        """Remove a member from a workspace (team alias)."""

        team_payload = self.remove_team_member(
            workspace_id,
            user_id,
            include_members=include_members,
        )
        return self._team_payload_to_workspace(
            team_payload, include_members=include_members
        )

    def set_active_workspace(
        self, workspace_id: str | None, *, enforce_membership: bool = True
    ) -> None:
        """Activate a workspace by delegating to team activation."""

        self.set_active_team(
            workspace_id, enforce_membership=enforce_membership
        )

    def clear_active_workspace(self) -> None:
        """Deactivate any active workspace context."""

        self.clear_active_team()

    def get_active_workspace(self) -> str | None:
        """Return the active workspace identifier."""

        return self.get_active_team()

    def list_team_spaces(
        self, *, include_members: bool = False
    ) -> list[dict[str, Any]]:
        self._ensure_team_memory_enabled()
        return [
            self._team_space_to_dict(space, include_members=include_members)
            for space in self.storage_service.list_team_spaces()
        ]

    def get_team_space(
        self, team_id: str, *, include_members: bool = True
    ) -> dict[str, Any]:
        self._ensure_team_memory_enabled()
        space = self.storage_service.get_team_space(team_id)
        if space is None:
            raise MemoriaError(f"Unknown team: {team_id}")
        return self._team_space_to_dict(space, include_members=include_members)

    def set_team_members(
        self,
        team_id: str,
        *,
        members: Sequence[str] | None = None,
        admins: Sequence[str] | None = None,
        include_members: bool = True,
    ) -> dict[str, Any]:
        self._ensure_team_memory_enabled()
        space = self.storage_service.set_team_members(
            team_id, members=members, admins=admins
        )
        return self._team_space_to_dict(space, include_members=include_members)

    def add_team_members(
        self,
        team_id: str,
        members: Sequence[str],
        *,
        as_admin: bool = False,
        include_members: bool = True,
    ) -> dict[str, Any]:
        self._ensure_team_memory_enabled()
        space = self.storage_service.add_team_members(
            team_id, members, as_admin=as_admin
        )
        return self._team_space_to_dict(space, include_members=include_members)

    def add_team_member(
        self,
        team_id: str,
        user_id: str,
        *,
        as_admin: bool = False,
        include_members: bool = True,
    ) -> dict[str, Any]:
        return self.add_team_members(
            team_id, [user_id], as_admin=as_admin, include_members=include_members
        )

    def remove_team_member(
        self, team_id: str, user_id: str, *, include_members: bool = True
    ) -> dict[str, Any]:
        self._ensure_team_memory_enabled()
        space = self.storage_service.remove_team_member(team_id, user_id)
        return self._team_space_to_dict(space, include_members=include_members)

    def set_active_team(
        self, team_id: str | None, *, enforce_membership: bool = True
    ) -> None:
        self._ensure_team_memory_enabled()
        if team_id is None:
            self._active_team_id = None
            return
        if enforce_membership:
            space = self.storage_service.require_team_access(team_id, self.user_id)
        else:
            space = self.storage_service.get_team_space(team_id)
            if space is None:
                raise MemoriaError(f"Unknown team: {team_id}")
        self._active_team_id = getattr(space, "team_id", team_id)
        self.storage_service.team_id = self._active_team_id
        self.storage_service.workspace_id = getattr(space, "team_id", team_id)

    def clear_active_team(self) -> None:
        self._active_team_id = None
        self.storage_service.team_id = None
        self.storage_service.workspace_id = None

    def get_active_team(self) -> str | None:
        return self._active_team_id

    def get_accessible_namespaces(self) -> set[str]:
        if not self.team_memory_enabled:
            return {self.namespace}
        return self.storage_service.get_accessible_namespaces(self.user_id)

    def _record_ingestion_results(self, results: list[dict[str, Any]]) -> None:
        self._last_ingestion_report = list(results)

    def run_daily_ingestion(
        self, *, promotion_weights: dict[str, float] | None = None
    ) -> list[dict[str, Any]]:
        if not self.ingestion_service:
            self._last_ingestion_report = []
            return []

        results: list[dict[str, Any]] = self.ingestion_service.run_ingestion_pass(
            promotion_weights=promotion_weights
        )
        self._last_ingestion_report = list(results)
        return results

    def get_ingestion_report(self) -> list[dict[str, Any]]:
        return list(self._last_ingestion_report)

    def ingest_thread(
        self,
        payload: ThreadIngestion | dict[str, Any],
        *,
        promotion_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Ingest an ordered conversational thread into storage."""

        if isinstance(payload, ThreadIngestion):
            thread = payload
        else:
            thread = cast(ThreadIngestion, model_validate(ThreadIngestion, payload))

        shared_anchors = canonicalize_symbolic_anchors(
            thread.shared_symbolic_anchors
        ) or []

        aggregated_anchors: list[str] = list(shared_anchors)
        transcript_lines: list[str] = []
        message_results: list[dict[str, Any]] = []
        link_payloads: list[dict[str, Any]] = []
        x_values: list[float] = []
        y_values: list[float] = []
        z_values: list[float] = []
        total_tokens = 0

        ritual_metadata = model_dump(thread.ritual) if thread.ritual else None
        thread_metadata = dict(thread.metadata or {})

        for index, message in enumerate(thread.messages):
            local_anchors = canonicalize_symbolic_anchors(message.symbolic_anchors) or []
            combined_anchor_sequence: list[str] = []
            for candidate in [*shared_anchors, *local_anchors, message.anchor]:
                if candidate and candidate not in combined_anchor_sequence:
                    combined_anchor_sequence.append(candidate)

            for anchor in combined_anchor_sequence:
                if anchor not in aggregated_anchors:
                    aggregated_anchors.append(anchor)

            transcript_lines.append(f"{message.role}: {message.text}")
            total_tokens += int(message.tokens)

            if message.x_coord is not None:
                x_values.append(float(message.x_coord))
            if message.y_coord is not None:
                y_values.append(float(message.y_coord))
            if message.z_coord is not None:
                z_values.append(float(message.z_coord))

            staging_metadata: dict[str, Any] = {
                "thread_id": thread.thread_id,
                "sequence_index": index,
                "role": message.role,
                "shared_symbolic_anchors": shared_anchors,
            }
            if ritual_metadata:
                staging_metadata["ritual"] = ritual_metadata
            if message.metadata:
                staging_metadata["message_metadata"] = message.metadata
            if thread_metadata:
                staging_metadata["thread_metadata"] = thread_metadata

            result = self.store_memory(
                anchor=message.anchor,
                text=message.text,
                tokens=message.tokens,
                timestamp=message.timestamp,
                x_coord=message.x_coord,
                y=message.y_coord,
                z=message.z_coord,
                symbolic_anchors=combined_anchor_sequence,
                emotional_intensity=message.emotional_intensity,
                chat_id=message.chat_id or thread.thread_id,
                promotion_weights=promotion_weights,
                return_status=True,
                metadata=staging_metadata,
            )

            if not isinstance(result, dict):
                raise MemoriaError("Unexpected store_memory result type")

            result_payload = dict(result)
            result_payload.update(
                {
                    "sequence_index": index,
                    "role": message.role,
                    "anchor": message.anchor,
                    "symbolic_anchors": combined_anchor_sequence,
                }
            )
            message_results.append(result_payload)

            link_payloads.append(
                {
                    "memory_id": result["memory_id"],
                    "sequence_index": index,
                    "role": message.role,
                    "anchor": message.anchor,
                    "timestamp": message.timestamp,
                }
            )

        centroid = {
            "x": sum(x_values) / len(x_values) if x_values else None,
            "y": sum(y_values) / len(y_values) if y_values else None,
            "z": sum(z_values) / len(z_values) if z_values else None,
        }

        transcript = "\n".join(transcript_lines)
        last_timestamp = thread.messages[-1].timestamp or datetime.now(timezone.utc)
        if last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)

        chat_metadata: dict[str, Any] = {
            "thread_id": thread.thread_id,
            "shared_symbolic_anchors": aggregated_anchors,
        }
        if ritual_metadata:
            chat_metadata["ritual"] = ritual_metadata
        if thread_metadata:
            chat_metadata["thread_metadata"] = thread_metadata

        self.db_manager.store_chat_history(
            chat_id=f"thread:{thread.thread_id}",
            user_input=transcript,
            ai_output="",
            timestamp=last_timestamp,
            session_id=thread.session_id or thread.thread_id,
            model=self.default_model,
            namespace=self.namespace,
            tokens_used=total_tokens,
            metadata=chat_metadata,
            edited_by_model=self._resolve_last_editor(self.default_model),
        )

        self.storage_service.store_thread(
            thread_id=thread.thread_id,
            shared_anchors=aggregated_anchors,
            ritual=ritual_metadata,
            centroid=centroid,
            message_links=link_payloads,
        )

        return {
            "thread_id": thread.thread_id,
            "shared_symbolic_anchors": aggregated_anchors,
            "ritual": ritual_metadata,
            "centroid": centroid,
            "messages": message_results,
        }

    def update_memory(
        self,
        memory_id: str,
        updates: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Update persisted memory fields via the storage service."""

        payload: dict[str, Any] = {}
        if updates:
            payload.update(updates)
        if kwargs:
            payload.update(kwargs)

        if not payload:
            return False

        try:
            return self.storage_service.update_memory(memory_id, payload)
        except MemoriaError:
            raise
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            raise MemoriaError(f"Failed to update memory {memory_id}: {e}")

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by its ID."""
        try:
            return self.storage_service.delete_memory(memory_id)
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise MemoriaError(f"Failed to delete memory {memory_id}: {e}")

    def get_conversation_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent conversation history."""
        return self.storage_service.get_conversation_history(
            session_id=self._session_id,
            shared_memory=self.shared_memory,
            limit=limit,
            user_id=self.user_id,
            workspace_id=self.get_active_workspace(),
        )

    def clear_memory(self, memory_type: str | None = None) -> None:
        """
        Clear memory data

        Args:
            memory_type: Type of memory to clear ('short_term', 'long_term', 'all')
        """
        try:
            self.storage_service.clear_memory(memory_type)
            logger.info(
                f"Cleared {memory_type or 'all'} memory for namespace: {self.namespace}"
            )
        except Exception as e:
            raise MemoriaError(f"Failed to clear memory: {e}")

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory statistics"""
        try:
            stats = self.db_manager.get_memory_stats(self.namespace)
            return dict(stats)
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}

    @staticmethod
    def _normalize_search_response(
        response: Union[list[dict[str, Any]], dict[str, Any]]
    ) -> dict[str, Any]:
        """Ensure search responses use a consistent dictionary structure."""

        if isinstance(response, dict):
            normalized = dict(response)
            results = normalized.get("results")
            if isinstance(results, list):
                normalized_results = results
            elif isinstance(results, Sequence) and not isinstance(
                results, (str, bytes)
            ):
                normalized_results = list(results)
            elif results is None:
                normalized_results = []
            else:
                normalized_results = [results]
            normalized["results"] = normalized_results
            normalized.setdefault("attempted", [])
            return normalized

        return {"results": list(response), "attempted": []}

    def get_memory_dashboard(
        self,
        limit: int = 100,
        category_filter: list[str] | None = None,
        memory_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Summarize recent memories for a dashboard view.

        Retrieves recent memories using :class:`SearchService` and aggregates
        counts by ``category_primary`` and ``importance_score``.

        Args:
            limit: Maximum number of recent memories to consider.
            category_filter: Optional list of categories to restrict results.
            memory_types: Optional list of memory types (``short_term`` or
                ``long_term``) to include.

        Returns:
            Dictionary containing total count and aggregations by category and
            importance score.
        """
        search_service = self.db_manager._get_search_service()
        try:
            memories = search_service._get_recent_memories(
                namespace=self.namespace,
                category_filter=category_filter,
                limit=limit,
                memory_types=memory_types,
            )
        except Exception as e:
            logger.error(f"Failed to build memory dashboard: {e}")
            return {}
        finally:
            try:
                search_service.session.close()
            except Exception:
                pass

        category_counts: dict[str, int] = defaultdict(int)
        importance_counts: dict[str, int] = defaultdict(int)

        for mem in memories:
            category = mem.get("category_primary") or "uncategorized"
            category_counts[category] += 1

            score = mem.get("importance_score")
            if score is not None:
                score_key = f"{float(score):.1f}"
                importance_counts[score_key] += 1

        return {
            "total_memories": len(memories),
            "by_category": dict(category_counts),
            "by_importance_score": dict(importance_counts),
        }

    @property
    def is_enabled(self) -> bool:
        """Check if memory recording is enabled"""
        return self._enabled

    @property
    def session_id(self) -> str:
        """Get current session ID"""
        return self._session_id

    def get_integration_stats(self) -> list[dict[str, Any]]:
        """Get statistics from the new interceptor system"""
        try:
            # Get system status first
            interceptor_status = self.get_interceptor_status()

            providers: dict[str, dict[str, Any]] = {}
            stats: dict[str, Any] = {
                "integration": "memoria_system",
                "enabled": self._enabled,
                "session_id": self._session_id,
                "namespace": self.namespace,
                "providers": providers,
            }

            # LiteLLM stats
            litellm_interceptor_status = interceptor_status.get("native", {})
            if LITELLM_AVAILABLE:
                providers["litellm"] = {
                    "available": True,
                    "method": "native_callbacks",
                    "enabled": litellm_interceptor_status.get("enabled", False),
                    "status": litellm_interceptor_status.get("status", "unknown"),
                }
            else:
                providers["litellm"] = {
                    "available": False,
                    "method": "native_callbacks",
                    "enabled": False,
                }

            # Get interceptor status instead of checking wrapped attributes
            interceptor_status = self.get_interceptor_status()

            # OpenAI stats
            try:
                import openai

                _ = openai  # Suppress unused import warning

                openai_interceptor_status = interceptor_status.get("openai", {})
                providers["openai"] = {
                    "available": True,
                    "method": "litellm_native",
                    "enabled": openai_interceptor_status.get("enabled", False),
                    "status": openai_interceptor_status.get("status", "unknown"),
                }
            except ImportError:
                providers["openai"] = {
                    "available": False,
                    "method": "litellm_native",
                    "enabled": False,
                }

            # Anthropic stats
            try:
                import anthropic

                _ = anthropic  # Suppress unused import warning

                anthropic_interceptor_status = interceptor_status.get("anthropic", {})
                providers["anthropic"] = {
                    "available": True,
                    "method": "litellm_native",
                    "enabled": anthropic_interceptor_status.get("enabled", False),
                    "status": anthropic_interceptor_status.get("status", "unknown"),
                }
            except ImportError:
                providers["anthropic"] = {
                    "available": False,
                    "method": "litellm_native",
                    "enabled": False,
                }

            # Google Gemini stats
            try:
                import google.generativeai  # noqa: F401

                gemini_interceptor_status = interceptor_status.get("gemini", {})
                providers["gemini"] = {
                    "available": True,
                    "method": "litellm_native",
                    "enabled": gemini_interceptor_status.get("enabled", False),
                    "status": gemini_interceptor_status.get("status", "unknown"),
                }
            except ImportError:
                providers["gemini"] = {
                    "available": False,
                    "method": "litellm_native",
                    "enabled": False,
                }

            return [stats]
        except Exception as e:
            logger.error(f"Failed to get integration stats: {e}")
            return []

    def update_user_context(
        self,
        current_projects: list[str] | None = None,
        relevant_skills: list[str] | None = None,
        user_preferences: list[str] | None = None,
    ) -> None:
        """Update user context for better memory processing"""
        if current_projects is not None:
            self._user_context["current_projects"] = current_projects
        if relevant_skills is not None:
            self._user_context["relevant_skills"] = relevant_skills
        if user_preferences is not None:
            self._user_context["user_preferences"] = user_preferences

        logger.debug(f"Updated user context: {self._user_context}")

    def search_memories(
        self,
        query: str,
        limit: int = 10,
        use_anchor: bool = True,
        keywords: list[str] | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        max_distance: float | None = None,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
        min_importance: float | None = None,
        max_importance: float | None = None,
        anchors: list[str] | None = None,
        namespace: str | None = None,
        team_id: str | None = None,
        share_with_team: bool | None = None,
        **filters: Any,
    ) -> dict[str, Any]:
        """Search memories using the database manager"""
        try:
            target_namespace, _, _ = self._prepare_team_context(
                namespace=namespace,
                team_id=team_id,
                share_with_team=share_with_team,
                require_team=self.team_mode == TeamMode.REQUIRED,
            )
            filters.pop("namespace", None)
            filters.pop("team_id", None)
            filters.pop("share_with_team", None)
            response = self.db_manager.search_memories(
                query=query,
                namespace=target_namespace,
                limit=limit,
                use_anchor=use_anchor,
                keywords=keywords,
                x=x,
                y=y,
                z=z,
                max_distance=max_distance,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                min_importance=min_importance,
                max_importance=max_importance,
                anchors=anchors,
                **filters,
            )
            return self._normalize_search_response(response)
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return {"results": [], "attempted": [], "hint": "Search failed"}

    def search_memories_by_category(
        self,
        category: str,
        limit: int = 10,
        *,
        namespace: str | None = None,
        team_id: str | None = None,
        share_with_team: bool | None = None,
    ) -> dict[str, Any]:
        """Search memories by specific category"""
        try:
            target_namespace, _, _ = self._prepare_team_context(
                namespace=namespace,
                team_id=team_id,
                share_with_team=share_with_team,
                require_team=self.team_mode == TeamMode.REQUIRED,
            )
            response = self.db_manager.search_memories(
                query="",
                namespace=target_namespace,
                category_filter=[category],
                limit=limit,
            )
            return self._normalize_search_response(response)
        except Exception as e:
            logger.error(f"Category search failed: {e}")
            return {"results": [], "attempted": [], "hint": "Search failed"}

    def get_entity_memories(
        self, entity_value: str, entity_type: str | None = None, limit: int = 10
    ) -> dict[str, Any]:
        """Get memories that contain a specific entity"""
        try:
            # This would use the entity index in the database
            # For now, use keyword search as fallback (entity_type is ignored for now)
            response = self.db_manager.search_memories(
                query=entity_value, namespace=self.namespace, limit=limit
            )
            return self._normalize_search_response(response)
        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            return {"results": [], "attempted": [], "hint": "Search failed"}

    def _snapshot_sync_settings(
        self, sync_settings: "SyncSettings" | None
    ) -> dict[str, Any] | None:
        if sync_settings is None:
            return None
        if hasattr(sync_settings, "dict"):
            try:
                return sync_settings.dict()
            except TypeError:
                pass
        if hasattr(sync_settings, "model_dump"):
            try:
                return sync_settings.model_dump()
            except TypeError:
                pass
        if isinstance(sync_settings, dict):
            return dict(sync_settings)
        return {"value": sync_settings}

    def _handle_sync_event(self, event: SyncEvent) -> None:
        storage = getattr(self, "storage_service", None)
        if storage is None:
            return
        try:
            storage.apply_sync_event(event)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.opt(exception=True).warning(
                "Failed to apply inbound sync event: %s", exc
            )

    def _teardown_sync(self) -> None:
        coordinator = self._sync_coordinator
        if coordinator is not None:
            try:
                coordinator.close()
            except Exception:  # pragma: no cover - defensive logging
                logger.opt(exception=True).debug("Error while stopping sync listener")
            self._sync_coordinator = None

        storage = getattr(self, "storage_service", None)
        if storage is not None:
            storage.set_sync_publisher(None)
            if hasattr(storage, "configure_sync_policy"):
                storage.configure_sync_policy(None)

        if self._sync_backend is not None and self._sync_backend_owned:
            try:
                self._sync_backend.close()
            except Exception:  # pragma: no cover - defensive logging
                logger.opt(exception=True).debug("Error while closing sync backend")
        self._sync_backend = None
        self._sync_backend_owned = False
        self._sync_enabled = False

    def configure_sync(
        self,
        sync_settings: "SyncSettings" | None,
        *,
        backend_override: "SyncBackend" | None = None,
    ) -> None:
        """(Re)configure the sync backend using runtime settings."""

        snapshot = self._snapshot_sync_settings(sync_settings)
        enabled = bool(getattr(sync_settings, "enabled", False))

        if not enabled:
            self._teardown_sync()
            self._sync_settings = sync_settings
            self._sync_settings_snapshot = snapshot
            return

        if (
            snapshot == self._sync_settings_snapshot
            and backend_override is None
            and self._sync_coordinator is not None
        ):
            return

        self._teardown_sync()

        backend = backend_override
        backend_owned = False
        if backend is None:
            backend = create_sync_backend(sync_settings)
            backend_owned = True

        coordinator = SyncCoordinator(
            backend=backend,
            namespace=self.namespace,
            origin=self._sync_origin,
            handler=self._handle_sync_event,
        )
        coordinator.start()

        self._sync_backend = backend
        self._sync_backend_owned = backend_owned
        self._sync_coordinator = coordinator
        self._sync_enabled = True
        self._sync_settings = sync_settings
        self._sync_settings_snapshot = snapshot
        if hasattr(self, "storage_service") and self.storage_service is not None:
            self.storage_service.configure_sync_policy(sync_settings)
            self.storage_service.set_sync_publisher(coordinator.publish_event)

    def cleanup(self) -> None:
        """Clean up all async tasks and resources"""
        try:
            self._teardown_sync()
            # Cancel background tasks
            self.conscious_manager.stop()

            # Clean up memory processing tasks
            if hasattr(self, "_memory_tasks"):
                for task in self._memory_tasks.copy():
                    if not task.done():
                        task.cancel()
                self._memory_tasks.clear()

            logger.debug("Memoria cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self) -> None:
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction

    def trigger_conscious_analysis(self) -> Any:
        """Delegate manual analysis trigger to conscious manager."""
        return self.conscious_manager.trigger_analysis()

    def _deduplicate_context_items(
        self, items: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        """Return context items with duplicates removed."""

        if not items:
            return []

        deduplicated: list[dict[str, Any]] = []
        seen_ids = set()
        seen_content = set()

        for item in items:
            memory_id = item.get("memory_id")
            content_value = (
                item.get("searchable_content")
                or item.get("summary")
                or item.get("text")
                or ""
            )

            if isinstance(content_value, str):
                normalized_content = content_value.strip().lower()
            else:
                normalized_content = str(content_value).strip().lower()

            if memory_id and memory_id in seen_ids:
                continue
            if normalized_content and normalized_content in seen_content:
                continue

            if memory_id:
                seen_ids.add(memory_id)
            if normalized_content:
                seen_content.add(normalized_content)

            deduplicated.append(item)

        return deduplicated

    def _get_conscious_context(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch short-term (essential) context for conscious ingest."""

        try:
            storage_service = getattr(self, "storage_service", None)
            if not storage_service:
                return []

            context = storage_service.get_essential_conversations(limit=limit)
            return self._deduplicate_context_items(context)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(f"Failed to retrieve conscious context: {e}")
            return []

    def _get_auto_ingest_context(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories for auto-ingest workflows."""

        if not query:
            return []

        try:
            results: list[dict[str, Any]] = []

            search_engine = getattr(self, "search_engine", None)
            if search_engine:
                try:
                    search_response = search_engine.execute_search(
                        query=query,
                        db_manager=self.db_manager,
                        namespace=self.namespace,
                        limit=limit,
                    )

                    if isinstance(search_response, dict):
                        results = search_response.get("results", []) or []
                    elif isinstance(search_response, list):
                        results = search_response
                    else:
                        results = []
                except Exception as search_error:
                    logger.warning(
                        f"Auto-ingest search engine retrieval failed: {search_error}"
                    )
                    results = []

            if not results:
                storage_service = getattr(self, "storage_service", None)
                if storage_service:
                    results = storage_service.retrieve_context(query, limit) or []
                else:
                    results = []

            return self._deduplicate_context_items(results)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(f"Failed to retrieve auto-ingest context: {e}")
            return []

    def prepare_context_window(
        self,
        mode: str,
        query: str | None = None,
        *,
        provider_name: str | None = None,  # noqa: ARG002 - reserved for future use
    ) -> tuple[list[dict[str, Any]], ContextPlan | None]:
        """Return context rows and orchestration plan for the requested mode."""

        orchestrator = self.context_orchestrator
        plan: ContextPlan | None = None
        limit = self.context_limit or 5

        query_text = (query or "").strip()

        if orchestrator is not None:
            plan = orchestrator.plan_initial_window(
                mode=mode,
                query=query_text,
                provider_name=provider_name,
            )
            limit = plan.max_items

        if mode == "conscious":
            context = self._get_conscious_context(limit=limit)
        elif mode == "auto":
            if not query_text:
                context = []
            elif self.search_engine:
                context = self._get_auto_ingest_context(query_text, limit=limit)
            else:
                context = self.retrieve_context(query_text, limit=limit)
        else:
            context = []

        if orchestrator is not None and context:
            context = orchestrator.apply_plan(plan, context)

        return context, plan

    def get_conscious_system_prompt(self) -> str:
        """
        Get conscious context as system prompt for direct injection.
        Returns ALL short-term memory as formatted system prompt.
        Use this for conscious_ingest mode.
        """
        try:
            context = self._get_conscious_context()
            if not context:
                return ""

            # Create system prompt with all short-term memory
            system_prompt = "--- Your Short-Term Memory (Conscious Context) ---\n"
            system_prompt += "This is your complete working memory. USE THIS INFORMATION TO ANSWER QUESTIONS:\n\n"

            # Deduplicate and format context
            seen_content = set()
            for mem in context:
                if isinstance(mem, dict):
                    content = mem.get("searchable_content", "") or mem.get(
                        "summary", ""
                    )
                    category = mem.get("category_primary", "")

                    # Skip duplicates
                    content_key = content.lower().strip()
                    if content_key in seen_content:
                        continue
                    seen_content.add(content_key)

                    system_prompt += f"[{category.upper()}] {content}\n"

            system_prompt += "\nIMPORTANT: Use the above information to answer questions about the user.\n"
            system_prompt += "-------------------------\n"

            return system_prompt

        except Exception as e:
            logger.error(f"Failed to generate conscious system prompt: {e}")
            return ""

    def get_auto_ingest_system_prompt(self, user_input: str) -> str:
        """
        Get auto-ingest context as system prompt for direct injection.
        Returns relevant memories based on user input as formatted system prompt.
        Use this for auto_ingest mode.
        """
        try:
            # Use intelligent retrieval based on current user input
            # This leverages the memory search engine rather than
            # relying solely on recent short-term memories.
            context = self._get_auto_ingest_context(user_input)

            if not context:
                return ""

            # Create system prompt with relevant memories (limited to prevent overwhelming)
            system_prompt = "--- Relevant Memory Context ---\n"

            # Take first 5 items to avoid too much context
            seen_content = set()
            for mem in context[:5]:
                if isinstance(mem, dict):
                    content = mem.get("searchable_content", "") or mem.get(
                        "summary", ""
                    )
                    category = mem.get("category_primary", "")

                    # Skip duplicates
                    content_key = content.lower().strip()
                    if content_key in seen_content:
                        continue
                    seen_content.add(content_key)

                    if category.startswith("essential_"):
                        system_prompt += f"[{category.upper()}] {content}\n"
                    else:
                        system_prompt += f"- {content}\n"

            system_prompt += "-------------------------\n"

            return system_prompt

        except Exception as e:
            logger.error(f"Failed to generate auto-ingest system prompt: {e}")
            return ""

    def add_memory_to_messages(
        self,
        messages: list[dict[str, Any]],
        user_input: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add appropriate memory context to messages based on ingest mode.

        Args:
            messages: list of messages for LLM
            user_input: User input for auto_ingest context retrieval (optional)

        Returns:
            Modified messages list with memory context added as system message
        """
        try:
            system_prompt = ""

            if self.conscious_ingest:
                # One-time conscious context injection
                if not self._conscious_context_injected:
                    system_prompt = self.get_conscious_system_prompt()
                    self._conscious_context_injected = True
                    logger.info(
                        "Conscious-ingest: Added complete working memory to system prompt"
                    )
                else:
                    logger.debug("Conscious-ingest: Context already injected, skipping")

            elif self.auto_ingest and user_input:
                # Dynamic auto-ingest based on user input
                system_prompt = self.get_auto_ingest_system_prompt(user_input)
                logger.debug("Auto-ingest: Added relevant context to system prompt")

            if system_prompt:
                # Add to existing system message or create new one
                messages_copy = list(messages)

                # Check if system message already exists
                for msg in messages_copy:
                    if msg.get("role") == "system":
                        msg["content"] = system_prompt + "\n" + msg.get("content", "")
                        return messages_copy

                # No system message exists, add one at the beginning
                messages_copy.insert(0, {"role": "system", "content": system_prompt})
                return messages_copy

            return messages

        except Exception as e:
            logger.error(f"Failed to add memory to messages: {e}")
            return messages

    def get_essential_conversations(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get essential conversations from short-term memory"""
        return self.storage_service.get_essential_conversations(limit)

    def create_openai_client(self, **kwargs: Any) -> Any:
        """
        Create an OpenAI client with automatic memory recording.

        This method creates a MemoriaOpenAIInterceptor that automatically records
        all OpenAI API calls to memory using the inheritance-based approach.

        Args:
            **kwargs: Additional arguments passed to OpenAI client (e.g., api_key)
                     These override any settings from the Memoria provider config

        Returns:
            MemoriaOpenAIInterceptor instance that works as a drop-in replacement
            for the standard OpenAI client

        Example:
            memoria = Memoria(api_key="sk-...")
            memoria.enable()

            # Create interceptor client
            client = memoria.create_openai_client()

            # Use exactly like standard OpenAI client
            response = client.chat.completions.create(
                model=memoria.model,
                messages=[{"role": "user", "content": "Hello!"}]
            )
            # Conversation is automatically recorded
        """
        try:
            from ..integrations.openai_client import create_openai_client

            return create_openai_client(self, self.provider_config, **kwargs)
        except ImportError as e:
            logger.error(f"Failed to import OpenAI integration: {e}")
            raise ImportError(
                "OpenAI integration not available. Install with: pip install openai"
            ) from e

    def create_openai_wrapper(self, **kwargs: Any) -> Any:
        """
        Create a legacy OpenAI wrapper (backward compatibility).

        DEPRECATED: Use create_openai_client() instead for better integration.

        Returns:
            MemoriaOpenAI wrapper instance
        """
        try:
            from ..integrations.openai_client import MemoriaOpenAI

            return MemoriaOpenAI(self, **kwargs)
        except ImportError as e:
            logger.error(f"Failed to import OpenAI integration: {e}")
            raise ImportError(
                "OpenAI integration not available. Install with: pip install openai"
            ) from e

    # Conversation management methods

    def get_conversation_stats(self) -> dict[str, Any]:
        """Get conversation manager statistics"""
        return self.conversation_manager.get_session_stats()

    def clear_conversation_history(self, session_id: str | None = None) -> None:
        """
        Clear conversation history

        Args:
            session_id: Specific session to clear. If None, clears current session.
        """
        if session_id is None:
            session_id = self._session_id
        self.conversation_manager.clear_session(session_id)
        logger.info(f"Cleared conversation history for session: {session_id}")

    def clear_all_conversations(self) -> None:
        """Clear all conversation histories"""
        self.conversation_manager.clear_all_sessions()
        logger.info("Cleared all conversation histories")

    def start_new_conversation(self) -> str:
        """
        Start a new conversation session

        Returns:
            New session ID
        """
        old_session_id = self._session_id
        self._session_id = str(uuid.uuid4())

        # Reset conscious context injection flag for new conversation
        self._conscious_context_injected = False

        logger.info(
            f"Started new conversation: {self._session_id} (previous: {old_session_id})"
        )
        return self._session_id

    def get_current_session_id(self) -> str:
        """Get current conversation session ID"""
        return self._session_id
