from __future__ import annotations

"""
Pydantic-based configuration settings for Memoria
"""

import inspect
import json
import os
from collections.abc import Mapping, MutableMapping, Sequence
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
)
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, root_validator, validator

from ..policy.schemas import PolicyDefinitions

try:  # pragma: no cover - compatibility shim for Pydantic < 2
    from pydantic import model_validator  # type: ignore
except ImportError:  # pragma: no cover - executed on Pydantic 1.x

    def model_validator(*, mode: str = "after", **kwargs):  # type: ignore
        if mode != "after":  # pragma: no cover - defensive guard
            raise RuntimeError("Only 'after' mode is supported with Pydantic 1.x")

        def decorator(func):
            def wrapper(cls, values):
                instance = cls.construct(**values)
                result = func(instance)
                target = result if isinstance(result, cls) else instance
                return target.dict()

            wrapper.__name__ = getattr(func, "__name__", "model_validator")
            return root_validator(pre=False, **kwargs)(wrapper)

        return decorator


class LogLevel(str, Enum):
    """Logging levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


SUPPORTED_DATABASE_SCHEME_PREFIXES = (
    "sqlite",
    "postgres",
    "postgresql",
    "mysql",
)


class DatabaseType(str, Enum):
    """Supported database types"""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class RetentionPolicy(str, Enum):
    """Memory retention policies"""

    DAYS_7 = "7_days"
    DAYS_30 = "30_days"
    DAYS_90 = "90_days"
    PERMANENT = "permanent"


class CoordinateAuditCadence(str, Enum):
    """Supported cadences for the coordinate audit scheduler."""

    WEEKLY = "weekly"
    DISABLED = "disabled"


class RetentionPolicyAction(str, Enum):
    """Actions available when a retention policy rule is triggered."""

    BLOCK = "block"
    ESCALATE = "escalate"
    LOG = "log"


class RetentionPolicyRuleSettings(BaseModel):
    """Declarative governance controls for memory retention."""

    name: str = Field(..., description="Friendly identifier for the policy rule")
    namespaces: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Namespaces or glob patterns this rule applies to",
    )
    privacy_ceiling: float | None = Field(
        default=None,
        description="Upper bound for the privacy (Y axis) value during retention updates",
    )
    importance_floor: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum importance score permitted by this policy",
    )
    lifecycle_days: float | None = Field(
        default=None,
        gt=0.0,
        description="Maximum age (in days) before a memory must be escalated instead of decayed",
    )
    action: RetentionPolicyAction = Field(
        default=RetentionPolicyAction.BLOCK,
        description="How the scheduler should respond when the rule is violated",
    )
    escalate_to: str | None = Field(
        default=None,
        description="Optional escalation target or queue identifier",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata forwarded with audit events",
    )

    @validator("namespaces", pre=True)
    def _coerce_namespaces(cls, value: Any) -> list[str]:
        if value in (None, "", []):
            return ["*"]
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            return parts or ["*"]
        if isinstance(value, (list, tuple, set)):
            cleaned: list[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned.append(text)
            return cleaned or ["*"]
        raise TypeError(
            "namespaces must be provided as a string or iterable of strings"
        )

    @validator("escalate_to")
    def _clean_escalation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @validator("metadata", pre=True)
    def _normalise_metadata(cls, value: Any) -> dict[str, Any]:
        if value in (None, "", {}):
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError("Policy metadata must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError("Policy metadata JSON must define an object")
            return parsed
        raise TypeError("Policy metadata must be a mapping or JSON object")

    @root_validator(skip_on_failure=True)
    def _ensure_constraints(
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover - validated via integration tests
        if (
            values.get("privacy_ceiling") is None
            and values.get("importance_floor") is None
            and values.get("lifecycle_days") is None
        ):
            raise ValueError(
                "Retention policy rules must define at least one constraint such as "
                "privacy_ceiling, importance_floor, or lifecycle_days"
            )
        return values


class TeamMode(str, Enum):
    """Operating modes for collaborative team memory."""

    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED = "required"


class IngestMode(str, Enum):
    """Supported ingestion profiles for memory processing."""

    STANDARD = "standard"
    PERSONAL = "personal"


class SyncBackendType(str, Enum):
    """Supported synchronisation backends."""

    NONE = "none"
    MEMORY = "memory"
    REDIS = "redis"
    POSTGRES = "postgres"


class DatabaseSettings(BaseModel):
    """Database configuration settings"""

    connection_string: str = Field(
        default="sqlite:///memoria.db", description="Database connection string"
    )
    database_type: DatabaseType = Field(
        default=DatabaseType.SQLITE, description="Type of database backend"
    )
    template: str = Field(default="basic", description="Database template to use")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    echo_sql: bool = Field(default=False, description="Echo SQL statements to logs")
    migration_auto: bool = Field(
        default=True, description="Automatically run migrations"
    )
    backup_enabled: bool = Field(default=False, description="Enable automatic backups")
    backup_interval_hours: int = Field(
        default=24, ge=1, description="Backup interval in hours"
    )

    @validator("connection_string")
    def validate_connection_string(cls, v):
        """Validate database connection string"""
        if not v:
            raise ValueError("Connection string cannot be empty")

        parsed = urlsplit(v)
        scheme = parsed.scheme.lower()
        if not scheme:
            raise ValueError(
                "Connection string must include a URI scheme (e.g. sqlite:///memoria.db)"
            )

        base_scheme = scheme.split("+", 1)[0]
        if not any(
            base_scheme.startswith(prefix)
            for prefix in SUPPORTED_DATABASE_SCHEME_PREFIXES
        ):
            raise ValueError(f"Unsupported database type in connection string: {v}")

        return v


class TaskModelRoute(BaseModel):
    """Mapping of a logical task to a provider/model combination."""

    provider: str = Field(..., description="Name of the provider or 'primary'.")
    model: str | None = Field(
        default=None,
        description="Optional model override for the task.",
    )
    fallback: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of fallback providers expressed as names or 'provider:model' pairs."
        ),
    )

    @validator("provider")
    def _validate_provider(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Task routing provider must be a non-empty string")
        return value.strip()

    @validator("fallback", pre=True)
    def _normalize_fallback(cls, value: Any) -> list[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, Mapping):
            normalized: list[str] = []
            for key, model in value.items():
                key_str = str(key).strip()
                if not key_str:
                    continue
                model_str = str(model).strip() if model is not None else ""
                normalized.append(f"{key_str}:{model_str}" if model_str else key_str)
            return [item for item in normalized if item]
        if isinstance(value, (list, tuple, set)):
            normalized: list[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    normalized.append(text)
            return normalized
        raise TypeError(
            "Fallback configuration must be provided as a string, list, or mapping"
        )

    def to_spec(self) -> "TaskRouteSpec":
        """Convert to a routing specification consumed by the provider registry."""

        from memoria.core.providers import TaskRouteSpec

        return TaskRouteSpec(
            provider=self.provider,
            model=self.model,
            fallback=tuple(self.fallback),
        )


def _default_task_model_routes() -> dict[str, TaskModelRoute]:
    """Return default routing preferences for Memoria agents."""

    return {
        "memory_ingest": TaskModelRoute(provider="primary", fallback=["openai"]),
        "search_planning": TaskModelRoute(
            provider="primary", fallback=["anthropic", "openai"]
        ),
    }


class AgentSettings(BaseModel):
    """AI agent configuration settings"""

    openai_api_key: str | None = Field(
        default=None, description="OpenAI API key for memory processing"
    )
    anthropic_api_key: str | None = Field(
        default=None, description="Anthropic API key for memory processing"
    )
    anthropic_model: str | None = Field(
        default=None,
        description="Preferred Anthropic model for ingestion or retrieval",
    )
    anthropic_base_url: str | None = Field(
        default=None, description="Optional override for the Anthropic API base URL"
    )
    gemini_api_key: str | None = Field(
        default=None, description="Google Gemini API key for memory processing"
    )
    gemini_model: str | None = Field(
        default=None, description="Preferred Gemini model for ingestion or retrieval"
    )
    default_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "Default model for memory processing (override with MEMORIA_AGENTS__DEFAULT_MODEL)"
        ),
    )
    fallback_model: str = Field(
        default="gpt-3.5-turbo",
        description="Fallback model if default fails",
    )
    max_tokens: int = Field(
        default=2000, ge=100, le=8000, description="Maximum tokens per API call"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Model temperature for memory processing",
    )
    timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="API timeout in seconds"
    )
    retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts for failed API calls",
    )
    conscious_ingest: bool = Field(
        default=False, description="Enable intelligent memory filtering"
    )
    task_model_routes: dict[str, TaskModelRoute] = Field(
        default_factory=_default_task_model_routes,
        description="Per-task routing preferences for model selection.",
    )

    @validator("openai_api_key")
    def validate_api_key(cls, v):
        """Validate API key format allowing OpenAI and other providers."""
        if v is None:
            return v

        key = v.strip()
        if not key:
            raise ValueError("API key must be a non-empty string")

        if key.startswith("sk-") and len(key) < 20:
            raise ValueError("OpenAI API key starting with 'sk-' appears malformed")

        return key

    @validator("anthropic_api_key", "gemini_api_key")
    def _validate_optional_provider_key(cls, value: str | None) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError("API key must be provided as a string")

        key = value.strip()
        if not key:
            raise ValueError("API key must be a non-empty string")

        return key

    @validator("anthropic_base_url", "anthropic_model", "gemini_model", pre=True)
    def _normalize_optional_strings(cls, value):
        if value is None:
            return None

        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None

        raise TypeError("Value must be provided as a string")

    @validator("task_model_routes", pre=True)
    def _normalize_task_routes(cls, value: Any) -> dict[str, TaskModelRoute]:
        if value in (None, "", {}):
            return {}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("Task model routes must be valid JSON") from exc
            value = parsed
        if isinstance(value, Mapping):
            normalized: dict[str, TaskModelRoute] = {}
            for task, route in value.items():
                if isinstance(route, TaskModelRoute):
                    normalized[str(task)] = route
                else:
                    normalized[str(task)] = TaskModelRoute.parse_obj(route)
            return normalized
        raise TypeError(
            "Task model routes must be provided as a mapping or JSON serialisable object"
        )

    def export_task_routes(self) -> dict[str, "TaskRouteSpec"]:
        """Return task routing information as provider registry specs."""


        return {
            task: route.to_spec()
            for task, route in (self.task_model_routes or {}).items()
        }


class LoggingSettings(BaseModel):
    """Logging configuration settings"""

    level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Log message format",
    )
    log_to_file: bool = Field(default=False, description="Enable logging to file")
    log_file_path: str = Field(default="logs/memoria.log", description="Log file path")
    log_rotation: str = Field(default="10 MB", description="Log rotation size")
    log_retention: str = Field(default="30 days", description="Log retention period")
    log_compression: str = Field(default="gz", description="Log compression format")
    structured_logging: bool = Field(
        default=False, description="Enable structured JSON logging"
    )


class MemorySettings(BaseModel):
    """Memory processing configuration"""

    ingest_mode: IngestMode = Field(
        default=IngestMode.STANDARD,
        description=(
            "Selects the ingestion workflow: 'standard' follows the default"
            " pipeline while 'personal' enables personal memory handling."
        ),
    )
    personal_documents_enabled: bool | None = Field(
        default=None,
        description=(
            "Toggle support for attaching structured personal documents during"
            " ingestion. Defaults to auto-enabling when ingest_mode is"
            " 'personal'."
        ),
    )
    namespace: str = Field(
        default="default", description="Default namespace for memory isolation"
    )
    shared_memory: bool = Field(
        default=False, description="Enable shared memory across agents"
    )
    team_memory_enabled: bool = Field(
        default=False,
        description="Enable collaborative team memory namespaces",
    )
    team_namespace_prefix: str = Field(
        default="team",
        description="Prefix used when generating team namespace identifiers",
    )
    team_enforce_membership: bool = Field(
        default=True,
        description="Require membership before granting access to team namespaces",
    )
    team_mode: TeamMode = Field(
        default=TeamMode.DISABLED,
        description=(
            "(Experimental) Controls how team collaboration behaves: 'disabled' (off),"
            " 'optional' (team context is available but not required), or"
            " 'required' (a team must be selected for memory operations)."
        ),
    )
    team_default_id: str | None = Field(
        default=None,
        description="Team identifier to activate automatically when available.",
    )
    team_share_by_default: bool = Field(
        default=False,
        description=(
            "Whether new team spaces should default to sharing stored memories"
            " with the team when an explicit preference is not provided."
        ),
    )
    workspace_mode: TeamMode = Field(
        default=TeamMode.DISABLED,
        description=(
            "Controls workspace collaboration behaviour. Mirrors team_mode but"
            " allows services to opt into workspace terminology."
        ),
    )
    workspace_default_id: str | None = Field(
        default=None,
        description=(
            "Workspace identifier to activate automatically when one is not"
            " provided explicitly."
        ),
    )

    @validator("team_mode", pre=True)
    def _coerce_team_mode(cls, value: Any) -> TeamMode:
        if value is None or value == "":
            return TeamMode.DISABLED
        if isinstance(value, TeamMode):
            return value
        if isinstance(value, bool):
            return TeamMode.OPTIONAL if value else TeamMode.DISABLED
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return TeamMode.DISABLED
            if normalized in {"true", "1", "enabled", "on"}:
                return TeamMode.OPTIONAL
            if normalized in {"false", "0", "disabled", "off"}:
                return TeamMode.DISABLED
            try:
                return TeamMode(normalized)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid team mode '{value}'") from exc
        raise TypeError("team_mode must be a string, boolean, or TeamMode value")

    @validator("team_default_id", pre=True)
    def _clean_team_default_id(cls, value: Any) -> str | None:
        if value in (None, "", b""):
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        raise TypeError("team_default_id must be a string or null")

    @validator("team_share_by_default", pre=True)
    def _coerce_team_share_default(cls, value: Any) -> bool:
        if value in (None, "", b""):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        try:
            return bool(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError("team_share_by_default must be coercible to bool") from exc

    @validator("workspace_mode", pre=True)
    def _coerce_workspace_mode(cls, value: Any) -> TeamMode:
        return cls._coerce_team_mode(value)

    @validator("workspace_default_id", pre=True)
    def _clean_workspace_default_id(cls, value: Any) -> str | None:
        return cls._clean_team_default_id(value)

    @validator("ingest_mode", pre=True)
    def _coerce_ingest_mode(cls, value: Any) -> IngestMode:
        if value in (None, "", b""):
            return IngestMode.STANDARD
        if isinstance(value, IngestMode):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return IngestMode.STANDARD
            try:
                return IngestMode(normalized)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Invalid ingest mode '{value}'") from exc
        raise TypeError("ingest_mode must be a string or IngestMode value")

    @validator("personal_documents_enabled", pre=True)
    def _coerce_personal_documents(cls, value: Any) -> bool | None:
        if value in (None, "", b""):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        try:
            return bool(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                "personal_documents_enabled must be coercible to bool"
            ) from exc

    @root_validator(skip_on_failure=True)
    def _synchronise_team_configuration(cls, values: dict[str, Any]) -> dict[str, Any]:
        mode = values.get("team_mode") or TeamMode.DISABLED
        enabled = bool(values.get("team_memory_enabled"))
        default_team = values.get("team_default_id")

        if enabled and mode == TeamMode.DISABLED:
            mode = TeamMode.OPTIONAL
        if default_team and mode == TeamMode.DISABLED:
            mode = TeamMode.OPTIONAL

        if mode == TeamMode.DISABLED:
            values["team_memory_enabled"] = False
            values["team_default_id"] = None
            values["team_share_by_default"] = False
        else:
            values["team_memory_enabled"] = True

        values["team_mode"] = mode
        return values

    @root_validator(skip_on_failure=True)
    def _synchronise_ingest_mode(cls, values: dict[str, Any]) -> dict[str, Any]:
        mode = values.get("ingest_mode") or IngestMode.STANDARD
        documents_enabled = values.get("personal_documents_enabled")

        if documents_enabled is None:
            values["personal_documents_enabled"] = mode == IngestMode.PERSONAL
        else:
            coerced_flag = bool(documents_enabled)
            if coerced_flag and mode != IngestMode.PERSONAL:
                raise ValueError(
                    "personal_documents_enabled can only be true when ingest_mode"
                    " is 'personal'"
                )
            values["personal_documents_enabled"] = coerced_flag

        values["ingest_mode"] = mode
        return values

    sovereign_ingest: bool = Field(
        default=False,
        description=(
            "(Experimental) Also known as 'Conscious Ingestion', this disables "
            "automatic callback registration so conversations are only persisted "
            "and processed manually."
        ),
    )
    retention_policy: RetentionPolicy = Field(
        default=RetentionPolicy.DAYS_30, description="Default memory retention policy"
    )
    auto_cleanup: bool = Field(
        default=True, description="Enable automatic cleanup of expired memories"
    )
    cleanup_interval_hours: int = Field(
        default=24, ge=1, description="Cleanup interval in hours"
    )
    importance_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum importance score for memory retention",
    )
    max_short_term_memories: int = Field(
        default=1000, ge=10, description="Maximum number of short-term memories"
    )
    max_long_term_memories: int = Field(
        default=10000, ge=100, description="Maximum number of long-term memories"
    )
    context_injection: bool = Field(
        default=False, description="Enable context injection in conversations"
    )
    context_limit: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of context memories to inject",
    )
    context_orchestration: bool = Field(
        default=True,
        description="Enable adaptive context window orchestration",
    )
    context_small_query_limit: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Target memories for short queries",
    )
    context_large_query_limit: int = Field(
        default=6,
        ge=1,
        le=25,
        description="Target memories for complex queries",
    )
    context_max_prompt_memories: int = Field(
        default=8,
        ge=1,
        le=25,
        description="Upper bound on context memories injected per prompt",
    )
    context_small_query_token_threshold: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Word/token threshold distinguishing small queries",
    )
    context_token_budget: int = Field(
        default=600,
        ge=50,
        le=4000,
        description="Soft token budget allocated to context memories",
    )
    context_privacy_floor: float = Field(
        default=-10.0,
        description="Privacy value at or below which memories are skipped",
    )
    context_usage_boost_threshold: int = Field(
        default=3,
        ge=0,
        le=100,
        description="Minimum access count before boosting a memory",
    )
    coordinate_audit_enabled: bool = Field(
        default=True,
        description="Enable the scheduled coordinate audit maintenance loop",
    )
    coordinate_audit_lookback_days: int = Field(
        default=7,
        ge=1,
        description="Number of days of history to rescan on subsequent coordinate audits",
    )
    coordinate_audit_cadence: CoordinateAuditCadence = Field(
        default=CoordinateAuditCadence.WEEKLY,
        description="Cadence controlling how often the coordinate audit runs",
    )
    context_usage_boost_limit: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Additional slots granted to boosted memories",
    )
    context_usage_cache_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="How long to cache analytics usage data in seconds",
    )
    context_analytics_window_days: int = Field(
        default=7,
        ge=1,
        le=365,
        description="Days of analytics history to consider for context planning",
    )
    context_analytics_top_n: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of high-usage memories to fetch from analytics",
    )
    retention_update_interval_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,
        description="Minutes between retention reinforcement cycles",
    )
    retention_decay_half_life_hours: float = Field(
        default=72.0,
        gt=0.0,
        description="Half-life (in hours) used to decay importance scores",
    )
    conscious_analysis_interval_seconds: int = Field(
        default=6 * 60 * 60,
        ge=60,
        description=("Interval in seconds between background conscious analysis runs"),
    )
    retention_reinforcement_bonus: float = Field(
        default=0.05,
        ge=0.0,
        description="Scaling factor applied when memories are frequently accessed",
    )
    retention_privacy_shift: float = Field(
        default=0.5,
        description="Amount to nudge privacy axis toward public when reinforced",
    )
    retention_importance_floor: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum allowable importance score after decay",
    )
    retention_policy_rules: list[RetentionPolicyRuleSettings] = Field(
        default_factory=list,
        description=(
            "Optional governance rules that limit retention updates or trigger "
            "escalations when privacy or lifecycle constraints are violated"
        ),
    )

    @validator("context_max_prompt_memories")
    def _validate_prompt_ceiling(
        cls, value, values
    ):  # noqa: ANN001 - pydantic signature
        small_limit = values.get("context_small_query_limit", value)
        large_limit = values.get("context_large_query_limit", value)
        if value < small_limit or value < large_limit:
            raise ValueError(
                "context_max_prompt_memories must be greater than or equal to both "
                "context_small_query_limit and context_large_query_limit"
            )
        return value

    @validator("retention_policy_rules", pre=True)
    def _normalise_retention_policies(cls, value: Any) -> list[Any]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("retention_policy_rules must be valid JSON") from exc
            value = parsed
        if isinstance(value, Mapping):
            return [value]
        if isinstance(value, list):
            return value
        raise TypeError(
            "retention_policy_rules must be provided as a list, mapping, or JSON string"
        )


class IntegrationSettings(BaseModel):
    """Integration configuration"""

    litellm_enabled: bool = Field(
        default=False,  # Default to False to prevent unintended LLM calls
        description="(Experimental) Enable LiteLLM integration for routing to over 100+ LLMs",
    )
    openai_wrapper_enabled: bool = Field(
        default=False, description="Enable OpenAI wrapper integration"
    )
    anthropic_wrapper_enabled: bool = Field(
        default=False, description="Enable Anthropic wrapper integration"
    )
    auto_enable_on_import: bool = Field(
        default=False, description="Automatically enable integrations on import"
    )
    callback_timeout: int = Field(
        default=5, ge=1, le=30, description="Callback timeout in seconds"
    )


class SyncSettings(BaseModel):
    """Configuration for cross-instance change propagation."""

    enabled: bool = Field(default=False, description="Enable storage synchronisation")
    backend: SyncBackendType = Field(
        default=SyncBackendType.NONE,
        description="Sync backend implementation to use",
    )
    connection_url: str | None = Field(
        default=None,
        description="Connection string for the selected backend",
    )
    channel: str = Field(
        default="memoria-sync",
        description="Logical pub/sub channel used by the backend",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend specific keyword arguments",
    )
    realtime_replication: bool = Field(
        default=False,
        description="Broadcast full payloads and apply inbound changes for near-real-time replicas",
    )
    privacy_floor: float | None = Field(
        default=None,
        description="Lower bound on the privacy (Y) axis for events to publish",
    )
    privacy_ceiling: float | None = Field(
        default=None,
        description="Upper bound on the privacy (Y) axis for events to publish",
    )

    @validator("backend", pre=True)
    def _normalize_backend(cls, value: Any) -> SyncBackendType:
        if isinstance(value, SyncBackendType):
            return value
        if isinstance(value, str) and value.strip():
            normalized = value.strip().lower()
            for member in SyncBackendType:
                if normalized == member.value:
                    return member
            if normalized in {"inmemory", "in-memory"}:
                return SyncBackendType.MEMORY
        return SyncBackendType.NONE

    @validator("connection_url")
    def _validate_connection(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        return stripped

    @validator("channel")
    def _validate_channel(cls, value: str) -> str:
        if value is None:
            return "memoria-sync"
        channel = value.strip()
        return channel or "memoria-sync"

    @validator("privacy_floor", "privacy_ceiling", pre=True)
    def _coerce_privacy_bounds(cls, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Privacy bounds must be numeric values") from exc

    @validator("privacy_ceiling")
    def _validate_privacy_order(
        cls, ceiling: float | None, values: dict[str, Any]
    ) -> float | None:
        floor = values.get("privacy_floor")
        if ceiling is not None and floor is not None and ceiling < floor:
            raise ValueError(
                "privacy_ceiling must be greater than or equal to privacy_floor"
            )
        return ceiling


class PluginSettings(BaseModel):
    """Configuration for optional Memoria plugins."""

    name: str | None = Field(
        default=None,
        description="Friendly identifier for the plugin (defaults to class name)",
    )
    import_path: str = Field(
        ...,
        description="Python import path to the plugin (module:Class or module.Class)",
    )
    enabled: bool = Field(default=True, description="Whether the plugin is active")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary keyword arguments passed to the plugin constructor",
    )

    @validator("import_path")
    def _validate_import_path(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Plugin import_path must be a non-empty string")
        return value.strip()

    @validator("options", pre=True)
    def _ensure_mapping(cls, value: Any) -> dict[str, Any]:
        if value in (None, "", {}):
            return {}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("Plugin options must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError("Plugin options JSON must define an object")
            return parsed
        if isinstance(value, dict):
            return value
        raise TypeError("Plugin options must be provided as a mapping or JSON string")


class MemoriaSettings(BaseModel):
    """Main Memoria configuration"""

    version: str = Field(default="1.0.0", description="Configuration version")
    debug: bool = Field(default=False, description="Enable debug mode")
    verbose: bool = Field(
        default=False, description="Enable verbose logging (loguru only)"
    )
    enable_cluster_indexing: bool = Field(
        default=True, description="Toggle automatic cluster index builds"
    )
    enable_heuristic_clustering: bool = Field(
        default=True, description="Enable heuristic clustering"
    )
    enable_vector_clustering: bool = Field(
        default=False,
        description="(Experimental) Enable vector embedding clustering",
    )
    enable_vector_search: bool = Field(
        default=False,
        description="(Experimental) Enable vector similarity scoring during search",
    )
    use_db_clusters: bool = Field(
        default=True, description="Store clusters in the database"
    )
    cluster_index_path: Path | None = Field(
        default=None,
        description="Optional path for legacy JSON-based cluster index",
    )
    cluster_gravity_lambda: float = Field(
        default=1.0 / (7 * 24 * 60 * 60),
        description="Decay rate λ for heuristic cluster gravity weighting",
    )

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    integrations: IntegrationSettings = Field(default_factory=IntegrationSettings)
    sync: SyncSettings = Field(
        default_factory=SyncSettings,
        description="(Experimental) Configuration for cross-instance change propagation.",
    )
    plugins: list[PluginSettings] = Field(
        default_factory=list,
        description="list of plugins to load when Memoria starts",
    )
    policy: PolicyDefinitions = Field(
        default_factory=PolicyDefinitions,
        description=(
            "(Experimental) Structured governance policies including retention ceilings, privacy "
            "floors, escalation contacts, and overrides"
        ),
    )

    # Custom settings
    custom_settings: dict[str, Any] = Field(
        default_factory=dict, description="Custom user-defined settings"
    )

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        """Return a JSON-schema-like description of the settings model."""

        return _build_model_json_schema(cls)

    @validator("plugins", pre=True)
    def _normalize_plugins(cls, value: Any) -> list[Any]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("Plugins must be valid JSON") from exc
            value = parsed
        if isinstance(value, dict):
            if all(str(key).isdigit() for key in value.keys()):
                return [
                    value[key] for key in sorted(value.keys(), key=lambda x: int(x))
                ]
            return [value]
        if isinstance(value, list):
            return value
        raise TypeError("Plugins configuration must be a list, mapping, or JSON string")

    @validator("policy", pre=True)
    def _normalise_policy(cls, value: Any) -> Any:
        if value in (None, "", {}):
            return {}
        if isinstance(value, PolicyDefinitions):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("Policy definitions must be valid JSON") from exc
        if isinstance(value, Mapping):
            return value
        raise TypeError(
            "policy must be provided as a mapping, JSON object, or PolicyDefinitions instance"
        )

    @classmethod
    def _collect_env_data(cls) -> tuple[dict[str, Any], set[str]]:
        """Return environment driven configuration data and the originating keys."""

        config = getattr(cls, "Config", None)
        prefix = getattr(config, "env_prefix", "") or ""
        delimiter = getattr(config, "env_nested_delimiter", "__")
        case_sensitive = getattr(config, "case_sensitive", False)
        compare_prefix = prefix if case_sensitive else prefix.lower()
        alias_mapping: dict[str, str] = {
            (alias if case_sensitive else alias.lower()): target
            for alias, target in getattr(config, "env_aliases", {}).items()
        }

        def _normalize(key: str) -> str:
            return key if case_sensitive else key.lower()

        def _split_path(path: str) -> list[str]:
            if delimiter and delimiter in path:
                parts = path.split(delimiter)
            elif "." in path:
                parts = path.split(".")
            else:
                parts = [path]
            return [part for part in parts if part]

        def _assign(data: dict[str, Any], keys: list[str], value: Any) -> None:
            current = data
            for part in keys[:-1]:
                part = _normalize(part)
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
            current[_normalize(keys[-1])] = value

        env_data: dict[str, Any] = {}
        consumed_aliases: set[str] = set()
        used_keys: set[str] = set()

        for env_key, env_value in os.environ.items():
            compare_key = env_key if case_sensitive else env_key.lower()

            if compare_key in alias_mapping:
                path = alias_mapping[compare_key]
                parts = _split_path(path)
                if parts:
                    _assign(env_data, parts, env_value)
                    used_keys.add(env_key)
                    consumed_aliases.add(compare_key)
                continue

            if prefix:
                if not compare_key.startswith(compare_prefix):
                    continue
                key_path = env_key[len(prefix) :]
            else:
                key_path = env_key

            normalized_path = _normalize(key_path)
            parts = _split_path(normalized_path)

            if not parts:
                continue

            _assign(env_data, parts, env_value)
            used_keys.add(env_key)

        # Include aliases that may not share the prefix when not already processed.
        for alias_key, path in alias_mapping.items():
            if alias_key in consumed_aliases:
                continue
            original_key = next(
                (key for key in os.environ.keys() if _normalize(key) == alias_key),
                None,
            )
            if original_key is None:
                continue
            parts = _split_path(path)
            if not parts:
                continue
            _assign(env_data, parts, os.environ[original_key])
            used_keys.add(original_key)

        return env_data, used_keys

    @classmethod
    def from_env(cls) -> "MemoriaSettings":
        """Create settings from environment variables"""

        env_data, _ = cls._collect_env_data()
        return cls(**env_data)

    @classmethod
    def from_env_with_metadata(
        cls,
    ) -> tuple["MemoriaSettings", set[str]]:
        """Return settings from the environment along with the keys that were used."""

        env_data, used_keys = cls._collect_env_data()
        return cls(**env_data), used_keys

    class Config:
        """Pydantic configuration"""

        env_prefix = "MEMORIA_"
        env_nested_delimiter = "__"
        case_sensitive = False
        env_aliases = {
            "MEMORIA_DB_URL": "database__connection_string",
            "MEMORIA_DATABASE_URL": "database__connection_string",
            "MEMORIA_DEFAULT_MODEL": "agents__default_model",
            "MEMORIA_DEFAULT_NAMESPACE": "memory__namespace",
            "MEMORIA_INGEST_MODE": "memory__ingest_mode",
            "MEMORIA_PERSONAL_DOCUMENTS_ENABLED": "memory__personal_documents_enabled",
        }

    @classmethod
    def from_file(cls, config_path: str | Path) -> "MemoriaSettings":
        """Load settings from JSON/YAML file"""
        import json
        from pathlib import Path

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            if config_path.suffix.lower() == ".json":
                data = json.load(f)
            elif config_path.suffix.lower() in [".yml", ".yaml"]:
                try:
                    import yaml

                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML configuration files")
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )

        return cls(**data)

    def to_file(self, config_path: str | Path, format: str = "json") -> None:
        """Save settings to file"""
        import json
        from pathlib import Path

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.dict()

        with open(config_path, "w") as f:
            if format.lower() == "json":
                json.dump(data, f, indent=2, default=str)
            elif format.lower() in ["yml", "yaml"]:
                try:
                    import yaml

                    yaml.safe_dump(data, f, default_flow_style=False)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML configuration files")
            else:
                raise ValueError(f"Unsupported format: {format}")

    def export(self, *, include_sensitive: bool = False) -> dict[str, Any]:
        """Return a serialisable representation of the settings."""

        try:
            data: dict[str, Any] = self.model_dump(mode="python")  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - pydantic v1 fallback
            data = self.dict()

        if include_sensitive:
            return data

        sync_settings = data.get("sync")
        if isinstance(sync_settings, dict):
            connection_value = sync_settings.get("connection_url")
            has_connection = bool(connection_value)
            if has_connection:
                sync_settings["connection_url"] = "***"
            sync_settings["has_connection"] = has_connection
        return data

    def get_database_url(self) -> str:
        """Get the database connection URL"""
        return self.database.connection_string

    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.debug and self.logging.level in [
            LogLevel.INFO,
            LogLevel.WARNING,
            LogLevel.ERROR,
        ]


def _build_model_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a JSON schema approximation for a Pydantic model."""

    return _describe_model(model, visited=set())


def _describe_model(
    model: type[BaseModel], *, visited: set[type[BaseModel]]
) -> dict[str, Any]:
    if model in visited:
        # Break potential recursion by returning an empty object schema
        return {"type": "object", "properties": {}}

    visited.add(model)

    schema: dict[str, Any] = {
        "title": getattr(model, "__name__", "Model"),
        "type": "object",
        "properties": {},
    }
    required_fields: list[str] = []

    for name, field in getattr(model, "__fields__", {}).items():
        field_schema = _describe_field(field, visited=visited)
        schema["properties"][name] = field_schema
        if field.required:
            required_fields.append(name)

    if required_fields:
        schema["required"] = required_fields

    visited.remove(model)
    return schema


def _describe_field(field, *, visited: set[type[BaseModel]]) -> dict[str, Any]:
    from pydantic.fields import Undefined

    schema = _schema_from_annotation(field.outer_type_, visited=visited)

    description = getattr(field.field_info, "description", None)
    if description:
        schema["description"] = description

    title = getattr(field.field_info, "title", None)
    if not title:
        title = _humanize_name(field.alias or field.name)
    schema.setdefault("title", title)

    default = field.default
    if default is not None and default is not Undefined:
        serialised = _serialise_default(default)
        if serialised is not _UNSET:
            schema["default"] = serialised
    elif default is Undefined and field.allow_none:
        schema.setdefault("default", None)

    return schema


def _schema_from_annotation(
    annotation: Any, *, visited: set[type[BaseModel]]
) -> dict[str, Any]:
    origin = get_origin(annotation)

    if origin is Union:
        args = tuple(get_args(annotation))
        has_none = any(arg is type(None) for arg in args)
        non_null = [arg for arg in args if arg is not type(None)]
        if len(non_null) == 1:
            schema = _schema_from_annotation(non_null[0], visited=visited)
            if has_none:
                schema = dict(schema)
                schema["nullable"] = True
            return schema
        schema = {"type": "string"}
        if has_none:
            schema["nullable"] = True
        return schema

    if origin in {list, list, Sequence, tuple, set}:
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        item_schema = _schema_from_annotation(item_type, visited=visited)
        return {"type": "array", "items": item_schema}

    if origin in {dict, Mapping, MutableMapping}:
        value_type = get_args(annotation)[1] if len(get_args(annotation)) > 1 else Any
        value_schema = _schema_from_annotation(value_type, visited=visited)
        return {"type": "object", "additionalProperties": value_schema}

    if origin is Literal:
        values = list(get_args(annotation))
        return {"type": "string", "enum": values}

    if inspect.isclass(annotation):
        if issubclass(annotation, BaseModel):
            nested = _describe_model(annotation, visited=visited)
            nested_schema: dict[str, Any] = {
                "type": "object",
                "properties": nested.get("properties", {}),
            }
            if nested.get("required"):
                nested_schema["required"] = nested["required"]
            return nested_schema
        if issubclass(annotation, Enum):
            values = [member.value for member in annotation]
            names = [member.name for member in annotation]
            schema = {"type": "string", "enum": values}
            if any(names):
                schema["enumNames"] = names
            return schema
        if issubclass(annotation, Path):
            return {"type": "string", "format": "path"}
        if issubclass(annotation, bool):
            return {"type": "boolean"}
        if issubclass(annotation, int):
            return {"type": "integer"}
        if issubclass(annotation, float):
            return {"type": "number"}
        if issubclass(annotation, str):
            return {"type": "string"}

    if annotation is Any:
        return {"type": "string"}

    return {"type": "string"}


def _humanize_name(raw: str) -> str:
    if not raw:
        return raw
    words = raw.replace("_", " ").split()
    return " ".join(word.capitalize() for word in words)


_UNSET = object()


def _serialise_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _UNSET
