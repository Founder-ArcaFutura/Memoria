"""Storage service handling persistence operations for Memoria."""

# ruff: noqa: UP006,UP007,UP045

from __future__ import annotations

import copy
import json
import math
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Tuple, Type, TypeAlias
from typing import cast as typing_cast

from loguru import logger
from pydantic import ValidationError
from sqlalchemy import (
    JSON,
    String,
    and_,
    bindparam,
    cast as sa_cast,
    func,
    literal,
    or_,
    select,
    text,
    union_all,
    DateTime,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Query, Session, selectinload
from sqlalchemy.sql import Select
from sqlalchemy.sql.selectable import Subquery
from sqlalchemy.inspection import inspect as sa_inspect

QueryType: TypeAlias = Query[Any]


from ..database.models import (
    Cluster,
    ClusterMember,
    LinkMemoryThread,
    LongTermMemory,
    RetentionPolicyAudit,
    ShortTermMemory,
    SpatialMetadata,
    ThreadEvent,
    ThreadMessageLink,
    Workspace,
    WorkspaceMember,
)
from ..heuristics.manual_promotion import StagedManualMemory
from ..heuristics.retention import RetentionPolicyRule
from ..policy import (
    EnforcementStage,
    PolicyAction,
    PolicyDecision,
    PolicyEnforcementEngine,
    PolicyViolationError,
    RedactionResult,
    apply_redactions,
)
from ..database.queries.memory_queries import MemoryQueries
from ..database.sqlalchemy_manager import SQLAlchemyDatabaseManager as DatabaseManager
from ..database.sqlite_utils import is_sqlite_json_error
from ..schemas import (
    PersonalMemoryDocument,
    PersonalMemoryEntry,
    canonicalize_symbolic_anchors,
)
from ..utils.exceptions import MemoriaError
from ..utils.pydantic_compat import model_dump, model_validate
from ..utils.pydantic_models import (
    MemoryClassification,
    MemoryImportanceLevel,
    ProcessedLongTermMemory,
)
from ..sync import SyncEvent, SyncEventAction
from .team_workspace import (
    TeamSpace,
    TeamSpaceCache,
    WorkspaceCache,
    WorkspaceContext,
    ensure_team_access,
    ensure_workspace_access,
    team_user_has_access,
)

if TYPE_CHECKING:
    from ..config.settings import SyncSettings


class StorageService:
    """Service layer encapsulating database access and search helpers."""

    AUTOGEN_USER_PLACEHOLDER = "[auto-generated user input]"
    AUTOGEN_AI_PLACEHOLDER = "[auto-generated ai output]"
    _TEAM_CACHE_SENTINEL = "<no-team>"
    _WORKSPACE_CACHE_SENTINEL = "<no-workspace>"

    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        namespace: str = "default",
        search_engine: Any | None = None,
        conscious_ingest: bool = False,
        sync_publisher: Callable[[SyncEventAction | str, str, str | None, dict[str, Any]], None] | None = None,
        team_id: str | None = None,
        workspace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        policy_engine: PolicyEnforcementEngine | None = None,

    ) -> None:
        self.db_manager = db_manager
        self.namespace = namespace
        self.search_engine = search_engine
        self.conscious_ingest = conscious_ingest
        self._sqlite_anchor_json_supported: bool | None = None
        self._sqlite_anchor_warning_logged = False
        self._sync_publisher = sync_publisher
        self._cache_lock = threading.RLock()
        self._memory_snapshot_cache: dict[str, dict[str, Any]] = {}
        self._thread_cache: dict[str, dict[str, Any]] = {}
        self._team_id: str | None = None
        self.team_id = team_id
        self._workspace_id: str | None = None
        self._active_workspace_context: WorkspaceContext | None = None
        self.workspace_id = workspace_id
        self._default_user_id = self._clean_identifier(user_id)
        self._default_agent_id = self._clean_identifier(agent_id)
        self._policy_engine = policy_engine or PolicyEnforcementEngine.get_global()
        self._sync_realtime = False
        self._sync_privacy_floor: float | None = None
        self._sync_privacy_ceiling: float | None = None
        self._personal_namespace = namespace or "default"
        self._team_namespace_prefix = "team"
        self._team_enforce_membership = True
        self._team_default_share = False
        self._team_cache = TeamSpaceCache()
        self._workspace_cache = WorkspaceCache()
        self._workspace_enforce_membership = True
        self._retention_policies: tuple[RetentionPolicyRule, ...] = ()
        self._agent_profiles: dict[str, dict[str, Any]] = {}

        self._refresh_agent_cache()

    @property
    def team_id(self) -> str | None:
        return self._team_id

    @team_id.setter
    def team_id(self, value: str | None) -> None:
        cleaned: str | None
        if isinstance(value, str):
            stripped = value.strip()
            cleaned = stripped or None
        else:
            cleaned = value

        previous = getattr(self, "_team_id", None)
        self._team_id = cleaned

        if previous == cleaned:
            return

        if hasattr(self, "_cache_lock"):
            with self._cache_lock:
                self._memory_snapshot_cache.clear()
                self._thread_cache.clear()

    @property
    def workspace_id(self) -> str | None:
        return self._workspace_id

    @workspace_id.setter
    def workspace_id(self, value: str | None) -> None:
        cleaned: str | None
        if isinstance(value, str):
            stripped = value.strip()
            cleaned = stripped or None
        else:
            cleaned = value

        previous = getattr(self, "_workspace_id", None)
        self._workspace_id = cleaned

        if previous == cleaned:
            return

        self._active_workspace_context = None
        if hasattr(self, "_cache_lock"):
            with self._cache_lock:
                self._memory_snapshot_cache.clear()
                self._thread_cache.clear()

    @property
    def policy_engine(self) -> PolicyEnforcementEngine:
        return self._policy_engine

    def enforce_policy(
        self,
        stage: EnforcementStage,
        payload: Mapping[str, Any] | None,
        *,
        allow_redaction: bool = False,
        on_redact: Callable[[PolicyDecision], None] | None = None,
    ) -> PolicyDecision:
        return self._run_policy_check(
            stage,
            payload,
            allow_redaction=allow_redaction,
            on_redact=on_redact,
        )

    def set_sync_publisher(
        self,
        publisher: Callable[[SyncEventAction | str, str, str | None, dict[str, Any]], None] | None,

    ) -> None:
        """Attach a callable that forwards events to the configured backend."""

        self._sync_publisher = publisher

    def configure_sync_policy(self, settings: "SyncSettings" | None) -> None:
        """Configure replication behaviour and privacy thresholds."""

        if settings is None:
            self._sync_realtime = False
            self._sync_privacy_floor = None
            self._sync_privacy_ceiling = None
            return

        self._sync_realtime = bool(getattr(settings, "realtime_replication", False))
        self._sync_privacy_floor = self._coerce_float(
            getattr(settings, "privacy_floor", None)
        )
        self._sync_privacy_ceiling = self._coerce_float(
            getattr(settings, "privacy_ceiling", None)
        )

    def configure_retention_policies(
        self, policies: Iterable[RetentionPolicyRule] | None
    ) -> None:
        """Store retention policy rules for local enforcement."""

        self._retention_policies = tuple(policies or ())

    @staticmethod
    def _clean_identifier(value: str | None) -> str | None:
        if value is None or not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned or None

    def _resolve_actor_identifier(self, user_id: str | None) -> str | None:
        identifier = self._clean_identifier(user_id)
        if identifier:
            return identifier
        return self._default_user_id or self._default_agent_id

    def _normalise_team_id(self, team_id: str) -> str:
        cleaned = self._clean_identifier(team_id)
        if not cleaned:
            raise MemoriaError("Team identifier must be a non-empty string")
        return cleaned

    def _collect_unique_identifiers(
        self, values: Iterable[str] | None, *, label: str
    ) -> set[str]:
        """Return cleaned identifiers ensuring no duplicates are supplied."""

        unique: set[str] = set()
        duplicates: set[str] = set()

        if values is None:
            return unique

        for raw in values:
            cleaned = self._clean_identifier(raw)
            if not cleaned:
                continue
            if cleaned in unique:
                duplicates.add(cleaned)
            else:
                unique.add(cleaned)

        if duplicates:
            formatted = ", ".join(sorted(duplicates))
            raise MemoriaError(f"Duplicate {label} identifiers: {formatted}")

        return unique

    def _default_policy_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {"namespace": self.namespace or "default"}
        if self.team_id:
            context["team_id"] = self.team_id
        if self.workspace_id:
            context["workspace_id"] = self.workspace_id
        if self._default_user_id:
            context["user_id"] = self._default_user_id
        elif self._default_agent_id:
            context["user_id"] = self._default_agent_id
        if self._default_agent_id:
            context["agent_id"] = self._default_agent_id
        return context

    # ------------------------------------------------------------------
    # Agent registry helpers
    # ------------------------------------------------------------------

    def _refresh_agent_cache(self) -> None:
        """Refresh the local cache of agent metadata."""

        try:
            records = self.db_manager.list_agents()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Unable to refresh agent cache: %s", exc)
            records = []

        with getattr(self, "_cache_lock", threading.RLock()):
            self._agent_profiles = {
                self._clean_identifier(record.get("agent_id")) or record.get("agent_id"): record
                for record in records
                if record.get("agent_id")
            }

    def _is_agent_identifier(self, identifier: str | None) -> bool:
        cleaned = self._clean_identifier(identifier)
        if not cleaned:
            return False
        return cleaned in self._agent_profiles

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        cleaned = self._clean_identifier(agent_id)
        if not cleaned:
            return None
        with self._cache_lock:
            profile = self._agent_profiles.get(cleaned)
        if profile is not None:
            return copy.deepcopy(profile)

        record = self.db_manager.get_agent(cleaned)
        if record is None:
            return None
        with self._cache_lock:
            self._agent_profiles[cleaned] = record
        return copy.deepcopy(record)

    def register_agent(
        self,
        agent_id: str,
        *,
        name: str | None = None,
        role: str | None = None,
        preferred_model: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = self.db_manager.register_agent(
            agent_id,
            name=name,
            role=role,
            preferred_model=preferred_model,
            metadata=metadata,
        )
        cleaned = self._clean_identifier(profile.get("agent_id"))
        if cleaned:
            with self._cache_lock:
                self._agent_profiles[cleaned] = profile
        return copy.deepcopy(profile)

    def list_agents(self) -> list[dict[str, Any]]:
        with self._cache_lock:
            if self._agent_profiles:
                return [copy.deepcopy(profile) for profile in self._agent_profiles.values()]

        self._refresh_agent_cache()
        with self._cache_lock:
            return [copy.deepcopy(profile) for profile in self._agent_profiles.values()]

    def _run_policy_check(
        self,
        stage: EnforcementStage,
        payload: Mapping[str, Any] | None,
        *,
        allow_redaction: bool,
        on_redact: Callable[[PolicyDecision], None] | None,
    ) -> PolicyDecision:
        if self._policy_engine is None:
            return PolicyDecision.allow()

        merged_payload: dict[str, Any] = self._default_policy_context()
        if payload:
            merged_payload.update(dict(payload))

        decision = self._policy_engine.evaluate(stage, merged_payload)

        if decision.action is PolicyAction.ALLOW:
            return decision

        if decision.action is PolicyAction.REDACT:
            if not allow_redaction:
                raise PolicyViolationError(decision, stage=stage)
            if on_redact is not None:
                on_redact(decision)
            return decision

        raise PolicyViolationError(decision, stage=stage)

    def _apply_collection_redactions(
        self,
        items: list[dict[str, Any]] | dict[str, Any] | None,
        decision: PolicyDecision,
    ) -> None:
        if decision.action is not PolicyAction.REDACT or not items:
            return

        if isinstance(items, list):
            for entry in items:
                if isinstance(entry, dict):
                    apply_redactions(entry, decision)
        elif isinstance(items, dict):
            apply_redactions(items, decision)
    def _personal_namespace_name(self) -> str:
        return self._personal_namespace or self.namespace or "default"

    def _apply_retention_policy_on_write(
        self,
        *,
        namespace: str,
        team_id: str | None,
        workspace_id: str | None,
        memory_id: str,
        proposed_privacy: float | None,
        proposed_importance: float | None,
        source: str,
    ) -> bool:
        if not self._retention_policies:
            return True

        resolved_namespace = namespace or self._personal_namespace_name()
        allowed = True

        for rule in self._retention_policies:
            if not rule.matches(resolved_namespace):
                continue

            privacy_violation = False
            importance_violation = False
            violations: list[str] = []

            if (
                proposed_privacy is not None
                and rule.privacy_ceiling is not None
                and proposed_privacy > rule.privacy_ceiling + 1e-6
            ):
                privacy_violation = True
                violations.append(
                    (
                        f"privacy {proposed_privacy:.3f} exceeds ceiling "
                        f"{rule.privacy_ceiling:.3f}"
                    )
                )

            if (
                proposed_importance is not None
                and rule.importance_floor is not None
                and proposed_importance < rule.importance_floor - 1e-6
            ):
                importance_violation = True
                violations.append(
                    (
                        f"importance {proposed_importance:.3f} below floor "
                        f"{rule.importance_floor:.3f}"
                    )
                )

            if not violations:
                continue

            details = {
                "source": source,
                "violations": violations,
                "proposed_privacy": proposed_privacy,
                "proposed_importance": proposed_importance,
                "team_id": team_id,
                "workspace_id": workspace_id,
            }

            self.record_retention_audit(
                memory_id=memory_id,
                namespace=resolved_namespace,
                policy_name=rule.name,
                action=rule.action,
                details=details,
                escalate_to=rule.escalate_to,
                team_id=team_id,
                workspace_id=workspace_id,
            )

            if rule.action == "block":
                allowed = False

        return allowed

    def record_retention_audit(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        memory_id: str | None = None,
        namespace: str | None = None,
        policy_name: str | None = None,
        action: str | None = None,
        details: Mapping[str, Any] | None = None,
        escalate_to: str | None = None,
        team_id: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        """Persist an audit trail entry for retention policy enforcement."""

        resolved_details: dict[str, Any] = dict(details or {})

        if isinstance(payload, Mapping):
            memory_id = payload.get("memory_id", memory_id)
            namespace = payload.get("namespace", namespace)
            policy_name = payload.get("policy_name", policy_name)
            action = payload.get("action", action)
            escalate_to = payload.get("escalate_to", escalate_to)
            team_id = payload.get("team_id", team_id)
            workspace_id = payload.get("workspace_id", workspace_id)

            for key in ("violations", "importance", "privacy", "age_days", "metadata"):
                if key in payload and payload[key] is not None and key not in resolved_details:
                    resolved_details[key] = payload[key]

        cleaned_namespace = self._clean_identifier(namespace)
        if not cleaned_namespace:
            cleaned_namespace = self.namespace or "default"

        audit_entry = RetentionPolicyAudit(
            memory_id=memory_id,
            namespace=cleaned_namespace,
            policy_name=policy_name or "unknown",  # pragma: no cover - defensive default
            action=action or "log",
            escalate_to=escalate_to,
            details=resolved_details,
            team_id=self._clean_identifier(team_id),
            workspace_id=self._clean_identifier(workspace_id),
        )

        try:
            with self.db_manager.SessionLocal() as session:
                session.add(audit_entry)
                session.commit()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to record retention audit for {memory_id}: {exc}")

    def configure_team_policy(
        self,
        *,
        namespace_prefix: str | None = None,
        enforce_membership: bool | None = None,
        share_by_default: bool | None = None,
    ) -> None:
        """Adjust defaults that govern team namespace resolution."""

        with self._cache_lock:
            if namespace_prefix is not None:
                cleaned = self._clean_identifier(namespace_prefix) or "team"
                self._team_namespace_prefix = cleaned
            if enforce_membership is not None:
                self._team_enforce_membership = bool(enforce_membership)
            if share_by_default is not None:
                self._team_default_share = bool(share_by_default)

    def register_team_space(
        self,
        team_id: str,
        *,
        namespace: str | None = None,
        display_name: str | None = None,
        members: Iterable[str] | None = None,
        admins: Iterable[str] | None = None,
        agents: Iterable[str] | None = None,
        share_by_default: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TeamSpace:
        """Create or replace a collaborative namespace definition."""

        normalized_id = self._normalise_team_id(team_id)
        resolved_namespace = (
            self._clean_identifier(namespace)
            or f"{self._team_namespace_prefix}:{normalized_id}"
        )
        member_set = self._collect_unique_identifiers(
            members, label="workspace member"
        )
        admin_set = self._collect_unique_identifiers(
            admins, label="workspace admin"
        )
        agent_set = self._collect_unique_identifiers(
            agents, label="agent identifier"
        )
        for candidate in member_set | admin_set:
            if self._is_agent_identifier(candidate):
                agent_set.add(candidate)
        default_share = (
            bool(share_by_default)
            if share_by_default is not None
            else self._team_default_share
        )

        agent_profiles: dict[str, dict[str, Any]] = {}
        for agent_id in agent_set:
            profile = self.get_agent(agent_id)
            if profile:
                agent_profiles[agent_id] = profile

        team = TeamSpace(
            team_id=normalized_id,
            namespace=resolved_namespace,
            display_name=self._clean_identifier(display_name),
            share_by_default=default_share,
            metadata=dict(metadata or {}),
            members=member_set,
            admins=admin_set,
            agent_members=agent_set,
            agent_profiles=agent_profiles,
        )

        with self._cache_lock:
            return self._team_cache.set(team)

    def get_team_space(self, team_id: str) -> TeamSpace | None:
        normalized_id = self._normalise_team_id(team_id)
        with self._cache_lock:
            return self._team_cache.get(normalized_id)

    def list_team_spaces(self) -> list[TeamSpace]:
        with self._cache_lock:
            return self._team_cache.list()

    def set_team_members(
        self,
        team_id: str,
        *,
        members: Iterable[str] | None = None,
        admins: Iterable[str] | None = None,
        agents: Iterable[str] | None = None,
    ) -> TeamSpace:
        normalized_id = self._normalise_team_id(team_id)
        member_set = (
            self._collect_unique_identifiers(members, label="workspace member")
            if members is not None
            else None
        )
        admin_set = (
            self._collect_unique_identifiers(admins, label="workspace admin")
            if admins is not None
            else None
        )
        agent_set = (
            self._collect_unique_identifiers(agents, label="agent identifier")
            if agents is not None
            else None
        )
        agent_profiles: dict[str, dict[str, Any]] | None = None
        if agent_set is not None:
            agent_profiles = {}
            for agent_id in agent_set:
                profile = self.get_agent(agent_id)
                if profile:
                    agent_profiles[agent_id] = profile
        with self._cache_lock:
            return self._team_cache.update_members(
                normalized_id,
                members=member_set,
                admins=admin_set,
                agents=agent_set,
                agent_profiles=agent_profiles,
            )

    def add_team_members(
        self,
        team_id: str,
        members: Iterable[str],
        *,
        as_admin: bool = False,
    ) -> TeamSpace:
        normalized_id = self._normalise_team_id(team_id)
        additions = self._collect_unique_identifiers(
            members,
            label="workspace admin" if as_admin else "workspace member",
        )
        with self._cache_lock:
            return self._team_cache.add_members(
                normalized_id, additions, as_admin=as_admin
            )

    def add_team_agents(
        self, team_id: str, agents: Iterable[str]
    ) -> TeamSpace:
        normalized_id = self._normalise_team_id(team_id)
        agent_set = self._collect_unique_identifiers(
            agents, label="agent identifier"
        )
        profiles = {
            agent_id: profile
            for agent_id in agent_set
            if (profile := self.get_agent(agent_id)) is not None
        }
        with self._cache_lock:
            return self._team_cache.add_agents(
                normalized_id, agent_set, profiles=profiles
            )

    def remove_team_member(self, team_id: str, user_id: str) -> TeamSpace:
        normalized_id = self._normalise_team_id(team_id)
        target = self._clean_identifier(user_id)
        if not target:
            raise MemoriaError("User identifier must be provided")
        with self._cache_lock:
            return self._team_cache.remove_member(normalized_id, target)

    def user_has_team_access(self, team_id: str, user_id: str | None) -> bool:
        normalized_id = self._normalise_team_id(team_id)
        with self._cache_lock:
            return team_user_has_access(
                self._team_cache,
                normalized_id,
                self._resolve_actor_identifier(user_id),
            )

    def require_team_access(self, team_id: str, user_id: str | None) -> TeamSpace:
        normalized_id = self._normalise_team_id(team_id)
        with self._cache_lock:
            return ensure_team_access(
                self._team_cache,
                normalized_id,
                self._resolve_actor_identifier(user_id),
                enforce_membership=self._team_enforce_membership,
            )

    def get_accessible_namespaces(self, user_id: str | None) -> set[str]:
        identifier = self._resolve_actor_identifier(user_id)
        with self._cache_lock:
            namespaces: set[str] = {self._personal_namespace_name()}
            if identifier is None:
                return namespaces
            for team_id in self._team_cache.team_ids_for_user(identifier):
                space = self._team_cache.get(team_id)
                if space is not None:
                    namespaces.add(space.namespace)
            return namespaces

    def configure_workspace_policy(
        self, *, enforce_membership: bool | None = None
    ) -> None:
        """Adjust workspace access policy controls."""

        if enforce_membership is not None:
            self._workspace_enforce_membership = bool(enforce_membership)

    def _ensure_service_metadata_table(self, connection: Any) -> None:
        """Ensure the lightweight ``service_metadata`` table exists."""

        try:
            connection.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS service_metadata "
                    "(key VARCHAR(255) PRIMARY KEY, value TEXT)"
                )
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Unable to ensure service_metadata table: %s", exc)

    def set_service_metadata_value(self, key: str, value: str | None) -> None:
        """Persist a key/value pair for background job bookkeeping."""

        if not isinstance(key, str) or not key.strip():
            raise ValueError("Metadata key must be a non-empty string")

        engine = getattr(self.db_manager, "engine", None)
        if engine is None:  # pragma: no cover - defensive guard
            raise MemoriaError("Database engine is not available")

        normalised_value = "" if value is None else str(value)
        payload = {"key": key.strip(), "value": normalised_value}

        with engine.begin() as connection:
            self._ensure_service_metadata_table(connection)
            update_stmt = text(
                "UPDATE service_metadata SET value = :value WHERE key = :key"
            )
            result = connection.execute(update_stmt, payload)
            if result.rowcount and result.rowcount > 0:
                return
            insert_stmt = text(
                "INSERT INTO service_metadata (key, value) VALUES (:key, :value)"
            )
            connection.execute(insert_stmt, payload)

    def get_service_metadata_value(self, key: str) -> str | None:
        """Return a previously stored ``service_metadata`` value when available."""

        if not isinstance(key, str) or not key.strip():
            raise ValueError("Metadata key must be a non-empty string")

        engine = getattr(self.db_manager, "engine", None)
        if engine is None:  # pragma: no cover - defensive guard
            return None

        with engine.begin() as connection:
            self._ensure_service_metadata_table(connection)
            select_stmt = text(
                "SELECT value FROM service_metadata WHERE key = :key"
            )
            result = connection.execute(select_stmt, {"key": key.strip()})
            row = result.first()
            return None if row is None else row[0]

    def _load_workspace_context(self, workspace_id: str) -> WorkspaceContext | None:
        with self.db_manager.SessionLocal() as session:
            workspace = (
                session.query(Workspace)
                .options(selectinload(Workspace.members))
                .filter(Workspace.workspace_id == workspace_id)
                .one_or_none()
            )
            if workspace is None:
                return None

            members: set[str] = set()
            admins: set[str] = set()
            agents: set[str] = set()
            agent_profiles: dict[str, dict[str, Any]] = {}
            for membership in workspace.members or []:
                identifier = self._clean_identifier(getattr(membership, "user_id", None))
                if not identifier:
                    continue
                is_agent_member = getattr(membership, "is_agent", False)
                if getattr(membership, "is_admin", False):
                    admins.add(identifier)
                if is_agent_member:
                    agents.add(identifier)
                    profile = self.get_agent(identifier)
                    if profile:
                        agent_profiles[identifier] = profile
                else:
                    members.add(identifier)

            metadata: dict[str, Any] = {}
            owner_id = getattr(workspace, "owner_id", None)
            if owner_id:
                metadata["owner_id"] = owner_id

            return WorkspaceContext(
                workspace_id=workspace.workspace_id,
                team_id=getattr(workspace, "team_id", None),
                name=getattr(workspace, "name", None),
                slug=getattr(workspace, "slug", None),
                description=getattr(workspace, "description", None),
                owner_id=owner_id,
                metadata=metadata,
                members=members,
                admins=admins,
                agents=agents,
                agent_profiles=agent_profiles,
            )

    def get_workspace_context(self, workspace_id: str) -> WorkspaceContext | None:
        normalized = self._clean_identifier(workspace_id)
        if not normalized:
            return None
        with self._cache_lock:
            cached = self._workspace_cache.get(normalized)
            if cached is not None:
                return cached

        loaded = self._load_workspace_context(normalized)
        if loaded is None:
            return None
        with self._cache_lock:
            return self._workspace_cache.set(loaded)

    def require_workspace_access(
        self, workspace_id: str, user_id: str | None = None
    ) -> WorkspaceContext:
        normalized = self._clean_identifier(workspace_id)
        if not normalized:
            raise MemoriaError("Workspace identifier must be provided")

        identifier = self._resolve_actor_identifier(user_id)
        context = self.get_workspace_context(normalized)
        if context is None:
            raise MemoriaError(f"Unknown workspace: {workspace_id}")

        with self._cache_lock:
            context = ensure_workspace_access(
                self._workspace_cache,
                normalized,
                identifier,
                enforce_membership=self._workspace_enforce_membership,
            )

        self._active_workspace_context = copy.deepcopy(context)
        return copy.deepcopy(context)

    def _resolve_workspace_context(
        self,
        *,
        workspace_id: str | None = None,
        user_id: str | None = None,
        enforce: bool | None = None,
    ) -> WorkspaceContext | None:
        target = self._clean_identifier(workspace_id) or self.workspace_id
        if not target:
            self._active_workspace_context = None
            return None

        identifier = self._clean_identifier(user_id) or self._default_user_id
        if enforce is None:
            enforce = self._workspace_enforce_membership

        if enforce:
            context = self.require_workspace_access(target, identifier)
        else:
            context = self.get_workspace_context(target)
            if context is None:
                raise MemoriaError(f"Unknown workspace: {target}")
        self._active_workspace_context = copy.deepcopy(context)
        return copy.deepcopy(context)

    def resolve_target_namespace(
        self,
        *,
        namespace: str | None = None,
        team_id: str | None = None,
        user_id: str | None = None,
        share_with_team: bool | None = None,
    ) -> str:
        explicit = self._clean_identifier(namespace)
        if explicit:
            return explicit

        if not team_id:
            return self._personal_namespace_name()

        normalized_id = self._normalise_team_id(team_id)
        with self._cache_lock:
            space = self._team_cache.get(normalized_id)
            if space is None:
                raise MemoriaError(f"Unknown team: {team_id}")
            share_flag = (
                space.share_by_default if share_with_team is None else bool(share_with_team)
            )
            if not share_flag:
                return self._personal_namespace_name()
            if self._team_enforce_membership and not space.is_member(
                self._clean_identifier(user_id)
            ):
                raise MemoriaError(
                    f"User '{user_id}' is not permitted to access team '{team_id}'"
                )
            return space.namespace

    def _emit_sync_event(
        self,
        action: SyncEventAction | str,
        entity_type: str,
        entity_id: str | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._sync_publisher is None:
            return
        if not self._should_publish_event(entity_type, payload):
            return
        try:
            prepared_payload: dict[str, Any]
            if payload is None:
                prepared_payload = {}
            elif isinstance(payload, dict):
                prepared_payload = copy.deepcopy(payload)
            else:
                try:
                    prepared_payload = dict(payload)
                except Exception:
                    prepared_payload = {}

            team_identifier = self._clean_identifier(self.team_id)
            if team_identifier and not prepared_payload.get("team_id"):
                prepared_payload["team_id"] = team_identifier
            workspace_identifier = self._clean_identifier(self.workspace_id)
            if workspace_identifier and not prepared_payload.get("workspace_id"):
                prepared_payload["workspace_id"] = workspace_identifier
                context = self.get_workspace_context(workspace_identifier)
                if context is not None:
                    prepared_payload.setdefault("workspace", context.to_metadata())
            if not prepared_payload.get("namespace"):
                prepared_payload.setdefault("namespace", self.namespace)
            policy_payload = {
                "action": getattr(action, "value", action),
                "entity_type": entity_type,
                "entity_id": entity_id,
                "payload": copy.deepcopy(prepared_payload),
            }

            def _apply_redaction(decision: PolicyDecision) -> None:
                apply_redactions(prepared_payload, decision)

            self.enforce_policy(
                EnforcementStage.SYNC,
                policy_payload,
                allow_redaction=True,
                on_redact=_apply_redaction,
            )

            self._sync_publisher(action, entity_type, entity_id, prepared_payload)
        except Exception:  # pragma: no cover - defensive logging
            logger.opt(exception=True).warning("Failed to publish sync event")

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            if isinstance(value, bool):
                return float(int(value))
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_privacy_from_payload(
        self, payload: dict[str, Any] | None
    ) -> float | None:
        if not isinstance(payload, dict):
            return None

        explicit = self._coerce_float(payload.get("privacy"))
        if explicit is not None:
            return explicit

        snapshot = payload.get("snapshot")
        if isinstance(snapshot, dict):
            value = self._coerce_float(snapshot.get("y_coord"))
            if value is not None:
                return value

        replica = payload.get("replica")
        if isinstance(replica, dict):
            record = replica.get("record")
            if isinstance(record, dict):
                value = self._coerce_float(record.get("y_coord"))
                if value is not None:
                    return value

        changes = payload.get("changes")
        if isinstance(changes, dict):
            value = self._coerce_float(changes.get("y_coord"))
            if value is not None:
                return value

        return None

    def _extract_team_identifier_from_mapping(
        self, payload: Mapping[str, Any] | None
    ) -> str | None:
        if not isinstance(payload, Mapping):
            return None
        team_value = payload.get("team_id")
        if isinstance(team_value, str):
            return self._clean_identifier(team_value)
        if team_value is not None:
            return self._clean_identifier(str(team_value))
        return None

    def _resolve_team_from_payload(
        self, payload: Mapping[str, Any] | None
    ) -> str | None:
        if not isinstance(payload, Mapping):
            return None

        team_identifier = self._extract_team_identifier_from_mapping(payload)
        if team_identifier:
            return team_identifier

        for key in ("snapshot", "replica", "changes"):
            nested_payload = payload.get(key)
            team_identifier = self._extract_team_identifier_from_mapping(
                nested_payload if isinstance(nested_payload, Mapping) else None
            )
            if team_identifier:
                return team_identifier

        return None

    def _extract_workspace_identifier_from_mapping(
        self, payload: Mapping[str, Any] | None
    ) -> str | None:
        if not isinstance(payload, Mapping):
            return None
        workspace_value = payload.get("workspace_id")
        if isinstance(workspace_value, str):
            return self._clean_identifier(workspace_value)
        if workspace_value is not None:
            return self._clean_identifier(str(workspace_value))
        return None

    def _resolve_workspace_from_payload(
        self, payload: Mapping[str, Any] | None
    ) -> str | None:
        if not isinstance(payload, Mapping):
            return None

        workspace_identifier = self._extract_workspace_identifier_from_mapping(payload)
        if workspace_identifier:
            return workspace_identifier

        for key in ("snapshot", "replica", "changes"):
            nested_payload = payload.get(key)
            workspace_identifier = self._extract_workspace_identifier_from_mapping(
                nested_payload if isinstance(nested_payload, Mapping) else None
            )
            if workspace_identifier:
                return workspace_identifier

        return None

    def _derive_privacy_axis(
        self,
        *,
        snapshot: dict[str, Any] | None = None,
        replica: dict[str, Any] | None = None,
        changes: dict[str, Any] | None = None,
        explicit: float | None = None,
    ) -> float | None:
        if explicit is not None:
            parsed = self._coerce_float(explicit)
            if parsed is not None:
                return parsed

        for source in (changes, snapshot):
            if isinstance(source, dict):
                parsed = self._coerce_float(source.get("y_coord"))
                if parsed is not None:
                    return parsed

        if isinstance(replica, dict):
            record = replica.get("record")
            if isinstance(record, dict):
                parsed = self._coerce_float(record.get("y_coord"))
                if parsed is not None:
                    return parsed

        return None

    def _should_publish_event(
        self, entity_type: str, payload: dict[str, Any] | None
    ) -> bool:
        if entity_type != "memory":
            return True

        floor = self._sync_privacy_floor
        ceiling = self._sync_privacy_ceiling
        if floor is None and ceiling is None:
            return True

        privacy_value = self._extract_privacy_from_payload(payload)
        if privacy_value is None:
            return True

        if floor is not None and privacy_value < floor:
            return False
        if ceiling is not None and privacy_value > ceiling:
            return False
        return True

    def _build_memory_event_payload(
        self,
        *,
        memory_id: str,
        snapshot: dict[str, Any] | None,
        replica: dict[str, Any] | None,
        changes: dict[str, Any] | None = None,
        explicit_privacy: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if snapshot:
            payload["snapshot"] = copy.deepcopy(snapshot)
        if replica:
            payload["replica"] = copy.deepcopy(replica)
        if changes:
            payload["changes"] = copy.deepcopy(changes)
        privacy = self._derive_privacy_axis(
            snapshot=snapshot,
            replica=replica,
            changes=changes,
            explicit=explicit_privacy,
        )
        if privacy is not None:
            payload["privacy"] = privacy
        if payload:
            payload.setdefault("memory_id", memory_id)
        else:
            payload = {"memory_id": memory_id}

        team_identifier = self._resolve_team_from_payload(payload)
        if not team_identifier:
            for source in (snapshot, replica, changes):
                if isinstance(source, Mapping):
                    team_identifier = self._extract_team_identifier_from_mapping(source)
                else:
                    team_identifier = None
                if team_identifier:
                    break
        if not team_identifier:
            team_identifier = self._clean_identifier(self.team_id)
        if team_identifier:
            payload["team_id"] = team_identifier

        workspace_identifier = self._resolve_workspace_from_payload(payload)
        if not workspace_identifier:
            for source in (snapshot, replica, changes):
                if isinstance(source, Mapping):
                    workspace_identifier = self._extract_workspace_identifier_from_mapping(
                        source
                    )
                else:
                    workspace_identifier = None
                if workspace_identifier:
                    break
        if not workspace_identifier:
            workspace_identifier = self._clean_identifier(self.workspace_id)
        if workspace_identifier:
            payload["workspace_id"] = workspace_identifier
            context = self.get_workspace_context(workspace_identifier)
            if context is not None:
                payload.setdefault("workspace", context.to_metadata())

        namespace_value: str | None
        existing_namespace = payload.get("namespace")
        if isinstance(existing_namespace, str):
            namespace_value = self._clean_identifier(existing_namespace)
        elif existing_namespace is not None:
            namespace_value = self._clean_identifier(str(existing_namespace))
        else:
            namespace_value = None
        if not namespace_value:
            for source in (snapshot, replica, changes):
                if isinstance(source, Mapping):
                    candidate = source.get("namespace")
                    if isinstance(candidate, str):
                        namespace_value = self._clean_identifier(candidate)
                    elif candidate is not None:
                        namespace_value = self._clean_identifier(str(candidate))
                    else:
                        namespace_value = None
                else:
                    namespace_value = None
                if namespace_value:
                    break
        if not namespace_value:
            namespace_value = self._clean_identifier(self.namespace)
        if namespace_value:
            payload["namespace"] = namespace_value
        elif "namespace" not in payload:
            payload["namespace"] = self.namespace
        return payload

    def _build_replication_payload(self, memory_id: str) -> dict[str, Any] | None:
        try:
            with self.db_manager.SessionLocal() as session:
                record, table = self._resolve_memory_record(session, memory_id)
                if record is None or table is None:
                    return None
                return {
                    "table": table,
                    "record": self._serialize_sa_record(record),
                }
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to serialise memory %s for replication", memory_id
            )
            return None

    @staticmethod
    def _serialize_sa_record(record: Any) -> dict[str, Any]:
        mapper = sa_inspect(record.__class__)
        data: dict[str, Any] = {}
        for column in mapper.columns:
            value = getattr(record, column.key)
            data[column.key] = StorageService._serialise_value(value)
        return data

    @staticmethod
    def _serialise_value(value: Any) -> Any:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", "ignore")
        if isinstance(value, (list, dict)):
            return copy.deepcopy(value)
        return value

    @staticmethod
    def _convert_datetime_for_storage(value: Any) -> tuple[datetime | None, bool]:
        if value is None:
            return None, True
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                value = value.astimezone(timezone.utc)
            return value.replace(tzinfo=None), True
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return None, False
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc)
            return parsed.replace(tzinfo=None), True
        return None, False

    def _normalise_replica_for_model(
        self, model: type, data: dict[str, Any]
    ) -> dict[str, Any]:
        mapper = sa_inspect(model)
        normalised: dict[str, Any] = {}
        for column in mapper.columns:
            key = column.key
            if key not in data:
                continue
            value = data[key]
            if isinstance(column.type, DateTime):
                converted, ok = self._convert_datetime_for_storage(value)
                if not ok:
                    continue
                normalised[key] = converted
            else:
                normalised[key] = copy.deepcopy(value)
        return normalised

    def _apply_memory_replica(self, replica: dict[str, Any]) -> None:
        if not isinstance(replica, dict):
            return
        record = replica.get("record")
        if not isinstance(record, dict):
            return
        table = replica.get("table") or record.get("table")
        model: type
        if table == "short_term":
            if not self.db_manager.enable_short_term:
                return
            model = ShortTermMemory
        else:
            model = LongTermMemory
            table = "long_term"

        normalised = self._normalise_replica_for_model(model, record)
        memory_id = normalised.get("memory_id")
        if not memory_id:
            return
        normalised["namespace"] = self.namespace
        if self.workspace_id is not None:
            normalised["workspace_id"] = self.workspace_id

        with self.db_manager.SessionLocal() as session:
            try:
                existing = (
                    session.query(model)
                    .filter(
                        model.memory_id == memory_id,  # type: ignore[attr-defined]
                        model.namespace == self.namespace,  # type: ignore[attr-defined]
                    )
                    .one_or_none()
                )
                if existing is None:
                    session.add(model(**normalised))
                else:
                    for key, value in normalised.items():
                        setattr(existing, key, value)
                session.commit()
            except Exception:
                session.rollback()
                logger.opt(exception=True).warning(
                    "Failed to apply replicated memory payload for %s", memory_id
                )

    def _replicate_memory_delete(self, memory_id: str) -> None:
        with self.db_manager.SessionLocal() as session:
            try:
                if self.db_manager.enable_short_term:
                    delete_query = session.query(ShortTermMemory).filter(
                        ShortTermMemory.memory_id == memory_id,
                        ShortTermMemory.namespace == self.namespace,
                    )
                    if self.workspace_id is not None:
                        delete_query = delete_query.filter(
                            ShortTermMemory.workspace_id == self.workspace_id
                        )
                    delete_query.delete()
                delete_long_query = session.query(LongTermMemory).filter(
                    LongTermMemory.memory_id == memory_id,
                    LongTermMemory.namespace == self.namespace,
                )
                if self.workspace_id is not None:
                    delete_long_query = delete_long_query.filter(
                        LongTermMemory.workspace_id == self.workspace_id
                    )
                delete_long_query.delete()
                session.commit()
            except Exception:
                session.rollback()
                logger.opt(exception=True).warning(
                    "Failed to apply replicated delete for memory %s", memory_id
                )

    def _cache_key(self, entity_id: str) -> str:
        team_component = self.team_id or self._TEAM_CACHE_SENTINEL
        workspace_component = self.workspace_id or self._WORKSPACE_CACHE_SENTINEL
        namespace_component = self.namespace or self._personal_namespace_name()
        return f"{workspace_component}::{team_component}::{namespace_component}::{entity_id}"

    def _set_memory_cache(self, memory_id: str, snapshot: dict[str, Any] | None) -> None:
        with self._cache_lock:
            cache_key = self._cache_key(memory_id)
            if snapshot is None:
                self._memory_snapshot_cache.pop(cache_key, None)
                return
            self._memory_snapshot_cache[cache_key] = copy.deepcopy(snapshot)

    def _get_cached_memory(self, memory_id: str) -> dict[str, Any] | None:
        with self._cache_lock:
            cache_key = self._cache_key(memory_id)
            cached = self._memory_snapshot_cache.get(cache_key)
        if cached is None:
            return None
        return copy.deepcopy(cached)

    def _remove_memory_cache(self, memory_id: str) -> None:
        self._set_memory_cache(memory_id, None)

    def _set_thread_cache(self, thread_id: str, payload: dict[str, Any] | None) -> None:
        with self._cache_lock:
            cache_key = self._cache_key(thread_id)
            if payload is None:
                self._thread_cache.pop(cache_key, None)
                return
            self._thread_cache[cache_key] = copy.deepcopy(payload)

    def _get_cached_thread(self, thread_id: str) -> dict[str, Any] | None:
        with self._cache_lock:
            cache_key = self._cache_key(thread_id)
            cached = self._thread_cache.get(cache_key)
        if cached is None:
            return None
        return copy.deepcopy(cached)

    def _remove_thread_cache(self, thread_id: str) -> None:
        self._set_thread_cache(thread_id, None)

    def _normalise_documents_payload(self, value: Any) -> list[dict[str, Any]] | None:
        """Convert arbitrary document payloads into serialisable mappings."""

        if value is None:
            return None

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                return None

        if isinstance(value, PersonalMemoryDocument):
            return [model_dump(value, mode="python")]

        items: Sequence[Any]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items = value
        else:
            items = [value]

        documents: list[dict[str, Any]] = []

        for item in items:
            if item in (None, "", {}):
                continue
            if isinstance(item, str):
                stripped_item = item.strip()
                if not stripped_item:
                    continue
                try:
                    parsed = json.loads(stripped_item)
                except json.JSONDecodeError:
                    continue
                item = parsed

            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                nested = self._normalise_documents_payload(item)
                if nested:
                    documents.extend(nested)
                continue

            if isinstance(item, PersonalMemoryDocument):
                document_model = item
            elif isinstance(item, Mapping):
                try:
                    document_model = model_validate(PersonalMemoryDocument, item)
                except Exception:
                    try:
                        document_model = PersonalMemoryDocument(**dict(item))
                    except Exception:
                        continue
            else:
                try:
                    document_model = model_validate(PersonalMemoryDocument, item)
                except Exception:
                    continue

            documents.append(model_dump(document_model, mode="python"))

        return documents or None

    def _build_memory_snapshot(self, record: Any, table: str | None) -> dict[str, Any]:
        processed = getattr(record, "processed_data", {}) or {}
        if isinstance(processed, str):
            try:
                loaded = json.loads(processed)
                if isinstance(loaded, dict):
                    processed = loaded
                else:
                    processed = {}
            except Exception:
                processed = {}
        text_value = processed.get("text") or processed.get("content")
        tokens_value = processed.get("tokens")
        intensity = processed.get("emotional_intensity")
        timestamp = getattr(record, "timestamp", None)
        timestamp_iso: str | None
        if isinstance(timestamp, datetime):
            timestamp_iso = timestamp.isoformat()
        else:
            timestamp_iso = None
        anchors = getattr(record, "symbolic_anchors", None) or []
        snapshot = {
            "memory_id": getattr(record, "memory_id", None),
            "text": text_value,
            "tokens": tokens_value,
            "emotional_intensity": intensity,
            "timestamp": timestamp_iso,
            "x_coord": getattr(record, "x_coord", None),
            "y_coord": getattr(record, "y_coord", None),
            "z_coord": getattr(record, "z_coord", None),
            "symbolic_anchors": list(anchors) if isinstance(anchors, list) else anchors,
            "namespace": getattr(record, "namespace", None),
            "team_id": getattr(record, "team_id", None),
            "workspace_id": getattr(record, "workspace_id", None),
            "table": table or (
                "short_term" if isinstance(record, ShortTermMemory) else "long_term"
            ),
        }
        documents = self._normalise_documents_payload(
            getattr(record, "documents_json", None)
        )
        if not documents and isinstance(processed, Mapping):
            documents = self._normalise_documents_payload(processed.get("documents"))
        if documents:
            snapshot["documents"] = documents
        return snapshot

    def get_memory_snapshot(
        self, memory_id: str, *, refresh: bool = False
    ) -> dict[str, Any] | None:
        """Return a cached representation of a memory, refreshing from storage if required."""

        if not memory_id:
            return None
        if not refresh:
            cached = self._get_cached_memory(memory_id)
            if cached is not None:
                return cached

        with self.db_manager.SessionLocal() as session:
            record, table = self._resolve_memory_record(session, memory_id)
            if record is None:
                self._remove_memory_cache(memory_id)
                return None
            snapshot = self._build_memory_snapshot(record, table)

        self._set_memory_cache(memory_id, snapshot)
        return copy.deepcopy(snapshot)

    def apply_sync_event(self, event: SyncEvent) -> None:
        """Update local caches when a remote change is observed."""

        if event.namespace and event.namespace != self.namespace:
            return

        local_team = self._clean_identifier(self.team_id)
        event_team = self._resolve_team_from_payload(event.payload)
        if local_team and event_team and event_team != local_team:
            return

        local_workspace = self._clean_identifier(self.workspace_id)
        event_workspace = self._resolve_workspace_from_payload(event.payload)
        if local_workspace and event_workspace and event_workspace != local_workspace:
            return

        if event.entity_type == "memory":
            memory_id = event.entity_id
            if not memory_id:
                return
            raw_payload = event.payload if isinstance(event.payload, dict) else {}
            if isinstance(raw_payload, dict) and any(
                key in raw_payload for key in ("snapshot", "replica", "changes", "privacy")
            ):
                snapshot_payload = raw_payload.get("snapshot")
                if not isinstance(snapshot_payload, dict):
                    snapshot_payload = None
                replica_payload = raw_payload.get("replica")
                if not isinstance(replica_payload, dict):
                    replica_payload = None
            else:
                snapshot_payload = raw_payload if isinstance(raw_payload, dict) else None
                replica_payload = None

            if event.action == SyncEventAction.MEMORY_DELETED.value:
                if self._sync_realtime:
                    self._replicate_memory_delete(memory_id)
                self._remove_memory_cache(memory_id)
            else:
                if self._sync_realtime and replica_payload:
                    self._apply_memory_replica(replica_payload)

                if snapshot_payload is not None:
                    self._set_memory_cache(memory_id, snapshot_payload)
                else:
                    self.get_memory_snapshot(memory_id, refresh=True)
        elif event.entity_type == "thread":
            thread_id = event.entity_id
            if not thread_id:
                return
            self.get_thread(thread_id, use_cache=False)
    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def _build_spatial_union(
        self, session: Session, namespace: str | None = None
    ) -> Subquery[Any]:
        """Return a combined selectable containing all spatial sources."""

        target_namespace = (namespace or self.namespace) or "default"
        queries: list[Select[Any]] = []
        team_id = self.team_id
        if self.db_manager.enable_short_term:

            short_term_conditions = [
                ShortTermMemory.namespace == target_namespace
            ]
            if self.team_id is not None:
                short_term_conditions.append(ShortTermMemory.team_id == self.team_id)
            if self.workspace_id is not None:
                short_term_conditions.append(
                    ShortTermMemory.workspace_id == self.workspace_id
                )

            queries.append(
                select(
                    ShortTermMemory.memory_id.label("memory_id"),
                    ShortTermMemory.processed_data.label("processed_data"),
                    ShortTermMemory.x_coord.label("x_coord"),
                    ShortTermMemory.y_coord.label("y_coord"),
                    ShortTermMemory.z_coord.label("z_coord"),
                    ShortTermMemory.symbolic_anchors.label("symbolic_anchors"),
                    literal("short_term").label("source_table"),
                ).where(and_(*short_term_conditions))
            )

        long_term_conditions = [LongTermMemory.namespace == target_namespace]
        if self.team_id is not None:
            long_term_conditions.append(LongTermMemory.team_id == self.team_id)
        if self.workspace_id is not None:
            long_term_conditions.append(LongTermMemory.workspace_id == self.workspace_id)

        queries.append(
            select(
                LongTermMemory.memory_id.label("memory_id"),
                LongTermMemory.processed_data.label("processed_data"),
                LongTermMemory.x_coord.label("x_coord"),
                LongTermMemory.y_coord.label("y_coord"),
                LongTermMemory.z_coord.label("z_coord"),
                LongTermMemory.symbolic_anchors.label("symbolic_anchors"),
                literal("long_term").label("source_table"),
            ).where(and_(*long_term_conditions))
        )

        spatial_conditions = [SpatialMetadata.namespace == target_namespace]
        if self.team_id is not None:
            spatial_conditions.append(SpatialMetadata.team_id == self.team_id)
        if self.workspace_id is not None:
            spatial_conditions.append(
                SpatialMetadata.workspace_id == self.workspace_id
            )

        queries.append(
            select(
                SpatialMetadata.memory_id.label("memory_id"),
                sa_cast(literal("{}"), JSON).label("processed_data"),
                SpatialMetadata.x.label("x_coord"),
                SpatialMetadata.y.label("y_coord"),
                SpatialMetadata.z.label("z_coord"),
                SpatialMetadata.symbolic_anchors.label("symbolic_anchors"),
                literal("spatial_metadata").label("source_table"),
            ).where(and_(*spatial_conditions))
        )


        return union_all(*queries).subquery()

    def _build_anchor_union(
        self,
        session: Session,
        *,
        exclude_ids: Sequence[str] | None = None,
    ) -> Subquery[Any]:
        """Return a selectable of memory IDs and anchors for recurrence checks."""

        exclusions = {value for value in (exclude_ids or []) if value}
        exclusion_list = list(exclusions)
        team_id = self.team_id
        queries: list[Select[Any]] = []
        if self.db_manager.enable_short_term:
            short_term_conditions = [ShortTermMemory.namespace == self.namespace]
            if self.team_id is not None:
                short_term_conditions.append(ShortTermMemory.team_id == self.team_id)
            if self.workspace_id is not None:
                short_term_conditions.append(
                    ShortTermMemory.workspace_id == self.workspace_id
                )

            short_term_query = (
                select(
                    ShortTermMemory.memory_id.label("memory_id"),
                    ShortTermMemory.symbolic_anchors.label("symbolic_anchors"),
                )
                .where(and_(*short_term_conditions))
            )
            if team_id:
                short_term_query = short_term_query.where(
                    ShortTermMemory.team_id == team_id
                )
            if exclusions:
                short_term_query = short_term_query.where(
                    ~ShortTermMemory.memory_id.in_(exclusion_list)
                )
            queries.append(short_term_query)

        long_term_conditions = [LongTermMemory.namespace == self.namespace]
        if self.team_id is not None:
            long_term_conditions.append(LongTermMemory.team_id == self.team_id)
        if self.workspace_id is not None:
            long_term_conditions.append(LongTermMemory.workspace_id == self.workspace_id)

        long_term_query = (
            select(
                LongTermMemory.memory_id.label("memory_id"),
                LongTermMemory.symbolic_anchors.label("symbolic_anchors"),
            )
            .where(and_(*long_term_conditions))
        )
        if team_id:
            long_term_query = long_term_query.where(LongTermMemory.team_id == team_id)
        if exclusions:
            long_term_query = long_term_query.where(
                ~LongTermMemory.memory_id.in_(exclusion_list)
            )
        queries.append(long_term_query)

        return union_all(*queries).subquery()

    def _normalize_anchor_filter(self, anchor: str | list[str] | None) -> list[str] | None:
        if anchor is None:
            return None
        if isinstance(anchor, str):
            return [anchor]
        return list(anchor)

    def _apply_anchor_filter(
        self,
        query: QueryType,
        combined: Subquery[Any],
        anchor_list: list[str] | None,
    ) -> tuple[QueryType, Callable[[], QueryType] | None]:

        if not anchor_list:
            return query, None

        if self.db_manager.database_type == "sqlite":
            normalized = list(anchor_list)
            base_query = query
            conditions = [
                combined.c.symbolic_anchors.contains(bindparam(f"anchor_{idx}", anchor))
                for idx, anchor in enumerate(normalized)
            ]
            fallback_conditions = [
                func.instr(
                func.lower(sa_cast(combined.c.symbolic_anchors, String)),
                    bindparam(
                        f"fallback_anchor_{idx}", f'"{anchor.lower()}"'
                    ),
                )
                > literal(0)
                for idx, anchor in enumerate(normalized)
            ]

            def _fallback(
                base_query: QueryType = base_query,
                fallback_conditions: Iterable[Any] = fallback_conditions,
            ) -> QueryType:
                return base_query.filter(or_(*fallback_conditions))

            return base_query.filter(or_(*conditions)), _fallback


        if self.db_manager.database_type == "postgresql":
            return (
                query.filter(
                    func.jsonb_exists_any(
                        sa_cast(combined.c.symbolic_anchors, JSONB),
                        bindparam("anchors", anchor_list, type_=ARRAY(String)),
                    )
                ),
                None,
            )

        if self.db_manager.database_type == "mysql":
            conditions = [
                func.json_contains(
                    sa_cast(combined.c.symbolic_anchors, JSON),
                    func.json_quote(bindparam(f"anchor_{idx}", anchor)),
                )
                for idx, anchor in enumerate(anchor_list)
            ]
            return query.filter(or_(*conditions)), None

        array_overlap_dialects = {"cockroachdb"}
        if self.db_manager.database_type not in array_overlap_dialects:
            raise MemoriaError(
                "Symbolic anchor filtering is not implemented for "
                f"database type '{self.db_manager.database_type}'"
            )

        return (
            query.filter(
                sa_cast(combined.c.symbolic_anchors, ARRAY(String)).overlap(
                    bindparam("anchors", anchor_list, type_=ARRAY(String))
                )
            ),
            None,
        )

    def compute_cluster_gravity(
        self,
        *,
        x_coord: float | None,
        y_coord: float | None,
        z_coord: float | None,
        anchors: Sequence[str] | None = None,
    ) -> float:
        """Calculate a gravitational pull score from cluster proximity."""

        if x_coord is None and y_coord is None and z_coord is None:
            return 0.0

        normalized = canonicalize_symbolic_anchors(anchors) or []
        anchor_terms = [anchor.lower() for anchor in normalized if anchor]

        with self.db_manager.SessionLocal() as session:
            base_query = session.query(
                Cluster.id,
                Cluster.weight,
                Cluster.centroid,
                Cluster.y_centroid,
                Cluster.z_centroid,
            ).filter(Cluster.weight.isnot(None))

            if anchor_terms:
                base_query = (
                    base_query.join(ClusterMember)
                    .filter(func.lower(ClusterMember.anchor).in_(anchor_terms))
                    .distinct()
                )

            clusters = (
                base_query.order_by(Cluster.weight.desc()).limit(50).all()
            )

            if not clusters:
                clusters = (
                    session.query(
                        Cluster.id,
                        Cluster.weight,
                        Cluster.centroid,
                        Cluster.y_centroid,
                        Cluster.z_centroid,
                    )
                    .filter(Cluster.weight.isnot(None))
                    .order_by(Cluster.weight.desc())
                    .limit(25)
                    .all()
                )

        total_influence = 0.0
        for cluster in clusters:
            weight = float(cluster.weight or 0.0)
            if weight <= 0.0:
                continue

            centroid = cluster.centroid or {}
            centroid_x = None
            centroid_y = None
            centroid_z = None

            if isinstance(centroid, dict):
                centroid_x = centroid.get("x")
                centroid_y = centroid.get("y")
                centroid_z = centroid.get("z")
            elif isinstance(centroid, (list, tuple)):
                if len(centroid) >= 1:
                    centroid_x = centroid[0]
                if len(centroid) >= 2:
                    centroid_y = centroid[1]
                if len(centroid) >= 3:
                    centroid_z = centroid[2]

            if cluster.y_centroid is not None:
                centroid_y = cluster.y_centroid
            if cluster.z_centroid is not None:
                centroid_z = cluster.z_centroid

            deltas: list[float] = []
            if x_coord is not None and centroid_x is not None:
                deltas.append((float(x_coord) - float(centroid_x)) ** 2)
            if y_coord is not None and centroid_y is not None:
                deltas.append((float(y_coord) - float(centroid_y)) ** 2)
            if z_coord is not None and centroid_z is not None:
                deltas.append((float(z_coord) - float(centroid_z)) ** 2)

            if not deltas:
                continue

            distance = math.sqrt(sum(deltas))
            influence = weight * math.exp(-distance)
            total_influence += influence

        if total_influence <= 0.0:
            return 0.0

        score = 1.0 - math.exp(-total_influence)
        return max(0.0, min(1.0, score))

    def count_anchor_occurrences(
        self,
        anchors: Sequence[str],
        *,
        exclude_memory_ids: Sequence[str] | None = None,
    ) -> dict[str, int]:
        """Count how often symbolic anchors appear across stored memories."""

        normalized = canonicalize_symbolic_anchors(anchors) or []
        ordered: list[str] = []
        for anchor in normalized:
            if anchor and anchor not in ordered:
                ordered.append(anchor)

        if not ordered:
            return {}

        exclusions = [value for value in (exclude_memory_ids or []) if value]
        with self.db_manager.SessionLocal() as session:
            dataset = self._build_anchor_union(session, exclude_ids=exclusions)
            results: dict[str, int] = {}

            for anchor in ordered:
                query = session.query(func.count()).select_from(dataset)
                filtered_query, fallback = self._apply_anchor_filter(
                    query, dataset, [anchor]
                )

                try:
                    count = filtered_query.scalar() or 0
                except OperationalError as exc:
                    if (
                        self.db_manager.database_type == "sqlite"
                        and fallback is not None
                        and is_sqlite_json_error(exc)
                    ):
                        session.rollback()
                        fallback_query = fallback()
                        count = fallback_query.scalar() or 0
                    else:
                        raise

                results[anchor] = int(count)

        return results

    def _log_sql_statement(self, session: Session, query: Any) -> None:
        try:
            sql_statement = str(
                query.statement.compile(
                    session.get_bind(),
                    compile_kwargs={"literal_binds": True},
                )
            )
            logger.debug(f"Generated SQL: {sql_statement}")
        except Exception as e:  # pragma: no cover - logging debug info only
            logger.debug(f"Could not compile SQL for logging: {e}")

    def _record_memory_touches(
        self, results: list[dict[str, Any]], *, event_source: str
    ) -> None:
        """Update access counters and log retrieval touches."""

        if not results:
            return

        payload = []
        for row in results:
            memory_id = row.get("memory_id")
            if not memory_id:
                continue

            table_hint = (
                row.get("table")
                or row.get("source_table")
                or row.get("memory_type")
            )
            if table_hint not in {"short_term", "long_term"}:
                continue

            payload.append(
                {
                    "memory_id": memory_id,
                    "table": table_hint,
                    "event_source": event_source,
                }
            )

        if not payload:
            return

        try:
            self.db_manager.record_memory_touches(
                self.namespace,
                payload,
                team_id=self.team_id,
                workspace_id=self.workspace_id,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(f"Failed to record memory touches: {exc}")

    def _format_spatial_results(
        self, rows: Iterable[Tuple[Any, ...]]

    ) -> list[dict[str, Any]]:
        def _parse_payload(raw_payload: Any) -> dict[str, Any]:
            if isinstance(raw_payload, dict):
                return raw_payload
            if isinstance(raw_payload, str):
                try:
                    parsed_payload = json.loads(raw_payload)
                except Exception:
                    parsed_payload = None
                if isinstance(parsed_payload, dict):
                    return parsed_payload
            return {}

        def _normalize_anchors(raw_anchors: Any) -> list[Any]:
            anchors_value = raw_anchors or []
            if isinstance(anchors_value, str):
                try:
                    parsed_anchors = json.loads(anchors_value)
                except Exception:
                    parsed_anchors = None
                if isinstance(parsed_anchors, list):
                    return parsed_anchors
                return [anchor.strip() for anchor in anchors_value.split(",") if anchor.strip()]
            if isinstance(anchors_value, (list, tuple)):
                return list(anchors_value)
            return []

        def _richness_score(entry: dict[str, Any]) -> tuple[int, int, int, int]:
            text_value = entry.get("text")
            source_table = entry.get("table")
            has_text = int(bool(text_value))
            is_primary_table = int(source_table in {"short_term", "long_term"})
            is_short_term = int(source_table == "short_term")
            text_length = len(text_value) if isinstance(text_value, str) else 0
            return has_text, is_primary_table, is_short_term, text_length

        best_rows: dict[str, dict[str, Any]] = {}
        best_scores: dict[str, tuple[int, int, int, int]] = {}
        order: list[str] = []
        metadata_rows: dict[str, list[dict[str, Any]]] = {}

        for mem_id, data, x_c, y_c, z_c, row_anchors, dist, source_table in rows:
            if mem_id is None:
                continue

            data_dict = _parse_payload(data)
            anchors = _normalize_anchors(row_anchors)
            emotion = data_dict.get("emotional_intensity")
            text_value = (
                data_dict.get("text")
                or data_dict.get("content")
                or data_dict.get("summary")
            )

            entry = {
                "memory_id": mem_id,
                "text": text_value,
                "tokens": data_dict.get("tokens"),
                "x": x_c,
                "y": y_c,
                "z": z_c,
                "symbolic_anchors": anchors,
                "emotional_intensity": emotion,
                "distance": dist,
                "table": source_table,
                "memory_type": source_table
                if source_table in {"short_term", "long_term"}
                else None,
            }

            if source_table == "spatial_metadata":
                metadata_rows.setdefault(mem_id, []).append(entry)

            if mem_id not in order:
                order.append(mem_id)

            score = _richness_score(entry)
            if mem_id not in best_rows or score > best_scores[mem_id]:
                best_rows[mem_id] = entry
                best_scores[mem_id] = score

        output: list[dict[str, Any]] = []
        for mem_id in order:
            entry = best_rows.get(mem_id)
            if not entry:
                continue

            if metadata_rows.get(mem_id):
                for metadata_entry in metadata_rows[mem_id]:
                    if entry.get("x") is None:
                        entry["x"] = metadata_entry.get("x")
                    if entry.get("y") is None:
                        entry["y"] = metadata_entry.get("y")
                    if entry.get("z") is None:
                        entry["z"] = metadata_entry.get("z")
                    anchors_value = entry.get("symbolic_anchors")
                    if not anchors_value:
                        entry["symbolic_anchors"] = metadata_entry.get("symbolic_anchors") or []

            output.append(entry)

        return output

    def _execute_spatial_query(
        self,
        session: Session,
        combined: Any,
        axes: Sequence[tuple[str, float]],
        max_distance: float,
        anchor_list: list[str] | None,
        limit: int,
    ) -> list[Tuple[Any, ...]]:

        distance_components = [
            func.pow(getattr(combined.c, axis) - value, 2) for axis, value in axes
        ]
        if not distance_components:
            raise ValueError("At least one axis must be provided for spatial query")

        distance_expr = distance_components[0]
        for component in distance_components[1:]:
            distance_expr = distance_expr + component
        distance_expr = func.sqrt(distance_expr)
        distance_col = distance_expr.label("distance")

        query = session.query(
            combined.c.memory_id,
            combined.c.processed_data,
            combined.c.x_coord,
            combined.c.y_coord,
            combined.c.z_coord,
            combined.c.symbolic_anchors,
            distance_col,
            combined.c.source_table,
        )

        filters = [getattr(combined.c, axis).isnot(None) for axis, _ in axes]
        query = query.filter(*filters, distance_expr <= max_distance)
        query, sqlite_fallback = self._apply_anchor_filter(
            query, combined, anchor_list
        )

        self._log_sql_statement(session, query)

        try:
            result = query.order_by(distance_col).limit(limit).all()
            return typing_cast(list[Tuple[Any, ...]], result)

        except OperationalError as exc:
            if (
                self.db_manager.database_type == "sqlite"
                and sqlite_fallback is not None
                and is_sqlite_json_error(exc)
            ):
                session.rollback()
                logger.warning(
                    "SQLite JSON1 functions unavailable; falling back to substring anchor search (anchors=%s): %s",
                    anchor_list,
                    exc,
                )
                fallback_query = sqlite_fallback()
                self._log_sql_statement(session, fallback_query)
                fallback_result = (
                    fallback_query.order_by(distance_col).limit(limit).all()
                )
                return typing_cast(list[Tuple[Any, ...]], fallback_result)

            raise

    def retrieve_context(
        self,
        query: str,
        limit: int = 5,
        *,
        namespace: str | None = None,
        team_id: str | None = None,
        user_id: str | None = None,
        share_with_team: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant context for a query with priority on essential facts."""
        workspace_context = self._resolve_workspace_context(user_id=user_id)
        effective_workspace_id = (
            workspace_context.workspace_id
            if workspace_context is not None
            else self.workspace_id
        )
        workspace_team_id = (
            self._clean_identifier(workspace_context.team_id)
            if workspace_context is not None and workspace_context.team_id
            else None
        )

        target_namespace = self.resolve_target_namespace(
            namespace=namespace,
            team_id=team_id,
            user_id=user_id,
            share_with_team=share_with_team,
        )

        policy_payload = {
            "query": query,
            "limit": limit,
            "namespace": target_namespace,
            "team_id": team_id or workspace_team_id or self.team_id,
            "workspace_id": effective_workspace_id,
            "user_id": self._clean_identifier(user_id) or self._default_user_id,
            "share_with_team": share_with_team,
        }

        decision = self.enforce_policy(
            EnforcementStage.RETRIEVAL,
            policy_payload,
            allow_redaction=True,
        )

        try:
            context_items: list[dict[str, Any]] = []

            essential_conversations: list[dict[str, Any]] = []
            if self.conscious_ingest:
                essentials_limit = min(limit, 3)
                try:
                    essential_conversations = self.get_essential_conversations(
                        limit=essentials_limit, namespace=target_namespace
                    )
                except TypeError as exc:
                    if "namespace" in str(exc):
                        essential_conversations = self.get_essential_conversations(
                            essentials_limit
                        )
                    else:
                        raise
                context_items.extend(essential_conversations)
                remaining_limit = max(0, limit - len(essential_conversations))
            else:
                remaining_limit = limit

            specific_context: list[dict[str, Any]] | dict[str, Any] = []
            if remaining_limit > 0:
                if self.search_engine:
                    specific_context = self.search_engine.execute_search(
                        query=query,
                        db_manager=self.db_manager,
                        namespace=target_namespace,
                        limit=remaining_limit,
                    )
                    if isinstance(specific_context, dict):
                        specific_context = specific_context.get("results", [])
                else:
                    specific_context = self.db_manager.search_memories(
                        query=query,
                        namespace=target_namespace,
                        limit=remaining_limit,
                        team_id=(
                            team_id
                            if team_id is not None
                            else (workspace_team_id or self.team_id)
                        ),
                        workspace_id=effective_workspace_id,

                    )

                if isinstance(specific_context, dict):
                    specific_context = specific_context.get("results", [])

                for item in specific_context or []:
                    if not any(
                        ctx.get("memory_id") == item.get("memory_id")
                        for ctx in context_items
                    ):
                        context_items.append(item)

            final_context_items = context_items[:limit]

            logger.debug(
                f"Retrieved {len(final_context_items)} context items for query: {query} "
                f"(Essential conversations: {len(essential_conversations) if self.conscious_ingest else 0})"
            )
            self._record_memory_touches(
                final_context_items, event_source="context_query"
            )
            self._apply_collection_redactions(final_context_items, decision)
            return final_context_items
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []

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
    ) -> list[dict[str, Any]]:
        """Retrieve memories near a given spatial coordinate within a threshold distance."""
        target_namespace = (namespace or self.namespace) or "default"
        policy_payload = {
            "namespace": target_namespace,
            "x": x,
            "y": y,
            "z": z,
            "max_distance": max_distance,
            "limit": limit,
            "anchor_filter": anchor,
            "dimensions": dimensions,
        }
        decision = self.enforce_policy(
            EnforcementStage.RETRIEVAL,
            policy_payload,
            allow_redaction=True,
        )
        try:
            with self.db_manager.SessionLocal() as session:
                combined = self._build_spatial_union(session, target_namespace)
                anchor_list = self._normalize_anchor_filter(anchor)

                logger.debug(
                    "retrieve_memories_near inputs: x=%s, y=%s, z=%s, max_distance=%s, anchor_list=%s",
                    x,
                    y,
                    z,
                    max_distance,
                    anchor_list,
                )

                dimension_mode = (dimensions or "3d").strip().lower()
                if dimension_mode == "2d":
                    axes = [("y_coord", y), ("z_coord", z)]
                else:
                    if dimension_mode != "3d":
                        logger.debug(
                            "Unknown dimension mode '%s'; defaulting to 3D distance calculation",
                            dimension_mode,
                        )
                    axes = [("x_coord", x), ("y_coord", y), ("z_coord", z)]

                results = self._execute_spatial_query(
                    session,
                    combined,
                    axes,
                    max_distance,
                    anchor_list,
                    limit,
                )

                logger.debug(f"Spatial query returned {len(results)} rows")

                formatted = self._format_spatial_results(results)
                event_source = "spatial_2d" if dimension_mode == "2d" else "spatial_3d"
                self._record_memory_touches(formatted, event_source=event_source)
                self._apply_collection_redactions(formatted, decision)
                return formatted
        except Exception as e:
            logger.error(f"Spatial retrieval failed: {e}")
            return []

    def retrieve_memories_near_2d(
        self,
        y: float,
        z: float,
        max_distance: float = 5.0,
        anchor: str | list[str] | None = None,
        limit: int = 10,
        *,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories near a point on the Y/Z plane within a threshold distance."""
        target_namespace = (namespace or self.namespace) or "default"
        policy_payload = {
            "namespace": target_namespace,
            "y": y,
            "z": z,
            "max_distance": max_distance,
            "limit": limit,
            "anchor_filter": anchor,
            "dimensions": "2d",
        }
        decision = self.enforce_policy(
            EnforcementStage.RETRIEVAL,
            policy_payload,
            allow_redaction=True,
        )

        try:
            with self.db_manager.SessionLocal() as session:
                combined = self._build_spatial_union(session, target_namespace)
                anchor_list = self._normalize_anchor_filter(anchor)

                logger.debug(
                    "retrieve_memories_near_2d inputs: y=%s, z=%s, max_distance=%s, anchor_list=%s",

                    y,
                    z,
                    max_distance,
                    anchor_list,
                )

                results = self._execute_spatial_query(
                    session,
                    combined,
                    [("y_coord", y), ("z_coord", z)],
                    max_distance,
                    anchor_list,
                    limit,
                )

                logger.debug(f"Spatial 2D query returned {len(results)} rows")

                formatted = self._format_spatial_results(results)
                self._record_memory_touches(formatted, event_source="spatial_2d")
                self._apply_collection_redactions(formatted, decision)
                return formatted
        except Exception as e:
            logger.error(f"Spatial retrieval (2D) failed: {e}")
            return []

    def retrieve_memories_by_anchor(
        self, anchors: list[str], *, namespace: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve memories that match symbolic anchors."""
        search_anchors = canonicalize_symbolic_anchors(anchors) or []
        if not search_anchors:
            return []

        # If namespace is None, search is cross-namespace. Otherwise, filter by it.
        target_namespace = namespace

        policy_payload = {
            "namespace": target_namespace or "*",
            "anchors": search_anchors,
            "limit": None,
        }
        decision = self.enforce_policy(
            EnforcementStage.RETRIEVAL,
            policy_payload,
            allow_redaction=True,
        )

        try:
            with self.db_manager.SessionLocal() as session:
                queries = []
                # Query ShortTermMemory if enabled
                if self.db_manager.enable_short_term:
                    stm_query = session.query(
                        ShortTermMemory.memory_id.label("memory_id"),
                        ShortTermMemory.processed_data.label("processed_data"),
                        ShortTermMemory.summary.label("summary"),
                        ShortTermMemory.x_coord.label("x_coord"),
                        ShortTermMemory.y_coord.label("y_coord"),
                        ShortTermMemory.z_coord.label("z_coord"),
                        ShortTermMemory.symbolic_anchors.label("symbolic_anchors"),
                        literal("short_term").label("source_table"),
                    )
                    if target_namespace:
                        stm_query = stm_query.filter(ShortTermMemory.namespace == target_namespace)
                    queries.append(stm_query)

                # Query LongTermMemory
                ltm_query = session.query(
                    LongTermMemory.memory_id.label("memory_id"),
                    LongTermMemory.processed_data.label("processed_data"),
                    LongTermMemory.summary.label("summary"),
                    LongTermMemory.x_coord.label("x_coord"),
                    LongTermMemory.y_coord.label("y_coord"),
                    LongTermMemory.z_coord.label("z_coord"),
                    LongTermMemory.symbolic_anchors.label("symbolic_anchors"),
                    literal("long_term").label("source_table"),
                )
                if target_namespace:
                    ltm_query = ltm_query.filter(LongTermMemory.namespace == target_namespace)
                queries.append(ltm_query)

                combined = union_all(*queries).subquery()

                base_query = session.query(
                    combined.c.memory_id,
                    combined.c.processed_data,
                    combined.c.summary,
                    combined.c.x_coord,
                    combined.c.y_coord,
                    combined.c.z_coord,
                    combined.c.symbolic_anchors,
                    combined.c.source_table,
                )

                filtered_query, fallback = self._apply_anchor_filter(
                    base_query, combined, search_anchors
                )

                try:
                    results = filtered_query.all()
                except OperationalError as exc:
                    if (
                        self.db_manager.database_type == "sqlite"
                        and fallback is not None
                        and is_sqlite_json_error(exc)
                    ):
                        session.rollback()
                        logger.warning(
                            "SQLite JSON1 functions unavailable; falling back to substring anchor search (anchors=%s): %s",
                            search_anchors,
                            exc,
                        )
                        results = fallback().all()
                    else:
                        raise

                output: list[dict[str, Any]] = []
                for (
                    memory_id,
                    data,
                    summary,
                    x_c,
                    y_c,
                    z_c,
                    row_anchors,
                    source_table,
                ) in results:
                    normalized: dict[str, Any] = {}
                    text_value: str | None = None

                    if isinstance(data, dict):
                        normalized = data
                    elif isinstance(data, str):
                        try:
                            parsed = json.loads(data)
                            if isinstance(parsed, dict):
                                normalized = parsed
                            else:
                                text_value = summary
                        except Exception:
                            text_value = summary
                    else:
                        normalized = {}

                    if text_value is None:
                        text_value = (
                            normalized.get("text")
                            or normalized.get("content")
                            or summary
                        )

                    emotion = normalized.get("emotional_intensity")
                    row_anchor_values = canonicalize_symbolic_anchors(row_anchors) or []

                    output.append(
                        {
                            "memory_id": memory_id,
                            "text": text_value,
                            "summary": summary,
                            "x": x_c,
                            "y": y_c,
                            "z": z_c,
                            "symbolic_anchors": row_anchor_values,
                            "emotional_intensity": emotion,
                            "table": source_table,
                            "memory_type": source_table
                            if source_table in {"short_term", "long_term"}
                            else None,
                        }
                    )
                self._record_memory_touches(output, event_source="anchor_retrieval")
                self._apply_collection_redactions(output, decision)
                return output
        except Exception as e:
            logger.error(
                f"Anchor retrieval failed for anchors={search_anchors}: {e}"
            )
            return []

    def find_related_memory_ids_by_anchors(
        self,
        anchors: Sequence[str],
        *,
        exclude_memory_ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Return IDs for memories sharing the provided symbolic anchors."""

        normalized = canonicalize_symbolic_anchors(anchors) or []
        ordered: list[str] = []
        for anchor in normalized:
            if anchor and anchor not in ordered:
                ordered.append(anchor)

        if not ordered:
            return []

        exclusions = [value for value in (exclude_memory_ids or []) if value]

        with self.db_manager.SessionLocal() as session:
            dataset = self._build_anchor_union(session, exclude_ids=exclusions)
            base_query = session.query(dataset.c.memory_id).distinct()
            filtered_query, fallback = self._apply_anchor_filter(
                base_query, dataset, ordered
            )

            try:
                rows = filtered_query.all()
            except OperationalError as exc:
                if (
                    self.db_manager.database_type == "sqlite"
                    and fallback is not None
                    and is_sqlite_json_error(exc)
                ):
                    session.rollback()
                    rows = fallback().all()
                else:
                    raise

        return [row.memory_id for row in rows if getattr(row, "memory_id", None)]

    def get_related_memories(
        self,
        memory_id: str,
        *,
        limit: int = 5,
        relation_types: Sequence[str] | None = None,
        privacy_floor: float | None = -10.0,
    ) -> list[dict[str, Any]]:
        """Return related memories for ``memory_id`` using the relationship graph."""

        if not memory_id or not getattr(self, "db_manager", None):
            return []

        getter = getattr(self.db_manager, "get_related_memories", None)
        if not callable(getter):
            return []

        return getter(
            memory_id=memory_id,
            namespace=self.namespace,
            limit=limit,
            relation_types=relation_types,
            privacy_floor=privacy_floor,
        )

    def get_relationship_candidates(
        self,
        *,
        symbolic_anchors: Sequence[str] | None = None,
        keywords: Sequence[str] | None = None,
        topic: str | None = None,
        exclude_ids: Sequence[str] | None = None,
        limit: int = 5,
        privacy_floor: float | None = -10.0,
    ) -> list[dict[str, Any]]:
        """Suggest related memories based on anchors and keywords."""

        if not getattr(self, "db_manager", None):
            return []

        finder = getattr(self.db_manager, "find_related_memory_candidates", None)
        if not callable(finder):
            return []

        return finder(
            namespace=self.namespace,
            symbolic_anchors=symbolic_anchors,
            keywords=keywords,
            topic=topic,
            exclude_ids=exclude_ids,
            limit=limit,
            privacy_floor=privacy_floor,
        )

    def retrieve_memories_by_time_range(
        self,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
        start_x: float | None = None,
        end_x: float | None = None,
        *,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories within a time or x-coordinate range."""
        target_namespace = (namespace or self.namespace) or "default"
        policy_payload = {
            "namespace": target_namespace,
            "start_timestamp": start_timestamp.isoformat() if start_timestamp else None,
            "end_timestamp": end_timestamp.isoformat() if end_timestamp else None,
            "start_x": start_x,
            "end_x": end_x,
        }
        decision = self.enforce_policy(
            EnforcementStage.RETRIEVAL,
            policy_payload,
            allow_redaction=True,
        )
        try:
            results = []
            with self.db_manager.SessionLocal() as session:
                models = [LongTermMemory]
                if self.db_manager.enable_short_term:
                    models.insert(0, ShortTermMemory)
                for model in models:
                    query = session.query(model).filter(
                        model.namespace == target_namespace
                    )
                    if start_timestamp:
                        query = query.filter(model.created_at >= start_timestamp)
                    if end_timestamp:
                        query = query.filter(model.created_at <= end_timestamp)
                    if start_x is not None:
                        query = query.filter(
                            model.x_coord.isnot(None), model.x_coord >= start_x
                        )
                    if end_x is not None:
                        query = query.filter(
                            model.x_coord.isnot(None), model.x_coord <= end_x
                        )
                    query = query.order_by(model.created_at)
                    results.extend(query.all())
            formatted: list[dict[str, Any]] = []
            for mem in results:
                processed = mem.processed_data or {}
                if isinstance(processed, str):
                    try:
                        processed = json.loads(processed)
                    except Exception:
                        processed = {}
                formatted.append(
                    {
                        "memory_id": mem.memory_id,
                        "text": processed.get("text")
                        or processed.get("content"),
                        "created_at": mem.created_at.isoformat(),
                        "x": mem.x_coord,
                        "y": mem.y_coord,
                        "z": mem.z_coord,
                        "symbolic_anchors": mem.symbolic_anchors or [],
                        "emotional_intensity": processed.get("emotional_intensity"),
                        "table": "short_term"
                        if isinstance(mem, ShortTermMemory)
                        else "long_term",
                        "memory_type": "short_term"
                        if isinstance(mem, ShortTermMemory)
                        else "long_term",
                    }
                )
            self._record_memory_touches(formatted, event_source="temporal_range")
            self._apply_collection_redactions(formatted, decision)
            return formatted
        except Exception as e:
            logger.error(f"Time-range retrieval failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------
    def update_memory(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update stored memory fields if the record exists."""

        allowed_fields = {
            "anchor",
            "text",
            "tokens",
            "timestamp",
            "x_coord",
            "y_coord",
            "z_coord",
            "symbolic_anchors",
            "emotional_intensity",
        }
        filtered_updates = {
            key: value for key, value in (updates or {}).items() if key in allowed_fields
        }

        if not filtered_updates:
            return False

        serialised_changes = {
            key: self._serialise_value(value)
            for key, value in filtered_updates.items()
        }

        anchor_value = filtered_updates.get("anchor")

        def _load_processed_data(raw: Any) -> dict[str, Any]:
            if isinstance(raw, dict):
                return dict(raw)
            if isinstance(raw, str):
                try:
                    loaded = json.loads(raw)
                    if isinstance(loaded, dict):
                        return dict(loaded)
                except Exception:
                    return {}
            return {}

        spatial_records: list[dict[str, Any]] = []

        def _safe_merge_spatial_records(records: list[dict[str, Any]]) -> None:
            if not records:
                return
            session_factory = getattr(self.db_manager, "SessionLocal", None)
            if session_factory is None:
                return
            try:
                with session_factory() as session:
                    for payload in records:
                        try:
                            session.merge(SpatialMetadata(**payload))
                        except OperationalError as exc:
                            session.rollback()
                            logger.debug(
                                "Spatial metadata merge skipped during update: %s", exc
                            )
                            return
                    session.commit()
            except OperationalError as exc:
                logger.debug(
                    "Spatial metadata merge skipped during update: %s", exc
                )

        def _apply_updates(instance: Any) -> bool:
            changed = False
            processed_data = _load_processed_data(getattr(instance, "processed_data", {}))
            processed_mutated = False
            spatial_changed = False

            if "text" in filtered_updates:
                text_value = filtered_updates["text"]
                if text_value is not None:
                    processed_data["text"] = text_value
                    processed_data["content"] = text_value
                    if hasattr(instance, "summary"):
                        instance.summary = text_value
                    if hasattr(instance, "searchable_content"):
                        instance.searchable_content = text_value
                else:
                    processed_data.pop("text", None)
                    processed_data.pop("content", None)
                processed_mutated = True
                changed = True

            if "tokens" in filtered_updates:
                tokens_value = filtered_updates["tokens"]
                if tokens_value is None:
                    processed_data.pop("tokens", None)
                else:
                    processed_data["tokens"] = tokens_value
                processed_mutated = True
                changed = True

            if "emotional_intensity" in filtered_updates:
                intensity = filtered_updates["emotional_intensity"]
                if intensity is None:
                    processed_data.pop("emotional_intensity", None)
                else:
                    processed_data["emotional_intensity"] = intensity
                processed_mutated = True
                changed = True

            if processed_mutated:
                instance.processed_data = processed_data

            if "x_coord" in filtered_updates:
                instance.x_coord = filtered_updates["x_coord"]
                changed = True
                spatial_changed = True

            if "y_coord" in filtered_updates:
                instance.y_coord = filtered_updates["y_coord"]
                changed = True
                spatial_changed = True

            if "z_coord" in filtered_updates:
                instance.z_coord = filtered_updates["z_coord"]
                changed = True
                spatial_changed = True

            if "symbolic_anchors" in filtered_updates:
                anchors = canonicalize_symbolic_anchors(
                    filtered_updates["symbolic_anchors"]
                )
                if anchors is None:
                    anchors = [anchor_value] if anchor_value else []
                instance.symbolic_anchors = anchors
                changed = True

            if (
                isinstance(instance, LongTermMemory)
                and "timestamp" in filtered_updates
                and filtered_updates["timestamp"] is not None
            ):
                instance.timestamp = filtered_updates["timestamp"]
                changed = True

            if changed:
                allowed = self._apply_retention_policy_on_write(
                    namespace=getattr(instance, "namespace", self.namespace)
                    or self.namespace
                    or "default",
                    team_id=getattr(instance, "team_id", None),
                    workspace_id=getattr(instance, "workspace_id", None),
                    memory_id=getattr(instance, "memory_id", memory_id),
                    proposed_privacy=getattr(instance, "y_coord", None),
                    proposed_importance=getattr(instance, "importance_score", None),
                    source="update_memory",
                )
                if not allowed:
                    raise MemoriaError(
                        f"Retention policy blocked updating memory '{memory_id}'"
                    )

            if (
                isinstance(instance, LongTermMemory)
                and spatial_changed
                and changed
            ):
                timestamp_value = (
                    getattr(instance, "timestamp", None)
                    or getattr(instance, "created_at", None)
                    or datetime.utcnow()
                )
                spatial_records.append(
                    {
                        "memory_id": getattr(instance, "memory_id", memory_id),
                        "namespace": getattr(instance, "namespace", self.namespace)
                        or self.namespace
                        or "default",
                        "team_id": getattr(instance, "team_id", None),
                        "workspace_id": getattr(instance, "workspace_id", None),
                        "timestamp": timestamp_value,
                        "x": getattr(instance, "x_coord", None),
                        "y": getattr(instance, "y_coord", None),
                        "z": getattr(instance, "z_coord", None),
                        "symbolic_anchors": list(
                            getattr(instance, "symbolic_anchors", []) or []
                        ),
                    }
                )

            return changed

        try:
            with self.db_manager.SessionLocal() as session:
                updated = False

                if self.db_manager.enable_short_term:
                    short_term = (
                        session.query(ShortTermMemory)
                        .filter(
                            ShortTermMemory.memory_id == memory_id,
                            ShortTermMemory.namespace == self.namespace,
                        )
                        .one_or_none()
                    )
                    if short_term is not None:
                        updated |= _apply_updates(short_term)

                long_term = (
                    session.query(LongTermMemory)
                    .filter(
                        LongTermMemory.memory_id == memory_id,
                        LongTermMemory.namespace == self.namespace,
                    )
                    .one_or_none()
                )
                if long_term is not None:
                    updated |= _apply_updates(long_term)

                if updated:
                    session.commit()
                else:
                    session.rollback()

            _safe_merge_spatial_records(spatial_records)

            if updated:
                snapshot = self.get_memory_snapshot(memory_id, refresh=True)
                replica = self._build_replication_payload(memory_id)
                payload = self._build_memory_event_payload(
                    memory_id=memory_id,
                    snapshot=snapshot,
                    replica=replica,
                    changes=serialised_changes,
                )
                self._emit_sync_event(
                    SyncEventAction.MEMORY_UPDATED,
                    "memory",
                    memory_id,
                    payload,
                )
            return updated
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(f"Failed to update memory {memory_id}: {e}")
            raise MemoriaError(f"Failed to update memory {memory_id}: {e}")

    def store_personal_memory(
        self,
        entry: PersonalMemoryEntry | Mapping[str, Any],
        *,
        namespace: str | None = None,
        team_id: str | None = None,
        workspace_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist a personal memory entry directly to long-term storage."""

        if entry is None:
            raise MemoriaError("Personal memory entry payload is required")

        if isinstance(entry, PersonalMemoryEntry):
            model = entry
        elif isinstance(entry, Mapping):
            try:
                model = PersonalMemoryEntry(**dict(entry))
            except ValidationError as exc:
                raise MemoriaError("Invalid personal memory entry payload") from exc
        else:
            raise MemoriaError(
                "Personal memory entry must be a mapping or PersonalMemoryEntry instance"
            )

        metadata = dict(model.metadata or {})

        workspace_context = self._resolve_workspace_context(
            workspace_id=workspace_id, user_id=user_id
        )
        effective_workspace_id = (
            workspace_context.workspace_id
            if workspace_context is not None
            else self.workspace_id
        )
        workspace_team_id = (
            self._clean_identifier(workspace_context.team_id)
            if workspace_context is not None and workspace_context.team_id
            else None
        )

        normalized_team_id: str | None = None
        if team_id is not None:
            normalized_team_id = self._normalise_team_id(team_id)

        effective_team_id = (
            normalized_team_id
            if normalized_team_id is not None
            else (workspace_team_id or self.team_id)
        )

        metadata_namespace = self._clean_identifier(metadata.get("namespace"))
        resolved_namespace = (
            self._clean_identifier(namespace)
            or metadata_namespace
            or self._personal_namespace_name()
        )

        ingestion_state: dict[str, Any] = {
            "anchor": model.anchor,
            "text": model.text,
            "tokens": model.tokens,
            "x_coord": model.x_coord,
            "y_coord": model.y_coord,
            "z_coord": model.z_coord,
            "symbolic_anchors": list(model.symbolic_anchors or []),
            "namespace": resolved_namespace,
            "team_id": effective_team_id,
            "workspace_id": effective_workspace_id,
            "metadata": copy.deepcopy(metadata),
        }

        removed_fields: set[str] = set()

        def _apply_redaction(decision: PolicyDecision) -> None:
            result: RedactionResult = apply_redactions(ingestion_state, decision)
            removed_fields.update(result.removed)

        policy_payload = {
            "anchor": model.anchor,
            "namespace": resolved_namespace,
            "team_id": effective_team_id,
            "workspace_id": effective_workspace_id,
            "user_id": self._clean_identifier(user_id) or self._default_user_id,
            "privacy": model.y_coord,
            "tokens": model.tokens,
            "source": "personal_store",
        }

        self.enforce_policy(
            EnforcementStage.INGESTION,
            policy_payload,
            allow_redaction=True,
            on_redact=_apply_redaction,
        )

        if "text" in removed_fields and not ingestion_state.get("text"):
            ingestion_state["text"] = ""
        if "tokens" in removed_fields and "tokens" not in ingestion_state:
            ingestion_state["tokens"] = 0
        if "symbolic_anchors" in removed_fields and not ingestion_state.get(
            "symbolic_anchors"
        ):
            ingestion_state["symbolic_anchors"] = []

        anchor = ingestion_state.get("anchor", model.anchor)
        text = ingestion_state.get("text", model.text)
        tokens = ingestion_state.get("tokens", model.tokens)
        x_coord = ingestion_state.get("x_coord", model.x_coord)
        y_coord = ingestion_state.get("y_coord", model.y_coord)
        z_coord = ingestion_state.get("z_coord", model.z_coord)
        symbolic_anchors = canonicalize_symbolic_anchors(
            ingestion_state.get("symbolic_anchors", model.symbolic_anchors)
        )
        if not symbolic_anchors and anchor:
            symbolic_anchors = [anchor]

        metadata = dict(ingestion_state.get("metadata", metadata) or {})
        resolved_namespace = (
            self._clean_identifier(ingestion_state.get("namespace"))
            or resolved_namespace
        )
        effective_team_id = ingestion_state.get("team_id", effective_team_id)
        effective_workspace_id = ingestion_state.get(
            "workspace_id", effective_workspace_id
        )

        def _to_string_list(value: Any) -> list[str]:
            if not value:
                return []
            if isinstance(value, str):
                candidate = value.strip()
                return [candidate] if candidate else []
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                result: list[str] = []
                for item in value:
                    if item is None:
                        continue
                    candidate = str(item).strip()
                    if candidate:
                        result.append(candidate)
                return result
            return []

        def _clamp_unit_float(value: Any) -> float | None:
            numeric = self._coerce_float(value)
            if numeric is None:
                return None
            return max(0.0, min(1.0, numeric))

        importance_value = metadata.get("importance") or metadata.get(
            "importance_level"
        )
        importance_level = MemoryImportanceLevel.HIGH
        if isinstance(importance_value, MemoryImportanceLevel):
            importance_level = importance_value
        elif isinstance(importance_value, str):
            try:
                importance_level = MemoryImportanceLevel(importance_value.lower())
            except ValueError:
                pass
        elif isinstance(importance_value, (int, float)):
            numeric = float(importance_value)
            if numeric >= 0.85:
                importance_level = MemoryImportanceLevel.CRITICAL
            elif numeric >= 0.7:
                importance_level = MemoryImportanceLevel.HIGH
            elif numeric >= 0.5:
                importance_level = MemoryImportanceLevel.MEDIUM
            else:
                importance_level = MemoryImportanceLevel.LOW

        summary = metadata.get("summary") or text
        topic = metadata.get("topic")
        if topic is not None:
            topic = str(topic)
        entities = _to_string_list(metadata.get("entities"))
        keywords = _to_string_list(metadata.get("keywords"))
        supersedes = _to_string_list(metadata.get("supersedes"))
        related_memories = _to_string_list(metadata.get("related_memories"))
        duplicate_of = metadata.get("duplicate_of")
        if duplicate_of is not None:
            duplicate_of = self._clean_identifier(str(duplicate_of))

        is_preference = bool(metadata.get("is_preference", False))
        is_skill_knowledge = bool(metadata.get("is_skill_knowledge", False))
        is_current_project = bool(metadata.get("is_current_project", False))
        promotion_eligible = bool(metadata.get("promotion_eligible", False))

        confidence_score = self._coerce_float(metadata.get("confidence_score"))
        if confidence_score is None:
            confidence_score = 0.9
        else:
            confidence_score = max(0.0, min(1.0, confidence_score))

        emotional_intensity = _clamp_unit_float(
            metadata.get("emotional_intensity")
        )

        classification_reason = (
            str(metadata.get("classification_reason"))
            if metadata.get("classification_reason")
            else "Personal memory ingestion"
        )

        timestamp = model.timestamp or datetime.utcnow()
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc)

        processed_memory = ProcessedLongTermMemory(
            content=text,
            summary=summary,
            classification=MemoryClassification.PERSONAL,
            importance=importance_level,
            topic=topic,
            entities=entities,
            keywords=keywords,
            is_user_context=True,
            is_preference=is_preference,
            is_skill_knowledge=is_skill_knowledge,
            is_current_project=is_current_project,
            duplicate_of=duplicate_of,
            supersedes=supersedes,
            related_memories=related_memories,
            conversation_id=model.chat_id,
            confidence_score=confidence_score,
            extraction_timestamp=timestamp,
            emotional_intensity=emotional_intensity,
            x_coord=x_coord,
            y_coord=y_coord,
            z_coord=z_coord,
            symbolic_anchors=symbolic_anchors,
            classification_reason=classification_reason,
            promotion_eligible=promotion_eligible,
            documents=model.documents,
        )

        proposed_memory_id = str(uuid.uuid4())
        allowed = self._apply_retention_policy_on_write(
            namespace=resolved_namespace,
            team_id=effective_team_id,
            workspace_id=effective_workspace_id,
            memory_id=proposed_memory_id,
            proposed_privacy=y_coord,
            proposed_importance=processed_memory.importance_score,
            source="store_personal_memory",
        )
        if not allowed:
            raise MemoriaError(
                f"Retention policy blocked storing personal memory in namespace '{resolved_namespace}'"
            )

        memory_id = self.db_manager.store_direct_long_term_memory(
            processed_memory,
            model.chat_id,
            resolved_namespace,
            team_id=effective_team_id,
            workspace_id=effective_workspace_id,
            documents=model.documents,
            memory_id=proposed_memory_id,
        )

        timestamp_naive = timestamp.replace(tzinfo=None)
        should_record_spatial = any(
            value is not None for value in (x_coord, y_coord, z_coord)
        ) or bool(symbolic_anchors)
        if should_record_spatial:
            anchors_for_spatial = list(symbolic_anchors or [])
            if anchor and anchor not in anchors_for_spatial:
                anchors_for_spatial.append(anchor)
            self._record_spatial_metadata(
                memory_id=memory_id,
                namespace=resolved_namespace,
                timestamp=timestamp_naive,
                x_coord=x_coord,
                y_coord=y_coord,
                z_coord=z_coord,
                symbolic_anchors=anchors_for_spatial,
                team_id=effective_team_id,
                workspace_id=effective_workspace_id,
            )

        snapshot = self.get_memory_snapshot(memory_id, refresh=True)
        replica = self._build_replication_payload(memory_id)

        document_payload = self._normalise_documents_payload(model.documents)

        changes: dict[str, Any] = {
            "anchor": anchor,
            "tokens": tokens,
            "metadata": copy.deepcopy(metadata),
            "chat_id": model.chat_id,
        }
        if document_payload:
            changes["documents"] = document_payload

        payload = self._build_memory_event_payload(
            memory_id=memory_id,
            snapshot=snapshot,
            replica=replica,
            changes=changes,
            explicit_privacy=self._coerce_float(y_coord),
        )
        self._emit_sync_event(
            SyncEventAction.MEMORY_CREATED,
            "memory",
            memory_id,
            payload,
        )

        return {
            "memory_id": memory_id,
            "chat_id": model.chat_id,
            "namespace": resolved_namespace,
            "team_id": effective_team_id,
            "workspace_id": effective_workspace_id,
            "status": "stored",
            "privacy": y_coord,
            "importance_score": processed_memory.importance_score,
        }

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
        workspace_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Directly store a memory into the database with spatial metadata."""
        symbolic_anchors = canonicalize_symbolic_anchors(symbolic_anchors)
        y_coord = y
        z_coord = z
        ts = timestamp or datetime.utcnow()
        if x_coord is None and ts:
            delta = datetime.utcnow() - ts
            x_coord = -delta.total_seconds() / (24 * 3600.0)
        namespace = self.namespace or "default"
        memory_id = str(uuid.uuid4())
        resolved_chat_id = chat_id or str(uuid.uuid4())
        workspace_context = self._resolve_workspace_context(
            workspace_id=workspace_id, user_id=user_id
        )
        effective_workspace_id = (
            workspace_context.workspace_id if workspace_context is not None else self.workspace_id
        )
        workspace_team_id = (
            self._clean_identifier(workspace_context.team_id)
            if workspace_context is not None
            else None
        )
        effective_team_id = workspace_team_id or self.team_id

        ingestion_state: dict[str, Any] = {
            "anchor": anchor,
            "text": text,
            "tokens": tokens,
            "y_coord": y_coord,
            "z_coord": z_coord,
            "symbolic_anchors": list(symbolic_anchors or []),
            "namespace": namespace,
            "team_id": effective_team_id,
            "workspace_id": effective_workspace_id,
            "chat_id": resolved_chat_id,
        }
        removed_fields: set[str] = set()

        def _apply_redaction(decision: PolicyDecision) -> None:
            result: RedactionResult = apply_redactions(ingestion_state, decision)
            removed_fields.update(result.removed)

        policy_payload = {
            "anchor": anchor,
            "namespace": namespace,
            "team_id": effective_team_id,
            "workspace_id": effective_workspace_id,
            "user_id": self._clean_identifier(user_id) or self._default_user_id,
            "privacy": y_coord,
            "tokens": tokens,
        }

        decision = self.enforce_policy(
            EnforcementStage.INGESTION,
            policy_payload,
            allow_redaction=True,
            on_redact=_apply_redaction,
        )

        if "text" in removed_fields and not ingestion_state.get("text"):
            ingestion_state["text"] = ""
        if "tokens" in removed_fields and "tokens" not in ingestion_state:
            ingestion_state["tokens"] = 0
        if "symbolic_anchors" in removed_fields and "symbolic_anchors" not in ingestion_state:
            ingestion_state["symbolic_anchors"] = []

        anchor = ingestion_state.get("anchor", anchor)
        text = ingestion_state.get("text", text)
        tokens = ingestion_state.get("tokens", tokens)
        y_coord = ingestion_state.get("y_coord", y_coord)
        z_coord = ingestion_state.get("z_coord", z_coord)
        symbolic_anchors = canonicalize_symbolic_anchors(
            ingestion_state.get("symbolic_anchors", symbolic_anchors)
        )
        namespace = ingestion_state.get("namespace", namespace)
        effective_team_id = ingestion_state.get("team_id", effective_team_id)
        effective_workspace_id = ingestion_state.get(
            "workspace_id", effective_workspace_id
        )
        resolved_chat_id = ingestion_state.get("chat_id", resolved_chat_id)

        if decision.action is PolicyAction.REDACT:
            policy_payload["redacted"] = True

        allowed = self._apply_retention_policy_on_write(
            namespace=namespace,
            team_id=effective_team_id,
            workspace_id=effective_workspace_id,
            memory_id=memory_id,
            proposed_privacy=y_coord,
            proposed_importance=0.5,
            source="store_memory",
        )
        if not allowed:
            raise MemoriaError(
                f"Retention policy blocked storing memory in namespace '{namespace}'"
            )

        try:
            with self.db_manager.SessionLocal() as session:
                if chat_id is None:
                    self.db_manager.store_chat_history(
                        chat_id=resolved_chat_id,
                        user_input=self.AUTOGEN_USER_PLACEHOLDER,
                        ai_output=self.AUTOGEN_AI_PLACEHOLDER,
                        timestamp=ts,
                        session_id=resolved_chat_id,
                        model="auto-generated",
                        namespace=namespace,
                        tokens_used=0,
                        metadata={"auto_generated": True},
                        team_id=effective_team_id,
                        workspace_id=effective_workspace_id,
                    )
                memory = LongTermMemory(
                    memory_id=memory_id,
                    original_chat_id=resolved_chat_id,
                    processed_data={
                        "text": text,
                        "tokens": tokens,
                        "emotional_intensity": emotional_intensity,
                    },
                    importance_score=0.5,
                    category_primary="manual",
                    retention_type="long_term",
                    namespace=namespace,
                    team_id=effective_team_id,
                    workspace_id=effective_workspace_id,
                    timestamp=ts,
                    created_at=datetime.utcnow(),
                    searchable_content=text,
                    summary=text,
                    x_coord=x_coord,
                    y_coord=y_coord,
                    z_coord=z_coord,
                    symbolic_anchors=symbolic_anchors or [anchor],
                )
                session.add(memory)
                session.commit()

            should_record_spatial = any(
                value is not None for value in (x_coord, y_coord, z_coord)
            ) or bool(symbolic_anchors)

            if should_record_spatial:
                self._record_spatial_metadata(
                    memory_id=memory_id,
                    namespace=namespace,
                    timestamp=ts,
                    x_coord=x_coord,
                    y_coord=y_coord,
                    z_coord=z_coord,
                    symbolic_anchors=symbolic_anchors or [anchor],
                    team_id=effective_team_id,
                    workspace_id=effective_workspace_id,
                )
            snapshot = self.get_memory_snapshot(memory_id, refresh=True)
            replica = self._build_replication_payload(memory_id)
            payload = self._build_memory_event_payload(
                memory_id=memory_id,
                snapshot=snapshot,
                replica=replica,
                changes=None,
            )
            self._emit_sync_event(
                SyncEventAction.MEMORY_CREATED,
                "memory",
                memory_id,
                payload,
            )
            return memory_id
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise

    def _parse_import_datetime(self, value: Any, *, field: str) -> datetime | None:
        """Convert import timestamps to naive UTC datetimes."""

        if value is None or value == "":
            return None

        dt_value: datetime
        if isinstance(value, datetime):
            dt_value = value
        elif isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            if cleaned.endswith("Z"):
                cleaned = f"{cleaned[:-1]}+00:00"
            try:
                dt_value = datetime.fromisoformat(cleaned)
            except ValueError as exc:  # pragma: no cover - invalid input guard
                raise MemoriaError(f"Invalid {field} value '{value}': {exc}") from exc
        else:  # pragma: no cover - defensive guard
            raise MemoriaError(
                f"Unsupported {field} type '{type(value).__name__}' for import"
            )

        if dt_value.tzinfo is not None:
            return dt_value.astimezone(timezone.utc).replace(tzinfo=None)
        return dt_value

    def _normalise_import_record(
        self,
        payload: Mapping[str, Any],
        namespace_map: Mapping[str, str],
        default_namespace: str,
        *,
        index: int,
    ) -> dict[str, Any]:
        """Validate and normalise a single import payload."""

        if not isinstance(payload, Mapping):
            raise MemoriaError("Import payload must be a mapping")

        processed_data = payload.get("processed_data")
        text_value = payload.get("text")
        if not text_value and isinstance(processed_data, Mapping):
            text_value = processed_data.get("text") or processed_data.get("content")
        if not text_value:
            text_value = payload.get("content")
        if not isinstance(text_value, str):
            raise MemoriaError("Import payload is missing text content")
        normalized_text = text_value.replace("\r\n", "\n").rstrip("\n")

        tokens_value = payload.get("tokens")
        if tokens_value is not None and tokens_value != "":
            try:
                tokens_value = int(tokens_value)
            except (TypeError, ValueError) as exc:
                raise MemoriaError("Invalid tokens value in import payload") from exc
        else:
            tokens_value = None

        intensity_value = payload.get("emotional_intensity")
        if intensity_value is not None and intensity_value != "":
            try:
                intensity_value = float(intensity_value)
            except (TypeError, ValueError) as exc:
                raise MemoriaError(
                    "Invalid emotional_intensity value in import payload"
                ) from exc
        else:
            intensity_value = None

        source_namespace = str(payload.get("namespace") or default_namespace or "default")
        target_namespace = namespace_map.get(source_namespace, source_namespace)
        target_namespace = target_namespace or "default"

        created_at = (
            self._parse_import_datetime(
                payload.get("created_at") or payload.get("timestamp"),
                field="created_at",
            )
            or datetime.utcnow()
        )
        timestamp = (
            self._parse_import_datetime(payload.get("timestamp"), field="timestamp")
            or created_at
        )

        def _coerce_optional_float(value: Any, field: str) -> float | None:
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise MemoriaError(f"Invalid {field} value in import payload") from exc

        x_coord = _coerce_optional_float(payload.get("x_coord"), "x_coord")
        y_coord = _coerce_optional_float(
            payload.get("y_coord", payload.get("privacy")), "y_coord"
        )
        z_coord = _coerce_optional_float(payload.get("z_coord"), "z_coord")

        anchors_source: Any = payload.get("symbolic_anchors")
        if anchors_source is None:
            anchors_source = payload.get("anchors")
        if anchors_source is None and isinstance(processed_data, Mapping):
            anchors_source = processed_data.get("symbolic_anchors")

        if isinstance(anchors_source, str):
            anchors_iterable = [
                item.strip()
                for item in anchors_source.split(",")
                if item and item.strip()
            ]
        elif isinstance(anchors_source, Iterable):
            anchors_iterable = [str(item).strip() for item in anchors_source if item]
        else:
            anchors_iterable = []

        canonical_anchors = canonicalize_symbolic_anchors(anchors_iterable) or []

        category_primary = payload.get("category_primary") or payload.get("category")
        category_primary = str(category_primary) if category_primary else "imported"

        importance_score = payload.get("importance_score")
        if importance_score is not None and importance_score != "":
            try:
                importance_score = float(importance_score)
            except (TypeError, ValueError) as exc:
                raise MemoriaError(
                    "Invalid importance_score value in import payload"
                ) from exc
        else:
            importance_score = 0.5

        summary_value = payload.get("summary")
        if not isinstance(summary_value, str) or not summary_value.strip():
            summary_value = normalized_text

        memory_id_value = payload.get("memory_id")
        memory_id = str(memory_id_value) if memory_id_value else str(uuid.uuid4())

        chat_id_value = (
            payload.get("original_chat_id")
            or payload.get("chat_id")
            or payload.get("session_id")
        )
        original_chat_id = str(chat_id_value) if chat_id_value else str(uuid.uuid4())

        processed_payload: dict[str, Any] = {}
        if isinstance(processed_data, Mapping):
            processed_payload.update(processed_data)

        processed_payload.setdefault("text", normalized_text)
        if tokens_value is not None:
            processed_payload.setdefault("tokens", tokens_value)
        if intensity_value is not None:
            processed_payload.setdefault("emotional_intensity", intensity_value)
        processed_payload.setdefault("symbolic_anchors", canonical_anchors)
        processed_payload.setdefault("imported", True)
        processed_payload.setdefault("import_source_namespace", source_namespace)

        metadata_value = payload.get("metadata")
        if isinstance(metadata_value, Mapping):
            processed_payload.setdefault("metadata", dict(metadata_value))

        return {
            "memory_id": memory_id,
            "namespace": target_namespace,
            "text": normalized_text,
            "tokens": tokens_value,
            "emotional_intensity": intensity_value,
            "created_at": created_at,
            "timestamp": timestamp,
            "x_coord": x_coord,
            "y_coord": y_coord,
            "z_coord": z_coord,
            "anchors": canonical_anchors,
            "importance_score": float(importance_score),
            "category_primary": category_primary,
            "summary": summary_value,
            "original_chat_id": original_chat_id,
            "processed_data": processed_payload,
            "source_namespace": source_namespace,
            "source_index": index,
        }

    def import_memories_bulk(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        namespace_map: Mapping[str, str] | None = None,
        default_namespace: str | None = None,
        skip_existing: bool = True,
    ) -> dict[str, Any]:
        """Bulk import long-term memories from exported payloads."""

        mapped_namespaces = {
            (key or "default"): value
            for key, value in (namespace_map or {}).items()
            if value
        }
        default_ns = default_namespace or self.namespace or "default"

        normalized_records: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_pairs: set[tuple[str, str, datetime]] = set()

        for idx, payload in enumerate(records or []):
            try:
                normalized = self._normalise_import_record(
                    payload,
                    mapped_namespaces,
                    default_ns,
                    index=idx,
                )
            except Exception as exc:
                errors.append({"index": idx, "error": str(exc)})
                continue

            mem_id = normalized["memory_id"]
            if mem_id in seen_ids:
                skipped.append(
                    {
                        "index": idx,
                        "memory_id": mem_id,
                        "reason": "duplicate memory_id in payload",
                    }
                )
                continue
            seen_ids.add(mem_id)

            pair_key = (
                normalized["namespace"],
                normalized["text"],
                normalized["created_at"],
            )
            if pair_key in seen_pairs:
                skipped.append(
                    {
                        "index": idx,
                        "memory_id": mem_id,
                        "reason": "duplicate text and timestamp in payload",
                    }
                )
                continue
            seen_pairs.add(pair_key)

            normalized_records.append(normalized)

        inserted_ids: list[str] = []

        with self.db_manager.SessionLocal() as session:
            for record in normalized_records:
                mem_id = record["memory_id"]
                namespace = record["namespace"]
                text_value = record["text"]
                created_at = record["created_at"]

                if skip_existing:
                    existing_by_id = (
                        session.query(LongTermMemory.memory_id)
                        .filter(
                            LongTermMemory.namespace == namespace,
                            LongTermMemory.memory_id == mem_id,
                        )
                    )
                    if self.team_id is not None:
                        existing_by_id = existing_by_id.filter(
                            LongTermMemory.team_id == self.team_id
                        )
                    if self.workspace_id is not None:
                        existing_by_id = existing_by_id.filter(
                            LongTermMemory.workspace_id == self.workspace_id
                        )
                    existing_by_id = existing_by_id.first()
                    if existing_by_id is not None:
                        skipped.append(
                            {
                                "index": record["source_index"],
                                "memory_id": mem_id,
                                "reason": "existing memory_id in target namespace",
                            }
                        )
                        continue

                    existing_by_content_query = session.query(LongTermMemory.memory_id).filter(
                        LongTermMemory.namespace == namespace,
                        and_(
                            LongTermMemory.searchable_content == text_value,
                            LongTermMemory.created_at == created_at,
                        ),
                    )
                    if self.team_id is not None:
                        existing_by_content_query = existing_by_content_query.filter(
                            LongTermMemory.team_id == self.team_id
                        )
                    if self.workspace_id is not None:
                        existing_by_content_query = existing_by_content_query.filter(
                            LongTermMemory.workspace_id == self.workspace_id
                        )
                    existing_by_content = existing_by_content_query.first()
                    if existing_by_content is not None:
                        skipped.append(
                            {
                                "index": record["source_index"],
                                "memory_id": mem_id,
                                "reason": "existing memory with matching text and created_at",
                            }
                        )
                        continue

                db_record = LongTermMemory(
                    memory_id=mem_id,
                    original_chat_id=record["original_chat_id"],
                    processed_data=record["processed_data"],
                    importance_score=record["importance_score"],
                    category_primary=record["category_primary"],
                    retention_type="long_term",
                    namespace=namespace,
                    team_id=self.team_id,
                    workspace_id=self.workspace_id,
                    timestamp=record["timestamp"],
                    created_at=created_at,
                    searchable_content=text_value,
                    summary=record["summary"],
                    x_coord=record["x_coord"],
                    y_coord=record["y_coord"],
                    z_coord=record["z_coord"],
                    symbolic_anchors=record["anchors"],
                )

                session.add(db_record)
                try:
                    session.commit()
                except Exception as exc:  # pragma: no cover - defensive logging
                    session.rollback()
                    errors.append(
                        {
                            "index": record["source_index"],
                            "memory_id": mem_id,
                            "error": str(exc),
                        }
                    )
                    continue

                inserted_ids.append(mem_id)
                self._record_spatial_metadata(
                    memory_id=mem_id,
                    namespace=namespace,
                    timestamp=record["timestamp"],
                    x_coord=record["x_coord"],
                    y_coord=record["y_coord"],
                    z_coord=record["z_coord"],
                    symbolic_anchors=record["anchors"],
                    team_id=self.team_id,
                    workspace_id=self.workspace_id,
                )
                snapshot = self.get_memory_snapshot(mem_id, refresh=True)
                self._emit_sync_event(
                    SyncEventAction.MEMORY_CREATED,
                    "memory",
                    mem_id,
                    snapshot or {"memory_id": mem_id},
                )

        return {"inserted": inserted_ids, "skipped": skipped, "errors": errors}

    def store_thread(
        self,
        thread_id: str,
        *,
        shared_anchors: list[str] | None = None,
        ritual: dict[str, Any] | None = None,
        centroid: dict[str, float | None] | None = None,
        message_links: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist thread-level metadata and message linkages."""

        canonical_anchors = canonicalize_symbolic_anchors(shared_anchors) or []
        centroid = centroid or {}
        message_links = message_links or []

        try:
            with self.db_manager.SessionLocal() as session:
                try:
                    event = (
                        session.query(ThreadEvent)
                        .filter(
                            ThreadEvent.thread_id == thread_id,
                            ThreadEvent.namespace == self.namespace,
                        )
                        .one_or_none()
                    )

                    payload = {
                        "thread_id": thread_id,
                        "namespace": self.namespace,
                        "symbolic_anchors": canonical_anchors or None,
                        "ritual_metadata": ritual or None,
                        "ritual_name": (ritual or {}).get("name") if ritual else None,
                        "ritual_phase": (ritual or {}).get("phase") if ritual else None,
                        "centroid_x": centroid.get("x"),
                        "centroid_y": centroid.get("y"),
                        "centroid_z": centroid.get("z"),
                    }

                    if event is None:
                        event = ThreadEvent(**payload)
                    else:
                        for key, value in payload.items():
                            setattr(event, key, value)

                    session.add(event)

                    session.query(ThreadMessageLink).filter(
                        ThreadMessageLink.thread_id == thread_id,
                        ThreadMessageLink.namespace == self.namespace,
                    ).delete()

                    session.query(LinkMemoryThread).filter(
                        LinkMemoryThread.relation == f"thread:{thread_id}"
                    ).delete()

                    ordered_links = sorted(
                        message_links,
                        key=lambda item: item.get("sequence_index", 0),
                    )

                    for link in ordered_links:
                        session.add(
                            ThreadMessageLink(
                                thread_id=thread_id,
                                memory_id=link["memory_id"],
                                namespace=self.namespace,
                                sequence_index=int(link.get("sequence_index", 0)),
                                role=link.get("role"),
                                anchor=link.get("anchor"),
                                timestamp=link.get("timestamp"),
                            )
                        )

                    for current, nxt in zip(ordered_links, ordered_links[1:]):
                        session.add(
                            LinkMemoryThread(
                                source_memory_id=current["memory_id"],
                                target_memory_id=nxt["memory_id"],
                                relation=f"thread:{thread_id}",
                            )
                        )

                    session.commit()
                except Exception:
                    session.rollback()
                    raise
            payload = self.get_thread(thread_id, use_cache=False)
            self._emit_sync_event(
                SyncEventAction.THREAD_UPDATED,
                "thread",
                thread_id,
                payload or {"thread_id": thread_id},
            )
        except Exception as exc:
            logger.error(f"Failed to store thread {thread_id}: {exc}")
            raise MemoriaError(f"Failed to store thread {thread_id}: {exc}")

    def _resolve_memory_record(
        self, session: Session, memory_id: str
    ) -> tuple[Any | None, str | None]:
        """Return the storage record and its table for a given memory id."""

        record = (
            session.query(LongTermMemory)
            .filter(
                LongTermMemory.memory_id == memory_id,
                LongTermMemory.namespace == self.namespace,
            )
            .one_or_none()
        )
        if record is not None:
            return record, "long_term"

        if not self.db_manager.enable_short_term:
            return None, None

        record = (
            session.query(ShortTermMemory)
            .filter(
                ShortTermMemory.memory_id == memory_id,
                ShortTermMemory.namespace == self.namespace,
            )
            .one_or_none()
        )
        if record is None:
            return None, None
        return record, "short_term"

    def get_thread(
        self, thread_id: str, *, use_cache: bool = True
    ) -> dict[str, Any] | None:
        """Return thread metadata and ordered message payloads."""

        try:
            if use_cache:
                cached = self._get_cached_thread(thread_id)
                if cached is not None:
                    return cached

            with self.db_manager.SessionLocal() as session:
                event = (
                    session.query(ThreadEvent)
                    .filter(
                        ThreadEvent.thread_id == thread_id,
                        ThreadEvent.namespace == self.namespace,
                    )
                    .one_or_none()
                )
                if event is None:
                    self._remove_thread_cache(thread_id)
                    return None

                links = (
                    session.query(ThreadMessageLink)
                    .filter(
                        ThreadMessageLink.thread_id == thread_id,
                        ThreadMessageLink.namespace == self.namespace,
                    )
                    .order_by(ThreadMessageLink.sequence_index.asc())
                    .all()
                )

                messages: list[dict[str, Any]] = []
                for link in links:
                    record, source = self._resolve_memory_record(session, link.memory_id)
                    if record is None:
                        continue

                    processed = getattr(record, "processed_data", {}) or {}
                    if isinstance(processed, str):
                        try:
                            processed = json.loads(processed)
                        except Exception:
                            processed = {}
                    summary = getattr(record, "summary", None)
                    symbolic = canonicalize_symbolic_anchors(
                        getattr(record, "symbolic_anchors", None)
                    )

                    messages.append(
                        {
                            "memory_id": link.memory_id,
                            "sequence_index": link.sequence_index,
                            "role": link.role,
                            "anchor": link.anchor,
                            "timestamp": link.timestamp.isoformat()
                            if link.timestamp
                            else None,
                            "source_table": source,
                            "symbolic_anchors": symbolic or [],
                            "summary": summary,
                            "content": processed.get("text")
                            if isinstance(processed, dict)
                            else summary,
                        }
                    )

                ritual = event.ritual_metadata or {}
                if event.ritual_name and (
                    not isinstance(ritual, dict) or "name" not in ritual
                ):
                    ritual = dict(ritual or {})
                    ritual.setdefault("name", event.ritual_name)
                if event.ritual_phase and (
                    not isinstance(ritual, dict) or "phase" not in ritual
                ):
                    ritual = dict(ritual or {})
                    ritual.setdefault("phase", event.ritual_phase)

                payload = {
                    "thread_id": thread_id,
                    "symbolic_anchors": event.symbolic_anchors or [],
                    "centroid": {
                        "x": event.centroid_x,
                        "y": event.centroid_y,
                        "z": event.centroid_z,
                    },
                    "ritual": ritual or None,
                    "messages": messages,
                }

            self._set_thread_cache(thread_id, payload)
            return copy.deepcopy(payload)
        except Exception as exc:
            logger.error(f"Failed to fetch thread {thread_id}: {exc}")
            raise MemoriaError(f"Failed to fetch thread {thread_id}: {exc}")

    def get_threads_for_memory(self, memory_id: str) -> list[dict[str, Any]]:
        """Return lightweight thread memberships for a memory identifier."""

        try:
            with self.db_manager.SessionLocal() as session:
                links = (
                    session.query(ThreadMessageLink)
                    .filter(
                        ThreadMessageLink.memory_id == memory_id,
                        ThreadMessageLink.namespace == self.namespace,
                    )
                    .order_by(ThreadMessageLink.sequence_index.asc())
                    .all()
                )

                results: list[dict[str, Any]] = []
                for link in links:
                    event = (
                        session.query(ThreadEvent)
                        .filter(
                            ThreadEvent.thread_id == link.thread_id,
                            ThreadEvent.namespace == self.namespace,
                        )
                        .one_or_none()
                    )

                    results.append(
                        {
                            "thread_id": link.thread_id,
                            "sequence_index": link.sequence_index,
                            "role": link.role,
                            "anchor": link.anchor,
                            "timestamp": link.timestamp.isoformat()
                            if link.timestamp
                            else None,
                            "symbolic_anchors": (event.symbolic_anchors if event else []),
                            "centroid": {
                                "x": getattr(event, "centroid_x", None),
                                "y": getattr(event, "centroid_y", None),
                                "z": getattr(event, "centroid_z", None),
                            },
                        }
                    )

                return results
        except Exception as exc:
            logger.error(
                f"Failed to fetch threads for memory {memory_id}: {exc}"
            )
            return []

    def _record_spatial_metadata(
        self,
        *,
        memory_id: str,
        namespace: str,
        timestamp: datetime,
        x_coord: float | None,
        y_coord: float | None,
        z_coord: float | None,
        symbolic_anchors: list[str],
        team_id: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        if not any(
            value is not None for value in (x_coord, y_coord, z_coord, symbolic_anchors)
        ):
            return

        with self.db_manager.SessionLocal() as session:
            payload = SpatialMetadata(
                memory_id=memory_id,
                namespace=namespace,
                team_id=team_id,
                workspace_id=workspace_id,
                timestamp=timestamp,
                x=x_coord,
                y=y_coord,
                z=z_coord,
                symbolic_anchors=symbolic_anchors or [],
            )
            try:
                session.merge(payload)
                session.commit()
            except OperationalError as exc:
                session.rollback()
                logger.debug(
                    "Spatial metadata merge skipped due to schema mismatch: %s", exc
                )

    def remove_short_term_memory(self, memory_id: str) -> None:
        if not getattr(self.db_manager, "enable_short_term", False):
            return

        with self.db_manager.SessionLocal() as session:
            query = session.query(ShortTermMemory).filter(
                ShortTermMemory.memory_id == memory_id,
                ShortTermMemory.namespace == self.namespace,
            )
            if self.team_id is not None:
                query = query.filter(ShortTermMemory.team_id == self.team_id)
            if self.workspace_id is not None:
                query = query.filter(ShortTermMemory.workspace_id == self.workspace_id)
            deleted = query.delete()
            if deleted:
                session.commit()
            else:
                session.rollback()

    def stage_manual_memory(
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
        namespace: str | None = None,
        team_id: str | None = None,
        user_id: str | None = None,
        share_with_team: bool | None = None,
        importance_score: float = 0.5,
        metadata: dict[str, Any] | None = None,
        workspace_id: str | None = None,
    ) -> StagedManualMemory:
        """Stage a manual memory by writing it to short-term storage and spatial metadata."""

        symbolic_anchors = canonicalize_symbolic_anchors(symbolic_anchors) or [anchor]
        normalized_team_id: str | None = None
        if team_id is not None:
            normalized_team_id = self._normalise_team_id(team_id)
        workspace_context = self._resolve_workspace_context(
            workspace_id=workspace_id, user_id=user_id
        )
        effective_workspace_id = (
            workspace_context.workspace_id if workspace_context is not None else None
        )
        workspace_team_id = (
            self._clean_identifier(workspace_context.team_id)
            if workspace_context is not None
            else None
        )
        resolved_namespace = self.resolve_target_namespace(
            namespace=namespace,
            team_id=normalized_team_id if normalized_team_id is not None else team_id,
            user_id=user_id,
            share_with_team=share_with_team,
        )
        ts = timestamp or datetime.utcnow()
        resolved_chat_id = chat_id or str(uuid.uuid4())
        if x_coord is None and ts:
            x_coord = float((ts.date() - datetime.utcnow().date()).days)

        effective_team_id = (
            normalized_team_id
            if normalized_team_id is not None
            else (workspace_team_id or self.team_id)
        )

        ingestion_state: dict[str, Any] = {
            "anchor": anchor,
            "text": text,
            "tokens": tokens,
            "y_coord": y,
            "z_coord": z,
            "symbolic_anchors": list(symbolic_anchors or []),
            "namespace": resolved_namespace,
            "team_id": effective_team_id,
            "workspace_id": effective_workspace_id,
            "chat_id": resolved_chat_id,
            "metadata": dict(metadata or {}) if metadata else {},
            "importance_score": importance_score,
        }
        removed_fields: set[str] = set()

        def _apply_redaction(decision: PolicyDecision) -> None:
            result: RedactionResult = apply_redactions(ingestion_state, decision)
            removed_fields.update(result.removed)

        policy_payload = {
            "anchor": anchor,
            "namespace": resolved_namespace,
            "team_id": effective_team_id,
            "workspace_id": effective_workspace_id,
            "user_id": self._clean_identifier(user_id) or self._default_user_id,
            "privacy": y,
            "tokens": tokens,
            "source": "manual_stage",
        }

        self.enforce_policy(
            EnforcementStage.INGESTION,
            policy_payload,
            allow_redaction=True,
            on_redact=_apply_redaction,
        )

        if "text" in removed_fields and not ingestion_state.get("text"):
            ingestion_state["text"] = ""
        if "tokens" in removed_fields and "tokens" not in ingestion_state:
            ingestion_state["tokens"] = 0
        if "symbolic_anchors" in removed_fields and "symbolic_anchors" not in ingestion_state:
            ingestion_state["symbolic_anchors"] = []
        if "metadata" in removed_fields:
            ingestion_state.setdefault("metadata", {})

        anchor = ingestion_state.get("anchor", anchor)
        text = ingestion_state.get("text", text)
        tokens = ingestion_state.get("tokens", tokens)
        y = ingestion_state.get("y_coord", y)
        z = ingestion_state.get("z_coord", z)
        symbolic_anchors = canonicalize_symbolic_anchors(
            ingestion_state.get("symbolic_anchors", symbolic_anchors)
        ) or [anchor]
        resolved_namespace = ingestion_state.get("namespace", resolved_namespace)
        effective_team_id = ingestion_state.get("team_id", effective_team_id)
        effective_workspace_id = ingestion_state.get(
            "workspace_id", effective_workspace_id
        )
        resolved_chat_id = ingestion_state.get("chat_id", resolved_chat_id)
        metadata = ingestion_state.get("metadata", metadata)
        importance_score = ingestion_state.get("importance_score", importance_score)

        processed_payload = {
            "text": text,
            "tokens": tokens,
            "anchor": anchor,
            "timestamp": ts.isoformat(),
            "emotional_intensity": emotional_intensity,
        }

        short_term_stored = False
        short_term_id: str | None = None
        if getattr(self.db_manager, "enable_short_term", False) and hasattr(
            self.db_manager, "store_manual_short_term_memory"
        ):
            try:
                short_term_id = self.db_manager.store_manual_short_term_memory(
                    chat_id=resolved_chat_id,
                    namespace=resolved_namespace,
                    processed_payload=processed_payload,
                    summary=text,
                    importance_score=importance_score,
                    x_coord=x_coord,
                    y_coord=y,
                    z_coord=z,
                    symbolic_anchors=symbolic_anchors,
                    team_id=effective_team_id,
                    workspace_id=effective_workspace_id,
                )
                short_term_stored = True
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(f"Failed to stage manual memory in short-term storage: {exc}")

        if short_term_id is None:
            short_term_id = str(uuid.uuid4())

        self._record_spatial_metadata(
            memory_id=short_term_id,
            namespace=resolved_namespace,
            timestamp=ts,
            x_coord=x_coord,
            y_coord=y,
            z_coord=z,
            symbolic_anchors=symbolic_anchors,
            team_id=effective_team_id,
            workspace_id=effective_workspace_id,
        )

        stage_metadata = dict(metadata or {})
        stage_metadata.setdefault("short_term_stored", short_term_stored)
        if effective_team_id:
            stage_metadata.setdefault("team_id", effective_team_id)
        if effective_workspace_id:
            stage_metadata.setdefault("workspace_id", effective_workspace_id)
        stage_metadata.setdefault("namespace", resolved_namespace)

        return StagedManualMemory(
            memory_id=short_term_id,
            chat_id=resolved_chat_id,
            namespace=resolved_namespace,
            anchor=anchor,
            text=text,
            tokens=tokens,
            timestamp=ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc),
            x_coord=x_coord,
            y_coord=y,
            z_coord=z,
            symbolic_anchors=symbolic_anchors,
            metadata=stage_metadata,
        )

    def transfer_spatial_metadata(
        self,
        source_id: str,
        target_id: str,
        *,
        team_id: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        """Move spatial metadata from a short-term record to its promoted counterpart."""

        with self.db_manager.SessionLocal() as session:
            normalized_team_id: str | None = None
            if team_id is not None:
                normalized_team_id = self._normalise_team_id(team_id)
            elif self.team_id is not None:
                normalized_team_id = self.team_id
            normalized_workspace_id = self._clean_identifier(workspace_id)
            if normalized_workspace_id is None:
                normalized_workspace_id = self.workspace_id

            query = session.query(SpatialMetadata).filter(
                SpatialMetadata.memory_id == source_id,
                SpatialMetadata.namespace == self.namespace,
            )
            if normalized_team_id is not None:
                query = query.filter(SpatialMetadata.team_id == normalized_team_id)
            if normalized_workspace_id is not None:
                query = query.filter(
                    SpatialMetadata.workspace_id == normalized_workspace_id
                )

            record = query.one_or_none()
            if not record:
                session.rollback()
                return

            record.memory_id = target_id
            if normalized_team_id is not None:
                record.team_id = normalized_team_id
            if normalized_workspace_id is not None:
                record.workspace_id = normalized_workspace_id
            session.commit()

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID from storage."""
        snapshot_before = self.get_memory_snapshot(memory_id, refresh=True)
        try:
            with self.db_manager.SessionLocal() as session:
                stm_deleted = 0
                if self.db_manager.enable_short_term:
                    stm_query = session.query(ShortTermMemory).filter(
                        ShortTermMemory.memory_id == memory_id,
                        ShortTermMemory.namespace == self.namespace,
                    )
                    if self.team_id is not None:
                        stm_query = stm_query.filter(
                            ShortTermMemory.team_id == self.team_id
                        )
                    if self.workspace_id is not None:
                        stm_query = stm_query.filter(
                            ShortTermMemory.workspace_id == self.workspace_id
                        )
                    stm_deleted = int(stm_query.delete())
                ltm_query = session.query(LongTermMemory).filter(
                    LongTermMemory.memory_id == memory_id,
                    LongTermMemory.namespace == self.namespace,
                )
                if self.team_id is not None:
                    ltm_query = ltm_query.filter(LongTermMemory.team_id == self.team_id)
                if self.workspace_id is not None:
                    ltm_query = ltm_query.filter(
                        LongTermMemory.workspace_id == self.workspace_id
                    )
                ltm_deleted = int(ltm_query.delete())
                session.commit()
                deleted = (stm_deleted + ltm_deleted) > 0

            if deleted:
                self._remove_memory_cache(memory_id)
                payload = self._build_memory_event_payload(
                    memory_id=memory_id,
                    snapshot=snapshot_before,
                    replica=None,
                    changes=None,
                )
                self._emit_sync_event(
                    SyncEventAction.MEMORY_DELETED,
                    "memory",
                    memory_id,
                    payload,
                )
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise MemoriaError(f"Failed to delete memory {memory_id}: {e}")

    def get_conversation_history(
        self,
        *,
        session_id: str | None,
        shared_memory: bool,
        limit: int = 10,
        user_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            workspace_context = self._resolve_workspace_context(
                workspace_id=workspace_id, user_id=user_id
            )
            effective_workspace_id = (
                workspace_context.workspace_id
                if workspace_context is not None
                else self.workspace_id
            )
            effective_team_id = (
                self._clean_identifier(workspace_context.team_id)
                if workspace_context is not None and workspace_context.team_id
                else self.team_id
            )
            history = self.db_manager.get_chat_history(
                namespace=self.namespace,
                session_id=session_id if not shared_memory else None,
                limit=limit,
                team_id=effective_team_id,
                workspace_id=effective_workspace_id,
            )
            if isinstance(history, list):
                return history
            if isinstance(history, dict):
                return [history]
            return []
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    def clear_memory(self, memory_type: str | None = None) -> None:
        self.db_manager.clear_memory(
            self.namespace,
            memory_type,
            team_id=self.team_id,
            workspace_id=self.workspace_id,
        )
        with self._cache_lock:
            self._memory_snapshot_cache.clear()
            self._thread_cache.clear()

    # ------------------------------------------------------------------
    # Helper queries
    # ------------------------------------------------------------------
    async def get_recent_memories_for_dedup(self) -> list[ProcessedLongTermMemory]:
        """Get recent memories for deduplication check."""
        try:
            with self.db_manager.get_connection() as connection:
                result = connection.execute(
                    text(MemoryQueries.SELECT_MEMORIES_FOR_DEDUPLICATION),
                    {
                        "namespace": self.namespace,
                        "processed_for_duplicates": False,
                        "workspace_id": self.workspace_id,
                        "limit": 20,
                    },
                )
                memories: list[ProcessedLongTermMemory] = []
                for row in result:
                    try:
                        memory = ProcessedLongTermMemory(
                            conversation_id=row[0],
                            summary=row[1] or "",
                            content=row[2] or "",
                            classification=row[3] or "conversational",
                            importance="medium",
                            promotion_eligible=False,
                            classification_reason="Existing memory loaded for deduplication check",
                        )
                        memories.append(memory)
                    except Exception as e:
                        logger.debug(f"Skipping malformed memory during dedup: {e}")
                        continue
                return memories
        except Exception as e:
            logger.error(f"Failed to get recent memories for dedup: {e}")
            return []

    def get_essential_conversations(
        self, limit: int = 10, *, namespace: str | None = None
    ) -> list[dict[str, Any]]:
        """Get essential conversations from short-term memory."""
        try:
            if not self.db_manager.enable_short_term:
                return []
            target_namespace = (
                self._clean_identifier(namespace) or self._personal_namespace_name()
            )
            with self.db_manager.get_connection() as connection:
                query = """
                SELECT memory_id, summary, category_primary, importance_score,
                       created_at, searchable_content, processed_data
                FROM short_term_memory
                WHERE namespace = :namespace
                  AND category_primary LIKE 'essential_%'
                  AND (:team_id IS NULL OR team_id = :team_id)
                  AND (:workspace_id IS NULL OR workspace_id = :workspace_id)
                ORDER BY importance_score DESC, created_at DESC
                LIMIT :limit
                """
                result = connection.execute(
                    text(query),
                    {
                        "namespace": target_namespace,
                        "limit": limit,
                        "team_id": self.team_id,
                        "workspace_id": self.workspace_id,
                    },
                )
                essential_conversations = []
                for row in result:
                    processed = row[6]
                    if not isinstance(processed, dict):
                        try:
                            processed = json.loads(processed)
                        except Exception:
                            processed = {}
                    emotion = processed.get("emotional_intensity")
                    essential_conversations.append(
                        {
                            "memory_id": row[0],
                            "summary": row[1],
                            "category_primary": row[2],
                            "importance_score": row[3],
                            "created_at": row[4],
                            "searchable_content": row[5],
                            "processed_data": processed,
                            "emotional_intensity": emotion,
                            "table": "short_term",
                            "memory_type": "short_term",
                        }
                    )
                return essential_conversations
        except Exception as e:
            logger.error(f"Failed to get essential conversations: {e}")
            return []

    def decrement_x_coords(self) -> None:
        """Decrement temporal x coordinates for all stored memories by one day."""
        try:
            with self.db_manager.SessionLocal() as session:
                if self.db_manager.enable_short_term:
                    short_term_query = session.query(ShortTermMemory).filter(
                        ShortTermMemory.namespace == self.namespace,
                        ShortTermMemory.x_coord.isnot(None),
                    )
                    if self.team_id is not None:
                        short_term_query = short_term_query.filter(
                            ShortTermMemory.team_id == self.team_id
                        )
                    if self.workspace_id is not None:
                        short_term_query = short_term_query.filter(
                            ShortTermMemory.workspace_id == self.workspace_id
                        )
                    short_term_query.update(
                        {ShortTermMemory.x_coord: ShortTermMemory.x_coord - 1}
                    )
                long_term_query = session.query(LongTermMemory).filter(
                    LongTermMemory.namespace == self.namespace,
                    LongTermMemory.x_coord.isnot(None),
                )
                if self.team_id is not None:
                    long_term_query = long_term_query.filter(
                        LongTermMemory.team_id == self.team_id
                    )
                if self.workspace_id is not None:
                    long_term_query = long_term_query.filter(
                        LongTermMemory.workspace_id == self.workspace_id
                    )
                long_term_query.update(
                    {LongTermMemory.x_coord: LongTermMemory.x_coord - 1}
                )
                spatial_query = session.query(SpatialMetadata).filter(
                    SpatialMetadata.namespace == self.namespace,
                    SpatialMetadata.x.isnot(None),
                )
                if self.team_id is not None:
                    spatial_query = spatial_query.filter(
                        SpatialMetadata.team_id == self.team_id
                    )
                if self.workspace_id is not None:
                    spatial_query = spatial_query.filter(
                        SpatialMetadata.workspace_id == self.workspace_id
                    )
                spatial_query.update({SpatialMetadata.x: SpatialMetadata.x - 1})
                session.commit()
                logger.info("Decremented x coordinates for all memories")
        except Exception as e:
            logger.error(f"Failed to decrement x coordinates: {e}")
            raise

    def recompute_x_coord_from_timestamp(self, memory_id: str) -> float:
        """Recompute and persist x_coord using the stored timestamp."""
        try:
            with self.db_manager.SessionLocal() as session:
                record = (
                    session.query(LongTermMemory)
                    .filter(
                        LongTermMemory.memory_id == memory_id,
                        LongTermMemory.namespace == self.namespace,
                    )
                    .one_or_none()
                )

                if record is None:
                    record = (
                        session.query(ShortTermMemory)
                        .filter(
                            ShortTermMemory.memory_id == memory_id,
                            ShortTermMemory.namespace == self.namespace,
                        )
                        .one_or_none()
                    )

                if record is None:
                    raise MemoriaError("Timestamp unavailable for recomputation")

                timestamp: datetime | None
                extraction_ts = getattr(record, "extraction_timestamp", None)
                if extraction_ts is not None:
                    timestamp = extraction_ts
                else:
                    timestamp = getattr(record, "timestamp", None)

                if timestamp is None:
                    raise MemoriaError("Timestamp unavailable for recomputation")

                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)

                now = datetime.now(timezone.utc)
                day_delta = (timestamp.date() - now.date()).days
                x_coord = float(day_delta)

                record.x_coord = x_coord
                if getattr(record, "timestamp", None) is None and extraction_ts is not None:
                    record.timestamp = extraction_ts

                session.query(SpatialMetadata).filter(
                    SpatialMetadata.memory_id == memory_id,
                    SpatialMetadata.namespace == self.namespace,
                ).update({SpatialMetadata.x: x_coord})
                session.commit()

                return float(x_coord)
        except Exception as e:
            logger.error(f"Failed to recompute x coordinate: {e}")
            raise
