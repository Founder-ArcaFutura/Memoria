"""Utilities for managing team and workspace membership metadata."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..utils.exceptions import MemoriaError


@dataclass
class TeamSpace:
    """Definition of a collaborative namespace shared across multiple users."""

    team_id: str
    namespace: str
    display_name: str | None = None
    share_by_default: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    members: set[str] = field(default_factory=set)
    admins: set[str] = field(default_factory=set)
    agent_members: set[str] = field(default_factory=set)
    agent_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    def iter_members(self, include_admins: bool = True) -> set[str]:
        """Return a copy of the member identifiers for this team."""

        people: set[str] = {member for member in self.members if member}
        if include_admins:
            people.update({admin for admin in self.admins if admin})
        people.update({agent for agent in self.agent_members if agent})
        return people

    def is_member(self, user_id: str | None) -> bool:
        """Return whether ``user_id`` is part of the team (members or admins)."""

        if user_id is None:
            return False
        if user_id in self.admins:
            return True
        if user_id in self.agent_members:
            return True
        return user_id in self.members

    def to_dict(self, *, include_members: bool = True) -> dict[str, Any]:
        """Serialize the team definition for API responses."""

        payload: dict[str, Any] = {
            "team_id": self.team_id,
            "namespace": self.namespace,
            "display_name": self.display_name,
            "share_by_default": self.share_by_default,
            "metadata": copy.deepcopy(self.metadata),
        }
        if include_members:
            payload["members"] = sorted(self.members)
            payload["admins"] = sorted(self.admins)
            payload["agents"] = sorted(self.agent_members)
            if self.agent_profiles:
                payload["agent_profiles"] = copy.deepcopy(self.agent_profiles)
        return payload


@dataclass
class WorkspaceContext:
    """Representation of a workspace scoped to storage interactions."""

    workspace_id: str
    team_id: str | None = None
    name: str | None = None
    slug: str | None = None
    description: str | None = None
    owner_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    members: set[str] = field(default_factory=set)
    admins: set[str] = field(default_factory=set)
    agents: set[str] = field(default_factory=set)
    agent_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    def iter_members(self, include_admins: bool = True) -> set[str]:
        """Return identifiers for members and optionally admins."""

        people: set[str] = {member for member in self.members if member}
        if include_admins:
            people.update({admin for admin in self.admins if admin})
        people.update({agent for agent in self.agents if agent})
        return people

    def is_member(self, user_id: str | None) -> bool:
        """Return whether ``user_id`` belongs to this workspace."""

        if user_id is None:
            return False
        if user_id in self.admins:
            return True
        if user_id in self.agents:
            return True
        return user_id in self.members

    def to_metadata(self) -> dict[str, Any]:
        """Serialise workspace metadata for sync payloads."""

        payload: dict[str, Any] = {"workspace_id": self.workspace_id}
        if self.team_id:
            payload["team_id"] = self.team_id
        if self.name:
            payload["name"] = self.name
        if self.slug:
            payload["slug"] = self.slug
        if self.description:
            payload["description"] = self.description
        if self.owner_id:
            payload["owner_id"] = self.owner_id
        if self.metadata:
            payload["metadata"] = copy.deepcopy(self.metadata)
        if self.admins:
            payload["admins"] = sorted(self.admins)
        if self.members:
            payload["members"] = sorted(self.members)
        if self.agents:
            payload["agents"] = sorted(self.agents)
        if self.agent_profiles:
            payload["agent_profiles"] = copy.deepcopy(self.agent_profiles)
        return payload


class TeamSpaceCache:
    """In-memory cache tracking team membership and namespaces."""

    def __init__(self) -> None:
        self._spaces: dict[str, TeamSpace] = {}
        self._member_index: dict[str, set[str]] = {}

    def _rebuild_member_index(self) -> None:
        index: dict[str, set[str]] = {}
        for team_id, space in self._spaces.items():
            for member in space.iter_members():
                index.setdefault(member, set()).add(team_id)
        self._member_index = index

    def set(self, space: TeamSpace) -> TeamSpace:
        """Store ``space`` in the cache and return a safe copy."""

        self._spaces[space.team_id] = copy.deepcopy(space)
        self._rebuild_member_index()
        return copy.deepcopy(self._spaces[space.team_id])

    def update_members(
        self,
        team_id: str,
        *,
        members: Iterable[str] | None = None,
        admins: Iterable[str] | None = None,
        agents: Iterable[str] | None = None,
        agent_profiles: Mapping[str, dict[str, Any]] | None = None,
    ) -> TeamSpace:
        space = self._spaces.get(team_id)
        if space is None:
            raise MemoriaError(f"Unknown team: {team_id}")
        if members is not None:
            space.members = set(members)
        if admins is not None:
            space.admins = set(admins)
        if agents is not None:
            space.agent_members = set(agents)
        if agent_profiles is not None:
            space.agent_profiles = {
                key: dict(value) for key, value in agent_profiles.items()
            }
        self._rebuild_member_index()
        return copy.deepcopy(space)

    def add_members(
        self, team_id: str, members: Iterable[str], *, as_admin: bool = False
    ) -> TeamSpace:
        space = self._spaces.get(team_id)
        if space is None:
            raise MemoriaError(f"Unknown team: {team_id}")
        additions = set(members)
        if as_admin:
            space.admins.update(additions)
        else:
            space.members.update(additions)
        self._rebuild_member_index()
        return copy.deepcopy(space)

    def add_agents(
        self,
        team_id: str,
        agents: Iterable[str],
        *,
        profiles: Mapping[str, dict[str, Any]] | None = None,
    ) -> TeamSpace:
        space = self._spaces.get(team_id)
        if space is None:
            raise MemoriaError(f"Unknown team: {team_id}")
        additions = set(agents)
        space.agent_members.update(additions)
        if profiles:
            for key, value in profiles.items():
                space.agent_profiles[key] = dict(value)
        self._rebuild_member_index()
        return copy.deepcopy(space)

    def remove_member(self, team_id: str, user_id: str) -> TeamSpace:
        space = self._spaces.get(team_id)
        if space is None:
            raise MemoriaError(f"Unknown team: {team_id}")
        space.members.discard(user_id)
        space.admins.discard(user_id)
        space.agent_members.discard(user_id)
        if user_id in space.agent_profiles:
            space.agent_profiles.pop(user_id, None)
        self._rebuild_member_index()
        return copy.deepcopy(space)

    def get(self, team_id: str) -> TeamSpace | None:
        space = self._spaces.get(team_id)
        return copy.deepcopy(space) if space is not None else None

    def list(self) -> list[TeamSpace]:
        return [copy.deepcopy(space) for space in self._spaces.values()]

    def user_has_access(self, team_id: str, user_id: str | None) -> bool:
        space = self._spaces.get(team_id)
        if space is None:
            return False
        return space.is_member(user_id)

    def require_access(
        self,
        team_id: str,
        user_id: str | None,
        *,
        enforce_membership: bool = True,
    ) -> TeamSpace:
        space = self._spaces.get(team_id)
        if space is None:
            raise MemoriaError(f"Unknown team: {team_id}")
        if enforce_membership and not space.is_member(user_id):
            raise MemoriaError(
                f"User '{user_id}' is not permitted to access team '{team_id}'"
            )
        return copy.deepcopy(space)

    def team_ids_for_user(self, user_id: str) -> set[str]:
        return set(self._member_index.get(user_id, set()))


class WorkspaceCache:
    """Cache encapsulating workspace membership checks."""

    def __init__(self) -> None:
        self._contexts: dict[str, WorkspaceContext] = {}
        self._member_index: dict[str, set[str]] = {}

    def _rebuild_member_index(self) -> None:
        index: dict[str, set[str]] = {}
        for context in self._contexts.values():
            for member in context.iter_members():
                index.setdefault(member, set()).add(context.workspace_id)
        self._member_index = index

    def set(self, context: WorkspaceContext) -> WorkspaceContext:
        self._contexts[context.workspace_id] = copy.deepcopy(context)
        self._rebuild_member_index()
        return copy.deepcopy(self._contexts[context.workspace_id])

    def get(self, workspace_id: str) -> WorkspaceContext | None:
        context = self._contexts.get(workspace_id)
        return copy.deepcopy(context) if context is not None else None

    def list(self) -> list[WorkspaceContext]:
        return [copy.deepcopy(context) for context in self._contexts.values()]

    def require_access(
        self,
        workspace_id: str,
        user_id: str | None,
        *,
        enforce_membership: bool = True,
    ) -> WorkspaceContext:
        context = self._contexts.get(workspace_id)
        if context is None:
            raise MemoriaError(f"Unknown workspace: {workspace_id}")
        if enforce_membership and not context.is_member(user_id):
            raise MemoriaError(
                f"User '{user_id}' is not permitted to access workspace '{workspace_id}'"
            )
        return copy.deepcopy(context)

    def workspace_ids_for_user(self, user_id: str) -> set[str]:
        return set(self._member_index.get(user_id, set()))


def team_user_has_access(
    cache: TeamSpaceCache, team_id: str, user_id: str | None
) -> bool:
    """Return whether ``user_id`` has access to the given ``team_id``."""

    return cache.user_has_access(team_id, user_id)


def ensure_team_access(
    cache: TeamSpaceCache,
    team_id: str,
    user_id: str | None,
    *,
    enforce_membership: bool = True,
) -> TeamSpace:
    """Validate access for ``user_id`` and return the cached ``TeamSpace``."""

    return cache.require_access(team_id, user_id, enforce_membership=enforce_membership)


def workspace_user_has_access(
    cache: WorkspaceCache, workspace_id: str, user_id: str | None
) -> bool:
    """Return whether ``user_id`` may access the given workspace."""

    context = cache.get(workspace_id)
    if context is None:
        return False
    return context.is_member(user_id)


def ensure_workspace_access(
    cache: WorkspaceCache,
    workspace_id: str,
    user_id: str | None,
    *,
    enforce_membership: bool = True,
) -> WorkspaceContext:
    """Validate workspace access and return the cached context."""

    return cache.require_access(
        workspace_id, user_id, enforce_membership=enforce_membership
    )
