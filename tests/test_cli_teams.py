"""Tests for the ``memoria cli teams`` subcommands."""

from __future__ import annotations

import json
from typing import Any

import pytest

from memoria import cli


class DummyMemoria:
    """Simple stand-in for :class:`memoria.core.memory.Memoria`."""

    def __init__(self) -> None:
        self.list_called_with: bool | None = None
        self.register_payload: dict[str, Any] | None = None
        self.activation_calls: list[tuple[str, bool]] = []
        self.active_team: Any = None
        self.cleared: bool = False

    def list_team_spaces(self, include_members: bool = False):
        self.list_called_with = include_members
        return [{"team_id": "alpha", "include_members": include_members}]

    def register_team_space(
        self,
        team_id: str,
        *,
        namespace: str | None = None,
        display_name: str | None = None,
        members: list[str] | None = None,
        admins: list[str] | None = None,
        share_by_default: bool | None = None,
        metadata: dict[str, Any] | None = None,
        include_members: bool = False,
    ):
        self.register_payload = {
            "team_id": team_id,
            "namespace": namespace,
            "display_name": display_name,
            "members": members,
            "admins": admins,
            "share_by_default": share_by_default,
            "metadata": metadata,
            "include_members": include_members,
        }
        return {"team_id": team_id, "namespace": namespace, "metadata": metadata}

    def set_active_team(self, team_id: str, *, enforce_membership: bool = True) -> None:
        self.activation_calls.append((team_id, enforce_membership))
        self.active_team = {
            "team_id": team_id,
            "enforce_membership": enforce_membership,
        }

    def get_active_team(self):
        return self.active_team

    def clear_active_team(self) -> None:
        self.cleared = True
        self.active_team = None


@pytest.fixture()
def dummy_memoria(monkeypatch: pytest.MonkeyPatch) -> DummyMemoria:
    client = DummyMemoria()

    def _loader(team_id: str | None = None) -> DummyMemoria:
        client.last_loaded_team = team_id  # type: ignore[attr-defined]
        return client

    monkeypatch.setattr(cli, "_load_memoria_instance_for_cli", _loader)
    return client


def test_teams_list_invokes_memoria(
    dummy_memoria: DummyMemoria, capsys: pytest.CaptureFixture[str]
) -> None:
    """Listing teams should call the Memoria client and print JSON."""

    exit_code = cli.main(["teams", "list", "--include-members", "--team-id", "beta"])

    assert exit_code == 0
    assert dummy_memoria.last_loaded_team == "beta"
    assert dummy_memoria.list_called_with is True

    output = json.loads(capsys.readouterr().out)
    assert output["active_team"] is None
    assert output["teams"][0]["team_id"] == "alpha"


def test_teams_create_registers_members(
    dummy_memoria: DummyMemoria, capsys: pytest.CaptureFixture[str]
) -> None:
    """Creating or updating a team should forward membership arguments."""

    metadata = {"region": "eu"}
    exit_code = cli.main(
        [
            "teams",
            "create",
            "team-42",
            "--namespace",
            "project",
            "--display-name",
            "Project Space",
            "--member",
            "alice",
            "--member",
            "bob",
            "--admin",
            "carol",
            "--share-by-default",
            "--metadata",
            json.dumps(metadata),
            "--include-members",
            "--activate",
            "--allow-guest",
        ]
    )

    assert exit_code == 0
    assert dummy_memoria.register_payload == {
        "team_id": "team-42",
        "namespace": "project",
        "display_name": "Project Space",
        "members": ["alice", "bob"],
        "admins": ["carol"],
        "share_by_default": True,
        "metadata": metadata,
        "include_members": True,
    }
    assert dummy_memoria.activation_calls == [("team-42", False)]

    payload = json.loads(capsys.readouterr().out)
    assert payload["team"]["team_id"] == "team-42"
    assert payload["active_team"]["team_id"] == "team-42"


def test_teams_activate_enforces_membership(
    dummy_memoria: DummyMemoria, capsys: pytest.CaptureFixture[str]
) -> None:
    """Activation should respect the ``--allow-guest`` flag."""

    exit_code = cli.main(["teams", "activate", "gamma"])

    assert exit_code == 0
    assert dummy_memoria.activation_calls == [("gamma", True)]

    payload = json.loads(capsys.readouterr().out)
    assert payload["active_team"]["team_id"] == "gamma"


def test_teams_clear_deactivates(
    dummy_memoria: DummyMemoria, capsys: pytest.CaptureFixture[str]
) -> None:
    """Clearing the team should call ``clear_active_team`` on the client."""

    exit_code = cli.main(["teams", "clear"])

    assert exit_code == 0
    assert dummy_memoria.cleared is True

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"active_team": None}
