"""Tests for the ``memoria.cli`` workspace subcommands."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from unittest import mock

import pytest

from memoria import cli
from memoria.utils.exceptions import MemoriaError

CliArgs = Sequence[str]


def _invoke_cli(args: CliArgs) -> int:
    """Invoke the CLI entrypoint in-process and return its exit code."""

    return cli.main(list(args))


def _json_stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


@pytest.fixture()
def memoria_loader(monkeypatch: pytest.MonkeyPatch) -> tuple[mock.Mock, mock.Mock]:
    """Provide a stub ``Memoria`` instance for CLI invocations."""

    memoria_mock = mock.Mock()
    loader = mock.Mock(return_value=memoria_mock)
    monkeypatch.setattr(cli, "_load_memoria_instance_for_cli", loader)
    return memoria_mock, loader


def test_workspace_list_outputs_json_and_membership_flag(
    memoria_loader: tuple[mock.Mock, mock.Mock],
    capsys: pytest.CaptureFixture[str],
) -> None:
    memoria_mock, loader = memoria_loader
    memoria_mock.list_workspaces.return_value = [
        {"workspace_id": "alpha", "display_name": "Alpha"}
    ]
    memoria_mock.get_active_workspace.return_value = "alpha"

    exit_code = _invoke_cli(
        ["workspaces", "list", "--workspace-id", "alpha", "--include-members"]
    )

    assert exit_code == 0
    loader.assert_called_once_with(team_id="alpha")
    memoria_mock.list_workspaces.assert_called_once_with(include_members=True)

    payload = _json_stdout(capsys)
    assert payload["active_workspace"] == "alpha"
    assert payload["workspaces"] == memoria_mock.list_workspaces.return_value


def test_workspace_create_with_activation_and_guest(
    memoria_loader: tuple[mock.Mock, mock.Mock],
    capsys: pytest.CaptureFixture[str],
) -> None:
    memoria_mock, loader = memoria_loader
    memoria_mock.register_workspace.return_value = {"workspace_id": "beta"}
    memoria_mock.get_active_workspace.return_value = "beta"

    exit_code = _invoke_cli(
        [
            "workspaces",
            "create",
            "beta",
            "--namespace",
            "team",
            "--display-name",
            "Beta",
            "--member",
            "alice",
            "--admin",
            "bob",
            "--share-by-default",
            "--metadata",
            '{"topic": "notes"}',
            "--include-members",
            "--activate",
            "--allow-guest",
        ]
    )

    assert exit_code == 0
    loader.assert_called_once_with()
    memoria_mock.register_workspace.assert_called_once_with(
        "beta",
        namespace="team",
        display_name="Beta",
        members=["alice"],
        admins=["bob"],
        share_by_default=True,
        metadata={"topic": "notes"},
        include_members=True,
    )
    memoria_mock.set_active_workspace.assert_called_once_with(
        "beta", enforce_membership=False
    )

    payload = _json_stdout(capsys)
    assert payload["workspace"] == {"workspace_id": "beta"}
    assert payload["active_workspace"] == "beta"


@pytest.mark.parametrize(
    "flags,expected_enforce",
    [([], True), (["--allow-guest"], False)],
)
def test_workspace_switch_membership_enforcement(
    memoria_loader: tuple[mock.Mock, mock.Mock],
    capsys: pytest.CaptureFixture[str],
    flags: Iterable[str],
    expected_enforce: bool,
) -> None:
    memoria_mock, _ = memoria_loader
    memoria_mock.get_active_workspace.return_value = "gamma"

    exit_code = _invoke_cli(["workspaces", "switch", "gamma", *flags])

    assert exit_code == 0
    memoria_mock.set_active_workspace.assert_called_once_with(
        "gamma", enforce_membership=expected_enforce
    )

    payload = _json_stdout(capsys)
    assert payload == {"active_workspace": "gamma"}


def test_workspace_create_invalid_metadata(
    memoria_loader: tuple[mock.Mock, mock.Mock],
    capsys: pytest.CaptureFixture[str],
) -> None:
    memoria_mock, _ = memoria_loader

    exit_code = _invoke_cli(["workspaces", "create", "delta", "--metadata", "not-json"])

    assert exit_code == 1
    memoria_mock.register_workspace.assert_not_called()

    captured = capsys.readouterr()
    assert "Invalid metadata JSON" in captured.err
    assert captured.out == ""


def test_workspace_switch_membership_error(
    memoria_loader: tuple[mock.Mock, mock.Mock],
    capsys: pytest.CaptureFixture[str],
) -> None:
    memoria_mock, _ = memoria_loader
    memoria_mock.set_active_workspace.side_effect = MemoriaError("Not a member")

    exit_code = _invoke_cli(["workspaces", "switch", "omega"])

    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "Not a member" in captured.err
    assert captured.out == ""

    memoria_mock.set_active_workspace.assert_called_once_with(
        "omega", enforce_membership=True
    )
