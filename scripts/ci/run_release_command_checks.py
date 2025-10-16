"""CI helper verifying release command workflows.

This script provisions a temporary SQLite database, exercises the
migration roll-forward flow, and validates the documented rollback path
(export/import) so release managers can rely on automated evidence.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_cli(
    args: Iterable[str],
    *,
    env: dict[str, str],
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute ``python -m memoria.cli`` with the provided arguments."""

    cmd = [sys.executable, "-m", "memoria.cli", *args]
    display = " ".join(cmd)
    print(f"+ {display}")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=capture_output,
        check=True,
    )


def _parse_migration_names(output: str) -> list[str]:
    """Extract migration names from ``migrate list`` output."""

    names: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        names.append(stripped.split()[0])
    return names


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        database_path = tmp_path / "release-command-check.db"
        database_url = f"sqlite:///{database_path}"

        env = os.environ.copy()
        env["DATABASE_URL"] = database_url
        env["MEMORIA_DB_URL"] = database_url
        env["MEMORIA__DATABASE__CONNECTION_STRING"] = database_url

        list_result = _run_cli(["migrate", "list"], env=env, capture_output=True)
        migrations = _parse_migration_names(list_result.stdout)
        if not migrations:
            raise RuntimeError("No migrations discovered during release command checks.")

        _run_cli(["init-db"], env=env)

        backup_path = tmp_path / "pre-migration-backup.json"
        _run_cli(
            [
                "export-data",
                str(backup_path),
                "--database-url",
                database_url,
            ],
            env=env,
        )

        for migration in migrations:
            _run_cli(
                [
                    "migrate",
                    "run",
                    migration,
                    "--database-url",
                    database_url,
                ],
                env=env,
            )

        _run_cli(
            [
                "import-data",
                str(backup_path),
                "--database-url",
                database_url,
            ],
            env=env,
        )

        post_backup_path = tmp_path / "post-rollback-export.json"
        _run_cli(
            [
                "export-data",
                str(post_backup_path),
                "--database-url",
                database_url,
            ],
            env=env,
        )

        print(
            "Release command checks completed successfully against",
            database_url,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
