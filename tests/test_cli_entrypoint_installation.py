from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def test_cli_help_from_built_wheel(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]

    if find_spec("setuptools") is None:  # pragma: no cover - environment guard
        pytest.skip("setuptools is required to build wheels for this test")

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()

    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(project_root),
            "-w",
            str(wheel_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    wheels = sorted(wheel_dir.glob("memoriasdk-*.whl"))
    assert wheels, f"No wheel produced: {build.stdout}\n{build.stderr}"
    wheel_path = wheels[-1]

    venv_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    python_bin = _venv_python(venv_dir)

    subprocess.run(
        [str(python_bin), "-m", "pip", "install", str(wheel_path)],
        check=True,
    )

    result = subprocess.run(
        [str(python_bin), "-m", "memoria.cli", "build-clusters", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "usage: memoria build-clusters" in result.stdout
