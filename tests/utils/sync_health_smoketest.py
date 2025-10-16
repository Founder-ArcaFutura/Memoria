"""Lightweight sync health validator for multi-region topologies.

Run inside an environment configured for Memoria. The script checks that required
configuration is present and that network endpoints accept TCP connections.
"""

from __future__ import annotations

import argparse
import os
import socket
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older runtimes
    import tomli as tomllib  # type: ignore[no-redef]


CHECK_ENV_KEYS = (
    "MEMORIA_MEMORY__NAMESPACE",
    "MEMORIA_SYNC__BACKEND",
    "MEMORIA_SYNC__CONNECTION_URL",
)


def _load_toml(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _lookup(config: dict[str, Any], path: Iterable[str]) -> Any:
    current: Any = config
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _resolve_value(
    env_var: str,
    config: dict[str, Any],
    *,
    path: Iterable[str],
) -> Any:
    if env_var in os.environ:
        return os.environ[env_var]
    return _lookup(config, path)


def _check_connection(name: str, url: str, timeout: float) -> tuple[bool, str]:
    if not url:
        return False, f"missing connection URL for {name}"
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port
    if host is None or port is None:
        return False, f"unable to parse host/port from {url!r}"
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"{name} reachable at {host}:{port}"
    except OSError as exc:
        return False, f"{name} unreachable at {host}:{port} ({exc})"


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def run_checks(args: argparse.Namespace) -> int:
    config = _load_toml(Path(args.config) if args.config else Path("memoria.toml"))
    missing_env = [key for key in CHECK_ENV_KEYS if key not in os.environ]
    for key in missing_env:
        value = _lookup(
            config,
            (
                ["sync", "connection_url"]
                if key == "MEMORIA_SYNC__CONNECTION_URL"
                else ["memory", "namespace"]
            ),
        )
        if value:
            continue
        print(f"[WARN] {key} not exported and no fallback found in memoria.toml")

    namespace = _resolve_value(
        "MEMORIA_MEMORY__NAMESPACE", config, path=("memory", "namespace")
    )
    if not namespace:
        print(
            "[ERROR] Namespace is not configured; sync coordinator will drop inbound events."
        )
        return 1
    print(f"[OK] Namespace configured: {namespace}")

    backend = (
        os.environ.get("MEMORIA_SYNC__BACKEND")
        or _lookup(config, ("sync", "backend"))
        or "none"
    ).lower()
    if backend not in {"redis", "postgres", "postgresql"}:
        print(f"[ERROR] Unsupported sync backend '{backend}'.")
        return 1
    print(f"[OK] Sync backend declared: {backend}")

    connection_url = _resolve_value(
        "MEMORIA_SYNC__CONNECTION_URL", config, path=("sync", "connection_url")
    )
    ok, message = _check_connection(
        args.topology, str(connection_url or ""), timeout=args.timeout
    )
    status = "[OK]" if ok else "[ERROR]"
    print(f"{status} {message}")
    if not ok:
        return 1

    floor = _resolve_value(
        "MEMORIA_SYNC__PRIVACY_FLOOR", config, path=("sync", "privacy_floor")
    )
    ceiling = _resolve_value(
        "MEMORIA_SYNC__PRIVACY_CEILING", config, path=("sync", "privacy_ceiling")
    )
    floor_val = _coerce_float(floor)
    ceiling_val = _coerce_float(ceiling)
    if floor_val is None:
        print(
            "[WARN] privacy_floor not set; defaulting to broadcast all Y-axis values."
        )
    else:
        print(f"[OK] privacy_floor set to {floor_val}")
    if ceiling_val is None:
        print(
            "[WARN] privacy_ceiling not set; values above floor will replicate without an upper bound."
        )
    elif floor_val is not None and ceiling_val < floor_val:
        print(
            "[ERROR] privacy_ceiling lower than privacy_floor; adjust configuration before failover drills."
        )
        return 1
    elif ceiling_val is not None:
        print(f"[OK] privacy_ceiling set to {ceiling_val}")

    if args.topology == "redis" and not os.environ.get(
        "MEMORIA_SYNC__REALTIME_REPLICATION"
    ):
        realtime = _lookup(config, ("sync", "realtime_replication"))
        if not realtime:
            print(
                "[WARN] realtime_replication disabled; Redis bridge will only fan-out metadata."
            )
        else:
            print("[OK] realtime_replication enabled for Redis bridge")

    if args.topology == "postgres" and not os.environ.get("MEMORIA_SYNC__TABLE"):
        table = _lookup(config, ("sync", "table"))
        if not table:
            print(
                "[WARN] No sync table configured; Memoria will fall back to defaults (memoria_sync_events)."
            )
        else:
            print(f"[OK] Sync table configured: {table}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Memoria sync connectivity and privacy guardrails."
    )
    parser.add_argument(
        "--topology",
        choices={"redis", "postgres"},
        required=True,
        help="Topology profile to validate (matches playbook sections).",
    )
    parser.add_argument(
        "--config",
        help="Path to memoria.toml. Defaults to ./memoria.toml if present.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Seconds to wait when probing Redis/PostgreSQL endpoints.",
    )
    args = parser.parse_args()
    raise SystemExit(run_checks(args))


if __name__ == "__main__":
    main()
