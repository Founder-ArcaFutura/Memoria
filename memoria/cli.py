"""Command line interface for common Memoria workflows."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import queue
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from memoria.cli_support.benchmark_runner import (
    BenchmarkConfigurationError,
    BenchmarkRunner,
    BenchmarkSuite,
    write_benchmark_ndjson,
)
from memoria.cli_support.evaluation_scaffold import (
    SuiteScaffoldConfig,
    parse_retrieval_queries,
    write_suite_scaffold,
)
from memoria.cli_support.governance import (
    RotationReport,
    build_roster_verification,
    build_rotation_report,
)
from memoria.cli_support.policy_tooling import (
    apply_policies,
    lint_policies,
    load_policies_for_cli,
    simulate_policies,
)
from memoria.config.manager import ConfigManager
from memoria.policy.enforcement import (
    PolicyEnforcementEngine,
    PolicyMetricsCollector,
    PolicyMetricsSnapshot,
)
from memoria.policy.utils import PolicyConfigurationError
from memoria.utils.exceptions import ConfigurationError, MemoriaError


@dataclass(slots=True)
class CapabilityStatus:
    """Normalised view of optional provider capability checks."""

    key: str
    label: str
    enabled: bool
    installed: bool
    module: str
    extra: str | None
    message: str
    resolution: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


_STRUCTURAL_MIGRATION_NAMES: tuple[str, ...] = (
    "add_cluster_centroids",
    "add_cluster_member_counts",
    "add_cluster_member_token_columns",
    "add_cluster_token_columns",
    "add_memory_embeddings",
    "add_policy_artifacts",
    "add_spatial_namespace",
    "add_team_support",
    "add_timestamp_columns",
    "add_workspace_support",
    "remove_manual_x_override",
)

_STRUCTURAL_MIGRATION_STATUS: dict[str, Any] = {
    "last_run": None,
    "results": [],
    "skipped": False,
    "error": None,
    "database_url": None,
    "auto_enabled": True,
}

_STRUCTURAL_MIGRATION_CHECK_RAN = False
_PROVIDER_CAPABILITIES: list[CapabilityStatus] = []
_CAPABILITY_CHECK_RAN = False


def get_structural_migration_status() -> dict[str, Any]:
    """Return the latest structural migration status snapshot."""

    return dict(_STRUCTURAL_MIGRATION_STATUS)


def get_provider_capability_status() -> list[dict[str, Any]]:
    """Return the cached provider capability status list."""

    return [status.to_dict() for status in _PROVIDER_CAPABILITIES]


def _timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _format_status(status: CapabilityStatus) -> str:
    if status.installed or not status.enabled:
        return f"✅ {status.label}"
    return f"⚠️ {status.label}: {status.message} {status.resolution}".strip()


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        return False
    except Exception:
        return True
    return True


def _resolve_capability_status(settings: Any | None) -> list[CapabilityStatus]:
    agent_settings = getattr(settings, "agents", None) if settings is not None else None
    sync_settings = getattr(settings, "sync", None) if settings is not None else None

    openai_enabled = True
    gemini_enabled = bool(
        getattr(agent_settings, "gemini_api_key", None)
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    sync_enabled = bool(getattr(sync_settings, "enabled", False))
    sync_backend = getattr(sync_settings, "backend", "none") if sync_settings else "none"
    if hasattr(sync_backend, "value"):
        sync_backend = sync_backend.value

    statuses: list[CapabilityStatus] = []

    statuses.append(
        CapabilityStatus(
            key="provider:openai",
            label="OpenAI provider",
            enabled=openai_enabled,
            installed=_module_available("openai"),
            module="openai",
            extra=None,
            message="Install the OpenAI Python client to enable default provider calls.",
            resolution="Run 'pip install openai' or disable OpenAI usage in settings.",
        )
    )

    statuses.append(
        CapabilityStatus(
            key="provider:gemini",
            label="Gemini provider",
            enabled=gemini_enabled,
            installed=_module_available("google.generativeai"),
            module="google-generativeai",
            extra="integrations",
            message="Gemini API support requires the integrations extra.",
            resolution=(
                "Run 'pip install -e "".[integrations]""' inside the Memoria repo or remove the Gemini API key."
            ),
        )
    )

    redis_required = sync_enabled and str(sync_backend).lower() == "redis"
    statuses.append(
        CapabilityStatus(
            key="sync:redis",
            label="Redis sync backend",
            enabled=redis_required,
            installed=_module_available("redis"),
            module="redis",
            extra="sync",
            message="Redis-based sync requires the sync extra.",
            resolution=(
                "Run 'pip install -e "".[sync]""' inside the Memoria repo or switch the sync backend to 'none'."
            ),
        )
    )

    postgres_required = sync_enabled and str(sync_backend).lower() == "postgres"
    statuses.append(
        CapabilityStatus(
            key="sync:postgres",
            label="PostgreSQL sync backend",
            enabled=postgres_required,
            installed=_module_available("psycopg") or _module_available("psycopg2"),
            module="psycopg",
            extra="sync",
            message="PostgreSQL sync requires a psycopg driver.",
            resolution="Run 'pip install psycopg[binary]' or configure a different sync backend.",
        )
    )

    return statuses


def check_provider_capabilities(
    settings: Any | None,
    *,
    interactive: bool,
    stream: Any | None = None,
) -> list[CapabilityStatus]:
    """Evaluate provider capability status and emit CLI guidance."""

    global _PROVIDER_CAPABILITIES, _CAPABILITY_CHECK_RAN

    statuses = _resolve_capability_status(settings)
    _PROVIDER_CAPABILITIES = statuses

    if stream is not None and (interactive or not _CAPABILITY_CHECK_RAN):
        unresolved = [status for status in statuses if status.enabled and not status.installed]
        if unresolved:
            print("", file=stream)
            print("⚠️  Optional provider dependencies are missing:", file=stream)
            for status in unresolved:
                print(f"   - {_format_status(status)}", file=stream)
            print("", file=stream)
        elif not _CAPABILITY_CHECK_RAN:
            print("✅ Optional provider dependencies satisfied.", file=stream)

    _CAPABILITY_CHECK_RAN = True
    return statuses


def run_structural_migrations(
    database_url: str | None,
    *,
    auto_enabled: bool = True,
    stream: Any | None = None,
) -> dict[str, Any]:
    """Execute bundled structural migrations for the configured database."""

    global _STRUCTURAL_MIGRATION_STATUS, _STRUCTURAL_MIGRATION_CHECK_RAN

    previous_ran = _STRUCTURAL_MIGRATION_CHECK_RAN

    status: dict[str, Any] = {
        "last_run": _timestamp(),
        "results": [],
        "skipped": False,
        "error": None,
        "database_url": database_url,
        "auto_enabled": auto_enabled,
    }

    if not database_url:
        _STRUCTURAL_MIGRATION_STATUS = status
        return status

    if not auto_enabled:
        status["skipped"] = True
        status["message"] = "Automatic migrations disabled via configuration."
        if stream is not None and not previous_ran:
            print(
                "⚠️  Automatic structural migrations are disabled (database.migration_auto=false).",
                file=stream,
            )
        _STRUCTURAL_MIGRATION_STATUS = status
        return status

    from memoria.cli_support.migration_runner import (
        MigrationExecutionError,
        discover_migrations,
        run_migration_script,
    )

    discovered = {
        migration.name: migration
        for migration in discover_migrations(include_archived=False)
        if migration.name in _STRUCTURAL_MIGRATION_NAMES
    }

    ordered = [discovered[name] for name in _STRUCTURAL_MIGRATION_NAMES if name in discovered]

    if not ordered:
        status["message"] = "No structural migrations discovered."
        _STRUCTURAL_MIGRATION_STATUS = status
        return status

    for migration in ordered:
        entry = {"name": migration.name, "status": "skipped", "detail": ""}
        try:
            run_migration_script(migration, database_url=database_url)
            entry["status"] = "applied"
        except MigrationExecutionError as exc:
            entry["status"] = "error"
            entry["detail"] = str(exc)
            status["error"] = str(exc)
            status["results"].append(entry)
            break
        except Exception as exc:  # pragma: no cover - defensive guard
            entry["status"] = "error"
            entry["detail"] = str(exc)
            status["error"] = str(exc)
            status["results"].append(entry)
            break
        else:
            status["results"].append(entry)

    if stream is not None and (not previous_ran or status.get("error")):
        if status.get("error"):
            print("❌ Structural migrations encountered an error:", file=stream)
            print(f"   {status['error']}", file=stream)
        else:
            print("✅ Structural migrations executed.", file=stream)
            for entry in status["results"]:
                print(f"   - {entry['name']}: {entry['status']}", file=stream)

    _STRUCTURAL_MIGRATION_STATUS = status
    _STRUCTURAL_MIGRATION_CHECK_RAN = True
    return status


def _handle_migrate_list(args: argparse.Namespace) -> int:
    """List available migration scripts."""

    from memoria.cli_support.migration_runner import discover_migrations

    migrations = discover_migrations(include_archived=args.include_archived)
    if not migrations:
        print("No migrations discovered.")
        return 0

    width = max(len(migration.name) for migration in migrations)
    for migration in migrations:
        desc = (migration.description or "").strip().splitlines()[0] if migration.description else ""
        suffix = " (archived)" if migration.archived else ""
        if desc:
            print(f"{migration.name:<{width}}  {desc}{suffix}")
        else:
            print(f"{migration.name:<{width}}{suffix}")
    return 0


def _handle_migrate_run(args: argparse.Namespace) -> int:
    """Execute a selected migration script."""

    from memoria.cli_support.migration_runner import (
        MigrationExecutionError,
        MigrationNotFoundError,
        get_migration,
        run_migration_script,
    )

    try:
        migration = get_migration(
            args.migration,
            include_archived=args.include_archived,
        )
    except MigrationNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except MigrationExecutionError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    database_url = _resolve_database_url(args.database_url)
    if args.dry_run:
        print(
            f"Dry run: would execute {migration.name} located at {migration.path} against {database_url}."
        )
        return 0

    try:
        run_migration_script(migration, database_url=database_url)
    except MigrationExecutionError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"Executed migration '{migration.name}' against {database_url}."
    )
    return 0


def _handle_bootstrap(args: argparse.Namespace) -> int:
    """Run the interactive bootstrap wizard."""

    from memoria.config.bootstrap import (
        BootstrapConfig,
        BootstrapError,
        run_bootstrap,
    )

    stream = sys.stderr if sys.stderr.isatty() else None

    non_interactive = bool(args.non_interactive or args.config_json)
    config: BootstrapConfig | None = None

    if args.config_json:
        try:
            with open(args.config_json, encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as exc:
            print(f"Unable to read configuration JSON: {exc}", file=sys.stderr)
            return 1
        except json.JSONDecodeError as exc:
            print(f"Configuration JSON is invalid: {exc}", file=sys.stderr)
            return 1
        try:
            config = BootstrapConfig.from_mapping(payload)
        except BootstrapError as exc:
            print(f"Invalid bootstrap configuration: {exc}", file=sys.stderr)
            return 1
    elif non_interactive:
        config = BootstrapConfig()

    try:
        run_bootstrap(
            env_path=args.env_file,
            config_path=args.config,
            force=bool(args.force or non_interactive),
            config=config,
        )
    except BootstrapError as exc:
        print(f"Bootstrap failed: {exc}", file=sys.stderr)
        return 1

    manager = ConfigManager.get_instance()
    try:
        manager.auto_load()
    except Exception:  # pragma: no cover - configuration may be partial
        pass

    settings = manager.get_settings()
    database_url = settings.get_database_url() if settings else None
    auto_enabled = True
    if settings and getattr(settings, "database", None):
        auto_enabled = bool(getattr(settings.database, "migration_auto", True))

    run_structural_migrations(database_url, auto_enabled=auto_enabled, stream=stream)
    check_provider_capabilities(settings, interactive=bool(stream), stream=stream)
    return 0


def _handle_runserver(args: argparse.Namespace) -> int:
    """Start the bundled Flask development server."""

    from memoria.api import app

    app.run(
        host=args.host,
        port=args.port,
        debug=args.reload,
        use_reloader=args.reload,
    )
    return 0


def _handle_init_db(_: argparse.Namespace) -> int:
    """Initialise the spatial database using the Flask application factory."""

    from memoria.api import init_spatial_db

    init_spatial_db()
    print("Database initialised successfully.")
    manager = ConfigManager.get_instance()
    try:
        manager.auto_load()
    except Exception:  # pragma: no cover - configuration may be partial
        settings = None
    else:
        settings = manager.get_settings()

    database_url = None
    auto_enabled = True
    if settings is not None:
        try:
            database_url = settings.get_database_url()
        except Exception:  # pragma: no cover - defensive
            database_url = None
        database_settings = getattr(settings, "database", None)
        if database_settings is not None:
            auto_enabled = bool(getattr(database_settings, "migration_auto", True))

    stream = sys.stderr if sys.stderr.isatty() else None
    run_structural_migrations(database_url, auto_enabled=auto_enabled, stream=stream)
    check_provider_capabilities(settings, interactive=bool(stream), stream=stream)
    return 0


def _handle_build_clusters(args: argparse.Namespace) -> int:
    """Build clustering indexes using the configured strategy."""

    if args.mode == "heuristic":
        from memoria.cli_support.heuristic_clusters import build_heuristic_clusters

        clusters, summary = build_heuristic_clusters(
            connection_string=args.database_url, verbose=args.verbose
        )
        print(summary)
        print(f"Generated {len(clusters)} heuristic clusters.")
        return 0

    from memoria.cli_support.index_clusters import build_index

    sources: Iterable[str] | None
    if args.sources:
        sources = list(dict.fromkeys(args.sources))
    else:
        sources = None

    clusters = build_index(sources=sources)
    print(f"Generated {len(clusters)} vector clusters.")
    return 0


def _resolve_database_url(override: str | None) -> str:
    if override:
        return override
    from memoria.config.manager import ConfigManager

    settings = ConfigManager.get_instance().get_settings()
    return settings.database.connection_string


def _handle_export_data(args: argparse.Namespace) -> int:
    """Export Memoria tables to JSON or NDJSON."""

    from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager

    connection = _resolve_database_url(args.database_url)
    manager = SQLAlchemyDatabaseManager(connection)
    try:
        destination: Path | None
        if args.destination == "-":
            destination = None
        else:
            destination = Path(args.destination)

        result = manager.export_dataset(
            destination=destination,
            format=args.format,
            tables=args.tables,
        )
    finally:
        manager.close()

    if args.destination == "-":
        sys.stdout.write(result.content)
    else:
        table_count = len(result.metadata.get("tables", []))
        row_total = sum(t["row_count"] for t in result.metadata.get("tables", []))
        print(
            f"Exported {row_total} rows across {table_count} tables to {destination}",
        )
    return 0


def _handle_import_data(args: argparse.Namespace) -> int:
    """Import data into the configured Memoria database."""

    from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager

    connection = _resolve_database_url(args.database_url)
    manager = SQLAlchemyDatabaseManager(connection)
    try:
        if args.source == "-":
            payload = sys.stdin.read()
            metadata = manager.import_dataset(
                payload,
                format=args.format,
                tables=args.tables,
                truncate=not args.no_truncate,
            )
        else:
            metadata = manager.import_dataset(
                Path(args.source),
                format=args.format,
                tables=args.tables,
                truncate=not args.no_truncate,
            )
    finally:
        manager.close()

    table_count = 0
    if isinstance(metadata, dict):
        table_count = len(metadata.get("tables", [])) or table_count
    print(
        f"Imported dataset for {table_count} tables (truncate={'no' if args.no_truncate else 'yes'})",
    )
    return 0


def _handle_assign_task_model(args: argparse.Namespace) -> int:
    """Update per-task model routing preferences."""

    manager = ConfigManager.get_instance()
    manager.auto_load()

    fallback_entries: list[str] = []
    if args.fallback:
        for item in args.fallback:
            parts = [segment.strip() for segment in item.split(",") if segment.strip()]
            fallback_entries.extend(parts)
    if args.clear_fallback:
        fallback_entries = []

    payload = {
        "provider": args.provider,
        "model": args.model,
        "fallback": fallback_entries,
    }

    manager.update_setting(f"agents.task_model_routes.{args.task}", payload)
    config_info = manager.get_config_info()
    sources = list(config_info.get("sources", []))
    destination: Path | None = None
    for source in sources:
        if not source or source == "defaults":
            continue
        if isinstance(source, str) and source.lower() == "environment":
            continue
        try:
            candidate = Path(source)
        except TypeError:
            continue
        destination = candidate
        break

    if destination is None:
        destination = Path(os.getenv("MEMORIA_CONFIG_PATH", "memoria.json"))

    format_hint = destination.suffix.lstrip(".").lower()
    if format_hint not in {"json", "yml", "yaml"}:
        format_hint = "json"

    manager.save_to_file(destination, format=format_hint)
    model_desc = args.model or "<default>"
    fallback_desc = fallback_entries if fallback_entries else "<none>"
    print(
        f"Updated task routing for '{args.task}': provider={args.provider}, model={model_desc}, fallback={fallback_desc}"
    )
    print(f"Persisted task routing update to {destination}")
    return 0


def _handle_benchmark(args: argparse.Namespace) -> int:
    try:
        suite = BenchmarkSuite.from_file(Path(args.config))
    except FileNotFoundError:
        print(f"Benchmark configuration not found: {args.config}", file=sys.stderr)
        return 1
    except BenchmarkConfigurationError as exc:
        print(f"Invalid benchmark configuration: {exc}", file=sys.stderr)
        return 1

    try:
        memoria = _load_memoria_instance_for_cli(team_id=args.team_id)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Failed to load Memoria configuration: {exc}", file=sys.stderr)
        return 1

    runner = BenchmarkRunner(
        memoria,
        suite.scenarios,
        base_namespace=args.namespace,
    )
    report = runner.run().to_dict()

    output = args.output
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "ndjson":
            write_benchmark_ndjson(destination, report)
        else:
            destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Benchmark results written to {destination}")
    else:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


def _handle_evaluation_scaffold(args: argparse.Namespace) -> int:
    suite_name = args.suite_name or args.suite_id.replace("_", " ").title()
    description = args.description or "Describe the goals of this evaluation suite."
    retrieval_queries = parse_retrieval_queries(args.retrieval_queries)

    config = SuiteScaffoldConfig(
        suite_id=args.suite_id,
        suite_name=suite_name,
        description=description,
        use_cases=args.use_cases or (),
        workspace=args.workspace,
        scenario_id=args.scenario_id,
        scenario_name=args.scenario_name,
        scenario_description=args.scenario_description,
        dataset_kind=args.dataset_kind,
        dataset_path=args.dataset_path,
        retrieval_queries=retrieval_queries,
    )

    try:
        output_path = write_suite_scaffold(
            Path(args.output), config, overwrite=args.force
        )
    except (RuntimeError, FileExistsError, ValueError) as exc:
        print(f"Failed to scaffold evaluation suite: {exc}", file=sys.stderr)
        return 1

    print(f"Created evaluation suite scaffold at {output_path}")
    return 0


def _load_memoria_instance_for_cli(team_id: str | None = None):
    """Instantiate :class:`memoria.core.memory.Memoria` using loaded configuration."""

    from memoria.config import ConfigManager
    from memoria.core.memory import Memoria, build_provider_options

    manager = ConfigManager.get_instance()
    try:
        manager.auto_load()
    except Exception:
        pass

    settings = manager.get_settings()
    stream = sys.stderr if sys.stderr.isatty() else None
    auto_enabled = True
    database_url = None
    if settings is not None:
        try:
            database_url = settings.get_database_url()
        except Exception:  # pragma: no cover - defensive
            database_url = None
        database_settings = getattr(settings, "database", None)
        if database_settings is not None:
            auto_enabled = bool(getattr(database_settings, "migration_auto", True))

    run_structural_migrations(database_url, auto_enabled=auto_enabled, stream=stream)
    check_provider_capabilities(settings, interactive=bool(stream), stream=stream)
    if settings is None:
        raise RuntimeError("Unable to load Memoria settings")

    provider_options = build_provider_options(settings)
    if team_id is not None:
        provider_options["team_id"] = team_id
    return Memoria(
        database_connect=settings.get_database_url(),
        **provider_options,
    )


def _handle_team_list(args: argparse.Namespace) -> int:
    try:
        memoria = _load_memoria_instance_for_cli(team_id=args.team_id)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Failed to load Memoria configuration: {exc}", file=sys.stderr)
        return 1

    try:
        teams = memoria.list_team_spaces(include_members=args.include_members)
    except MemoriaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    payload = {"teams": teams, "active_team": memoria.get_active_team()}
    print(json.dumps(payload, indent=2))
    return 0


def _handle_team_create(args: argparse.Namespace) -> int:
    try:
        memoria = _load_memoria_instance_for_cli()
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Failed to load Memoria configuration: {exc}", file=sys.stderr)
        return 1

    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as exc:
            print(f"Invalid metadata JSON: {exc}", file=sys.stderr)
            return 1
        if not isinstance(metadata, dict):
            print("metadata must be a JSON object", file=sys.stderr)
            return 1

    try:
        team = memoria.register_team_space(
            args.team_id,
            namespace=args.namespace,
            display_name=args.display_name,
            members=args.members or None,
            admins=args.admins or None,
            share_by_default=args.share_by_default,
            metadata=metadata,
            include_members=args.include_members,
        )
    except MemoriaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    activation_result = None
    if args.activate:
        try:
            memoria.set_active_team(
                args.team_id,
                enforce_membership=not args.allow_guest,
            )
            activation_result = memoria.get_active_team()
        except MemoriaError as exc:
            print(f"Warning: {exc}", file=sys.stderr)

    payload = {"team": team}
    if activation_result is not None:
        payload["active_team"] = activation_result
    print(json.dumps(payload, indent=2))
    return 0


def _handle_team_activate(args: argparse.Namespace) -> int:
    try:
        memoria = _load_memoria_instance_for_cli()
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Failed to load Memoria configuration: {exc}", file=sys.stderr)
        return 1

    try:
        memoria.set_active_team(
            args.team_id,
            enforce_membership=not args.allow_guest,
        )
    except MemoriaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"active_team": memoria.get_active_team()}, indent=2))
    return 0


def _handle_team_clear(_: argparse.Namespace) -> int:
    try:
        memoria = _load_memoria_instance_for_cli()
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Failed to load Memoria configuration: {exc}", file=sys.stderr)
        return 1

    memoria.clear_active_team()
    print(json.dumps({"active_team": None}, indent=2))
    return 0


def _handle_workspace_list(args: argparse.Namespace) -> int:
    try:
        memoria = _load_memoria_instance_for_cli(team_id=args.team_id)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Failed to load Memoria configuration: {exc}", file=sys.stderr)
        return 1

    try:
        workspaces = memoria.list_workspaces(include_members=args.include_members)
    except MemoriaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    payload = {"workspaces": workspaces, "active_workspace": memoria.get_active_workspace()}
    print(json.dumps(payload, indent=2))
    return 0


def _handle_workspace_create(args: argparse.Namespace) -> int:
    try:
        memoria = _load_memoria_instance_for_cli()
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Failed to load Memoria configuration: {exc}", file=sys.stderr)
        return 1

    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as exc:
            print(f"Invalid metadata JSON: {exc}", file=sys.stderr)
            return 1
        if not isinstance(metadata, dict):
            print("metadata must be a JSON object", file=sys.stderr)
            return 1

    try:
        workspace = memoria.register_workspace(
            args.workspace_id,
            namespace=args.namespace,
            display_name=args.display_name,
            members=args.members or None,
            admins=args.admins or None,
            share_by_default=args.share_by_default,
            metadata=metadata,
            include_members=args.include_members,
        )
    except MemoriaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    activation_result = None
    if args.activate:
        try:
            memoria.set_active_workspace(
                args.workspace_id,
                enforce_membership=not args.allow_guest,
            )
            activation_result = memoria.get_active_workspace()
        except MemoriaError as exc:
            print(f"Warning: {exc}", file=sys.stderr)

    payload = {"workspace": workspace}
    if activation_result is not None:
        payload["active_workspace"] = activation_result
    print(json.dumps(payload, indent=2))
    return 0


def _handle_workspace_switch(args: argparse.Namespace) -> int:
    try:
        memoria = _load_memoria_instance_for_cli()
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Failed to load Memoria configuration: {exc}", file=sys.stderr)
        return 1

    try:
        memoria.set_active_workspace(
            args.workspace_id,
            enforce_membership=not args.allow_guest,
        )
    except MemoriaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"active_workspace": memoria.get_active_workspace()}, indent=2))
    return 0


def _handle_policy_lint(args: argparse.Namespace) -> int:
    try:
        policies = load_policies_for_cli(args.source)
    except PolicyConfigurationError as exc:
        print(f"Policy validation failed: {exc}", file=sys.stderr)
        return 1

    warnings = lint_policies(policies)
    print(f"Validated {len(policies)} policy rule(s).")
    for rule in policies:
        namespaces = ", ".join(rule.namespaces or ["*"])
        action = getattr(rule.action, "value", str(rule.action))
        print(f" - {rule.name}: action={action}, namespaces={namespaces}")

    if warnings:
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
    return 0


def _handle_policy_test(args: argparse.Namespace) -> int:
    try:
        policies = load_policies_for_cli(args.source)
    except PolicyConfigurationError as exc:
        print(f"Policy validation failed: {exc}", file=sys.stderr)
        return 1

    sample_path = Path(args.samples) if args.samples else None
    try:
        reports = simulate_policies(policies, sample_path=sample_path)
    except (PolicyConfigurationError, OSError) as exc:
        print(f"Policy simulation failed: {exc}", file=sys.stderr)
        return 1

    if not reports:
        print("No policies provided; nothing to test.")
        return 0

    for report in reports:
        action = getattr(report.rule.action, "value", str(report.rule.action))
        trigger_desc = f"{report.trigger_count}/{report.total_samples}"
        print(f"{report.rule.name}: {trigger_desc} triggers (action={action})")
        if not report.hits:
            print("  (no triggers)")
            continue
        for hit in report.hits:
            identifier = hit.memory_id or f"sample#{hit.sample_index}"
            reasons = ", ".join(hit.reasons)
            print(f"  - {identifier}: {reasons}")
    return 0


def _resolve_policy_config_target(manager: ConfigManager) -> Path | None:
    """Identify a writable configuration file for persisting policy updates."""

    candidates: list[Path] = []

    env_path = os.getenv("MEMORIA_CONFIG_PATH")
    if env_path:
        candidates.append(Path(env_path))

    try:
        info = manager.get_config_info()
    except Exception:
        info = {}

    for source in info.get("sources", []):
        if not source or source in {"defaults", "environment"}:
            continue
        candidates.append(Path(source))

    candidates.extend(
        [
            Path("memoria.json"),
            Path("memoria.yaml"),
            Path("memoria.yml"),
            Path("config") / "memoria.json",
            Path("config") / "memoria.yaml",
            Path("config") / "memoria.yml",
            Path.home() / ".memoria" / "config.json",
            Path.home() / ".memoria" / "config.yaml",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path in seen:
            continue
        seen.add(path)

        parent = path.parent if str(path.parent) else Path(".")
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue

        if path.exists():
            if os.access(path, os.W_OK):
                return path
            continue

        if os.access(parent, os.W_OK):
            return path

    return None


def _handle_policy_apply(args: argparse.Namespace) -> int:
    try:
        policies = load_policies_for_cli(args.source)
    except PolicyConfigurationError as exc:
        print(f"Policy validation failed: {exc}", file=sys.stderr)
        return 1

    manager = ConfigManager.get_instance()
    try:
        manager.auto_load()
    except ConfigurationError as exc:
        print(f"warning: Failed to auto-load configuration: {exc}", file=sys.stderr)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"warning: Failed to auto-load configuration: {exc}", file=sys.stderr)

    try:
        existing = manager.get_setting("memory.retention_policy_rules", []) or []
    except ConfigurationError:
        existing = []
    except Exception:
        existing = []
    existing_count = len(existing)

    try:
        apply_policies(policies, dry_run=args.dry_run)
    except PolicyConfigurationError as exc:
        print(f"Failed to apply policies: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(
            f"Dry run complete. {len(policies)} rule(s) validated against {existing_count} currently configured.",
        )
        for rule in policies:
            namespaces = ", ".join(rule.namespaces or ["*"])
            action = getattr(rule.action, "value", str(rule.action))
            print(f" - {rule.name}: action={action}, namespaces={namespaces}")
        return 0

    target = _resolve_policy_config_target(manager)
    if target is None:
        print(
            f"Applied {len(policies)} policy rule(s) (previously {existing_count}).",
        )
        print(
            "warning: Unable to locate a writable configuration file; rules not persisted.",
            file=sys.stderr,
        )
        return 0

    format_hint = target.suffix.lower()
    save_format = "yaml" if format_hint in {".yaml", ".yml"} else "json"

    try:
        manager.save_to_file(target, format=save_format)
    except ConfigurationError as exc:
        print(f"Failed to persist policy rules: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - filesystem dependent
        print(f"Failed to persist policy rules: {exc}", file=sys.stderr)
        return 1

    print(
        f"Applied {len(policies)} policy rule(s) (previously {existing_count}). Saved to {target}."
    )
    return 0


def _load_policy_metrics_collector_for_cli() -> tuple[PolicyMetricsCollector, Any]:
    """Initialise the API app and return the shared policy metrics collector."""

    try:
        from memoria_server.api.app_factory import create_app
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"Failed to import app factory: {exc}") from exc

    app = create_app()
    collector = app.config.get("policy_metrics_collector")
    if isinstance(collector, PolicyMetricsCollector):
        return collector, app

    engine = PolicyEnforcementEngine.get_global()
    return engine.metrics, app


def _snapshot_to_csv(payload: dict[str, Any], *, include_header: bool) -> str:
    fieldnames = [
        "generated_at",
        "generated_at_epoch",
        "policy",
        "action",
        "count",
        "total_duration_ms",
        "average_duration_ms",
        "min_duration_ms",
        "max_duration_ms",
        "last_triggered_at",
        "stage_counts",
        "collector_stage_counts",
        "collector_counts",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    if include_header:
        writer.writeheader()

    stage_totals = json.dumps(payload.get("stage_counts", {}), sort_keys=True)
    counts_totals = json.dumps(payload.get("counts", {}), sort_keys=True)
    actions = payload.get("policy_actions") or []

    if not actions:
        writer.writerow(
            {
                "generated_at": payload.get("generated_at"),
                "generated_at_epoch": payload.get("generated_at_epoch"),
                "policy": None,
                "action": None,
                "count": 0,
                "total_duration_ms": 0,
                "average_duration_ms": None,
                "min_duration_ms": None,
                "max_duration_ms": None,
                "last_triggered_at": None,
                "stage_counts": "{}",
                "collector_stage_counts": stage_totals,
                "collector_counts": counts_totals,
            }
        )
    else:
        for entry in actions:
            writer.writerow(
                {
                    "generated_at": payload.get("generated_at"),
                    "generated_at_epoch": payload.get("generated_at_epoch"),
                    "policy": entry.get("policy"),
                    "action": entry.get("action"),
                    "count": entry.get("count"),
                    "total_duration_ms": entry.get("total_duration_ms"),
                    "average_duration_ms": entry.get("average_duration_ms"),
                    "min_duration_ms": entry.get("min_duration_ms"),
                    "max_duration_ms": entry.get("max_duration_ms"),
                    "last_triggered_at": entry.get("last_triggered_at"),
                    "stage_counts": json.dumps(
                        entry.get("stage_counts") or {}, sort_keys=True
                    ),
                    "collector_stage_counts": stage_totals,
                    "collector_counts": counts_totals,
                }
            )

    return buffer.getvalue()


def _normalise_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _rows_to_csv(rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _normalise_csv_value(row.get(field)) for field in fieldnames})
    return buffer.getvalue()


_VERIFICATION_FIELDNAMES: tuple[str, ...] = (
    "name",
    "channel",
    "target",
    "priority",
    "status",
    "issue_count",
    "issues",
    "namespaces",
    "triggers",
    "coverage",
    "next_rotation",
    "rotation_entries",
    "integrations",
)

_ROTATION_FIELDNAMES: tuple[str, ...] = (
    "name",
    "channel",
    "target",
    "checked_at",
    "active_primary",
    "active_secondary",
    "active_date",
    "next_primary",
    "next_secondary",
    "next_date",
    "overdue_windows",
    "invalid_entries",
    "rotation_count",
    "metadata_updated",
    "history_entries",
)


def _handle_roster_verify(args: argparse.Namespace) -> int:
    cadence = int(getattr(args, "cadence", 60) or 60)
    try:
        payload, rows = build_roster_verification(cadence_minutes=cadence)
    except ConfigurationError as exc:
        print(f"Failed to verify escalation roster: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        text = _rows_to_csv(rows, _VERIFICATION_FIELDNAMES)
        if not text.endswith("\n"):
            text += "\n"
        sys.stdout.write(text)

    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    total = summary.get("total_contacts")
    if total is not None:
        status_counts = summary.get("status_counts", {})
        print(
            "Roster verification for {total} contact(s): ok={ok}, warning={warning}, error={error}.".format(
                total=total,
                ok=status_counts.get("ok", 0),
                warning=status_counts.get("warning", 0),
                error=status_counts.get("error", 0),
            ),
            file=sys.stderr,
        )

    return 0


def _handle_roster_rotate(args: argparse.Namespace) -> int:
    cadence = int(getattr(args, "cadence", 60) or 60)
    persist = bool(getattr(args, "persist", False))
    try:
        report: RotationReport = build_rotation_report(
            cadence_minutes=cadence,
            persist=persist,
        )
    except ConfigurationError as exc:
        print(f"Failed to apply rotation metadata: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(report.payload, indent=2))
    else:
        text = _rows_to_csv(report.rows, _ROTATION_FIELDNAMES)
        if not text.endswith("\n"):
            text += "\n"
        sys.stdout.write(text)

    if report.metadata_updates:
        message = (
            f"Applied rotation metadata for {report.metadata_updates} contact(s)."
            if persist
            else f"{report.metadata_updates} contact(s) require metadata updates. Re-run with --persist to save changes."
        )
        print(message, file=sys.stderr)

    summary = report.payload.get("summary", {}) if isinstance(report.payload, dict) else {}
    overdue = summary.get("overdue_contacts")
    if overdue:
        print(
            f"Warning: {overdue} contact(s) have overdue rotation windows.",
            file=sys.stderr,
        )

    return 0


class _PolicyTelemetryEmitter:
    """Format and dispatch policy telemetry snapshots for the CLI."""

    def __init__(self, args: argparse.Namespace, *, limit: int | None) -> None:
        self._limit = limit
        self._format = args.format
        self._follow = bool(args.follow)
        self._pretty = bool(getattr(args, "pretty", False))
        self._webhook = getattr(args, "webhook", None)
        self._webhook_timeout = float(getattr(args, "webhook_timeout", 10.0) or 10.0)
        self._header_written = False
        self._output_path = args.output or "-"
        self._stream: Any | None = None
        if self._output_path != "-":
            mode = "a" if self._follow else "w"
            self._stream = open(self._output_path, mode, encoding="utf-8")

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    def emit(self, snapshot: PolicyMetricsSnapshot) -> None:
        payload = snapshot.to_payload(limit=self._limit, round_durations=3)
        if self._format == "csv":
            text = _snapshot_to_csv(payload, include_header=not self._header_written)
            self._header_written = True
            self._write(text if text.endswith("\n") else text + "\n")
            self._post_webhook(text, "text/csv")
            return

        indent = 2 if (self._pretty and not self._follow) else None
        text = json.dumps(payload, indent=indent)
        if not text.endswith("\n"):
            text = text + "\n"
        self._write(text)
        self._post_webhook(text, "application/json")

    def _write(self, text: str) -> None:
        target = self._stream or sys.stdout
        target.write(text)
        target.flush()

    def _post_webhook(self, payload: str, content_type: str) -> None:
        if not self._webhook:
            return

        data = payload.rstrip("\n").encode("utf-8")
        request = urllib.request.Request(
            self._webhook,
            data=data,
            headers={
                "Content-Type": content_type,
                "User-Agent": "memoria-cli/telemetry",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._webhook_timeout):
                pass
        except urllib.error.URLError as exc:
            print(
                f"warning: Failed to post telemetry snapshot: {exc}",
                file=sys.stderr,
            )


def _handle_policy_telemetry(args: argparse.Namespace) -> int:
    try:
        collector, app = _load_policy_metrics_collector_for_cli()
    except Exception as exc:
        print(f"Failed to initialise policy telemetry: {exc}", file=sys.stderr)
        return 1

    _ = app  # Keep reference alive for scheduled background tasks.

    limit = args.limit if args.limit is not None else None
    if limit is not None and limit < 0:
        limit = None

    emitter = _PolicyTelemetryEmitter(args, limit=limit)

    unsubscribe: Callable[[], None] | None = None
    event_queue: queue.Queue[PolicyMetricsSnapshot] | None = None
    stop_event: threading.Event | None = None

    try:
        if args.follow:
            queue_size = args.queue_size if args.queue_size and args.queue_size > 0 else 256
            event_queue = queue.Queue(maxsize=queue_size)
            stop_event = threading.Event()

            def _observer(snapshot: PolicyMetricsSnapshot) -> None:
                if stop_event.is_set():
                    return
                try:
                    event_queue.put_nowait(snapshot)
                except queue.Full:
                    try:
                        event_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        event_queue.put_nowait(snapshot)
                    except queue.Full:
                        pass

            unsubscribe = collector.register_observer(_observer)

        snapshot = collector.capture_snapshot(limit=limit)
        emitter.emit(snapshot)

        if not args.follow:
            return 0

        assert event_queue is not None and stop_event is not None
        max_events = args.max_events if args.max_events else None
        events_sent = 0

        try:
            while True:
                try:
                    next_snapshot = event_queue.get(timeout=1.0)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue

                emitter.emit(next_snapshot)
                events_sent += 1
                if max_events is not None and events_sent >= max_events:
                    break
        except KeyboardInterrupt:
            print("Telemetry stream interrupted by user.", file=sys.stderr)
        finally:
            stop_event.set()
    finally:
        if unsubscribe is not None:
            try:
                unsubscribe()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
        emitter.close()

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memoria",
        description=(
            "Utilities for local Memoria development, including running the API "
            "server and preparing databases."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    bootstrap = subparsers.add_parser(
        "bootstrap",
        help="Generate .env and configuration files via an interactive wizard.",
    )
    bootstrap.add_argument(
        "--config",
        default="memoria.json",
        help="Path to the Memoria configuration file to write (default: memoria.json).",
    )
    bootstrap.add_argument(
        "--env-file",
        default=".env",
        help="Path to the environment file to write (default: .env).",
    )
    bootstrap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files without prompting.",
    )
    bootstrap.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run the bootstrap wizard without interactive prompts.",
    )
    bootstrap.add_argument(
        "--config-json",
        help="Path to a JSON file providing configuration for non-interactive runs.",
    )
    bootstrap.set_defaults(func=_handle_bootstrap)

    runserver = subparsers.add_parser(
        "runserver", help="Start the Flask development server."
    )
    runserver.add_argument("--host", default="0.0.0.0", help="Hostname to bind.")
    runserver.add_argument("--port", type=int, default=8080, help="Port to listen on.")
    runserver.add_argument(
        "--reload",
        action="store_true",
        help="Enable Flask's debug reloader (development only).",
    )
    runserver.set_defaults(func=_handle_runserver)

    init_db = subparsers.add_parser(
        "init-db", help="Create or upgrade the configured Memoria database."
    )
    init_db.set_defaults(func=_handle_init_db)

    migrate = subparsers.add_parser(
        "migrate",
        help="Inspect and run schema migration scripts bundled with Memoria.",
    )
    migrate_sub = migrate.add_subparsers(dest="migrate_command")
    migrate_sub.required = True

    migrate_list = migrate_sub.add_parser(
        "list",
        help="List available migration scripts and their descriptions.",
    )
    migrate_list.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived or legacy migrations in the output.",
    )
    migrate_list.set_defaults(func=_handle_migrate_list)

    migrate_run = migrate_sub.add_parser(
        "run",
        help="Execute a migration script against the configured database.",
    )
    migrate_run.add_argument(
        "migration",
        help="Migration name or filename to execute.",
    )
    migrate_run.add_argument(
        "--database-url",
        help="Override the database URL (defaults to the configured connection).",
    )
    migrate_run.add_argument(
        "--include-archived",
        action="store_true",
        help="Allow running migrations that live under the archive folder.",
    )
    migrate_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the migration without executing it.",
    )
    migrate_run.set_defaults(func=_handle_migrate_run)

    build_clusters = subparsers.add_parser(
        "build-clusters",
        help="Generate memory clusters for faster retrieval.",
        description="Generate memory clusters for faster retrieval.",
    )
    build_clusters.add_argument(
        "--mode",
        choices=("heuristic", "vector"),
        default="heuristic",
        help="Select the clustering strategy to run.",
    )
    build_clusters.add_argument(
        "--database-url",
        help="Optional database URL override when using heuristic clustering.",
    )
    build_clusters.add_argument(
        "--source",
        dest="sources",
        action="append",
        help="Limit clustering to the provided memory source (repeatable).",
    )
    build_clusters.add_argument(
        "--verbose",
        action="store_true",
        help="Emit verbose logging when using heuristic clustering.",
    )
    build_clusters.set_defaults(func=_handle_build_clusters)

    export_data = subparsers.add_parser(
        "export-data",
        help="Export key Memoria tables to JSON or NDJSON.",
    )
    export_data.add_argument(
        "destination",
        help="Output file path or '-' to stream to stdout.",
    )
    export_data.add_argument(
        "--database-url",
        help="Optional database URL override (defaults to configured connection).",
    )
    export_data.add_argument(
        "--format",
        choices=("json", "ndjson"),
        default="json",
        help="Serialisation format to emit.",
    )
    export_data.add_argument(
        "--table",
        dest="tables",
        action="append",
        help="Limit the export to the specified table (repeatable).",
    )
    export_data.set_defaults(func=_handle_export_data, tables=None)

    import_data = subparsers.add_parser(
        "import-data",
        help="Import a previously exported Memoria dataset.",
    )
    import_data.add_argument(
        "source",
        help="Path to the JSON/NDJSON dataset or '-' to read from stdin.",
    )
    import_data.add_argument(
        "--database-url",
        help="Optional database URL override (defaults to configured connection).",
    )
    import_data.add_argument(
        "--format",
        choices=("json", "ndjson"),
        help="Input format. Defaults to JSON unless the filename ends with .ndjson.",
    )
    import_data.add_argument(
        "--table",
        dest="tables",
        action="append",
        help="Limit the import to a subset of tables (repeatable).",
    )
    import_data.add_argument(
        "--no-truncate",
        action="store_true",
        help="Append data without clearing existing rows first.",
    )
    import_data.set_defaults(func=_handle_import_data, tables=None)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run benchmark scenarios from a JSON or YAML configuration file.",
    )
    benchmark.add_argument(
        "config",
        help="Path to the benchmark configuration file.",
    )
    benchmark.add_argument(
        "--namespace",
        help="Base namespace used when staging temporary benchmark data.",
    )
    benchmark.add_argument(
        "--team-id",
        help="Optional team context to activate for the benchmark run.",
    )
    benchmark.add_argument(
        "--output",
        help="Write benchmark results to this file instead of stdout.",
    )
    benchmark.add_argument(
        "--format",
        choices=("json", "ndjson"),
        default="json",
        help="Serialisation format for --output destinations.",
    )
    benchmark.set_defaults(
        func=_handle_benchmark,
        namespace=None,
        team_id=None,
        output=None,
    )

    evaluation_scaffold = subparsers.add_parser(
        "evaluation-scaffold",
        help="Generate a starter evaluation suite YAML file.",
    )
    evaluation_scaffold.add_argument(
        "output",
        help="Destination path for the generated YAML document.",
    )
    evaluation_scaffold.add_argument(
        "--suite-id",
        required=True,
        help="Unique identifier for the evaluation suite.",
    )
    evaluation_scaffold.add_argument(
        "--suite-name",
        help="Human readable suite name.",
    )
    evaluation_scaffold.add_argument(
        "--description",
        help="High level description of the evaluation goals.",
    )
    evaluation_scaffold.add_argument(
        "--use-case",
        dest="use_cases",
        action="append",
        help="Add a use case label to the suite (repeatable).",
    )
    evaluation_scaffold.add_argument(
        "--workspace",
        default="workspace",
        help="Default workspace directory for materialised scenarios.",
    )
    evaluation_scaffold.add_argument(
        "--scenario-id",
        default="scenario",
        help="Identifier for the initial scenario.",
    )
    evaluation_scaffold.add_argument(
        "--scenario-name",
        default="New Scenario",
        help="Display name for the initial scenario.",
    )
    evaluation_scaffold.add_argument(
        "--scenario-description",
        default="Describe the workflow being evaluated.",
        help="Narrative description for the initial scenario.",
    )
    evaluation_scaffold.add_argument(
        "--dataset-kind",
        default="fixture",
        choices=("fixture", "generated"),
        help="Dataset provisioning strategy for the scenario.",
    )
    evaluation_scaffold.add_argument(
        "--dataset-path",
        default="datasets/example.jsonl",
        help="Dataset filename or relative path placeholder.",
    )
    evaluation_scaffold.add_argument(
        "--retrieval-query",
        dest="retrieval_queries",
        action="append",
        help="Add a retrieval query placeholder (repeatable).",
    )
    evaluation_scaffold.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination if it already exists.",
    )
    evaluation_scaffold.set_defaults(
        func=_handle_evaluation_scaffold,
        use_cases=None,
        retrieval_queries=None,
    )

    assign_model = subparsers.add_parser(
        "assign-task-model",
        help="Override the provider/model routing for a specific task.",
    )
    assign_model.add_argument("task", help="Logical task name (e.g. memory_ingest).")
    assign_model.add_argument(
        "--provider",
        required=True,
        help="Provider name or alias to use for the task.",
    )
    assign_model.add_argument(
        "--model",
        help="Optional model override for the task.",
    )
    assign_model.add_argument(
        "--fallback",
        action="append",
        help=(
            "Fallback providers in priority order. Accepts comma-separated lists; "
            "use provider:model to specify model overrides."
        ),
    )
    assign_model.add_argument(
        "--clear-fallback",
        action="store_true",
        help="Remove any fallback providers for the task.",
    )
    assign_model.set_defaults(func=_handle_assign_task_model)

    teams = subparsers.add_parser(
        "teams", help="Inspect and manage collaborative team spaces.",
    )
    teams_sub = teams.add_subparsers(dest="teams_command")
    teams_sub.required = True

    teams_list = teams_sub.add_parser(
        "list", help="List configured teams and the active context.",
    )
    teams_list.add_argument(
        "--include-members",
        action="store_true",
        help="Include membership details in the response.",
    )
    teams_list.add_argument(
        "--team-id",
        help="Temporarily activate a team before listing.",
    )
    teams_list.set_defaults(func=_handle_team_list, team_id=None)

    teams_create = teams_sub.add_parser(
        "create", help="Create or update a collaborative team space.",
    )
    teams_create.add_argument("team_id", help="Identifier for the team space.")
    teams_create.add_argument(
        "--namespace", help="Override the namespace associated with the team."
    )
    teams_create.add_argument(
        "--display-name", help="Human readable name for the team.",
    )
    teams_create.add_argument(
        "--member",
        dest="members",
        action="append",
        default=None,
        help="Member identifier (repeatable).",
    )
    teams_create.add_argument(
        "--admin",
        dest="admins",
        action="append",
        default=None,
        help="Admin identifier (repeatable).",
    )
    share_group = teams_create.add_mutually_exclusive_group()
    share_group.add_argument(
        "--share-by-default",
        dest="share_by_default",
        action="store_const",
        const=True,
        help="Share stored memories with the team by default.",
    )
    share_group.add_argument(
        "--no-share-by-default",
        dest="share_by_default",
        action="store_const",
        const=False,
        help="Disable default sharing for the team.",
    )
    teams_create.add_argument(
        "--metadata",
        help="Optional JSON object describing additional team metadata.",
    )
    teams_create.add_argument(
        "--include-members",
        action="store_true",
        help="Include membership details in the response payload.",
    )
    teams_create.add_argument(
        "--activate",
        action="store_true",
        help="Activate the team immediately after creation.",
    )
    teams_create.add_argument(
        "--allow-guest",
        action="store_true",
        help="Skip membership enforcement when activating.",
    )
    teams_create.set_defaults(
        func=_handle_team_create,
        share_by_default=None,
    )

    teams_activate = teams_sub.add_parser(
        "activate", help="Activate a team for subsequent CLI operations.",
    )
    teams_activate.add_argument("team_id", help="Identifier of the team to activate.")
    teams_activate.add_argument(
        "--allow-guest",
        action="store_true",
        help="Skip membership enforcement when activating.",
    )
    teams_activate.set_defaults(func=_handle_team_activate)

    teams_clear = teams_sub.add_parser(
        "clear", help="Clear the currently active team context.",
    )
    teams_clear.set_defaults(func=_handle_team_clear)

    workspaces = subparsers.add_parser(
        "workspaces", help="Manage shared workspaces (alias for team spaces).",
    )
    workspaces_sub = workspaces.add_subparsers(dest="workspaces_command")
    workspaces_sub.required = True

    workspaces_list = workspaces_sub.add_parser(
        "list", help="List configured workspaces and the active context.",
    )
    workspaces_list.add_argument(
        "--include-members",
        action="store_true",
        help="Include membership details in the response.",
    )
    workspaces_list.add_argument(
        "--workspace-id",
        dest="team_id",
        help="Temporarily activate a workspace before listing.",
    )
    workspaces_list.set_defaults(func=_handle_workspace_list, team_id=None)

    workspaces_create = workspaces_sub.add_parser(
        "create", help="Create or update a collaborative workspace.",
    )
    workspaces_create.add_argument(
        "workspace_id", help="Identifier for the workspace.",
    )
    workspaces_create.add_argument(
        "--namespace", help="Override the namespace associated with the workspace.",
    )
    workspaces_create.add_argument(
        "--display-name", help="Human readable name for the workspace.",
    )
    workspaces_create.add_argument(
        "--member",
        dest="members",
        action="append",
        default=None,
        help="Member identifier (repeatable).",
    )
    workspaces_create.add_argument(
        "--admin",
        dest="admins",
        action="append",
        default=None,
        help="Admin identifier (repeatable).",
    )
    workspace_share_group = workspaces_create.add_mutually_exclusive_group()
    workspace_share_group.add_argument(
        "--share-by-default",
        dest="share_by_default",
        action="store_const",
        const=True,
        help="Share stored memories with the workspace by default.",
    )
    workspace_share_group.add_argument(
        "--no-share-by-default",
        dest="share_by_default",
        action="store_const",
        const=False,
        help="Disable default sharing for the workspace.",
    )
    workspaces_create.add_argument(
        "--metadata",
        help="Optional JSON object describing additional workspace metadata.",
    )
    workspaces_create.add_argument(
        "--include-members",
        action="store_true",
        help="Include membership details in the response payload.",
    )
    workspaces_create.add_argument(
        "--activate",
        action="store_true",
        help="Activate the workspace immediately after creation.",
    )
    workspaces_create.add_argument(
        "--allow-guest",
        action="store_true",
        help="Skip membership enforcement when activating.",
    )
    workspaces_create.set_defaults(
        func=_handle_workspace_create,
        share_by_default=None,
    )

    workspaces_switch = workspaces_sub.add_parser(
        "switch", help="Activate a workspace for subsequent CLI operations.",
    )
    workspaces_switch.add_argument(
        "workspace_id", help="Identifier of the workspace to activate.",
    )
    workspaces_switch.add_argument(
        "--allow-guest",
        action="store_true",
        help="Skip membership enforcement when activating.",
    )
    workspaces_switch.set_defaults(func=_handle_workspace_switch)

    roster = subparsers.add_parser(
        "roster", help="Inspect escalation roster automation outputs.",
    )
    roster_sub = roster.add_subparsers(dest="roster_command")
    roster_sub.required = True

    roster_verify = roster_sub.add_parser(
        "verify", help="Run escalation roster verification checks.",
    )
    roster_verify.add_argument(
        "--cadence",
        type=int,
        default=60,
        help="Verification cadence in minutes (default: 60).",
    )
    roster_verify.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format suitable for ticket attachments (default: json).",
    )
    roster_verify.set_defaults(func=_handle_roster_verify)

    roster_rotate = roster_sub.add_parser(
        "rotate", help="Apply rotation metadata updates for escalation contacts.",
    )
    roster_rotate.add_argument(
        "--cadence",
        type=int,
        default=60,
        help="Rotation check cadence in minutes (default: 60).",
    )
    roster_rotate.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format suitable for ticket attachments (default: json).",
    )
    roster_rotate.add_argument(
        "--persist",
        action="store_true",
        help="Persist rotation metadata updates back to the configuration.",
    )
    roster_rotate.set_defaults(func=_handle_roster_rotate, persist=False)

    policy = subparsers.add_parser(
        "policy", help="Validate and manage retention policy rules.",
    )
    policy_sub = policy.add_subparsers(dest="policy_command")
    policy_sub.required = True

    policy_lint = policy_sub.add_parser(
        "lint", help="Validate a policy file without applying any changes.",
    )
    policy_lint.add_argument(
        "source", help="Path to the JSON/YAML policy file or '-' for stdin.",
    )
    policy_lint.set_defaults(func=_handle_policy_lint)

    policy_test = policy_sub.add_parser(
        "test", help="Simulate policies against a sample dataset.",
    )
    policy_test.add_argument(
        "source", help="Path to the JSON/YAML policy file or '-' for stdin.",
    )
    policy_test.add_argument(
        "--samples",
        help="Optional JSON or NDJSON dataset of memories to test against.",
    )
    policy_test.set_defaults(func=_handle_policy_test)

    policy_apply = policy_sub.add_parser(
        "apply", help="Persist policy rules to the active configuration.",
    )
    policy_apply.add_argument(
        "source", help="Path to the JSON/YAML policy file or '-' for stdin.",
    )
    policy_apply.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the file without saving configuration changes.",
    )
    policy_apply.set_defaults(func=_handle_policy_apply)

    policy_telemetry = policy_sub.add_parser(
        "telemetry", help="Export policy enforcement telemetry snapshots.",
    )
    policy_telemetry.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Snapshot output format (default: json).",
    )
    policy_telemetry.add_argument(
        "--output",
        default="-",
        help="Destination path or '-' for stdout.",
    )
    policy_telemetry.add_argument(
        "--webhook",
        help="Optional webhook URL to POST snapshots.",
    )
    policy_telemetry.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of policy action records per snapshot.",
    )
    policy_telemetry.add_argument(
        "--follow",
        action="store_true",
        help="Stream snapshots as policy events occur.",
    )
    policy_telemetry.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Stop streaming after N updates (requires --follow).",
    )
    policy_telemetry.add_argument(
        "--queue-size",
        type=int,
        default=256,
        help="Observer queue depth when streaming (default: 256).",
    )
    policy_telemetry.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output (ignored with --follow).",
    )
    policy_telemetry.add_argument(
        "--webhook-timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds when posting to a webhook (default: 10).",
    )
    policy_telemetry.set_defaults(func=_handle_policy_telemetry)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1
    return int(handler(args) or 0)


if __name__ == "__main__":  # pragma: no cover - invoked manually
    sys.exit(main())
