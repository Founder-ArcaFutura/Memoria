"""Minimal runner module for the Memoria Flask application."""

from __future__ import annotations

import os

from memoria_server.api.app_factory import create_app
from memoria_server.api.scheduler import (
    schedule_daily_decrement as _schedule_daily_decrement_impl,
)
from memoria_server.api.spatial_setup import init_spatial_db as _init_spatial_db_impl
from memoria_server.api.utility_routes import (
    build_heuristic_clusters,
    build_index,
    query_cluster_index,
)

app = create_app()


def _sync_module_state() -> None:
    """Keep module-level shortcuts aligned with the Flask app config."""

    global memoria, config_manager, DATABASE_URL, DB_PATH, EXPECTED_API_KEY, DECREMENT_TZ, engine

    memoria = app.config["memoria"]
    config_manager = app.config["config_manager"]
    DATABASE_URL = app.config["DATABASE_URL"]
    DB_PATH = app.config.get("DB_PATH")
    EXPECTED_API_KEY = app.config.get("EXPECTED_API_KEY")
    DECREMENT_TZ = app.config.get("DECREMENT_TZ")
    engine = app.config.get("ENGINE")


_sync_module_state()


def init_spatial_db() -> None:
    """Proxy for :func:`memoria_server.api.spatial_setup.init_spatial_db`."""

    _init_spatial_db_impl(app)
    _sync_module_state()


def _schedule_daily_decrement() -> None:
    """Proxy for :func:`memoria_server.api.scheduler.schedule_daily_decrement`."""

    _schedule_daily_decrement_impl(app)


def main() -> None:
    """Run the Flask development server."""

    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


__all__ = [
    "app",
    "build_index",
    "build_heuristic_clusters",
    "query_cluster_index",
    "init_spatial_db",
    "_schedule_daily_decrement",
    "_sync_module_state",
    "config_manager",
    "memoria",
    "DATABASE_URL",
    "DB_PATH",
    "EXPECTED_API_KEY",
    "DECREMENT_TZ",
    "engine",
    "main",
]


if __name__ == "__main__":
    main()
