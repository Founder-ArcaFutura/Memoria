from __future__ import annotations

import json
import sqlite3
from functools import lru_cache, wraps
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from loguru import logger
from sqlalchemy import text

from memoria.database.analytics import (
    CategoryCount,
    RetentionSeries,
    UsageRecord,
    get_analytics_summary,
    get_category_counts,
    get_retention_trends,
    get_usage_frequency,
)
from memoria.utils import get_cluster_activity, query_cluster_index
from .memory_routes import _parse_positive_int as _parse_positive_query_int

try:  # Cluster indexing is optional
    from memoria.cli_support.index_clusters import (
        build_index,
        get_status as get_cluster_status,
    )
except ImportError:  # pragma: no cover - handled during runtime
    try:  # Compatibility fallback for legacy paths
        from scripts.index_clusters import (
            build_index,
            get_status as get_cluster_status,
        )
    except ImportError:  # pragma: no cover - handled during runtime
        build_index = None
        get_cluster_status = lambda: {
            "state": "unavailable",
            "current": 0,
            "total": 0,
            "error": "clustering unavailable",
        }
        logger.warning(
            "Cluster helpers could not be imported; clustering disabled",
        )

try:  # Heuristic clustering is optional
    from memoria.cli_support.heuristic_clusters import build_heuristic_clusters
except ImportError:  # pragma: no cover - handled during runtime
    try:  # Legacy fallback
        from scripts.heuristic_clusters import build_heuristic_clusters
    except ImportError:  # pragma: no cover - handled during runtime
        build_heuristic_clusters = None

utility_bp = Blueprint("utility", __name__)


def _current_connection_string(settings) -> str:
    """Determine the active database connection string for clustering."""

    url = current_app.config.get("DATABASE_URL")
    if url:
        return url

    db_path = current_app.config.get("DB_PATH")
    if db_path:
        path = str(db_path)
        return path if path.startswith("sqlite://") else f"sqlite:///{path}"

    if hasattr(settings, "get_database_url"):
        try:
            derived = settings.get_database_url()
            if derived:
                return derived
        except Exception:  # pragma: no cover - defensive
            pass

    database = getattr(settings, "database", None)
    connection_string = getattr(database, "connection_string", None)
    if connection_string:
        return connection_string

    return "sqlite:///memoria.db"


def _existing_clusters() -> list:
    """Safely return any currently stored clusters."""

    try:
        return query_cluster_index()
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to load existing clusters")
        return []


def _with_session(view):
    """Provide a SQLAlchemy session to analytics endpoints."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        memoria = current_app.config.get("memoria")
        db_manager = getattr(memoria, "db_manager", None) if memoria else None
        session_factory = getattr(db_manager, "SessionLocal", None)
        include_short_term = bool(getattr(db_manager, "enable_short_term", False))

        if session_factory is None:
            logger.warning("Analytics requested but SessionLocal is unavailable")
            return jsonify({"status": "error", "message": "Database session unavailable"}), 503

        session = session_factory()
        try:
            return view(session, include_short_term, *args, **kwargs)
        finally:
            session.close()

    return wrapped


def _handle_heuristic_cluster_rebuild(settings, sources):
    """Execute the heuristic clustering rebuild workflow."""

    if not getattr(settings, "enable_heuristic_clustering", False):
        msg = "Heuristic clustering is disabled"
        logger.warning(msg)
        return (
            jsonify({"status": "disabled", "message": msg, "clusters": _existing_clusters()}),
            200,
        )

    if build_heuristic_clusters is None:
        msg = "Heuristic clustering unavailable"
        logger.warning(msg)
        return (
            jsonify({"status": "unavailable", "message": msg, "clusters": _existing_clusters()}),
            200,
        )

    connection_string = _current_connection_string(settings)
    try:
        result = build_heuristic_clusters(connection_string)
        if isinstance(result, tuple):
            clusters, summary = result
        else:  # backwards compatibility
            clusters, summary = result, None
        from memoria.database.queries.cluster_queries import replace_clusters

        replace_clusters(clusters)
        body = {"status": "success", "clusters": clusters}
        if summary:
            body["summary"] = summary
        return jsonify(body)

    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Failed to rebuild heuristic clusters")
        return (
            jsonify({"status": "error", "message": str(e), "clusters": []}),
            500,
        )


def _handle_vector_cluster_rebuild(settings, sources):
    """Execute the vector clustering rebuild workflow."""

    if not getattr(settings, "enable_vector_clustering", False):
        msg = "Vector clustering is disabled"
        logger.warning(msg)
        return (
            jsonify({"status": "disabled", "message": msg, "clusters": _existing_clusters()}),
            200,
        )

    if build_index is None:
        msg = "Cluster indexing dependencies missing"
        logger.warning(msg)
        return (
            jsonify({"status": "unavailable", "message": msg, "clusters": _existing_clusters()}),
            200,
        )

    try:
        if sources:
            clusters = build_index(sources=sources)
        else:
            clusters = build_index()
        return jsonify({"status": "success", "clusters": clusters})

    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Failed to rebuild cluster index")
        return (
            jsonify({"status": "error", "message": str(e), "clusters": []}),
            500,
        )


@utility_bp.route("/debug/anchors", methods=["GET"])
def list_anchors():
    """Return all unique symbolic anchors stored in memory tables."""

    def _normalize_anchor_value(value, *, allow_json: bool = True) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if allow_json:
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                else:
                    return _normalize_anchor_value(parsed, allow_json=False)
            parts = [part.strip() for part in stripped.split(",")]
            return [part for part in parts if part]

        if isinstance(value, (list, tuple, set)):
            anchors: list[str] = []
            for item in value:
                anchors.extend(_normalize_anchor_value(item, allow_json=False))
            return anchors

        return [str(value)]

    def _collect_rows(rows):
        for (raw,) in rows:
            for anchor in _normalize_anchor_value(raw):
                anchors.add(anchor)

    try:
        anchors: set[str] = set()
        memoria = current_app.config.get("memoria")
        db_manager = getattr(memoria, "db_manager", None) if memoria else None
        engine = getattr(db_manager, "engine", None) if db_manager else None
        include_short_term = bool(getattr(db_manager, "enable_short_term", False))
        db_path = current_app.config.get("DB_PATH")

        if engine is not None:
            with engine.begin() as conn:
                result = conn.execute(
                    text("SELECT symbolic_anchors FROM long_term_memory")
                )
                _collect_rows(result.fetchall())

                if include_short_term:
                    try:
                        result = conn.execute(
                            text("SELECT symbolic_anchors FROM short_term_memory")
                        )
                    except Exception:
                        logger.debug(
                            "short_term_memory table unavailable when listing anchors",
                            exc_info=True,
                        )
                    else:
                        _collect_rows(result.fetchall())

        elif db_path:
            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.cursor()
                try:
                    cur.execute("SELECT symbolic_anchors FROM long_term_memory")
                    _collect_rows(cur.fetchall())
                except sqlite3.Error:
                    logger.exception("Failed to load long-term anchors from SQLite")
                    raise

                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='short_term_memory'"
                )
                if cur.fetchone():
                    try:
                        cur.execute("SELECT symbolic_anchors FROM short_term_memory")
                        _collect_rows(cur.fetchall())
                    except sqlite3.Error:
                        logger.warning(
                            "short_term_memory table exists but could not be queried for anchors"
                        )
            finally:
                conn.close()

        return jsonify({"anchors": sorted(anchors)})
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"error": str(e)}), 500


def _openapi_spec_path() -> Path:
    """Return the absolute path to the repository OpenAPI specification."""

    return Path(__file__).resolve().parents[2] / "openapi.json"


@lru_cache(maxsize=1)
def _load_openapi_spec() -> dict[str, object]:
    """Load and cache the OpenAPI specification from disk."""

    spec_path = _openapi_spec_path()
    with spec_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


@utility_bp.route("/openapi.json", methods=["GET"])
def get_openapi_spec():
    """Serve the repository OpenAPI specification file."""

    spec_path = _openapi_spec_path()

    try:
        spec = _load_openapi_spec()
    except FileNotFoundError:
        logger.warning("OpenAPI specification missing at %s", spec_path)
        return jsonify({"error": "OpenAPI specification not found"}), 404
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode OpenAPI specification: %s", exc)
        return (
            jsonify({"error": "OpenAPI specification is invalid"}),
            500,
        )

    return jsonify(spec)


def _parse_int_param(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = request.args.get(name)
    if raw in (None, ""):
        return default
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return default
    parsed = max(parsed, minimum)
    return min(parsed, maximum)


def _parse_positive_float_param(
    name: str, raw_value: str | None, *, default: float
) -> tuple[float | None, str | None]:
    """Parse a positive floating point parameter from the request query string."""

    if raw_value in (None, ""):
        return default, None

    message = f"Parameter '{name}' must be a positive number"

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None, message

    if value <= 0:
        return None, message

    return value, None


@utility_bp.route("/analytics/categories", methods=["GET"])
@_with_session
def analytics_categories(session, include_short_term):
    namespace = request.args.get("namespace")
    payload = get_category_counts(
        session,
        namespace=namespace,
        include_short_term=include_short_term,
    )
    return jsonify(
        {
            "status": "ok",
            "namespace": namespace or "all",
            "include_short_term": include_short_term,
            "counts": [
                {
                    "memory_type": memory_type,
                    "categories": [
                        {"category": item.category, "count": item.count}
                        for item in categories
                    ],
                }
                for memory_type, categories in payload.items()
            ],
        }
    )


@utility_bp.route("/analytics/retention", methods=["GET"])
@_with_session
def analytics_retention(session, include_short_term):
    namespace = request.args.get("namespace")
    days = _parse_int_param("days", default=30, minimum=1, maximum=365)
    payload = get_retention_trends(
        session,
        namespace=namespace,
        days=days,
        include_short_term=include_short_term,
    )
    return jsonify(
        {
            "status": "ok",
            "namespace": namespace or "all",
            "include_short_term": include_short_term,
            "range": payload.get("range"),
            "series": {
                key: [
                    {
                        "retention_type": series.retention_type,
                        "daily_counts": series.daily_counts,
                    }
                    for series in section.get("series", [])
                ]
                for key, section in payload.items()
                if key in {"long_term", "short_term"}
            },
        }
    )


@utility_bp.route("/analytics/usage", methods=["GET"])
@_with_session
def analytics_usage(session, include_short_term):
    namespace = request.args.get("namespace")
    top_n = _parse_int_param("top", default=10, minimum=1, maximum=100)
    payload = get_usage_frequency(
        session,
        namespace=namespace,
        include_short_term=include_short_term,
        top_n=top_n,
    )

    def _serialize(records: list[UsageRecord]):
        return [
            {
                "memory_id": record.memory_id,
                "summary": record.summary,
                "category": record.category,
                "access_count": record.access_count,
                "last_accessed": record.last_accessed.isoformat()
                if record.last_accessed
                else None,
            }
            for record in records
        ]

    return jsonify(
        {
            "status": "ok",
            "namespace": namespace or "all",
            "include_short_term": include_short_term,
            "usage": {
                key: {
                    "total_records": section["total_records"],
                    "total_accesses": section["total_accesses"],
                    "average_accesses": section["average_accesses"],
                    "top_records": _serialize(section["top_records"]),
                }
                for key, section in payload.items()
            },
        }
    )


@utility_bp.route("/analytics/summary", methods=["GET"])
@_with_session
def analytics_summary(session, include_short_term):
    namespace = request.args.get("namespace")
    days = _parse_int_param("days", default=30, minimum=1, maximum=365)
    top_n = _parse_int_param("top", default=10, minimum=1, maximum=100)
    payload = get_analytics_summary(
        session,
        namespace=namespace,
        include_short_term=include_short_term,
        days=days,
        top_n=top_n,
    )

    def _encode_counts(items: list[CategoryCount]):
        return [
            {"category": item.category, "count": item.count}
            for item in items
        ]

    def _encode_trends(series_list: list[RetentionSeries]):
        return [
            {
                "retention_type": series.retention_type,
                "daily_counts": series.daily_counts,
            }
            for series in series_list
        ]

    def _encode_usage(records: list[UsageRecord]):
        return [
            {
                "memory_id": record.memory_id,
                "summary": record.summary,
                "category": record.category,
                "access_count": record.access_count,
                "last_accessed": record.last_accessed.isoformat()
                if record.last_accessed
                else None,
            }
            for record in records
        ]

    return jsonify(
        {
            "status": "ok",
            "namespace": payload["namespace"],
            "include_short_term": payload["include_short_term"],
            "category_counts": {
                key: _encode_counts(items)
                for key, items in payload["category_counts"].items()
            },
            "retention_trends": {
                "range": payload["retention_trends"].get("range"),
                "long_term": _encode_trends(
                    payload["retention_trends"].get("long_term", {}).get("series", [])
                ),
                "short_term": _encode_trends(
                    payload["retention_trends"].get("short_term", {}).get("series", [])
                )
                if include_short_term
                else [],
            },
            "usage_frequency": {
                key: {
                    "total_records": section["total_records"],
                    "total_accesses": section["total_accesses"],
                    "average_accesses": section["average_accesses"],
                    "top_records": _encode_usage(section["top_records"]),
                }
                for key, section in payload["usage_frequency"].items()
            },
        }
    )


@utility_bp.route("/clusters", methods=["POST"])
def rebuild_clusters():
    """Rebuild the cluster index and return the new clusters."""
    mode = request.args.get("mode")
    payload = request.get_json(silent=True) or {}
    sources_param = payload.get("sources")
    if sources_param is None:
        sources_param = payload.get("source")
    query_sources = request.args.getlist("source")
    normalized_sources: list[str] = []
    if isinstance(sources_param, str):
        normalized_sources.append(sources_param)
    elif isinstance(sources_param, (list, tuple)):
        normalized_sources.extend(s for s in sources_param if isinstance(s, str))
    elif sources_param is not None:
        logger.warning(
            "Ignoring unsupported sources payload; expected string or list of strings"
        )
    if not normalized_sources and query_sources:
        normalized_sources.extend(s for s in query_sources if isinstance(s, str) and s)
    # Preserve order while removing duplicates
    seen_sources: set[str] = set()
    deduped_sources: list[str] = []
    for source in normalized_sources:
        if source in seen_sources:
            continue
        seen_sources.add(source)
        deduped_sources.append(source)
    sources = deduped_sources or None
    if mode is None:
        mode = payload.get("mode")
    logger.bind(
        endpoint="rebuild_clusters",
        mode=mode,
        query_params=request.args.to_dict(),
        payload=payload,
        sources=sources,
    ).info("Cluster rebuild requested")
    if mode is not None and mode not in {"heuristic", "vector"}:
        msg = "Invalid mode; must be 'heuristic' or 'vector'"
        logger.warning(msg)
        return jsonify({"status": "error", "message": msg}), 400
    config_manager = current_app.config["config_manager"]
    settings = config_manager.get_settings()

    if (
        not getattr(settings, "enable_heuristic_clustering", False)
        and not getattr(settings, "enable_vector_clustering", False)
    ):
        msg = "Clustering is disabled"
        logger.warning(msg)
        return jsonify({"status": "disabled", "message": msg, "clusters": _existing_clusters()}), 200

    if mode is None:
        if getattr(settings, "enable_heuristic_clustering", False) and not getattr(
            settings, "enable_vector_clustering", False
        ):
            mode = "heuristic"
        else:
            mode = "vector"

    if mode == "heuristic":
        return _handle_heuristic_cluster_rebuild(settings, sources)

    return _handle_vector_cluster_rebuild(settings, sources)



@utility_bp.route("/clusters/status", methods=["GET"])
def cluster_status():
    """Return progress information for the current cluster rebuild."""
    return jsonify(get_cluster_status())


@utility_bp.route("/clusters", methods=["GET"])
def get_clusters():
    """Retrieve clusters from ``latest_clusters.json`` with optional filters."""
    try:
        params = request.args.to_dict()
        logger.bind(endpoint="get_clusters", query_params=params).info(
            "Cluster query received"
        )

        keyword = params.get("keyword")
        min_pol = params.get("min_polarity")
        max_pol = params.get("max_polarity")
        min_size = params.get("min_size")
        max_size = params.get("max_size")
        min_imp = params.get("min_importance")
        max_imp = params.get("max_importance")
        min_weight = params.get("min_weight")
        max_weight = params.get("max_weight")
        min_age = params.get("min_age_seconds")
        max_age = params.get("max_age_seconds")
        sort_by = params.get("sort_by")
        include_members = params.get("include_members") in ("1", "true", "True")

        emotion_range = None
        if min_pol is not None or max_pol is not None:
            low = float(min_pol) if min_pol is not None else -1.0
            high = float(max_pol) if max_pol is not None else 1.0
            emotion_range = (low, high)
        size_range = None
        if min_size is not None or max_size is not None:
            low = int(min_size) if min_size is not None else 0
            high = int(max_size) if max_size is not None else 1_000_000
            size_range = (low, high)
        importance_range = None
        if min_imp is not None or max_imp is not None:
            low = float(min_imp) if min_imp is not None else 0.0
            high = float(max_imp) if max_imp is not None else 1.0
            importance_range = (low, high)
        weight_range = None
        if min_weight is not None or max_weight is not None:
            low = float(min_weight) if min_weight is not None else 0.0
            high = float(max_weight) if max_weight is not None else 1e12
            weight_range = (low, high)
        time_since_update_range = None
        if min_age is not None or max_age is not None:
            # Convert provided seconds to days for query_cluster_index
            low = (float(min_age) / 86400) if min_age is not None else 0.0
            high = (float(max_age) / 86400) if max_age is not None else 1e12
            time_since_update_range = (low, high)

        clusters = query_cluster_index(
            keyword=keyword,
            emotion_range=emotion_range,
            size_range=size_range,
            importance_range=importance_range,
            weight_range=weight_range,
            time_since_update_range=time_since_update_range,
            sort_by=sort_by,
            include_members=include_members,
        )
        sample_ids = [c.get("id") for c in clusters[:5] if isinstance(c, dict)]
        logger.bind(
            endpoint="get_clusters", num_clusters=len(clusters), sample_ids=sample_ids
        ).info("Cluster query completed")
        return jsonify({"clusters": clusters})
    except Exception as e:  # pragma: no cover - defensive
        logger.bind(endpoint="get_clusters", error=str(e)).exception(
            "Cluster retrieval failed"
        )
        return jsonify({"status": "error", "message": str(e)}), 500


@utility_bp.route("/clusters/activity", methods=["GET"])
def get_cluster_activity_endpoint():
    """Return top active clusters and fading clusters based on average importance."""
    try:
        top_raw = request.args.get("top_n")
        top_n, error = _parse_positive_query_int("top_n", top_raw, default=5)
        if error:
            return jsonify({"status": "error", "message": error}), 400

        fading_raw = request.args.get("fading_threshold")
        fading_threshold, error = _parse_positive_float_param(
            "fading_threshold", fading_raw, default=0.3
        )
        if error:
            return jsonify({"status": "error", "message": error}), 400

        activity = get_cluster_activity(
            top_n=top_n, fading_threshold=fading_threshold
        )
        return jsonify(activity)
    except Exception as e:  # pragma: no cover - defensive
        return jsonify({"status": "error", "message": str(e)}), 500
