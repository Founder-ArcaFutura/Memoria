"""Build and persist memory clusters used by the Memoria platform."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memoria.config.manager import ConfigManager
from memoria.database.backup_guard import backup_guard
from memoria.database.models import LongTermMemory, ShortTermMemory
from memoria.database.queries.cluster_queries import replace_clusters

if TYPE_CHECKING:
    import numpy as np


# Paths
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = PACKAGE_ROOT if (PACKAGE_ROOT / "logs").exists() else Path.cwd()
LOG_DIR = Path(os.getenv("MEMORIA_LOG_DIR", str(DEFAULT_WORKSPACE / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "index_clusters.log"

# Logging configuration
logger.add(LOG_FILE)

MODEL_NAME = "all-MiniLM-L6-v2"

# Similarity threshold to join an existing cluster
SIM_THRESHOLD = float(os.getenv("CLUSTER_SIM_THRESHOLD", 0.8))
# Exponential decay time constant in seconds (default: 7 days)
DECAY_SECONDS = float(os.getenv("CLUSTER_DECAY_SECONDS", 7 * 24 * 60 * 60))
try:
    CENTROID_SMOOTHING = float(os.getenv("CLUSTER_CENTROID_SMOOTHING", "1.0"))
except ValueError:
    CENTROID_SMOOTHING = 1.0
if CENTROID_SMOOTHING <= 0:
    CENTROID_SMOOTHING = 1.0
CENTROID_SMOOTHING = min(CENTROID_SMOOTHING, 1.0)

# Limit the number of clusters logged for brevity
MAX_LOG_CLUSTERS = 5

# Index build status shared with the API
INDEX_STATUS: dict[str, object] = {
    "state": "idle",  # idle | running | complete | error
    "current": 0,
    "total": 0,
    "error": None,
}

# Determine whether vector clustering is enabled and if dependencies are present
try:
    _settings = ConfigManager().get_settings()
    ENABLE_VECTOR_CLUSTERING = getattr(_settings, "enable_vector_clustering", False)
except Exception:
    ENABLE_VECTOR_CLUSTERING = False

VECTOR_DEPS_AVAILABLE = False
if ENABLE_VECTOR_CLUSTERING:
    try:  # pragma: no cover - import guarded by config
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from textblob import TextBlob

        VECTOR_DEPS_AVAILABLE = True
    except Exception:  # pragma: no cover - depends on optional deps
        logger.warning(
            "Vector clustering enabled but dependencies are missing; falling back to heuristic clustering"
        )


def _blend_centroid(previous: float | None, target: float) -> float:
    """Blend the existing centroid toward ``target`` using smoothing."""

    if previous is None or CENTROID_SMOOTHING >= 1.0:
        return target
    alpha = CENTROID_SMOOTHING
    return previous + alpha * (target - previous)


def _update_cluster_centroids(
    cluster: dict, y_value: float | None, z_value: float | None
) -> None:
    """Update running centroid statistics for ``cluster``."""

    if y_value is not None:
        y_sum = cluster.get("_y_sum", 0.0) + y_value
        y_count = cluster.get("_y_count", 0) + 1
        cluster["_y_sum"] = y_sum
        cluster["_y_count"] = y_count
        avg_y = y_sum / y_count
        cluster["y_centroid"] = _blend_centroid(cluster.get("y_centroid"), avg_y)
    elif cluster.get("_y_count", 0) == 0:
        cluster["y_centroid"] = None

    if z_value is not None:
        z_sum = cluster.get("_z_sum", 0.0) + z_value
        z_count = cluster.get("_z_count", 0) + 1
        cluster["_z_sum"] = z_sum
        cluster["_z_count"] = z_count
        avg_z = z_sum / z_count
        cluster["z_centroid"] = _blend_centroid(cluster.get("z_centroid"), avg_z)
    elif cluster.get("_z_count", 0) == 0:
        cluster["z_centroid"] = None


def _finalize_centroid_state(cluster: dict) -> None:
    """Remove running centroid bookkeeping keys before persistence."""

    cluster.pop("_y_sum", None)
    cluster.pop("_y_count", None)
    cluster.pop("_z_sum", None)
    cluster.pop("_z_count", None)


def _reset_status() -> None:
    """Reset progress tracking info."""
    INDEX_STATUS.update({"state": "idle", "current": 0, "total": 0, "error": None})


def get_status() -> dict[str, object]:
    """Return a copy of the current index build status."""
    return INDEX_STATUS.copy()


def set_vector_search_enabled(
    enabled: bool,
    *,
    persist: bool = False,
    config_path: Path | None = None,
) -> bool:
    """Toggle vector search support within the persisted configuration."""

    config = ConfigManager()
    try:  # pragma: no cover - defensive load
        config.auto_load()
    except Exception:
        pass

    config.update_setting("enable_vector_search", bool(enabled))
    state = "enabled" if enabled else "disabled"
    logger.info(f"Vector search {state}")

    if persist:
        target = Path(config_path) if config_path else Path("memoria.json")
        try:
            config.save_to_file(target)
        except Exception as exc:  # pragma: no cover - filesystem dependent
            logger.warning(f"Failed to persist vector search setting: {exc}")
    return bool(enabled)


def enable_vector_search(
    *, persist: bool = False, config_path: Path | None = None
) -> bool:
    """Convenience wrapper to enable vector search."""

    return set_vector_search_enabled(True, persist=persist, config_path=config_path)


def disable_vector_search(
    *, persist: bool = False, config_path: Path | None = None
) -> bool:
    """Convenience wrapper to disable vector search."""

    return set_vector_search_enabled(False, persist=persist, config_path=config_path)


def resolve_database_url(config: ConfigManager | None = None) -> str:
    """Determine the database connection string used for clustering.

    Priority order:
    1. ``DATABASE_URL`` environment variable
    2. Flask application's ``DATABASE_URL`` or ``DB_PATH``
    3. Existing ``ConfigManager`` connection string
    """

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    try:  # Attempt to pull from an active Flask application
        from flask import current_app

        try:
            url = current_app.config.get("DATABASE_URL")
            if url:
                return url
            db_path = current_app.config.get("DB_PATH")
            if db_path:
                return f"sqlite:///{db_path}"
        except RuntimeError:
            # No active application context
            pass
    except Exception:
        pass

    if config is None:
        config = ConfigManager()
        try:
            config.auto_load()
        except Exception:
            pass
    return config.get_settings().database.connection_string


MEMORY_SOURCE_MODELS: dict[str, type] = {
    "LongTermMemory": LongTermMemory,
    "ShortTermMemory": ShortTermMemory,
}


def fetch_texts(
    conn_str: str,
    sources: Sequence[str] = ("LongTermMemory", "ShortTermMemory"),
) -> list[dict[str, object]]:
    """Fetch memory metadata used for clustering."""
    engine = create_engine(conn_str)
    Session = sessionmaker(bind=engine)
    rows: list[dict[str, object]] = []

    with Session() as session:
        seen_models: set[type] = set()
        selected_sources: list[str] = []
        for source in sources:
            model = MEMORY_SOURCE_MODELS.get(source)
            if model is None:
                logger.warning(f"Skipping unknown memory source '{source}'")
                continue
            if model in seen_models:
                continue
            seen_models.add(model)
            selected_sources.append(source)
            try:
                result = session.query(
                    model.memory_id,
                    model.summary,
                    model.searchable_content,
                    model.symbolic_anchors,
                    model.importance_score,
                    model.y_coord,
                    model.z_coord,
                ).all()
                for (
                    memory_id,
                    summary,
                    text,
                    anchors,
                    importance,
                    y_coord,
                    z_coord,
                ) in result:
                    content = summary or text
                    if not content:
                        continue
                    anchor = ""
                    if anchors:
                        if isinstance(anchors, str):
                            try:
                                anchor_list = json.loads(anchors)
                            except Exception:
                                anchor_list = [
                                    a.strip() for a in anchors.split(",") if a.strip()
                                ]
                        else:
                            anchor_list = anchors
                        anchor = anchor_list[0] if anchor_list else ""
                    rows.append(
                        {
                            "memory_id": memory_id,
                            "anchor": anchor,
                            "summary": content,
                            "importance": float(importance or 0),
                            "y_coord": float(y_coord) if y_coord is not None else None,
                            "z_coord": float(z_coord) if z_coord is not None else None,
                        }
                    )

            except Exception as exc:  # pragma: no cover - table may not exist
                logger.warning(
                    f"Skipping table {getattr(model, '__tablename__', model)}: {exc}"
                )
    if selected_sources:
        logger.info(
            f"Fetched {len(rows)} rows from the database using sources: {', '.join(selected_sources)}"
        )
    else:
        logger.info(f"Fetched {len(rows)} rows from the database")
    return rows


def embed_texts(texts: list[str], SentenceTransformer) -> np.ndarray:
    """Encode texts into embeddings."""
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(texts, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _heuristic_summary(texts: list[str], num_words: int = 40) -> str:
    """Generate a simple summary from the first words and common keywords."""
    combined = " ".join(texts)
    words = combined.split()
    excerpt = " ".join(words[:num_words])

    import re
    from collections import Counter

    tokens = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in words]
    stopwords = {
        "the",
        "and",
        "of",
        "to",
        "a",
        "in",
        "is",
        "it",
        "for",
        "on",
        "with",
        "as",
        "that",
        "this",
        "by",
        "an",
        "be",
        "are",
        "or",
        "from",
        "at",
        "which",
    }
    tokens = [t for t in tokens if t and t not in stopwords]
    keywords = [w for w, _ in Counter(tokens).most_common(5)]
    if keywords:
        return f"{excerpt} (keywords: {', '.join(keywords)})"
    return excerpt


def summarize_cluster(texts: list[str]) -> str:
    """Summarize a cluster of texts using either heuristics or an LLM."""
    env_flag = os.getenv("MEMORIA_ENABLE_VECTOR_CLUSTERING")
    if env_flag is not None:
        use_llm = env_flag.lower() not in ("0", "false", "no")
    else:
        try:
            use_llm = ConfigManager().get_settings().enable_vector_clustering
        except Exception:
            use_llm = False

    if not use_llm:
        return _heuristic_summary(texts)

    client = OpenAI()
    prompt = (
        "Summarize the following texts, capturing the main theme and emotional tone:\n"
        + "\n".join(texts)
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - depends on external API
        logger.error(f"LLM summarization failed: {exc}")
        return ""


def analyze_emotions(texts: list[str], TextBlob, np) -> dict:
    """Compute average sentiment scores for a list of texts."""
    polarities = [TextBlob(t).sentiment.polarity for t in texts]
    subjectivities = [TextBlob(t).sentiment.subjectivity for t in texts]
    return {
        "polarity": float(np.mean(polarities)),
        "subjectivity": float(np.mean(subjectivities)),
    }


def _heuristic_clusters(rows: list[dict[str, object]], conn_str: str) -> list[dict]:
    """Build clusters using simple anchor grouping without vector embeddings."""
    clusters: list[dict] = []
    anchor_map: dict[str, dict] = {}
    for i, row in enumerate(rows):
        INDEX_STATUS["current"] = i + 1
        memory_id = row["memory_id"]
        anchor = row.get("anchor", "") or ""
        summary = row.get("summary", "")
        if not isinstance(summary, str):
            summary = str(summary)
        importance = float(row.get("importance", 0.0) or 0.0)
        y_coord = row.get("y_coord")
        z_coord = row.get("z_coord")
        if anchor and anchor in anchor_map:
            cluster = anchor_map[anchor]
        else:
            cluster = {
                "id": len(clusters),
                "anchor": anchor,
                "members": [],
                "token_count": 0,
                "char_count": 0,
                "update_count": 0,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "weight": 1.0,
                "y_centroid": None,
                "z_centroid": None,
                "_y_sum": 0.0,
                "_y_count": 0,
                "_z_sum": 0.0,
                "_z_count": 0,
            }
            clusters.append(cluster)
            if anchor:
                anchor_map[anchor] = cluster
        cluster["members"].append(
            {
                "memory_id": memory_id,
                "anchor": anchor,
                "summary": summary,
                "importance_score": importance,
                "y_coord": y_coord,
                "z_coord": z_coord,
            }
        )
        tokens = len(summary.split())
        chars = len(summary)
        cluster["token_count"] += tokens
        cluster["char_count"] += chars
        cluster["update_count"] += 1
        cluster["last_updated"] = datetime.now(timezone.utc).isoformat()
        _update_cluster_centroids(cluster, y_coord, z_coord)

    for cluster in clusters:
        member_summaries = [m["summary"] for m in cluster["members"]]
        cluster["summary"] = _heuristic_summary(member_summaries)
        cluster["centroid"] = None
        cluster["emotions"] = {"polarity": 0.0, "subjectivity": 0.0}
        cluster["size"] = len(cluster["members"])
        cluster["avg_importance"] = (
            sum(m["importance_score"] for m in cluster["members"]) / cluster["size"]
            if cluster["size"]
            else 0.0
        )
        _finalize_centroid_state(cluster)

    replace_clusters(clusters, connection_string=conn_str)
    total_tokens = sum(c.get("token_count", 0) for c in clusters)
    total_chars = sum(c.get("char_count", 0) for c in clusters)
    logger.info(
        f"Stored {len(clusters)} heuristic clusters in database ({total_tokens} tokens, {total_chars} chars)"
    )
    INDEX_STATUS.update(state="complete", current=INDEX_STATUS["total"])
    return clusters


def build_index(
    output_path: Path | ConfigManager | None = None,
    sources: Sequence[str] = ("LongTermMemory", "ShortTermMemory"),
) -> list[dict]:
    """Build cluster index, store results, and return the clusters.

    Parameters
    ----------
    output_path:
        Optional path or ``ConfigManager`` instance used when invoking the
        index builder from the command line.  The value is ignored when called
        from the API.
    sources:
        Ordered collection of memory tables to include when fetching texts.
        Unknown names are skipped.  Defaults to long- and short-term memory.

    Returns
    -------
    list[dict]
        The list of cluster dictionaries that were persisted to the database.

    Raises
    ------
    RuntimeError
        If there are no texts available for clustering.

    Other exceptions are propagated to the caller.
    """
    _reset_status()
    INDEX_STATUS["state"] = "running"
    config = ConfigManager()
    try:
        config.auto_load()
    except Exception:
        pass
    conn_str = resolve_database_url(config)
    logger.info(f"Resolved DATABASE_URL: {conn_str}")
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        config.update_setting("database.connection_string", database_url)

    rows = fetch_texts(conn_str, sources=sources)
    INDEX_STATUS["total"] = len(rows)
    if not rows:
        msg = "No texts available for clustering"
        logger.error(msg)
        INDEX_STATUS.update(state="error", error=msg)
        raise RuntimeError(msg)

    settings = config.get_settings()
    if (
        not getattr(settings, "enable_vector_clustering", False)
        or not VECTOR_DEPS_AVAILABLE
    ):
        if not getattr(settings, "enable_vector_clustering", False):
            logger.info("Vector clustering disabled; using heuristic clustering")
        else:
            logger.warning(
                "Vector clustering dependencies are missing; falling back to heuristic clustering"
            )
        return _heuristic_clusters(rows, conn_str)

    try:
        summaries = [str(r.get("summary", "")) for r in rows]
        embeddings = embed_texts(summaries, SentenceTransformer)

        clusters: list[dict] = []
        for i, row in enumerate(rows):
            INDEX_STATUS["current"] = i + 1
            memory_id = row["memory_id"]
            anchor = row.get("anchor", "") or ""
            summary = row.get("summary", "")
            if not isinstance(summary, str):
                summary = str(summary)
            importance = float(row.get("importance", 0.0) or 0.0)
            y_coord = row.get("y_coord")
            z_coord = row.get("z_coord")
            embedding = embeddings[i]
            tokens = len(summary.split())
            chars = len(summary)
            best_idx, best_sim = None, -1.0
            for idx, cluster in enumerate(clusters):
                if anchor and anchor == cluster.get("anchor"):
                    sim = 1.0
                else:
                    sim = cosine_similarity(embedding, cluster["centroid"])
                if sim > best_sim:
                    best_idx, best_sim = idx, sim
            if best_idx is not None and best_sim >= SIM_THRESHOLD:
                cluster = clusters[best_idx]
                n = len(cluster["members"])
                cluster["centroid"] = (cluster["centroid"] * n + embedding) / (n + 1)
                cluster["members"].append(
                    {
                        "memory_id": memory_id,
                        "anchor": anchor,
                        "summary": summary,
                        "importance_score": importance,
                        "y_coord": y_coord,
                        "z_coord": z_coord,
                    }
                )
                cluster["token_count"] += tokens
                cluster["char_count"] += chars
                cluster["update_count"] += 1
                cluster["last_updated"] = datetime.now(timezone.utc).isoformat()
                age = datetime.now(timezone.utc) - datetime.fromisoformat(
                    cluster["last_updated"]
                )

                cluster["weight"] = cluster["update_count"] * math.exp(
                    -age.total_seconds() / DECAY_SECONDS
                )
                _update_cluster_centroids(cluster, y_coord, z_coord)
                logger.info(
                    f"Added memory {memory_id} to cluster {best_idx} (sim={best_sim:.3f}, weight={cluster['weight']:.3f})"
                )
            else:
                cluster = {
                    "id": len(clusters),
                    "anchor": anchor,
                    "centroid": embedding,
                    "members": [
                        {
                            "memory_id": memory_id,
                            "anchor": anchor,
                            "summary": summary,
                            "importance_score": importance,
                            "y_coord": y_coord,
                            "z_coord": z_coord,
                        }
                    ],
                    "update_count": 1,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "weight": 1.0,
                    "token_count": tokens,
                    "char_count": chars,
                    "y_centroid": None,
                    "z_centroid": None,
                    "_y_sum": 0.0,
                    "_y_count": 0,
                    "_z_sum": 0.0,
                    "_z_count": 0,
                }
                clusters.append(cluster)
                _update_cluster_centroids(cluster, y_coord, z_coord)
                logger.info(f"Memory {memory_id} started new cluster {cluster['id']}")

        for cluster in clusters:
            member_summaries = [m["summary"] for m in cluster["members"]]
            summary = summarize_cluster(member_summaries)
            emotions = analyze_emotions(member_summaries, TextBlob, np)
            age = datetime.now(timezone.utc) - datetime.fromisoformat(
                cluster["last_updated"]
            )
            cluster["weight"] = cluster["update_count"] * math.exp(
                -age.total_seconds() / DECAY_SECONDS
            )
            cluster["summary"] = summary
            cluster["centroid"] = cluster["centroid"].tolist()
            cluster["emotions"] = emotions
            cluster["size"] = len(cluster["members"])
            cluster["avg_importance"] = (
                sum(m["importance_score"] for m in cluster["members"]) / cluster["size"]
                if cluster["size"]
                else 0.0
            )
            _finalize_centroid_state(cluster)

        replace_clusters(clusters, connection_string=conn_str)
        total_tokens = sum(c.get("token_count", 0) for c in clusters)
        total_chars = sum(c.get("char_count", 0) for c in clusters)
        logger.info(
            f"Stored {len(clusters)} clusters in database ({total_tokens} tokens, {total_chars} chars)"
        )
        INDEX_STATUS.update(state="complete", current=INDEX_STATUS["total"])

        # Return the clusters we just persisted so callers don't need to
        # perform a separate query.

        return clusters
    except Exception as exc:
        INDEX_STATUS.update(state="error", error=str(exc))
        raise


def main() -> None:
    config = ConfigManager()
    try:
        config.auto_load()
    except Exception:
        pass
    if not config.get_settings().enable_cluster_indexing:
        logger.info("Cluster indexing disabled; exiting")
        return
    conn_str = resolve_database_url(config)
    with backup_guard(conn_str, ignore_tables={"clusters", "cluster_members"}):
        build_index()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # pragma: no cover - top-level logging

        logger.exception("Indexing failed")
        raise
