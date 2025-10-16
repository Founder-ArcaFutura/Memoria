"""Database operations for cluster tables."""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import create_engine, func, or_
from sqlalchemy.orm import Session, sessionmaker

from memoria.database.models import Base, Cluster, ClusterMember


def _get_session(connection_string: str | None = None) -> Session:
    """Create a database session using Flask's DB_PATH or configuration."""
    from memoria.config.manager import ConfigManager

    if connection_string is None:
        connection_string = os.getenv("DATABASE_URL")
        if not connection_string:
            db_path = None
            try:  # Prefer Flask app configuration when available
                from flask import current_app

                db_path = current_app.config.get("DB_PATH")  # type: ignore[assignment]
            except Exception:  # pragma: no cover - current_app not available
                db_path = None

            if db_path:
                connection_string = f"sqlite:///{db_path}"
            else:
                config = ConfigManager()
                connection_string = config.get_settings().database.connection_string

    engine = create_engine(connection_string)
    # Ensure tables exist
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def replace_clusters(
    clusters: Iterable[dict],
    connection_string: str | None = None,
    session: Session | None = None,
) -> None:
    """Replace all clusters with provided data."""
    close_session = False
    if session is None:
        session = _get_session(connection_string)
        if session.bind is not None and connection_string is None:
            connection_string = str(session.bind.url)
        close_session = True
    else:
        if connection_string is None and session.bind is not None:
            connection_string = str(session.bind.url)
    if connection_string:
        logger.info(f"Replacing clusters using database: {connection_string}")
    try:
        with session.begin():
            session.query(ClusterMember).delete()
            session.query(Cluster).delete()
            for idx, c in enumerate(clusters):
                members = c.get("members", [])
                lu = c.get("last_updated")
                if isinstance(lu, str):
                    try:
                        lu = datetime.fromisoformat(lu.replace("Z", "+00:00"))
                    except ValueError:
                        lu = None

                cluster_tokens = c.get("token_count")
                cluster_chars = c.get("char_count")
                if cluster_tokens is None or cluster_chars is None:
                    cluster_tokens = 0
                    cluster_chars = 0
                processed_members = []
                for m in members:
                    summary = m.get("summary", "")
                    tokens = m.get("tokens")
                    chars = m.get("chars")
                    if tokens is None:
                        tokens = len(summary.split())
                    if chars is None:
                        chars = len(summary)
                    cluster_tokens += tokens
                    cluster_chars += chars
                    processed_members.append(
                        {
                            "memory_id": m.get("memory_id"),
                            "anchor": m.get("anchor"),
                            "summary": summary,
                            "tokens": tokens,
                            "chars": chars,
                        }
                    )

                cluster = Cluster(
                    id=c.get("id", idx),
                    summary=c.get("summary", ""),
                    centroid=c.get("centroid"),
                    y_centroid=c.get("y_centroid"),
                    z_centroid=c.get("z_centroid"),
                    polarity=c.get("emotions", {}).get("polarity"),
                    subjectivity=c.get("emotions", {}).get("subjectivity"),
                    size=c.get("size"),
                    avg_importance=c.get("avg_importance"),
                    update_count=c.get("update_count", 0),
                    last_updated=lu,
                    weight=c.get("weight", 0.0),
                    total_tokens=cluster_tokens,
                    total_chars=cluster_chars,
                )
                session.add(cluster)
                session.flush()
                for m in processed_members:
                    session.add(
                        ClusterMember(
                            cluster_id=cluster.id,
                            memory_id=m["memory_id"],
                            anchor=m.get("anchor"),
                            summary=m.get("summary", ""),
                            tokens=m.get("tokens"),
                            chars=m.get("chars"),
                        )
                    )
    finally:
        if close_session:
            session.close()


def query_clusters(
    keyword: str | None = None,
    emotion_range: tuple[float, float] | None = None,
    size_range: tuple[int, int] | None = None,
    importance_range: tuple[float, float] | None = None,
    weight_range: tuple[float, float] | None = None,
    time_since_update_range: tuple[float, float] | None = None,
    sort_by: str | None = None,
    include_members: bool = False,
) -> list[dict]:
    """Query clusters with optional filters."""
    session = _get_session()
    try:
        q = session.query(Cluster)
        if keyword:
            kw = f"%{keyword.lower()}%"
            q = q.filter(
                or_(
                    func.lower(Cluster.summary).like(kw),
                    Cluster.members.any(func.lower(ClusterMember.anchor).like(kw)),
                    Cluster.members.any(func.lower(ClusterMember.summary).like(kw)),
                )
            )
        if emotion_range:
            q = q.filter(
                Cluster.polarity >= emotion_range[0],
                Cluster.polarity <= emotion_range[1],
            )
        if size_range:
            q = q.filter(
                Cluster.size >= size_range[0],
                Cluster.size <= size_range[1],
            )
        if importance_range:
            q = q.filter(
                Cluster.avg_importance >= importance_range[0],
                Cluster.avg_importance <= importance_range[1],
            )
        if weight_range:
            q = q.filter(
                Cluster.weight >= weight_range[0],
                Cluster.weight <= weight_range[1],
            )

        clusters = q.all()
        now = datetime.now(timezone.utc)
        results = []
        for cluster in clusters:
            time_since = None
            if cluster.last_updated:
                dt = cluster.last_updated
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                time_since = (now - dt).total_seconds() / 86400
            data = {
                "id": cluster.id,
                "summary": cluster.summary,
                "centroid": cluster.centroid,
                "y_centroid": cluster.y_centroid,
                "z_centroid": cluster.z_centroid,
                "emotions": {
                    "polarity": cluster.polarity,
                    "subjectivity": cluster.subjectivity,
                },
                "size": cluster.size,
                "avg_importance": cluster.avg_importance,
                "update_count": cluster.update_count,
                "last_updated": (
                    cluster.last_updated.isoformat() if cluster.last_updated else None
                ),
                "weight": cluster.weight,
                "time_since_update": time_since,
                "token_count": cluster.total_tokens,
                "char_count": cluster.total_chars,
            }
            if include_members:
                data["members"] = [
                    {
                        "memory_id": m.memory_id,
                        "anchor": m.anchor,
                        "summary": m.summary,
                        "tokens": m.tokens,
                        "chars": m.chars,
                    }
                    for m in cluster.members
                ]
            results.append(data)
        if time_since_update_range:
            low, high = time_since_update_range
            results = [
                r
                for r in results
                if r["time_since_update"] is not None
                and low <= r["time_since_update"] <= high
            ]
        if sort_by == "weight":
            results.sort(key=lambda c: c.get("weight", 0.0), reverse=True)
        elif sort_by == "time_since_update":
            results.sort(key=lambda c: c.get("time_since_update", float("inf")))
        return results
    finally:
        session.close()


def get_heaviest_clusters(top_n: int = 5, include_members: bool = False) -> list[dict]:
    clusters = query_clusters(sort_by="weight", include_members=include_members)
    return clusters[:top_n]


def get_cluster_activity(
    top_n: int = 5, fading_threshold: float = 0.3
) -> dict[str, list[dict]]:
    clusters = query_clusters()
    clusters.sort(key=lambda c: c.get("avg_importance", 0.0), reverse=True)
    active = clusters[:top_n]
    fading = [c for c in clusters if c.get("avg_importance", 0.0) < fading_threshold]
    return {"active": active, "fading": fading}


__all__ = [
    "replace_clusters",
    "query_clusters",
    "get_heaviest_clusters",
    "get_cluster_activity",
]
