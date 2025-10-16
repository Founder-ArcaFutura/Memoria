"""Utility wrappers for cluster-related database queries."""

from memoria.database.queries.cluster_queries import (
    get_cluster_activity as _get_cluster_activity,
)
from memoria.database.queries.cluster_queries import (
    get_heaviest_clusters as _get_heaviest_clusters,
)
from memoria.database.queries.cluster_queries import (
    query_clusters,
)


def query_cluster_index(
    keyword: str | None = None,
    emotion_range: tuple[float, float] | None = None,
    size_range: tuple[int, int] | None = None,
    importance_range: tuple[float, float] | None = None,
    weight_range: tuple[float, float] | None = None,
    time_since_update_range: tuple[float, float] | None = None,
    sort_by: str | None = None,
    include_members: bool = False,
) -> list[dict]:
    """Wrapper around :func:`query_clusters` for backward compatibility."""
    return query_clusters(
        keyword=keyword,
        emotion_range=emotion_range,
        size_range=size_range,
        importance_range=importance_range,
        weight_range=weight_range,
        time_since_update_range=time_since_update_range,
        sort_by=sort_by,
        include_members=include_members,
    )


def get_heaviest_clusters(top_n: int = 5, include_members: bool = False) -> list[dict]:
    return _get_heaviest_clusters(top_n=top_n, include_members=include_members)


def get_cluster_activity(
    top_n: int = 5, fading_threshold: float = 0.3
) -> dict[str, list[dict]]:
    return _get_cluster_activity(top_n=top_n, fading_threshold=fading_threshold)


__all__ = [
    "query_cluster_index",
    "get_heaviest_clusters",
    "get_cluster_activity",
]
