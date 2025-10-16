"""Embedding helpers for vector search features."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

from loguru import logger

try:  # pragma: no cover - optional OpenAI dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    OpenAI = None  # type: ignore

from memoria.config.manager import ConfigManager

_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _deterministic_embedding(text: str, dimensions: int = 16) -> list[float]:
    """Return a deterministic pseudo-embedding for offline usage."""

    if not text:
        return []
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    required_bytes = dimensions * 4
    if len(digest) < required_bytes:
        digest = (digest * ((required_bytes // len(digest)) + 1))[:required_bytes]
    values: list[float] = []
    for i in range(0, required_bytes, 4):
        chunk = digest[i : i + 4]
        values.append(int.from_bytes(chunk, "big") / 0xFFFFFFFF)
    return values


def vector_search_enabled() -> bool:
    """Return ``True`` when vector search is enabled in configuration."""

    try:
        settings = ConfigManager().get_settings()
        return bool(getattr(settings, "enable_vector_search", False))
    except Exception:  # pragma: no cover - defensive, config always available
        return False


def generate_embedding(
    text: str,
    *,
    client: Any | None = None,
    model: str | None = None,
) -> list[float]:
    """Generate an embedding for ``text`` using OpenAI or a deterministic fallback."""

    text = text or ""
    if not text.strip():
        return []

    api_client = client
    if api_client is None and OpenAI is not None:
        try:  # pragma: no cover - depends on OpenAI availability
            api_client = OpenAI()
        except Exception as exc:  # pragma: no cover - no credentials or network
            logger.debug(f"OpenAI client unavailable for embeddings: {exc}")
            api_client = None

    if api_client is not None:
        try:  # pragma: no cover - depends on remote service
            response = api_client.embeddings.create(
                model=model or _DEFAULT_EMBEDDING_MODEL,
                input=text,
            )
            data = getattr(response, "data", None)
            if data:
                first = data[0]
                vector = getattr(first, "embedding", None)
                if vector is None and isinstance(first, dict):
                    vector = first.get("embedding")
                if isinstance(vector, Iterable):
                    return [float(value) for value in vector]
        except Exception as exc:  # pragma: no cover - service/credential failure
            logger.debug(f"Falling back to deterministic embedding: {exc}")

    return _deterministic_embedding(text)


def normalize_embedding(value: Any) -> list[float] | None:
    """Convert stored embedding payloads into a list of floats."""

    if value is None:
        return None
    if isinstance(value, list):
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, tuple):
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return normalize_embedding(parsed)
    return None


def vector_similarity(
    query_embedding: Sequence[float],
    memory_embedding: Sequence[float],
) -> float:
    """Return a similarity score using inverse-squared Euclidean distance."""

    if not query_embedding or not memory_embedding:
        return 0.0
    dims = min(len(query_embedding), len(memory_embedding))
    if dims == 0:
        return 0.0
    distance_squared = 0.0
    for i in range(dims):
        diff = query_embedding[i] - memory_embedding[i]
        distance_squared += diff * diff
    return 1.0 / (distance_squared + 1e-6)


__all__ = [
    "generate_embedding",
    "normalize_embedding",
    "vector_similarity",
    "vector_search_enabled",
]
