"""Dynamic context window planning for context injection."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Any

from loguru import logger

TokenEstimator = Callable[[Mapping[str, Any]], int]
AnalyticsFetcher = Callable[[], Mapping[str, Any] | None]


def _resolve_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:  # pragma: no cover - defensive conversion
        return default


def _resolve_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:  # pragma: no cover - defensive conversion
        return default


@dataclass(frozen=True)
class ContextPlan:
    """Immutable representation of a context orchestration decision."""

    mode: str
    query_complexity: str
    max_items: int
    token_budget: int
    privacy_floor: float | None
    boosted_ids: frozenset[str]


@dataclass(frozen=True)
class ContextOrchestrationConfig:
    """Configuration knobs controlling context window orchestration."""

    enabled: bool = True
    baseline_limit: int = 3
    small_query_limit: int = 3
    large_query_limit: int = 6
    max_limit: int = 8
    small_query_token_threshold: int = 32
    token_budget: int = 600
    privacy_floor: float | None = -10.0
    usage_boost_threshold: int = 3
    usage_boost_limit: int = 2
    usage_cache_seconds: float = 30.0
    analytics_window_days: int = 7
    analytics_top_n: int = 10

    @classmethod
    def from_settings(cls, settings: Any | None) -> ContextOrchestrationConfig:
        """Build configuration from a :class:`~memoria.config.settings.MemorySettings`."""

        if settings is None:
            return cls()

        data = {
            "enabled": bool(getattr(settings, "context_orchestration", True)),
            "baseline_limit": _resolve_int(getattr(settings, "context_limit", 3), 3),
            "small_query_limit": _resolve_int(
                getattr(settings, "context_small_query_limit", None)
                or getattr(settings, "context_limit", 3),
                3,
            ),
            "large_query_limit": _resolve_int(
                getattr(settings, "context_large_query_limit", None) or 6,
                6,
            ),
            "max_limit": _resolve_int(
                getattr(settings, "context_max_prompt_memories", None) or 8,
                8,
            ),
            "small_query_token_threshold": _resolve_int(
                getattr(settings, "context_small_query_token_threshold", None) or 32,
                32,
            ),
            "token_budget": _resolve_int(
                getattr(settings, "context_token_budget", None) or 600,
                600,
            ),
            "privacy_floor": _resolve_float(
                getattr(settings, "context_privacy_floor", None),
                -10.0,
            ),
            "usage_boost_threshold": _resolve_int(
                getattr(settings, "context_usage_boost_threshold", None) or 3,
                3,
            ),
            "usage_boost_limit": _resolve_int(
                getattr(settings, "context_usage_boost_limit", None) or 2,
                2,
            ),
            "usage_cache_seconds": float(
                getattr(settings, "context_usage_cache_seconds", 30.0)
            ),
            "analytics_window_days": _resolve_int(
                getattr(settings, "context_analytics_window_days", None) or 7,
                7,
            ),
            "analytics_top_n": _resolve_int(
                getattr(settings, "context_analytics_top_n", None) or 10,
                10,
            ),
        }
        return cls(**data)


class ContextOrchestrator:
    """Derive context window sizes using search results and analytics telemetry."""

    def __init__(
        self,
        config: ContextOrchestrationConfig,
        *,
        analytics_fetcher: AnalyticsFetcher | None = None,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        self.config = config
        self._analytics_fetcher = analytics_fetcher
        self._token_estimator = token_estimator or self._estimate_tokens
        self._usage_cache: tuple[float, dict[str, int]] | None = None

    def plan_initial_window(
        self,
        *,
        mode: str,
        query: str | None = None,
        provider_name: str | None = None,  # noqa: ARG002 - reserved for future use
    ) -> ContextPlan:
        """Return an initial plan controlling limits for context retrieval."""

        if not self.config.enabled:
            return ContextPlan(
                mode=mode,
                query_complexity="disabled",
                max_items=self.config.baseline_limit,
                token_budget=self.config.token_budget,
                privacy_floor=self.config.privacy_floor,
                boosted_ids=frozenset(),
            )

        query_text = (query or "").strip()
        word_count = len(query_text.split()) if query_text else 0
        complexity = (
            "short" if word_count <= self.config.small_query_token_threshold else "long"
        )

        base_limit = (
            self.config.small_query_limit
            if complexity == "short"
            else self.config.large_query_limit
        )
        base_limit = max(base_limit, self.config.baseline_limit)
        base_limit = min(base_limit, self.config.max_limit)

        boosted_ids = frozenset(self._get_boosted_ids())

        return ContextPlan(
            mode=mode,
            query_complexity=complexity,
            max_items=base_limit,
            token_budget=self.config.token_budget,
            privacy_floor=self.config.privacy_floor,
            boosted_ids=boosted_ids,
        )

    def apply_plan(
        self,
        plan: ContextPlan,
        results: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        """Apply orchestration rules to the retrieved results."""

        if not results:
            return []

        allowed: list[Mapping[str, Any]] = []
        boosted_bucket: list[Mapping[str, Any]] = []
        regular_bucket: list[Mapping[str, Any]] = []

        for item in results:
            if plan.privacy_floor is not None:
                privacy_value = self._resolve_privacy(item)
                if privacy_value is not None and privacy_value <= plan.privacy_floor:
                    logger.debug(
                        "Skipping memory %s due to privacy floor %.2f",
                        item.get("memory_id"),
                        plan.privacy_floor,
                    )
                    continue

            if item.get("memory_id") in plan.boosted_ids:
                boosted_bucket.append(item)
            else:
                regular_bucket.append(item)

        boosted_count = len(boosted_bucket)
        effective_limit = plan.max_items
        if boosted_count:
            effective_limit = min(
                plan.max_items + min(boosted_count, self.config.usage_boost_limit),
                self.config.max_limit,
            )

        ordered = boosted_bucket + regular_bucket
        total_tokens = 0
        for item in ordered:
            if len(allowed) >= effective_limit:
                break
            token_estimate = max(self._token_estimator(item), 0)
            if plan.token_budget > 0 and allowed:
                if total_tokens + token_estimate > plan.token_budget:
                    logger.debug(
                        "Token budget reached (budget=%s, current=%s, next=%s)",
                        plan.token_budget,
                        total_tokens,
                        token_estimate,
                    )
                    break
            total_tokens += token_estimate
            allowed.append(item)

        if not allowed and ordered:
            # Guarantee at least one memory if nothing else satisfied budget thresholds.
            first = ordered[0]
            allowed.append(first)

        return allowed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_tokens(self, item: Mapping[str, Any]) -> int:
        value = (
            item.get("token_count") or item.get("tokens") or item.get("approx_tokens")
        )
        if value is not None:
            return _resolve_int(value, 0)
        summary = item.get("searchable_content") or item.get("summary") or ""
        return max(len(str(summary).split()), 0)

    def _resolve_privacy(self, item: Mapping[str, Any]) -> float | None:
        privacy_keys = ("privacy", "privacy_score", "y_coord", "privacy_axis")
        for key in privacy_keys:
            if key in item:
                return _resolve_float(item.get(key))
        return None

    def _get_boosted_ids(self) -> Iterable[str]:
        usage_counts = self._get_usage_counts()
        threshold = max(0, self.config.usage_boost_threshold)
        for memory_id, access_count in usage_counts.items():
            if access_count >= threshold:
                yield memory_id

    def _get_usage_counts(self) -> dict[str, int]:
        if self._analytics_fetcher is None:
            return {}

        now = monotonic()
        if (
            self._usage_cache
            and now - self._usage_cache[0] < self.config.usage_cache_seconds
        ):
            return dict(self._usage_cache[1])

        try:
            metrics = self._analytics_fetcher() or {}
        except Exception as exc:  # pragma: no cover - analytics backends optional
            logger.debug("Analytics fetcher failed: %s", exc)
            metrics = {}

        usage_payload = (
            metrics.get("usage_frequency") if isinstance(metrics, Mapping) else {}
        )
        counts: dict[str, int] = {}

        def _ingest(records: Iterable[Any]) -> None:
            for record in records:
                memory_id = None
                access_count = None
                if isinstance(record, Mapping):
                    memory_id = record.get("memory_id")
                    access_count = record.get("access_count")
                else:
                    memory_id = getattr(record, "memory_id", None)
                    access_count = getattr(record, "access_count", None)
                if not memory_id:
                    continue
                counts[str(memory_id)] = max(
                    counts.get(str(memory_id), 0), _resolve_int(access_count, 0)
                )

        if isinstance(usage_payload, Mapping):
            for key in ("long_term", "short_term"):
                scope = usage_payload.get(key)
                if isinstance(scope, Mapping):
                    _ingest(scope.get("top_records") or [])

        self._usage_cache = (now, counts)
        return counts


__all__ = ["ContextOrchestrator", "ContextOrchestrationConfig", "ContextPlan"]
