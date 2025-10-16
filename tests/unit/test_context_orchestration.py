"""Tests for the adaptive context orchestration module."""

from memoria.core.context_orchestration import (
    ContextOrchestrationConfig,
    ContextOrchestrator,
)


def _usage_payload():
    return {
        "usage_frequency": {
            "long_term": {
                "top_records": [
                    {"memory_id": "mem-1", "access_count": 5},
                    {"memory_id": "hot", "access_count": 8},
                ]
            }
        }
    }


def test_plan_adjusts_limits_for_query_complexity():
    config = ContextOrchestrationConfig(
        enabled=True,
        baseline_limit=3,
        small_query_limit=2,
        large_query_limit=6,
        max_limit=6,
        small_query_token_threshold=3,
        token_budget=800,
        privacy_floor=-10.0,
        usage_boost_threshold=2,
        usage_boost_limit=2,
    )
    orchestrator = ContextOrchestrator(config, analytics_fetcher=_usage_payload)

    short_plan = orchestrator.plan_initial_window(mode="auto", query="hi there")
    assert short_plan.query_complexity == "short"
    assert short_plan.max_items == 3  # baseline enforces minimum

    long_query = "this request describes a complex multi-part objective for the agent"
    long_plan = orchestrator.plan_initial_window(mode="auto", query=long_query)
    assert long_plan.query_complexity == "long"
    assert long_plan.max_items == 6


def test_apply_plan_respects_privacy_and_tokens():
    config = ContextOrchestrationConfig(
        enabled=True,
        baseline_limit=2,
        small_query_limit=2,
        large_query_limit=4,
        max_limit=4,
        small_query_token_threshold=5,
        token_budget=6,
        privacy_floor=-10.0,
        usage_boost_threshold=2,
        usage_boost_limit=1,
    )
    orchestrator = ContextOrchestrator(config, analytics_fetcher=_usage_payload)
    plan = orchestrator.plan_initial_window(mode="auto", query="short question")

    rows = [
        {"memory_id": "mem-1", "summary": "Boosted item", "tokens": 4, "y_coord": 0},
        {"memory_id": "mem-2", "summary": "Secondary", "tokens": 3, "y_coord": 0},
        {"memory_id": "mem-3", "summary": "Private", "tokens": 2, "y_coord": -15},
    ]

    selected = orchestrator.apply_plan(plan, rows)
    assert [row["memory_id"] for row in selected] == ["mem-1"]


def test_apply_plan_expands_limit_for_high_usage():
    config = ContextOrchestrationConfig(
        enabled=True,
        baseline_limit=2,
        small_query_limit=2,
        large_query_limit=2,
        max_limit=4,
        small_query_token_threshold=5,
        token_budget=100,
        privacy_floor=-10.0,
        usage_boost_threshold=1,
        usage_boost_limit=2,
    )
    orchestrator = ContextOrchestrator(config, analytics_fetcher=_usage_payload)
    plan = orchestrator.plan_initial_window(mode="auto", query="short question")

    rows = [
        {"memory_id": "hot", "summary": "Most popular", "tokens": 2},
        {"memory_id": "mem-2", "summary": "Secondary", "tokens": 2},
        {"memory_id": "mem-3", "summary": "Tertiary", "tokens": 2},
    ]

    selected = orchestrator.apply_plan(plan, rows)
    assert [row["memory_id"] for row in selected] == ["hot", "mem-2", "mem-3"]
