from __future__ import annotations

import json

import pytest

from memoria.tools.benchmark import (
    BenchmarkRecord,
    compute_precision,
    compute_recall,
    export_markdown_report,
    generate_markdown_report,
    iter_records,
    persist_benchmark_records,
    summarise_by_suite,
    summarise_records,
)


def make_records() -> list[BenchmarkRecord]:
    return [
        BenchmarkRecord(
            query_id="q1",
            true_positives=4,
            false_positives=1,
            false_negatives=2,
            policy_compliant=True,
            cost=0.0025,
            latency_ms=1200.0,
            provider="openai",
            metadata={"suite": "alpha", "scenario": "s1"},
        ),
        BenchmarkRecord(
            query_id="q2",
            true_positives=1,
            false_positives=1,
            false_negatives=0,
            policy_compliant=False,
            cost=0.0010,
            latency_ms=800.0,
            provider="anthropic",
            metadata={"suite": "alpha", "scenario": "s1"},
        ),
        BenchmarkRecord(
            query_id="q3",
            true_positives=0,
            false_positives=0,
            false_negatives=3,
            latency_ms=1500.0,
            provider="openai",
            metadata={"suite": "beta", "scenario": "s2"},
        ),
    ]


def test_precision_recall_helpers() -> None:
    assert compute_precision(5, 5) == pytest.approx(0.5)
    assert compute_recall(5, 5) == pytest.approx(0.5)


def test_summarise_records_returns_expected_metrics() -> None:
    summary = summarise_records(make_records())
    assert summary.total_queries == 3
    assert summary.total_true_positives == 5
    assert summary.total_false_positives == 2
    assert summary.total_false_negatives == 5
    assert summary.precision == pytest.approx(5 / 7)
    assert summary.recall == pytest.approx(5 / 10)
    assert summary.policy_compliance_rate == pytest.approx(0.5)
    assert summary.total_cost == pytest.approx(0.0035)
    assert summary.cost_per_query == pytest.approx(0.0035 / 2)
    assert summary.latency_average == pytest.approx((1200.0 + 800.0 + 1500.0) / 3)
    assert summary.latency_p50 == pytest.approx(1200.0)
    assert summary.latency_p95 == pytest.approx(1470.0)
    assert summary.provider_mix["openai"]["count"] == 2
    assert summary.provider_mix["anthropic"]["count"] == 1
    assert len(summary.cost_curve) == 2


def test_summarise_by_suite_groups_records() -> None:
    grouped = summarise_by_suite(make_records())
    assert set(grouped) == {"alpha", "beta"}
    assert grouped["alpha"].total_queries == 2
    assert grouped["beta"].total_queries == 1


def test_persist_benchmark_records_jsonl(tmp_path) -> None:
    path = tmp_path / "benchmark.jsonl"
    persist_benchmark_records(make_records(), path)

    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    assert len(rows) == 3
    assert rows[0]["query_id"] == "q1"
    assert rows[1]["cost"] == pytest.approx(0.0010)
    assert rows[0]["latency_ms"] == pytest.approx(1200.0)


def test_persist_benchmark_records_parquet(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "benchmark.parquet"

    persist_benchmark_records(make_records(), path, format="parquet")

    assert path.exists()


def test_persist_benchmark_records_sql(tmp_path) -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    path = tmp_path / "benchmark.sql.json"
    dsn = f"sqlite:///{tmp_path/'analytics.db'}"

    persist_benchmark_records(make_records(), path, format="sql", dsn=dsn)

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["destination"] == "sql"
    assert manifest["row_count"] == 3

    engine = sqlalchemy.create_engine(dsn)
    with engine.connect() as connection:
        result = connection.execute(
            sqlalchemy.text("SELECT COUNT(*) FROM benchmark_records")
        )
        assert result.scalar() == 3


def test_persist_benchmark_records_sql_requires_dsn(tmp_path) -> None:
    with pytest.raises(ValueError):
        persist_benchmark_records(
            make_records(), tmp_path / "benchmark.sql.json", format="sql"
        )


def test_generate_markdown_report_contains_metrics(tmp_path) -> None:
    report = generate_markdown_report(make_records())
    assert "Benchmark Summary" in report
    assert "Precision" in report
    assert "Latency" in report
    assert "Provider Mix" in report

    output_path = tmp_path / "summary.md"
    export_markdown_report(make_records(), output_path)
    assert output_path.read_text(encoding="utf-8").startswith("# Benchmark Summary")


def test_iter_records_round_trip() -> None:
    raw = [
        {
            "query_id": "q1",
            "true_positives": 1,
            "false_positives": 0,
            "false_negatives": 0,
            "policy_compliant": True,
            "latency_ms": 500.0,
            "provider": "openai",
            "model": "gpt-4",
            "cost_components": {"ingest": 0.5},
            "metadata": {"note": "ok"},
        }
    ]

    records = iter_records(raw)
    assert len(records) == 1
    assert records[0].precision() == pytest.approx(1.0)
    assert records[0].metadata == {"note": "ok"}
    assert records[0].latency_ms == 500.0
    assert records[0].provider == "openai"
    assert records[0].model == "gpt-4"
    assert records[0].cost_components == {"ingest": 0.5}
