"""Benchmarking utilities for quality and cost evaluations."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BenchmarkRecord:
    """Container for storing per-query benchmark outcomes."""

    query_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    policy_compliant: bool | None = None
    cost: float | None = None
    latency_ms: float | None = None
    provider: str | None = None
    model: str | None = None
    cost_components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def precision(self) -> float:
        """Return the precision for this record."""

        return compute_precision(self.true_positives, self.false_positives)

    def recall(self) -> float:
        """Return the recall for this record."""

        return compute_recall(self.true_positives, self.false_negatives)

    def to_serializable_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable representation of the record."""

        payload: dict[str, Any] = {
            "query_id": self.query_id,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }
        if self.policy_compliant is not None:
            payload["policy_compliant"] = self.policy_compliant
        if self.cost is not None:
            payload["cost"] = self.cost
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        if self.provider is not None:
            payload["provider"] = self.provider
        if self.model is not None:
            payload["model"] = self.model
        if self.cost_components:
            payload["cost_components"] = self.cost_components
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(slots=True)
class BenchmarkSummary:
    """Aggregate view of benchmark metrics."""

    total_queries: int
    total_true_positives: int
    total_false_positives: int
    total_false_negatives: int
    precision: float
    recall: float
    f1_score: float
    policy_compliance_rate: float | None
    total_cost: float | None
    cost_per_query: float | None
    latency_p50: float | None
    latency_p95: float | None
    latency_average: float | None
    provider_mix: dict[str, Any] = field(default_factory=dict)
    cost_curve: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation."""

        return {
            "total_queries": self.total_queries,
            "true_positives": self.total_true_positives,
            "false_positives": self.total_false_positives,
            "false_negatives": self.total_false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "policy_compliance_rate": self.policy_compliance_rate,
            "total_cost": self.total_cost,
            "cost_per_query": self.cost_per_query,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_average": self.latency_average,
            "provider_mix": self.provider_mix,
            "cost_curve": self.cost_curve,
        }


def compute_precision(true_positives: int, false_positives: int) -> float:
    """Compute precision as TP / (TP + FP)."""

    denominator = true_positives + false_positives
    if denominator == 0:
        return 0.0
    return true_positives / denominator


def compute_recall(true_positives: int, false_negatives: int) -> float:
    """Compute recall as TP / (TP + FN)."""

    denominator = true_positives + false_negatives
    if denominator == 0:
        return 0.0
    return true_positives / denominator


def compute_f1(precision: float, recall: float) -> float:
    """Return the harmonic mean of precision and recall."""

    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_policy_compliance_rate(
    compliant_count: int, evaluated_count: int
) -> float | None:
    """Compute the policy compliance rate."""

    if evaluated_count == 0:
        return None
    return compliant_count / evaluated_count


def compute_cost_per_query(total_cost: float, total_queries: int) -> float | None:
    """Return the average cost per query."""

    if total_queries == 0:
        return None
    return total_cost / total_queries


def _compute_percentile(values: Sequence[float], percentile: float) -> float:
    """Return the percentile (0-1) for ``values`` using linear interpolation."""

    if not values:
        raise ValueError("cannot compute percentile of empty sequence")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be within [0, 1]")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    if lower_index == upper_index:
        return lower_value
    fraction = position - lower_index
    return lower_value + (upper_value - lower_value) * fraction


def _build_provider_mix(records: Sequence[BenchmarkRecord]) -> dict[str, Any]:
    provider_counts: Counter[str] = Counter()
    provider_costs: dict[str, float] = {}

    for record in records:
        provider = record.provider or record.metadata.get("provider")
        if not provider:
            continue
        provider = str(provider)
        provider_counts[provider] += 1
        cost = record.cost
        if cost is None and record.cost_components:
            cost = sum(record.cost_components.values())
        if cost is not None:
            provider_costs[provider] = provider_costs.get(provider, 0.0) + float(cost)

    if not provider_counts:
        return {}

    total_count = sum(provider_counts.values())
    total_cost = sum(provider_costs.values())
    provider_mix: dict[str, Any] = {}
    for provider, count in provider_counts.items():
        entry: dict[str, Any] = {
            "count": count,
            "share": count / total_count if total_count else 0.0,
        }
        if provider_costs.get(provider):
            entry["cost"] = float(provider_costs[provider])
            entry["cost_share"] = (
                float(provider_costs[provider]) / total_cost if total_cost else None
            )
        provider_mix[provider] = entry
    return provider_mix


def _build_cost_curve(records: Sequence[BenchmarkRecord]) -> list[dict[str, Any]]:
    cumulative_cost = 0.0
    curve: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        incremental_cost: float | None = None
        if record.cost is not None:
            incremental_cost = float(record.cost)
        elif record.cost_components:
            incremental_cost = float(sum(record.cost_components.values()))
        if incremental_cost is None:
            continue
        cumulative_cost += incremental_cost
        curve.append(
            {
                "query_index": index,
                "query_id": record.query_id,
                "cumulative_cost": cumulative_cost,
                "provider": record.provider or record.metadata.get("provider"),
            }
        )
    return curve


def summarise_records(records: Sequence[BenchmarkRecord]) -> BenchmarkSummary:
    """Aggregate a collection of records into a summary."""

    total_tp = sum(record.true_positives for record in records)
    total_fp = sum(record.false_positives for record in records)
    total_fn = sum(record.false_negatives for record in records)
    precision = compute_precision(total_tp, total_fp)
    recall = compute_recall(total_tp, total_fn)
    f1_score = compute_f1(precision, recall)

    compliance_values = [
        record.policy_compliant
        for record in records
        if record.policy_compliant is not None
    ]
    compliant_count = sum(1 for value in compliance_values if value)
    compliance_rate = compute_policy_compliance_rate(
        compliant_count, len(compliance_values)
    )

    costs: list[float] = []
    for record in records:
        if record.cost is not None:
            costs.append(float(record.cost))
        elif record.cost_components:
            costs.append(float(sum(record.cost_components.values())))
    total_cost = sum(costs) if costs else None
    cost_per_query = None
    if total_cost is not None:
        cost_per_query = compute_cost_per_query(total_cost, len(costs))

    latencies = [
        float(record.latency_ms) for record in records if record.latency_ms is not None
    ]
    latency_p50: float | None = None
    latency_p95: float | None = None
    latency_average: float | None = None
    if latencies:
        latency_average = sum(latencies) / len(latencies)
        latency_p50 = _compute_percentile(latencies, 0.5)
        latency_p95 = _compute_percentile(latencies, 0.95)

    provider_mix = _build_provider_mix(records)
    cost_curve = _build_cost_curve(records)

    return BenchmarkSummary(
        total_queries=len(records),
        total_true_positives=total_tp,
        total_false_positives=total_fp,
        total_false_negatives=total_fn,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        policy_compliance_rate=compliance_rate,
        total_cost=total_cost,
        cost_per_query=cost_per_query,
        latency_p50=latency_p50,
        latency_p95=latency_p95,
        latency_average=latency_average,
        provider_mix=provider_mix,
        cost_curve=cost_curve,
    )


def summarise_by_suite(
    records: Sequence[BenchmarkRecord],
) -> dict[str, BenchmarkSummary]:
    """Aggregate records by their ``suite`` metadata entry."""

    grouped: dict[str, list[BenchmarkRecord]] = {}
    for record in records:
        suite_id = record.metadata.get("suite")
        if not suite_id:
            continue
        grouped.setdefault(str(suite_id), []).append(record)

    return {suite: summarise_records(items) for suite, items in grouped.items()}


def persist_benchmark_records(
    records: Sequence[BenchmarkRecord],
    output_path: str | Path,
    format: str | None = None,
    *,
    dsn: str | None = None,
    table_name: str = "benchmark_records",
) -> Path:
    """Persist benchmark records to JSONL, Parquet, or SQL."""

    path = Path(output_path)
    serialisable_records = [record.to_serializable_dict() for record in records]

    if format is None:
        format = path.suffix.lstrip(".").lower()

    if format in {"", None}:
        format = "jsonl"

    format = str(format).lower()

    if format == "jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in serialisable_records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif format == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
            msg = "pyarrow is required to write Parquet outputs"
            raise RuntimeError(msg) from error

        table = pa.Table.from_pylist(serialisable_records)
        pq.write_table(table, path)
    elif format == "sql":
        if dsn is None:
            raise ValueError(
                "A DSN must be provided when persisting benchmark records to SQL"
            )
        _write_records_to_sql(serialisable_records, dsn, table_name)
        manifest = {
            "destination": "sql",
            "table": table_name,
            "row_count": len(serialisable_records),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        msg = f"Unsupported persistence format: {format}"
        raise ValueError(msg)

    if dsn is not None and format != "sql":
        _write_records_to_sql(serialisable_records, dsn, table_name)

    return path


def _write_records_to_sql(
    records: Sequence[dict[str, Any]],
    dsn: str,
    table_name: str,
) -> None:
    """Persist serialised benchmark records to a SQL table."""

    try:
        from sqlalchemy import (
            JSON,
            Boolean,
            Column,
            Float,
            Integer,
            MetaData,
            String,
            Table,
            create_engine,
        )
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "SQL persistence requires SQLAlchemy to be installed"
        ) from error

    engine = create_engine(dsn)
    metadata_obj = MetaData()
    table = Table(
        table_name,
        metadata_obj,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("query_id", String(255), nullable=False),
        Column("true_positives", Integer, nullable=False),
        Column("false_positives", Integer, nullable=False),
        Column("false_negatives", Integer, nullable=False),
        Column("policy_compliant", Boolean, nullable=True),
        Column("cost", Float, nullable=True),
        Column("latency_ms", Float, nullable=True),
        Column("provider", String(255), nullable=True),
        Column("model", String(255), nullable=True),
        Column("cost_components", JSON, nullable=True),
        Column("metadata", JSON, nullable=True),
    )

    metadata_obj.create_all(engine, checkfirst=True)

    if not records:
        return

    rows = [
        {
            "query_id": row["query_id"],
            "true_positives": row["true_positives"],
            "false_positives": row["false_positives"],
            "false_negatives": row["false_negatives"],
            "policy_compliant": row.get("policy_compliant"),
            "cost": row.get("cost"),
            "latency_ms": row.get("latency_ms"),
            "provider": row.get("provider"),
            "model": row.get("model"),
            "cost_components": row.get("cost_components"),
            "metadata": row.get("metadata"),
        }
        for row in records
    ]

    with engine.begin() as connection:
        connection.execute(table.insert(), rows)


def generate_markdown_report(records: Sequence[BenchmarkRecord]) -> str:
    """Generate a markdown summary for quick sharing."""

    summary = summarise_records(records)
    lines = [
        "# Benchmark Summary",
        "",
        f"- Total queries: **{summary.total_queries}**",
        f"- True positives: **{summary.total_true_positives}**",
        f"- False positives: **{summary.total_false_positives}**",
        f"- False negatives: **{summary.total_false_negatives}**",
        "",
        "## Quality Metrics",
        f"- Precision: **{summary.precision:.3f}**",
        f"- Recall: **{summary.recall:.3f}**",
        f"- F1 score: **{summary.f1_score:.3f}**",
    ]

    if summary.policy_compliance_rate is not None:
        lines.append(f"- Policy compliance: **{summary.policy_compliance_rate:.3%}**")
    else:
        lines.append("- Policy compliance: _no policy annotations provided_")

    lines.append("\n## Cost Metrics")

    if summary.total_cost is not None:
        lines.append(f"- Total cost: **${summary.total_cost:.4f}**")
    else:
        lines.append("- Total cost: _not reported_")

    if summary.cost_per_query is not None:
        lines.append(f"- Cost per query: **${summary.cost_per_query:.4f}**")
    else:
        lines.append("- Cost per query: _not reported_")

    lines.append("\n## Latency Metrics")
    if summary.latency_average is not None:
        lines.append(f"- Average latency: **{summary.latency_average:.1f} ms**")
        if summary.latency_p50 is not None:
            lines.append(f"- Median latency (p50): **{summary.latency_p50:.1f} ms**")
        if summary.latency_p95 is not None:
            lines.append(f"- P95 latency: **{summary.latency_p95:.1f} ms**")
    else:
        lines.append("- Latency metrics: _not reported_")

    lines.append("\n## Provider Mix")
    if summary.provider_mix:
        lines.append("| Provider | Share | Queries | Cost Share |")
        lines.append("| --- | --- | --- | --- |")
        for provider, entry in sorted(summary.provider_mix.items()):
            share = entry.get("share")
            share_display = f"{share:.1%}" if isinstance(share, float) else "—"
            count = entry.get("count", 0)
            cost_share = entry.get("cost_share")
            cost_display = f"{cost_share:.1%}" if isinstance(cost_share, float) else "—"
            lines.append(f"| {provider} | {share_display} | {count} | {cost_display} |")
    else:
        lines.append("- Provider mix: _not captured_")

    lines.append("\n## Cost Curve")
    if summary.cost_curve:
        last_point = summary.cost_curve[-1]
        lines.append(
            "- Cumulative spend after {count} tracked queries: **${total:.4f}**".format(
                count=last_point.get("query_index"),
                total=last_point.get("cumulative_cost", 0.0),
            )
        )
        sample_points = summary.cost_curve[:5]
        bullet_lines = ", ".join(
            "{query} (${cost:.4f})".format(
                query=point.get("query_id"),
                cost=point.get("cumulative_cost", 0.0),
            )
            for point in sample_points
        )
        lines.append(f"- Sample progression: {bullet_lines}")
    else:
        lines.append("- No costed queries were recorded.")

    lines.extend(
        [
            "\n## Troubleshooting Guide",
            "- Precision or recall dips: inspect scenario manifests for missing `expected_anchors` or stale datasets.",
            "- Rising cost per query: review the provider mix table to rebalance traffic or update fallback routes.",
            "- Latency spikes: confirm provider SLAs and consider lowering concurrency for impacted suites.",
        ]
    )

    return "\n".join(lines)


def export_markdown_report(
    records: Sequence[BenchmarkRecord], output_path: str | Path
) -> Path:
    """Write a markdown summary to ``output_path``."""

    path = Path(output_path)
    report = generate_markdown_report(records)
    path.write_text(report, encoding="utf-8")
    return path


def iter_records(source: Iterable[dict[str, Any]]) -> list[BenchmarkRecord]:
    """Create :class:`BenchmarkRecord` instances from dictionaries."""

    records: list[BenchmarkRecord] = []
    for item in source:
        records.append(
            BenchmarkRecord(
                query_id=item["query_id"],
                true_positives=item.get("true_positives", 0),
                false_positives=item.get("false_positives", 0),
                false_negatives=item.get("false_negatives", 0),
                policy_compliant=item.get("policy_compliant"),
                cost=item.get("cost"),
                latency_ms=item.get("latency_ms"),
                provider=item.get("provider"),
                model=item.get("model"),
                cost_components=item.get("cost_components", {}),
                metadata=item.get("metadata", {}),
            )
        )
    return records
