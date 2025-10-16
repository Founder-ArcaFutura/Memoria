"""CI helper to materialise evaluation suites and emit benchmark summaries."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memoria.evaluation import (
    EvaluationSpec,
    load_default_spec,
    load_spec_from_path,
    materialize_suite,
)
from memoria.tools.benchmark import (
    BenchmarkRecord,
    generate_markdown_report,
    persist_benchmark_records,
    summarise_by_suite,
    summarise_records,
)


def _load_spec(spec_path: str | None) -> EvaluationSpec:
    if spec_path:
        resolved = Path(spec_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Evaluation spec override not found: {resolved}")
        return load_spec_from_path(resolved)
    return load_default_spec()


def _load_dataset(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _apply_dataset_override(destination: Path, source_root: Path | None) -> None:
    if source_root is None:
        return
    override = source_root / destination.name
    if override.exists():
        shutil.copy2(override, destination)


def _load_thresholds(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Threshold file must contain a JSON object")
    thresholds: dict[str, float] = {}
    for key, value in payload.items():
        try:
            thresholds[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid threshold for {key!r}: {value!r}") from exc
    return thresholds


def _check_thresholds(summary: "BenchmarkSummary", thresholds: dict[str, float]) -> list[str]:
    violations: list[str] = []

    def _require(metric: str, actual: float | None, comparator: str, target: float) -> None:
        if actual is None:
            violations.append(
                f"Metric '{metric}' was not reported but a threshold of {target} was configured."
            )
            return
        if comparator == "min" and actual < target:
            violations.append(
                f"Metric '{metric}'={actual:.3f} fell below the threshold {target:.3f}."
            )
        if comparator == "max" and actual > target:
            violations.append(
                f"Metric '{metric}'={actual:.3f} exceeded the threshold {target:.3f}."
            )

    if "min_precision" in thresholds:
        _require("precision", summary.precision, "min", thresholds["min_precision"])
    if "min_recall" in thresholds:
        _require("recall", summary.recall, "min", thresholds["min_recall"])
    if "min_f1" in thresholds:
        _require("f1", summary.f1_score, "min", thresholds["min_f1"])
    if "min_compliance" in thresholds:
        _require(
            "policy_compliance",
            summary.policy_compliance_rate,
            "min",
            thresholds["min_compliance"],
        )
    if "max_cost_per_query" in thresholds:
        _require(
            "cost_per_query",
            summary.cost_per_query,
            "max",
            thresholds["max_cost_per_query"],
        )
    if "max_latency_p95" in thresholds:
        _require(
            "latency_p95_ms",
            summary.latency_p95,
            "max",
            thresholds["max_latency_p95"],
        )

    return violations


def _format_suite_table(summaries: dict[str, "BenchmarkSummary"]) -> str:
    if not summaries:
        return "_No suite-specific summaries available._"
    header = "| Suite | Precision | Recall | F1 | P95 Latency (ms) | Cost/Query |\n| --- | --- | --- | --- | --- | --- |"
    rows = [header]
    for suite_id, summary in sorted(summaries.items()):
        latency = f"{summary.latency_p95:.1f}" if summary.latency_p95 is not None else "—"
        cost_per_query = (
            f"${summary.cost_per_query:.4f}"
            if summary.cost_per_query is not None
            else "—"
        )
        rows.append(
            "| `{suite}` | {precision:.3f} | {recall:.3f} | {f1:.3f} | {latency} | {cost} |".format(
                suite=suite_id,
                precision=summary.precision,
                recall=summary.recall,
                f1=summary.f1_score,
                latency=latency,
                cost=cost_per_query,
            )
        )
    return "\n".join(rows)


def _write_pr_summary(
    destination: Path,
    overall: "BenchmarkSummary",
    suites: dict[str, "BenchmarkSummary"],
    threshold_messages: list[str],
) -> Path:
    lines = [
        "# Evaluation Results",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Precision | {overall.precision:.3f} |",
        f"| Recall | {overall.recall:.3f} |",
        f"| F1 score | {overall.f1_score:.3f} |",
    ]
    if overall.policy_compliance_rate is not None:
        lines.append(
            f"| Policy compliance | {overall.policy_compliance_rate:.3%} |"
        )
    if overall.cost_per_query is not None:
        lines.append(f"| Cost per query | ${overall.cost_per_query:.4f} |")
    if overall.latency_p95 is not None:
        lines.append(f"| P95 latency | {overall.latency_p95:.1f} ms |")

    if threshold_messages:
        lines.append("\n> **Regression guardrails failed**:")
        for message in threshold_messages:
            lines.append(f"> - {message}")
    else:
        lines.append("\n> ✅ Regression guardrails passed.")

    lines.append("\n## Suite Breakdown")
    lines.append(_format_suite_table(suites))

    lines.append(
        "\nArtifacts saved under `evaluation-artifacts/`. Upload or link this file in PR discussions for full context."
    )

    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def _iter_task_records(
    suite_id: str,
    scenario_id: str,
    retrieval_tasks: Iterable[dict[str, object]],
    dataset_rows: list[dict[str, object]],
    workspace: str,
) -> list[BenchmarkRecord]:
    anchors: Counter[str] = Counter()
    for row in dataset_rows:
        values = row.get("symbolic_anchors")
        if isinstance(values, list):
            anchors.update(str(item) for item in values if isinstance(item, str))

    records: list[BenchmarkRecord] = []
    for index, task in enumerate(retrieval_tasks, start=1):
        query = str(task.get("query", f"task-{index}"))
        expected = [
            str(anchor)
            for anchor in task.get("expected_anchors", [])
            if isinstance(anchor, str)
        ]
        missing = [anchor for anchor in expected if anchors[anchor] == 0]
        records.append(
            BenchmarkRecord(
                query_id=f"{suite_id}:{scenario_id}:{index}",
                true_positives=len(expected) - len(missing),
                false_positives=0,
                false_negatives=len(missing),
                policy_compliant=None if not expected else not missing,
                metadata={
                    "suite": suite_id,
                    "scenario": scenario_id,
                    "workspace": workspace,
                    "query": query,
                    "expected_anchors": expected,
                    "missing_anchors": missing,
                    "dataset_entries": len(dataset_rows),
                },
            )
        )
    return records


def _fallback_record(
    suite_id: str,
    scenario_id: str,
    dataset_rows: list[dict[str, object]],
    workspace: str,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        query_id=f"{suite_id}:{scenario_id}:dataset",
        true_positives=1 if dataset_rows else 0,
        false_positives=0 if dataset_rows else 1,
        false_negatives=0,
        policy_compliant=bool(dataset_rows),
        metadata={
            "suite": suite_id,
            "scenario": scenario_id,
            "workspace": workspace,
            "dataset_entries": len(dataset_rows),
            "note": "No retrieval_tasks defined; reporting dataset availability.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suites",
        nargs="+",
        help="Evaluation suite identifiers to run. Defaults to smoke suites from documentation.",
    )
    parser.add_argument(
        "--spec",
        type=str,
        default=os.environ.get("MEMORIA_EVALUATION_SPEC_PATH"),
        help="Optional path to an evaluation spec override.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("MEMORIA_EVALUATION_DATASET_ROOT"),
        help="Optional directory containing fixture overrides (filenames must match packaged datasets).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation-artifacts"),
        help="Directory for generated reports.",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        help="Optional JSON file containing regression guardrail thresholds.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        dest="min_precision",
        help="Minimum acceptable precision before failing the run.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        dest="min_recall",
        help="Minimum acceptable recall before failing the run.",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        dest="min_f1",
        help="Minimum acceptable F1 score before failing the run.",
    )
    parser.add_argument(
        "--min-compliance",
        type=float,
        dest="min_compliance",
        help="Minimum acceptable policy compliance rate before failing the run.",
    )
    parser.add_argument(
        "--max-cost-per-query",
        type=float,
        dest="max_cost_per_query",
        help="Maximum allowable average cost per query before failing the run.",
    )
    parser.add_argument(
        "--max-latency-p95",
        type=float,
        dest="max_latency_p95",
        help="Maximum allowable P95 latency in milliseconds before failing the run.",
    )
    parser.add_argument(
        "--analytics-format",
        choices=["jsonl", "parquet", "sql"],
        help="Optional output format override for benchmark analytics.",
    )
    parser.add_argument(
        "--analytics-dsn",
        help="SQLAlchemy-compatible DSN used when persisting analytics to SQL.",
    )
    args = parser.parse_args()

    spec = _load_spec(args.spec)
    suite_ids = list(args.suites or [])
    if not suite_ids:
        env_suites = os.environ.get("MEMORIA_EVALUATION_SUITES", "")
        if env_suites:
            suite_ids = [
                value
                for value in re.split(r"[\s,]+", env_suites)
                if value.strip()
            ]
    if not suite_ids:
        suite_ids = [
            "customer_support_v1",
            "revenue_enablement_v1",
        ]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[BenchmarkRecord] = []
    metadata: dict[str, object] = {
        "suites": suite_ids,
        "spec_override": str(args.spec) if args.spec else None,
        "dataset_override_root": str(args.dataset_root) if args.dataset_root else None,
        "env_suite_override": os.environ.get("MEMORIA_EVALUATION_SUITES"),
        "analytics_format": args.analytics_format,
        "analytics_dsn": bool(args.analytics_dsn),
    }

    for suite_id in suite_ids:
        suite_context = materialize_suite(suite_id, spec=spec, keep_root=True)
        try:
            for scenario in suite_context.scenarios:
                dataset_path = scenario.dataset_path
                if args.dataset_root:
                    _apply_dataset_override(dataset_path, Path(args.dataset_root))
                dataset_rows = _load_dataset(dataset_path)

                tasks = list(scenario.spec.retrieval_tasks)
                scenario_records = _iter_task_records(
                    suite_id,
                    scenario.spec.id,
                    tasks,
                    dataset_rows,
                    scenario.spec.workspace,
                )
                if not scenario_records:
                    scenario_records = [
                        _fallback_record(
                            suite_id,
                            scenario.spec.id,
                            dataset_rows,
                            scenario.spec.workspace,
                        )
                    ]
                records.extend(scenario_records)
        finally:
            shutil.rmtree(suite_context.root, ignore_errors=True)

    if not records:
        raise RuntimeError("No benchmark records generated from the requested suites")

    analytics_format = args.analytics_format
    analytics_dsn = args.analytics_dsn
    if analytics_format == "sql" and not analytics_dsn:
        raise SystemExit("--analytics-dsn is required when --analytics-format=sql")

    if analytics_format == "parquet":
        records_filename = "benchmark.parquet"
    elif analytics_format == "sql":
        records_filename = "benchmark.sql.json"
    else:
        records_filename = "benchmark.jsonl"

    records_path = persist_benchmark_records(
        records,
        output_dir / records_filename,
        format=analytics_format,
        dsn=analytics_dsn,
    )
    summary = summarise_records(records)
    suite_summaries = summarise_by_suite(records)

    threshold_config = _load_thresholds(args.thresholds)
    for key in (
        "min_precision",
        "min_recall",
        "min_f1",
        "min_compliance",
        "max_cost_per_query",
        "max_latency_p95",
    ):
        value = getattr(args, key)
        if value is not None:
            threshold_config[key] = float(value)
    threshold_messages = _check_thresholds(summary, threshold_config)

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    markdown = generate_markdown_report(records)
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    suite_summary_path = output_dir / "suite_summaries.json"
    suite_summary_payload = {
        suite_id: suite_summary.to_dict()
        for suite_id, suite_summary in suite_summaries.items()
    }
    suite_summary_path.write_text(
        json.dumps(suite_summary_payload, indent=2), encoding="utf-8"
    )

    metadata_path = output_dir / "run_metadata.json"
    metadata_payload = {
        **metadata,
        "summary": summary.to_dict(),
        "suite_summaries": suite_summary_payload,
        "thresholds": threshold_config,
        "threshold_violations": threshold_messages,
        "git_sha": os.environ.get("GITHUB_SHA"),
        "git_ref": os.environ.get("GITHUB_REF"),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    pr_comment_path = output_dir / "pull_request_comment.md"
    _write_pr_summary(pr_comment_path, summary, suite_summaries, threshold_messages)

    print(f"Saved benchmark records to {records_path}")
    print(f"Saved markdown summary to {markdown_path}")
    print(f"Saved JSON summary to {summary_json}")
    print(f"Saved per-suite summaries to {suite_summary_path}")
    print(f"Saved run metadata to {metadata_path}")
    print(f"Saved PR summary to {pr_comment_path}")

    if threshold_messages:
        for message in threshold_messages:
            print(f"::error title=Regression guardrail violated::{message}")
        joined = "\n - ".join(threshold_messages)
        raise SystemExit(
            "Regression guardrails failed:\n - {messages}".format(messages=joined)
        )


if __name__ == "__main__":
    main()
