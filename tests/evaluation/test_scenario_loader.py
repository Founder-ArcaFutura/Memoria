from __future__ import annotations

import json
from collections import Counter

from memoria.evaluation import load_default_spec, materialize_suite
from memoria.evaluation.starter_datasets import list_datasets


def test_default_spec_structure():
    spec = load_default_spec()
    assert "customer_support_v1" in spec.suites
    support_suite = spec.get_suite("customer_support_v1")
    assert support_suite.privacy_mix["public"] == 0.25
    assert any(scenario.id == "cs_public_kb" for scenario in support_suite.scenarios)


def test_materialize_suite_creates_datasets(tmp_path):
    spec = load_default_spec()
    suite_context = materialize_suite(
        "customer_support_v1", spec=spec, output_root=tmp_path
    )

    assert suite_context.suite.id == "customer_support_v1"
    assert len(suite_context.scenarios) == len(
        spec.get_suite("customer_support_v1").scenarios
    )

    generated = next(
        scenario
        for scenario in suite_context.scenarios
        if scenario.spec.id == "cs_privacy_mix_generated"
    )
    assert generated.dataset_path.exists()

    with generated.dataset_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    assert len(records) == 36
    privacy_counts = Counter(record["privacy_bucket"] for record in records)
    assert set(privacy_counts) == {"public", "internal", "restricted", "private"}


def test_list_datasets_matches_packaged_files():
    datasets = list_datasets()
    assert "customer_support_public.jsonl" in datasets
    assert "revenue_enablement_mixed.jsonl" in datasets
