# Evaluation Datasets & Contribution Guide

This guide explains how to use the packaged starter datasets and how to
contribute organisation-specific corpora for Memoria's evaluation harness.

## Starter Collection

The repository ships with a lightweight set of fixtures located at
`memoria/evaluation/starter_datasets/`. They are aligned with the baseline suite
in `memoria/evaluation/specs/default_suites.yaml` and serve three goals:

1. **Smoke testing.** Validate loader and scoring logic without requiring access
   to sensitive data.
2. **Reference modelling.** Demonstrate how privacy coordinates, symbolic
   anchors, and importance scores are encoded in JSONL records.
3. **Reproducibility.** Allow contributors to compare behaviour across branches
   with deterministic datasets.

Available datasets can be listed at runtime:

```python
from memoria.evaluation.starter_datasets import list_datasets

print(list_datasets())
```

Each dataset stores one memory per line with the full spatial-symbolic schema.

## Adding Organisation-specific Corpora

1. **Create a new dataset directory.** Store internal fixtures outside of the
   repository or behind encrypted storage. Reference them via absolute or
   environment-resolved paths in your custom evaluation suite.
2. **Preserve schema fidelity.** Include `anchor`, `text`, `x_coord`,
   `y_coord`, `z_coord`, `symbolic_anchors`, `timestamp`, `importance_score`,
   `retention_type`, and `namespace`. Additional keys are allowed for bespoke
   scoring but should be namespaced (e.g., `org_privacy_label`).
3. **Document provenance.** In your suite's `metadata`, describe the source
   system, anonymisation steps, and review cadence. This helps security teams
   audit compliance.
4. **Set privacy expectations.** Ensure the suite's `privacy_expectations`
   matches the classification of each dataset. For example, private escalations
   should set `enforce_namespace_boundary: true` and `require_admin_review: true`.
5. **Share contributions safely.** When submitting pull requests, only include
   synthetic or fully anonymised datasets. Reference proprietary corpora via
   documentation rather than raw files.

## Synthetic Dataset Generators

The module `memoria/evaluation/datasets.py` exposes pluggable generators that
mirror production privacy mixes without leaking sensitive content. The default
`synthetic_privacy_mix` function accepts parameters such as `count`,
`namespaces`, `privacy_buckets`, and `anchor_pool` to emulate your distribution.

To add a new generator:

1. Implement a function returning a list of JSON-serialisable memory records.
2. Register the function in `DATASET_GENERATORS` with a unique key.
3. Document the expected parameters in this file and update the evaluation suite
   spec to reference the new generator.

## Submission Checklist

- [ ] New suites validated with `pytest tests/evaluation/test_scenario_loader.py`
- [ ] Documentation updated (`docs/configuration/evaluation-suites.md` or this
      guide) describing new scenarios and datasets
- [ ] Sensitive data removed or replaced with generated fixtures
- [ ] Privacy expectations documented for every scenario in the suite

Following these steps keeps evaluation assets reproducible while protecting
customer and employee data.
