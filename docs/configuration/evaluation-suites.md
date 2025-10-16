# Evaluation Suite Specification

Memoria's evaluation harness uses declarative YAML/JSON suites to define
repeatable benchmark scenarios. Each suite aligns with a real-world use case and
captures the privacy mix, memory distribution, and scoring goals required to
measure changes over time.

## File Locations

- Packaged baseline suites: `memoria/evaluation/specs/default_suites.yaml`
- Starter datasets: `memoria/evaluation/starter_datasets/`
- Generated data helpers: `memoria/evaluation/datasets.py`

Suites can also be authored in JSON with the same schema; the loader infers the
format from the file extension.

## Top-level Structure

```yaml
version: "1.0"
metadata:
  description: ...
  last_updated: ...
suites:
  - id: customer_support_v1
    name: Customer Support Baseline
    description: ...
    use_cases:
      - ...
    privacy_mix:
      public: 0.25
      internal: 0.45
      restricted: 0.15
      private: 0.15
    memory_distribution:
      long_term:
        entries: 72
        anchors: [product, troubleshooting, compliance]
      short_term:
        entries: 18
        anchors: [ticket, resolution]
    scenarios:
      - id: cs_public_kb
        name: Knowledge Base Lookup
        workspace: support_public
        dataset:
          kind: fixture
          path: customer_support_public.jsonl
        ingest_profile:
          cadence: daily
          batch_size: 12
        retrieval_tasks:
          - query: "Reset procedure for Horizon router"
            expected_anchors: [troubleshooting, firmware]
            metrics:
              recall_at_3: 0.8
              precision_at_3: 0.7
        privacy_expectations:
          allow_public_overlap: true
          max_private_leakage: 0
```

### Required Keys

| Level      | Keys                                                                                     |
| ---------- | ----------------------------------------------------------------------------------------- |
| Suite      | `id`, `name`, `description`, `use_cases`, `privacy_mix`, `memory_distribution`, `scenarios` |
| Scenario   | `id`, `name`, `description`, `workspace`, `dataset`                                       |
| Dataset    | `kind` (either `fixture` or `generated`)                                                  |

### Optional Keys

- `metadata`: free-form dictionary for tagging suites or scenarios.
- `ingest_profile`: cadence, batch sizing, and ingestion channel hints.
- `retrieval_tasks`: queries, expected anchors, and metric targets.
- `privacy_expectations`: policy assertions validated during scoring.

## Authoring Guidance

1. **Quantify the privacy mix.** Ratios should add up to 1.0 when possible to
   simplify comparison. Use canonical buckets: `public`, `internal`,
   `restricted`, `private`.
2. **Describe memory distribution.** Capture the volume and anchor clusters for
   long-term, short-term, and conscious memories to provide context for scoring.
3. **Reference datasets.** Point fixtures to files inside
   `memoria/evaluation/starter_datasets/` or reference generators defined in
   `memoria/evaluation/datasets.py`.
4. **List retrieval tasks.** Each task should encode the target anchors and
   desired metric thresholds so future regressions are easy to interpret.
5. **Annotate privacy expectations.** Declare the expected enforcement behaviour
   (for example, `enforce_namespace_boundary: true`) to align with policy
   validation tests.

### CLI scaffolding

Run `memoria evaluation-scaffold` to generate a starter YAML document with the
required structure:

```bash
memoria evaluation-scaffold suites/new_suite.yaml \
  --suite-id my_use_case_v1 \
  --suite-name "My use case" \
  --use-case support \
  --scenario-id onboarding_flow \
  --retrieval-query "How do I escalate?"
```

The command emits a ready-to-edit specification alongside placeholder retrieval
queries and dataset references. Use `--force` to overwrite an existing file or
repeat `--retrieval-query` to seed multiple probes before collecting labels.

## Loading Suites Programmatically

```python
from memoria.evaluation import load_default_spec, materialize_suite

spec = load_default_spec()
materialized = materialize_suite("customer_support_v1", spec=spec)

for scenario in materialized.scenarios:
    print("Scenario", scenario.spec.name)
    print("Dataset located at", scenario.dataset_path)
    print("Manifest", scenario.manifest_path.read_text())
```

The loader copies fixture datasets into ephemeral workspace directories and
invokes synthetic generators when specified. Each workspace receives a
`scenario_manifest.json` describing the scenario metadata for downstream tools.

## Overriding with Organisation Data

- Create a new YAML or JSON document following the schema above.
- Point `dataset.path` to organisation-specific fixtures stored outside the
  package (absolute or relative paths are supported).
- When referencing generated data, supply tuned parameters (for example,
  namespace names or anchor pools) that reflect your deployment.
- Use versioned filenames (e.g., `customer_support_v2.yaml`) and include a
  `metadata.change_log` array to track major updates.

## Validation

The parser raises descriptive errors for missing keys or malformed structures.
Run `pytest tests/evaluation/test_scenario_loader.py` after authoring a new
suite to confirm it loads correctly.

## CI automation

Memoria ships with an evaluation workflow (`.github/workflows/evaluation.yml`)
that re-materialises a small set of smoke suites and publishes benchmark
artifacts for reviewers. The job calls `scripts/ci/run_evaluation_suites.py`,
which loads each requested suite, inspects the packaged datasets, and then
summarises anchor coverage with `memoria.tools.benchmark.generate_markdown_report`.

The workflow runs automatically for pull requests that touch retrieval,
evaluation, or policy code paths and can also be triggered manually via
**Run workflow** from the Actions tab. Provide optional inputs to override the
suite list (`suites`), spec path (`spec`), or dataset root (`dataset_root`).

On pull requests the job now completes the review loop automatically: once the
`evaluation-artifacts/pull_request_comment.md` file is generated, the workflow
posts it back to the PR as a summary comment so reviewers can see guardrail
status and suite metrics inline. Threshold violations also surface as failing
check annotations (via `::error` markers) so regressions are visible directly in
the Actions log and the GitHub Checks UI.

When running the workflow manually for a dry run, leave the `post_pr_comment`
input set to the default (`false`) to avoid publishing a comment. You can still
download the uploaded artifacts for local review without notifying reviewers.

### Default smoke suites

- `customer_support_v1` &mdash; validates the support privacy mix and ensures the
  knowledge base fixtures cover the retrieval probes documented in this file.
- `revenue_enablement_v1` &mdash; checks that revenue enablement datasets still ship
  the competitive and launch anchors used by the GTM scenarios.

Override the default selection via the workflow dispatch form or by setting the
`MEMORIA_EVALUATION_SUITES` environment variable (comma or space separated IDs).

### Required secrets and overrides

Add the following secrets to repository or organisation settings before running
the workflow:

| Secret | Required | Purpose |
| ------ | -------- | ------- |
| `OPENAI_API_KEY` | Yes (for provider-backed probes) | Enables suites that rely on OpenAI completions or embeddings. |
| `ANTHROPIC_API_KEY` | Yes (for provider-backed probes) | Supports Anthropic-backed probes and policy checks. |
| `GOOGLE_API_KEY` | Optional | Allows evaluation of Google Generative AI scenarios when configured. |
| `MEMORIA_EVALUATION_DATASET_ARCHIVE` | Optional | Base64-encoded `tar.gz` containing proprietary dataset overrides. When present and no manual dataset root is provided, the workflow extracts the archive and exports `MEMORIA_EVALUATION_DATASET_ROOT` for the runner. |

If neither the `dataset_root` input nor `MEMORIA_EVALUATION_DATASET_ARCHIVE`
are supplied, the packaged starter datasets are used.

You can also pass a custom spec or dataset location at dispatch time:

- `spec` input &rarr; forwarded to `MEMORIA_EVALUATION_SPEC_PATH` so the runner
  loads an alternative YAML/JSON document.
- `dataset_root` input &rarr; forwarded to
  `MEMORIA_EVALUATION_DATASET_ROOT`, allowing local fixture directories to
  replace the packaged datasets without encoding them as a secret.

### Published artifacts

Each run uploads an `evaluation-reports` artifact containing:

- `benchmark.jsonl` &mdash; raw records for each retrieval task.
- `summary.json` &mdash; aggregate precision/recall metrics.
- `summary.md` &mdash; reviewer-friendly Markdown generated by
  `generate_markdown_report`.
- `suite_summaries.json` &mdash; per-suite aggregates including latency and cost.
- `run_metadata.json` &mdash; the suite list, overrides, guardrail thresholds, and
  GitHub context for the run.
- `pull_request_comment.md` &mdash; drop-in PR summary with guardrail status and a
  breakdown table suitable for review threads.

The Markdown report now includes latency percentiles, provider mix shares,
cost-curve highlights, and troubleshooting tips so reviewers can quickly
interpret regressions.

### Regression guardrails

`run_evaluation_suites.py` enforces configurable guardrails before exiting. Pass
thresholds on the command line or via a JSON file:

```bash
python scripts/ci/run_evaluation_suites.py \
  --min-precision 0.65 \
  --min-recall 0.6 \
  --max-cost-per-query 0.30 \
  --max-latency-p95 1800
```

The CI workflow sets defaults for smoke runs and will fail the job when any
threshold is violated. The error message lists the failing metrics, the step log
emits GitHub check annotations for each guardrail breach, and the uploaded
`pull_request_comment.md` highlights the guardrail status alongside per suite
metrics so maintainers can copy/paste (or rely on the automated PR comment
mentioned above) during review.
