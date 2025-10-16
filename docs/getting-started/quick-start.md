# Quick start

Bring Memoria online in a few minutes and validate retrieval quality with the bundled evaluation harness. The workflow mirrors what CI executes so you can trust the results locally.

## 1. Set up your environment

1. **Install the SDK from source**:

   ```bash
   git clone https://github.com/Founder-ArcaFutura/memoria_staging.git
   cd memoria_staging
   pip install -e .
   pip install -e ".[sync]"
   pip install openai
   ```

   Development installs include the test suite and docs tooling:

   ```bash
   pip install -e ".[dev]"
   ```

2. **Export provider credentials** if you plan to exercise live models:

   ```bash
   export OPENAI_API_KEY="sk-your-openai-key-here"
   ```

3. **Bootstrap configuration** (generates `.env`, API keys, and database defaults):

   ```bash
   memoria bootstrap
   ```

4. **Smoke test the stack** to confirm imports and Docker orchestration succeed:

   ```bash
   python scripts/run_local.py smoke  # or: make smoke
   ```

## 2. Author a scenario

The repository ships with a template at `examples/evaluation_quickstart.json`. Duplicate it to capture the behaviour you want to validate:

```bash
cp examples/evaluation_quickstart.json scenarios/my_feature.json
```

Each scenario contains two sections:

| Section | Purpose | Key fields |
| --- | --- | --- |
| `facts` | Seeds the memory graph with curated entries. | `text`, optional `symbolic_anchors`, and optional spatial coordinates (`x`, `y`, `z`). |
| `probes` | Queries that assert the right memories surface. | `query`, `expects` (list of substrings that must appear in the auto-ingest prompt), and optional `label`. |

Adjust the template to reflect the behaviour you expect—change namespaces, add symbolic anchors, or tweak coordinates. Point to a remote database via `database_url` when you need scenario isolation.

## 3. Run the harness

Execute the helper script against your scenario:

```bash
python examples/evaluation_quickstart.py --scenario scenarios/my_feature.json
```

The runner stages each fact with `Memoria.store_memory`, executes the probes with `Memoria.get_auto_ingest_system_prompt`, and prints a structured summary. Append `--report reports/my_feature.json` to write machine-readable output for CI or regression dashboards.

## 4. Interpret the report

Successful runs look similar to:

```
Scenario: developer-onboarding
Description: Seeds developer background facts and confirms retrieval with targeted probes.
Namespace: quickstart_onboarding

Memories staged:
  • identity-riley
  • architecture-update

Probe results:
- Identity recall: PASS
  query   : Who is leading the platform migration?
  expects : ['Riley', 'analytics platform migration']
  prompt  : --- Relevant Memory Context ---
            [ESSENTIAL_FACT] Riley Chen is leading the analytics platform migration for Q4.
            - The team is replacing the legacy warehouse with a lakehouse pattern backed by Delta Lake.
            -------------------------
- Architecture decision: PASS
  query   : What architecture change is planned for data storage?
  expects : ['lakehouse', 'Delta Lake']
  prompt  : --- Relevant Memory Context ---
            - The team is replacing the legacy warehouse with a lakehouse pattern backed by Delta Lake.
            -------------------------

Memory stats:
  chat_history_count: 0
  long_term_count: 2
  short_term_count: 0
```

Use the summary table to debug quickly:

- **Missing expectations** signal retrieval drift—inspect the generated prompt to see which memory displaced the expected item.
- **Low long-term counts** indicate staging failed or the wrong namespace/database was targeted. Delete the SQLite file or update `database_url` before rerunning.
- **High short-term counts** suggest previous runs polluted the scenario. Reset the database or change the namespace to isolate results.

Attach the console or JSON output to pull requests whenever heuristics, ranking, or dataset changes alter behaviour. Review the [contribution guidelines](../../CONTRIBUTING.md#keep-evaluation-scenarios-up-to-date) for the evaluation checklist used during code review.

## 5. Troubleshooting

| Symptom | Likely cause | What to try |
| --- | --- | --- |
| `Rate limit hit` or long pauses between probes | Provider throttling while the scenario seeds or probes live models. | Lower the number of probe queries, add `time.sleep` between probes, or configure provider-specific retries (for example `LITELLM_EXTRA_HEADERS`). |
| Expected memories no longer surface after updating the dataset | Dataset drift or namespace mismatch. | Regenerate the scenario database by deleting the SQLite file, confirm the `namespace` matches your probes, and review recent schema migrations. |
| Probe fails with policy or privacy errors | Retention or privacy policies block promotion. | Inspect the policy settings in `memoria/config/settings.py`, adjust the `y` coordinate or symbolic anchors in your facts, and retry with `verbose=true` to trace policy checks. |
| CLI or dashboard shows a missing provider capability | Optional extras for OpenAI, Gemini, or sync backends are not installed. | Follow the suggested `pip install` command (for example `pip install -e ".[integrations]"`) or disable the capability by clearing the provider key or switching the sync backend to `none`. |

---

Next steps:

- Expand scenarios with end-to-end conversations or policy edge cases.
- Schedule the quickstart script in CI with the `--report` flag to monitor regressions.
- Share new probes in documentation or examples so the community can keep benchmarks fresh.
