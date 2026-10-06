# CI Qualification

Machine-readable qualification for pipelines:

```bash
arsenic qualify --ci openai:gpt-x
arsenic qualify --json openai:gpt-x anthropic:claude-y
```

With `--ci` / `--json`, stdout is JSON only (no human banners). The envelope includes
authoritative effective fields so CI does not reimplement Arsenic policy:

- `recorded_decision` / `effective_decision`
- `evidence_validity` (`VALID` · `STALE` · `INCOMPLETE`)
- `is_stale`, `stale_reasons`, `incomplete_reasons`
- `migration_recommendation` / `migration_recommendation_text`
- `contract_hash`, `baseline_hash`, `input_fingerprint`
- `aggregate_migration_recommendation` / `ci_exit_code`

## Exit codes

Driven by effective `MigrationRecommendation` (not raw historical `decision`):

| Code | Meaning |
|-----:|---------|
| 0 | `SAFE TO MIGRATE` |
| 1 | `MIGRATION BLOCKED` |
| 2 | `REVIEW REQUIRED` or `STALE — REQUALIFY REQUIRED` |
| 3 | `QUALIFICATION INCOMPLETE` or `NO EVIDENCE` / fatal config error |

Offline smoke gate:

```bash
./scripts/application_contract_smoke.sh
```

## GitHub Actions

```yaml
name: model-qualification
on:
  pull_request:
  workflow_dispatch:
    inputs:
      candidate:
        description: provider:model to qualify
        required: true
        default: openai:gpt-x

jobs:
  qualify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Arsenic
        run: |
          curl -fsSL https://raw.githubusercontent.com/markndg/arsenic/main/install.sh | bash
      - name: Qualify candidate model
        run: arsenic qualify --ci ${{ inputs.candidate || 'openai:gpt-x' }}
        # For offline demos, pass fixtures:
        # run: arsenic qualify --ci openai:gpt-safe --from-fixtures examples/customer-support/fixtures/candidates.json --project examples/customer-support
```

Evidence remains in `.arsenic/qualifications/`; never treat the summary score alone as the decision.
Never treat lock `qualified` or historical PASS as current safety after contract/baseline change.
