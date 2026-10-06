# CI Qualification

Machine-readable qualification for pipelines:

```bash
arsenic qualify --ci openai:gpt-x
arsenic qualify --json openai:gpt-x anthropic:claude-y
```

## Exit codes

| Code | Meaning |
|-----:|---------|
| 0 | `PASS` / `PASS_WITH_WARNINGS` / `PASS_WITH_PATCH` |
| 1 | `BLOCK` |
| 2 | `REVIEW` (or `STALE`) |
| 3 | Configuration / runtime error |

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
