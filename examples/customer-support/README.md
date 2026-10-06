# Customer Support — Application Contract demo

Realistic demo for Arsenic Application Contracts.

Deliberately produces:

| Candidate | Outcome |
|-----------|---------|
| `openai:gpt-safe` | Safe migration (PASS, lower cost) |
| `anthropic:claude-y` | Presentation-only drift (PASS_WITH_WARNINGS) |
| `google:gemini-z` | Tool-argument + missing claim + schema BLOCK |
| `openai:gpt-repairable` | Missing claim → validated repair → PASS_WITH_PATCH |

## Offline run

```bash
# from repo root
./scripts/application_contract_smoke.sh
```

Or manually:

```bash
arsenic init --project examples/customer-support --contract examples/customer-support/contract.json
arsenic baseline openai:gpt-current --project examples/customer-support \
  --from-fixtures examples/customer-support/fixtures/production.json
arsenic qualify openai:gpt-safe anthropic:claude-y google:gemini-z \
  --project examples/customer-support \
  --from-fixtures examples/customer-support/fixtures/candidates.json
arsenic report --project examples/customer-support --output /tmp/app.html
```
