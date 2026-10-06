# Application Contracts

An **Application Contract** describes the behavioural expectations of an AI application
independently of any specific model.

Arsenic uses contracts to answer a narrow question:

> If we change the model, does the application still get the behaviours it depends on?

This is **compatibility**, not model quality.

## Schema

Contracts are versioned JSON (`schema_version = 1`) stored at:

```text
.arsenic/contract/contract.json
```

Each `ContractItem` has:

| Field | Purpose |
|-------|---------|
| `id` | Stable ID, e.g. `refund_decision.required_claim.001` |
| `kind` | `required_claim`, `tool_argument`, `structured_output`, … |
| `severity` | `info` · `warn` · `review` · `block` |
| `prompt` | Logical feature / prompt name |
| `expectation` | Typed expectation payload |
| `provenance` | How the item was derived (never silent for critical items) |

## Workflow

```bash
arsenic init
arsenic baseline openai:gpt-current          # live provider capture
arsenic qualify openai:gpt-next              # live candidate qualification
arsenic report
```

### Offline fixtures

For reproducible tests without API keys:

```bash
arsenic baseline openai:gpt-current --from-fixtures path/to/production.json
arsenic qualify openai:gpt-next --from-fixtures path/to/candidates.json
```

Fixture and live captures share the same evaluator after normalisation.

## Live providers

Uses the same adapters as `arsenic compare` (`openai`, `anthropic`, `google`, Ollama via OpenAI-compatible endpoint).

| Flag | Purpose |
|------|---------|
| `--key-env` | Env var name holding the API key |
| `--endpoint` | OpenAI-compatible base URL |
| `--concurrency` | Bounded scenario concurrency (default 4) |
| `--repair` | Deterministic validated repair (explicit apply) |
| `--replay <id>` | Re-evaluate stored evidence offline |
| `--ci` | JSON + exit codes 0/1/2/3 |

Credentials are never persisted; raw responses are redacted before storage.

## Diffing

```bash
arsenic contract diff
```

When a contract or production baseline changes, prior qualifications are marked `STALE`
and must not appear as currently qualified.

## Guardrails

- Discovery proposes; it does not silently invent block-severity expectations.
- Evidence is authoritative; compatibility % is a summary only.
- Compatibility ≠ intelligence ≠ benchmark performance.
