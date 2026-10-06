# Application Contracts

An **Application Contract** describes the behavioural expectations of an AI application
independently of any specific model.

Arsenic uses contracts to answer a narrow question:

> If we change the model, does the application still get the behaviours it depends on
> **based on current, complete evidence?**

This is **compatibility**, not model quality. Arsenic must never communicate
`SAFE TO MIGRATE` when evidence is stale, incomplete, failed, unsupported, or otherwise invalid.

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

## Decisions and evidence validity

Historical qualification JSON is **immutable**. Current migration recommendations are
derived via `assess_qualification` / `aggregate_migration` using the current contract and
baseline hashes.

| State | Meaning |
|-------|---------|
| **PASS** / **PASS_WITH_WARNINGS** / **PASS_WITH_PATCH** | Behavioural outcome on the recorded run. Only certifies migration when evidence is currently **VALID**. |
| **REVIEW** | Unresolved review-severity findings. Not safe to migrate. |
| **BLOCK** | Blocking contract requirement failed. Migration blocked. |
| **STALE** | Contract and/or baseline hash differs from the qualification. Historical only — requalify required. Never SAFE. |
| **INCOMPLETE** (`QUALIFICATION INCOMPLETE`) | Provider/runtime execution errors or missing trustworthy evidence. Not a behavioural PASS. Never SAFE. |

Evidence validity is orthogonal to the recorded behavioural decision:

1. Check staleness (contract/baseline hashes).
2. Check completeness (execution errors, empty evidence).
3. Only then interpret the behavioural decision as a migration recommendation.

### When requalification is required

- Contract content or version changes (hash differs)
- Production baseline changes (hash differs)
- Input fingerprint changes (prompts, thresholds, runtime params)
- Prior run was Incomplete or Stale
- Validated patch was bound to a different contract/baseline/model

### Validated repair

A validated patch only yields `PASS_WITH_PATCH` → `SAFE TO MIGRATE` when it binds to the
**exact** current contract hash, baseline hash, and candidate model. A patch validated
against stale evidence cannot certify a changed contract.

Patches are never applied silently — `arsenic patch apply` is explicit.

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

Offline smoke (CI):

```bash
./scripts/application_contract_smoke.sh
```

## Live providers

Uses the same adapters as `arsenic compare` (`openai`, `anthropic`, `google`, Ollama via OpenAI-compatible endpoint).

| Flag | Purpose |
|------|---------|
| `--key-env` | Env var name holding the API key |
| `--endpoint` | OpenAI-compatible base URL |
| `--concurrency` | Bounded scenario concurrency (default 4) |
| `--repair` | Deterministic validated repair (explicit apply) |
| `--replay <id>` | Re-evaluate stored evidence offline |
| `--ci` / `--json` | Machine-readable JSON only on stdout |

### Provider capability limits

Unsupported **required** capabilities (e.g. tool calling on Anthropic/Google adapters
that do not yet expose tools in Arsenic) produce behavioural Fail / Block — never silent
PASS or NA. Capability gaps are not Incomplete; they are explicit contract failures.

| Adapter | Tools | Structured output |
|---------|-------|-------------------|
| OpenAI / Ollama | yes | yes |
| Anthropic | no (in Arsenic) | no |
| Google | no (in Arsenic) | no |

### CI exit codes

Driven by `MigrationRecommendation` (authoritative effective state):

| Code | Meaning |
|------|---------|
| 0 | `SAFE TO MIGRATE` |
| 1 | `MIGRATION BLOCKED` |
| 2 | `REVIEW REQUIRED` or `STALE — REQUALIFY REQUIRED` |
| 3 | `QUALIFICATION INCOMPLETE` or `NO EVIDENCE` |

JSON envelopes expose `recorded_decision`, `effective_decision`, `evidence_validity`,
`is_stale`, `stale_reasons`, `incomplete_reasons`, `migration_recommendation`, hashes,
and `ci_exit_code` so automation does not reimplement Arsenic policy.

## Lock file

`arsenic.lock` records historical references (qualification ids) under the contract and
baseline hashes at write time. After contract/baseline change, lock `qualified` entries
must not be treated as current safety — reporting always reassesses via effective state.
Incomplete or stale results are never placed in `qualified`.

## Relation to ordinary model comparison

`arsenic compare` scores probe suites for relative model behaviour. Application Contracts
answer a different question: whether a **specific application** may safely migrate models
given its contract and production baseline. A high compare score does not imply
`SAFE TO MIGRATE`.

## Guardrails

- Discovery proposes; it does not silently invent block-severity expectations.
- Evidence is authoritative; compatibility % is a summary only.
- Compatibility ≠ intelligence ≠ benchmark performance.
- Corrupt qualification JSON fails closed (load errors), never silent skip → false PASS.
- CLI, HTML, JSON, and CI are renderings of the same effective state.
