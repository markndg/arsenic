# Model Qualification

Continuous Model Qualification compares candidate models against the current
**Application Contract** and **production baseline**.

```bash
arsenic qualify openai:gpt-x anthropic:claude-y google:gemini-z
```

Omit models to run the configured candidate pool from `arsenic.toml` / `arsenic.lock`.

## Decisions vs effective state

Recorded behavioural decisions are immutable. **Current** migration recommendations
are derived with `assess_qualification` against the live contract/baseline hashes.

| Recorded decision | Rule |
|-------------------|------|
| `BLOCK` | Any block-severity behavioural failure |
| `REVIEW` | Any review-severity failure |
| `PASS_WITH_WARNINGS` | Warnings only |
| `PASS_WITH_PATCH` | Pass after validated repair (must bind to current hashes/model) |
| `PASS` | Otherwise |

| Effective overlay | Meaning |
|-------------------|---------|
| `STALE` | Contract/baseline hash diverged — requalify; never SAFE |
| `INCOMPLETE` | Provider/runtime errors — never SAFE; CI exit 3 |

Provider/runtime failures (`AUTH_ERROR`, `TIMEOUT`, `RATE_LIMITED`, `INVALID_RESPONSE`, …)
are **not** behavioural BLOCK/PASS. Migration recommendation is `QUALIFICATION INCOMPLETE`.

## `--changed`

Re-runs a candidate unless a **currently valid** SAFE qualification exists for the same
input fingerprint (contract, baseline, model, prompts, tools, temperature/max_tokens/endpoint,
thresholds). Stale or incomplete priors never skip requalification.

## Lockfile

`arsenic.lock` records production baseline and historical qualification references under
the contract/baseline hashes at write time. Entries in `qualified` are never written for
incomplete or stale evidence. After a contract/baseline change, treat lock `qualified`
as historical — reporting reassesses effective state.

```toml
application = "customer-support"

[production]
model = "openai:gpt-current"
baseline = "baseline-0001"

[[qualified]]
model = "openai:gpt-x"
qualification = "qual-0001"
```

## Candidate pool

Configure candidates in `arsenic.toml` with optional tags:

```toml
[[candidates]]
model = "local:qwen"
tags = ["local", "cheap"]
```

```bash
arsenic qualify --tag local
arsenic qualify --changed
```

## Repair

```bash
arsenic qualify --repair openai:gpt-x
arsenic patch show <qualification-id>
arsenic patch apply <qualification-id> --yes
```

Repairs are explicit, diffable, reversible, and never silently modify application prompts.
A patch validated on stale evidence cannot certify the current contract.

## Impact

```bash
arsenic impact openai:gpt-x
```

Groups BLOCKERS / REVIEW / PRESENTATION DRIFT / UNAFFECTED and prints the **effective**
migration recommendation (SAFE / BLOCKED / STALE / INCOMPLETE / REVIEW).
