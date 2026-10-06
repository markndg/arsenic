# Model Qualification

Continuous Model Qualification compares candidate models against the current
**Application Contract** and **production baseline**.

```bash
arsenic qualify openai:gpt-x anthropic:claude-y google:gemini-z
```

Omit models to run the configured candidate pool from `arsenic.toml` / `arsenic.lock`.

## Decisions

Overall decision is deterministic from contract-item outcomes:

| Decision | Rule |
|----------|------|
| `BLOCK` | Any block-severity behavioural failure |
| `REVIEW` | Any review-severity failure, or infrastructure-only failures |
| `PASS_WITH_WARNINGS` | Warnings only |
| `PASS_WITH_PATCH` | Pass after validated repair |
| `PASS` | Otherwise |
| `STALE` | Derived when contract/baseline hashes diverge (history preserved) |

Provider/runtime failures (`AUTH_ERROR`, `TIMEOUT`, `RATE_LIMITED`, …) are **not** behavioural BLOCK/PASS. CI returns exit code **3**.

## `--changed`

Re-runs a candidate when any of these change: contract hash, baseline hash, model, prompt hashes, tools hash, temperature/max_tokens/endpoint, qualification thresholds. Matching prior PASS fingerprints are skipped.

## Lockfile

`arsenic.lock` records production baseline and qualification outcomes:

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

## Impact

```bash
arsenic impact openai:gpt-x
```

Groups BLOCKERS / REVIEW / PRESENTATION DRIFT / UNAFFECTED for change review.
