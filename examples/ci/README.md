# CI Examples

## GitHub Actions

`github-actions.yaml` — gates model upgrades on pull requests that touch your prompt corpus.

It uses the [`markndg/arsenic-action`](https://github.com/markndg/arsenic-action) wrapper, which packages the binary, the standard probe suite, and the comparison run into a single step.

### Setup

1. Copy `github-actions.yaml` into your repo's `.github/workflows/` directory.
2. Add `OPENAI_API_KEY` as a repository secret (or whichever provider key your models need).
3. Edit `old-model` and `new-model` to the pair you want to gate the upgrade on.
4. Point `corpus` at the directory holding your prompt TOML files.

### Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `old-model` | yes | — | Current production model (e.g. `gpt-4.1-mini`). |
| `new-model` | yes | — | Candidate upgrade (e.g. `gpt-5-mini`). |
| `corpus` | yes | — | Path to a directory of probe TOML files. |
| `fail-on-risk` | no | `high` | Risk threshold that fails the job (`low` / `medium` / `high`). |

### Running arsenic directly

If you'd rather call the binary yourself — for example to use `--baseline` for cache replay, customise `--consistency-runs`, or upload the HTML/JSON reports as artefacts — install it from the GitHub Releases page and invoke `arsenic compare` directly:

```yaml
- name: Install Arsenic
  run: |
    curl -L https://github.com/markndg/arsenic/releases/latest/download/arsenic-linux-x86_64.tar.gz \
      | tar -xz -C /usr/local/bin arsenic

- name: Compare against cached baseline
  run: |
    arsenic compare \
      --baseline prod-gpt-4.1-mini \
      --v2 "openai:gpt-5-mini" \
      --v2-key-env OPENAI_API_KEY \
      --standard-suite full \
      --output report.html \
      --json report.json
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

- name: Gate on blocking regressions
  run: arsenic report summary report.json --fail-on-blocking

- uses: actions/upload-artifact@v4
  if: always()
  with:
    name: arsenic-report
    path: |
      report.html
      report.json
```

This direct-invocation path requires arsenic `v0.3.0+` for baseline replay; the `arsenic-action` wrapper is the simpler choice for most teams.
