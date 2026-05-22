# CI Examples

## GitHub Actions

`github-actions.yaml` — gates model upgrades on pull requests that touch your prompt corpus.

Copy it into your own repo's `.github/workflows/` directory, set `OPENAI_API_KEY` as a repository secret, and adjust `--v1` and `--v2` to the models you're comparing.

Requires `--fail-on-blocking` support (arsenic v0.2.0+).
