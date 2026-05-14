# ARSENIC

Model-agnostic LLM drift certification: run a probe suite against two endpoints, compare responses on multiple dimensions (including v2 claim cross-matching, latency, consistency, and optional prompt mutations), and emit HTML/JSON/Markdown reports.

## Quickstart

Build the CLI from the repository root:

```bash
cargo build --release
# binary: target/release/arsenic
```

List standard probes:

```bash
cargo run --release -- probe list
```

Compare two OpenAI-compatible models (for example OpenAI or Ollama at `http://localhost:11434/v1`):

```bash
export OPENAI_API_KEY=sk-...
cargo run --release -- compare \
  --v1 "openai:gpt-4o-mini" \
  --v2 "openai:gpt-4o-mini" \
  --v1-key-env OPENAI_API_KEY \
  --v2-key-env OPENAI_API_KEY \
  --standard-suite factual \
  --output ./report.html \
  --json ./report.json
```

### v2 flags (additive)

- **`--consistency-runs N`** (default **3**): runs each probe **N** times per model for `ConsistencyDiff`. Use **`--consistency-runs 1`** for v1-style cost (single run per side, no consistency dimension).
- **`--latency-affects-risk`**: when set, Amber latency contributes to worst-case probe risk like other dimensions; by default only **Red** latency is merged into overall risk unless this flag is set.
- **`--mutate`**: after the main compare, runs the rule-based mutation engine against **v2** only, validates with `compare_one`, attaches `mutation_results`, and refreshes the structured **upgrade path** in the report.
- **Probe TOML** optional fields: `expected_verbosity`, `expected_tone`, `refusal_expectation`, `mutation_hint`, `custom_assertions` (see `ARSENIC_v2_spec.md`).

In `arsenic.toml`, `[run]` may set `consistency_runs` and `latency_affects_risk` to override CLI defaults.

Ollama (OpenAI-compatible API):

```bash
export OLLAMA_KEY=dummy
cargo run --release -- compare \
  --v1 "openai:llama3.1:8b" \
  --v2 "openai:llama3.2:3b" \
  --v1-endpoint "http://localhost:11434/v1" \
  --v2-endpoint "http://localhost:11434/v1" \
  --v1-key-env OLLAMA_KEY \
  --v2-key-env OLLAMA_KEY \
  --standard-suite full \
  --output ./report.html \
  --json ./report.json
```

Disable semantic similarity (no sentence-level hash embedding path) and rely on other dimensions only:

```bash
cargo run --release -- compare ... --no-semantic
```

### Download embedding weights (optional)

For future Candle/BGE integration, model weights can be pulled without the Hugging Face CLI:

```bash
cargo run --release -- models download bge-small-en-v1.5 --output ~/.arsenic/models/
```

Files are fetched from `https://huggingface.co/<repo>/resolve/main/…` (bare names default to `BAAI/<name>`). Each artifact gets a companion `*.sha256` checksum file after download.

### Config file

You can use `arsenic.toml` instead of long flags; see `ARSENIC_v1_spec.md` and `ARSENIC_v2_spec.md` for the layout. Example:

```bash
cargo run --release -- compare --config arsenic.toml
```

### Environment

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google Generative Language API key |
| `ARSENIC_LOG` | Log filter (e.g. `info`, `debug`) |
| `ARSENIC_SUITE_PATH` | Override default probe directory (`probe-suite/standard`) |

### Probe format

Probe definitions live under `probe-suite/standard/` as TOML files. See the specs and inline examples in those files.

### Commands

| Command | Description |
|---------|-------------|
| `arsenic compare` | Run probes against v1 and v2, write reports |
| `arsenic probe list` | List standard probes (optional `--category`) |
| `arsenic probe show <name>` | Print one probe as JSON |
| `arsenic probe validate <path>` | Validate user corpus TOML |
| `arsenic validate <path>` | Same as probe validate |
| `arsenic report render <json> --format html\|md\|json --output <path>` | Re-render from saved JSON |
| `arsenic report summary <json>` | Print JSON summary to stdout |
| `arsenic models download <name>` | Download public HF weights (e.g. `bge-small-en-v1.5` → `BAAI/bge-small-en-v1.5`) |

## Workspace layout

- `crates/arsenic-core` — types, comparison engine, probe runner, claim/mutation/embedding helpers
- `crates/arsenic-probes` — TOML loader
- `crates/arsenic-adapters` — OpenAI-compatible, Anthropic, Google HTTP adapters
- `crates/arsenic-report` — HTML / JSON / Markdown rendering
- `crates/arsenic-cli` — `arsenic` binary
- `probe-suite/standard/` — default probe suite
- `report-templates/` — Tera templates for reports

## Notes

- **Semantic similarity** in this build uses deterministic hash-based sentence embeddings and weighted sentence similarity when semantic scoring is enabled; `--no-semantic` sets similarity to `None` and marks the dimension as disabled in the report.
- **Adapters:** `--v1 "anthropic:claude-3-5-sonnet-20241022"` with `--v1-key-env ANTHROPIC_API_KEY`; Google uses `--v1 "google:gemini-1.5-flash"` and `--v1-key-env GOOGLE_API_KEY`.
