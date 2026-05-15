# ARSENIC
 
You get a deprecation notice. Your model is going away in 90 days. You upgrade to the new version, your tests pass, and three weeks later someone notices the support bot sounds different. The sales assistant lost something. The legal tool started hedging in ways that undermine confidence. Nothing broke — it just changed. And you had no way to see it coming.
 
That's the problem ARSENIC solves.
 
Most eval frameworks tell you whether your model passed your tests. ARSENIC tells you what changed about your model's behaviour whether you anticipated it or not — and for regressions it can't automatically fix, it tells you exactly what prompt changes will recover the original behaviour on the new model.
 
---
 
## What it does
 
ARSENIC runs a structured probe suite against two model endpoints in parallel and produces a behavioural drift report across seven dimensions:
 
- **Morphology** — did the response shape change? Length, structure, paragraph count, response type
- **Tone** — formality, assertiveness, hedging, contraction rate
- **Factual** — did known-answer probes regress?
- **Schema** — did structured JSON output stay valid and schema-compliant?
- **Instruction** — did the model continue following explicit instructions?
- **Refusal** — did refusal boundaries shift? Things answered that now aren't, or vice versa
- **Claim** — sentence-level cross-matching: does v2 convey the same information as v1, or did it drop claims, add new ones, or drift on specific values?
Every dimension gets a risk level (Green / Amber / Red) and a direction (Improvement / Regression / Neutral). The upgrade path section splits results into blocking regressions, improvements worth verifying, and neutral changes — so you know exactly what needs attention before you cut over.
 
The mutation engine goes one step further. For regressions it can address, it generates candidate prompt mutations, validates them against v2, and produces certified prompt replacements. You get the diff, a copy button, and confidence the fix actually works.
 
---
 
## Quickstart
 
```bash
cargo build --release
```
 
List the standard probe suite:
 
```bash
./target/release/arsenic probe list
```
 
Compare two models (OpenAI):
 
```bash
export OPENAI_API_KEY=sk-...
 
./target/release/arsenic compare \
  --v1 "openai:gpt-4o-mini" \
  --v2 "openai:gpt-4.1-mini" \
  --v1-key-env OPENAI_API_KEY \
  --v2-key-env OPENAI_API_KEY \
  --standard-suite full \
  --consistency-runs 3 \
  --mutate \
  --output ./report.html \
  --json ./report.json
```
 
Compare local models via Ollama:
 
```bash
export OLLAMA_KEY=ollama
 
./target/release/arsenic compare \
  --v1 "openai:llama3.1:8b" \
  --v2 "openai:llama3.2:3b" \
  --v1-endpoint "http://localhost:11434/v1" \
  --v2-endpoint "http://localhost:11434/v1" \
  --v1-key-env OLLAMA_KEY \
  --v2-key-env OLLAMA_KEY \
  --standard-suite full \
  --consistency-runs 3 \
  --mutate \
  --timeout-secs 120 \
  --output ./report.html \
  --json ./report.json
```
 
The report is a self-contained HTML file. Open it in a browser. Share it with whoever needs to make the upgrade decision.
 
---
 
## Model support
 
ARSENIC is model-agnostic. Any OpenAI-compatible endpoint works out of the box — OpenAI, Ollama, vLLM, LM Studio, Groq. Anthropic and Google have native adapters.
 
```bash
# Anthropic
./target/release/arsenic compare \
  --v1 "anthropic:claude-3-haiku-20240307" \
  --v2 "anthropic:claude-3-5-haiku-20241022" \
  --v1-key-env ANTHROPIC_API_KEY \
  --v2-key-env ANTHROPIC_API_KEY \
  --standard-suite full \
  --mutate \
  --output ./report.html
 
# Google
./target/release/arsenic compare \
  --v1 "google:gemini-1.5-flash" \
  --v2 "google:gemini-2.0-flash" \
  --v1-key-env GOOGLE_API_KEY \
  --v2-key-env GOOGLE_API_KEY \
  --standard-suite full \
  --output ./report.html
```
 
---
 
## Probe suite
 
The standard suite ships with 18 probes across 7 categories covering factual accuracy, schema compliance, instruction following, refusal boundaries, tone, morphology, and open-ended semantic consistency.
 
Bring your own production prompts alongside the standard suite:
 
```bash
./target/release/arsenic compare \
  --v1 "openai:gpt-4o-mini" \
  --v2 "openai:gpt-4.1-mini" \
  --v1-key-env OPENAI_API_KEY \
  --v2-key-env OPENAI_API_KEY \
  --standard-suite full \
  --user-corpus ./my-prompts/ \
  --mutate \
  --output ./report.html
```
 
User corpus probes are TOML files in the same format as the standard suite. You can annotate them with expected behaviour to make valence scoring more precise:
 
```toml
[[probes]]
name = "support_greeting"
category = "Tone"
prompt = "Hi, I'm having trouble with my order."
expected_verbosity = "Moderate"
expected_tone = "Formal"
refusal_expectation = "ShouldAnswer"
mutation_hint = "If tone regresses, add: respond in a warm, professional tone."
tags = ["support", "tone", "production"]
```
 
Validate a corpus before running:
 
```bash
./target/release/arsenic probe validate ./my-prompts/
```
 
---
 
## Claim cross-matching
 
The claim dimension is where ARSENIC differs from a standard eval framework.
 
Whole-response similarity scores miss the things that actually matter. Two responses can look similar in embedding space but one says "the rate is 4.5%" and the other says "the rate varies." Two responses can use completely different phrasing and convey identical information. Cosine similarity on the full response can't tell these apart.
 
ARSENIC extracts informationally significant sentences from each response, strips scaffolding ("Great question!", "I hope this helps", "In conclusion"), identifies claim anchors — numeric values, dates, named entities — and cross-matches claims between v1 and v2 at the sentence level. Dropped claims, new claims, and anchor drift (where a specific value changes between versions) are surfaced separately.
 
A probe that drops "the interest rate is 4.5%" and replaces it with "interest rates vary" is a different finding from one that says the same thing in a longer sentence. The claim dimension catches the first. Cosine similarity doesn't.
 
---
 
## Mutation engine
 
Run with `--mutate` to enable the prompt mutation engine.
 
For each blocking regression, ARSENIC generates a candidate prompt mutation, runs it against v2, and checks whether the risk improves. Strategies are rule-based and drift-informed — if v2 dropped specific claim anchors, the mutation adds an explicit instruction to cover them. If v2 became more verbose, it adds a length constraint. If v2 over-hedged, it adds a directness instruction.
 
Mutations that validate are certified — the report shows the original prompt, the mutated prompt, and a copy button. Mutations that don't validate after three strategy attempts are marked for manual review.
 
The engine is deterministic. No LLM is used to generate mutations. The certified prompt is something you can put in a test and trust.
 
---
 
## Consistency scoring
 
By default ARSENIC runs each probe 3 times per model (`--consistency-runs 3`). Variance across runs is measured and reported as a separate dimension.
 
A model that gives inconsistent answers on repeated identical prompts is a different problem from one that gives consistently different answers. The consistency dimension surfaces both — a v2 that's more variable than v1 is flagged as a regression even if each individual response looks acceptable.
 
Use `--consistency-runs 1` to match v1 behaviour and halve your API spend.
 
---
 
## Flags
 
| Flag | Default | Description |
|------|---------|-------------|
| `--standard-suite` | — | Probe categories to run: `full`, `factual`, `tone`, `morphology`, `schema`, `instruction`, `refusal`, `semantic`. Comma-separate multiple. |
| `--user-corpus` | — | Path to directory of user-defined probe TOML files |
| `--consistency-runs` | `3` | Runs per probe per model for consistency scoring |
| `--mutate` | off | Run the prompt mutation engine after comparison |
| `--no-semantic` | off | Disable semantic similarity dimension |
| `--latency-affects-risk` | off | Include Amber latency in overall probe risk |
| `--concurrency` | `10` | Max parallel requests per endpoint |
| `--timeout-secs` | `30` | Request timeout — increase for slow local models |
| `--output` | — | HTML report output path |
| `--json` | — | JSON report output path |
| `--config` | — | Path to `arsenic.toml` config file |
 
---
 
## Config file
 
```toml
[v1]
adapter = "openai"
api_key_env = "OPENAI_API_KEY"
model_id = "gpt-4o-mini"
temperature = 0.0
 
[v2]
adapter = "openai"
api_key_env = "OPENAI_API_KEY"
model_id = "gpt-4.1-mini"
temperature = 0.0
 
[run]
consistency_runs = 3
timeout_secs = 60
standard_suite = "full"
user_corpus = "./my-prompts/"
 
[output]
html = "./reports/latest.html"
json = "./reports/latest.json"
```
 
```bash
./target/release/arsenic compare --config arsenic.toml
```
 
---
 
## Commands
 
```
arsenic compare                    Run probe suite, write reports
arsenic probe list                 List standard probes
arsenic probe list --category tone Filter by category
arsenic probe show <name>          Show one probe as JSON
arsenic probe validate <path>      Validate user corpus TOML
arsenic report render <json>       Re-render a saved JSON report
arsenic report summary <json>      Print summary to stdout
arsenic models download <name>     Download HuggingFace model weights
```
 
---
 
## Environment variables
 
| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google API key |
| `OLLAMA_KEY` | Any non-empty string for Ollama (not validated) |
| `ARSENIC_LOG` | Log level: `error`, `warn`, `info`, `debug` |
| `ARSENIC_SUITE_PATH` | Override default probe suite directory |
 
---
 
## Workspace
 
```
crates/
  arsenic-core/       Types, comparison engine, claim matching, mutation engine
  arsenic-probes/     TOML probe loader
  arsenic-adapters/   OpenAI-compatible, Anthropic, Google adapters
  arsenic-report/     HTML / JSON / Markdown report rendering
  arsenic-cli/        arsenic binary
probe-suite/standard/ Standard probe suite (18 probes, 7 categories)
report-templates/     Tera templates
```
 
Built in Rust. Fast. No runtime dependencies. The report is a single self-contained HTML file with no external CDN calls after the font load.
 
---
 
## Licence
 
Apache 2.0