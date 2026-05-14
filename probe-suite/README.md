# Standard probe suite

Each `.toml` file defines a `[[probes]]` array of tables. Required fields per probe:

- `name` — unique name for `arsenic probe show`
- `category` — `Morphology` | `Tone` | `Factual` | `Schema` | `Instruction` | `Refusal` | `Semantic`
- `prompt` — user message text
- `tags` — optional string list

Optional fields:

- `system_prompt`
- `known_answer` — for factual probes (substring match in the response)
- `expected_schema` — JSON Schema-like object (`type`, `required`, `properties`) for schema probes
- `[[probes.instructions]]` — instruction checks with `description` and `check = { type = "MaxWords", value = 20 }` style tables

See `ARSENIC_v1_spec.md` in the repo root for full examples.
