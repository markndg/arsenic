# Contract Discovery

```bash
arsenic init
```

Discovery inspects the project tree for AI-application artefacts:

- plain prompt files (`.txt`, `.md`, `.prompt`)
- JSON / YAML prompt definitions
- Python / TypeScript / JavaScript source hints
- OpenAI-style message structures and tool schemas
- JSON Schema structured outputs
- existing Arsenic probe suites

## Honesty

Discovery is **approximate**. Every hit reports:

- source file
- line range when available
- parser name
- confidence
- whether it needs confirmation (`auto_accepted` is false by default for proposals)

Critical behavioural expectations are **never silently invented**.

```bash
# Explicit opt-in to accept proposed items into the draft contract
arsenic init --accept-proposals

# Or seed from a hand-authored contract
arsenic init --contract path/to/contract.json
```

Users can accept, reject, edit, or add requirements manually by editing
`.arsenic/contract/contract.json`.
