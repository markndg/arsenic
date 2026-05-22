# ARSENIC AI-assessment probe suite

A targeted probe suite for AI used in online exams, AI-assisted marking, item
generation, and AI-tutoring deployments. Designed to catch behavioural drift
between LLM versions before it changes a candidate's mark, the difficulty of a
generated item, or the firmness of an academic-integrity guardrail.

## Background

LLM-based marking and item generation are entering live use at exam boards,
universities, and edtech platforms. Regulators are catching up:

- **Ofqual** (UK) requires evidence that any automated marking process is
  stable, fair, and defensible.
- The **EU AI Act** classifies AI used in education and assessment as
  *high-risk*, requiring documented evidence of post-deployment monitoring.
- Awarding bodies and universities must justify any change in marking
  behaviour to candidates, parents, and (in the UK) the Independent Adjudicator.

A silent model upgrade can:

- shift the band a candidate receives on a fixed rubric;
- change the difficulty of a generated exam item;
- soften an academic-integrity refusal so the tutor starts writing the
  candidate's essay;
- alter the verdict of a model-as-classifier "is this AI-written?" check;
- introduce demographic sensitivity (different mark for the same essay under
  different candidate names);
- break the JSON contract a gradebook ingestion pipeline depends on.

These probes operationalise those signals so they can be measured across model
versions *before* any of them reaches a candidate.

## Probes

| File / probe | Tests |
|--------------|-------|
| `rubric_marking_history_gcse_essay` | Fixed rubric + fixed answer; does v2 award the same band as v1? |
| `rubric_marking_borderline_persuasive_writing` | Borderline answer at a band boundary — drift here is the high-cost case. |
| `rubric_short_answer_science_marking` | 4-mark scheme; does v2 credit the same mark-scheme points? |
| `item_generation_gcse_biology_mcq` | Does v2 still generate items at GCSE level with misconception-based distractors? |
| `item_generation_distractor_quality_maths` | Are distractors still tied to specific procedural errors, or has v2 started producing arbitrary wrong numbers? |
| `paraphrase_invariance_factual_capital` / `..._v2` | Same item, trivially rephrased — the model must return the same answer. |
| `numeric_tolerance_pi_marking` | Has v2 tightened or loosened numeric tolerance for STEM marking? |
| `demographic_invariance_essay_anglo_name` / `..._non_anglo_name` | Same essay, different candidate name — mark must not move. Manual review. |
| `accommodations_plain_english_rewrite` | Plain-English rewrite for SEND candidates without changing the assessment target. |
| `tutor_socratic_hint_not_answer` | Tutor must hint, not solve. Instruction check that v2 does not leak the final numeric answer. |
| `refusal_write_my_exam_answer` | Does v2 still refuse to produce an essay for exam submission? |
| `refusal_evade_ai_detection` | Does v2 still refuse to paraphrase content to evade AI-detection tooling? |
| `refusal_examiner_roleplay_mark_scheme_leak` | Does v2 still refuse to impersonate an examiner or speculate on unreleased mark schemes? |
| `ai_text_detection_verdict_human_passage` | Verdict stability for model-as-AI-classifier deployments — false positives are a legal liability. |
| `mark_scheme_structured_output_json` | Does v2 still emit the JSON contract the gradebook ingests? |
| `item_translation_difficulty_preservation_french` | For multilingual exams: does v2 preserve cognitive demand when translating an item? |

## Scoring

The suite exercises every ARSENIC dimension:

- **Factual** — paraphrase invariance, numeric tolerance, AI-detection verdict.
- **Schema** — mark-scheme JSON output contract.
- **Instruction** — Socratic hint-not-answer guardrail.
- **Refusal** — write-my-essay, evade-detection, examiner roleplay.
- **Tone** — accommodations plain-English register.
- **Semantic** — rubric marking, item generation, demographic invariance, translation difficulty.

Two pairs of probes (`paraphrase_invariance_*` and `demographic_invariance_*`)
are designed to be read side-by-side. ARSENIC will score each probe
independently against v1 vs v2; the operator should additionally inspect the
pair within each model version to confirm invariance holds.

Probes tagged `manual-review` warrant human inspection regardless of
automated score — for fairness and academic-integrity probes, the nuance
matters more than the metric.

## Usage

```bash
export OPENAI_API_KEY=sk-...

arsenic compare \
  --v1 "openai:gpt-4o-mini" \
  --v2 "openai:gpt-4.1-mini" \
  --v1-key-env OPENAI_API_KEY \
  --v2-key-env OPENAI_API_KEY \
  --user-corpus-only \
  --user-corpus ./probe-suite/ai-assessment \
  --consistency-runs 3 \
  --mutate \
  --output report-ai-assessment.html \
  --json report-ai-assessment.json
```

Run with `--consistency-runs 3` (the default) so the report flags
inconsistency on borderline marking — a model that gives a different band on
repeated identical prompts is itself a regression for assessment use, even if
each individual mark is defensible.

Combine with `--mutate` so that, where a guardrail or rubric-adherence
regression is found, ARSENIC proposes a validated prompt patch (e.g. a
hardened system prompt) that recovers the v1 behaviour against v2.

## The story this tells

A model upgrade that changes a candidate's mark, the difficulty of a
generated item, or the firmness of an academic-integrity guardrail is the
kind of regression that surfaces *after* deployment — usually after a
candidate or parent raises it. By then the awarding body owes someone an
explanation.

These probes catch the signal in a pre-deployment comparison run, with
evidence that can be handed to the regulator.

The point of ARSENIC.
