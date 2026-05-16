//! Signal-ranked reconcile: delta analysis, strategy selection, and validation loop.

use std::collections::HashSet;

use chrono::Utc;
use uuid::Uuid;

use crate::adapter::ModelAdapter;
use crate::category_infer::infer_probe_category;
use crate::claim::is_spurious_anchor_value;
use crate::comparison::ComparisonEngine;
use crate::mutation::apply_mutations;
use crate::reconcile::{
    ReconcileAttempt, ReconcileDimension, ReconcileResult, ReconcileSignal, SignalDetail,
};
use crate::types::{
    Claim, DriftDirection, FinishReason, ModelInfo, ModelResponse, MutationStrategy, Probe,
    ProbeCategory, ProbeResult, ProbeSource, ResponsePair, RiskLevel,
};

/// Default validation attempts when `max_strategies` is unset or zero.
pub const DEFAULT_MAX_STRATEGIES: usize = 5;

/// Baseline word count treated as long-form open-ended content.
const LONG_FORM_WORD_THRESHOLD: usize = 150;

/// Dropped-claim count above which topic-coverage strategies apply on semantic probes.
const HIGH_DROPPED_CLAIM_COUNT: usize = 5;

/// Anchors per cumulative `AddClaimInstruction` step when splitting a large value list.
const CLAIM_ANCHOR_CHUNK: usize = 4;

/// Build the synthetic probe used for single-prompt reconcile.
pub fn build_reconcile_probe(prompt: String, system_prompt: Option<String>) -> Probe {
    Probe {
        id: Uuid::new_v4(),
        name: "reconcile_target".to_string(),
        category: infer_probe_category(&prompt),
        prompt,
        system_prompt,
        known_answer: None,
        expected_schema: None,
        instructions: vec![],
        tags: vec!["reconcile".to_string()],
        source: ProbeSource::UserDefined,
        expected_verbosity: None,
        expected_tone: None,
        refusal_expectation: None,
        mutation_hint: None,
        custom_assertions: vec![],
    }
}

/// Construct a [`ModelResponse`] from supplied text (Modes 2/3).
pub fn synthetic_model_response(
    probe_id: Uuid,
    model_label: &str,
    model_id: &str,
    content: &str,
) -> ModelResponse {
    let word_count = content.split_whitespace().count();
    ModelResponse {
        probe_id,
        model_label: model_label.to_string(),
        model_id: model_id.to_string(),
        content: content.to_string(),
        token_count: word_count.max(1),
        latency_ms: 0,
        finish_reason: FinishReason::Stop,
        timestamp: Utc::now(),
        raw: serde_json::Value::Null,
    }
}

fn risk_rank(r: &RiskLevel) -> u8 {
    match r {
        RiskLevel::Green => 0,
        RiskLevel::Amber => 1,
        RiskLevel::Red => 2,
    }
}

fn risk_improved(before: &RiskLevel, after: &RiskLevel) -> bool {
    risk_rank(after) < risk_rank(before)
}

/// Extract dimension signals and rank by magnitude (descending).
pub fn rank_signals(delta: &ProbeResult) -> Vec<ReconcileSignal> {
    let mut signals = Vec::new();
    let claim = &delta.dimensions.claim;
    let v1_claim_n = claim.v1_claims.len().max(1);

    if !claim.drifted_claims.is_empty() {
        let drifted: Vec<_> = claim
            .drifted_claims
            .iter()
            .flat_map(|d| d.drifted_anchors.clone())
            .collect();
        if !drifted.is_empty() {
            signals.push(ReconcileSignal {
                dimension: ReconcileDimension::Claim,
                magnitude: 1.0,
                direction: DriftDirection::Regression,
                detail: SignalDetail::AnchorDrift { drifted },
            });
        }
    }

    if !claim.dropped_claims.is_empty() {
        let mut anchors: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for c in &claim.dropped_claims {
            for a in &c.anchors {
                if is_spurious_anchor_value(&a.value) {
                    continue;
                }
                if seen.insert(a.value.clone()) {
                    anchors.push(a.value.clone());
                }
            }
        }
        let count = claim.dropped_claims.len();
        let magnitude = (count as f64 / v1_claim_n as f64).clamp(0.0, 1.0);
        if magnitude > 0.0 {
            signals.push(ReconcileSignal {
                dimension: ReconcileDimension::Claim,
                magnitude,
                direction: DriftDirection::Regression,
                detail: SignalDetail::DroppedClaims { anchors, count },
            });
        }
    }

    if let Some(f) = &delta.dimensions.factual {
        if f.regression {
            signals.push(ReconcileSignal {
                dimension: ReconcileDimension::Factual,
                magnitude: 1.0,
                direction: DriftDirection::Regression,
                detail: SignalDetail::FactualRegression {
                    v1_answer: f.v1_answer_extract.clone(),
                    v2_answer: f.v2_answer_extract.clone(),
                },
            });
        }
    }

    if delta.dimensions.refusal.new_refusal {
        signals.push(ReconcileSignal {
            dimension: ReconcileDimension::Refusal,
            magnitude: 1.0,
            direction: DriftDirection::Regression,
            detail: SignalDetail::RefusalNew,
        });
    }

    if let Some(s) = &delta.dimensions.schema {
        if !s.v2_schema_valid || !s.v2_valid_json {
            signals.push(ReconcileSignal {
                dimension: ReconcileDimension::Schema,
                magnitude: 0.9,
                direction: s.direction,
                detail: SignalDetail::SchemaInvalid {
                    missing_fields: s.v2_missing_fields.clone(),
                },
            });
        }
    }

    let morph = &delta.dimensions.morphology;
    let token_pct = morph.delta.token_delta_pct.abs();
    if token_pct > 5.0 || morph.delta.response_type_changed {
        let v2_shorter = morph.v2.word_count < morph.v1.word_count;
        let magnitude = (token_pct / 100.0).min(1.0).max(if morph.delta.response_type_changed {
            0.35
        } else {
            0.0
        });
        if magnitude > 0.0 {
            signals.push(ReconcileSignal {
                dimension: ReconcileDimension::Morphology,
                magnitude,
                direction: morph.direction,
                detail: SignalDetail::MorphologyDelta {
                    token_delta_pct: morph.delta.token_delta_pct,
                    response_type_changed: morph.delta.response_type_changed,
                    v2_shorter,
                },
            });
        }
    }

    let tone = &delta.dimensions.tone;
    let formality = tone.delta.formality_delta.abs();
    let v2_less_formal = tone.delta.formality_delta < -0.08;
    let v2_over_hedged = tone.delta.hedge_word_delta > 3;
    if formality > 0.05 || v2_over_hedged {
        let magnitude = formality.max(if v2_over_hedged { 0.25 } else { 0.0 });
        signals.push(ReconcileSignal {
            dimension: ReconcileDimension::Tone,
            magnitude,
            direction: tone.direction,
            detail: SignalDetail::ToneDelta {
                formality_delta: tone.delta.formality_delta,
                assertiveness_delta: tone.delta.assertiveness_delta,
                v2_less_formal,
                v2_over_hedged,
            },
        });
    }

    let sem = &delta.dimensions.semantic;
    if !sem.semantic_scoring_disabled {
        if let Some(sim) = sem.cosine_similarity {
            if sim < sem.similarity_threshold {
                let magnitude = (1.0 - sim).clamp(0.0, 1.0);
                signals.push(ReconcileSignal {
                    dimension: ReconcileDimension::Semantic,
                    magnitude,
                    direction: sem.direction,
                    detail: SignalDetail::SemanticDrift { similarity: sim },
                });
            }
        }
    }

    signals.sort_by(|a, b| {
        b.magnitude
            .partial_cmp(&a.magnitude)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    signals
}

/// Map ranked signals to mutation strategies in magnitude order (not the fixed compare/mutate sequence).
pub fn signals_to_strategies(signals: &[ReconcileSignal], delta: &ProbeResult) -> Vec<MutationStrategy> {
    let mut ordered = signals.to_vec();
    ordered.sort_by(|a, b| {
        b.magnitude
            .partial_cmp(&a.magnitude)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out: Vec<MutationStrategy> = Vec::new();
    let morph = &delta.dimensions.morphology;

    for signal in &ordered {
        match &signal.detail {
            SignalDetail::DroppedClaims { anchors, count } => {
                append_dropped_claim_strategies(&mut out, anchors, *count, delta, morph);
            }
            SignalDetail::AnchorDrift { drifted } => {
                push_unique(&mut out, MutationStrategy::AddPrecisionInstruction);
                let mut values: Vec<String> = Vec::new();
                let mut seen = HashSet::new();
                for d in drifted {
                    if is_spurious_anchor_value(&d.v1_value) {
                        continue;
                    }
                    if seen.insert(d.v1_value.clone()) {
                        values.push(d.v1_value.clone());
                    }
                }
                if !values.is_empty() {
                    push_unique(
                        &mut out,
                        MutationStrategy::AddClaimInstruction { required_values: values },
                    );
                }
            }
            SignalDetail::FactualRegression { v1_answer, .. } => {
                if !v1_answer.trim().is_empty() {
                    push_unique(
                        &mut out,
                        MutationStrategy::AddClaimInstruction {
                            required_values: vec![v1_answer.clone()],
                        },
                    );
                }
            }
            SignalDetail::RefusalNew => {
                push_unique(&mut out, MutationStrategy::SoftenPhrasing);
                push_unique(&mut out, MutationStrategy::AddEducationalContext);
            }
            SignalDetail::SchemaInvalid { missing_fields } => {
                let hint = if missing_fields.is_empty() {
                    "Return only valid JSON matching the requested schema with all required fields."
                        .to_string()
                } else {
                    format!(
                        "Return valid JSON including required fields: {}.",
                        missing_fields.join(", ")
                    )
                };
                push_unique(
                    &mut out,
                    MutationStrategy::ReinforceInstruction {
                        instruction_text: hint,
                    },
                );
            }
            SignalDetail::MorphologyDelta {
                v2_shorter: true,
                ..
            } => {
                let min_words = ((morph.v1.word_count as f64) * 0.85).ceil() as usize;
                push_unique(
                    &mut out,
                    MutationStrategy::AddDetailInstruction {
                        min_words: min_words.max(8),
                    },
                );
            }
            SignalDetail::MorphologyDelta {
                v2_shorter: false,
                ..
            } => {
                if morph.v2.word_count > morph.v1.word_count {
                    let max_words = ((morph.v1.word_count as f64) * 1.15).ceil() as usize;
                    push_unique(
                        &mut out,
                        MutationStrategy::AddLengthConstraint {
                            max_words: max_words.max(8),
                        },
                    );
                }
                if morph.delta.response_type_changed {
                    push_unique(&mut out, MutationStrategy::AddDirectnessInstruction);
                }
            }
            SignalDetail::ToneDelta {
                v2_less_formal: true,
                ..
            } => {
                push_unique(&mut out, MutationStrategy::AddFormalityInstruction);
            }
            SignalDetail::ToneDelta {
                v2_over_hedged: true,
                ..
            } => {
                push_unique(&mut out, MutationStrategy::AddConfidenceInstruction);
            }
            SignalDetail::ToneDelta { .. } => {}
            SignalDetail::SemanticDrift { .. } => {
                let mut anchors: Vec<String> = Vec::new();
                let mut seen = HashSet::new();
                for c in &delta.dimensions.claim.dropped_claims {
                    for a in &c.anchors {
                        if is_spurious_anchor_value(&a.value) {
                            continue;
                        }
                        if seen.insert(a.value.clone()) {
                            anchors.push(a.value.clone());
                        }
                    }
                }
                if !anchors.is_empty() {
                    push_unique(
                        &mut out,
                        MutationStrategy::AddClaimInstruction { required_values: anchors },
                    );
                }
            }
        }
    }
    expand_strategies_for_attempts(out)
}

fn is_long_form_semantic(delta: &ProbeResult) -> bool {
    delta.probe.category == ProbeCategory::Semantic
        && delta.dimensions.morphology.v1.word_count >= LONG_FORM_WORD_THRESHOLD
}

fn append_dropped_claim_strategies(
    out: &mut Vec<MutationStrategy>,
    anchors: &[String],
    dropped_count: usize,
    delta: &ProbeResult,
    morph: &crate::types::MorphologyDiff,
) {
    if anchors.is_empty() && dropped_count == 0 {
        push_unique(out, MutationStrategy::AddDirectnessInstruction);
        return;
    }

    let long_form = is_long_form_semantic(delta);
    if long_form && dropped_count > HIGH_DROPPED_CLAIM_COUNT {
        let topics = extract_coverage_topics(
            &delta.dimensions.claim.v1_claims,
            &delta.dimensions.claim.dropped_claims,
        );
        if !topics.is_empty() {
            push_unique(
                out,
                MutationStrategy::AddTopicCoverageInstruction { topics },
            );
        }
        if !anchors.is_empty() {
            let subset: Vec<_> = anchors.iter().take(8).cloned().collect();
            push_unique(
                out,
                MutationStrategy::AddClaimInstruction {
                    required_values: subset,
                },
            );
        }
        if morph.v2.word_count < morph.v1.word_count {
            let min_words = ((morph.v1.word_count as f64) * 0.85).ceil() as usize;
            push_unique(
                out,
                MutationStrategy::AddDetailInstruction {
                    min_words: min_words.max(8),
                },
            );
        }
        return;
    }

    if anchors.is_empty() {
        push_unique(out, MutationStrategy::AddDirectnessInstruction);
        return;
    }

    push_unique(
        out,
        MutationStrategy::AddClaimInstruction {
            required_values: anchors.to_vec(),
        },
    );
}

/// Headings and key phrases from baseline claims for topic-coverage mutations.
///
/// Prefers markdown bold headings, then `#` line headers, then title-case noun phrases.
/// Sentence openers and fragments with verbs are rejected.
pub fn extract_coverage_topics(v1_claims: &[Claim], dropped_claims: &[Claim]) -> Vec<String> {
    const MAX_TOPICS: usize = 6;
    /// When bold + markdown headers yield this many topics, noun-phrase pass is skipped.
    const SKIP_NOUN_PHRASE_PASS_AT: usize = 3;
    let mut seen = HashSet::new();
    let mut topics = Vec::new();
    let claims: Vec<&Claim> = dropped_claims.iter().chain(v1_claims.iter()).collect();

    // Pass 1: every `**bold**` span (headings anywhere in the claim).
    for c in &claims {
        for bold in extract_all_bold_spans(&c.text) {
            try_push_topic(&mut topics, &mut seen, MAX_TOPICS, bold);
        }
    }

    // Pass 2: markdown `#` / `##` line headers.
    if topics.len() < MAX_TOPICS {
        for c in &claims {
            for line in c.text.lines() {
                if let Some(h) = extract_markdown_line_header(line) {
                    try_push_topic(&mut topics, &mut seen, MAX_TOPICS, h);
                }
            }
        }
    }

    // Pass 3: title-case noun phrases only when passes 1–2 found fewer than 3 topics.
    if topics.len() < SKIP_NOUN_PHRASE_PASS_AT && topics.len() < MAX_TOPICS {
        for c in &claims {
            for phrase in extract_title_case_noun_phrases(&c.text) {
                try_push_topic(&mut topics, &mut seen, MAX_TOPICS, phrase);
            }
        }
    }

    topics
}

fn try_push_topic(
    topics: &mut Vec<String>,
    seen: &mut HashSet<String>,
    max_topics: usize,
    candidate: String,
) {
    if topics.len() >= max_topics {
        return;
    }
    let label = normalize_topic_label(&candidate);
    let label = strip_leading_topic_stopwords(&label);
    if is_usable_topic(&label) && seen.insert(label.clone()) {
        topics.push(label);
    }
}

fn extract_all_bold_spans(text: &str) -> Vec<String> {
    let mut spans = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("**") {
        rest = &rest[start + 2..];
        if let Some(end) = rest.find("**") {
            let inner = rest[..end].trim();
            if !inner.is_empty() {
                spans.push(inner.to_string());
            }
            rest = &rest[end + 2..];
        } else {
            break;
        }
    }
    spans
}

fn extract_markdown_line_header(line: &str) -> Option<String> {
    let t = line.trim();
    let hashes = t.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let title = t[hashes..].trim();
    if title.is_empty() {
        None
    } else {
        Some(title.to_string())
    }
}

fn extract_title_case_noun_phrases(text: &str) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut phrases = Vec::new();
    let mut i = 0;
    while i < words.len() {
        if !is_topic_phrase_word(words[i], true) {
            i += 1;
            continue;
        }
        let start = i;
        i += 1;
        while i < words.len() && is_topic_phrase_word(words[i], false) {
            i += 1;
        }
        let len = i - start;
        if (2..=5).contains(&len) {
            let phrase = words[start..i]
                .iter()
                .map(|w| trim_word_punctuation(w))
                .collect::<Vec<_>>()
                .join(" ");
            phrases.push(phrase);
        }
    }
    phrases
}

fn is_topic_phrase_word(word: &str, is_first: bool) -> bool {
    let w = trim_word_punctuation(word);
    if w.is_empty() || w.len() < 2 {
        return false;
    }
    if is_topic_verb_word(&w) {
        return false;
    }
    if w.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-') && w.len() <= 8 {
        return true;
    }
    if is_first && topic_leading_stopword(&w) {
        return false;
    }
    w.chars().next().is_some_and(|c| c.is_ascii_uppercase())
}

fn trim_word_punctuation(word: &str) -> String {
    word.trim_matches(|c: char| {
        matches!(
            c,
            ',' | '.' | ':' | ';' | '!' | '?' | '(' | ')' | '[' | ']' | '"' | '\''
        )
    })
    .to_string()
}

fn normalize_topic_label(s: &str) -> String {
    trim_word_punctuation(
        &s.trim()
            .trim_end_matches(':')
            .trim_end_matches(|c: char| c == '.' || c == ','),
    )
}

fn strip_leading_topic_stopwords(s: &str) -> String {
    let mut words: Vec<&str> = s.split_whitespace().collect();
    while let Some(first) = words.first() {
        if topic_leading_stopword(first) {
            words.remove(0);
        } else {
            break;
        }
    }
    words.join(" ")
}

fn topic_leading_stopword(w: &str) -> bool {
    matches!(
        w.to_lowercase().as_str(),
        "the" | "a" | "an" | "when" | "if" | "this" | "that" | "they" | "you" | "it" | "as"
            | "in" | "on" | "for" | "to" | "and" | "or" | "but" | "so" | "because" | "while"
            | "with" | "without" | "by" | "from" | "at" | "into" | "through"
    )
}

fn is_topic_verb_word(w: &str) -> bool {
    matches!(
        w.to_lowercase().as_str(),
        "is" | "are" | "was" | "were" | "be" | "been" | "being" | "am" | "work" | "works"
            | "working" | "worked" | "should" | "can" | "could" | "will" | "would" | "have"
            | "has" | "had" | "do" | "does" | "did" | "use" | "uses" | "using" | "used"
            | "allow" | "allows" | "provide" | "provides" | "make" | "makes" | "help"
            | "helps" | "need" | "needs" | "include" | "includes" | "mean" | "means"
            | "similar" | "often" | "also" | "just" | "very"
    )
}

fn looks_like_sentence_fragment(s: &str) -> bool {
    let lower = s.to_lowercase();
    if s.chars().next().is_some_and(|c| c.is_ascii_lowercase()) {
        return true;
    }
    const PHRASE_VERBS: &[&str] = &[
        " is ", " are ", " was ", " were ", " work ", " works ", " working ", " should ",
        " can ", " will ", " have ", " has ", " do ", " does ", " use ", " uses ", " allow ",
        " provides ", " when ", " if ", " that ", " this ", " they ", " you ", " in a ",
        " in the ", " of the ", " to the ", " for the ", " similar way", " such as",
    ];
    if PHRASE_VERBS.iter().any(|p| lower.contains(p)) {
        return true;
    }
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() > 6 {
        return true;
    }
    if words
        .last()
        .is_some_and(|w| matches!(w.to_lowercase().as_str(), "in" | "to" | "for" | "and" | "or" | "a" | "an" | "the"))
    {
        return true;
    }
    words.iter().any(|w| is_topic_verb_word(w))
}

fn is_usable_topic(s: &str) -> bool {
    let word_count = s.split_whitespace().count();
    (2..=6).contains(&word_count)
        && s.len() >= 4
        && s.len() <= 48
        && !is_spurious_anchor_value(s)
        && !looks_like_sentence_fragment(s)
}

/// Split oversized claim-instruction steps so the validation loop can accumulate one chunk per attempt.
/// Identical strategies (e.g. the same anchor chunk from two signals) are dropped — order preserved.
pub fn expand_strategies_for_attempts(strategies: Vec<MutationStrategy>) -> Vec<MutationStrategy> {
    let mut out = Vec::new();
    for s in strategies {
        match s {
            MutationStrategy::AddClaimInstruction { required_values }
                if required_values.len() > CLAIM_ANCHOR_CHUNK =>
            {
                for chunk in required_values.chunks(CLAIM_ANCHOR_CHUNK) {
                    push_unique(
                        &mut out,
                        MutationStrategy::AddClaimInstruction {
                            required_values: chunk.to_vec(),
                        },
                    );
                }
            }
            other => push_unique(&mut out, other),
        }
    }
    out
}

fn push_unique(out: &mut Vec<MutationStrategy>, strategy: MutationStrategy) {
    if !out.iter().any(|s| same_strategy(s, &strategy)) {
        out.push(strategy);
    }
}

fn same_strategy(a: &MutationStrategy, b: &MutationStrategy) -> bool {
    match (a, b) {
        (MutationStrategy::AddDirectnessInstruction, MutationStrategy::AddDirectnessInstruction)
        | (MutationStrategy::AddPrecisionInstruction, MutationStrategy::AddPrecisionInstruction)
        | (MutationStrategy::AddFormalityInstruction, MutationStrategy::AddFormalityInstruction)
        | (MutationStrategy::AddConfidenceInstruction, MutationStrategy::AddConfidenceInstruction)
        | (MutationStrategy::SoftenPhrasing, MutationStrategy::SoftenPhrasing)
        | (MutationStrategy::AddEducationalContext, MutationStrategy::AddEducationalContext) => true,
        (
            MutationStrategy::AddLengthConstraint { max_words: m1 },
            MutationStrategy::AddLengthConstraint { max_words: m2 },
        ) => m1 == m2,
        (
            MutationStrategy::AddDetailInstruction { min_words: m1 },
            MutationStrategy::AddDetailInstruction { min_words: m2 },
        ) => m1 == m2,
        (
            MutationStrategy::AddClaimInstruction {
                required_values: v1,
            },
            MutationStrategy::AddClaimInstruction {
                required_values: v2,
            },
        ) => v1 == v2,
        (
            MutationStrategy::AddTopicCoverageInstruction { topics: t1 },
            MutationStrategy::AddTopicCoverageInstruction { topics: t2 },
        ) => t1 == t2,
        (
            MutationStrategy::ReinforceInstruction {
                instruction_text: t1,
            },
            MutationStrategy::ReinforceInstruction {
                instruction_text: t2,
            },
        ) => t1 == t2,
        (MutationStrategy::Custom { hint: h1 }, MutationStrategy::Custom { hint: h2 }) => h1 == h2,
        _ => false,
    }
}

/// Full reconcile pipeline: analyse delta, rank signals, validate cumulative strategies against the target model.
pub async fn run_reconcile(
    engine: &ComparisonEngine,
    probe: Probe,
    v1_response: ModelResponse,
    v2_response: ModelResponse,
    v1_model: ModelInfo,
    v2_model: ModelInfo,
    target_adapter: &dyn ModelAdapter,
    max_strategies: usize,
) -> anyhow::Result<ReconcileResult> {
    let pair = ResponsePair {
        probe: probe.clone(),
        v1: v1_response.clone(),
        v2: v2_response.clone(),
        v1_runs: Vec::new(),
        v2_runs: Vec::new(),
    };
    let delta = engine.compare_one(pair)?;
    let baseline_risk = delta.overall_risk.clone();
    let signals = rank_signals(&delta);
    let strategies = signals_to_strategies(&signals, &delta);
    let max_attempts = if max_strategies == 0 {
        DEFAULT_MAX_STRATEGIES
    } else {
        max_strategies
    };
    let attempts_limit = max_attempts.min(strategies.len());

    let mut attempts = Vec::new();
    let mut certified = false;
    let mut certified_prompt = None;
    let mut strategies_applied = Vec::new();
    let mut validation_risk = None;
    let mut validation_response = None;
    let mut notes = Vec::new();
    #[allow(unused_assignments)]
    let mut last_mutated = probe.prompt.clone();

    if strategies.is_empty() {
        notes.push("No actionable drift signals; prompt unchanged.".into());
    } else if attempts_limit == 0 {
        notes.push("No strategies to apply; skipped validation.".into());
    } else {
        for i in 0..attempts_limit {
            let trial: Vec<_> = strategies[..=i].to_vec();
            last_mutated = apply_mutations(&probe, &trial);
            let mut trial_probe = probe.clone();
            trial_probe.prompt = last_mutated.clone();
            let v2_trial = target_adapter.complete(&trial_probe).await?;
            let trial_pair = ResponsePair {
                probe: probe.clone(),
                v1: v1_response.clone(),
                v2: v2_trial.clone(),
                v1_runs: Vec::new(),
                v2_runs: Vec::new(),
            };
            let trial_delta = engine.compare_one(trial_pair)?;
            let risk_after = trial_delta.overall_risk.clone();
            let improved = risk_improved(&baseline_risk, &risk_after);
            attempts.push(ReconcileAttempt {
                attempt_number: i + 1,
                mutated_prompt: last_mutated.clone(),
                strategies: trial.clone(),
                v2_response: v2_trial.content.clone(),
                risk_after: risk_after.clone(),
                improved,
            });
            if improved {
                certified = true;
                certified_prompt = Some(last_mutated.clone());
                strategies_applied = trial;
                validation_risk = Some(risk_after);
                validation_response = Some(v2_trial.content);
                notes.push("Cumulative strategies reduced overall risk.".into());
                break;
            }
            // On failure, continue accumulating strategies through remaining attempts.
        }
        if !certified {
            notes.push(format!(
                "No improvement after {} cumulative attempt(s); manual review suggested.",
                attempts.len()
            ));
        }
    }

    Ok(ReconcileResult {
        run_id: Uuid::new_v4(),
        generated_at: Utc::now(),
        prompt: probe.prompt.clone(),
        system_prompt: probe.system_prompt.clone(),
        v1_model,
        v2_model,
        v1_response: v1_response.content,
        v2_response: v2_response.content,
        delta,
        signals,
        attempts,
        certified,
        certified_prompt,
        strategies_applied,
        validation_risk,
        validation_response,
        requires_manual_review: !certified && !strategies.is_empty(),
        notes,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use async_trait::async_trait;

    use super::*;
    use crate::comparison::{ComparisonEngine, RiskThresholds};
    use crate::reconcile::SignalDetail;
    use crate::types::ProbeCategory;

    struct ScriptedAdapter {
        model_id: String,
        by_prompt_contains: HashMap<String, String>,
        default: String,
    }

    #[async_trait]
    impl ModelAdapter for ScriptedAdapter {
        async fn complete(&self, probe: &Probe) -> anyhow::Result<ModelResponse> {
            let content = self
                .by_prompt_contains
                .iter()
                .find(|(k, _)| probe.prompt.contains(k.as_str()))
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| self.default.clone());
            Ok(synthetic_model_response(
                probe.id,
                "target",
                &self.model_id,
                &content,
            ))
        }

        fn model_id(&self) -> &str {
            &self.model_id
        }

        fn adapter_name(&self) -> &str {
            "scripted"
        }

        fn endpoint(&self) -> &str {
            "memory://"
        }
    }

    #[test]
    fn rank_signals_orders_anchor_drift_before_morphology() {
        let mut delta = minimal_probe_result();
        delta.dimensions.claim.drifted_claims.push(crate::types::ClaimDrift {
            v1_claim: crate::types::Claim {
                text: "Rate is 4.5%".into(),
                information_density: 0.5,
                anchors: vec![crate::types::ClaimAnchor {
                    anchor_type: crate::types::AnchorType::NumericValue,
                    value: "4.5%".into(),
                }],
            },
            v2_claim: crate::types::Claim {
                text: "Rate is 5%".into(),
                information_density: 0.5,
                anchors: vec![crate::types::ClaimAnchor {
                    anchor_type: crate::types::AnchorType::NumericValue,
                    value: "5%".into(),
                }],
            },
            similarity: 0.7,
            drifted_anchors: vec![crate::types::AnchorDrift {
                anchor_type: crate::types::AnchorType::NumericValue,
                v1_value: "4.5%".into(),
                v2_value: "5%".into(),
            }],
        });
        delta.dimensions.morphology.delta.token_delta_pct = 40.0;
        let signals = rank_signals(&delta);
        assert!(!signals.is_empty());
        assert!(matches!(
            signals[0].detail,
            SignalDetail::AnchorDrift { .. }
        ));
        assert!(signals[0].magnitude >= signals.last().unwrap().magnitude);
    }

    #[test]
    fn signals_to_strategies_uses_magnitude_order_not_fixed_mutate_order() {
        let signals = vec![
            ReconcileSignal {
                dimension: ReconcileDimension::Morphology,
                magnitude: 0.6,
                direction: DriftDirection::Regression,
                detail: SignalDetail::MorphologyDelta {
                    token_delta_pct: -60.0,
                    response_type_changed: false,
                    v2_shorter: true,
                },
            },
            ReconcileSignal {
                dimension: ReconcileDimension::Tone,
                magnitude: 0.2,
                direction: DriftDirection::Regression,
                detail: SignalDetail::ToneDelta {
                    formality_delta: -0.1,
                    assertiveness_delta: 0.0,
                    v2_less_formal: true,
                    v2_over_hedged: false,
                },
            },
        ];
        let delta = minimal_probe_result();
        let strategies = signals_to_strategies(&signals, &delta);
        assert!(strategies.len() >= 2);
        assert!(matches!(
            strategies[0],
            MutationStrategy::AddDetailInstruction { .. }
        ));
        assert!(matches!(
            strategies[1],
            MutationStrategy::AddFormalityInstruction
        ));
    }

    #[test]
    fn long_form_high_drops_use_topic_coverage_not_detail_when_v2_longer() {
        use crate::types::*;
        let mut delta = minimal_probe_result();
        delta.probe.category = ProbeCategory::Semantic;
        delta.dimensions.morphology.v1.word_count = 350;
        delta.dimensions.morphology.v2.word_count = 400;
        delta.dimensions.claim.dropped_claims = (0..12)
            .map(|i| Claim {
                text: format!("**Section {i}**: Detail about topic {i}."),
                information_density: 0.3,
                anchors: vec![ClaimAnchor {
                    anchor_type: AnchorType::ProperNoun,
                    value: format!("Topic{i}"),
                }],
            })
            .collect();

        let signals = rank_signals(&delta);
        let dropped = signals
            .iter()
            .find(|s| matches!(s.detail, SignalDetail::DroppedClaims { .. }));
        assert!(dropped.is_some());

        let strategies = signals_to_strategies(&signals, &delta);
        assert!(
            strategies
                .iter()
                .any(|s| matches!(s, MutationStrategy::AddTopicCoverageInstruction { .. })),
            "expected topic coverage strategy"
        );
        assert!(
            !strategies
                .iter()
                .any(|s| matches!(s, MutationStrategy::AddDetailInstruction { .. })),
            "detail instruction should not apply when target response is not shorter"
        );
    }

    #[test]
    fn expand_strategies_splits_large_claim_instruction() {
        let raw = vec![MutationStrategy::AddClaimInstruction {
            required_values: (0..10).map(|i| format!("v{i}")).collect(),
        }];
        let expanded = expand_strategies_for_attempts(raw);
        assert_eq!(expanded.len(), 3);
        assert!(matches!(
            expanded[0],
            MutationStrategy::AddClaimInstruction { .. }
        ));
    }

    #[test]
    fn expand_strategies_dedupes_identical_claim_chunks() {
        let chunk = vec![
            "Programming".into(),
            "Interfaces".into(),
            "APIs".into(),
            "URLs".into(),
        ];
        let raw = vec![
            MutationStrategy::AddTopicCoverageInstruction {
                topics: vec!["API endpoints".into()],
            },
            MutationStrategy::AddClaimInstruction {
                required_values: chunk.clone(),
            },
            MutationStrategy::AddClaimInstruction {
                required_values: chunk,
            },
        ];
        let expanded = expand_strategies_for_attempts(raw);
        let claim_steps: Vec<_> = expanded
            .iter()
            .filter(|s| matches!(s, MutationStrategy::AddClaimInstruction { .. }))
            .collect();
        assert_eq!(claim_steps.len(), 1, "duplicate claim chunks must be removed: {expanded:?}");
    }

    #[test]
    fn extract_coverage_topics_from_bold_headings() {
        let claims = vec![Claim {
            text: "**Team Experience**: Consider skills.".into(),
            information_density: 0.4,
            anchors: vec![],
        }];
        let topics = extract_coverage_topics(&claims, &claims);
        assert!(topics.iter().any(|t| t.contains("Team Experience")));
    }

    #[test]
    fn extract_coverage_topics_prefers_inline_bold_over_sentence_text() {
        let claims = vec![Claim {
            text: "APIs work in a similar way to other services. **API endpoints** accept requests. **Request methods** include GET and POST.".into(),
            information_density: 0.4,
            anchors: vec![],
        }];
        let topics = extract_coverage_topics(&claims, &claims);
        assert!(topics.iter().any(|t| t == "API endpoints"));
        assert!(topics.iter().any(|t| t == "Request methods"));
        assert!(
            !topics.iter().any(|t| t.contains("work in a similar")),
            "sentence openers must not become topics: {topics:?}"
        );
    }

    #[test]
    fn extract_coverage_topics_skips_noun_pass_when_three_bold_headings() {
        let claims = vec![Claim {
            text: "**API endpoints** accept traffic. **Request methods** vary. **Response codes** matter. REST APIs work in a similar way to other distributed systems.".into(),
            information_density: 0.4,
            anchors: vec![],
        }];
        let topics = extract_coverage_topics(&claims, &claims);
        assert_eq!(topics.len(), 3);
        assert!(topics.iter().any(|t| t == "API endpoints"));
        assert!(topics.iter().any(|t| t == "Request methods"));
        assert!(topics.iter().any(|t| t == "Response codes"));
        assert!(!topics.iter().any(|t| t.contains("REST APIs")));
    }

    #[test]
    fn extract_coverage_topics_caps_at_six() {
        let claims: Vec<Claim> = (0..10)
            .map(|i| Claim {
                text: format!("**Topic {i}**: Some detail here."),
                information_density: 0.4,
                anchors: vec![],
            })
            .collect();
        let topics = extract_coverage_topics(&claims, &claims);
        assert_eq!(topics.len(), 6);
    }

    #[test]
    fn extract_coverage_topics_rejects_sentence_fragment_fallback() {
        let claims = vec![Claim {
            text: "When evaluating options, teams should weigh requirements carefully.".into(),
            information_density: 0.4,
            anchors: vec![],
        }];
        let topics = extract_coverage_topics(&claims, &claims);
        assert!(
            topics.is_empty() || !topics.iter().any(|t| t.contains("should weigh")),
            "expected no sentence-fragment topics, got {topics:?}"
        );
    }

    #[tokio::test]
    async fn validation_loop_runs_multiple_attempts_on_failure() {
        let probe = build_reconcile_probe(
            "What are the most important things to consider when choosing a programming language for a new project?"
                .into(),
            None,
        );
        let mut v1_parts = Vec::new();
        for i in 0..14 {
            v1_parts.push(format!(
                "**Factor {i}**: When evaluating options, teams should weigh requirement {i}, trade-offs around Topic{i}, and how the choice affects delivery timelines."
            ));
        }
        let v1_text = v1_parts.join("\n\n");
        let v2_text =
            "Pick a language your team knows. Performance and ecosystem matter too.".to_string();
        let v1 = synthetic_model_response(probe.id, "baseline", "b", &v1_text);
        let v2 = synthetic_model_response(probe.id, "target", "t", &v2_text);

        let engine = ComparisonEngine::new(true, 0.85, RiskThresholds::default());
        let adapter = Arc::new(ScriptedAdapter {
            model_id: "t".into(),
            by_prompt_contains: HashMap::new(),
            default: v2_text.to_string(),
        });

        let result = run_reconcile(
            &engine,
            probe,
            v1,
            v2,
            ModelInfo {
                label: "baseline".into(),
                model_id: "b".into(),
                adapter: "inline".into(),
                endpoint: String::new(),
            },
            ModelInfo {
                label: "target".into(),
                model_id: "t".into(),
                adapter: "scripted".into(),
                endpoint: "memory://".into(),
            },
            adapter.as_ref(),
            5,
        )
        .await
        .expect("reconcile");

        assert!(
            result.attempts.len() > 1,
            "expected more than one attempt when risk does not improve, got {}",
            result.attempts.len()
        );
        assert!(
            result.attempts.iter().all(|a| !a.improved),
            "scripted adapter should not improve"
        );
    }

    #[tokio::test]
    async fn reconcile_pipeline_mode3_without_network() {
        let prompt = "What was the US unemployment rate in 2008?".to_string();
        let probe = build_reconcile_probe(prompt.clone(), None);
        assert!(matches!(probe.category, ProbeCategory::Factual));

        let v1_text = "The US unemployment rate peaked at 4.5% in early 2008 before rising later that year.";
        let v2_text = "Unemployment fluctuated during the financial crisis period.";

        let v1 = synthetic_model_response(probe.id, "baseline", "baseline-model", v1_text);
        let v2 = synthetic_model_response(probe.id, "target", "target-model", v2_text);

        let engine = ComparisonEngine::new(true, 0.85, RiskThresholds::default());
        let mut responses = HashMap::new();
        responses.insert(
            "specific values:".to_string(),
            v1_text.to_string(),
        );
        let adapter = Arc::new(ScriptedAdapter {
            model_id: "target-model".into(),
            by_prompt_contains: responses,
            default: v2_text.to_string(),
        });

        let result = run_reconcile(
            &engine,
            probe,
            v1,
            v2,
            ModelInfo {
                label: "baseline".into(),
                model_id: "baseline-model".into(),
                adapter: "inline".into(),
                endpoint: String::new(),
            },
            ModelInfo {
                label: "target".into(),
                model_id: "target-model".into(),
                adapter: "scripted".into(),
                endpoint: "memory://".into(),
            },
            adapter.as_ref(),
            5,
        )
        .await
        .expect("reconcile run");

        assert!(!result.signals.is_empty());
        assert!(!result.attempts.is_empty());
        assert!(
            result.certified || result.requires_manual_review,
            "expected certification or manual review flag"
        );
        if result.certified {
            assert!(result.certified_prompt.is_some());
            assert!(result.validation_risk.is_some());
        }
    }

    fn minimal_probe_result() -> ProbeResult {
        use crate::types::*;
        ProbeResult {
            probe: build_reconcile_probe("test".into(), None),
            v1_content: String::new(),
            v2_content: String::new(),
            overall_risk: RiskLevel::Amber,
            overall_direction: DriftDirection::Neutral,
            drift_category: DriftCategory::NoSignificantDrift,
            dimensions: ProbeDimensions {
                morphology: MorphologyDiff {
                    risk: RiskLevel::Green,
                    direction: DriftDirection::Neutral,
                    v1: MorphologyMetrics {
                        token_count: 10,
                        word_count: 10,
                        sentence_count: 1,
                        paragraph_count: 1,
                        has_lists: false,
                        has_headers: false,
                        has_code_blocks: false,
                        has_caveats: false,
                        response_type: ResponseType::SingleLine,
                    },
                    v2: MorphologyMetrics {
                        token_count: 4,
                        word_count: 4,
                        sentence_count: 1,
                        paragraph_count: 1,
                        has_lists: false,
                        has_headers: false,
                        has_code_blocks: false,
                        has_caveats: false,
                        response_type: ResponseType::SingleLine,
                    },
                    delta: MorphologyDelta {
                        token_delta: -6,
                        token_delta_pct: -60.0,
                        response_type_changed: false,
                        structure_changed: false,
                    },
                },
                tone: ToneDiff {
                    risk: RiskLevel::Green,
                    direction: DriftDirection::Neutral,
                    v1: ToneMetrics {
                        formality_score: 0.6,
                        assertiveness_score: 0.5,
                        hedge_word_count: 0,
                        contraction_count: 0,
                        average_sentence_length: 10.0,
                        passive_voice_ratio: 0.0,
                    },
                    v2: ToneMetrics {
                        formality_score: 0.5,
                        assertiveness_score: 0.5,
                        hedge_word_count: 0,
                        contraction_count: 0,
                        average_sentence_length: 10.0,
                        passive_voice_ratio: 0.0,
                    },
                    delta: ToneDelta {
                        formality_delta: -0.1,
                        assertiveness_delta: 0.0,
                        hedge_word_delta: 0,
                        significant_shift: false,
                    },
                },
                factual: None,
                schema: None,
                instruction: None,
                refusal: RefusalDiff {
                    risk: RiskLevel::Green,
                    direction: DriftDirection::Neutral,
                    v1_refused: false,
                    v2_refused: false,
                    new_refusal: false,
                    refusal_lifted: false,
                },
                semantic: SemanticDiff {
                    risk: RiskLevel::Green,
                    direction: DriftDirection::Neutral,
                    cosine_similarity: Some(0.95),
                    semantic_scoring_disabled: false,
                    disabled_reason: None,
                    flagged_for_review: false,
                    similarity_threshold: 0.85,
                },
                claim: ClaimDiff {
                    risk: RiskLevel::Green,
                    direction: DriftDirection::Neutral,
                    v1_claims: vec![],
                    v2_claims: vec![],
                    matched_pairs: vec![],
                    dropped_claims: vec![],
                    new_claims: vec![],
                    drifted_claims: vec![],
                    preservation_score: 1.0,
                    preservation_threshold: 0.5,
                },
                latency: LatencyDiff {
                    risk: RiskLevel::Green,
                    direction: DriftDirection::Neutral,
                    v1_latency_ms: 0,
                    v2_latency_ms: 0,
                    delta_ms: 0,
                    delta_pct: 0.0,
                },
                consistency: None,
                custom_assertions: None,
            },
            notes: vec![],
        }
    }
}
