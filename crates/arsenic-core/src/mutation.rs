//! Rule-based prompt mutations (v2). Validation runs in the CLI against v2.

use std::collections::HashSet;

use crate::claim::is_spurious_anchor_value;
use crate::types::{
    ClaimDiff, MorphologyDiff, MutationStrategy, Probe, RefusalDiff, ToneDiff,
};

/// Build ordered mutation strategies from drift characterisation.
pub fn propose_strategies(
    probe: &Probe,
    morphology: &MorphologyDiff,
    tone: &ToneDiff,
    refusal: &RefusalDiff,
    claim: &ClaimDiff,
    instruction_regressions: &[String],
) -> Vec<MutationStrategy> {
    let mut out = Vec::new();

    for line in instruction_regressions {
        out.push(MutationStrategy::ReinforceInstruction {
            instruction_text: line.clone(),
        });
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut dropped_values: Vec<String> = Vec::new();
    for v in claim
        .dropped_claims
        .iter()
        .flat_map(|c| c.anchors.iter().map(|a| a.value.clone()))
    {
        if is_spurious_anchor_value(&v) {
            continue;
        }
        if seen.insert(v.clone()) {
            dropped_values.push(v);
        }
    }
    if !dropped_values.is_empty() {
        out.push(MutationStrategy::AddClaimInstruction {
            required_values: dropped_values,
        });
    }
    if !claim.drifted_claims.is_empty() {
        out.push(MutationStrategy::AddPrecisionInstruction);
    }

    if morphology.delta.token_delta > 0 && morphology.v2.word_count > morphology.v1.word_count {
        if let Some(ev) = &probe.expected_verbosity {
            use crate::types::ExpectedVerbosity;
            if matches!(ev, ExpectedVerbosity::Concise) {
                let max_words = ((morphology.v1.word_count as f64) * 1.15).ceil() as usize;
                out.push(MutationStrategy::AddLengthConstraint {
                    max_words: max_words.max(8),
                });
            }
        }
    }
    if morphology.delta.response_type_changed {
        out.push(MutationStrategy::AddDirectnessInstruction);
    }

    if tone.delta.formality_delta < -0.08 {
        if let Some(et) = &probe.expected_tone {
            use crate::types::ExpectedTonePreference;
            if matches!(et, ExpectedTonePreference::Formal) {
                out.push(MutationStrategy::AddFormalityInstruction);
            }
        }
    }
    if tone.delta.hedge_word_delta > 3 {
        out.push(MutationStrategy::AddConfidenceInstruction);
    }

    if refusal.new_refusal {
        out.push(MutationStrategy::SoftenPhrasing);
        out.push(MutationStrategy::AddEducationalContext);
        if let Some(h) = &probe.mutation_hint {
            out.push(MutationStrategy::Custom { hint: h.clone() });
        }
    }

    out
}

/// Apply strategies as prompt suffixes (default per v2 spec).
pub fn apply_mutations(probe: &Probe, strategies: &[MutationStrategy]) -> String {
    let mut suffixes: Vec<String> = Vec::new();
    for s in strategies {
        match s {
            MutationStrategy::AddLengthConstraint { max_words } => {
                suffixes.push(format!(" Respond concisely in {max_words} words or fewer."));
            }
            MutationStrategy::AddDetailInstruction { min_words } => {
                suffixes.push(format!(
                    " Provide a thorough and detailed answer of at least {min_words} words."
                ));
            }
            MutationStrategy::AddDirectnessInstruction => {
                suffixes.push(" Provide a direct answer without elaboration.".into());
            }
            MutationStrategy::AddFormalityInstruction => {
                suffixes.push(" Use formal, professional language.".into());
            }
            MutationStrategy::AddConfidenceInstruction => {
                suffixes.push(" Give a direct, confident answer.".into());
            }
            MutationStrategy::SoftenPhrasing => {
                suffixes.push(" Could you help with the following request?".into());
            }
            MutationStrategy::AddEducationalContext => {
                suffixes.push(" This is for educational purposes.".into());
            }
            MutationStrategy::AddClaimInstruction { required_values } => {
                let joined = required_values.join(", ");
                suffixes.push(format!(
                    " Your answer must address or include these specific values: {joined}."
                ));
            }
            MutationStrategy::AddTopicCoverageInstruction { topics } => {
                let joined = topics.join(", ");
                suffixes.push(format!(
                    " Ensure your response covers the following topics: {joined}."
                ));
            }
            MutationStrategy::AddPrecisionInstruction => {
                suffixes.push(" Be precise with specific values and dates.".into());
            }
            MutationStrategy::ReinforceInstruction { instruction_text } => {
                suffixes.push(format!(" Important: {instruction_text}"));
            }
            MutationStrategy::Custom { hint } => {
                suffixes.push(format!(" {hint}"));
            }
        }
    }
    let joined = suffixes.join("");
    format!("{}{}", probe.prompt, joined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AnchorType, Claim, ClaimAnchor, ClaimDrift, DriftDirection, ExpectedTonePreference,
        ExpectedVerbosity, MorphologyDelta, MorphologyMetrics, ProbeCategory, ProbeSource,
        ResponseType, RiskLevel, ToneDelta, ToneMetrics,
    };
    use uuid::Uuid;

    fn metrics(words: usize) -> MorphologyMetrics {
        MorphologyMetrics {
            token_count: words,
            word_count: words,
            sentence_count: 1,
            paragraph_count: 1,
            has_lists: false,
            has_headers: false,
            has_code_blocks: false,
            has_caveats: false,
            response_type: ResponseType::ShortParagraph,
        }
    }

    fn morph(token_delta: i64, w1: usize, w2: usize, type_changed: bool) -> MorphologyDiff {
        MorphologyDiff {
            risk: RiskLevel::Green,
            direction: DriftDirection::Neutral,
            v1: metrics(w1),
            v2: metrics(w2),
            delta: MorphologyDelta {
                token_delta,
                token_delta_pct: 0.0,
                response_type_changed: type_changed,
                structure_changed: false,
            },
        }
    }

    fn tone_metrics() -> ToneMetrics {
        ToneMetrics {
            formality_score: 0.5,
            assertiveness_score: 0.5,
            hedge_word_count: 0,
            contraction_count: 0,
            average_sentence_length: 0.0,
            passive_voice_ratio: 0.0,
        }
    }

    fn tone(formality_delta: f64, hedge_delta: i64) -> ToneDiff {
        ToneDiff {
            risk: RiskLevel::Green,
            direction: DriftDirection::Neutral,
            v1: tone_metrics(),
            v2: tone_metrics(),
            delta: ToneDelta {
                formality_delta,
                assertiveness_delta: 0.0,
                hedge_word_delta: hedge_delta,
                significant_shift: false,
            },
        }
    }

    fn refusal(new_refusal: bool) -> RefusalDiff {
        RefusalDiff {
            risk: RiskLevel::Green,
            direction: DriftDirection::Neutral,
            v1_refused: false,
            v2_refused: new_refusal,
            new_refusal,
            refusal_lifted: false,
        }
    }

    fn empty_claim() -> ClaimDiff {
        ClaimDiff {
            risk: RiskLevel::Green,
            direction: DriftDirection::Neutral,
            v1_claims: vec![],
            v2_claims: vec![],
            matched_pairs: vec![],
            dropped_claims: vec![],
            new_claims: vec![],
            drifted_claims: vec![],
            preservation_score: 1.0,
            preservation_threshold: 0.7,
        }
    }

    fn dropped_claim(values: &[&str]) -> Claim {
        Claim {
            text: "claim".into(),
            information_density: 1.0,
            anchors: values
                .iter()
                .map(|v| ClaimAnchor {
                    value: (*v).into(),
                    anchor_type: AnchorType::ProperNoun,
                })
                .collect(),
        }
    }

    fn mk_probe() -> Probe {
        Probe {
            id: Uuid::new_v4(),
            name: "t".into(),
            category: ProbeCategory::Factual,
            prompt: "Say hi.".into(),
            system_prompt: None,
            known_answer: None,
            expected_schema: None,
            instructions: vec![],
            tags: vec![],
            source: ProbeSource::Standard,
            expected_verbosity: None,
            expected_tone: None,
            refusal_expectation: None,
            mutation_hint: None,
            custom_assertions: vec![],
        }
    }

    #[test]
    fn no_drift_yields_no_strategies() {
        let s = propose_strategies(
            &mk_probe(),
            &morph(0, 10, 10, false),
            &tone(0.0, 0),
            &refusal(false),
            &empty_claim(),
            &[],
        );
        assert!(s.is_empty(), "got: {s:?}");
    }

    #[test]
    fn instruction_regressions_become_reinforce_strategies() {
        let s = propose_strategies(
            &mk_probe(),
            &morph(0, 10, 10, false),
            &tone(0.0, 0),
            &refusal(false),
            &empty_claim(),
            &["Always cite sources.".into(), "Use bullet points.".into()],
        );
        let count = s
            .iter()
            .filter(|x| matches!(x, MutationStrategy::ReinforceInstruction { .. }))
            .count();
        assert_eq!(count, 2);
    }

    #[test]
    fn dropped_claim_anchors_become_add_claim_instruction() {
        let mut claim = empty_claim();
        claim.dropped_claims.push(dropped_claim(&["Paris", "1789"]));
        let s = propose_strategies(
            &mk_probe(),
            &morph(0, 10, 10, false),
            &tone(0.0, 0),
            &refusal(false),
            &claim,
            &[],
        );
        let found = s.iter().find_map(|x| match x {
            MutationStrategy::AddClaimInstruction { required_values } => Some(required_values),
            _ => None,
        });
        let values = found.expect("AddClaimInstruction expected");
        assert!(values.contains(&"Paris".to_string()));
        assert!(values.contains(&"1789".to_string()));
    }

    #[test]
    fn drifted_claims_request_precision() {
        let mut claim = empty_claim();
        let v1c = Claim {
            text: "Paris is the capital".into(),
            information_density: 1.0,
            anchors: vec![],
        };
        let v2c = Claim {
            text: "Lyon is the capital".into(),
            information_density: 1.0,
            anchors: vec![],
        };
        claim.drifted_claims.push(ClaimDrift {
            v1_claim: v1c,
            v2_claim: v2c,
            similarity: 0.4,
            drifted_anchors: vec![],
        });
        let s = propose_strategies(
            &mk_probe(),
            &morph(0, 10, 10, false),
            &tone(0.0, 0),
            &refusal(false),
            &claim,
            &[],
        );
        assert!(s.iter().any(|x| matches!(x, MutationStrategy::AddPrecisionInstruction)));
    }

    #[test]
    fn concise_expected_plus_inflation_triggers_length_constraint() {
        let mut probe = mk_probe();
        probe.expected_verbosity = Some(ExpectedVerbosity::Concise);
        let s = propose_strategies(
            &probe,
            &morph(50, 20, 60, false),
            &tone(0.0, 0),
            &refusal(false),
            &empty_claim(),
            &[],
        );
        let max_words = s.iter().find_map(|x| match x {
            MutationStrategy::AddLengthConstraint { max_words } => Some(*max_words),
            _ => None,
        });
        assert!(
            max_words.is_some_and(|m| m >= 8),
            "expected AddLengthConstraint with sensible cap, got {s:?}"
        );
    }

    #[test]
    fn response_type_changed_requests_directness() {
        let s = propose_strategies(
            &mk_probe(),
            &morph(0, 10, 10, true),
            &tone(0.0, 0),
            &refusal(false),
            &empty_claim(),
            &[],
        );
        assert!(s
            .iter()
            .any(|x| matches!(x, MutationStrategy::AddDirectnessInstruction)));
    }

    #[test]
    fn formality_drop_with_formal_expectation_adds_formality_instruction() {
        let mut probe = mk_probe();
        probe.expected_tone = Some(ExpectedTonePreference::Formal);
        let s = propose_strategies(
            &probe,
            &morph(0, 10, 10, false),
            &tone(-0.15, 0),
            &refusal(false),
            &empty_claim(),
            &[],
        );
        assert!(s
            .iter()
            .any(|x| matches!(x, MutationStrategy::AddFormalityInstruction)));

        // Without the expectation, formality drop alone should NOT trigger it.
        let s2 = propose_strategies(
            &mk_probe(),
            &morph(0, 10, 10, false),
            &tone(-0.15, 0),
            &refusal(false),
            &empty_claim(),
            &[],
        );
        assert!(!s2
            .iter()
            .any(|x| matches!(x, MutationStrategy::AddFormalityInstruction)));
    }

    #[test]
    fn hedge_spike_triggers_confidence_instruction() {
        let s = propose_strategies(
            &mk_probe(),
            &morph(0, 10, 10, false),
            &tone(0.0, 5),
            &refusal(false),
            &empty_claim(),
            &[],
        );
        assert!(s
            .iter()
            .any(|x| matches!(x, MutationStrategy::AddConfidenceInstruction)));
    }

    #[test]
    fn new_refusal_adds_softening_education_and_custom_hint() {
        let mut probe = mk_probe();
        probe.mutation_hint = Some("Rephrase as a research question.".into());
        let s = propose_strategies(
            &probe,
            &morph(0, 10, 10, false),
            &tone(0.0, 0),
            &refusal(true),
            &empty_claim(),
            &[],
        );
        assert!(s.iter().any(|x| matches!(x, MutationStrategy::SoftenPhrasing)));
        assert!(s
            .iter()
            .any(|x| matches!(x, MutationStrategy::AddEducationalContext)));
        assert!(s
            .iter()
            .any(|x| matches!(x, MutationStrategy::Custom { hint } if hint.contains("Rephrase"))));
    }

    #[test]
    fn apply_mutations_appends_each_strategy_suffix() {
        let probe = {
            let mut p = mk_probe();
            p.prompt = "Q.".into();
            p
        };
        let strategies = vec![
            MutationStrategy::AddDirectnessInstruction,
            MutationStrategy::AddConfidenceInstruction,
            MutationStrategy::AddClaimInstruction {
                required_values: vec!["Paris".into(), "1789".into()],
            },
            MutationStrategy::Custom {
                hint: "Frame as Q&A.".into(),
            },
        ];
        let out = apply_mutations(&probe, &strategies);
        assert!(out.starts_with("Q."));
        assert!(out.contains("Provide a direct answer"));
        assert!(out.contains("direct, confident answer"));
        assert!(out.contains("Paris, 1789"));
        assert!(out.contains("Frame as Q&A."));
    }

    #[test]
    fn apply_mutations_with_no_strategies_returns_original_prompt() {
        let probe = mk_probe();
        let out = apply_mutations(&probe, &[]);
        assert_eq!(out, "Say hi.");
    }
}
