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
