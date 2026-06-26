//! Rollout impact classification — separates blocking behavioural regressions from
//! presentation/format drift and observational telemetry.

use crate::code_equivalence::{code_formatting_only, formatting_blocks, is_code_probe};
use crate::types::*;

#[derive(Debug, Clone)]
pub struct ProbeImpactAssessment {
    pub drift_impact: DriftImpact,
    pub overall_risk: RiskLevel,
    pub drift_category: DriftCategory,
    pub drift_severity: DriftSeverity,
    pub blocking_reasons: Vec<String>,
}

/// Classify probe-level rollout impact from dimension findings and probe metadata.
pub fn assess_probe_impact(probe: &Probe, dims: &ProbeDimensions) -> ProbeImpactAssessment {
    let mut blocking_reasons = Vec::new();
    let code_fmt_only = code_formatting_only(dims);

    if let Some(f) = &dims.factual {
        if f.regression {
            blocking_reasons.push("factual regression".into());
        }
    }

    if let Some(s) = &dims.schema {
        if (!s.v2_valid_json || !s.v2_schema_valid)
            && (s.v1_valid_json && s.v1_schema_valid || !s.v2_valid_json)
        {
            blocking_reasons.push("schema or JSON regression".into());
        } else if matches!(s.risk, RiskLevel::Red) && !s.v2_schema_valid {
            blocking_reasons.push("schema validation failure".into());
        }
    }

    if let Some(i) = &dims.instruction {
        if !i.regressions.is_empty() {
            blocking_reasons.push(format!("instruction: {}", i.regressions.join("; ")));
        }
    }

    if let Some(c) = &dims.custom_assertions {
        if !c.regressions.is_empty() {
            blocking_reasons.push(format!("custom assertion: {}", c.regressions.join("; ")));
        }
    }

    match probe.refusal_expectation {
        Some(RefusalExpectation::ShouldAnswer) if dims.refusal.new_refusal => {
            blocking_reasons.push("unexpected refusal".into());
        }
        Some(RefusalExpectation::ShouldRefuse) if dims.refusal.refusal_lifted => {
            blocking_reasons.push("refusal lifted".into());
        }
        _ => {}
    }

    if dims.claim.has_material_anchor_drift && !(code_fmt_only && !formatting_blocks(probe)) {
        blocking_reasons.push("material claim anchor drift".into());
    }

    if !dims.claim.material_dropped_claims.is_empty()
        && material_drops_block(probe, dims)
        && !(code_fmt_only && !formatting_blocks(probe))
    {
        blocking_reasons.push("material claim loss".into());
    }

    if let Some(slo) = probe.latency_slo_ms {
        if dims.latency.v2_latency_ms > slo {
            blocking_reasons.push(format!("latency SLO breach ({slo} ms)"));
        }
    }

    if formatting_blocks(probe) {
        if matches!(dims.morphology.risk, RiskLevel::Red)
            && probe.presentation_drift == PresentationDriftPolicy::Blocking
        {
            blocking_reasons.push("task-critical structure/format change".into());
        } else if dims.morphology.v1.has_code_blocks != dims.morphology.v2.has_code_blocks {
            blocking_reasons.push("task-critical code formatting change".into());
        }
    }

    if is_code_probe(probe)
        && dims.morphology.v1.has_code_blocks
        && !dims.morphology.v2.has_code_blocks
        && !code_fmt_only
        && formatting_blocks(probe)
    {
        blocking_reasons.push("code block lost".into());
    }

    if is_code_probe(probe)
        && dims
            .code_equivalence
            .as_ref()
            .is_some_and(|c| c.applies && c.equivalence == CodeEquivalenceLevel::Different)
        && (dims.claim.has_material_anchor_drift
            || !dims.claim.drifted_claims.is_empty()
            || dims.claim.material_preservation_score < dims.claim.preservation_threshold
            || dims.claim.preservation_score < 1.0)
    {
        blocking_reasons.push("code semantics changed".into());
    }

    if !blocking_reasons.is_empty() {
        return ProbeImpactAssessment {
            drift_impact: DriftImpact::Blocking,
            overall_risk: RiskLevel::Red,
            drift_category: DriftCategory::CriticalRegression,
            drift_severity: DriftSeverity::Critical,
            blocking_reasons,
        };
    }

    if code_fmt_only && !formatting_blocks(probe) {
        let fence_only = dims.morphology.v1.has_code_blocks != dims.morphology.v2.has_code_blocks;
        if !fence_only {
            return ProbeImpactAssessment {
                drift_impact: DriftImpact::Informational,
                overall_risk: RiskLevel::Green,
                drift_category: DriftCategory::NoSignificantDrift,
                drift_severity: DriftSeverity::Informational,
                blocking_reasons: vec![],
            };
        }
        return ProbeImpactAssessment {
            drift_impact: DriftImpact::Presentation,
            overall_risk: if matches!(probe.presentation_drift, PresentationDriftPolicy::Review) {
                RiskLevel::Amber
            } else {
                RiskLevel::Green
            },
            drift_category: presentation_drift_category(dims),
            drift_severity: DriftSeverity::Medium,
            blocking_reasons: vec![],
        };
    }

    if has_review_signals(probe, dims) {
        return ProbeImpactAssessment {
            drift_impact: DriftImpact::Review,
            overall_risk: RiskLevel::Amber,
            drift_category: review_drift_category(dims),
            drift_severity: DriftSeverity::High,
            blocking_reasons: vec![],
        };
    }

    if has_presentation_drift(probe, dims) {
        return ProbeImpactAssessment {
            drift_impact: DriftImpact::Presentation,
            overall_risk: RiskLevel::Amber,
            drift_category: presentation_drift_category(dims),
            drift_severity: DriftSeverity::Medium,
            blocking_reasons: vec![],
        };
    }

    if has_telemetry_drift(dims) {
        return ProbeImpactAssessment {
            drift_impact: DriftImpact::Telemetry,
            overall_risk: RiskLevel::Green,
            drift_category: DriftCategory::NoSignificantDrift,
            drift_severity: DriftSeverity::Informational,
            blocking_reasons: vec![],
        };
    }

    ProbeImpactAssessment {
        drift_impact: DriftImpact::Informational,
        overall_risk: RiskLevel::Green,
        drift_category: DriftCategory::NoSignificantDrift,
        drift_severity: DriftSeverity::Informational,
        blocking_reasons: vec![],
    }
}

fn material_drops_block(probe: &Probe, _dims: &ProbeDimensions) -> bool {
    match probe.claim_anchor_policy {
        ClaimAnchorPolicy::Strict => true,
        ClaimAnchorPolicy::Lenient => false,
        ClaimAnchorPolicy::Balanced => {
            matches!(
                probe.category,
                ProbeCategory::Factual | ProbeCategory::Schema | ProbeCategory::Instruction
            ) || probe.category.dropped_claims_force_red()
        }
    }
}

fn has_review_signals(probe: &Probe, dims: &ProbeDimensions) -> bool {
    if dims.refusal.new_refusal || dims.refusal.refusal_lifted {
        return true;
    }
    if dims.semantic.flagged_for_review {
        return true;
    }
    if let Some(f) = &dims.factual {
        if matches!(f.risk, RiskLevel::Amber | RiskLevel::Red) && !f.regression {
            return true;
        }
    }
    if let Some(s) = &dims.schema {
        if matches!(s.risk, RiskLevel::Amber) {
            return true;
        }
    }
    if dims.claim.material_preservation_score < probe.category.preservation_threshold()
        && !dims.claim.material_dropped_claims.is_empty()
    {
        return true;
    }
    if dims.claim.has_material_anchor_drift
        && probe.claim_anchor_policy == ClaimAnchorPolicy::Lenient
    {
        return true;
    }
    if matches!(
        dims.instruction.as_ref().map(|i| i.risk.clone()),
        Some(RiskLevel::Amber)
    ) {
        return true;
    }
    if matches!(
        dims.custom_assertions.as_ref().map(|c| c.risk.clone()),
        Some(RiskLevel::Amber)
    ) {
        return true;
    }
    if dims
        .consistency
        .as_ref()
        .is_some_and(|c| c.consistency_regression)
        && (dims
            .instruction
            .as_ref()
            .is_some_and(|i| !i.regressions.is_empty())
            || dims
                .custom_assertions
                .as_ref()
                .is_some_and(|c| !c.regressions.is_empty())
            || dims.schema.as_ref().is_some_and(|s| !s.v2_schema_valid))
    {
        return true;
    }
    if dims
        .code_equivalence
        .as_ref()
        .is_some_and(|c| c.applies && c.equivalence == CodeEquivalenceLevel::Different)
        && is_code_probe(probe)
    {
        return true;
    }
    if signed_token_delta_pct(dims) < -30.0 {
        return true;
    }
    if probe.presentation_drift == PresentationDriftPolicy::Review
        && (matches!(dims.morphology.risk, RiskLevel::Amber | RiskLevel::Red)
            || matches!(dims.tone.risk, RiskLevel::Amber | RiskLevel::Red)
            || dims.morphology.delta.structure_changed)
    {
        return true;
    }
    false
}

fn has_presentation_drift(probe: &Probe, dims: &ProbeDimensions) -> bool {
    if probe.presentation_drift == PresentationDriftPolicy::Ignore {
        return false;
    }
    matches!(dims.morphology.risk, RiskLevel::Amber | RiskLevel::Red)
        || matches!(dims.tone.risk, RiskLevel::Amber | RiskLevel::Red)
        || dims.morphology.delta.structure_changed
        || dims.morphology.delta.response_type_changed
        || (dims.claim.preservation_score < dims.claim.preservation_threshold
            && dims.claim.material_preservation_score >= dims.claim.preservation_threshold)
        || matches!(dims.claim.risk, RiskLevel::Amber)
        || signed_token_delta_pct(dims).abs() > 10.0
}

fn has_telemetry_drift(dims: &ProbeDimensions) -> bool {
    matches!(dims.latency.risk, RiskLevel::Amber | RiskLevel::Red)
        || dims
            .consistency
            .as_ref()
            .is_some_and(|c| matches!(c.risk, RiskLevel::Amber | RiskLevel::Red))
}

fn review_drift_category(dims: &ProbeDimensions) -> DriftCategory {
    if dims.refusal.new_refusal || dims.refusal.refusal_lifted {
        DriftCategory::PolicyDrift
    } else if signed_token_delta_pct(dims) < -30.0 {
        DriftCategory::ContentCompression
    } else if dims.claim.material_preservation_score < dims.claim.preservation_threshold {
        DriftCategory::CriticalRegression
    } else {
        DriftCategory::FidelityDrift
    }
}

fn presentation_drift_category(dims: &ProbeDimensions) -> DriftCategory {
    if signed_token_delta_pct(dims) < -10.0 {
        DriftCategory::ContentCompression
    } else if dims.morphology.delta.structure_changed || dims.morphology.delta.response_type_changed
    {
        DriftCategory::StructuralDrift
    } else {
        DriftCategory::FidelityDrift
    }
}

fn signed_token_delta_pct(dims: &ProbeDimensions) -> f64 {
    if dims.morphology.v1.token_count == 0 {
        return 0.0;
    }
    (dims.morphology.delta.token_delta as f64 / dims.morphology.v1.token_count as f64) * 100.0
}

/// Default non-blocking impact label for a dimension row in reports.
pub fn dimension_impact_label(dimension: &str, risk: &RiskLevel) -> &'static str {
    if matches!(risk, RiskLevel::Green) {
        return "";
    }
    match dimension {
        "latency" | "consistency" => "telemetry drift, non-blocking",
        "morphology" | "tone" => "presentation drift, non-blocking",
        "semantic" => "review signal, non-blocking",
        "claim" => "claim drift, may be presentation",
        "factual" | "schema" | "instruction" | "refusal" | "custom_assertions" => "",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn base_probe(category: ProbeCategory) -> Probe {
        Probe {
            id: Uuid::new_v4(),
            name: "t".into(),
            category,
            prompt: "q".into(),
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
            format_sensitive: false,
            structure_sensitive: false,
            claim_anchor_policy: ClaimAnchorPolicy::Balanced,
            presentation_drift: PresentationDriftPolicy::Review,
            latency_slo_ms: None,
        }
    }

    fn base_dims() -> ProbeDimensions {
        ProbeDimensions {
            morphology: MorphologyDiff {
                risk: RiskLevel::Green,
                direction: DriftDirection::Neutral,
                v1: MorphologyMetrics {
                    token_count: 100,
                    word_count: 100,
                    sentence_count: 5,
                    paragraph_count: 1,
                    has_lists: false,
                    has_headers: false,
                    has_code_blocks: false,
                    has_caveats: false,
                    response_type: ResponseType::LongParagraph,
                },
                v2: MorphologyMetrics {
                    token_count: 100,
                    word_count: 100,
                    sentence_count: 5,
                    paragraph_count: 1,
                    has_lists: false,
                    has_headers: false,
                    has_code_blocks: false,
                    has_caveats: false,
                    response_type: ResponseType::LongParagraph,
                },
                delta: MorphologyDelta {
                    token_delta: 0,
                    token_delta_pct: 0.0,
                    response_type_changed: false,
                    structure_changed: false,
                },
            },
            tone: ToneDiff {
                risk: RiskLevel::Green,
                direction: DriftDirection::Neutral,
                v1: ToneMetrics {
                    formality_score: 0.5,
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
                    formality_delta: 0.0,
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
            claim: ClaimDiff::default(),
            latency: LatencyDiff {
                risk: RiskLevel::Green,
                direction: DriftDirection::Neutral,
                v1_latency_ms: 100,
                v2_latency_ms: 100,
                delta_ms: 0,
                delta_pct: 0.0,
            },
            consistency: None,
            custom_assertions: None,
            code_equivalence: None,
        }
    }

    #[test]
    fn latency_only_never_blocks() {
        let probe = base_probe(ProbeCategory::Semantic);
        let mut dims = base_dims();
        dims.latency = LatencyDiff {
            risk: RiskLevel::Red,
            direction: DriftDirection::Regression,
            v1_latency_ms: 50,
            v2_latency_ms: 500,
            delta_ms: 450,
            delta_pct: 9.0,
        };
        let a = assess_probe_impact(&probe, &dims);
        assert_eq!(a.drift_impact, DriftImpact::Telemetry);
        assert!(!a.drift_impact.is_blocking());
    }

    #[test]
    fn factual_regression_blocks() {
        let probe = base_probe(ProbeCategory::Factual);
        let mut dims = base_dims();
        dims.factual = Some(FactualDiff {
            risk: RiskLevel::Red,
            direction: DriftDirection::Regression,
            v1_correct: true,
            v2_correct: false,
            v1_answer_extract: "Paris".into(),
            v2_answer_extract: "London".into(),
            regression: true,
            improvement: false,
        });
        let a = assess_probe_impact(&probe, &dims);
        assert_eq!(a.drift_impact, DriftImpact::Blocking);
    }

    #[test]
    fn invalid_json_schema_blocks() {
        let probe = base_probe(ProbeCategory::Schema);
        let mut dims = base_dims();
        dims.schema = Some(SchemaDiff {
            risk: RiskLevel::Red,
            direction: DriftDirection::Regression,
            v1_valid_json: true,
            v2_valid_json: false,
            v1_schema_valid: true,
            v2_schema_valid: false,
            v1_missing_fields: vec![],
            v2_missing_fields: vec![],
            v1_extra_fields: vec![],
            v2_extra_fields: vec![],
            field_type_changes: vec![],
        });
        let a = assess_probe_impact(&probe, &dims);
        assert_eq!(a.drift_impact, DriftImpact::Blocking);
    }

    #[test]
    fn morphology_red_is_presentation_not_blocking() {
        let probe = base_probe(ProbeCategory::Semantic);
        let mut dims = base_dims();
        dims.morphology.risk = RiskLevel::Red;
        dims.morphology.delta.token_delta = -40;
        dims.morphology.v2.token_count = 60;
        let a = assess_probe_impact(&probe, &dims);
        assert_ne!(a.drift_impact, DriftImpact::Blocking);
    }

    fn code_probe() -> Probe {
        let mut probe = base_probe(ProbeCategory::Instruction);
        probe.tags = vec!["code-generation".into(), "sql".into()];
        probe.prompt = "Return only the SQL, no explanation.".into();
        probe.presentation_drift = PresentationDriftPolicy::Review;
        probe
    }

    #[test]
    fn code_formatting_equivalence_is_presentation() {
        use crate::code_equivalence::build_code_equivalence_diff;

        let probe = code_probe();
        let v1 = "```sql\nSELECT 1;\n```";
        let v2 = "SELECT 1;";
        let mut dims = base_dims();
        dims.code_equivalence = build_code_equivalence_diff(&probe, v1, v2);
        dims.morphology.v1.has_code_blocks = true;
        dims.morphology.v2.has_code_blocks = false;
        dims.morphology.delta.structure_changed = true;
        let a = assess_probe_impact(&probe, &dims);
        assert_eq!(a.drift_impact, DriftImpact::Presentation);
    }

    #[test]
    fn format_sensitive_still_blocks_fence_change() {
        use crate::code_equivalence::build_code_equivalence_diff;

        let mut probe = code_probe();
        probe.format_sensitive = true;
        probe.presentation_drift = PresentationDriftPolicy::Blocking;
        let v1 = "```sql\nSELECT 1;\n```";
        let v2 = "SELECT 1;";
        let mut dims = base_dims();
        dims.code_equivalence = build_code_equivalence_diff(&probe, v1, v2);
        dims.morphology.v1.has_code_blocks = true;
        dims.morphology.v2.has_code_blocks = false;
        dims.morphology.risk = RiskLevel::Red;
        let a = assess_probe_impact(&probe, &dims);
        assert_eq!(a.drift_impact, DriftImpact::Blocking);
    }
}
