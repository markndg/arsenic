//! Canonical material-change decisions shared by dimension summaries and the
//! behavioural fingerprint.
//!
//! These helpers encode the same threshold bands that produce Amber/Red risk in
//! [`crate::comparison::ComparisonEngine`]. Fingerprint retention must use these
//! predicates — not raw numeric inequality, and not risk colour as a synonym for
//! unchanged/changed.
//!
//! Default numeric thresholds match [`crate::comparison::RiskThresholds::default`].

use crate::types::*;

/// Default morphology amber band: absolute token-delta fraction of baseline.
pub const DEFAULT_MORPHOLOGY_TOKEN_DELTA_AMBER: f64 = 0.5;

/// Consistency aggregate retention at or above this stays in changelog telemetry
/// (not high-impact). A 5pp repeatability gap is a conservative behavioural signal;
/// 99% retention (≈1pp) must not rank as a major compatibility drop.
pub const CONSISTENCY_HIGH_IMPACT_RETENTION_BELOW: f64 = 95.0;

/// Morphology crosses Arsenic's material band (structure / response-type / length).
///
/// Mirrors the Amber-or-worse branch of `morphology_risk_level` in `comparison.rs`.
pub fn morphology_crosses_material_band(
    token_delta_pct: f64,
    response_type_changed: bool,
    structure_changed: bool,
    amber_token_delta_pct: f64,
) -> bool {
    token_delta_pct >= amber_token_delta_pct || structure_changed || response_type_changed
}

/// Material morphology change using default amber token-delta threshold (0.5).
pub fn morphology_materially_changed(m: &MorphologyDiff) -> bool {
    morphology_crosses_material_band(
        m.delta.token_delta_pct,
        m.delta.response_type_changed,
        m.delta.structure_changed,
        DEFAULT_MORPHOLOGY_TOKEN_DELTA_AMBER,
    )
}

/// Material tone change: the compare layer's `significant_shift` flag (formality /
/// assertiveness amber band or |hedge| ≥ 4). Raw hedge deltas below that band do not count.
pub fn tone_materially_changed(t: &ToneDiff) -> bool {
    t.delta.significant_shift
}

/// Material factual change: regression or both-wrong. Improvements are Green and
/// are not counted as `probes_affected` in dimension summaries.
pub fn factual_materially_changed(f: &FactualDiff) -> bool {
    f.regression || (!f.v1_correct && !f.v2_correct)
}

/// Material schema change matching Amber/Red assignment in `compare_schema`.
pub fn schema_materially_changed(s: &SchemaDiff) -> bool {
    !s.v2_valid_json || !s.v2_schema_valid || !s.v1_schema_valid || !s.field_type_changes.is_empty()
}

/// Material instruction change: listed regressions or lower pass rate.
/// A higher pass rate alone (improvement) stays Green / unaffected.
pub fn instruction_materially_changed(i: &InstructionDiff) -> bool {
    !i.regressions.is_empty() || i.v2_pass_rate < i.v1_pass_rate
}

/// Material refusal change: new refusal or refusal lifted.
pub fn refusal_materially_changed(r: &RefusalDiff) -> bool {
    r.new_refusal || r.refusal_lifted
}

/// Material semantic change: the compare layer's `flagged_for_review` flag
/// (similarity below the amber/scoring threshold). Raw cosine `< 1.0` above the
/// threshold does not count. Formatting soften clears this flag when semantic
/// risk is forced Green.
pub fn semantic_materially_changed(s: &SemanticDiff) -> bool {
    if s.semantic_scoring_disabled {
        return false;
    }
    s.flagged_for_review
}

/// Material claim change matching Amber/Red bands in `ClaimMatcher::match_claims`.
pub fn claim_materially_changed(c: &ClaimDiff, category: ProbeCategory) -> bool {
    let amber = category.preservation_amber_threshold();
    c.material_preservation_score < c.preservation_threshold
        || c.has_material_anchor_drift
        || !c.material_dropped_claims.is_empty()
        || !c.drifted_claims.is_empty()
        || c.preservation_score < amber
}

/// Assessed consistency materiality for fingerprint / drift reporting.
///
/// Absolute inconsistency (`baseline_consistent` / `candidate_consistent`) drives
/// risk and `DimensionSummary::probes_affected`. Fingerprint retention uses only
/// [`Self::materially_changed`] (band-crossing drift relative to baseline).
///
/// No same-band variance-delta materiality threshold exists in Arsenic; raw
/// `|v1_variance − v2_variance|` while both sides remain on the same side of the
/// 0.12 band is telemetry only (never fingerprint drift).
#[derive(Debug, Clone, PartialEq)]
pub struct ConsistencyMateriality {
    pub baseline_consistent: bool,
    pub candidate_consistent: bool,
    pub crossed_band: bool,
    pub absolute_variance_delta: f64,
    pub materially_changed: bool,
    pub direction: DriftDirection,
}

/// Evaluate consistency drift relative to baseline (not absolute inconsistency).
pub fn assess_consistency_materiality(c: &ConsistencyDiff) -> ConsistencyMateriality {
    let crossed_band = c.v1_consistent != c.v2_consistent;
    let direction = if c.v1_consistent && !c.v2_consistent {
        DriftDirection::Regression
    } else if !c.v1_consistent && c.v2_consistent {
        DriftDirection::Improvement
    } else {
        DriftDirection::Neutral
    };
    ConsistencyMateriality {
        baseline_consistent: c.v1_consistent,
        candidate_consistent: c.v2_consistent,
        crossed_band,
        absolute_variance_delta: (c.v1_variance - c.v2_variance).abs(),
        // Band crossing only — no same-band delta threshold is defined.
        materially_changed: crossed_band,
        direction,
    }
}

/// Material consistency *drift*: baseline and candidate sit on opposite sides of
/// the absolute consistency band (`v1_consistent != v2_consistent`).
///
/// Both inconsistent with equal (or near-equal) variance is **not** drift.
/// Absolute inconsistency remains visible via risk / `probes_affected`.
pub fn consistency_materially_changed(c: &ConsistencyDiff) -> bool {
    assess_consistency_materiality(c).materially_changed
}

/// One applicable fingerprint observation after canonical materiality is applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FingerprintObservation {
    pub applicable: bool,
    pub materially_changed: bool,
    pub direction: DriftDirection,
    pub risk: RiskLevel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morphology_one_token_below_band_is_immaterial() {
        // 1 token on a 10-token baseline = 10% << 50% amber.
        assert!(!morphology_crosses_material_band(0.1, false, false, 0.5));
    }

    #[test]
    fn morphology_structure_is_material() {
        assert!(morphology_crosses_material_band(0.0, false, true, 0.5));
    }

    #[test]
    fn consistency_both_consistent_immaterial_despite_raw_delta() {
        let c = ConsistencyDiff {
            risk: RiskLevel::Green,
            direction: DriftDirection::Neutral,
            v1_runs: 3,
            v2_runs: 3,
            v1_variance: 0.01,
            v2_variance: 0.02,
            v1_consistent: true,
            v2_consistent: true,
            consistency_regression: false,
            consistency_improvement: false,
        };
        assert!(!consistency_materially_changed(&c));
    }

    #[test]
    fn consistency_loose_candidate_is_material() {
        let c = ConsistencyDiff {
            risk: RiskLevel::Amber,
            direction: DriftDirection::Regression,
            v1_runs: 3,
            v2_runs: 3,
            v1_variance: 0.01,
            v2_variance: 0.20,
            v1_consistent: true,
            v2_consistent: false,
            consistency_regression: true,
            consistency_improvement: false,
        };
        let a = assess_consistency_materiality(&c);
        assert!(a.materially_changed);
        assert!(a.crossed_band);
        assert_eq!(a.direction, DriftDirection::Regression);
    }

    #[test]
    fn consistency_both_inconsistent_equal_is_not_drift() {
        let c = ConsistencyDiff {
            risk: RiskLevel::Amber,
            direction: DriftDirection::Neutral,
            v1_runs: 3,
            v2_runs: 3,
            v1_variance: 0.18,
            v2_variance: 0.18,
            v1_consistent: false,
            v2_consistent: false,
            consistency_regression: false,
            consistency_improvement: false,
        };
        let a = assess_consistency_materiality(&c);
        assert!(!a.materially_changed);
        assert!(!a.crossed_band);
        assert!(!a.baseline_consistent && !a.candidate_consistent);
        assert_eq!(a.direction, DriftDirection::Neutral);
    }

    #[test]
    fn consistency_both_inconsistent_tiny_delta_is_not_drift() {
        let c = ConsistencyDiff {
            risk: RiskLevel::Amber,
            direction: DriftDirection::Neutral,
            v1_runs: 3,
            v2_runs: 3,
            v1_variance: 0.18,
            v2_variance: 0.181,
            v1_consistent: false,
            v2_consistent: false,
            consistency_regression: false,
            consistency_improvement: false,
        };
        assert!(!consistency_materially_changed(&c));
    }

    #[test]
    fn consistency_inconsistent_to_consistent_is_improvement() {
        let c = ConsistencyDiff {
            risk: RiskLevel::Amber,
            direction: DriftDirection::Improvement,
            v1_runs: 3,
            v2_runs: 3,
            v1_variance: 0.14,
            v2_variance: 0.10,
            v1_consistent: false,
            v2_consistent: true,
            consistency_regression: false,
            consistency_improvement: true,
        };
        let a = assess_consistency_materiality(&c);
        assert!(a.materially_changed);
        assert_eq!(a.direction, DriftDirection::Improvement);
    }

    #[test]
    fn consistency_truth_table_both_consistent_equal() {
        let c = ConsistencyDiff {
            risk: RiskLevel::Green,
            direction: DriftDirection::Neutral,
            v1_runs: 3,
            v2_runs: 3,
            v1_variance: 0.05,
            v2_variance: 0.05,
            v1_consistent: true,
            v2_consistent: true,
            consistency_regression: false,
            consistency_improvement: false,
        };
        assert!(!consistency_materially_changed(&c));
    }

    #[test]
    fn consistency_truth_table_consistent_to_inconsistent() {
        let c = ConsistencyDiff {
            risk: RiskLevel::Red,
            direction: DriftDirection::Regression,
            v1_runs: 3,
            v2_runs: 3,
            v1_variance: 0.10,
            v2_variance: 0.14,
            v1_consistent: true,
            v2_consistent: false,
            consistency_regression: true,
            consistency_improvement: false,
        };
        let a = assess_consistency_materiality(&c);
        assert!(a.materially_changed);
        assert_eq!(a.direction, DriftDirection::Regression);
    }
}
