//! Behavioural fingerprint — baseline-retention scores across evaluated dimensions.
//!
//! A score of 100 means no detected drift from the baseline in that dimension.
//! A score of 0 means every applicable probe showed drift (regression, improvement,
//! or neutral behavioural change). This is a compatibility fingerprint, not a
//! global quality score.
//!
//! Retention counts use canonical material-change helpers from [`crate::materiality`].
//! Risk colour is severity only — never a synonym for unchanged or changed.
//!
//! # Consistency variance metric
//!
//! `ConsistencyDiff::{v1,v2}_variance` is the mean pairwise distance
//! `1 − cosine_similarity` over L2-normalised non-negative hash-bag embeddings of
//! multi-run responses (`run_variance` in `comparison.rs`). Because bag-of-counts
//! embeddings are non-negative, cosine ∈ [0, 1] and the distance is intrinsically
//! ∈ [0, 1]. Clamping to [0, 1] is therefore a defensive float guard, not a
//! normalisation of unbounded statistical variance. Repeatability and retention
//! are derived from that bounded distance (see [`variance_to_repeatability`] and
//! [`consistency_retention`]).

use serde::{Deserialize, Serialize};

use crate::materiality::{
    assess_consistency_materiality, claim_materially_changed, factual_materially_changed,
    instruction_materially_changed, morphology_materially_changed, refusal_materially_changed,
    schema_materially_changed, semantic_materially_changed, tone_materially_changed,
    CONSISTENCY_HIGH_IMPACT_RETENTION_BELOW,
};
use crate::types::*;
use thiserror::Error;

pub const FINGERPRINT_VERSION: u32 = 1;
pub const MIN_AXES_FOR_RADAR: usize = 3;

/// Interpretation of fingerprint scores.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FingerprintInterpretation {
    #[default]
    BaselineRetention,
}

/// Confidence in an axis score based on sample size.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FingerprintConfidence {
    #[default]
    Unavailable,
    Limited,
    Sufficient,
}

/// Why a preferred fingerprint axis is absent from the radar/table.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OmittedAxisReason {
    /// Dimension was considered but no probe produced applicable observations.
    NoApplicableProbes,
    /// Feature intentionally off for this run (e.g. semantic scoring disabled).
    Disabled,
    /// Capability not available (e.g. multi-run consistency not sampled).
    Unavailable,
}

/// Preferred axis omitted from the polygon with an explicit reason.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OmittedFingerprintAxis {
    pub id: String,
    pub label: String,
    pub reason: OmittedAxisReason,
    pub detail: String,
}

/// Explicit retention class for one applicable observation (independent of risk).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetentionClass {
    Unchanged,
    Regression,
    Improvement,
    NeutralChange,
}

/// How an axis score is derived from observations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FingerprintAggregationKind {
    /// `100 × unchanged / applicable` from material per-probe decisions.
    #[default]
    ProbeRetention,
    /// Aggregate similarity of repeatability (consistency telemetry axis).
    AggregateSimilarity,
}

/// One behavioural axis on the fingerprint radar.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FingerprintAxis {
    pub id: String,
    pub label: String,
    /// Compatibility retention score in [0, 100].
    pub score: f64,
    pub applicable_probes: usize,
    pub unchanged_probes: usize,
    /// Materially changed probes (canonical dimension materiality).
    pub changed_probes: usize,
    pub regressions: usize,
    pub improvements: usize,
    pub neutral_changes: usize,
    /// Worst Arsenic risk/severity among applicable observations (separate from retention).
    pub risk: RiskLevel,
    pub confidence: FingerprintConfidence,
    /// Probe UUIDs (as strings) that contributed materially changed evidence.
    pub evidence_probe_ids: Vec<String>,
    pub explanation: String,
    /// Score aggregation semantics for this axis.
    #[serde(default)]
    pub aggregation_kind: FingerprintAggregationKind,
    /// Alias of [`Self::changed_probes`] for consistency clarity in JSON consumers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub materially_changed_probes: Option<usize>,
    /// Probes with any raw variance inequality (consistency only; not used for retention).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_changed_probes: Option<usize>,
    /// Baseline probes outside the absolute consistency band (consistency only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_inconsistent_probes: Option<usize>,
    /// Candidate probes outside the absolute consistency band (consistency only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_inconsistent_probes: Option<usize>,
    /// Optional raw baseline value for telemetry-style axes (e.g. consistency %).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_raw: Option<f64>,
    /// Optional raw candidate value for telemetry-style axes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_raw: Option<f64>,
    /// Absolute difference between candidate and baseline raw values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub absolute_difference: Option<f64>,
    /// Unit label for raw values (e.g. "repeatability_pct").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_unit: Option<String>,
}

/// Structured failures when derived rollups disagree with canonical probe results.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RollupValidationError {
    #[error(
        "fingerprint/summary mismatch on {axis}: fingerprint changed={fingerprint}, summary material={summary}"
    )]
    FingerprintSummaryMismatch {
        axis: String,
        fingerprint: usize,
        summary: usize,
    },
    #[error("invalid fingerprint score on {axis}: {score}")]
    InvalidFingerprintScore { axis: String, score: f64 },
    #[error("invalid fingerprint identity on {axis}")]
    InvalidFingerprintIdentity { axis: String },
    #[error("fingerprint validation: {0}")]
    FingerprintInvalid(String),
}

/// Run-level behavioural fingerprint relative to the baseline model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BehaviourFingerprint {
    pub version: u32,
    pub interpretation: FingerprintInterpretation,
    pub baseline_label: String,
    pub candidate_label: String,
    pub axes: Vec<FingerprintAxis>,
    /// Preferred axes omitted from the polygon/table, each with an explicit reason.
    #[serde(default)]
    pub omitted_axes: Vec<OmittedFingerprintAxis>,
    /// True when at least [`MIN_AXES_FOR_RADAR`] axes are available for a radar polygon.
    pub radar_available: bool,
    /// True when some preferred axes were omitted due to missing evidence.
    pub partial: bool,
    /// Non-empty when rollup validation failed after recompute from probe results.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation_diagnostics: Vec<String>,
}

impl Default for BehaviourFingerprint {
    fn default() -> Self {
        Self {
            version: FINGERPRINT_VERSION,
            interpretation: FingerprintInterpretation::BaselineRetention,
            baseline_label: String::new(),
            candidate_label: String::new(),
            axes: Vec::new(),
            omitted_axes: Vec::new(),
            radar_available: false,
            partial: false,
            validation_diagnostics: Vec::new(),
        }
    }
}

/// Precomputed SVG geometry for deterministic HTML rendering (no browser score calc).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FingerprintSvgModel {
    pub size: f64,
    pub center: f64,
    pub max_radius: f64,
    pub grid_polygons: Vec<String>,
    pub spokes: Vec<FingerprintSvgSpoke>,
    pub baseline_points: String,
    pub candidate_points: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FingerprintSvgSpoke {
    pub axis_id: String,
    pub label: String,
    pub score: f64,
    pub angle_deg: f64,
    pub label_x: f64,
    pub label_y: f64,
    pub point_x: f64,
    pub point_y: f64,
    pub baseline_x: f64,
    pub baseline_y: f64,
    pub spoke_x2: f64,
    pub spoke_y2: f64,
}

struct AxisAccum {
    id: &'static str,
    label: &'static str,
    applicable: usize,
    unchanged: usize,
    regressions: usize,
    improvements: usize,
    neutrals: usize,
    worst_risk: RiskLevel,
    evidence: Vec<String>,
    /// For consistency: sum of normalised repeatability.
    v1_raw_sum: f64,
    v2_raw_sum: f64,
    raw_n: usize,
    /// Consistency only: raw variance inequality count (not materiality).
    raw_changed: usize,
    baseline_inconsistent: usize,
    candidate_inconsistent: usize,
}

impl AxisAccum {
    fn new(id: &'static str, label: &'static str) -> Self {
        Self {
            id,
            label,
            applicable: 0,
            unchanged: 0,
            regressions: 0,
            improvements: 0,
            neutrals: 0,
            worst_risk: RiskLevel::Green,
            evidence: Vec::new(),
            v1_raw_sum: 0.0,
            v2_raw_sum: 0.0,
            raw_n: 0,
            raw_changed: 0,
            baseline_inconsistent: 0,
            candidate_inconsistent: 0,
        }
    }

    fn observe(&mut self, risk: &RiskLevel, class: RetentionClass, probe_id: &str) {
        self.applicable += 1;
        self.worst_risk = self.worst_risk.clone().max(risk.clone());
        match class {
            RetentionClass::Unchanged => self.unchanged += 1,
            RetentionClass::Regression => {
                self.regressions += 1;
                self.evidence.push(probe_id.to_string());
            }
            RetentionClass::Improvement => {
                self.improvements += 1;
                self.evidence.push(probe_id.to_string());
            }
            RetentionClass::NeutralChange => {
                self.neutrals += 1;
                self.evidence.push(probe_id.to_string());
            }
        }
    }

    fn changed(&self) -> usize {
        self.regressions + self.improvements + self.neutrals
    }

    fn into_axis(self) -> Option<FingerprintAxis> {
        if self.applicable == 0 {
            return None;
        }
        let changed = self.changed();
        let score = retention_score(self.unchanged, self.applicable);
        let confidence = confidence_for(self.applicable);
        let explanation = explain_axis(
            self.label,
            score,
            self.unchanged,
            self.applicable,
            changed,
            self.regressions,
            self.improvements,
            self.neutrals,
        );
        Some(FingerprintAxis {
            id: self.id.into(),
            label: self.label.into(),
            score,
            applicable_probes: self.applicable,
            unchanged_probes: self.unchanged,
            changed_probes: changed,
            regressions: self.regressions,
            improvements: self.improvements,
            neutral_changes: self.neutrals,
            risk: self.worst_risk,
            confidence,
            evidence_probe_ids: self.evidence,
            explanation,
            aggregation_kind: FingerprintAggregationKind::ProbeRetention,
            materially_changed_probes: None,
            raw_changed_probes: None,
            baseline_inconsistent_probes: None,
            candidate_inconsistent_probes: None,
            baseline_raw: None,
            candidate_raw: None,
            absolute_difference: None,
            raw_unit: None,
        })
    }

    fn into_consistency_axis(self) -> Option<FingerprintAxis> {
        if self.applicable == 0 || self.raw_n == 0 {
            return None;
        }
        let baseline = round2(self.v1_raw_sum / self.raw_n as f64);
        let candidate = round2(self.v2_raw_sum / self.raw_n as f64);
        let abs_diff = round2((candidate - baseline).abs());
        let score = round2(consistency_retention(baseline, candidate));
        let changed = self.changed();
        let confidence = confidence_for(self.applicable);
        let explanation = format!(
            "Aggregate consistency similarity: {score:.2}%. Baseline repeatability {baseline:.2}%, candidate {candidate:.2}%, difference {abs_diff:.2} percentage points. Material consistency drift: {changed} of {} probes crossed the band ({}); absolute inconsistency: {} baseline / {} candidate probes outside the 0.12 band.",
            self.applicable,
            direction_summary_phrase(
                changed,
                self.regressions,
                self.improvements,
                self.neutrals
            ),
            self.baseline_inconsistent,
            self.candidate_inconsistent
        );
        Some(FingerprintAxis {
            id: self.id.into(),
            label: self.label.into(),
            score,
            applicable_probes: self.applicable,
            unchanged_probes: self.unchanged,
            changed_probes: changed,
            regressions: self.regressions,
            improvements: self.improvements,
            neutral_changes: self.neutrals,
            risk: self.worst_risk,
            confidence,
            evidence_probe_ids: self.evidence,
            explanation,
            aggregation_kind: FingerprintAggregationKind::AggregateSimilarity,
            materially_changed_probes: Some(changed),
            raw_changed_probes: Some(self.raw_changed),
            baseline_inconsistent_probes: Some(self.baseline_inconsistent),
            candidate_inconsistent_probes: Some(self.candidate_inconsistent),
            baseline_raw: Some(baseline),
            candidate_raw: Some(candidate),
            absolute_difference: Some(abs_diff),
            raw_unit: Some("repeatability_pct".into()),
        })
    }
}

/// Classify from material change + direction. Direction alone never marks a change.
fn classify_retention(direction: DriftDirection, materially_changed: bool) -> RetentionClass {
    if !materially_changed {
        return RetentionClass::Unchanged;
    }
    match direction {
        DriftDirection::Regression => RetentionClass::Regression,
        DriftDirection::Improvement => RetentionClass::Improvement,
        DriftDirection::Neutral | DriftDirection::NotApplicable => RetentionClass::NeutralChange,
    }
}

/// Convert mean pairwise embedding distance (0 = identical runs) to repeatability %.
///
/// Source metric is intrinsically in [0, 1] for hash-bag embeddings; clamp is defensive.
pub fn variance_to_repeatability(variance: f64) -> f64 {
    if !variance.is_finite() {
        return 0.0;
    }
    let clamped = variance.clamp(0.0, 1.0);
    round2(100.0 * (1.0 - clamped))
}

/// Retention of consistency relative to baseline: `100 - |candidate - baseline|`.
///
/// This measures similarity to baseline consistency, not absolute candidate consistency.
pub fn consistency_retention(baseline_pct: f64, candidate_pct: f64) -> f64 {
    if !baseline_pct.is_finite() || !candidate_pct.is_finite() {
        return 0.0;
    }
    let b = baseline_pct.clamp(0.0, 100.0);
    let c = candidate_pct.clamp(0.0, 100.0);
    (100.0 - (c - b).abs()).clamp(0.0, 100.0)
}

pub fn retention_score(unchanged: usize, applicable: usize) -> f64 {
    if applicable == 0 {
        return 0.0;
    }
    let raw = 100.0 * (unchanged as f64) / (applicable as f64);
    if !raw.is_finite() {
        return 0.0;
    }
    round2(raw.clamp(0.0, 100.0))
}

fn confidence_for(applicable: usize) -> FingerprintConfidence {
    match applicable {
        0 => FingerprintConfidence::Unavailable,
        1 | 2 => FingerprintConfidence::Limited,
        _ => FingerprintConfidence::Sufficient,
    }
}

fn round2(v: f64) -> f64 {
    if !v.is_finite() {
        return 0.0;
    }
    (v * 100.0).round() / 100.0
}

fn explain_axis(
    label: &str,
    score: f64,
    unchanged: usize,
    applicable: usize,
    changed: usize,
    regressions: usize,
    improvements: usize,
    neutrals: usize,
) -> String {
    if changed == 0 {
        return format!(
            "{label}: {unchanged} of {applicable} applicable probes retained baseline-equivalent behaviour (score {score:.0}%)."
        );
    }
    format!(
        "{label}: {unchanged} of {applicable} applicable probes retained baseline-equivalent behaviour (score {score:.0}%). Material changes: {}.",
        direction_summary_phrase(changed, regressions, improvements, neutrals)
    )
}

fn direction_summary_phrase(
    changed: usize,
    regressions: usize,
    improvements: usize,
    neutrals: usize,
) -> String {
    let mut parts = Vec::new();
    if regressions > 0 {
        parts.push(format!(
            "{regressions} regression{}",
            if regressions == 1 { "" } else { "s" }
        ));
    }
    if improvements > 0 {
        parts.push(format!(
            "{improvements} improvement{}",
            if improvements == 1 { "" } else { "s" }
        ));
    }
    if neutrals > 0 {
        parts.push(format!(
            "{neutrals} neutral-direction change{}",
            if neutrals == 1 { "" } else { "s" }
        ));
    }
    if parts.is_empty() {
        format!(
            "{changed} material change{}",
            if changed == 1 { "" } else { "s" }
        )
    } else {
        parts.join(", ")
    }
}

fn changelog_axis_line(axis: &FingerprintAxis) -> String {
    let label = axis.label.trim_end_matches(" retention");
    let drop_note = format!(
        "{label} retention fell to {:.0}%: {} of {} probes changed",
        axis.score, axis.changed_probes, axis.applicable_probes
    );
    let detail = direction_summary_phrase(
        axis.changed_probes,
        axis.regressions,
        axis.improvements,
        axis.neutral_changes,
    );
    if axis.regressions == 0
        && axis.improvements == 0
        && axis.neutral_changes == axis.changed_probes
        && axis.changed_probes > 0
    {
        format!("{drop_note}, all neutral in quality direction.")
    } else if axis.regressions == 0 && axis.improvements == 0 {
        format!("{drop_note} ({detail}).")
    } else {
        format!("{drop_note} ({detail}).")
    }
}

/// Preferred stable axis order. Axes with zero applicable probes are omitted.
const AXIS_ORDER: &[(&str, &str)] = &[
    ("morphology", "Morphology"),
    ("tone", "Tone"),
    ("factual", "Factual"),
    ("schema", "Schema"),
    ("instruction", "Instruction"),
    ("refusal", "Refusal"),
    ("consistency", "Consistency retention"),
    ("claim", "Claim retention"),
    ("semantic", "Semantic retention"),
];

/// Build a behavioural fingerprint from probe results.
pub fn compute_behaviour_fingerprint(
    results: &[ProbeResult],
    v1_model: &ModelInfo,
    v2_model: &ModelInfo,
) -> BehaviourFingerprint {
    let mut morph = AxisAccum::new("morphology", "Morphology");
    let mut tone = AxisAccum::new("tone", "Tone");
    let mut factual = AxisAccum::new("factual", "Factual");
    let mut schema = AxisAccum::new("schema", "Schema");
    let mut instruction = AxisAccum::new("instruction", "Instruction");
    let mut refusal = AxisAccum::new("refusal", "Refusal");
    let mut consistency = AxisAccum::new("consistency", "Consistency retention");
    let mut claim = AxisAccum::new("claim", "Claim retention");
    let mut semantic = AxisAccum::new("semantic", "Semantic retention");

    let mut semantic_enabled = false;
    let mut semantic_disabled = false;
    let mut consistency_insufficient_runs = false;
    let mut consistency_absent = true;

    for pr in results {
        let pid = pr.probe.id.to_string();
        let d = &pr.dimensions;

        morph.observe(
            &d.morphology.risk,
            classify_retention(
                d.morphology.direction,
                morphology_materially_changed(&d.morphology),
            ),
            &pid,
        );
        tone.observe(
            &d.tone.risk,
            classify_retention(d.tone.direction, tone_materially_changed(&d.tone)),
            &pid,
        );
        refusal.observe(
            &d.refusal.risk,
            classify_retention(d.refusal.direction, refusal_materially_changed(&d.refusal)),
            &pid,
        );

        if let Some(f) = &d.factual {
            factual.observe(
                &f.risk,
                classify_retention(f.direction, factual_materially_changed(f)),
                &pid,
            );
        }
        if let Some(s) = &d.schema {
            schema.observe(
                &s.risk,
                classify_retention(s.direction, schema_materially_changed(s)),
                &pid,
            );
        }
        if let Some(i) = &d.instruction {
            instruction.observe(
                &i.risk,
                classify_retention(i.direction, instruction_materially_changed(i)),
                &pid,
            );
        }

        // Claim: always evaluated; empty N/A claim sets still count as applicable.
        if !matches!(d.claim.direction, DriftDirection::NotApplicable)
            || !matches!(d.claim.risk, RiskLevel::Green)
            || !d.claim.v1_claims.is_empty()
            || !d.claim.v2_claims.is_empty()
            || d.claim.preservation_score < 1.0
        {
            claim.observe(
                &d.claim.risk,
                classify_retention(
                    d.claim.direction,
                    claim_materially_changed(&d.claim, pr.probe.category),
                ),
                &pid,
            );
        } else {
            claim.observe(
                &RiskLevel::Green,
                classify_retention(DriftDirection::Neutral, false),
                &pid,
            );
        }

        if d.semantic.semantic_scoring_disabled {
            semantic_disabled = true;
        } else {
            semantic_enabled = true;
            semantic.observe(
                &d.semantic.risk,
                classify_retention(
                    d.semantic.direction,
                    semantic_materially_changed(&d.semantic),
                ),
                &pid,
            );
        }

        if let Some(c) = &d.consistency {
            consistency_absent = false;
            if c.v1_runs > 1 && c.v2_runs > 1 {
                let assessed = assess_consistency_materiality(c);
                consistency.observe(
                    &c.risk,
                    classify_retention(assessed.direction, assessed.materially_changed),
                    &pid,
                );
                if assessed.absolute_variance_delta > 1e-9 {
                    consistency.raw_changed += 1;
                }
                if !assessed.baseline_consistent {
                    consistency.baseline_inconsistent += 1;
                }
                if !assessed.candidate_consistent {
                    consistency.candidate_inconsistent += 1;
                }
                consistency.v1_raw_sum += variance_to_repeatability(c.v1_variance);
                consistency.v2_raw_sum += variance_to_repeatability(c.v2_variance);
                consistency.raw_n += 1;
            } else {
                consistency_insufficient_runs = true;
            }
        }
    }

    let preferred = AXIS_ORDER.len();
    let mut axes = Vec::new();
    let built = [
        morph.into_axis(),
        tone.into_axis(),
        factual.into_axis(),
        schema.into_axis(),
        instruction.into_axis(),
        refusal.into_axis(),
        consistency.into_consistency_axis(),
        claim.into_axis(),
        semantic.into_axis(),
    ];
    for axis in built.into_iter().flatten() {
        axes.push(axis);
    }

    let present: std::collections::HashSet<&str> = axes.iter().map(|a| a.id.as_str()).collect();
    let mut omitted_axes = Vec::new();
    for &(id, label) in AXIS_ORDER {
        if present.contains(id) {
            continue;
        }
        let (reason, detail) = match id {
            "factual" => (
                OmittedAxisReason::NoApplicableProbes,
                "No applicable probes: none had a known answer for factual scoring.".into(),
            ),
            "schema" => (
                OmittedAxisReason::NoApplicableProbes,
                "No applicable probes: none declared an expected schema.".into(),
            ),
            "instruction" => (
                OmittedAxisReason::NoApplicableProbes,
                "No applicable probes: none declared instruction checks.".into(),
            ),
            "semantic" if semantic_disabled && !semantic_enabled => (
                OmittedAxisReason::Disabled,
                "Semantic scoring was disabled for this run.".into(),
            ),
            "semantic" => (
                OmittedAxisReason::NoApplicableProbes,
                "No applicable probes: semantic scoring produced no observations.".into(),
            ),
            "consistency" if consistency_insufficient_runs => (
                OmittedAxisReason::Unavailable,
                "Consistency sampling present but fewer than 2 runs per model.".into(),
            ),
            "consistency" if consistency_absent => (
                OmittedAxisReason::Unavailable,
                "No multi-run consistency observations (enable --consistency-runs > 1).".into(),
            ),
            "consistency" => (
                OmittedAxisReason::Unavailable,
                "Consistency retention unavailable: insufficient multi-run observations.".into(),
            ),
            _ if results.is_empty() => (
                OmittedAxisReason::NoApplicableProbes,
                format!("No applicable probes: empty comparison for {label}."),
            ),
            _ => (
                OmittedAxisReason::NoApplicableProbes,
                format!("No applicable probes for {label}."),
            ),
        };
        omitted_axes.push(OmittedFingerprintAxis {
            id: id.into(),
            label: label.into(),
            reason,
            detail,
        });
    }

    let radar_available = axes.len() >= MIN_AXES_FOR_RADAR;
    let partial = !omitted_axes.is_empty() || axes.len() < preferred;

    BehaviourFingerprint {
        version: FINGERPRINT_VERSION,
        interpretation: FingerprintInterpretation::BaselineRetention,
        baseline_label: format!("{} ({})", v1_model.label, v1_model.model_id),
        candidate_label: format!("{} ({})", v2_model.label, v2_model.model_id),
        axes,
        omitted_axes,
        radar_available,
        partial,
        validation_diagnostics: Vec::new(),
    }
}

/// Validate fingerprint data before rendering. Returns human-readable errors.
pub fn validate_fingerprint(fp: &BehaviourFingerprint) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if fp.version == 0 {
        errors.push("fingerprint version must be >= 1".into());
    }
    let mut seen = std::collections::HashSet::new();
    for axis in &fp.axes {
        if !seen.insert(axis.id.clone()) {
            errors.push(format!("duplicate axis id: {}", axis.id));
        }
        if !axis.score.is_finite() || !(0.0..=100.0).contains(&axis.score) {
            errors.push(format!(
                "axis {}: score must be finite in [0,100], got {}",
                axis.id, axis.score
            ));
        }
        if axis.unchanged_probes + axis.changed_probes != axis.applicable_probes {
            errors.push(format!(
                "axis {}: unchanged + changed must equal applicable",
                axis.id
            ));
        }
        let dir_sum = axis.regressions + axis.improvements + axis.neutral_changes;
        if dir_sum != axis.changed_probes {
            errors.push(format!(
                "axis {}: regressions + improvements + neutral_changes must equal changed",
                axis.id
            ));
        }
        if axis.unchanged_probes + dir_sum != axis.applicable_probes {
            errors.push(format!(
                "axis {}: unchanged + regressions + improvements + neutral_changes must equal applicable",
                axis.id
            ));
        }
        if axis.applicable_probes == 0 {
            errors.push(format!("axis {}: applicable_probes must be > 0", axis.id));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Build deterministic SVG geometry for the retention radar.
pub fn build_fingerprint_svg(fp: &BehaviourFingerprint) -> Option<FingerprintSvgModel> {
    if !fp.radar_available || fp.axes.len() < MIN_AXES_FOR_RADAR {
        return None;
    }
    let n = fp.axes.len();
    let size = 360.0;
    let center = size / 2.0;
    let max_radius = 120.0;
    let label_r = max_radius + 28.0;

    let mut grid_polygons = Vec::new();
    for pct in [20.0, 40.0, 60.0, 80.0, 100.0] {
        let r = max_radius * (pct / 100.0);
        grid_polygons.push(polygon_points(n, center, r, |_| 100.0));
    }

    let mut spokes = Vec::new();
    let mut baseline_pts = Vec::new();
    let mut candidate_pts = Vec::new();

    for (i, axis) in fp.axes.iter().enumerate() {
        let angle =
            -std::f64::consts::FRAC_PI_2 + (i as f64) * (2.0 * std::f64::consts::PI / n as f64);
        let angle_deg = angle.to_degrees();
        let (bx, by) = polar(center, max_radius, angle);
        let score_r = max_radius * (axis.score.clamp(0.0, 100.0) / 100.0);
        let (cx, cy) = polar(center, score_r, angle);
        let (lx, ly) = polar(center, label_r, angle);
        baseline_pts.push(format!("{:.2},{:.2}", bx, by));
        candidate_pts.push(format!("{:.2},{:.2}", cx, cy));
        spokes.push(FingerprintSvgSpoke {
            axis_id: axis.id.clone(),
            label: axis.label.clone(),
            score: axis.score,
            angle_deg: round2(angle_deg),
            label_x: round2(lx),
            label_y: round2(ly),
            point_x: round2(cx),
            point_y: round2(cy),
            baseline_x: round2(bx),
            baseline_y: round2(by),
            spoke_x2: round2(bx),
            spoke_y2: round2(by),
        });
    }

    Some(FingerprintSvgModel {
        size,
        center,
        max_radius,
        grid_polygons,
        spokes,
        baseline_points: baseline_pts.join(" "),
        candidate_points: candidate_pts.join(" "),
    })
}

fn polar(center: f64, radius: f64, angle: f64) -> (f64, f64) {
    (center + radius * angle.cos(), center + radius * angle.sin())
}

fn polygon_points<F>(n: usize, center: f64, radius: f64, score_fn: F) -> String
where
    F: Fn(usize) -> f64,
{
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let angle =
            -std::f64::consts::FRAC_PI_2 + (i as f64) * (2.0 * std::f64::consts::PI / n as f64);
        let r = radius * (score_fn(i).clamp(0.0, 100.0) / 100.0);
        let (x, y) = polar(center, r, angle);
        pts.push(format!("{:.2},{:.2}", x, y));
    }
    pts.join(" ")
}

/// Deterministic high-impact / stable / telemetry changelog entries for the renderer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FingerprintChangelog {
    pub high_impact: Vec<String>,
    pub stable_contracts: Vec<String>,
    pub telemetry: Vec<String>,
    pub text_summary: String,
}

pub fn build_fingerprint_changelog(
    fp: &BehaviourFingerprint,
    latency: &LatencySummary,
) -> FingerprintChangelog {
    // Ordinary axes: rank by compatibility drop, then stable AXIS_ORDER id.
    // Skip limited-confidence axes and the consistency aggregate axis (telemetry).
    let mut ranked: Vec<&FingerprintAxis> = fp
        .axes
        .iter()
        .filter(|a| {
            a.id != "consistency"
                && matches!(a.confidence, FingerprintConfidence::Sufficient)
                && a.changed_probes > 0
                && a.score < 100.0
        })
        .collect();
    ranked.sort_by(|a, b| {
        let drop_a = 100.0 - a.score;
        let drop_b = 100.0 - b.score;
        drop_b
            .partial_cmp(&drop_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });

    let mut high_impact = Vec::new();
    for axis in ranked.iter().take(5) {
        high_impact.push(changelog_axis_line(axis));
    }

    // Consistency enters high-impact only when aggregate retention falls below the
    // documented telemetry threshold (see CONSISTENCY_HIGH_IMPACT_RETENTION_BELOW).
    if let Some(c) = fp.axes.iter().find(|a| a.id == "consistency") {
        if c.score < CONSISTENCY_HIGH_IMPACT_RETENTION_BELOW && high_impact.len() < 5 {
            high_impact.push(changelog_axis_line(c));
        }
    }

    let mut stable_contracts = Vec::new();
    for axis in &fp.axes {
        if axis.id == "consistency" {
            continue;
        }
        if axis.regressions == 0 && axis.changed_probes == 0 {
            stable_contracts.push(format!(
                "{}: no material changes detected ({} probes retained)",
                axis.label, axis.unchanged_probes
            ));
        } else if axis.regressions == 0 && axis.score >= 99.99 {
            stable_contracts.push(format!(
                "{}: no regressions detected ({} probes retained)",
                axis.label, axis.unchanged_probes
            ));
        }
    }

    let mut telemetry = Vec::new();
    if latency.v1_avg_latency_ms > 0 || latency.v2_avg_latency_ms > 0 {
        if latency.delta_pct.abs() >= 10.0 {
            if latency.delta_pct < 0.0 {
                telemetry.push(format!(
                    "Average latency improved by {:.1}% ({} ms → {} ms).",
                    latency.delta_pct.abs(),
                    latency.v1_avg_latency_ms,
                    latency.v2_avg_latency_ms
                ));
            } else {
                telemetry.push(format!(
                    "Average latency increased by {:.1}% ({} ms → {} ms).",
                    latency.delta_pct, latency.v1_avg_latency_ms, latency.v2_avg_latency_ms
                ));
            }
        } else {
            telemetry.push(format!(
                "Average latency roughly unchanged ({} ms → {} ms).",
                latency.v1_avg_latency_ms, latency.v2_avg_latency_ms
            ));
        }
    }
    if let Some(c) = fp.axes.iter().find(|a| a.id == "consistency") {
        if let (Some(b), Some(cand)) = (c.baseline_raw, c.candidate_raw) {
            let abs = c.absolute_difference.unwrap_or_else(|| (cand - b).abs());
            if c.score >= CONSISTENCY_HIGH_IMPACT_RETENTION_BELOW {
                telemetry.push(format!(
                    "Repeatability remained effectively stable: {b:.2}% → {cand:.2}%, a difference of {abs:.2} percentage points (aggregate retention {:.2}%).",
                    c.score
                ));
            } else {
                telemetry.push(format!(
                    "Consistency aggregate retention {:.2}%: baseline repeatability {b:.2}% → candidate {cand:.2}% (Δ {abs:.2} pp).",
                    c.score
                ));
            }
            telemetry.push(format!(
                "{} probe{} crossed the material consistency-drift rule ({}).",
                c.changed_probes,
                if c.changed_probes == 1 { "" } else { "s" },
                direction_summary_phrase(
                    c.changed_probes,
                    c.regressions,
                    c.improvements,
                    c.neutral_changes,
                )
            ));
            let b_abs = c.baseline_inconsistent_probes.unwrap_or(0);
            let c_abs = c.candidate_inconsistent_probes.unwrap_or(0);
            telemetry.push(format!(
                "{b_abs} baseline probe{} and {c_abs} candidate probe{} were outside the absolute consistency band.",
                if b_abs == 1 { "" } else { "s" },
                if c_abs == 1 { "" } else { "s" },
            ));
        }
    }

    let text_summary = if high_impact.is_empty() {
        "Behavioural fingerprint: candidate retained baseline-equivalent behaviour across evaluated dimensions."
            .into()
    } else {
        format!(
            "Behavioural fingerprint: largest compatibility drops — {}.",
            high_impact
                .iter()
                .take(3)
                .map(|s| s.split(':').next().unwrap_or(s.as_str()))
                .collect::<Vec<_>>()
                .join("; ")
        )
    };

    FingerprintChangelog {
        high_impact,
        stable_contracts,
        telemetry,
        text_summary,
    }
}

/// Ensure fingerprint material counts reconcile with dimension summaries.
///
/// Compares `changed_probes` to [`DimensionSummary::materially_changed_probes`]
/// (for consistency: band-crossing drift; for other axes: same as historical
/// `probes_affected`). Does **not** compare aggregate consistency retention to
/// the per-probe drift ratio.
pub fn fingerprint_summary_mismatches(
    fp: &BehaviourFingerprint,
    summaries: &DimensionSummaries,
) -> Vec<String> {
    validate_fingerprint_rollups(fp, summaries)
        .err()
        .unwrap_or_default()
        .into_iter()
        .map(|e| e.to_string())
        .collect()
}

/// Validate fingerprint identity, scores, and reconciliation with summaries.
///
/// Call once after recomputing summaries and fingerprint from probe results.
/// A second recompute cannot fix a persistent semantic mismatch — callers must
/// surface [`RollupValidationError`] rather than looping.
pub fn validate_fingerprint_rollups(
    fp: &BehaviourFingerprint,
    summaries: &DimensionSummaries,
) -> Result<(), Vec<RollupValidationError>> {
    let mut errors = Vec::new();
    if let Err(msgs) = validate_fingerprint(fp) {
        for m in msgs {
            errors.push(RollupValidationError::FingerprintInvalid(m));
        }
    }
    for axis in &fp.axes {
        if !axis.score.is_finite() || !(0.0..=100.0).contains(&axis.score) {
            errors.push(RollupValidationError::InvalidFingerprintScore {
                axis: axis.id.clone(),
                score: axis.score,
            });
        }
        let dir_sum = axis.regressions + axis.improvements + axis.neutral_changes;
        if axis.unchanged_probes + dir_sum != axis.applicable_probes
            || dir_sum != axis.changed_probes
        {
            errors.push(RollupValidationError::InvalidFingerprintIdentity {
                axis: axis.id.clone(),
            });
        }
    }

    let expected = [
        ("morphology", summaries.morphology.materially_changed_probes),
        ("tone", summaries.tone.materially_changed_probes),
        ("factual", summaries.factual.materially_changed_probes),
        ("schema", summaries.schema.materially_changed_probes),
        (
            "instruction",
            summaries.instruction.materially_changed_probes,
        ),
        ("refusal", summaries.refusal.materially_changed_probes),
        ("claim", summaries.claim.materially_changed_probes),
        ("semantic", summaries.semantic.materially_changed_probes),
        (
            "consistency",
            summaries.consistency.materially_changed_probes,
        ),
    ];
    for (id, material) in expected {
        match fp.axes.iter().find(|a| a.id == id) {
            Some(axis) if axis.changed_probes != material => {
                errors.push(RollupValidationError::FingerprintSummaryMismatch {
                    axis: id.into(),
                    fingerprint: axis.changed_probes,
                    summary: material,
                });
            }
            None if material > 0 => {
                errors.push(RollupValidationError::FingerprintSummaryMismatch {
                    axis: id.into(),
                    fingerprint: 0,
                    summary: material,
                });
            }
            _ => {}
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn morph_metrics() -> MorphologyMetrics {
        MorphologyMetrics {
            token_count: 1,
            word_count: 1,
            sentence_count: 1,
            paragraph_count: 1,
            has_lists: false,
            has_headers: false,
            has_code_blocks: false,
            has_caveats: false,
            response_type: ResponseType::SingleLine,
        }
    }

    fn tone_metrics() -> ToneMetrics {
        ToneMetrics {
            formality_score: 0.5,
            assertiveness_score: 0.5,
            hedge_word_count: 0,
            contraction_count: 0,
            average_sentence_length: 10.0,
            passive_voice_ratio: 0.0,
        }
    }

    fn green_dims() -> ProbeDimensions {
        ProbeDimensions {
            morphology: MorphologyDiff {
                risk: RiskLevel::Green,
                direction: DriftDirection::Neutral,
                v1: morph_metrics(),
                v2: morph_metrics(),
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
                v1: tone_metrics(),
                v2: tone_metrics(),
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
                cosine_similarity: Some(1.0),
                semantic_scoring_disabled: false,
                disabled_reason: None,
                flagged_for_review: false,
                similarity_threshold: 0.85,
            },
            claim: ClaimDiff {
                risk: RiskLevel::Green,
                direction: DriftDirection::Neutral,
                preservation_score: 1.0,
                preservation_threshold: 0.7,
                material_preservation_score: 1.0,
                ..Default::default()
            },
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

    fn probe_result(dims: ProbeDimensions) -> ProbeResult {
        ProbeResult {
            probe: Probe {
                id: Uuid::new_v4(),
                name: "p".into(),
                category: ProbeCategory::Semantic,
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
                claim_anchor_policy: ClaimAnchorPolicy::default(),
                presentation_drift: PresentationDriftPolicy::default(),
                latency_slo_ms: None,
            },
            v1_content: "a".into(),
            v2_content: "b".into(),
            overall_risk: RiskLevel::Green,
            overall_direction: DriftDirection::Neutral,
            drift_category: DriftCategory::NoSignificantDrift,
            drift_severity: DriftSeverity::Informational,
            drift_impact: DriftImpact::Informational,
            dimensions: dims,
            notes: vec![],
        }
    }

    fn models() -> (ModelInfo, ModelInfo) {
        (
            ModelInfo {
                label: "v1".into(),
                model_id: "base".into(),
                adapter: "openai".into(),
                endpoint: String::new(),
            },
            ModelInfo {
                label: "v2".into(),
                model_id: "cand".into(),
                adapter: "openai".into(),
                endpoint: String::new(),
            },
        )
    }

    #[test]
    fn all_unchanged_scores_100() {
        let (v1, v2) = models();
        let results = vec![probe_result(green_dims()), probe_result(green_dims())];
        let fp = compute_behaviour_fingerprint(&results, &v1, &v2);
        assert!(fp.radar_available);
        for axis in &fp.axes {
            assert_eq!(axis.score, 100.0, "{}", axis.id);
            assert_eq!(axis.changed_probes, 0);
        }
        validate_fingerprint(&fp).unwrap();
    }

    #[test]
    fn all_changed_scores_0() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.morphology.risk = RiskLevel::Red;
        dims.morphology.direction = DriftDirection::Regression;
        dims.morphology.delta.structure_changed = true;
        dims.tone.risk = RiskLevel::Amber;
        dims.tone.direction = DriftDirection::Improvement;
        dims.tone.delta.significant_shift = true;
        dims.refusal.risk = RiskLevel::Amber;
        dims.refusal.direction = DriftDirection::Neutral;
        dims.refusal.new_refusal = true;
        dims.refusal.v2_refused = true;
        dims.semantic.risk = RiskLevel::Red;
        dims.semantic.direction = DriftDirection::Regression;
        dims.semantic.cosine_similarity = Some(0.4);
        dims.semantic.flagged_for_review = true;
        dims.claim.risk = RiskLevel::Amber;
        dims.claim.direction = DriftDirection::Neutral;
        dims.claim.preservation_score = 0.5;
        dims.claim.material_preservation_score = 0.5;
        let results = vec![probe_result(dims)];
        let fp = compute_behaviour_fingerprint(&results, &v1, &v2);
        for axis in &fp.axes {
            if matches!(
                axis.id.as_str(),
                "morphology" | "tone" | "refusal" | "semantic" | "claim"
            ) {
                assert_eq!(axis.score, 0.0, "{}", axis.id);
            }
        }
        let morph = fp.axes.iter().find(|a| a.id == "morphology").unwrap();
        assert_eq!(morph.regressions, 1);
        let tone = fp.axes.iter().find(|a| a.id == "tone").unwrap();
        assert_eq!(tone.improvements, 1);
        let refusal = fp.axes.iter().find(|a| a.id == "refusal").unwrap();
        assert_eq!(refusal.neutral_changes, 1);
        validate_fingerprint(&fp).unwrap();
    }

    #[test]
    fn improvement_counts_as_drift() {
        let (v1, v2) = models();
        let d1 = green_dims();
        let mut d2 = green_dims();
        d2.morphology.risk = RiskLevel::Amber;
        d2.morphology.direction = DriftDirection::Improvement;
        d2.morphology.delta.structure_changed = true;
        let fp = compute_behaviour_fingerprint(&[probe_result(d1), probe_result(d2)], &v1, &v2);
        let morph = fp.axes.iter().find(|a| a.id == "morphology").unwrap();
        assert_eq!(morph.score, 50.0);
        assert_eq!(morph.improvements, 1);
        assert_eq!(morph.changed_probes, 1);
    }

    #[test]
    fn green_plus_unchanged_retains_fully() {
        let (v1, v2) = models();
        let dims = green_dims();
        assert!(!morphology_materially_changed(&dims.morphology));
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let morph = fp.axes.iter().find(|a| a.id == "morphology").unwrap();
        assert_eq!(morph.score, 100.0);
        assert_eq!(morph.unchanged_probes, 1);
        assert_eq!(morph.changed_probes, 0);
    }

    #[test]
    fn one_token_difference_does_not_reduce_morphology_retention() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.morphology.delta.token_delta = 1;
        dims.morphology.delta.token_delta_pct = 0.1; // << 0.5 amber
        dims.morphology.risk = RiskLevel::Green;
        assert!(!morphology_materially_changed(&dims.morphology));
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let morph = fp.axes.iter().find(|a| a.id == "morphology").unwrap();
        assert_eq!(morph.score, 100.0);
        assert_eq!(morph.changed_probes, 0);
    }

    #[test]
    fn morphology_structure_change_reduces_retention() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.morphology.delta.structure_changed = true;
        dims.morphology.risk = RiskLevel::Amber;
        dims.morphology.direction = DriftDirection::Neutral;
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let morph = fp.axes.iter().find(|a| a.id == "morphology").unwrap();
        assert_eq!(morph.score, 0.0);
        assert_eq!(morph.neutral_changes, 1);
    }

    #[test]
    fn green_risk_with_material_morphology_still_reduces_retention() {
        // Risk colour and materiality are independent: Green can still be material
        // if stored deltas cross the band (e.g. unusual persisted state).
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.morphology.risk = RiskLevel::Green;
        dims.morphology.direction = DriftDirection::Improvement;
        dims.morphology.delta.structure_changed = true;
        assert!(morphology_materially_changed(&dims.morphology));
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let morph = fp.axes.iter().find(|a| a.id == "morphology").unwrap();
        assert_eq!(morph.score, 0.0);
        assert_eq!(morph.improvements, 1);
        assert!(matches!(morph.risk, RiskLevel::Green));
    }

    #[test]
    fn improvement_direction_without_material_does_not_reduce_retention() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.morphology.risk = RiskLevel::Green;
        dims.morphology.direction = DriftDirection::Improvement;
        // No material morphology signals.
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let morph = fp.axes.iter().find(|a| a.id == "morphology").unwrap();
        assert_eq!(morph.score, 100.0);
        assert_eq!(morph.improvements, 0);
    }

    #[test]
    fn tone_hedge_below_band_does_not_reduce_retention() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.tone.delta.hedge_word_delta = 2; // significant_shift needs |hedge| >= 4
        dims.tone.delta.significant_shift = false;
        dims.tone.risk = RiskLevel::Green;
        assert!(!tone_materially_changed(&dims.tone));
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let tone = fp.axes.iter().find(|a| a.id == "tone").unwrap();
        assert_eq!(tone.score, 100.0);
    }

    #[test]
    fn tone_significant_shift_reduces_retention() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.tone.delta.significant_shift = true;
        dims.tone.risk = RiskLevel::Amber;
        dims.tone.direction = DriftDirection::Neutral;
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let tone = fp.axes.iter().find(|a| a.id == "tone").unwrap();
        assert_eq!(tone.score, 0.0);
        assert_eq!(tone.neutral_changes, 1);
    }

    #[test]
    fn semantic_near_identical_below_one_does_not_count() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.semantic.risk = RiskLevel::Green;
        dims.semantic.cosine_similarity = Some(0.95);
        dims.semantic.similarity_threshold = 0.85;
        dims.semantic.flagged_for_review = false;
        assert!(!semantic_materially_changed(&dims.semantic));
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let sem = fp.axes.iter().find(|a| a.id == "semantic").unwrap();
        assert_eq!(sem.score, 100.0);
    }

    #[test]
    fn semantic_below_threshold_reduces_retention() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.semantic.risk = RiskLevel::Amber;
        dims.semantic.cosine_similarity = Some(0.80);
        dims.semantic.similarity_threshold = 0.85;
        dims.semantic.flagged_for_review = true;
        dims.semantic.direction = DriftDirection::Neutral;
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let sem = fp.axes.iter().find(|a| a.id == "semantic").unwrap();
        assert_eq!(sem.score, 0.0);
        assert_eq!(sem.neutral_changes, 1);
    }

    #[test]
    fn amber_and_red_regression_same_retention() {
        let (v1, v2) = models();
        let mut amber = green_dims();
        amber.morphology.risk = RiskLevel::Amber;
        amber.morphology.direction = DriftDirection::Regression;
        amber.morphology.delta.structure_changed = true;
        let mut red = amber.clone();
        red.morphology.risk = RiskLevel::Red;
        let fp_a = compute_behaviour_fingerprint(&[probe_result(amber)], &v1, &v2);
        let fp_r = compute_behaviour_fingerprint(&[probe_result(red)], &v1, &v2);
        let a = fp_a.axes.iter().find(|x| x.id == "morphology").unwrap();
        let r = fp_r.axes.iter().find(|x| x.id == "morphology").unwrap();
        assert_eq!(a.score, r.score);
        assert_eq!(a.regressions, r.regressions);
        assert_eq!(a.score, 0.0);
    }

    #[test]
    fn risk_severity_alone_does_not_alter_retention() {
        let (v1, v2) = models();
        // Same material morphology (structure); only risk colour differs.
        let mut amber = green_dims();
        amber.morphology.risk = RiskLevel::Amber;
        amber.morphology.direction = DriftDirection::Neutral;
        amber.morphology.delta.structure_changed = true;
        amber.morphology.delta.token_delta_pct = 0.6;
        let mut red = amber.clone();
        red.morphology.risk = RiskLevel::Red;
        let scores: Vec<f64> = [amber, red]
            .into_iter()
            .map(|d| {
                let fp = compute_behaviour_fingerprint(&[probe_result(d)], &v1, &v2);
                fp.axes.iter().find(|a| a.id == "morphology").unwrap().score
            })
            .collect();
        assert_eq!(scores[0], scores[1]);
        assert_eq!(scores[0], 0.0);
    }

    #[test]
    fn mixed_risk_and_direction_states() {
        let (v1, v2) = models();
        let unchanged = green_dims();
        let mut green_imp = green_dims();
        green_imp.morphology.risk = RiskLevel::Green;
        green_imp.morphology.direction = DriftDirection::Improvement;
        green_imp.morphology.delta.structure_changed = true;
        let mut amber_reg = green_dims();
        amber_reg.morphology.risk = RiskLevel::Amber;
        amber_reg.morphology.direction = DriftDirection::Regression;
        amber_reg.morphology.delta.structure_changed = true;
        let mut red_reg = green_dims();
        red_reg.morphology.risk = RiskLevel::Red;
        red_reg.morphology.direction = DriftDirection::Regression;
        red_reg.morphology.delta.token_delta_pct = 1.0;
        let mut green_neu = green_dims();
        green_neu.morphology.risk = RiskLevel::Green;
        green_neu.morphology.direction = DriftDirection::Neutral;
        green_neu.morphology.delta.token_delta = 2;
        green_neu.morphology.delta.token_delta_pct = 0.1; // immaterial
        let fp = compute_behaviour_fingerprint(
            &[
                probe_result(unchanged),
                probe_result(green_imp),
                probe_result(amber_reg),
                probe_result(red_reg),
                probe_result(green_neu),
            ],
            &v1,
            &v2,
        );
        let morph = fp.axes.iter().find(|a| a.id == "morphology").unwrap();
        assert_eq!(morph.applicable_probes, 5);
        assert_eq!(morph.unchanged_probes, 2); // unchanged + immaterial green_neu
        assert_eq!(morph.improvements, 1);
        assert_eq!(morph.regressions, 2);
        assert_eq!(morph.neutral_changes, 0);
        assert_eq!(morph.score, 40.0);
        assert!(matches!(morph.risk, RiskLevel::Red));
    }

    #[test]
    fn consistency_raw_variance_diff_not_material_when_both_consistent() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.consistency = Some(ConsistencyDiff {
            risk: RiskLevel::Green,
            direction: DriftDirection::Neutral,
            v1_runs: 3,
            v2_runs: 3,
            v1_variance: 0.01,
            v2_variance: 0.05,
            v1_consistent: true,
            v2_consistent: true,
            consistency_regression: false,
            consistency_improvement: false,
        });
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let c = fp.axes.iter().find(|a| a.id == "consistency").unwrap();
        assert_eq!(c.changed_probes, 0);
        assert_eq!(c.raw_changed_probes, Some(1));
        assert_eq!(
            c.aggregation_kind,
            FingerprintAggregationKind::AggregateSimilarity
        );
        assert!(
            (c.score
                - consistency_retention(
                    variance_to_repeatability(0.01),
                    variance_to_repeatability(0.05)
                ))
            .abs()
                < 0.01
        );
    }

    #[test]
    fn consistency_material_when_candidate_loose() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.consistency = Some(ConsistencyDiff {
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
        });
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let c = fp.axes.iter().find(|a| a.id == "consistency").unwrap();
        assert_eq!(c.changed_probes, 1);
        assert_eq!(c.regressions, 1);
        assert_eq!(c.materially_changed_probes, Some(1));
        assert_eq!(c.baseline_inconsistent_probes, Some(0));
        assert_eq!(c.candidate_inconsistent_probes, Some(1));
    }

    #[test]
    fn consistency_both_inconsistent_equal_does_not_reduce_retention() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.consistency = Some(ConsistencyDiff {
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
        });
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let c = fp.axes.iter().find(|a| a.id == "consistency").unwrap();
        assert_eq!(c.changed_probes, 0);
        assert_eq!(c.unchanged_probes, 1);
        assert_eq!(c.score, 100.0);
        assert_eq!(c.baseline_inconsistent_probes, Some(1));
        assert_eq!(c.candidate_inconsistent_probes, Some(1));
        assert!(matches!(c.risk, RiskLevel::Amber));
    }

    #[test]
    fn consistency_aggregate_retention_independent_of_drift_count() {
        let ret = consistency_retention(
            variance_to_repeatability(0.0665),
            variance_to_repeatability(0.0760),
        );
        assert!((ret - 99.05).abs() < 1e-9, "ret={ret}");
    }

    #[test]
    fn consistency_99_pct_not_in_high_impact_changelog() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        // Aggregate ~99% retention; both inconsistent same band → no drift.
        dims.consistency = Some(ConsistencyDiff {
            risk: RiskLevel::Amber,
            direction: DriftDirection::Neutral,
            v1_runs: 3,
            v2_runs: 3,
            v1_variance: 0.0665, // ~93.35%
            v2_variance: 0.0760, // ~92.40%
            v1_consistent: false,
            v2_consistent: false,
            consistency_regression: false,
            consistency_improvement: false,
        });
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let c = fp.axes.iter().find(|a| a.id == "consistency").unwrap();
        assert!(c.score >= CONSISTENCY_HIGH_IMPACT_RETENTION_BELOW);
        assert_eq!(c.changed_probes, 0);
        let lat = LatencySummary {
            v1_avg_latency_ms: 1000,
            v2_avg_latency_ms: 1000,
            delta_ms: 0,
            delta_pct: 0.0,
            direction: DriftDirection::Neutral,
            note: String::new(),
        };
        let log = build_fingerprint_changelog(&fp, &lat);
        assert!(
            !log.high_impact.iter().any(|s| s.contains("Consistency")),
            "high_impact={:?}",
            log.high_impact
        );
        assert!(log
            .telemetry
            .iter()
            .any(|s| s.contains("effectively stable")));
        assert!(log
            .telemetry
            .iter()
            .any(|s| s.contains("absolute consistency band")));
        assert!(!log.high_impact.iter().any(|s| s.contains("including 0")));
    }

    #[test]
    fn rollup_validation_rejects_identity_break() {
        let mut fp = BehaviourFingerprint {
            radar_available: true,
            ..Default::default()
        };
        fp.axes.push(FingerprintAxis {
            id: "morphology".into(),
            label: "Morphology".into(),
            score: 50.0,
            applicable_probes: 2,
            unchanged_probes: 1,
            changed_probes: 1,
            regressions: 0,
            improvements: 0,
            neutral_changes: 0,
            risk: RiskLevel::Amber,
            confidence: FingerprintConfidence::Sufficient,
            evidence_probe_ids: vec![],
            explanation: String::new(),
            aggregation_kind: FingerprintAggregationKind::ProbeRetention,
            materially_changed_probes: None,
            raw_changed_probes: None,
            baseline_inconsistent_probes: None,
            candidate_inconsistent_probes: None,
            baseline_raw: None,
            candidate_raw: None,
            absolute_difference: None,
            raw_unit: None,
        });
        let summaries = DimensionSummaries::default();
        let err = validate_fingerprint_rollups(&fp, &summaries).unwrap_err();
        assert!(err
            .iter()
            .any(|e| matches!(e, RollupValidationError::InvalidFingerprintIdentity { .. })));
    }

    #[test]
    fn rollup_validation_detects_summary_mismatch_without_loop() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.morphology.delta.structure_changed = true;
        dims.morphology.risk = RiskLevel::Amber;
        let results = vec![
            probe_result(dims.clone()),
            probe_result(dims.clone()),
            probe_result(dims),
        ];
        let fp = compute_behaviour_fingerprint(&results, &v1, &v2);
        let mut summaries = DimensionSummaries::default();
        summaries.morphology.materially_changed_probes = 99;
        let err = validate_fingerprint_rollups(&fp, &summaries).unwrap_err();
        assert!(err.iter().any(|e| matches!(
            e,
            RollupValidationError::FingerprintSummaryMismatch { axis, .. } if axis == "morphology"
        )));
        let err2 = validate_fingerprint_rollups(&fp, &summaries).unwrap_err();
        assert_eq!(err, err2);
    }

    #[test]
    fn changelog_omits_zero_regression_wording() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.morphology.delta.structure_changed = true;
        dims.morphology.risk = RiskLevel::Amber;
        dims.morphology.direction = DriftDirection::Neutral;
        // Enough probes for Sufficient confidence.
        let results = vec![
            probe_result(dims.clone()),
            probe_result(dims.clone()),
            probe_result(dims),
        ];
        let fp = compute_behaviour_fingerprint(&results, &v1, &v2);
        let lat = LatencySummary {
            v1_avg_latency_ms: 0,
            v2_avg_latency_ms: 0,
            delta_ms: 0,
            delta_pct: 0.0,
            direction: DriftDirection::Neutral,
            note: String::new(),
        };
        let log = build_fingerprint_changelog(&fp, &lat);
        assert!(!log.high_impact.is_empty());
        for line in &log.high_impact {
            assert!(!line.contains("including 0"));
            assert!(!line.contains("0 regressions"));
        }
        assert!(log.high_impact[0].contains("neutral"));
    }

    #[test]
    fn factual_absent_omits_axis() {
        let (v1, v2) = models();
        let fp = compute_behaviour_fingerprint(&[probe_result(green_dims())], &v1, &v2);
        assert!(!fp.axes.iter().any(|a| a.id == "factual"));
        assert!(fp.partial);
        let omitted = fp
            .omitted_axes
            .iter()
            .find(|o| o.id == "factual")
            .expect("factual omission reason");
        assert!(matches!(
            omitted.reason,
            OmittedAxisReason::NoApplicableProbes
        ));
        assert!(omitted.detail.contains("known answer"));
    }

    #[test]
    fn semantic_disabled_omits_axis() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.semantic.semantic_scoring_disabled = true;
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        assert!(!fp.axes.iter().any(|a| a.id == "semantic"));
        let omitted = fp
            .omitted_axes
            .iter()
            .find(|o| o.id == "semantic")
            .expect("semantic omission reason");
        assert!(matches!(omitted.reason, OmittedAxisReason::Disabled));
    }

    #[test]
    fn schema_absent_has_explicit_omission() {
        let (v1, v2) = models();
        let fp = compute_behaviour_fingerprint(&[probe_result(green_dims())], &v1, &v2);
        assert!(!fp.axes.iter().any(|a| a.id == "schema"));
        let omitted = fp
            .omitted_axes
            .iter()
            .find(|o| o.id == "schema")
            .expect("schema omission");
        assert!(matches!(
            omitted.reason,
            OmittedAxisReason::NoApplicableProbes
        ));
        assert!(omitted.detail.contains("expected schema"));
    }

    #[test]
    fn consistency_identical() {
        assert_eq!(consistency_retention(90.0, 90.0), 100.0);
        assert_eq!(variance_to_repeatability(0.0), 100.0);
    }

    #[test]
    fn consistency_candidate_more_variable() {
        // variance 0.2 → 80%, variance 0.5 → 50%, retention = 100 - 30 = 70
        let b = variance_to_repeatability(0.2);
        let c = variance_to_repeatability(0.5);
        assert_eq!(b, 80.0);
        assert_eq!(c, 50.0);
        assert_eq!(consistency_retention(b, c), 70.0);
    }

    #[test]
    fn consistency_candidate_less_variable() {
        let b = variance_to_repeatability(0.4);
        let c = variance_to_repeatability(0.1);
        assert_eq!(consistency_retention(b, c), 70.0);
    }

    #[test]
    fn consistency_bounds_and_non_finite() {
        assert_eq!(consistency_retention(-10.0, 150.0), 0.0);
        assert_eq!(consistency_retention(f64::NAN, 50.0), 0.0);
        assert_eq!(variance_to_repeatability(f64::INFINITY), 0.0);
        assert_eq!(variance_to_repeatability(2.0), 0.0);
    }

    #[test]
    fn consistency_axis_from_probes() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.consistency = Some(ConsistencyDiff {
            risk: RiskLevel::Green,
            direction: DriftDirection::Neutral,
            v1_runs: 3,
            v2_runs: 3,
            v1_variance: 0.0,
            v2_variance: 0.0,
            v1_consistent: true,
            v2_consistent: true,
            consistency_regression: false,
            consistency_improvement: false,
        });
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let c = fp.axes.iter().find(|a| a.id == "consistency").unwrap();
        assert_eq!(c.label, "Consistency retention");
        assert_eq!(c.score, 100.0);
        assert_eq!(c.baseline_raw, Some(100.0));
        assert_eq!(c.candidate_raw, Some(100.0));
        assert_eq!(c.absolute_difference, Some(0.0));
        assert!(c.explanation.contains("Aggregate consistency similarity"));
        assert!(c.explanation.contains("absolute inconsistency"));
    }

    #[test]
    fn missing_consistency_omits_axis() {
        let (v1, v2) = models();
        let fp = compute_behaviour_fingerprint(&[probe_result(green_dims())], &v1, &v2);
        assert!(!fp.axes.iter().any(|a| a.id == "consistency"));
        let omitted = fp
            .omitted_axes
            .iter()
            .find(|o| o.id == "consistency")
            .expect("consistency omission");
        assert!(matches!(omitted.reason, OmittedAxisReason::Unavailable));
    }

    #[test]
    fn zero_consistency_runs_omits() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.consistency = Some(ConsistencyDiff {
            risk: RiskLevel::Green,
            direction: DriftDirection::Neutral,
            v1_runs: 1,
            v2_runs: 1,
            v1_variance: 0.0,
            v2_variance: 0.0,
            v1_consistent: true,
            v2_consistent: true,
            consistency_regression: false,
            consistency_improvement: false,
        });
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        assert!(!fp.axes.iter().any(|a| a.id == "consistency"));
    }

    #[test]
    fn axis_order_stable() {
        let (v1, v2) = models();
        let mut dims = green_dims();
        dims.factual = Some(FactualDiff {
            risk: RiskLevel::Green,
            direction: DriftDirection::Neutral,
            v1_correct: true,
            v2_correct: true,
            v1_answer_extract: "x".into(),
            v2_answer_extract: "x".into(),
            regression: false,
            improvement: false,
        });
        dims.consistency = Some(ConsistencyDiff {
            risk: RiskLevel::Green,
            direction: DriftDirection::Neutral,
            v1_runs: 3,
            v2_runs: 3,
            v1_variance: 0.1,
            v2_variance: 0.1,
            v1_consistent: true,
            v2_consistent: true,
            consistency_regression: false,
            consistency_improvement: false,
        });
        let fp = compute_behaviour_fingerprint(&[probe_result(dims)], &v1, &v2);
        let ids: Vec<_> = fp.axes.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "morphology",
                "tone",
                "factual",
                "refusal",
                "consistency",
                "claim",
                "semantic"
            ]
        );
    }

    #[test]
    fn svg_requires_three_axes() {
        let fp = BehaviourFingerprint {
            axes: vec![],
            radar_available: false,
            ..Default::default()
        };
        assert!(build_fingerprint_svg(&fp).is_none());
    }

    #[test]
    fn changelog_deterministic() {
        let (v1, v2) = models();
        let mut d = green_dims();
        d.semantic.risk = RiskLevel::Amber;
        d.semantic.direction = DriftDirection::Regression;
        d.semantic.cosine_similarity = Some(0.5);
        d.semantic.flagged_for_review = true;
        let results = vec![
            probe_result(d.clone()),
            probe_result(d.clone()),
            probe_result(d),
        ];
        let fp = compute_behaviour_fingerprint(&results, &v1, &v2);
        let lat = LatencySummary {
            v1_avg_latency_ms: 1000,
            v2_avg_latency_ms: 750,
            delta_ms: -250,
            delta_pct: -25.0,
            direction: DriftDirection::Improvement,
            note: "faster".into(),
        };
        let c1 = build_fingerprint_changelog(&fp, &lat);
        let c2 = build_fingerprint_changelog(&fp, &lat);
        assert_eq!(c1, c2);
        assert!(!c1.high_impact.is_empty());
        assert!(c1.telemetry.iter().any(|t| t.contains("25")));
        assert!(!c1.high_impact.iter().any(|s| s.contains("including 0")));
    }

    #[test]
    fn score_bounds_no_nan() {
        assert_eq!(retention_score(0, 0), 0.0);
        assert_eq!(retention_score(3, 3), 100.0);
        assert_eq!(retention_score(1, 3), 33.33);
    }
}
