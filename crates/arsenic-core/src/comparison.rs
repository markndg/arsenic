use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::claim::{ClaimExtractor, ClaimMatcher};
use crate::embedding::weighted_sentence_similarity;
use crate::morphology::MorphologyAnalyser;
use crate::refusal::RefusalDetector;
use crate::semantic::SemanticAnalyser;
use crate::tone::ToneAnalyser;
use crate::types::*;

#[derive(Debug, Clone)]
pub struct RiskThresholds {
    pub morphology_token_delta_amber: f64,
    pub morphology_token_delta_red: f64,
    pub tone_formality_delta_amber: f64,
    pub tone_formality_delta_red: f64,
    pub semantic_similarity_amber: f64,
    pub semantic_similarity_red: f64,
    /// Mean pairwise response distance (1 − cosine on hash embeddings) below this counts as
    /// "consistent" across runs. Default `0.12`. Above this on v1 or v2 → at least Amber risk.
    pub consistency_variance_threshold: f64,
}

impl Default for RiskThresholds {
    fn default() -> Self {
        Self {
            morphology_token_delta_amber: 0.5,
            morphology_token_delta_red: 1.0,
            tone_formality_delta_amber: 0.15,
            tone_formality_delta_red: 0.30,
            semantic_similarity_amber: 0.85,
            semantic_similarity_red: 0.70,
            consistency_variance_threshold: 0.12,
        }
    }
}

/// Below this v1 latency (ms), %-based slowdowns are often measurement noise on tiny baselines.
const LATENCY_PCT_FLOOR_MS: u64 = 80;
const LATENCY_SMALL_BASELINE_RED_MS: i64 = 280;
const LATENCY_SMALL_BASELINE_AMBER_MS: i64 = 120;
const LATENCY_SMALL_BASELINE_REG_DIR_MS: i64 = 72;
const LATENCY_SMALL_BASELINE_IMP_DIR_MS: i64 = -48;

fn compute_latency_diff(v1: u64, v2: u64) -> LatencyDiff {
    let delta_ms = v2 as i64 - v1 as i64;
    let delta_pct = if v1 == 0 {
        if v2 == 0 {
            0.0
        } else {
            1.0
        }
    } else {
        delta_ms as f64 / v1 as f64
    };
    let (risk, direction) = if v1 < LATENCY_PCT_FLOOR_MS {
        let risk = if delta_ms > LATENCY_SMALL_BASELINE_RED_MS {
            RiskLevel::Red
        } else if delta_ms > LATENCY_SMALL_BASELINE_AMBER_MS {
            RiskLevel::Amber
        } else {
            RiskLevel::Green
        };
        let direction = if delta_ms < LATENCY_SMALL_BASELINE_IMP_DIR_MS {
            DriftDirection::Improvement
        } else if delta_ms > LATENCY_SMALL_BASELINE_REG_DIR_MS {
            DriftDirection::Regression
        } else {
            DriftDirection::Neutral
        };
        (risk, direction)
    } else {
        let risk = if delta_pct > 1.0 {
            RiskLevel::Red
        } else if delta_pct > 0.5 {
            RiskLevel::Amber
        } else {
            RiskLevel::Green
        };
        let direction = if delta_pct < -0.2 {
            DriftDirection::Improvement
        } else if delta_pct > 0.1 {
            DriftDirection::Regression
        } else {
            DriftDirection::Neutral
        };
        (risk, direction)
    };
    LatencyDiff {
        risk,
        direction,
        v1_latency_ms: v1,
        v2_latency_ms: v2,
        delta_ms,
        delta_pct,
    }
}

pub struct ComparisonEngine {
    pub semantic_analyser: SemanticAnalyser,
    pub semantic_threshold: f64,
    pub risk_thresholds: RiskThresholds,
    pub claim_matcher: ClaimMatcher,
}

impl ComparisonEngine {
    pub fn new(
        semantic_enabled: bool,
        semantic_threshold: f64,
        risk_thresholds: RiskThresholds,
    ) -> Self {
        Self {
            semantic_analyser: SemanticAnalyser::new(semantic_enabled),
            semantic_threshold,
            risk_thresholds,
            // ClaimMatcher uses hash embeddings today (same as `weighted_sentence_similarity`), not BGE.
            // Relaxed cutoffs avoid ~0.71 rephrases landing in "drift" vs "match"; switch to `true` when
            // claim lines use BGE (then align with `semantic_enabled` or a dedicated flag).
            claim_matcher: ClaimMatcher::for_embedding_tier(false),
        }
    }

    /// Compare a single probe pair (used by the CLI mutation validation loop).
    pub fn compare_one(&self, pair: ResponsePair) -> anyhow::Result<ProbeResult> {
        self.compare_pair(pair)
    }

    pub fn compare(
        &self,
        run_id: Uuid,
        pairs: Vec<ResponsePair>,
        v1_model: ModelInfo,
        v2_model: ModelInfo,
    ) -> anyhow::Result<DriftReport> {
        let mut probe_results = Vec::with_capacity(pairs.len());
        for pair in pairs {
            probe_results.push(self.compare_pair(pair)?);
        }
        let dimension_summaries = self.compute_dimension_summaries(&probe_results);
        let overall_risk = self.compute_overall_risk(&probe_results);
        let valence_summary = valence_counts(&probe_results);
        let summary = self.build_summary(&probe_results, &overall_risk, &valence_summary);
        let upgrade_path = build_upgrade_path(&probe_results, &[]);
        let latency_summary = compute_latency_summary(&probe_results);
        let migration_profile = compute_migration_profile(&probe_results, &latency_summary);
        Ok(DriftReport {
            run_id,
            generated_at: chrono::Utc::now(),
            v1_model,
            v2_model,
            overall_risk,
            summary,
            probe_results,
            dimension_summaries,
            valence_summary,
            upgrade_path,
            mutation_results: Vec::new(),
            latency_summary,
            migration_profile,
        })
    }

    /// Attach mutation results and rebuild upgrade path (call from CLI after `--mutate`).
    pub fn attach_mutations(report: &mut DriftReport, mutations: Vec<MutationResult>) {
        report.mutation_results = mutations;
        report.upgrade_path = build_upgrade_path(&report.probe_results, &report.mutation_results);
        report.summary.auto_remediation_candidates = report
            .mutation_results
            .iter()
            .filter(|m| m.validated)
            .count();
        report.sync_valence_from_probe_results();
    }

    fn compare_pair(&self, pair: ResponsePair) -> anyhow::Result<ProbeResult> {
        let mut notes = Vec::new();
        if matches!(pair.v1.finish_reason, FinishReason::Error) {
            notes.push(format!("v1 probe error: {}", pair.v1.content));
        }
        if matches!(pair.v2.finish_reason, FinishReason::Error) {
            notes.push(format!("v2 probe error: {}", pair.v2.content));
        }

        let morphology = self.compare_morphology(&pair);
        let tone = self.compare_tone(&pair);
        let factual = self.compare_factual(&pair);
        let schema = self.compare_schema(&pair);
        let instruction = self.compare_instruction(&pair);
        let refusal = self.compare_refusal(&pair);
        let claim = self.compare_claim(&pair)?;
        let semantic = self.compare_semantic(&pair, &claim)?;
        let latency = self.compare_latency(&pair);
        let consistency = self.compare_consistency(&pair);
        let custom_assertions = self.compare_custom(&pair);

        let dimensions = ProbeDimensions {
            morphology,
            tone,
            factual,
            schema,
            instruction,
            refusal,
            semantic,
            claim,
            latency,
            consistency,
            custom_assertions,
        };
        let (overall_risk, drift_category, drift_severity) =
            compute_probe_risk(&pair.probe, &dimensions);
        let overall_direction = probe_overall_direction(&dimensions, &overall_risk);
        Ok(ProbeResult {
            probe: pair.probe,
            v1_content: pair.v1.content.clone(),
            v2_content: pair.v2.content.clone(),
            overall_risk,
            overall_direction,
            drift_category,
            drift_severity,
            dimensions,
            notes,
        })
    }

    fn compare_claim(&self, pair: &ResponsePair) -> anyhow::Result<ClaimDiff> {
        // Claims are extracted from assistant message text only (`ModelResponse.content`), never
        // from `raw` JSON or adapter endpoint strings.
        let c1 = ClaimExtractor::extract(&pair.v1.content);
        let c2 = ClaimExtractor::extract(&pair.v2.content);
        self.claim_matcher
            .match_claims(c1, c2, pair.probe.category)
    }

    fn compare_morphology(&self, pair: &ResponsePair) -> MorphologyDiff {
        let mut m1 = MorphologyAnalyser::analyse(&pair.v1.content, pair.v1.token_count);
        let mut m2 = MorphologyAnalyser::analyse(&pair.v2.content, pair.v2.token_count);
        if RefusalDetector::is_refusal(&pair.v1) {
            m1.response_type = ResponseType::Refusal;
        }
        if RefusalDetector::is_refusal(&pair.v2) {
            m2.response_type = ResponseType::Refusal;
        }
        let token_delta = m2.token_count as i64 - m1.token_count as i64;
        let token_delta_pct = if m1.token_count == 0 {
            if m2.token_count == 0 {
                0.0
            } else {
                1.0
            }
        } else {
            token_delta.unsigned_abs() as f64 / m1.token_count as f64
        };
        let response_type_changed = m1.response_type != m2.response_type;
        let structure_changed = m1.has_lists != m2.has_lists
            || m1.has_headers != m2.has_headers
            || m1.has_code_blocks != m2.has_code_blocks
            || m1.paragraph_count != m2.paragraph_count;
        let t = &self.risk_thresholds;
        let risk = morphology_risk_level(
            token_delta_pct,
            response_type_changed,
            structure_changed,
            t.morphology_token_delta_amber,
            t.morphology_token_delta_red,
        );
        let direction = direction_morphology(&pair.probe, m1.word_count, m2.word_count, response_type_changed);
        MorphologyDiff {
            risk,
            direction,
            v1: m1,
            v2: m2,
            delta: MorphologyDelta {
                token_delta,
                token_delta_pct,
                response_type_changed,
                structure_changed,
            },
        }
    }

    fn compare_tone(&self, pair: &ResponsePair) -> ToneDiff {
        let v1 = ToneAnalyser::analyse(&pair.v1.content);
        let v2 = ToneAnalyser::analyse(&pair.v2.content);
        let formality_delta = (v2.formality_score - v1.formality_score).abs();
        let assertiveness_delta = (v2.assertiveness_score - v1.assertiveness_score).abs();
        let hedge_word_delta = v2.hedge_word_count as i64 - v1.hedge_word_count as i64;
        let t = &self.risk_thresholds;
        let significant_shift = formality_delta >= t.tone_formality_delta_amber
            || assertiveness_delta >= t.tone_formality_delta_amber
            || hedge_word_delta.unsigned_abs() >= 4;
        let risk = if formality_delta >= t.tone_formality_delta_red
            || assertiveness_delta >= t.tone_formality_delta_red
        {
            RiskLevel::Red
        } else if significant_shift {
            RiskLevel::Amber
        } else {
            RiskLevel::Green
        };
        let direction = direction_tone(&pair.probe, v1.formality_score, v2.formality_score);
        ToneDiff {
            risk,
            direction,
            v1,
            v2,
            delta: ToneDelta {
                formality_delta: v2.formality_score - v1.formality_score,
                assertiveness_delta: v2.assertiveness_score - v1.assertiveness_score,
                hedge_word_delta,
                significant_shift,
            },
        }
    }

    fn compare_factual(&self, pair: &ResponsePair) -> Option<FactualDiff> {
        let known = pair.probe.known_answer.as_ref()?;
        let v1_ok = answer_matches_known(&pair.v1.content, known);
        let v2_ok = answer_matches_known(&pair.v2.content, known);
        let v1_extract = extract_snippet(&pair.v1.content, known);
        let v2_extract = extract_snippet(&pair.v2.content, known);
        let regression = v1_ok && !v2_ok;
        let improvement = !v1_ok && v2_ok;
        let risk = if regression {
            RiskLevel::Red
        } else if !v1_ok && !v2_ok {
            RiskLevel::Amber
        } else if improvement {
            RiskLevel::Green
        } else {
            RiskLevel::Green
        };
        let direction = if regression {
            DriftDirection::Regression
        } else if improvement {
            DriftDirection::Improvement
        } else {
            DriftDirection::Neutral
        };
        Some(FactualDiff {
            risk,
            direction,
            v1_correct: v1_ok,
            v2_correct: v2_ok,
            v1_answer_extract: v1_extract,
            v2_answer_extract: v2_extract,
            regression,
            improvement,
        })
    }

    fn compare_schema(&self, pair: &ResponsePair) -> Option<SchemaDiff> {
        let schema = pair.probe.expected_schema.as_ref()?;
        let v1_parsed = extract_json_value(&pair.v1.content);
        let v2_parsed = extract_json_value(&pair.v2.content);
        let v1_valid_json = v1_parsed.is_ok();
        let v2_valid_json = v2_parsed.is_ok();
        let (v1_schema_ok, v1_miss, v1_extra, v1_types) = if let Ok(v) = &v1_parsed {
            validate_json_schema(v, schema)
        } else {
            let empty: HashMap<String, String> = HashMap::new();
            (false, vec![], vec![], empty)
        };
        let (v2_schema_ok, v2_miss, v2_extra, v2_types) = if let Ok(v) = &v2_parsed {
            validate_json_schema(v, schema)
        } else {
            let empty: HashMap<String, String> = HashMap::new();
            (false, vec![], vec![], empty)
        };
        let field_type_changes = diff_field_types(&v1_types, &v2_types);
        let risk = if !v2_valid_json || !v2_schema_ok {
            RiskLevel::Red
        } else if !v1_schema_ok || !field_type_changes.is_empty() {
            RiskLevel::Amber
        } else {
            RiskLevel::Green
        };
        let direction = if v2_schema_ok && !v1_schema_ok {
            DriftDirection::Improvement
        } else if !v2_schema_ok && v1_schema_ok {
            DriftDirection::Regression
        } else {
            DriftDirection::Neutral
        };
        Some(SchemaDiff {
            risk,
            direction,
            v1_valid_json,
            v2_valid_json,
            v1_schema_valid: v1_schema_ok,
            v2_schema_valid: v2_schema_ok,
            v1_missing_fields: v1_miss,
            v2_missing_fields: v2_miss,
            v1_extra_fields: v1_extra,
            v2_extra_fields: v2_extra,
            field_type_changes,
        })
    }

    fn compare_instruction(&self, pair: &ResponsePair) -> Option<InstructionDiff> {
        if pair.probe.instructions.is_empty() {
            return None;
        }
        let mut v1_results = Vec::new();
        let mut v2_results = Vec::new();
        for ins in &pair.probe.instructions {
            v1_results.push(run_instruction_check(ins, &pair.v1.content));
            v2_results.push(run_instruction_check(ins, &pair.v2.content));
        }
        let v1_pass = v1_results.iter().filter(|r| r.passed).count() as f64 / v1_results.len() as f64;
        let v2_pass = v2_results.iter().filter(|r| r.passed).count() as f64 / v2_results.len() as f64;
        let mut regressions = Vec::new();
        for (a, b) in v1_results.iter().zip(v2_results.iter()) {
            if a.passed && !b.passed {
                regressions.push(a.instruction.clone());
            }
        }
        let risk = if !regressions.is_empty() {
            RiskLevel::Red
        } else if v2_pass < v1_pass {
            RiskLevel::Amber
        } else {
            RiskLevel::Green
        };
        let direction = if v2_pass > v1_pass {
            DriftDirection::Improvement
        } else if v2_pass < v1_pass {
            DriftDirection::Regression
        } else {
            DriftDirection::Neutral
        };
        Some(InstructionDiff {
            risk,
            direction,
            v1_results,
            v2_results,
            v1_pass_rate: v1_pass,
            v2_pass_rate: v2_pass,
            regressions,
        })
    }

    fn compare_refusal(&self, pair: &ResponsePair) -> RefusalDiff {
        let v1_refused = RefusalDetector::is_refusal(&pair.v1);
        let v2_refused = RefusalDetector::is_refusal(&pair.v2);
        let new_refusal = !v1_refused && v2_refused;
        let refusal_lifted = v1_refused && !v2_refused;
        let risk = if new_refusal {
            RiskLevel::Red
        } else if refusal_lifted {
            RiskLevel::Amber
        } else {
            RiskLevel::Green
        };
        let direction = direction_refusal(&pair.probe, new_refusal, refusal_lifted);
        RefusalDiff {
            risk,
            direction,
            v1_refused,
            v2_refused,
            new_refusal,
            refusal_lifted,
        }
    }

    fn compare_semantic(&self, pair: &ResponsePair, claim: &ClaimDiff) -> anyhow::Result<SemanticDiff> {
        if !self.semantic_analyser.is_enabled() {
            return Ok(SemanticDiff {
                risk: RiskLevel::Green,
                direction: DriftDirection::NotApplicable,
                cosine_similarity: None,
                semantic_scoring_disabled: true,
                disabled_reason: Some("semantic scoring disabled (--no-semantic or engine off)".into()),
                flagged_for_review: false,
                similarity_threshold: self.semantic_threshold,
            });
        }
        let sim = weighted_sentence_similarity(&pair.v1.content, &pair.v2.content);
        let t = &self.risk_thresholds;
        let flagged = sim < self.semantic_threshold;
        // Hash / token similarity: rephrased long answers often score below the red cutoff — cap at
        // Amber so overall probe risk (max across dimensions) is not forced Red on semantic alone.
        let hash_embedding_tier = self.claim_matcher.drift_threshold < 0.5;
        let risk = if sim < t.semantic_similarity_red {
            if hash_embedding_tier {
                RiskLevel::Amber
            } else {
                RiskLevel::Red
            }
        } else if sim < t.semantic_similarity_amber {
            RiskLevel::Amber
        } else {
            RiskLevel::Green
        };
        let direction = claim.direction.clone();
        Ok(SemanticDiff {
            risk,
            direction,
            cosine_similarity: Some(sim),
            semantic_scoring_disabled: false,
            disabled_reason: None,
            flagged_for_review: flagged,
            similarity_threshold: self.semantic_threshold,
        })
    }

    fn compare_latency(&self, pair: &ResponsePair) -> LatencyDiff {
        compute_latency_diff(pair.v1.latency_ms, pair.v2.latency_ms)
    }

    fn compare_consistency(&self, pair: &ResponsePair) -> Option<ConsistencyDiff> {
        let v1_series = if pair.v1_runs.is_empty() {
            vec![pair.v1.clone()]
        } else {
            pair.v1_runs.clone()
        };
        let v2_series = if pair.v2_runs.is_empty() {
            vec![pair.v2.clone()]
        } else {
            pair.v2_runs.clone()
        };
        if v1_series.len() <= 1 && v2_series.len() <= 1 {
            return None;
        }
        let v1_var = run_variance(&v1_series, self.risk_thresholds.consistency_variance_threshold);
        let v2_var = run_variance(&v2_series, self.risk_thresholds.consistency_variance_threshold);
        let t = self.risk_thresholds.consistency_variance_threshold;
        let v1_consistent = v1_var <= t;
        let v2_consistent = v2_var <= t;
        let consistency_regression = v1_consistent && !v2_consistent;
        let consistency_improvement = !v1_consistent && v2_consistent;
        // Red: v1 tight across runs, v2 loose (upgrade got worse). Amber: either side exceeds the
        // consistency band (mean pairwise 1−cosine on hash embeddings); threshold default 0.12.
        let risk = if consistency_regression {
            RiskLevel::Red
        } else if v1_var > t || v2_var > t {
            RiskLevel::Amber
        } else {
            RiskLevel::Green
        };
        let direction = if consistency_regression {
            DriftDirection::Regression
        } else if consistency_improvement {
            DriftDirection::Improvement
        } else {
            DriftDirection::Neutral
        };
        Some(ConsistencyDiff {
            risk,
            direction,
            v1_runs: v1_series.len(),
            v2_runs: v2_series.len(),
            v1_variance: v1_var,
            v2_variance: v2_var,
            v1_consistent,
            v2_consistent,
            consistency_regression,
            consistency_improvement,
        })
    }

    fn compare_custom(&self, pair: &ResponsePair) -> Option<CustomAssertionDiff> {
        if pair.probe.custom_assertions.is_empty() {
            return None;
        }
        let mut v1_results = Vec::new();
        let mut v2_results = Vec::new();
        for ca in &pair.probe.custom_assertions {
            let ins = ProbeInstruction {
                description: ca.description.clone(),
                check: ca.check.clone(),
            };
            v1_results.push(run_instruction_check(&ins, &pair.v1.content));
            v2_results.push(run_instruction_check(&ins, &pair.v2.content));
        }
        let v1_pass = v1_results.iter().filter(|r| r.passed).count() as f64 / v1_results.len() as f64;
        let v2_pass = v2_results.iter().filter(|r| r.passed).count() as f64 / v2_results.len() as f64;
        let mut regressions = Vec::new();
        for (a, b) in v1_results.iter().zip(v2_results.iter()) {
            if a.passed && !b.passed {
                regressions.push(a.instruction.clone());
            }
        }
        let risk = if !regressions.is_empty() {
            RiskLevel::Red
        } else if v2_pass < v1_pass {
            RiskLevel::Amber
        } else {
            RiskLevel::Green
        };
        let direction = if v2_pass > v1_pass {
            DriftDirection::Improvement
        } else if v2_pass < v1_pass {
            DriftDirection::Regression
        } else {
            DriftDirection::Neutral
        };
        Some(CustomAssertionDiff {
            risk,
            direction,
            v1_results,
            v2_results,
            v1_pass_rate: v1_pass,
            v2_pass_rate: v2_pass,
            regressions,
        })
    }

    pub fn compute_probe_risk(&self, probe: &Probe, dimensions: &ProbeDimensions) -> RiskLevel {
        compute_probe_risk(probe, dimensions).0
    }

    pub fn compute_overall_risk(&self, results: &[ProbeResult]) -> RiskLevel {
        let mut worst = RiskLevel::Green;
        for pr in results {
            worst = worst.max(pr.overall_risk.clone());
        }
        worst
    }

    fn build_summary(
        &self,
        results: &[ProbeResult],
        _overall: &RiskLevel,
        valence: &ValenceSummary,
    ) -> ReportSummary {
        let total = results.len();
        let mut green = 0usize;
        let mut amber = 0usize;
        let mut red = 0usize;
        for pr in results {
            match pr.overall_risk {
                RiskLevel::Green => green += 1,
                RiskLevel::Amber => amber += 1,
                RiskLevel::Red => red += 1,
            }
        }
        let drift_counts = count_drift_categories(results);
        let safe = !has_critical_blocking(results);
        let manual = amber + red;
        ReportSummary {
            total_probes: total,
            probes_green: green,
            probes_amber: amber,
            probes_red: red,
            safe_to_upgrade: safe,
            requires_manual_review: manual,
            auto_remediation_candidates: 0,
            probe_regressions: valence.probe_regressions,
            probe_improvements: valence.probe_improvements,
            probe_neutral: valence.probe_neutral,
            drift_critical_regressions: drift_counts.critical_regressions,
            drift_policy: drift_counts.policy,
            drift_fidelity: drift_counts.fidelity,
            drift_structural: drift_counts.structural,
            drift_content_compression: drift_counts.content_compression,
        }
    }

    pub fn compute_dimension_summaries(&self, results: &[ProbeResult]) -> DimensionSummaries {
        DimensionSummaries {
            morphology: dim_summary_core(results, |d| Some((&d.morphology.risk, d.morphology.direction))),
            tone: dim_summary_core(results, |d| Some((&d.tone.risk, d.tone.direction))),
            factual: dim_summary_core(results, |d| d.factual.as_ref().map(|f| (&f.risk, f.direction))),
            schema: dim_summary_core(results, |d| d.schema.as_ref().map(|s| (&s.risk, s.direction))),
            instruction: dim_summary_core(results, |d| d.instruction.as_ref().map(|i| (&i.risk, i.direction))),
            refusal: dim_summary_core(results, |d| Some((&d.refusal.risk, d.refusal.direction))),
            semantic: dim_summary_core(results, |d| Some((&d.semantic.risk, d.semantic.direction))),
            claim: dim_summary_core(results, |d| Some((&d.claim.risk, d.claim.direction))),
            latency: dim_summary_core(results, |d| Some((&d.latency.risk, d.latency.direction))),
            consistency: dim_summary_core(results, |d| d.consistency.as_ref().map(|c| (&c.risk, c.direction))),
            custom_assertions: dim_summary_core(results, |d| {
                d.custom_assertions.as_ref().map(|c| (&c.risk, c.direction))
            }),
        }
    }
}

/// Worst (max) risk among dimension levels — used only in unit tests for legacy rollup checks.
#[cfg(test)]
fn max_risk_level(levels: impl IntoIterator<Item = RiskLevel>) -> RiskLevel {
    levels
        .into_iter()
        .max()
        .unwrap_or(RiskLevel::Green)
}

/// Per-dimension risks for legacy max-rollup tests. Latency is excluded (observational only).
#[cfg(test)]
fn probe_dimension_risk_levels(dimensions: &ProbeDimensions) -> Vec<RiskLevel> {
    let semantic_for_overall = if dimensions.semantic.semantic_scoring_disabled {
        RiskLevel::Green
    } else {
        dimensions.semantic.risk.clone()
    };

    let mut risks = vec![
        dimensions.morphology.risk.clone(),
        dimensions.tone.risk.clone(),
        dimensions.refusal.risk.clone(),
        semantic_for_overall,
        dimensions.claim.risk.clone(),
    ];
    if let Some(f) = &dimensions.factual {
        risks.push(f.risk.clone());
    }
    if let Some(s) = &dimensions.schema {
        risks.push(s.risk.clone());
    }
    if let Some(i) = &dimensions.instruction {
        risks.push(i.risk.clone());
    }
    if let Some(c) = &dimensions.custom_assertions {
        risks.push(c.risk.clone());
    }
    if let Some(co) = &dimensions.consistency {
        risks.push(co.risk.clone());
    }
    risks
}

/// Run-level latency rollup for the dedicated report section (not used for risk or routing).
/// `delta_pct` is a **percentage** (e.g. `-60.8` = 60.8% faster), not a unit ratio.
pub fn compute_latency_summary(results: &[ProbeResult]) -> LatencySummary {
    const NEUTRAL_BAND_PCT: f64 = 10.0;
    let n = results.len();
    if n == 0 {
        return LatencySummary {
            v1_avg_latency_ms: 0,
            v2_avg_latency_ms: 0,
            delta_ms: 0,
            delta_pct: 0.0,
            direction: DriftDirection::Neutral,
            note: "No probes in run.".into(),
        };
    }
    let v1_avg = results
        .iter()
        .map(|r| r.dimensions.latency.v1_latency_ms)
        .sum::<u64>()
        / n as u64;
    let v2_avg = results
        .iter()
        .map(|r| r.dimensions.latency.v2_latency_ms)
        .sum::<u64>()
        / n as u64;
    let delta_ms = v2_avg as i64 - v1_avg as i64;
    let delta_pct = if v1_avg == 0 {
        if v2_avg == 0 {
            0.0
        } else {
            1.0
        }
    } else {
        (delta_ms as f64 / v1_avg as f64) * 100.0
    };
    let direction = if delta_pct < -NEUTRAL_BAND_PCT {
        DriftDirection::Improvement
    } else if delta_pct > NEUTRAL_BAND_PCT {
        DriftDirection::Regression
    } else {
        DriftDirection::Neutral
    };
    let note = latency_summary_note(direction, delta_pct, n);
    LatencySummary {
        v1_avg_latency_ms: v1_avg,
        v2_avg_latency_ms: v2_avg,
        delta_ms,
        delta_pct,
        direction,
        note,
    }
}

/// Run-level plain-English fingerprint of v2 vs v1 for the migration profile card.
pub fn compute_migration_profile(
    results: &[ProbeResult],
    latency: &LatencySummary,
) -> MigrationProfile {
    const SPEED_THRESHOLD_PCT: f64 = 20.0;
    const VERBOSITY_THRESHOLD: f64 = 0.20;
    const FORMALITY_THRESHOLD: f64 = 0.15;
    const STRUCTURAL_DRIFT_SHARE: f64 = 0.40;

    let speed_change = if latency.delta_pct < -SPEED_THRESHOLD_PCT {
        Some(format!("{}% faster", latency.delta_pct.abs().round() as i64))
    } else if latency.delta_pct > SPEED_THRESHOLD_PCT {
        Some(format!("{}% slower", latency.delta_pct.round() as i64))
    } else {
        None
    };

    let (total_v1_tokens, total_v2_tokens) = results.iter().fold((0usize, 0usize), |(a, b), pr| {
        (
            a + pr.dimensions.morphology.v1.token_count,
            b + pr.dimensions.morphology.v2.token_count,
        )
    });
    let verbosity_change = if total_v1_tokens == 0 {
        None
    } else {
        let pct = (total_v2_tokens as f64 - total_v1_tokens as f64) / total_v1_tokens as f64;
        if pct < -VERBOSITY_THRESHOLD {
            Some("more concise".into())
        } else if pct > VERBOSITY_THRESHOLD {
            Some("more verbose".into())
        } else {
            None
        }
    };

    let style_change = if results.is_empty() {
        None
    } else {
        let avg_formality: f64 = results
            .iter()
            .map(|pr| pr.dimensions.tone.delta.formality_delta)
            .sum::<f64>()
            / results.len() as f64;
        if avg_formality < -FORMALITY_THRESHOLD {
            Some("more conversational".into())
        } else if avg_formality > FORMALITY_THRESHOLD {
            Some("more formal".into())
        } else {
            None
        }
    };

    let n = results.len();
    let structural_compression = results
        .iter()
        .filter(|pr| {
            matches!(
                pr.drift_category,
                DriftCategory::ContentCompression | DriftCategory::StructuralDrift
            )
        })
        .count();
    let reliability_change = if n > 0
        && (structural_compression as f64 / n as f64) > STRUCTURAL_DRIFT_SHARE
    {
        Some("less structurally consistent".into())
    } else {
        None
    };

    let headline = compose_migration_headline(
        results,
        speed_change.as_deref(),
        verbosity_change.as_deref(),
        style_change.as_deref(),
        reliability_change.as_deref(),
    );
    let safe_to_upgrade = critical_blocking_count(results) == 0;

    MigrationProfile {
        speed_change,
        verbosity_change,
        style_change,
        reliability_change,
        headline,
        safe_to_upgrade,
    }
}

fn critical_blocking_count(results: &[ProbeResult]) -> usize {
    results
        .iter()
        .filter(|p| is_blocking_probe(p))
        .count()
}

fn attention_count(results: &[ProbeResult]) -> usize {
    results
        .iter()
        .filter(|p| needs_attention_probe(p))
        .count()
}

fn compression_count(results: &[ProbeResult]) -> usize {
    results
        .iter()
        .filter(|p| p.drift_category == DriftCategory::ContentCompression)
        .count()
}

fn join_profile_traits(items: &[&str]) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].to_string(),
        2 => format!("{} and {}", items[0], items[1]),
        _ => format!(
            "{}, and {}",
            items[..items.len() - 1].join(", "),
            items[items.len() - 1]
        ),
    }
}

fn compose_migration_headline(
    results: &[ProbeResult],
    speed: Option<&str>,
    verbosity: Option<&str>,
    style: Option<&str>,
    reliability: Option<&str>,
) -> String {
    let critical = critical_blocking_count(results);
    let attention = attention_count(results);
    let compression = compression_count(results);

    if critical > 0 {
        return format!(
            "v2 introduces {critical} critical regression{} — upgrade not recommended without prompt fixes",
            if critical == 1 { "" } else { "s" }
        );
    }

    if attention > 0 {
        return format!(
            "v2 is safe to upgrade — {attention} probe{} warrant review before switching",
            if attention == 1 { "" } else { "s" }
        );
    }

    if compression > 0 {
        let only_compression = results
            .iter()
            .all(|p| {
                p.drift_category == DriftCategory::ContentCompression
                    || p.drift_category == DriftCategory::NoSignificantDrift
                    || matches!(p.overall_risk, RiskLevel::Green)
            });
        if only_compression {
            return format!(
                "v2 is more concise across {compression} probe{} — review compression before upgrading",
                if compression == 1 { "" } else { "s" }
            );
        }
    }

    let mut secondary: Vec<&str> = Vec::new();
    if let Some(v) = verbosity {
        secondary.push(v);
    }
    if let Some(st) = style {
        secondary.push(st);
    }
    if let Some(r) = reliability {
        secondary.push(r);
    }

    if let Some(s) = speed {
        if secondary.is_empty() {
            return format!("v2 is {s}");
        }
        return format!(
            "v2 is {s} but {}{}",
            join_profile_traits(&secondary),
            if reliability.is_some() {
                " across open-ended prompts"
            } else {
                ""
            }
        );
    }

    if secondary.is_empty() {
        return "v2 is behaviourally equivalent — safe to upgrade".into();
    }

    format!(
        "v2 is {}{}",
        join_profile_traits(&secondary),
        if reliability.is_some() {
            " across open-ended prompts"
        } else {
            ""
        }
    )
}

fn latency_summary_note(direction: DriftDirection, delta_pct: f64, probe_count: usize) -> String {
    let pct = delta_pct.abs().round() as i64;
    let n = probe_count;
    match direction {
        DriftDirection::Improvement => {
            format!("v2 responded {pct}% faster on average across {n} probes")
        }
        DriftDirection::Regression => {
            format!("v2 is {pct}% slower on average across {n} probes")
        }
        DriftDirection::Neutral | DriftDirection::NotApplicable => {
            format!("v2 latency within 10% of v1 on average across {n} probes")
        }
    }
}

fn run_variance(runs: &[ModelResponse], _thresh: f64) -> f64 {
    if runs.len() < 2 {
        return 0.0;
    }
    let contents: Vec<String> = runs.iter().map(|r| r.content.clone()).collect();
    let embs = crate::embedding::embed_batch_hash(&contents);
    let mut dists = Vec::new();
    for i in 0..runs.len() {
        for j in (i + 1)..runs.len() {
            let d = 1.0 - crate::embedding::cosine_f32(&embs[i], &embs[j]);
            dists.push(d);
        }
    }
    if dists.is_empty() {
        0.0
    } else {
        dists.iter().sum::<f64>() / dists.len() as f64
    }
}

fn direction_morphology(
    probe: &Probe,
    w1: usize,
    w2: usize,
    response_type_changed: bool,
) -> DriftDirection {
    use ExpectedVerbosity::*;
    if let Some(ev) = &probe.expected_verbosity {
        return match ev {
            Concise => {
                if w2 > w1 {
                    DriftDirection::Regression
                } else if w2 < w1 {
                    DriftDirection::Improvement
                } else {
                    DriftDirection::Neutral
                }
            }
            Detailed => {
                if w2 < w1 {
                    DriftDirection::Regression
                } else if w2 > w1 {
                    DriftDirection::Improvement
                } else {
                    DriftDirection::Neutral
                }
            }
            Moderate => {
                if response_type_changed {
                    DriftDirection::Neutral
                } else {
                    DriftDirection::Neutral
                }
            }
        };
    }
    if response_type_changed {
        DriftDirection::Neutral
    } else {
        DriftDirection::Neutral
    }
}

fn direction_tone(probe: &Probe, f1: f64, f2: f64) -> DriftDirection {
    use ExpectedTonePreference::*;
    if let Some(et) = &probe.expected_tone {
        return match et {
            Formal => {
                if f2 > f1 {
                    DriftDirection::Improvement
                } else if f2 < f1 {
                    DriftDirection::Regression
                } else {
                    DriftDirection::Neutral
                }
            }
            Casual => {
                if f2 < f1 {
                    DriftDirection::Improvement
                } else if f2 > f1 {
                    DriftDirection::Regression
                } else {
                    DriftDirection::Neutral
                }
            }
            Neutral => DriftDirection::Neutral,
        };
    }
    DriftDirection::Neutral
}

fn direction_refusal(probe: &Probe, new_refusal: bool, lifted: bool) -> DriftDirection {
    let should_answer = probe.tags.iter().any(|t| t == "should-answer")
        || matches!(
            probe.refusal_expectation,
            Some(RefusalExpectation::ShouldAnswer)
        );
    let should_refuse = probe.tags.iter().any(|t| t == "should-refuse")
        || matches!(
            probe.refusal_expectation,
            Some(RefusalExpectation::ShouldRefuse)
        );
    if new_refusal {
        if should_refuse {
            DriftDirection::Improvement
        } else {
            DriftDirection::Regression
        }
    } else if lifted {
        if should_answer {
            DriftDirection::Improvement
        } else {
            DriftDirection::Neutral
        }
    } else {
        DriftDirection::Neutral
    }
}

/// True when a substantive dimension regressed (semantic, refusal, claim, factual, etc.).
/// Stylistic-only regressions (morphology, tone) do not cancel a claim-level improvement when
/// aggregating per-probe drift direction.
fn substantive_regression_direction(d: &ProbeDimensions) -> bool {
    if matches!(d.semantic.direction, DriftDirection::Regression) {
        return true;
    }
    if matches!(d.refusal.direction, DriftDirection::Regression) {
        return true;
    }
    if matches!(d.claim.direction, DriftDirection::Regression) {
        return true;
    }
    if let Some(f) = &d.factual {
        if matches!(f.direction, DriftDirection::Regression) {
            return true;
        }
    }
    if let Some(s) = &d.schema {
        if matches!(s.direction, DriftDirection::Regression) {
            return true;
        }
    }
    if let Some(i) = &d.instruction {
        if matches!(i.direction, DriftDirection::Regression) {
            return true;
        }
    }
    if let Some(c) = &d.custom_assertions {
        if matches!(c.direction, DriftDirection::Regression) {
            return true;
        }
    }
    if let Some(co) = &d.consistency {
        if matches!(co.direction, DriftDirection::Regression) {
            return true;
        }
    }
    false
}

fn probe_overall_direction(d: &ProbeDimensions, overall_risk: &RiskLevel) -> DriftDirection {
    if matches!(overall_risk, RiskLevel::Green) {
        return DriftDirection::Neutral;
    }
    let mut reg = false;
    let mut imp = false;
    let dims: Vec<DriftDirection> = vec![
        d.morphology.direction.clone(),
        d.tone.direction.clone(),
        d.refusal.direction.clone(),
        d.semantic.direction.clone(),
        d.claim.direction.clone(),
    ];
    for x in dims {
        if matches!(x, DriftDirection::Regression) {
            reg = true;
        }
        if matches!(x, DriftDirection::Improvement) {
            imp = true;
        }
    }
    if let Some(f) = &d.factual {
        if matches!(f.direction, DriftDirection::Regression) {
            reg = true;
        }
        if matches!(f.direction, DriftDirection::Improvement) {
            imp = true;
        }
    }
    if let Some(s) = &d.schema {
        if matches!(s.direction, DriftDirection::Regression) {
            reg = true;
        }
        if matches!(s.direction, DriftDirection::Improvement) {
            imp = true;
        }
    }
    if let Some(i) = &d.instruction {
        if matches!(i.direction, DriftDirection::Regression) {
            reg = true;
        }
        if matches!(i.direction, DriftDirection::Improvement) {
            imp = true;
        }
    }
    if let Some(c) = &d.custom_assertions {
        if matches!(c.direction, DriftDirection::Regression) {
            reg = true;
        }
        if matches!(c.direction, DriftDirection::Improvement) {
            imp = true;
        }
    }
    if let Some(co) = &d.consistency {
        if matches!(co.direction, DriftDirection::Regression) {
            reg = true;
        }
        if matches!(co.direction, DriftDirection::Improvement) {
            imp = true;
        }
    }
    if reg && !imp {
        DriftDirection::Regression
    } else if imp && !reg {
        DriftDirection::Improvement
    } else if reg && imp {
        if matches!(d.claim.direction, DriftDirection::Improvement) && !substantive_regression_direction(d) {
            DriftDirection::Improvement
        } else {
            DriftDirection::Neutral
        }
    } else {
        DriftDirection::Neutral
    }
}

fn valence_counts(results: &[ProbeResult]) -> ValenceSummary {
    let mut probe_regressions = 0usize;
    let mut probe_improvements = 0usize;
    let mut probe_neutral = 0usize;
    let mut dimension_regressions = 0usize;
    let mut dimension_improvements = 0usize;
    for pr in results {
        match pr.overall_direction {
            DriftDirection::Regression => probe_regressions += 1,
            DriftDirection::Improvement => probe_improvements += 1,
            _ => probe_neutral += 1,
        }
        let d = &pr.dimensions;
        for dir in [
            &d.morphology.direction,
            &d.tone.direction,
            &d.refusal.direction,
            &d.semantic.direction,
            &d.claim.direction,
        ] {
            if matches!(dir, DriftDirection::Regression) {
                dimension_regressions += 1;
            }
            if matches!(dir, DriftDirection::Improvement) {
                dimension_improvements += 1;
            }
        }
        if let Some(f) = &d.factual {
            if matches!(f.direction, DriftDirection::Regression) {
                dimension_regressions += 1;
            }
            if matches!(f.direction, DriftDirection::Improvement) {
                dimension_improvements += 1;
            }
        }
        if let Some(s) = &d.schema {
            if matches!(s.direction, DriftDirection::Regression) {
                dimension_regressions += 1;
            }
            if matches!(s.direction, DriftDirection::Improvement) {
                dimension_improvements += 1;
            }
        }
        if let Some(i) = &d.instruction {
            if matches!(i.direction, DriftDirection::Regression) {
                dimension_regressions += 1;
            }
            if matches!(i.direction, DriftDirection::Improvement) {
                dimension_improvements += 1;
            }
        }
        if let Some(c) = &d.custom_assertions {
            if matches!(c.direction, DriftDirection::Regression) {
                dimension_regressions += 1;
            }
            if matches!(c.direction, DriftDirection::Improvement) {
                dimension_improvements += 1;
            }
        }
        if let Some(co) = &d.consistency {
            if matches!(co.direction, DriftDirection::Regression) {
                dimension_regressions += 1;
            }
            if matches!(co.direction, DriftDirection::Improvement) {
                dimension_improvements += 1;
            }
        }
    }
    ValenceSummary {
        probe_regressions,
        probe_improvements,
        probe_neutral,
        dimension_regressions,
        dimension_improvements,
    }
}

impl DriftReport {
    /// Recomputes valence counts from `probe_results` and copies probe-level buckets into `summary`.
    ///
    /// Use after loading older JSON or before rendering so executive stats match per-probe directions.
    pub fn sync_valence_from_probe_results(&mut self) {
        let v = valence_counts(&self.probe_results);
        self.summary.probe_regressions = v.probe_regressions;
        self.summary.probe_improvements = v.probe_improvements;
        self.summary.probe_neutral = v.probe_neutral;
        self.valence_summary = v;
    }
}

struct DriftCategoryCounts {
    critical_regressions: usize,
    policy: usize,
    fidelity: usize,
    structural: usize,
    content_compression: usize,
}

fn count_drift_categories(results: &[ProbeResult]) -> DriftCategoryCounts {
    let mut counts = DriftCategoryCounts {
        critical_regressions: 0,
        policy: 0,
        fidelity: 0,
        structural: 0,
        content_compression: 0,
    };
    for pr in results {
        match pr.drift_category {
            DriftCategory::CriticalRegression => counts.critical_regressions += 1,
            DriftCategory::PolicyDrift => counts.policy += 1,
            DriftCategory::FidelityDrift => counts.fidelity += 1,
            DriftCategory::StructuralDrift => counts.structural += 1,
            DriftCategory::ContentCompression => counts.content_compression += 1,
            DriftCategory::NoSignificantDrift => {}
        }
    }
    counts
}

fn has_critical_blocking(results: &[ProbeResult]) -> bool {
    critical_blocking_count(results) > 0
}

fn is_blocking_probe(pr: &ProbeResult) -> bool {
    pr.drift_category
        .is_blocking(&pr.overall_risk)
}

fn needs_attention_probe(pr: &ProbeResult) -> bool {
    matches!(pr.drift_category, DriftCategory::CriticalRegression)
        && matches!(pr.overall_risk, RiskLevel::Amber)
}

/// Signed token change as a percentage of v1 length (negative = v2 shorter).
fn signed_token_delta_pct(dims: &ProbeDimensions) -> f64 {
    if dims.morphology.v1.token_count == 0 {
        return 0.0;
    }
    (dims.morphology.delta.token_delta as f64 / dims.morphology.v1.token_count as f64) * 100.0
}

/// Map dimension findings to a severity level (primary driver of probe risk).
pub fn dimension_severity(probe: &Probe, dims: &ProbeDimensions) -> DriftSeverity {
    if let Some(s) = &dims.schema {
        if matches!(s.risk, RiskLevel::Red) {
            return DriftSeverity::Critical;
        }
    }
    if let Some(i) = &dims.instruction {
        if !i.regressions.is_empty() {
            return DriftSeverity::Critical;
        }
    }
    if let Some(f) = &dims.factual {
        if f.regression {
            return DriftSeverity::Critical;
        }
    }
    if !dims.claim.drifted_claims.is_empty()
        && dims
            .claim
            .drifted_claims
            .iter()
            .any(|d| !d.drifted_anchors.is_empty())
    {
        return DriftSeverity::Critical;
    }

    if dims.refusal.new_refusal || dims.refusal.refusal_lifted {
        return DriftSeverity::High;
    }
    if matches!(
        probe.category,
        ProbeCategory::Factual | ProbeCategory::Schema | ProbeCategory::Instruction
    ) && dims.claim.preservation_score < 0.70
    {
        return DriftSeverity::High;
    }

    if matches!(dims.tone.risk, RiskLevel::Red) {
        return DriftSeverity::Medium;
    }
    if dims.morphology.delta.response_type_changed
        && matches!(
            probe.category,
            ProbeCategory::Schema | ProbeCategory::Instruction
        )
    {
        return DriftSeverity::Medium;
    }

    if signed_token_delta_pct(dims) < -30.0 {
        return DriftSeverity::Low;
    }
    if matches!(
        probe.category,
        ProbeCategory::Semantic | ProbeCategory::Tone | ProbeCategory::Morphology
    ) && dims.claim.preservation_score < 0.70
    {
        return DriftSeverity::Low;
    }

    DriftSeverity::Informational
}

/// Severity-weighted probe risk and drift category (replaces max dimension risk rollup).
pub fn compute_probe_risk(
    probe: &Probe,
    dims: &ProbeDimensions,
) -> (RiskLevel, DriftCategory, DriftSeverity) {
    let severity = dimension_severity(probe, dims);
    let (risk, category) = match severity {
        DriftSeverity::Critical => (RiskLevel::Red, DriftCategory::CriticalRegression),
        DriftSeverity::High => (RiskLevel::Amber, DriftCategory::CriticalRegression),
        DriftSeverity::Medium => (RiskLevel::Amber, DriftCategory::FidelityDrift),
        DriftSeverity::Low => (RiskLevel::Amber, DriftCategory::ContentCompression),
        DriftSeverity::Informational => {
            if signed_token_delta_pct(dims).abs() > 10.0 || dims.claim.preservation_score < 0.90 {
                (RiskLevel::Amber, DriftCategory::StructuralDrift)
            } else {
                (RiskLevel::Green, DriftCategory::NoSignificantDrift)
            }
        }
    };
    (risk, category, severity)
}

fn build_upgrade_path(results: &[ProbeResult], mutations: &[MutationResult]) -> UpgradePathReport {
    let mut critical = Vec::new();
    let mut blocking = Vec::new();
    let mut verify = Vec::new();
    let mut neutral = Vec::new();
    let mut certified = Vec::new();
    for pr in results {
        let certified_mutation = mutations
            .iter()
            .find(|m| m.probe_name == pr.probe.name && m.validated)
            .map(|m| m.mutated_prompt.clone());
        let item = UpgradePathItem {
            probe_name: pr.probe.name.clone(),
            category: pr.probe.category.clone(),
            overall_risk: pr.overall_risk.clone(),
            overall_direction: pr.overall_direction.clone(),
            drift_category: pr.drift_category,
            summary: format!(
                "{:?} / {:?} / {:?} / {:?}",
                pr.drift_severity, pr.overall_risk, pr.overall_direction, pr.drift_category
            ),
            certified_mutation,
        };
        if is_blocking_probe(pr) {
            blocking.push(item.clone());
            critical.push(item);
        } else if needs_attention_probe(pr) {
            let mut attention_item = item;
            attention_item.summary = format!(
                "{} — warrants attention before switching",
                attention_item.summary
            );
            verify.push(attention_item);
        } else if matches!(pr.overall_risk, RiskLevel::Red | RiskLevel::Amber)
            && matches!(pr.overall_direction, DriftDirection::Improvement)
        {
            verify.push(item);
        } else if matches!(pr.overall_risk, RiskLevel::Red | RiskLevel::Amber) {
            neutral.push(item);
        }
    }
    for m in mutations {
        if m.validated {
            certified.push(CertifiedPromptDiff {
                probe_name: m.probe_name.clone(),
                original_prompt: m.original_prompt.clone(),
                mutated_prompt: m.mutated_prompt.clone(),
                validated: true,
                validation_risk: m.validation_risk.clone(),
                strategies_applied: m.strategies_applied.clone(),
            });
        }
    }
    let remediation = RemediationCounts {
        prompt_changes_suggested: mutations.len(),
        manual_review: mutations.iter().filter(|m| m.requires_manual_review).count(),
        auto_certified: mutations.iter().filter(|m| m.validated).count(),
    };
    UpgradePathReport {
        critical_regressions: critical,
        policy_changes: Vec::new(),
        blocking_regressions: blocking,
        improvements_to_verify: verify,
        neutral_changes: neutral,
        certified_prompts: certified,
        remediation,
    }
}

fn dim_summary_core<Rd>(results: &[ProbeResult], mut get: Rd) -> DimensionSummary
where
    Rd: FnMut(&ProbeDimensions) -> Option<(&RiskLevel, DriftDirection)>,
{
    let mut affected = 0usize;
    let mut worst = RiskLevel::Green;
    let mut notes = Vec::new();
    let mut drift_regressions = 0usize;
    let mut drift_improvements = 0usize;
    let mut drift_neutral = 0usize;
    let mut drift_not_applicable = 0usize;
    for pr in results {
        if let Some((r, dir)) = get(&pr.dimensions) {
            if !matches!(r, RiskLevel::Green) {
                affected += 1;
                worst = worst.max(r.clone());
                match dir {
                    DriftDirection::Regression => drift_regressions += 1,
                    DriftDirection::Improvement => drift_improvements += 1,
                    DriftDirection::Neutral => drift_neutral += 1,
                    DriftDirection::NotApplicable => drift_not_applicable += 1,
                }
                if matches!(r, RiskLevel::Red) {
                    notes.push(format!("probe {}: {:?}", pr.probe.name, r));
                }
            }
        }
    }
    DimensionSummary {
        probes_affected: affected,
        worst_risk: worst,
        notes: notes.into_iter().take(20).collect(),
        drift_regressions,
        drift_improvements,
        drift_neutral,
        drift_not_applicable,
    }
}

fn morphology_risk_level(
    token_delta_pct: f64,
    response_type_changed: bool,
    structure_changed: bool,
    amber: f64,
    red: f64,
) -> RiskLevel {
    if token_delta_pct >= red || (response_type_changed && token_delta_pct >= amber) {
        RiskLevel::Red
    } else if token_delta_pct >= amber || structure_changed || response_type_changed {
        RiskLevel::Amber
    } else {
        RiskLevel::Green
    }
}

fn answer_matches_known(content: &str, known: &str) -> bool {
    let c = content.to_lowercase();
    let k = known.to_lowercase();
    if c.contains(&k) {
        return true;
    }
    if known.chars().all(|ch| ch.is_ascii_digit()) {
        for word in c.split(|ch: char| !ch.is_ascii_digit()) {
            if word == k {
                return true;
            }
        }
    }
    false
}

fn extract_snippet(content: &str, _known: &str) -> String {
    let c = content.trim();
    if c.len() > 200 {
        format!("{}…", &c[..200])
    } else {
        c.to_string()
    }
}

fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

fn run_instruction_check(ins: &ProbeInstruction, content: &str) -> InstructionCheckResult {
    let passed = match &ins.check {
        InstructionCheck::MaxWords(n) => word_count(content) <= *n,
        InstructionCheck::MinWords(n) => word_count(content) >= *n,
        InstructionCheck::MustContain(s) => content.to_lowercase().contains(&s.to_lowercase()),
        InstructionCheck::MustNotContain(s) => !content.to_lowercase().contains(&s.to_lowercase()),
        InstructionCheck::MustStartWith(s) => content.trim_start().starts_with(s),
        InstructionCheck::MustEndWith(s) => content.trim_end().ends_with(s),
        InstructionCheck::OutputFormat(f) => check_output_format(content, f),
    };
    let detail = if passed {
        "ok".into()
    } else {
        format!("failed check {:?}", ins.check)
    };
    InstructionCheckResult {
        instruction: ins.description.clone(),
        passed,
        detail,
    }
}

/// Parse JSON from raw model text: whole body, fenced code blocks, or first balanced object/array.
fn extract_json_value(content: &str) -> Result<serde_json::Value, serde_json::Error> {
    let trimmed = content.trim();
    if let Ok(v) = serde_json::from_str(trimmed) {
        return Ok(v);
    }
    for block in extract_fenced_code_blocks(trimmed) {
        if let Ok(v) = serde_json::from_str(block) {
            return Ok(v);
        }
    }
    if let Some(slice) = extract_balanced_json_slice(trimmed) {
        return serde_json::from_str(slice);
    }
    serde_json::from_str(trimmed)
}

fn extract_fenced_code_blocks(content: &str) -> Vec<&str> {
    let mut blocks = Vec::new();
    let mut rest = content;
    while let Some(start) = rest.find("```") {
        rest = &rest[start + 3..];
        rest = rest.trim_start();
        if rest.starts_with("json") {
            rest = rest[4..].trim_start();
        } else if let Some((lang, tail)) = rest.split_once('\n') {
            if !lang.contains(' ') && lang.len() <= 12 && !lang.contains('{') {
                rest = tail;
            }
        }
        if let Some(end) = rest.find("```") {
            let block = rest[..end].trim();
            if !block.is_empty() {
                blocks.push(block);
            }
            rest = &rest[end + 3..];
        } else {
            break;
        }
    }
    blocks
}

fn extract_balanced_json_slice(content: &str) -> Option<&str> {
    let start = content.find(['{', '['])?;
    let open = content.as_bytes()[start];
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, b) in content[start..].bytes().enumerate() {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            c if c == open => depth += 1,
            c if c == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(&content[start..start + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

fn check_output_format(content: &str, fmt: &OutputFormat) -> bool {
    match fmt {
        OutputFormat::Json => extract_json_value(content).is_ok(),
        OutputFormat::Markdown => content.contains('#') || content.contains("**"),
        OutputFormat::PlainText => !content.contains("```") && !content.contains('|'),
        OutputFormat::BulletList => {
            content.lines().filter(|l| l.trim_start().starts_with("- ") || l.trim_start().starts_with("* ")).count() >= 2
        }
        OutputFormat::NumberedList => {
            content.lines().any(|l| {
                let t = l.trim_start();
                t.len() > 2 && t.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) && t.contains('.')
            })
        }
    }
}

fn validate_json_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
) -> (bool, Vec<String>, Vec<String>, HashMap<String, String>) {
    let obj = match value.as_object() {
        Some(o) => o,
        None => return (false, vec!["<root>".into()], vec![], HashMap::new()),
    };
    let mut missing = Vec::new();
    let mut extra = Vec::new();
    let mut types = HashMap::new();
    if let Some(req) = schema.get("required").and_then(|r| r.as_array()) {
        for r in req {
            if let Some(name) = r.as_str() {
                if !obj.contains_key(name) {
                    missing.push(name.to_string());
                }
            }
        }
    }
    if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
        let allowed: HashSet<_> = props.keys().cloned().collect();
        for k in obj.keys() {
            if !allowed.contains(k) {
                extra.push(k.clone());
            }
        }
        for (k, spec) in props {
            if let Some(v) = obj.get(k) {
                let expected = spec.get("type").and_then(|t| t.as_str()).unwrap_or("");
                let actual = json_type_name(v);
                types.insert(k.clone(), actual.clone());
                if !expected.is_empty() && actual != expected {}
            }
        }
    }
    let schema_ok = missing.is_empty() && field_types_ok(obj, schema);
    (schema_ok, missing, extra, types)
}

fn field_types_ok(obj: &serde_json::Map<String, serde_json::Value>, schema: &serde_json::Value) -> bool {
    let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
        return true;
    };
    for (k, spec) in props {
        let Some(v) = obj.get(k) else { continue };
        let expected = spec.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if expected.is_empty() {
            continue;
        }
        let actual = json_type_name(v);
        if actual != expected {
            return false;
        }
    }
    true
}

fn json_type_name(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "null".into(),
        serde_json::Value::Bool(_) => "boolean".into(),
        serde_json::Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "integer".into()
            } else {
                "number".into()
            }
        }
        serde_json::Value::String(_) => "string".into(),
        serde_json::Value::Array(_) => "array".into(),
        serde_json::Value::Object(_) => "object".into(),
    }
}

fn diff_field_types(
    a: &HashMap<String, String>,
    b: &HashMap<String, String>,
) -> Vec<FieldTypeChange> {
    let mut out = Vec::new();
    for (k, t1) in a {
        if let Some(t2) = b.get(k) {
            if t1 != t2 {
                out.push(FieldTypeChange {
                    field: k.clone(),
                    v1_type: t1.clone(),
                    v2_type: t2.clone(),
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_from_markdown_fence() {
        let body = "Here is the JSON object representing a person:\n\n```\n{\n  \"name\": \"Jane Smith\",\n  \"age\": 34,\n  \"email\": \"jane@example.com\",\n  \"active\": true\n}\n```";
        let v = extract_json_value(body).expect("parse fenced json");
        assert_eq!(v["name"], "Jane Smith");
        assert_eq!(v["age"], 34);
    }

    #[test]
    fn certified_prompt_diff_carries_validation_risk() {
        let mutations = vec![MutationResult {
            probe_name: "p".into(),
            original_prompt: "orig".into(),
            mutated_prompt: "mut".into(),
            strategies_applied: vec![],
            validated: true,
            validation_risk: RiskLevel::Amber,
            requires_manual_review: false,
            notes: String::new(),
        }];
        let path = build_upgrade_path(&[], &mutations);
        assert_eq!(path.certified_prompts.len(), 1);
        assert!(matches!(
            path.certified_prompts[0].validation_risk,
            RiskLevel::Amber
        ));
    }

    #[test]
    fn latency_small_baseline_does_not_use_pct_spike() {
        let d = compute_latency_diff(12, 28);
        assert_eq!(d.delta_ms, 16);
        assert!(matches!(d.risk, RiskLevel::Green));
        assert!(matches!(d.direction, DriftDirection::Neutral));
    }

    #[test]
    fn latency_small_baseline_still_flags_large_absolute_slowdown() {
        let d = compute_latency_diff(10, 400);
        assert!(matches!(d.risk, RiskLevel::Red));
        assert!(matches!(d.direction, DriftDirection::Regression));
    }

    #[test]
    fn latency_large_baseline_uses_percent_bands() {
        let d = compute_latency_diff(200, 460);
        assert!(d.delta_pct > 1.0);
        assert!(matches!(d.risk, RiskLevel::Red));
        assert!(matches!(d.direction, DriftDirection::Regression));
    }

    #[test]
    fn max_risk_level_is_simple_maximum() {
        assert_eq!(
            max_risk_level([RiskLevel::Green, RiskLevel::Amber, RiskLevel::Amber]),
            RiskLevel::Amber
        );
        assert_eq!(
            max_risk_level([RiskLevel::Amber, RiskLevel::Green]),
            RiskLevel::Amber
        );
        assert_eq!(
            max_risk_level([RiskLevel::Green, RiskLevel::Red, RiskLevel::Amber]),
            RiskLevel::Red
        );
    }

    #[test]
    fn aggregate_probe_risk_amber_only_when_no_red_dimension() {
        let dims = test_probe_dimensions(
            RiskLevel::Amber,
            RiskLevel::Green,
            RiskLevel::Amber,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
        );
        let overall = max_risk_level(probe_dimension_risk_levels(&dims));
        assert!(
            matches!(overall, RiskLevel::Amber),
            "expected Amber overall, got {overall:?}"
        );
        assert!(!matches!(overall, RiskLevel::Red));
    }

    #[test]
    fn aggregate_probe_risk_red_only_when_a_dimension_is_red() {
        let dims = test_probe_dimensions(
            RiskLevel::Amber,
            RiskLevel::Green,
            RiskLevel::Amber,
            RiskLevel::Red,
            RiskLevel::Green,
            RiskLevel::Green,
        );
        let overall = max_risk_level(probe_dimension_risk_levels(&dims));
        assert!(matches!(overall, RiskLevel::Red));
    }

    #[test]
    fn latency_risk_does_not_affect_probe_overall_risk() {
        let dims = test_probe_dimensions(
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Red,
            RiskLevel::Green,
        );
        let overall = max_risk_level(probe_dimension_risk_levels(&dims));
        assert!(
            matches!(overall, RiskLevel::Green),
            "latency Red must not raise overall probe risk, got {overall:?}"
        );
    }

    #[test]
    fn compute_latency_summary_direction_and_note() {
        use crate::types::Probe;
        let probe = Probe {
            id: Uuid::new_v4(),
            name: "p".into(),
            category: ProbeCategory::Semantic,
            prompt: "x".into(),
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
        };
        let mk = |v1_ms, v2_ms| ProbeResult {
            probe: probe.clone(),
            v1_content: String::new(),
            v2_content: String::new(),
            overall_risk: RiskLevel::Green,
            overall_direction: DriftDirection::Neutral,
            drift_category: DriftCategory::NoSignificantDrift,
            drift_severity: DriftSeverity::Informational,
            dimensions: ProbeDimensions {
                latency: LatencyDiff {
                    risk: RiskLevel::Green,
                    direction: DriftDirection::Neutral,
                    v1_latency_ms: v1_ms,
                    v2_latency_ms: v2_ms,
                    delta_ms: v2_ms as i64 - v1_ms as i64,
                    delta_pct: 0.0,
                },
                ..test_probe_dimensions(
                    RiskLevel::Green,
                    RiskLevel::Green,
                    RiskLevel::Green,
                    RiskLevel::Green,
                    RiskLevel::Green,
                    RiskLevel::Green,
                )
            },
            notes: vec![],
        };
        let results = vec![mk(100, 70), mk(200, 150)];
        let summary = compute_latency_summary(&results);
        assert!(matches!(summary.direction, DriftDirection::Improvement));
        assert!(summary.note.contains("faster"));
        assert!(summary.note.contains("2 probes"));
    }

    #[test]
    fn latency_summary_delta_pct_is_percentage_not_ratio() {
        use crate::types::Probe;
        let probe = Probe {
            id: Uuid::new_v4(),
            name: "p".into(),
            category: ProbeCategory::Semantic,
            prompt: "x".into(),
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
        };
        let v1_ms = 7012_u64;
        let v2_ms = v1_ms - 4261;
        let pr = ProbeResult {
            probe,
            v1_content: String::new(),
            v2_content: String::new(),
            overall_risk: RiskLevel::Green,
            overall_direction: DriftDirection::Neutral,
            drift_category: DriftCategory::NoSignificantDrift,
            drift_severity: DriftSeverity::Informational,
            dimensions: ProbeDimensions {
                latency: LatencyDiff {
                    risk: RiskLevel::Green,
                    direction: DriftDirection::Neutral,
                    v1_latency_ms: v1_ms,
                    v2_latency_ms: v2_ms,
                    delta_ms: -(4261),
                    delta_pct: 0.0,
                },
                ..test_probe_dimensions(
                    RiskLevel::Green,
                    RiskLevel::Green,
                    RiskLevel::Green,
                    RiskLevel::Green,
                    RiskLevel::Green,
                    RiskLevel::Green,
                )
            },
            notes: vec![],
        };
        let summary = compute_latency_summary(&[pr]);
        assert!(
            (summary.delta_pct + 60.8).abs() < 0.5,
            "expected ~-60.8% stored as delta_pct, got {}",
            summary.delta_pct
        );
        assert!(summary.note.contains("61%") || summary.note.contains("61 %"));
    }

    fn test_probe_from_dims(category: ProbeCategory, dimensions: ProbeDimensions) -> ProbeResult {
        let probe = Probe {
            id: Uuid::new_v4(),
            name: "test".into(),
            category,
            prompt: String::new(),
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
        };
        let (overall_risk, drift_category, drift_severity) = compute_probe_risk(&probe, &dimensions);
        ProbeResult {
            probe,
            v1_content: String::new(),
            v2_content: String::new(),
            overall_risk,
            overall_direction: DriftDirection::Regression,
            drift_category,
            drift_severity,
            dimensions,
            notes: vec![],
        }
    }

    #[test]
    fn soap_explanation_compression_is_low_not_critical() {
        let mut dims = test_probe_dimensions(
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Amber,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
        );
        dims.morphology.v1.token_count = 100;
        dims.morphology.v2.token_count = 55;
        dims.morphology.delta.token_delta = -45;
        dims.morphology.delta.token_delta_pct = 0.45;
        dims.claim.preservation_score = 0.45;
        dims.claim.preservation_threshold = 0.50;
        dims.claim.drifted_claims.clear();
        let pr = test_probe_from_dims(ProbeCategory::Semantic, dims);
        assert_eq!(pr.drift_severity, DriftSeverity::Low);
        assert_eq!(pr.drift_category, DriftCategory::ContentCompression);
        assert!(matches!(pr.overall_risk, RiskLevel::Amber));
        assert!(!is_blocking_probe(&pr));
    }

    #[test]
    fn instruction_violation_is_critical() {
        let mut dims = test_probe_dimensions(
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
        );
        dims.instruction = Some(InstructionDiff {
            risk: RiskLevel::Red,
            direction: DriftDirection::Regression,
            v1_results: vec![],
            v2_results: vec![],
            v1_pass_rate: 1.0,
            v2_pass_rate: 0.0,
            regressions: vec!["failed check".into()],
        });
        let pr = test_probe_from_dims(ProbeCategory::Instruction, dims);
        assert_eq!(pr.drift_severity, DriftSeverity::Critical);
        assert_eq!(pr.drift_category, DriftCategory::CriticalRegression);
        assert!(matches!(pr.overall_risk, RiskLevel::Red));
        assert!(is_blocking_probe(&pr));
    }

    #[test]
    fn factual_anchor_drift_is_critical() {
        let mut dims = test_probe_dimensions(
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Red,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
        );
        dims.claim.drifted_claims.push(ClaimDrift {
            v1_claim: Claim {
                text: "rate is 4.5%".into(),
                information_density: 1.0,
                anchors: vec![],
            },
            v2_claim: Claim {
                text: "rate varies".into(),
                information_density: 1.0,
                anchors: vec![],
            },
            similarity: 0.5,
            drifted_anchors: vec![AnchorDrift {
                anchor_type: AnchorType::NumericValue,
                v1_value: "4.5%".into(),
                v2_value: "varies".into(),
            }],
        });
        let pr = test_probe_from_dims(ProbeCategory::Semantic, dims);
        assert_eq!(pr.drift_severity, DriftSeverity::Critical);
        assert!(is_blocking_probe(&pr));
    }

    #[test]
    fn refusal_flip_is_high_not_critical() {
        let mut dims = test_probe_dimensions(
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Red,
        );
        dims.refusal.new_refusal = true;
        dims.refusal.direction = DriftDirection::Regression;
        let pr = test_probe_from_dims(ProbeCategory::Refusal, dims);
        assert_eq!(pr.drift_severity, DriftSeverity::High);
        assert_eq!(pr.drift_category, DriftCategory::CriticalRegression);
        assert!(matches!(pr.overall_risk, RiskLevel::Amber));
        assert!(!is_blocking_probe(&pr));
        assert!(needs_attention_probe(&pr));
    }

    #[test]
    fn open_ended_low_preservation_is_low_severity() {
        let mut dims = test_probe_dimensions(
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Red,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
        );
        dims.claim.preservation_score = 0.30;
        dims.claim.preservation_threshold = 0.50;
        dims.claim.drifted_claims.clear();
        let pr = test_probe_from_dims(ProbeCategory::Semantic, dims);
        assert_eq!(pr.drift_severity, DriftSeverity::Low);
        assert_eq!(pr.drift_category, DriftCategory::ContentCompression);
        assert!(!is_blocking_probe(&pr));
    }

    #[test]
    fn migration_profile_speed_faster_headline() {
        use crate::types::Probe;
        let probe = Probe {
            id: Uuid::new_v4(),
            name: "p".into(),
            category: ProbeCategory::Semantic,
            prompt: "x".into(),
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
        };
        let mut dims = test_probe_dimensions(
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
        );
        dims.morphology.v1.token_count = 100;
        dims.morphology.v2.token_count = 100;
        let pr = ProbeResult {
            probe,
            v1_content: String::new(),
            v2_content: String::new(),
            overall_risk: RiskLevel::Green,
            overall_direction: DriftDirection::Neutral,
            drift_category: DriftCategory::NoSignificantDrift,
            drift_severity: DriftSeverity::Informational,
            dimensions: dims,
            notes: vec![],
        };
        let latency = LatencySummary {
            v1_avg_latency_ms: 200,
            v2_avg_latency_ms: 98,
            delta_ms: -102,
            delta_pct: -51.0,
            direction: DriftDirection::Improvement,
            note: String::new(),
        };
        let profile = compute_migration_profile(&[pr], &latency);
        assert_eq!(profile.speed_change.as_deref(), Some("51% faster"));
        assert!(profile.safe_to_upgrade);
        assert!(profile.headline.contains("51% faster"));
    }

    #[test]
    fn migration_profile_factual_error_headline() {
        let mut dims = test_probe_dimensions(
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
        );
        dims.factual = Some(FactualDiff {
            risk: RiskLevel::Red,
            direction: DriftDirection::Regression,
            v1_correct: true,
            v2_correct: false,
            v1_answer_extract: "yes".into(),
            v2_answer_extract: "no".into(),
            regression: true,
            improvement: false,
        });
        let pr = test_probe_from_dims(ProbeCategory::Factual, dims);
        let latency = compute_latency_summary(&[pr.clone()]);
        let profile = compute_migration_profile(&[pr], &latency);
        assert!(!profile.safe_to_upgrade);
        assert!(profile.headline.contains("critical regression"));
        assert!(profile.headline.contains("not recommended"));
    }

    #[test]
    fn migration_profile_instruction_and_content_drift_headline() {
        let mut instr_dims = test_probe_dimensions(
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
        );
        instr_dims.schema = Some(SchemaDiff {
            risk: RiskLevel::Red,
            direction: DriftDirection::Regression,
            v1_valid_json: true,
            v2_valid_json: false,
            v1_schema_valid: true,
            v2_schema_valid: false,
            v1_missing_fields: vec![],
            v2_missing_fields: vec!["active".into()],
            v1_extra_fields: vec![],
            v2_extra_fields: vec![],
            field_type_changes: vec![],
        });
        let instr = test_probe_from_dims(ProbeCategory::Schema, instr_dims);

        let mut content_dims = test_probe_dimensions(
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Red,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
        );
        content_dims.claim.drifted_claims.push(ClaimDrift {
            v1_claim: Claim {
                text: "rate 4.5%".into(),
                information_density: 1.0,
                anchors: vec![],
            },
            v2_claim: Claim {
                text: "rate varies".into(),
                information_density: 1.0,
                anchors: vec![],
            },
            similarity: 0.5,
            drifted_anchors: vec![AnchorDrift {
                anchor_type: AnchorType::NumericValue,
                v1_value: "4.5%".into(),
                v2_value: "varies".into(),
            }],
        });
        let content = test_probe_from_dims(ProbeCategory::Semantic, content_dims);

        let latency = compute_latency_summary(&[instr.clone(), content.clone()]);
        let profile = compute_migration_profile(&[instr, content], &latency);
        assert!(profile.headline.contains("critical regression"));
        assert!(profile.headline.contains("not recommended"));
    }

    #[test]
    fn migration_profile_policy_only_headline() {
        let mut dims = test_probe_dimensions(
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Red,
        );
        dims.refusal.new_refusal = true;
        dims.refusal.direction = DriftDirection::Regression;
        let pr = test_probe_from_dims(ProbeCategory::Refusal, dims);
        let latency = compute_latency_summary(&[pr.clone()]);
        let profile = compute_migration_profile(&[pr], &latency);
        assert!(profile.safe_to_upgrade);
        assert!(profile.headline.contains("warrant review"));
    }

    #[test]
    fn migration_profile_safe_equivalent_headline() {
        let dims = test_probe_dimensions(
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
        );
        let pr = test_probe_from_dims(ProbeCategory::Semantic, dims);
        let latency = compute_latency_summary(std::slice::from_ref(&pr));
        let profile = compute_migration_profile(std::slice::from_ref(&pr), &latency);
        assert!(profile.safe_to_upgrade);
        assert_eq!(
            profile.headline,
            "v2 is behaviourally equivalent — safe to upgrade"
        );
    }

    #[test]
    fn migration_profile_structural_inconsistency() {
        let mut probes = Vec::new();
        for _ in 0..5 {
            let mut dims = test_probe_dimensions(
                RiskLevel::Amber,
                RiskLevel::Green,
                RiskLevel::Green,
                RiskLevel::Green,
                RiskLevel::Green,
                RiskLevel::Green,
            );
            dims.morphology.v1.token_count = 100;
            dims.morphology.v2.token_count = 50;
            dims.morphology.delta.token_delta = -50;
            dims.morphology.delta.token_delta_pct = 0.50;
            probes.push(test_probe_from_dims(ProbeCategory::Semantic, dims));
        }
        let latency = compute_latency_summary(&probes);
        let profile = compute_migration_profile(&probes, &latency);
        assert_eq!(
            profile.reliability_change.as_deref(),
            Some("less structurally consistent")
        );
    }

    #[test]
    fn upgrade_path_compression_is_not_blocking() {
        let mut dims = test_probe_dimensions(
            RiskLevel::Red,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
            RiskLevel::Green,
        );
        dims.morphology.v1.token_count = 100;
        dims.morphology.v2.token_count = 50;
        dims.morphology.delta.token_delta = -50;
        dims.morphology.delta.token_delta_pct = 0.50;
        let pr = test_probe_from_dims(ProbeCategory::Semantic, dims);
        assert_eq!(pr.drift_category, DriftCategory::ContentCompression);
        assert_eq!(pr.drift_severity, DriftSeverity::Low);
        let path = build_upgrade_path(&[pr], &[]);
        assert!(path.blocking_regressions.is_empty());
        assert_eq!(path.neutral_changes.len(), 1);
    }

    fn test_probe_dimensions(
        morphology: RiskLevel,
        tone: RiskLevel,
        claim: RiskLevel,
        semantic: RiskLevel,
        latency: RiskLevel,
        refusal: RiskLevel,
    ) -> ProbeDimensions {
        let semantic_has_score = !matches!(semantic, RiskLevel::Green);
        ProbeDimensions {
            morphology: MorphologyDiff {
                risk: morphology,
                direction: DriftDirection::Neutral,
                v1: MorphologyMetrics {
                    token_count: 1,
                    word_count: 1,
                    sentence_count: 1,
                    paragraph_count: 1,
                    has_lists: false,
                    has_headers: false,
                    has_code_blocks: false,
                    has_caveats: false,
                    response_type: ResponseType::SingleLine,
                },
                v2: MorphologyMetrics {
                    token_count: 1,
                    word_count: 1,
                    sentence_count: 1,
                    paragraph_count: 1,
                    has_lists: false,
                    has_headers: false,
                    has_code_blocks: false,
                    has_caveats: false,
                    response_type: ResponseType::SingleLine,
                },
                delta: MorphologyDelta {
                    token_delta: 0,
                    token_delta_pct: 0.0,
                    response_type_changed: false,
                    structure_changed: false,
                },
            },
            tone: ToneDiff {
                risk: tone,
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
                risk: refusal,
                direction: DriftDirection::Neutral,
                v1_refused: false,
                v2_refused: false,
                new_refusal: false,
                refusal_lifted: false,
            },
            semantic: SemanticDiff {
                risk: semantic,
                direction: DriftDirection::Neutral,
                cosine_similarity: if semantic_has_score { Some(0.5) } else { None },
                semantic_scoring_disabled: false,
                disabled_reason: None,
                flagged_for_review: false,
                similarity_threshold: 0.85,
            },
            claim: ClaimDiff {
                risk: claim,
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
                risk: latency,
                direction: DriftDirection::Neutral,
                v1_latency_ms: 100,
                v2_latency_ms: 100,
                delta_ms: 0,
                delta_pct: 0.0,
            },
            consistency: None,
            custom_assertions: None,
        }
    }
}