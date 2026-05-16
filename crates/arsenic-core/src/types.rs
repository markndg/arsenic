use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probe {
    pub id: Uuid,
    pub name: String,
    pub category: ProbeCategory,
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub known_answer: Option<String>,
    pub expected_schema: Option<serde_json::Value>,
    pub instructions: Vec<ProbeInstruction>,
    pub tags: Vec<String>,
    pub source: ProbeSource,
    /// v2: expected response length for morphology drift direction.
    #[serde(default)]
    pub expected_verbosity: Option<ExpectedVerbosity>,
    /// v2: expected tone for tone drift direction.
    #[serde(default)]
    pub expected_tone: Option<ExpectedTonePreference>,
    /// v2: whether refusals are expected for this probe.
    #[serde(default)]
    pub refusal_expectation: Option<RefusalExpectation>,
    /// v2: hint consumed by the mutation engine.
    #[serde(default)]
    pub mutation_hint: Option<String>,
    /// v2: user-defined checks (corpus probes).
    #[serde(default)]
    pub custom_assertions: Vec<CustomAssertion>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "PascalCase")]
pub enum ProbeCategory {
    Morphology,
    Tone,
    Factual,
    Schema,
    Instruction,
    Refusal,
    Semantic,
}

impl ProbeCategory {
    /// Minimum preservation score before claim drift is Red (category-specific).
    pub fn preservation_threshold(self) -> f64 {
        match self {
            Self::Factual | Self::Schema | Self::Instruction => 0.70,
            Self::Morphology | Self::Tone | Self::Semantic | Self::Refusal => 0.50,
        }
    }

    /// Upper preservation band before Green (Amber between this and [`Self::preservation_threshold`]).
    pub fn preservation_amber_threshold(self) -> f64 {
        match self {
            Self::Factual | Self::Schema | Self::Instruction => 0.90,
            Self::Morphology | Self::Tone | Self::Semantic | Self::Refusal => 0.70,
        }
    }

    /// Unmatched v1 claims alone force Red (tight factual/schema probes).
    pub fn dropped_claims_force_red(self) -> bool {
        matches!(self, Self::Factual | Self::Schema | Self::Instruction)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProbeSource {
    Standard,
    UserDefined,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeInstruction {
    pub description: String,
    pub check: InstructionCheck,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum ExpectedVerbosity {
    Concise,
    Moderate,
    Detailed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum ExpectedTonePreference {
    Formal,
    Neutral,
    Casual,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum RefusalExpectation {
    ShouldAnswer,
    ShouldRefuse,
    Either,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CustomAssertion {
    pub description: String,
    pub check: InstructionCheck,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "PascalCase", tag = "type", content = "value")]
pub enum InstructionCheck {
    MaxWords(usize),
    MinWords(usize),
    MustContain(String),
    MustNotContain(String),
    MustStartWith(String),
    MustEndWith(String),
    OutputFormat(OutputFormat),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub enum OutputFormat {
    Json,
    Markdown,
    PlainText,
    BulletList,
    NumberedList,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    pub probe_id: Uuid,
    pub model_label: String,
    pub model_id: String,
    pub content: String,
    pub token_count: usize,
    pub latency_ms: u64,
    pub finish_reason: FinishReason,
    pub timestamp: DateTime<Utc>,
    pub raw: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Refusal,
    Error,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsePair {
    pub probe: Probe,
    /// Primary run (first) — used for display and single-run comparisons.
    pub v1: ModelResponse,
    pub v2: ModelResponse,
    /// v2: all runs when `--consistency-runs` > 1; empty means single-run only.
    #[serde(default)]
    pub v1_runs: Vec<ModelResponse>,
    #[serde(default)]
    pub v2_runs: Vec<ModelResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub run_id: Uuid,
    pub generated_at: DateTime<Utc>,
    pub v1_model: ModelInfo,
    pub v2_model: ModelInfo,
    pub overall_risk: RiskLevel,
    pub summary: ReportSummary,
    pub probe_results: Vec<ProbeResult>,
    pub dimension_summaries: DimensionSummaries,
    /// v2: valence counts across all dimension rows (approximate from probe-level worst direction).
    #[serde(default)]
    pub valence_summary: ValenceSummary,
    /// v2: structured upgrade path (blocking / verify / neutral / certified prompts).
    #[serde(default)]
    pub upgrade_path: UpgradePathReport,
    /// v2: prompt mutation results when `--mutate` was used.
    #[serde(default)]
    pub mutation_results: Vec<MutationResult>,
    /// Run-level latency comparison (observational; does not affect risk or upgrade path).
    #[serde(default)]
    pub latency_summary: LatencySummary,
    /// Plain-English behavioural fingerprint of v2 relative to v1.
    #[serde(default)]
    pub migration_profile: MigrationProfile,
}

/// Headline summary of how v2 behaves relative to v1 across the run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MigrationProfile {
    pub speed_change: Option<String>,
    pub verbosity_change: Option<String>,
    pub style_change: Option<String>,
    pub reliability_change: Option<String>,
    pub headline: String,
    pub safe_to_upgrade: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencySummary {
    pub v1_avg_latency_ms: u64,
    pub v2_avg_latency_ms: u64,
    pub delta_ms: i64,
    /// Percentage change in average latency (e.g. `-60.8` = target 60.8% faster than baseline).
    pub delta_pct: f64,
    pub direction: DriftDirection,
    /// Plain-language summary, e.g. "v2 responded 23% faster on average across 18 probes".
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub label: String,
    pub model_id: String,
    pub adapter: String,
    pub endpoint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_probes: usize,
    pub probes_green: usize,
    pub probes_amber: usize,
    pub probes_red: usize,
    pub safe_to_upgrade: bool,
    pub requires_manual_review: usize,
    pub auto_remediation_candidates: usize,
    /// v2: probes whose overall drift direction is regression / improvement / neutral.
    #[serde(default)]
    pub probe_regressions: usize,
    #[serde(default)]
    pub probe_improvements: usize,
    #[serde(default)]
    pub probe_neutral: usize,
    /// Probes classified as critical regressions (blocking when Red/Amber).
    #[serde(default, alias = "drift_behavioural_regressions")]
    pub drift_critical_regressions: usize,
    #[serde(default)]
    pub drift_policy: usize,
    #[serde(default)]
    pub drift_fidelity: usize,
    #[serde(default)]
    pub drift_structural: usize,
    #[serde(default)]
    pub drift_content_compression: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub probe: Probe,
    pub v1_content: String,
    pub v2_content: String,
    pub overall_risk: RiskLevel,
    /// v2: worst-case drift direction for this probe.
    #[serde(default)]
    pub overall_direction: DriftDirection,
    /// What kind of drift this probe represents for upgrade-path routing.
    #[serde(default)]
    pub drift_category: DriftCategory,
    /// Severity-weighted drift level (drives overall risk and category).
    #[serde(default)]
    pub drift_severity: DriftSeverity,
    pub dimensions: ProbeDimensions,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeDimensions {
    pub morphology: MorphologyDiff,
    pub tone: ToneDiff,
    pub factual: Option<FactualDiff>,
    pub schema: Option<SchemaDiff>,
    pub instruction: Option<InstructionDiff>,
    pub refusal: RefusalDiff,
    pub semantic: SemanticDiff,
    #[serde(default)]
    pub claim: ClaimDiff,
    #[serde(default)]
    pub latency: LatencyDiff,
    pub consistency: Option<ConsistencyDiff>,
    pub custom_assertions: Option<CustomAssertionDiff>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "PascalCase")]
pub enum RiskLevel {
    #[default]
    Green,
    Amber,
    Red,
}

/// Severity-weighted drift level — primary driver of probe risk and upgrade routing.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Default)]
#[serde(rename_all = "PascalCase")]
pub enum DriftSeverity {
    #[default]
    Informational = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Taxonomy for what kind of drift a probe shows — drives upgrade-path blocking rules.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "PascalCase")]
pub enum DriftCategory {
    /// Factual error, broken schema, or anchor value changed.
    #[serde(alias = "BehaviouralRegression")]
    CriticalRegression,
    /// Refusal boundary shifted.
    PolicyDrift,
    /// Same answer, different style or detail level.
    FidelityDrift,
    /// Format or layout changed; content equivalent.
    StructuralDrift,
    /// Shorter, less exhaustive, but not wrong.
    ContentCompression,
    /// Green — no meaningful change.
    #[default]
    NoSignificantDrift,
}

impl DriftCategory {
    /// Only Red + CriticalRegression blocks rollout (see [`DriftSeverity::Critical`]).
    pub fn is_blocking(self, risk: &RiskLevel) -> bool {
        matches!(self, DriftCategory::CriticalRegression) && matches!(risk, RiskLevel::Red)
    }
}

/// v2: whether drift on a dimension is better, worse, or neutral for the upgrade.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "PascalCase")]
pub enum DriftDirection {
    #[default]
    Neutral,
    Improvement,
    Regression,
    NotApplicable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphologyDiff {
    pub risk: RiskLevel,
    pub direction: DriftDirection,
    pub v1: MorphologyMetrics,
    pub v2: MorphologyMetrics,
    pub delta: MorphologyDelta,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct MorphologyMetrics {
    pub token_count: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub paragraph_count: usize,
    pub has_lists: bool,
    pub has_headers: bool,
    pub has_code_blocks: bool,
    pub has_caveats: bool,
    pub response_type: ResponseType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum ResponseType {
    SingleLine,
    ShortParagraph,
    LongParagraph,
    MultiParagraph,
    Structured,
    Refusal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphologyDelta {
    pub token_delta: i64,
    pub token_delta_pct: f64,
    pub response_type_changed: bool,
    pub structure_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneDiff {
    pub risk: RiskLevel,
    pub direction: DriftDirection,
    pub v1: ToneMetrics,
    pub v2: ToneMetrics,
    pub delta: ToneDelta,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ToneMetrics {
    pub formality_score: f64,
    pub assertiveness_score: f64,
    pub hedge_word_count: usize,
    pub contraction_count: usize,
    pub average_sentence_length: f64,
    pub passive_voice_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneDelta {
    pub formality_delta: f64,
    pub assertiveness_delta: f64,
    pub hedge_word_delta: i64,
    pub significant_shift: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactualDiff {
    pub risk: RiskLevel,
    pub direction: DriftDirection,
    pub v1_correct: bool,
    pub v2_correct: bool,
    pub v1_answer_extract: String,
    pub v2_answer_extract: String,
    pub regression: bool,
    pub improvement: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaDiff {
    pub risk: RiskLevel,
    pub direction: DriftDirection,
    pub v1_valid_json: bool,
    pub v2_valid_json: bool,
    pub v1_schema_valid: bool,
    pub v2_schema_valid: bool,
    pub v1_missing_fields: Vec<String>,
    pub v2_missing_fields: Vec<String>,
    pub v1_extra_fields: Vec<String>,
    pub v2_extra_fields: Vec<String>,
    pub field_type_changes: Vec<FieldTypeChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldTypeChange {
    pub field: String,
    pub v1_type: String,
    pub v2_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionDiff {
    pub risk: RiskLevel,
    pub direction: DriftDirection,
    pub v1_results: Vec<InstructionCheckResult>,
    pub v2_results: Vec<InstructionCheckResult>,
    pub v1_pass_rate: f64,
    pub v2_pass_rate: f64,
    pub regressions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionCheckResult {
    pub instruction: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefusalDiff {
    pub risk: RiskLevel,
    pub direction: DriftDirection,
    pub v1_refused: bool,
    pub v2_refused: bool,
    pub new_refusal: bool,
    pub refusal_lifted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDiff {
    pub risk: RiskLevel,
    pub direction: DriftDirection,
    /// `None` when `--no-semantic` or semantic engine unavailable (no misleading score).
    pub cosine_similarity: Option<f64>,
    pub semantic_scoring_disabled: bool,
    pub disabled_reason: Option<String>,
    pub flagged_for_review: bool,
    pub similarity_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyDiff {
    pub risk: RiskLevel,
    pub direction: DriftDirection,
    pub v1_latency_ms: u64,
    pub v2_latency_ms: u64,
    pub delta_ms: i64,
    pub delta_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyDiff {
    pub risk: RiskLevel,
    pub direction: DriftDirection,
    pub v1_runs: usize,
    pub v2_runs: usize,
    pub v1_variance: f64,
    pub v2_variance: f64,
    pub v1_consistent: bool,
    pub v2_consistent: bool,
    pub consistency_regression: bool,
    pub consistency_improvement: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimDiff {
    pub risk: RiskLevel,
    pub direction: DriftDirection,
    pub v1_claims: Vec<Claim>,
    pub v2_claims: Vec<Claim>,
    pub matched_pairs: Vec<ClaimMatch>,
    pub dropped_claims: Vec<Claim>,
    pub new_claims: Vec<Claim>,
    pub drifted_claims: Vec<ClaimDrift>,
    pub preservation_score: f64,
    /// Red-band preservation cutoff used for this probe (from [`ProbeCategory::preservation_threshold`]).
    #[serde(default = "default_preservation_threshold")]
    pub preservation_threshold: f64,
}

fn default_preservation_threshold() -> f64 {
    0.70
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub text: String,
    pub information_density: f64,
    pub anchors: Vec<ClaimAnchor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimAnchor {
    pub anchor_type: AnchorType,
    pub value: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum AnchorType {
    NumericValue,
    DateOrYear,
    ProperNoun,
    KeyTerm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimMatch {
    pub v1_claim: Claim,
    pub v2_claim: Claim,
    pub similarity: f64,
    pub anchor_agreement: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimDrift {
    pub v1_claim: Claim,
    pub v2_claim: Claim,
    pub similarity: f64,
    pub drifted_anchors: Vec<AnchorDrift>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorDrift {
    pub anchor_type: AnchorType,
    pub v1_value: String,
    pub v2_value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomAssertionDiff {
    pub risk: RiskLevel,
    pub direction: DriftDirection,
    pub v1_results: Vec<InstructionCheckResult>,
    pub v2_results: Vec<InstructionCheckResult>,
    pub v1_pass_rate: f64,
    pub v2_pass_rate: f64,
    pub regressions: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValenceSummary {
    pub probe_regressions: usize,
    pub probe_improvements: usize,
    pub probe_neutral: usize,
    pub dimension_regressions: usize,
    pub dimension_improvements: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpgradePathReport {
    /// Blocking probes with critical regressions (factual, schema, anchor drift).
    #[serde(default, alias = "behavioural_regressions")]
    pub critical_regressions: Vec<UpgradePathItem>,
    /// Blocking probes with policy / refusal drift.
    #[serde(default)]
    pub policy_changes: Vec<UpgradePathItem>,
    /// All blocking items (critical + policy); kept for backward-compatible JSON consumers.
    pub blocking_regressions: Vec<UpgradePathItem>,
    pub improvements_to_verify: Vec<UpgradePathItem>,
    pub neutral_changes: Vec<UpgradePathItem>,
    pub certified_prompts: Vec<CertifiedPromptDiff>,
    pub remediation: RemediationCounts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpgradePathItem {
    pub probe_name: String,
    pub category: ProbeCategory,
    pub overall_risk: RiskLevel,
    pub overall_direction: DriftDirection,
    #[serde(default)]
    pub drift_category: DriftCategory,
    pub summary: String,
    pub certified_mutation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertifiedPromptDiff {
    pub probe_name: String,
    pub original_prompt: String,
    pub mutated_prompt: String,
    pub validated: bool,
    /// Overall probe risk after re-running the mutated prompt against v2.
    #[serde(default)]
    pub validation_risk: RiskLevel,
    /// Strategies cumulatively applied to produce `mutated_prompt` (mirrors validated [`MutationResult`]).
    #[serde(default)]
    pub strategies_applied: Vec<MutationStrategy>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RemediationCounts {
    pub prompt_changes_suggested: usize,
    pub manual_review: usize,
    pub auto_certified: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationResult {
    pub probe_name: String,
    pub original_prompt: String,
    pub mutated_prompt: String,
    pub strategies_applied: Vec<MutationStrategy>,
    pub validated: bool,
    pub validation_risk: RiskLevel,
    pub requires_manual_review: bool,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationStrategy {
    AddLengthConstraint { max_words: usize },
    AddDetailInstruction { min_words: usize },
    AddDirectnessInstruction,
    AddFormalityInstruction,
    AddConfidenceInstruction,
    SoftenPhrasing,
    AddEducationalContext,
    AddClaimInstruction { required_values: Vec<String> },
    /// Long-form drift: require coverage of baseline section topics (not specific values).
    AddTopicCoverageInstruction { topics: Vec<String> },
    AddPrecisionInstruction,
    ReinforceInstruction { instruction_text: String },
    Custom { hint: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionSummaries {
    pub morphology: DimensionSummary,
    pub tone: DimensionSummary,
    pub factual: DimensionSummary,
    pub schema: DimensionSummary,
    pub instruction: DimensionSummary,
    pub refusal: DimensionSummary,
    pub semantic: DimensionSummary,
    #[serde(default)]
    pub claim: DimensionSummary,
    #[serde(default)]
    pub latency: DimensionSummary,
    #[serde(default)]
    pub consistency: DimensionSummary,
    #[serde(default)]
    pub custom_assertions: DimensionSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionSummary {
    pub probes_affected: usize,
    pub worst_risk: RiskLevel,
    pub notes: Vec<String>,
    /// Among non-Green probes for this dimension, count of regression / improvement / neutral / N/A drift directions.
    #[serde(default)]
    pub drift_regressions: usize,
    #[serde(default)]
    pub drift_improvements: usize,
    #[serde(default)]
    pub drift_neutral: usize,
    #[serde(default)]
    pub drift_not_applicable: usize,
}

impl Default for DimensionSummary {
    fn default() -> Self {
        Self {
            probes_affected: 0,
            worst_risk: RiskLevel::Green,
            notes: Vec::new(),
            drift_regressions: 0,
            drift_improvements: 0,
            drift_neutral: 0,
            drift_not_applicable: 0,
        }
    }
}

impl Default for ClaimDiff {
    fn default() -> Self {
        Self {
            risk: RiskLevel::Green,
            direction: DriftDirection::NotApplicable,
            v1_claims: Vec::new(),
            v2_claims: Vec::new(),
            matched_pairs: Vec::new(),
            dropped_claims: Vec::new(),
            new_claims: Vec::new(),
            drifted_claims: Vec::new(),
            preservation_score: 1.0,
            preservation_threshold: ProbeCategory::Semantic.preservation_threshold(),
        }
    }
}

impl Default for LatencyDiff {
    fn default() -> Self {
        Self {
            risk: RiskLevel::Green,
            direction: DriftDirection::NotApplicable,
            v1_latency_ms: 0,
            v2_latency_ms: 0,
            delta_ms: 0,
            delta_pct: 0.0,
        }
    }
}
