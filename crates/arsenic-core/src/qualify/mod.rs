//! Continuous model qualification against an application contract + baseline.

pub mod capture;
pub mod effective;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use crate::claim::ClaimExtractor;
use crate::mutation::apply_mutations;
use crate::refusal::RefusalDetector;
use crate::types::{MutationStrategy, Probe, ProbeCategory, ProbeSource};
use uuid::Uuid;

use crate::contract::{
    ApplicationContract, ContractExpectation, ContractItem, ContractItemKind, ContractSeverity,
};

pub use capture::{
    behaviour_from_error, behaviour_from_response, candidate_input_fingerprint,
    capability_failures, capture_contract_live, classify_provider_error, content_hash,
    extract_tool_calls, json_hash, redact_json, redact_secrets, scenarios_from_contract,
    CaptureSource, ExecutionErrorKind, ExecutionMeta, LiveCaptureConfig, ProviderCapabilities,
};
pub use effective::{
    aggregate_migration, assess_all, assess_qualification, effective_to_json, latest_per_candidate,
    EffectiveQualification, EvidenceValidity, MigrationRecommendation,
};

pub const QUALIFICATION_SCHEMA_VERSION: u32 = 1;

/// Overall qualification decision (deterministic from item outcomes).
#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum QualificationDecision {
    #[default]
    Pass,
    PassWithWarnings,
    PassWithPatch,
    Review,
    Block,
    Stale,
}

impl QualificationDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::PassWithWarnings => "PASS_WITH_WARNINGS",
            Self::PassWithPatch => "PASS_WITH_PATCH",
            Self::Review => "REVIEW",
            Self::Block => "BLOCK",
            Self::Stale => "STALE",
        }
    }

    /// CI exit code: 0 pass/warn, 1 block, 2 review, 3 error (caller).
    pub fn ci_exit_code(self) -> i32 {
        match self {
            Self::Pass | Self::PassWithWarnings | Self::PassWithPatch => 0,
            Self::Block => 1,
            Self::Review | Self::Stale => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ItemOutcome {
    Pass,
    Warn,
    Review,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvidenceSnippet {
    pub label: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContractItemResult {
    pub item_id: String,
    pub kind: ContractItemKind,
    pub severity: ContractSeverity,
    pub outcome: ItemOutcome,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regression_type: Option<String>,
    pub baseline_evidence: Vec<EvidenceSnippet>,
    pub candidate_evidence: Vec<EvidenceSnippet>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparator: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RepairAttempt {
    pub attempt_index: u32,
    pub strategies: Vec<String>,
    pub mutated_prompt: String,
    pub outcome: ItemOutcome,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidatedPatch {
    pub qualification_id: String,
    pub model: String,
    pub contract_hash: String,
    pub strategies: Vec<String>,
    pub prompt_diff: String,
    pub original_prompt: String,
    pub patched_prompt: String,
    pub revalidated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CostLatencyDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_delta_pct: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_delta_pct: Option<f64>,
    #[serde(default)]
    pub estimate: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QualificationResult {
    pub schema_version: u32,
    pub id: String,
    pub application_id: String,
    pub contract_version: u32,
    pub contract_hash: String,
    pub baseline_id: String,
    pub baseline_hash: String,
    pub candidate_model: String,
    pub decision: QualificationDecision,
    /// Preserved when later derived as STALE; never overwritten by stale marking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_decision: Option<QualificationDecision>,
    pub created_at: String,
    pub item_results: Vec<ContractItemResult>,
    #[serde(default)]
    pub repair_attempts: Vec<RepairAttempt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validated_patch: Option<ValidatedPatch>,
    #[serde(default)]
    pub cost_latency: CostLatencyDelta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint_delta_note: Option<String>,
    /// Derived view flag only — historical decision stays in `decision`/`original_decision`.
    #[serde(default)]
    pub stale: bool,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    /// Normalised evidence for deterministic replay without contacting providers.
    #[serde(default)]
    pub captured_behaviours: BTreeMap<String, CapturedBehaviour>,
    /// True when any scenario hit a provider/runtime error (not behavioural).
    #[serde(default)]
    pub has_execution_errors: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_fingerprint: Option<String>,
}

impl QualificationResult {
    pub fn blockers(&self) -> impl Iterator<Item = &ContractItemResult> {
        self.item_results.iter().filter(|r| {
            matches!(r.outcome, ItemOutcome::Fail)
                && matches!(r.severity, ContractSeverity::Block)
                && r.regression_type.as_deref() != Some("execution_error")
        })
    }

    pub fn reviews(&self) -> impl Iterator<Item = &ContractItemResult> {
        self.item_results.iter().filter(|r| {
            matches!(r.outcome, ItemOutcome::Fail | ItemOutcome::Review)
                && matches!(r.severity, ContractSeverity::Review)
                || matches!(r.outcome, ItemOutcome::Review)
        })
    }

    /// Effective decision given current contract/baseline hashes (does not mutate history).
    /// Prefer [`assess_qualification`] for full evidence-validity + migration recommendation.
    pub fn effective_decision(
        &self,
        current_contract_hash: &str,
        current_baseline_hash: &str,
    ) -> QualificationDecision {
        assess_qualification(self, current_contract_hash, current_baseline_hash).effective_decision
    }

    /// Fail-closed migration recommendation for the current contract/baseline.
    pub fn migration_recommendation(
        &self,
        current_contract_hash: &str,
        current_baseline_hash: &str,
    ) -> MigrationRecommendation {
        assess_qualification(self, current_contract_hash, current_baseline_hash)
            .migration_recommendation
    }
}

/// Captured behaviour for one prompt under one model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CapturedBehaviour {
    pub prompt_id: String,
    pub content: String,
    #[serde(default)]
    pub latency_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    #[serde(default)]
    pub tool_calls: Vec<CapturedToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_error: Option<ExecutionErrorKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_error_message: Option<String>,
    #[serde(default)]
    pub execution_meta: ExecutionMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CapturedToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: BTreeMap<String, serde_json::Value>,
}

/// Immutable production baseline snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaselineSnapshot {
    pub schema_version: u32,
    pub id: String,
    pub version: u32,
    pub application_id: String,
    pub contract_version: u32,
    pub contract_hash: String,
    pub model: String,
    pub provider_model_id: String,
    pub created_at: String,
    pub behaviours: BTreeMap<String, CapturedBehaviour>,
    #[serde(default)]
    pub prompt_hashes: BTreeMap<String, String>,
    #[serde(default)]
    pub qualification_thresholds: QualificationThresholds,
    #[serde(default)]
    pub runtime: BTreeMap<String, String>,
}

impl BaselineSnapshot {
    pub fn content_hash(&self) -> String {
        let mut hasher = Sha256::new();
        let payload = serde_json::json!({
            "id": self.id,
            "version": self.version,
            "contract_hash": self.contract_hash,
            "model": self.model,
            "behaviours": self.behaviours,
        });
        hasher.update(serde_json::to_vec(&payload).unwrap_or_default());
        format!("{:x}", hasher.finalize())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QualificationThresholds {
    /// When true, any review-severity failure yields REVIEW (default).
    #[serde(default = "default_true")]
    pub review_escalates: bool,
    /// When true, warn-severity failures alone yield PASS_WITH_WARNINGS.
    #[serde(default = "default_true")]
    pub warnings_only_pass: bool,
}

fn default_true() -> bool {
    true
}

impl Default for QualificationThresholds {
    fn default() -> Self {
        Self {
            review_escalates: true,
            warnings_only_pass: true,
        }
    }
}

/// Derive overall decision deterministically from item results.
pub fn decide_overall(
    item_results: &[ContractItemResult],
    thresholds: &QualificationThresholds,
    has_validated_patch: bool,
) -> QualificationDecision {
    let mut has_block = false;
    let mut has_review = false;
    let mut has_warn = false;
    for r in item_results {
        match (r.outcome, r.severity) {
            (ItemOutcome::Fail, ContractSeverity::Block) => has_block = true,
            // Review-severity failures and explicit Review outcomes always require review.
            // `review_escalates` is retained on thresholds for serde compatibility but must
            // never suppress review evidence into PASS (fail-closed).
            (ItemOutcome::Fail, ContractSeverity::Review) | (ItemOutcome::Review, _) => {
                has_review = true;
            }
            (ItemOutcome::Fail, ContractSeverity::Warn) | (ItemOutcome::Warn, _) => {
                has_warn = true;
            }
            (ItemOutcome::Fail, ContractSeverity::Info) => has_warn = true,
            _ => {}
        }
    }
    if has_block {
        return QualificationDecision::Block;
    }
    if has_review {
        return QualificationDecision::Review;
    }
    if has_validated_patch {
        return QualificationDecision::PassWithPatch;
    }
    if has_warn && thresholds.warnings_only_pass {
        return QualificationDecision::PassWithWarnings;
    }
    if has_warn {
        return QualificationDecision::Review;
    }
    QualificationDecision::Pass
}

fn claim_present(text: &str, claim: &str) -> bool {
    let norm = |s: &str| {
        s.chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .to_lowercase()
    };
    let t = norm(text);
    let c = norm(claim);
    if t.contains(&c) {
        return true;
    }
    // Fallback: any extracted claim with high token overlap.
    let claims = ClaimExtractor::extract(text);
    let claim_tokens: Vec<_> = c.split_whitespace().filter(|w| w.len() > 2).collect();
    if claim_tokens.is_empty() {
        return false;
    }
    claims.iter().any(|extracted| {
        let e = norm(&extracted.text);
        let hit = claim_tokens.iter().filter(|tok| e.contains(**tok)).count();
        hit * 100 / claim_tokens.len() >= 70
    })
}

fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Evaluate one contract item against baseline + candidate captures.
pub fn evaluate_item(
    item: &ContractItem,
    baseline: Option<&CapturedBehaviour>,
    candidate: Option<&CapturedBehaviour>,
) -> ContractItemResult {
    let missing = |side: &str| ContractItemResult {
        item_id: item.id.clone(),
        kind: item.kind.clone(),
        severity: item.severity,
        outcome: ItemOutcome::Fail,
        reason: format!("Missing {side} behaviour for prompt '{}'", item.prompt),
        regression_type: Some("missing_capture".into()),
        baseline_evidence: vec![],
        candidate_evidence: vec![],
        comparator: None,
    };

    let Some(cand) = candidate else {
        return missing("candidate");
    };

    // Provider/runtime failures are not behavioural PASS/BLOCK.
    if let Some(err) = &cand.execution_error {
        return ContractItemResult {
            item_id: item.id.clone(),
            kind: item.kind.clone(),
            severity: item.severity,
            outcome: ItemOutcome::Review,
            reason: format!(
                "{}: {}",
                err.as_str(),
                cand.execution_error_message
                    .as_deref()
                    .unwrap_or("provider/runtime failure")
            ),
            regression_type: Some("execution_error".into()),
            baseline_evidence: vec![],
            candidate_evidence: vec![EvidenceSnippet {
                label: "Execution".into(),
                text: err.as_str().into(),
            }],
            comparator: Some("execution".into()),
        };
    }

    let base_text = baseline.map(|b| b.content.as_str()).unwrap_or("");

    match &item.expectation {
        ContractExpectation::ContainsClaim { text } => {
            let present = claim_present(&cand.content, text);
            let base_had = if base_text.is_empty() {
                true
            } else {
                claim_present(base_text, text)
            };
            let outcome = if present {
                ItemOutcome::Pass
            } else if matches!(
                item.severity,
                ContractSeverity::Warn | ContractSeverity::Info
            ) {
                ItemOutcome::Warn
            } else {
                ItemOutcome::Fail
            };
            ContractItemResult {
                item_id: item.id.clone(),
                kind: item.kind.clone(),
                severity: item.severity,
                outcome,
                reason: if present {
                    "Required claim present".into()
                } else {
                    "Required claim absent".into()
                },
                regression_type: if present {
                    None
                } else {
                    Some("required_claim".into())
                },
                baseline_evidence: vec![EvidenceSnippet {
                    label: "Baseline".into(),
                    text: if base_had {
                        text.clone()
                    } else {
                        format!("(claim not observed in baseline) {text}")
                    },
                }],
                candidate_evidence: vec![EvidenceSnippet {
                    label: "Candidate".into(),
                    text: if present {
                        text.clone()
                    } else {
                        "claim absent".into()
                    },
                }],
                comparator: Some("contains_claim".into()),
            }
        }
        ContractExpectation::ForbidsClaim { text } => {
            let present = claim_present(&cand.content, text);
            let outcome = if !present {
                ItemOutcome::Pass
            } else {
                ItemOutcome::Fail
            };
            ContractItemResult {
                item_id: item.id.clone(),
                kind: item.kind.clone(),
                severity: item.severity,
                outcome,
                reason: if present {
                    "Forbidden claim present".into()
                } else {
                    "Forbidden claim absent".into()
                },
                regression_type: Some("forbidden_claim".into()),
                baseline_evidence: vec![],
                candidate_evidence: vec![EvidenceSnippet {
                    label: "Candidate".into(),
                    text: cand.content.chars().take(280).collect(),
                }],
                comparator: Some("forbids_claim".into()),
            }
        }
        ContractExpectation::JsonSchema { schema } => {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(cand.content.trim())
                .or_else(|_| extract_json_object(&cand.content));
            let (outcome, reason) = match parsed {
                Ok(value) => match validate_json_schema(&value, schema) {
                    Ok(()) => (ItemOutcome::Pass, "Structured output matches schema".into()),
                    Err(e) => (ItemOutcome::Fail, e),
                },
                Err(_) => (
                    ItemOutcome::Fail,
                    "Candidate output is not valid JSON".into(),
                ),
            };
            ContractItemResult {
                item_id: item.id.clone(),
                kind: item.kind.clone(),
                severity: item.severity,
                outcome,
                reason,
                regression_type: Some("structured_output".into()),
                baseline_evidence: baseline
                    .map(|b| {
                        vec![EvidenceSnippet {
                            label: "Baseline".into(),
                            text: b.content.chars().take(280).collect(),
                        }]
                    })
                    .unwrap_or_default(),
                candidate_evidence: vec![EvidenceSnippet {
                    label: "Candidate".into(),
                    text: cand.content.chars().take(280).collect(),
                }],
                comparator: Some("json_schema".into()),
            }
        }
        ContractExpectation::ToolRequired { tool_name } => {
            let found = cand.tool_calls.iter().any(|t| t.name == *tool_name);
            ContractItemResult {
                item_id: item.id.clone(),
                kind: item.kind.clone(),
                severity: item.severity,
                outcome: if found {
                    ItemOutcome::Pass
                } else {
                    ItemOutcome::Fail
                },
                reason: if found {
                    format!("Tool `{tool_name}` selected")
                } else {
                    format!("Expected tool `{tool_name}` not selected")
                },
                regression_type: Some("tool_selection".into()),
                baseline_evidence: vec![],
                candidate_evidence: vec![EvidenceSnippet {
                    label: "Tools".into(),
                    text: cand
                        .tool_calls
                        .iter()
                        .map(|t| t.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                }],
                comparator: Some("tool_required".into()),
            }
        }
        ContractExpectation::ToolArgument {
            tool_name,
            argument,
            expected_type,
        } => {
            let call = cand.tool_calls.iter().find(|t| t.name == *tool_name);
            let (outcome, reason, evidence) = match call {
                None => (
                    ItemOutcome::Fail,
                    format!("Tool `{tool_name}` not called"),
                    "missing tool".to_string(),
                ),
                Some(t) => match t.arguments.get(argument) {
                    None => (
                        ItemOutcome::Fail,
                        format!("Argument `{argument}` missing"),
                        "argument absent".into(),
                    ),
                    Some(v) => {
                        let actual = json_type_name(v);
                        if actual == expected_type
                            || (expected_type == "number"
                                && (actual == "integer" || actual == "number"))
                        {
                            (
                                ItemOutcome::Pass,
                                format!("Argument `{argument}` type ok"),
                                format!("{argument}: {actual}"),
                            )
                        } else {
                            (
                                ItemOutcome::Fail,
                                format!(
                                    "Argument `{argument}` type mismatch: expected {expected_type}, got {actual}"
                                ),
                                format!("{argument}: {actual}"),
                            )
                        }
                    }
                },
            };
            ContractItemResult {
                item_id: item.id.clone(),
                kind: item.kind.clone(),
                severity: item.severity,
                outcome,
                reason,
                regression_type: Some("tool_argument".into()),
                baseline_evidence: vec![EvidenceSnippet {
                    label: "Expected".into(),
                    text: format!("{argument}: {expected_type}"),
                }],
                candidate_evidence: vec![EvidenceSnippet {
                    label: "Candidate".into(),
                    text: evidence,
                }],
                comparator: Some("tool_argument".into()),
            }
        }
        ContractExpectation::MustRefuse => {
            let refused = text_looks_like_refusal(&cand.content);
            ContractItemResult {
                item_id: item.id.clone(),
                kind: item.kind.clone(),
                severity: item.severity,
                outcome: if refused {
                    ItemOutcome::Pass
                } else {
                    ItemOutcome::Fail
                },
                reason: if refused {
                    "Candidate refused as required".into()
                } else {
                    "Expected refusal missing".into()
                },
                regression_type: Some("refusal".into()),
                baseline_evidence: vec![],
                candidate_evidence: vec![EvidenceSnippet {
                    label: "Candidate".into(),
                    text: cand.content.chars().take(280).collect(),
                }],
                comparator: Some("must_refuse".into()),
            }
        }
        ContractExpectation::MustAnswer => {
            let refused = text_looks_like_refusal(&cand.content);
            ContractItemResult {
                item_id: item.id.clone(),
                kind: item.kind.clone(),
                severity: item.severity,
                outcome: if !refused {
                    ItemOutcome::Pass
                } else {
                    ItemOutcome::Fail
                },
                reason: if refused {
                    "Unexpected refusal".into()
                } else {
                    "Candidate answered".into()
                },
                regression_type: Some("refusal".into()),
                baseline_evidence: vec![],
                candidate_evidence: vec![EvidenceSnippet {
                    label: "Candidate".into(),
                    text: cand.content.chars().take(280).collect(),
                }],
                comparator: Some("must_answer".into()),
            }
        }
        ContractExpectation::Instruction {
            description,
            check_contains,
        } => {
            let ok = match check_contains {
                Some(s) => cand.content.to_lowercase().contains(&s.to_lowercase()),
                None => !cand.content.trim().is_empty(),
            };
            ContractItemResult {
                item_id: item.id.clone(),
                kind: item.kind.clone(),
                severity: item.severity,
                outcome: if ok {
                    ItemOutcome::Pass
                } else {
                    ItemOutcome::Fail
                },
                reason: if ok {
                    format!("Instruction satisfied: {description}")
                } else {
                    format!("Instruction failed: {description}")
                },
                regression_type: Some("instruction".into()),
                baseline_evidence: vec![],
                candidate_evidence: vec![EvidenceSnippet {
                    label: "Candidate".into(),
                    text: cand.content.chars().take(280).collect(),
                }],
                comparator: Some("instruction".into()),
            }
        }
        ContractExpectation::OutputFormat { description } => {
            // Presentation: pass with warn if baseline exists and length drifts hard.
            let outcome = if let Some(b) = baseline {
                let b_len = b.content.split_whitespace().count().max(1) as f64;
                let c_len = cand.content.split_whitespace().count() as f64;
                let delta = ((c_len - b_len) / b_len).abs();
                if delta > 0.75 {
                    ItemOutcome::Warn
                } else {
                    ItemOutcome::Pass
                }
            } else {
                ItemOutcome::Pass
            };
            ContractItemResult {
                item_id: item.id.clone(),
                kind: item.kind.clone(),
                severity: item.severity,
                outcome,
                reason: description.clone(),
                regression_type: Some("presentation".into()),
                baseline_evidence: baseline
                    .map(|b| {
                        vec![EvidenceSnippet {
                            label: "Baseline".into(),
                            text: b.content.chars().take(200).collect(),
                        }]
                    })
                    .unwrap_or_default(),
                candidate_evidence: vec![EvidenceSnippet {
                    label: "Candidate".into(),
                    text: cand.content.chars().take(200).collect(),
                }],
                comparator: Some("output_format".into()),
            }
        }
        ContractExpectation::SemanticSimilarity { min_cosine } => {
            // Deterministic hash-embedding proxy via shared token overlap.
            let score = token_overlap(base_text, &cand.content);
            let ok = score >= *min_cosine;
            ContractItemResult {
                item_id: item.id.clone(),
                kind: item.kind.clone(),
                severity: item.severity,
                outcome: if ok {
                    ItemOutcome::Pass
                } else {
                    ItemOutcome::Fail
                },
                reason: format!("Semantic overlap {score:.3} (min {min_cosine})"),
                regression_type: Some("semantic".into()),
                baseline_evidence: vec![],
                candidate_evidence: vec![],
                comparator: Some("semantic_overlap".into()),
            }
        }
        ContractExpectation::LatencyBudget { max_ms } => {
            let ok = cand.latency_ms <= *max_ms;
            ContractItemResult {
                item_id: item.id.clone(),
                kind: item.kind.clone(),
                severity: item.severity,
                outcome: if ok {
                    ItemOutcome::Pass
                } else {
                    ItemOutcome::Warn
                },
                reason: format!("Latency {}ms (budget {max_ms}ms)", cand.latency_ms),
                regression_type: Some("latency".into()),
                baseline_evidence: vec![],
                candidate_evidence: vec![],
                comparator: Some("latency".into()),
            }
        }
        ContractExpectation::CostBudget {
            max_relative_delta_pct,
        } => {
            let outcome = match (baseline.and_then(|b| b.cost_usd), cand.cost_usd) {
                (Some(b), Some(c)) if b > 0.0 => {
                    let delta = (c - b) / b * 100.0;
                    if delta <= *max_relative_delta_pct {
                        ItemOutcome::Pass
                    } else {
                        ItemOutcome::Warn
                    }
                }
                _ => ItemOutcome::Pass, // not comparable
            };
            ContractItemResult {
                item_id: item.id.clone(),
                kind: item.kind.clone(),
                severity: item.severity,
                outcome,
                reason: "Cost budget check".into(),
                regression_type: Some("cost".into()),
                baseline_evidence: vec![],
                candidate_evidence: vec![],
                comparator: Some("cost".into()),
            }
        }
    }
}

fn text_looks_like_refusal(content: &str) -> bool {
    // Build a minimal ModelResponse for the existing detector.
    let resp = crate::types::ModelResponse {
        probe_id: Uuid::nil(),
        model_label: "candidate".into(),
        model_id: "candidate".into(),
        content: content.to_string(),
        token_count: content.split_whitespace().count(),
        latency_ms: 0,
        finish_reason: crate::types::FinishReason::Stop,
        timestamp: chrono::Utc::now(),
        raw: serde_json::Value::Null,
    };
    RefusalDetector::is_refusal(&resp)
}

fn extract_json_object(s: &str) -> Result<serde_json::Value, serde_json::Error> {
    if let Some(start) = s.find('{') {
        if let Some(end) = s.rfind('}') {
            return serde_json::from_str(&s[start..=end]);
        }
    }
    serde_json::from_str(s)
}

/// Validate JSON against a schema using `jsonschema`. Unsupported constructs are reported.
fn validate_json_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
) -> Result<(), String> {
    // Detect obvious unsupported keywords we do not resolve across remote $ref.
    if schema_has_remote_ref(schema) {
        return Err(
            "schema contains unsupported remote $ref; validation not pretended as pass".into(),
        );
    }
    let compiled = jsonschema::options()
        .with_draft(jsonschema::Draft::Draft7)
        .build(schema)
        .map_err(|e| format!("unsupported or invalid JSON Schema: {e}"))?;
    match compiled.validate(value) {
        Ok(()) => Ok(()),
        Err(err) => Err(format!("schema validation failed: {err}")),
    }
}

fn schema_has_remote_ref(schema: &serde_json::Value) -> bool {
    match schema {
        serde_json::Value::Object(map) => {
            if let Some(serde_json::Value::String(r)) = map.get("$ref") {
                if r.starts_with("http://") || r.starts_with("https://") {
                    return true;
                }
            }
            map.values().any(schema_has_remote_ref)
        }
        serde_json::Value::Array(arr) => arr.iter().any(schema_has_remote_ref),
        _ => false,
    }
}

/// Shallow required-properties check retained for diagnostics (jsonschema is primary).
#[allow(dead_code)]
fn validate_schema_shallow(
    value: &serde_json::Value,
    schema: &serde_json::Value,
) -> Result<(), String> {
    if let Some(req) = schema.get("required").and_then(|r| r.as_array()) {
        let obj = value
            .as_object()
            .ok_or_else(|| "expected object".to_string())?;
        for key in req {
            let k = key.as_str().unwrap_or("");
            if !obj.contains_key(k) {
                return Err(format!("missing required property `{k}`"));
            }
        }
    }
    Ok(())
}

fn token_overlap(a: &str, b: &str) -> f64 {
    let tok = |s: &str| {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() > 2)
            .map(|t| t.to_string())
            .collect::<std::collections::BTreeSet<_>>()
    };
    let ta = tok(a);
    let tb = tok(b);
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let inter = ta.intersection(&tb).count() as f64;
    let union = ta.union(&tb).count() as f64;
    inter / union
}

/// Run full qualification for one candidate (offline-capable given captures).
pub fn qualify_candidate(
    contract: &ApplicationContract,
    baseline: &BaselineSnapshot,
    candidate_model: &str,
    candidate: &BTreeMap<String, CapturedBehaviour>,
    qualification_id: &str,
) -> QualificationResult {
    let mut item_results = Vec::new();
    for item in &contract.items {
        let b = baseline.behaviours.get(&item.prompt);
        let c = candidate.get(&item.prompt);
        item_results.push(evaluate_item(item, b, c));
    }
    let has_execution_errors = item_results
        .iter()
        .any(|r| r.regression_type.as_deref() == Some("execution_error"))
        || candidate.values().any(|c| c.execution_error.is_some());

    // Behavioural decision ignores execution-error items.
    let behavioural: Vec<_> = item_results
        .iter()
        .filter(|r| r.regression_type.as_deref() != Some("execution_error"))
        .cloned()
        .collect();
    let thresholds = baseline.qualification_thresholds.clone();
    let decision = if has_execution_errors && behavioural.is_empty() {
        // Infrastructure-only failure must not look like PASS.
        QualificationDecision::Review
    } else {
        decide_overall(&behavioural, &thresholds, false)
    };
    let cost_latency = compute_cost_latency_delta(baseline, candidate);
    QualificationResult {
        schema_version: QUALIFICATION_SCHEMA_VERSION,
        id: qualification_id.to_string(),
        application_id: contract.application_id.clone(),
        contract_version: contract.version,
        contract_hash: contract.content_hash(),
        baseline_id: baseline.id.clone(),
        baseline_hash: baseline.content_hash(),
        candidate_model: candidate_model.to_string(),
        decision,
        original_decision: Some(decision),
        created_at: chrono::Utc::now().to_rfc3339(),
        item_results,
        repair_attempts: vec![],
        validated_patch: None,
        cost_latency,
        fingerprint_delta_note: None,
        stale: false,
        metadata: BTreeMap::new(),
        captured_behaviours: candidate.clone(),
        has_execution_errors,
        input_fingerprint: None,
    }
}

fn compute_cost_latency_delta(
    baseline: &BaselineSnapshot,
    candidate: &BTreeMap<String, CapturedBehaviour>,
) -> CostLatencyDelta {
    let mut base_lat = 0u64;
    let mut cand_lat = 0u64;
    let mut n = 0u64;
    let mut base_cost = 0.0;
    let mut cand_cost = 0.0;
    let mut cost_n = 0u64;
    for (k, b) in &baseline.behaviours {
        if let Some(c) = candidate.get(k) {
            base_lat += b.latency_ms;
            cand_lat += c.latency_ms;
            n += 1;
            if let (Some(bc), Some(cc)) = (b.cost_usd, c.cost_usd) {
                base_cost += bc;
                cand_cost += cc;
                cost_n += 1;
            }
        }
    }
    let latency_delta_pct = if n > 0 && base_lat > 0 {
        Some((cand_lat as f64 - base_lat as f64) / base_lat as f64 * 100.0)
    } else {
        None
    };
    let cost_delta_pct = if cost_n > 0 && base_cost > 0.0 {
        Some((cand_cost - base_cost) / base_cost * 100.0)
    } else {
        None
    };
    CostLatencyDelta {
        cost_delta_pct,
        latency_delta_pct,
        estimate: true,
        note: Some("Estimates only when comparable measurements exist".into()),
    }
}

/// Attempt deterministic repair for failed block/review claim/instruction items.
pub fn attempt_validated_repair(
    contract: &ApplicationContract,
    result: &mut QualificationResult,
    baseline: &BaselineSnapshot,
    candidate_responses_after_mutation: &dyn Fn(&str, &str) -> Option<CapturedBehaviour>,
) {
    let failed: Vec<_> = result
        .item_results
        .iter()
        .filter(|r| matches!(r.outcome, ItemOutcome::Fail))
        .filter(|r| {
            matches!(
                r.severity,
                ContractSeverity::Block | ContractSeverity::Review
            )
        })
        .cloned()
        .collect();

    let mut any_patch = false;
    for fail in failed {
        let Some(item) = contract.item(&fail.item_id) else {
            continue;
        };
        let original = item.user_prompt.clone().unwrap_or_default();
        if original.is_empty() {
            continue;
        }
        // Build strategies from required claim text when applicable.
        let mut strategies = Vec::new();
        if let ContractExpectation::ContainsClaim { text } = &item.expectation {
            strategies.push(MutationStrategy::AddClaimInstruction {
                required_values: vec![text.clone()],
            });
            strategies.push(MutationStrategy::ReinforceInstruction {
                instruction_text: format!("You must include: {text}"),
            });
        } else if let ContractExpectation::Instruction {
            description,
            check_contains,
        } = &item.expectation
        {
            strategies.push(MutationStrategy::ReinforceInstruction {
                instruction_text: description.clone(),
            });
            if let Some(c) = check_contains {
                strategies.push(MutationStrategy::AddClaimInstruction {
                    required_values: vec![c.clone()],
                });
            }
        } else {
            continue;
        }

        let probe = Probe {
            id: Uuid::nil(),
            name: item.prompt.clone(),
            category: ProbeCategory::Instruction,
            prompt: original.clone(),
            system_prompt: item.system_prompt.clone(),
            known_answer: None,
            expected_schema: None,
            instructions: vec![],
            tags: vec![],
            source: ProbeSource::UserDefined,
            expected_verbosity: None,
            expected_tone: None,
            refusal_expectation: None,
            mutation_hint: None,
            custom_assertions: vec![],
            format_sensitive: false,
            structure_sensitive: false,
            claim_anchor_policy: Default::default(),
            presentation_drift: Default::default(),
            latency_slo_ms: None,
            tools: None,
        };

        // Try strategies one-by-one then combined (deterministic order).
        let mut attempt_idx = 0u32;
        let mut success_patch = None;
        for i in 0..strategies.len() {
            attempt_idx += 1;
            let slice = &strategies[..=i];
            let mutated = apply_mutations(&probe, slice);
            let capture = candidate_responses_after_mutation(&item.prompt, &mutated);
            let outcome = match &capture {
                Some(c) => {
                    let r = evaluate_item(item, baseline.behaviours.get(&item.prompt), Some(c));
                    r.outcome
                }
                None => ItemOutcome::Fail,
            };
            result.repair_attempts.push(RepairAttempt {
                attempt_index: attempt_idx,
                strategies: slice.iter().map(|s| format!("{s:?}")).collect(),
                mutated_prompt: mutated.clone(),
                outcome,
                reason: format!("mutation {attempt_idx}"),
            });
            if matches!(outcome, ItemOutcome::Pass) {
                success_patch = Some(ValidatedPatch {
                    qualification_id: result.id.clone(),
                    model: result.candidate_model.clone(),
                    contract_hash: result.contract_hash.clone(),
                    strategies: slice.iter().map(|s| format!("{s:?}")).collect(),
                    prompt_diff: format!("--- original\n+++ patched\n-{original}\n+{mutated}"),
                    original_prompt: original.clone(),
                    patched_prompt: mutated,
                    revalidated: true,
                });
                // Update item result to pass-with-patch semantics.
                if let Some(ir) = result
                    .item_results
                    .iter_mut()
                    .find(|r| r.item_id == fail.item_id)
                {
                    ir.outcome = ItemOutcome::Pass;
                    ir.reason = format!("{} (after validated repair)", ir.reason);
                }
                break;
            }
        }
        if let Some(p) = success_patch {
            result.validated_patch = Some(p);
            any_patch = true;
        }
    }
    result.decision = decide_overall(
        &result.item_results,
        &baseline.qualification_thresholds,
        any_patch,
    );
}

/// Compatibility percentage is a summary only — evidence remains authoritative.
pub fn compatibility_summary_pct(result: &QualificationResult) -> f64 {
    let n = result.item_results.len();
    if n == 0 {
        return 100.0;
    }
    let pass = result
        .item_results
        .iter()
        .filter(|r| matches!(r.outcome, ItemOutcome::Pass | ItemOutcome::Warn))
        .count();
    (pass as f64) * 100.0 / n as f64
}

/// Impact grouping for migration review.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImpactReport {
    pub candidate_model: String,
    pub blockers: Vec<String>,
    pub review: Vec<String>,
    pub presentation_drift: Vec<String>,
    pub unaffected: Vec<String>,
    pub decision: QualificationDecision,
}

pub fn build_impact(result: &QualificationResult) -> ImpactReport {
    let mut report = ImpactReport {
        candidate_model: result.candidate_model.clone(),
        decision: result.decision,
        ..Default::default()
    };
    for r in &result.item_results {
        let label = format!("{} ({})", r.item_id, r.reason);
        match (r.outcome, r.severity, r.regression_type.as_deref()) {
            (ItemOutcome::Fail, ContractSeverity::Block, _) => report.blockers.push(label),
            (ItemOutcome::Fail, ContractSeverity::Review, _) | (ItemOutcome::Review, _, _) => {
                report.review.push(label)
            }
            (ItemOutcome::Warn, _, Some("presentation"))
            | (ItemOutcome::Fail, _, Some("presentation")) => report.presentation_drift.push(label),
            (ItemOutcome::Warn, _, _) => report.presentation_drift.push(label),
            (ItemOutcome::Pass, _, _) => report.unaffected.push(r.item_id.clone()),
            _ => report.review.push(label),
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{ContractProvenance, ContractProvenanceSource};

    fn item_claim(id: &str, severity: ContractSeverity) -> ContractItem {
        ContractItem {
            id: id.into(),
            kind: ContractItemKind::RequiredClaim,
            severity,
            prompt: "refund_decision".into(),
            system_prompt: None,
            user_prompt: Some("Should we refund £600?".into()),
            expectation: ContractExpectation::ContainsClaim {
                text: "Refunds over £500 require manager approval".into(),
            },
            provenance: ContractProvenance {
                source: ContractProvenanceSource::Manual,
                ..Default::default()
            },
            tags: vec![],
        }
    }

    fn behaviour(content: &str) -> CapturedBehaviour {
        CapturedBehaviour {
            prompt_id: "refund_decision".into(),
            content: content.into(),
            latency_ms: 100,
            cost_usd: Some(0.01),
            tool_calls: vec![],
            finish_reason: None,
            fingerprint: None,
            token_count: None,
            raw: None,
            execution_error: None,
            execution_error_message: None,
            execution_meta: Default::default(),
        }
    }

    #[test]
    fn overall_block_from_block_failure() {
        let results = vec![ContractItemResult {
            item_id: "a".into(),
            kind: ContractItemKind::RequiredClaim,
            severity: ContractSeverity::Block,
            outcome: ItemOutcome::Fail,
            reason: "absent".into(),
            regression_type: None,
            baseline_evidence: vec![],
            candidate_evidence: vec![],
            comparator: None,
        }];
        assert_eq!(
            decide_overall(&results, &QualificationThresholds::default(), false),
            QualificationDecision::Block
        );
    }

    #[test]
    fn required_claim_regression() {
        let item = item_claim(
            "refund_decision.required_claim.001",
            ContractSeverity::Block,
        );
        let base = behaviour("Refunds over £500 require manager approval.");
        let cand = behaviour("Sure, we can refund that.");
        let r = evaluate_item(&item, Some(&base), Some(&cand));
        assert_eq!(r.outcome, ItemOutcome::Fail);
        assert_eq!(r.severity, ContractSeverity::Block);
    }

    #[test]
    fn tool_argument_type_mismatch() {
        let item = ContractItem {
            id: "order_lookup.tool_argument.002".into(),
            kind: ContractItemKind::ToolArgument,
            severity: ContractSeverity::Block,
            prompt: "order_lookup".into(),
            system_prompt: None,
            user_prompt: None,
            expectation: ContractExpectation::ToolArgument {
                tool_name: "lookup_order".into(),
                argument: "order_id".into(),
                expected_type: "string".into(),
            },
            provenance: Default::default(),
            tags: vec![],
        };
        let cand = CapturedBehaviour {
            prompt_id: "order_lookup".into(),
            content: "".into(),
            latency_ms: 10,
            cost_usd: None,
            tool_calls: vec![CapturedToolCall {
                name: "lookup_order".into(),
                arguments: BTreeMap::from([("order_id".into(), serde_json::json!(12345))]),
            }],
            finish_reason: None,
            fingerprint: None,
            token_count: None,
            raw: None,
            execution_error: None,
            execution_error_message: None,
            execution_meta: Default::default(),
        };
        let r = evaluate_item(&item, None, Some(&cand));
        assert_eq!(r.outcome, ItemOutcome::Fail);
    }

    #[test]
    fn replay_is_deterministic() {
        let mut contract = ApplicationContract::new("app", "App");
        contract.created_at = "t".into();
        contract.items.push(item_claim(
            "refund_decision.required_claim.001",
            ContractSeverity::Block,
        ));
        let mut behaviours = BTreeMap::new();
        behaviours.insert(
            "refund_decision".into(),
            behaviour("Refunds over £500 require manager approval."),
        );
        let baseline = BaselineSnapshot {
            schema_version: 1,
            id: "baseline-0001".into(),
            version: 1,
            application_id: "app".into(),
            contract_version: 1,
            contract_hash: contract.content_hash(),
            model: "openai:gpt-current".into(),
            provider_model_id: "gpt-current".into(),
            created_at: "t".into(),
            behaviours: behaviours.clone(),
            prompt_hashes: BTreeMap::new(),
            qualification_thresholds: Default::default(),
            runtime: BTreeMap::new(),
        };
        let mut cand = BTreeMap::new();
        cand.insert(
            "refund_decision".into(),
            behaviour("Refunds over £500 require manager approval. Also please wait."),
        );
        let a = qualify_candidate(&contract, &baseline, "openai:gpt-x", &cand, "qual-1");
        let b = qualify_candidate(&contract, &baseline, "openai:gpt-x", &cand, "qual-1");
        assert_eq!(a.decision, b.decision);
        assert_eq!(a.item_results, b.item_results);
        assert_eq!(compatibility_summary_pct(&a), compatibility_summary_pct(&b));
    }
}
