//! Authoritative current-state assessment for Application Contract qualifications.
//!
//! Historical [`QualificationResult`] records remain immutable. Every current-state
//! decision (SAFE TO MIGRATE, CI exit, report buckets, lock confidence) must go
//! through [`assess_qualification`] / [`aggregate_migration`].
//!
//! Invariant: Arsenic must never communicate SAFE TO MIGRATE when evidence is
//! stale, incomplete, failed, unsupported, or otherwise invalid.

use serde::{Deserialize, Serialize};

use super::{QualificationDecision, QualificationResult, ValidatedPatch};

/// Whether captured evidence is trustworthy for a *current* migration decision.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvidenceValidity {
    /// Evidence matches current contract+baseline and executions completed.
    Valid,
    /// Contract and/or baseline hash no longer match; historical only.
    Stale,
    /// Provider/runtime failure, missing required captures, or corrupt context.
    Incomplete,
}

impl EvidenceValidity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Valid => "VALID",
            Self::Stale => "STALE",
            Self::Incomplete => "INCOMPLETE",
        }
    }
}

/// Fail-closed migration recommendation for humans and automation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MigrationRecommendation {
    SafeToMigrate,
    MigrationBlocked,
    ReviewRequired,
    Incomplete,
    Stale,
    NoEvidence,
}

impl MigrationRecommendation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SafeToMigrate => "SAFE TO MIGRATE",
            Self::MigrationBlocked => "MIGRATION BLOCKED",
            Self::ReviewRequired => "REVIEW REQUIRED",
            Self::Incomplete => "QUALIFICATION INCOMPLETE",
            Self::Stale => "STALE — REQUALIFY REQUIRED",
            Self::NoEvidence => "NO EVIDENCE",
        }
    }

    pub fn allows_safe_migrate(self) -> bool {
        matches!(self, Self::SafeToMigrate)
    }

    /// CI exit: 0 safe, 1 block, 2 review/stale, 3 incomplete/no evidence/error.
    pub fn ci_exit_code(self) -> i32 {
        match self {
            Self::SafeToMigrate => 0,
            Self::MigrationBlocked => 1,
            Self::ReviewRequired | Self::Stale => 2,
            Self::Incomplete | Self::NoEvidence => 3,
        }
    }
}

/// Fully assessed current view of one historical qualification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EffectiveQualification {
    pub qualification_id: String,
    pub candidate_model: String,
    /// Immutable recorded behavioural decision from the run.
    pub recorded_decision: QualificationDecision,
    /// Decision after applying current-context validity (Stale overrides).
    pub effective_decision: QualificationDecision,
    pub evidence_validity: EvidenceValidity,
    pub is_stale: bool,
    pub stale_reasons: Vec<String>,
    pub incomplete_reasons: Vec<String>,
    pub migration_recommendation: MigrationRecommendation,
    pub blockers: usize,
    pub review_items: usize,
    pub has_validated_patch: bool,
    pub patch_applies_to_current: bool,
    pub contract_hash: String,
    pub baseline_hash: String,
    pub current_contract_hash: String,
    pub current_baseline_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_fingerprint: Option<String>,
}

impl EffectiveQualification {
    pub fn is_currently_passing(&self) -> bool {
        self.evidence_validity == EvidenceValidity::Valid
            && matches!(
                self.recorded_decision,
                QualificationDecision::Pass
                    | QualificationDecision::PassWithWarnings
                    | QualificationDecision::PassWithPatch
            )
            && self.migration_recommendation.allows_safe_migrate()
    }
}

fn patch_binds_to_current(
    patch: &ValidatedPatch,
    q: &QualificationResult,
    current_contract_hash: &str,
    current_baseline_hash: &str,
) -> bool {
    patch.contract_hash == current_contract_hash
        && q.contract_hash == current_contract_hash
        && q.baseline_hash == current_baseline_hash
        && patch.model == q.candidate_model
}

fn incomplete_reasons(q: &QualificationResult) -> Vec<String> {
    let mut reasons = Vec::new();
    if q.has_execution_errors {
        reasons.push("provider/runtime execution errors present".into());
    }
    if q.item_results.is_empty() && q.captured_behaviours.is_empty() {
        reasons.push("zero captured behaviours and no item results".into());
    }
    if q.item_results.is_empty() && !q.captured_behaviours.is_empty() {
        reasons.push("no contract item results recorded".into());
    }
    for r in &q.item_results {
        if r.regression_type.as_deref() == Some("execution_error") {
            reasons.push(format!("{}: {}", r.item_id, r.reason));
        }
    }
    // missing_capture is a behavioural Fail (Block/Review), not incomplete evidence.
    // Capability gaps are behavioural Fail — handled as Block/Review.
    reasons.sort();
    reasons.dedup();
    reasons
}

/// Assess one qualification against the *current* contract and baseline hashes.
pub fn assess_qualification(
    q: &QualificationResult,
    current_contract_hash: &str,
    current_baseline_hash: &str,
) -> EffectiveQualification {
    let mut stale_reasons = Vec::new();
    if q.contract_hash != current_contract_hash {
        stale_reasons.push("contract hash differs from current application contract".into());
    }
    if q.baseline_hash != current_baseline_hash {
        stale_reasons.push("baseline hash differs from current production baseline".into());
    }
    let is_stale = !stale_reasons.is_empty();

    let incomplete = incomplete_reasons(q);
    let has_incomplete = !incomplete.is_empty();

    let patch_ok = q
        .validated_patch
        .as_ref()
        .map(|p| patch_binds_to_current(p, q, current_contract_hash, current_baseline_hash))
        .unwrap_or(false);

    let (evidence_validity, effective_decision, migration) = if is_stale {
        (
            EvidenceValidity::Stale,
            QualificationDecision::Stale,
            MigrationRecommendation::Stale,
        )
    } else if has_incomplete {
        // Incomplete evidence: never SAFE. Preserve recorded behavioural decision
        // for inspection, but surface Incomplete as the migration outcome.
        (
            EvidenceValidity::Incomplete,
            q.decision,
            MigrationRecommendation::Incomplete,
        )
    } else {
        let recorded = q.decision;
        // PassWithPatch only certifies current inputs when patch binds.
        let migration = match recorded {
            QualificationDecision::Pass | QualificationDecision::PassWithWarnings => {
                MigrationRecommendation::SafeToMigrate
            }
            QualificationDecision::PassWithPatch if patch_ok || q.validated_patch.is_none() => {
                MigrationRecommendation::SafeToMigrate
            }
            QualificationDecision::PassWithPatch => MigrationRecommendation::Stale,
            QualificationDecision::Block => MigrationRecommendation::MigrationBlocked,
            QualificationDecision::Review => MigrationRecommendation::ReviewRequired,
            QualificationDecision::Stale => MigrationRecommendation::Stale,
        };
        (EvidenceValidity::Valid, recorded, migration)
    };

    EffectiveQualification {
        qualification_id: q.id.clone(),
        candidate_model: q.candidate_model.clone(),
        recorded_decision: q.decision,
        effective_decision,
        evidence_validity,
        is_stale,
        stale_reasons,
        incomplete_reasons: incomplete,
        migration_recommendation: migration,
        blockers: q.blockers().count(),
        review_items: q
            .item_results
            .iter()
            .filter(|r| {
                (matches!(r.outcome, super::ItemOutcome::Fail)
                    && matches!(r.severity, crate::contract::ContractSeverity::Review))
                    || matches!(r.outcome, super::ItemOutcome::Review)
            })
            .count(),
        has_validated_patch: q.validated_patch.is_some(),
        patch_applies_to_current: patch_ok,
        contract_hash: q.contract_hash.clone(),
        baseline_hash: q.baseline_hash.clone(),
        current_contract_hash: current_contract_hash.to_string(),
        current_baseline_hash: current_baseline_hash.to_string(),
        input_fingerprint: q.input_fingerprint.clone(),
    }
}

/// Deterministically pick the latest qualification per candidate (by `created_at`, then id).
pub fn latest_per_candidate(quals: &[QualificationResult]) -> Vec<&QualificationResult> {
    let mut by_model: std::collections::BTreeMap<String, &QualificationResult> =
        std::collections::BTreeMap::new();
    let mut ordered: Vec<&QualificationResult> = quals.iter().collect();
    ordered.sort_by(|a, b| {
        a.created_at
            .cmp(&b.created_at)
            .then_with(|| a.id.cmp(&b.id))
    });
    for q in ordered {
        by_model.insert(q.candidate_model.clone(), q);
    }
    by_model.into_values().collect()
}

/// Assess latest-per-candidate qualifications against current hashes.
pub fn assess_all(
    quals: &[QualificationResult],
    current_contract_hash: &str,
    current_baseline_hash: &str,
) -> Vec<EffectiveQualification> {
    let latest = latest_per_candidate(quals);
    let mut out: Vec<_> = latest
        .into_iter()
        .map(|q| assess_qualification(q, current_contract_hash, current_baseline_hash))
        .collect();
    out.sort_by(|a, b| a.candidate_model.cmp(&b.candidate_model));
    out
}

/// Aggregate migration recommendation across candidates (fail closed).
pub fn aggregate_migration(effectives: &[EffectiveQualification]) -> MigrationRecommendation {
    if effectives.is_empty() {
        return MigrationRecommendation::NoEvidence;
    }
    let current: Vec<_> = effectives.iter().filter(|e| !e.is_stale).collect();
    if current.is_empty() {
        return MigrationRecommendation::Stale;
    }
    if current
        .iter()
        .any(|e| e.evidence_validity == EvidenceValidity::Incomplete)
    {
        return MigrationRecommendation::Incomplete;
    }
    if current
        .iter()
        .any(|e| e.migration_recommendation == MigrationRecommendation::MigrationBlocked)
    {
        return MigrationRecommendation::MigrationBlocked;
    }
    if current
        .iter()
        .any(|e| e.migration_recommendation == MigrationRecommendation::ReviewRequired)
    {
        return MigrationRecommendation::ReviewRequired;
    }
    if current
        .iter()
        .any(|e| e.migration_recommendation.allows_safe_migrate())
    {
        return MigrationRecommendation::SafeToMigrate;
    }
    // Fail closed: unknown combination is not safe.
    MigrationRecommendation::ReviewRequired
}

/// Machine-readable envelope for one qualification assessment.
pub fn effective_to_json(e: &EffectiveQualification) -> serde_json::Value {
    serde_json::json!({
        "qualification_id": e.qualification_id,
        "candidate_model": e.candidate_model,
        "recorded_decision": e.recorded_decision,
        "effective_decision": e.effective_decision,
        "evidence_validity": e.evidence_validity,
        "is_stale": e.is_stale,
        "stale_reasons": e.stale_reasons,
        "incomplete_reasons": e.incomplete_reasons,
        "migration_recommendation": e.migration_recommendation,
        "migration_recommendation_text": e.migration_recommendation.as_str(),
        "blockers": e.blockers,
        "review_items": e.review_items,
        "has_validated_patch": e.has_validated_patch,
        "patch_applies_to_current": e.patch_applies_to_current,
        "contract_hash": e.contract_hash,
        "baseline_hash": e.baseline_hash,
        "current_contract_hash": e.current_contract_hash,
        "current_baseline_hash": e.current_baseline_hash,
        "input_fingerprint": e.input_fingerprint,
        "ci_exit_code": e.migration_recommendation.ci_exit_code(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{
        ApplicationContract, ContractExpectation, ContractItem, ContractItemKind, ContractSeverity,
    };
    use crate::qualify::{
        qualify_candidate, BaselineSnapshot, CapturedBehaviour, QualificationThresholds,
    };
    use std::collections::BTreeMap;

    fn claim_contract(version: u32) -> ApplicationContract {
        let mut c = ApplicationContract::new("app", "App");
        c.version = version;
        c.created_at = "t".into();
        c.items.push(ContractItem {
            id: "refund_decision.required_claim.001".into(),
            kind: ContractItemKind::RequiredClaim,
            severity: ContractSeverity::Block,
            prompt: "refund_decision".into(),
            system_prompt: None,
            user_prompt: Some("refund?".into()),
            expectation: ContractExpectation::ContainsClaim {
                text: "Refunds over £500 require manager approval".into(),
            },
            provenance: Default::default(),
            tags: vec![],
        });
        c
    }

    fn beh(ok: bool) -> CapturedBehaviour {
        CapturedBehaviour {
            prompt_id: "refund_decision".into(),
            content: if ok {
                "Refunds over £500 require manager approval".into()
            } else {
                "approve".into()
            },
            latency_ms: 1,
            cost_usd: None,
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

    fn baseline_for(contract: &ApplicationContract) -> BaselineSnapshot {
        let mut behaviours = BTreeMap::new();
        behaviours.insert("refund_decision".into(), beh(true));
        BaselineSnapshot {
            schema_version: 1,
            id: "baseline-0001".into(),
            version: 1,
            application_id: "app".into(),
            contract_version: contract.version,
            contract_hash: contract.content_hash(),
            model: "openai:prod".into(),
            provider_model_id: "prod".into(),
            created_at: "t".into(),
            behaviours,
            prompt_hashes: BTreeMap::new(),
            qualification_thresholds: QualificationThresholds::default(),
            runtime: BTreeMap::new(),
        }
    }

    #[test]
    fn stale_pass_never_safe() {
        let c1 = claim_contract(1);
        let baseline = baseline_for(&c1);
        let mut cand = BTreeMap::new();
        cand.insert("refund_decision".into(), beh(true));
        let q = qualify_candidate(&c1, &baseline, "openai:gpt-x", &cand, "qual-1");
        let c2 = claim_contract(2);
        let eff = assess_qualification(&q, &c2.content_hash(), &baseline.content_hash());
        assert!(eff.is_stale);
        assert_eq!(eff.evidence_validity, EvidenceValidity::Stale);
        assert_eq!(eff.migration_recommendation, MigrationRecommendation::Stale);
        assert!(!eff.migration_recommendation.allows_safe_migrate());
        assert_eq!(eff.recorded_decision, QualificationDecision::Pass);
    }

    #[test]
    fn execution_errors_never_safe() {
        let c1 = claim_contract(1);
        let baseline = baseline_for(&c1);
        let mut cand = BTreeMap::new();
        let mut b = beh(true);
        b.execution_error = Some(super::super::ExecutionErrorKind::Timeout);
        b.execution_error_message = Some("timeout".into());
        cand.insert("refund_decision".into(), b);
        let q = qualify_candidate(&c1, &baseline, "openai:gpt-x", &cand, "qual-1");
        assert!(q.has_execution_errors);
        let eff = assess_qualification(&q, &c1.content_hash(), &baseline.content_hash());
        assert_eq!(eff.evidence_validity, EvidenceValidity::Incomplete);
        assert_eq!(
            eff.migration_recommendation,
            MigrationRecommendation::Incomplete
        );
        assert!(!eff.is_currently_passing());
    }

    #[test]
    fn aggregate_omitting_stale_cannot_invent_safe_from_stale_only() {
        let c1 = claim_contract(1);
        let baseline = baseline_for(&c1);
        let mut cand = BTreeMap::new();
        cand.insert("refund_decision".into(), beh(true));
        let q = qualify_candidate(&c1, &baseline, "openai:gpt-x", &cand, "qual-1");
        let c2 = claim_contract(2);
        let eff = assess_all(&[q], &c2.content_hash(), &baseline.content_hash());
        assert_eq!(aggregate_migration(&eff), MigrationRecommendation::Stale);
    }

    #[test]
    fn current_block_dominates_current_pass() {
        let c1 = claim_contract(1);
        let baseline = baseline_for(&c1);
        let mut pass_c = BTreeMap::new();
        pass_c.insert("refund_decision".into(), beh(true));
        let mut block_c = BTreeMap::new();
        block_c.insert("refund_decision".into(), beh(false));
        let q_pass = qualify_candidate(&c1, &baseline, "openai:good", &pass_c, "qual-1");
        let q_block = qualify_candidate(&c1, &baseline, "openai:bad", &block_c, "qual-2");
        let eff = assess_all(
            &[q_pass, q_block],
            &c1.content_hash(),
            &baseline.content_hash(),
        );
        assert_eq!(
            aggregate_migration(&eff),
            MigrationRecommendation::MigrationBlocked
        );
    }

    #[test]
    fn review_escalates_false_cannot_neutralise_execution_errors() {
        let c1 = claim_contract(1);
        let mut baseline = baseline_for(&c1);
        baseline.qualification_thresholds.review_escalates = false;
        let mut cand = BTreeMap::new();
        let mut b = beh(true);
        b.execution_error = Some(super::super::ExecutionErrorKind::AuthError);
        cand.insert("refund_decision".into(), b);
        let q = qualify_candidate(&c1, &baseline, "openai:gpt-x", &cand, "qual-1");
        let eff = assess_qualification(&q, &c1.content_hash(), &baseline.content_hash());
        assert_eq!(
            eff.migration_recommendation,
            MigrationRecommendation::Incomplete
        );
    }

    #[test]
    fn latest_per_candidate_prefers_newest() {
        let c1 = claim_contract(1);
        let baseline = baseline_for(&c1);
        let mut cand = BTreeMap::new();
        cand.insert("refund_decision".into(), beh(true));
        let mut older = qualify_candidate(&c1, &baseline, "openai:gpt-x", &cand, "qual-1");
        older.created_at = "2020-01-01T00:00:00Z".into();
        let mut newer = qualify_candidate(&c1, &baseline, "openai:gpt-x", &cand, "qual-2");
        newer.created_at = "2026-01-01T00:00:00Z".into();
        newer.decision = QualificationDecision::Block;
        let quals = [older, newer];
        let latest = latest_per_candidate(&quals);
        assert_eq!(latest.len(), 1);
        assert_eq!(latest[0].id, "qual-2");
    }
}
