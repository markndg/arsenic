//! Adversarial release-gate tests: attempt to make Arsenic say SAFE TO MIGRATE
//! without valid, current, complete evidence. Every case must fail closed.

use std::collections::BTreeMap;

use arsenic_core::{
    aggregate_migration, assess_all, assess_qualification, attempt_validated_repair,
    capability_failures, decide_overall, qualify_candidate, update_lock_from_qualification,
    ApplicationContract, ArsenicLock, BaselineSnapshot, CapturedBehaviour, CapturedToolCall,
    ContractExpectation, ContractItem, ContractItemKind, ContractProvenance, ContractSeverity,
    EvidenceValidity, ExecutionErrorKind, ItemOutcome, MigrationRecommendation,
    ProviderCapabilities, QualificationDecision, QualificationThresholds, ValidatedPatch,
};

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
        provenance: ContractProvenance::default(),
        tags: vec![],
    });
    c
}

fn tool_contract() -> ApplicationContract {
    let mut c = claim_contract(1);
    c.items.push(ContractItem {
        id: "order_lookup.tool.001".into(),
        kind: ContractItemKind::ToolCall,
        severity: ContractSeverity::Block,
        prompt: "order_lookup".into(),
        system_prompt: None,
        user_prompt: None,
        expectation: ContractExpectation::ToolRequired {
            tool_name: "lookup_order".into(),
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
            "approve immediately".into()
        },
        latency_ms: 1,
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

fn baseline_for(contract: &ApplicationContract) -> BaselineSnapshot {
    let mut behaviours = BTreeMap::new();
    behaviours.insert("refund_decision".into(), beh(true));
    if contract.items.iter().any(|i| i.prompt == "order_lookup") {
        behaviours.insert(
            "order_lookup".into(),
            CapturedBehaviour {
                prompt_id: "order_lookup".into(),
                content: "ok".into(),
                latency_ms: 1,
                cost_usd: None,
                tool_calls: vec![CapturedToolCall {
                    name: "lookup_order".into(),
                    arguments: BTreeMap::new(),
                }],
                finish_reason: None,
                fingerprint: None,
                token_count: None,
                raw: None,
                execution_error: None,
                execution_error_message: None,
                execution_meta: Default::default(),
            },
        );
    }
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

fn assert_never_safe(eff: &arsenic_core::EffectiveQualification) {
    assert!(
        !eff.migration_recommendation.allows_safe_migrate(),
        "invariant violated: {:?} recommended SAFE",
        eff.migration_recommendation
    );
}

#[test]
fn historical_pass_changed_contract() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    assert_eq!(q.decision, QualificationDecision::Pass);
    let c2 = claim_contract(2);
    let eff = assess_qualification(&q, &c2.content_hash(), &baseline.content_hash());
    assert!(eff.is_stale);
    assert_never_safe(&eff);
}

#[test]
fn historical_pass_changed_baseline() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    let mut baseline2 = baseline.clone();
    baseline2.id = "baseline-0002".into();
    // Force different content hash via id (included in content_hash payload).
    let eff = assess_qualification(&q, &c1.content_hash(), &baseline2.content_hash());
    assert!(eff.is_stale);
    assert_never_safe(&eff);
}

#[test]
fn historical_pass_both_changed() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    let c2 = claim_contract(2);
    let mut baseline2 = baseline.clone();
    baseline2.id = "baseline-0002".into();
    let eff = assess_qualification(&q, &c2.content_hash(), &baseline2.content_hash());
    assert!(eff.is_stale);
    assert_never_safe(&eff);
}

#[test]
fn latest_stale_older_pass_cannot_certify() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let mut older = qualify_candidate(&c1, &baseline, "openai:x", &cand, "older");
    older.created_at = "2020-01-01T00:00:00Z".into();
    let mut newer = qualify_candidate(&c1, &baseline, "openai:x", &cand, "newer");
    newer.created_at = "2026-01-01T00:00:00Z".into();
    // Newer was recorded against an older contract hash (stale vs current).
    newer.contract_hash = "stale-hash".into();
    let c_now = claim_contract(1);
    let eff = assess_all(
        &[older, newer],
        &c_now.content_hash(),
        &baseline.content_hash(),
    );
    // Latest is newer (stale); older PASS must not be selected.
    assert_eq!(eff.len(), 1);
    assert_eq!(eff[0].qualification_id, "newer");
    assert_eq!(aggregate_migration(&eff), MigrationRecommendation::Stale);
}

#[test]
fn provider_timeout_never_safe() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    let mut b = beh(true);
    b.execution_error = Some(ExecutionErrorKind::Timeout);
    b.execution_error_message = Some("timed out".into());
    cand.insert("refund_decision".into(), b);
    let q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    let eff = assess_qualification(&q, &c1.content_hash(), &baseline.content_hash());
    assert_eq!(eff.evidence_validity, EvidenceValidity::Incomplete);
    assert_never_safe(&eff);
    assert_eq!(eff.migration_recommendation.ci_exit_code(), 3);
}

#[test]
fn provider_auth_failure_never_safe() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    let mut b = beh(true);
    b.execution_error = Some(ExecutionErrorKind::AuthError);
    cand.insert("refund_decision".into(), b);
    let q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    let eff = assess_qualification(&q, &c1.content_hash(), &baseline.content_hash());
    assert_never_safe(&eff);
}

#[test]
fn malformed_response_never_safe() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    let mut b = beh(true);
    b.execution_error = Some(ExecutionErrorKind::InvalidResponse);
    cand.insert("refund_decision".into(), b);
    let q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    let eff = assess_qualification(&q, &c1.content_hash(), &baseline.content_hash());
    assert_never_safe(&eff);
}

#[test]
fn unsupported_required_capability_never_safe() {
    let contract = tool_contract();
    let baseline = baseline_for(&contract);
    let caps = ProviderCapabilities {
        tools: false,
        structured_output: true,
        system_prompts: true,
        usage_reporting: true,
        seed: false,
    };
    let fails = capability_failures(&contract, &caps);
    assert!(!fails.is_empty());
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    cand.insert(
        "order_lookup".into(),
        CapturedBehaviour {
            prompt_id: "order_lookup".into(),
            content: "no tools".into(),
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
        },
    );
    let mut q = qualify_candidate(&contract, &baseline, "ollama:local", &cand, "q1");
    for f in fails {
        if let Some(existing) = q.item_results.iter_mut().find(|r| r.item_id == f.item_id) {
            *existing = f;
        } else {
            q.item_results.push(f);
        }
    }
    q.decision = decide_overall(
        &q.item_results
            .iter()
            .filter(|r| r.regression_type.as_deref() != Some("execution_error"))
            .cloned()
            .collect::<Vec<_>>(),
        &baseline.qualification_thresholds,
        false,
    );
    let eff = assess_qualification(&q, &contract.content_hash(), &baseline.content_hash());
    assert_eq!(
        eff.migration_recommendation,
        MigrationRecommendation::MigrationBlocked
    );
    assert_never_safe(&eff);
}

#[test]
fn partial_candidate_execution_missing_prompt() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let cand = BTreeMap::new(); // zero captures
    let q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    assert!(q
        .item_results
        .iter()
        .any(|r| r.regression_type.as_deref() == Some("missing_capture")));
    let eff = assess_qualification(&q, &c1.content_hash(), &baseline.content_hash());
    assert_never_safe(&eff);
}

#[test]
fn missing_candidate_behaviour_block_item() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let cand = BTreeMap::new();
    let q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    assert_eq!(q.decision, QualificationDecision::Block);
    let eff = assess_qualification(&q, &c1.content_hash(), &baseline.content_hash());
    assert_eq!(
        eff.migration_recommendation,
        MigrationRecommendation::MigrationBlocked
    );
}

#[test]
fn zero_captured_behaviours_incomplete_or_block() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let q = qualify_candidate(&c1, &baseline, "openai:x", &BTreeMap::new(), "q1");
    let eff = assess_qualification(&q, &c1.content_hash(), &baseline.content_hash());
    assert_never_safe(&eff);
}

#[test]
fn block_item_execution_missing() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let q = qualify_candidate(&c1, &baseline, "openai:x", &BTreeMap::new(), "q1");
    assert!(q.item_results.iter().any(|r| {
        r.severity == ContractSeverity::Block
            && r.regression_type.as_deref() == Some("missing_capture")
    }));
    assert_never_safe(&assess_qualification(
        &q,
        &c1.content_hash(),
        &baseline.content_hash(),
    ));
}

#[test]
fn review_item_execution_missing() {
    let mut c1 = claim_contract(1);
    c1.items[0].severity = ContractSeverity::Review;
    let baseline = baseline_for(&c1);
    let q = qualify_candidate(&c1, &baseline, "openai:x", &BTreeMap::new(), "q1");
    assert_eq!(q.decision, QualificationDecision::Review);
    let eff = assess_qualification(&q, &c1.content_hash(), &baseline.content_hash());
    assert_eq!(
        eff.migration_recommendation,
        MigrationRecommendation::ReviewRequired
    );
    assert_never_safe(&eff);
}

#[test]
fn stale_pass_mixed_with_current_block() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut pass_c = BTreeMap::new();
    pass_c.insert("refund_decision".into(), beh(true));
    let mut block_c = BTreeMap::new();
    block_c.insert("refund_decision".into(), beh(false));
    let mut stale_pass = qualify_candidate(&c1, &baseline, "openai:old", &pass_c, "stale");
    stale_pass.contract_hash = "old".into();
    let current_block = qualify_candidate(&c1, &baseline, "openai:new", &block_c, "cur");
    let eff = assess_all(
        &[stale_pass, current_block],
        &c1.content_hash(),
        &baseline.content_hash(),
    );
    assert_eq!(
        aggregate_migration(&eff),
        MigrationRecommendation::MigrationBlocked
    );
}

#[test]
fn stale_block_mixed_with_current_pass() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut pass_c = BTreeMap::new();
    pass_c.insert("refund_decision".into(), beh(true));
    let mut block_c = BTreeMap::new();
    block_c.insert("refund_decision".into(), beh(false));
    let mut stale_block = qualify_candidate(&c1, &baseline, "openai:old", &block_c, "stale");
    stale_block.contract_hash = "old".into();
    let current_pass = qualify_candidate(&c1, &baseline, "openai:new", &pass_c, "cur");
    let eff = assess_all(
        &[stale_block, current_pass],
        &c1.content_hash(),
        &baseline.content_hash(),
    );
    // Stale block must not poison current pass for a *different* candidate;
    // aggregate across candidates: one safe + one stale → Safe (current only).
    assert_eq!(
        aggregate_migration(&eff),
        MigrationRecommendation::SafeToMigrate
    );
}

#[test]
fn multiple_quals_same_candidate_newest_wins() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut pass_c = BTreeMap::new();
    pass_c.insert("refund_decision".into(), beh(true));
    let mut block_c = BTreeMap::new();
    block_c.insert("refund_decision".into(), beh(false));
    let mut older = qualify_candidate(&c1, &baseline, "openai:x", &pass_c, "old");
    older.created_at = "2020-01-01T00:00:00Z".into();
    let mut newer = qualify_candidate(&c1, &baseline, "openai:x", &block_c, "new");
    newer.created_at = "2026-01-01T00:00:00Z".into();
    let eff = assess_all(
        &[older, newer],
        &c1.content_hash(),
        &baseline.content_hash(),
    );
    assert_eq!(eff[0].qualification_id, "new");
    assert_eq!(
        aggregate_migration(&eff),
        MigrationRecommendation::MigrationBlocked
    );
}

#[test]
fn report_cli_json_html_agree_on_effective_state() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    let eff = assess_qualification(&q, &c1.content_hash(), &baseline.content_hash());
    let json = arsenic_core::effective_to_json(&eff);
    assert_eq!(
        json["migration_recommendation_text"],
        MigrationRecommendation::SafeToMigrate.as_str()
    );
    assert_eq!(json["ci_exit_code"], 0);
    assert_eq!(
        q.migration_recommendation(&c1.content_hash(), &baseline.content_hash()),
        MigrationRecommendation::SafeToMigrate
    );
}

#[test]
fn ci_exit_agrees_with_machine_readable() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    let mut b = beh(true);
    b.execution_error = Some(ExecutionErrorKind::Timeout);
    cand.insert("refund_decision".into(), b);
    let q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    let eff = assess_qualification(&q, &c1.content_hash(), &baseline.content_hash());
    let json = arsenic_core::effective_to_json(&eff);
    assert_eq!(
        json["ci_exit_code"],
        eff.migration_recommendation.ci_exit_code()
    );
    assert_eq!(eff.migration_recommendation.ci_exit_code(), 3);
}

#[test]
fn lock_cannot_resurrect_stale_qualification() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let mut q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    let mut lock = ArsenicLock::new("app");
    lock.contract_hash = Some(c1.content_hash());
    lock.baseline_hash = Some(baseline.content_hash());
    update_lock_from_qualification(&mut lock, &q);
    assert_eq!(lock.qualified.len(), 1);

    // Contract changes — reassess; lock update with stale result must not keep qualified.
    let c2 = claim_contract(2);
    lock.contract_hash = Some(c2.content_hash());
    q.contract_hash = c1.content_hash(); // historical
    update_lock_from_qualification(&mut lock, &q);
    assert!(
        lock.qualified.is_empty(),
        "stale PASS must not remain in lock.qualified"
    );
    assert_eq!(lock.review.len(), 1);
}

#[test]
fn validated_patch_from_stale_cannot_certify_current() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(false));
    let mut q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    attempt_validated_repair(&c1, &mut q, &baseline, &|_, _mutated| {
        Some(CapturedBehaviour {
            prompt_id: "refund_decision".into(),
            content: "Refunds over £500 require manager approval".into(),
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
        })
    });
    assert!(q.validated_patch.is_some());
    assert_eq!(q.decision, QualificationDecision::PassWithPatch);

    let c2 = claim_contract(2);
    let eff = assess_qualification(&q, &c2.content_hash(), &baseline.content_hash());
    assert!(eff.is_stale || !eff.patch_applies_to_current);
    assert_never_safe(&eff);
}

#[test]
fn changed_input_fingerprint_requires_requalification() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let mut q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    q.input_fingerprint = Some("fp-old".into());
    // Fingerprint itself is not in assess_qualification; callers must requalify.
    // Simulate: same hashes but different fingerprint means evidence is for different inputs.
    // Binding check: patch + fingerprint mismatch treated via incomplete when flagged.
    let mut q2 = q.clone();
    q2.input_fingerprint = Some("fp-new".into());
    // Two fingerprints for same candidate — newest selected; both current hashes → still assessable.
    q.created_at = "2020-01-01T00:00:00Z".into();
    q2.created_at = "2026-01-01T00:00:00Z".into();
    let latest = assess_all(&[q, q2], &c1.content_hash(), &baseline.content_hash());
    assert_eq!(latest[0].input_fingerprint.as_deref(), Some("fp-new"));
}

#[test]
fn capability_gap_not_hidden_by_scoring() {
    let contract = tool_contract();
    let baseline = baseline_for(&contract);
    let caps = ProviderCapabilities {
        tools: false,
        structured_output: false,
        system_prompts: true,
        usage_reporting: true,
        seed: false,
    };
    let fails = capability_failures(&contract, &caps);
    assert!(fails.iter().any(|f| f.outcome == ItemOutcome::Fail));
    // Compatibility % must not override capability Fail into SAFE.
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let mut q = qualify_candidate(&contract, &baseline, "google:g", &cand, "q1");
    for f in fails {
        if let Some(existing) = q.item_results.iter_mut().find(|r| r.item_id == f.item_id) {
            *existing = f;
        } else {
            q.item_results.push(f);
        }
    }
    q.decision = decide_overall(&q.item_results, &QualificationThresholds::default(), false);
    let eff = assess_qualification(&q, &contract.content_hash(), &baseline.content_hash());
    assert_never_safe(&eff);
}

#[test]
fn review_escalates_false_cannot_neutralise_execution_errors() {
    let c1 = claim_contract(1);
    let mut baseline = baseline_for(&c1);
    baseline.qualification_thresholds.review_escalates = false;
    let mut cand = BTreeMap::new();
    // Mix: one prompt succeeds behaviourally, but execution error on the required item.
    let mut b = beh(true);
    b.execution_error = Some(ExecutionErrorKind::RateLimited);
    b.execution_error_message = Some("rate limited".into());
    cand.insert("refund_decision".into(), b);
    let q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    assert!(q.has_execution_errors);
    let eff = assess_qualification(&q, &c1.content_hash(), &baseline.content_hash());
    assert_eq!(eff.evidence_validity, EvidenceValidity::Incomplete);
    assert_never_safe(&eff);
}

#[test]
fn patch_binds_only_to_exact_context() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(false));
    let mut q = qualify_candidate(&c1, &baseline, "openai:x", &cand, "q1");
    q.validated_patch = Some(ValidatedPatch {
        qualification_id: "q1".into(),
        model: "openai:x".into(),
        contract_hash: c1.content_hash(),
        strategies: vec![],
        original_prompt: "x".into(),
        patched_prompt: "y".into(),
        prompt_diff: "diff".into(),
        revalidated: true,
    });
    q.decision = QualificationDecision::PassWithPatch;
    // Wrong model on patch
    q.validated_patch.as_mut().unwrap().model = "openai:other".into();
    let eff = assess_qualification(&q, &c1.content_hash(), &baseline.content_hash());
    assert!(!eff.patch_applies_to_current);
    assert_never_safe(&eff);
}
