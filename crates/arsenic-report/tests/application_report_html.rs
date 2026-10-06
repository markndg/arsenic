//! Qualification HTML report rendering tests — visual semantics + fail-closed messaging.

use std::collections::BTreeMap;

use arsenic_core::{
    qualify_candidate, ApplicationContract, BaselineSnapshot, CapturedBehaviour,
    ContractExpectation, ContractItem, ContractItemKind, ContractProvenance, ContractSeverity,
    ExecutionErrorKind, QualificationDecision, QualificationThresholds, ValidatedPatch,
};
use arsenic_report::render_application_report;

fn claim_contract(version: u32) -> ApplicationContract {
    let mut c = ApplicationContract::new("customer-support", "Customer Support");
    c.version = version;
    c.created_at = "2026-01-01T00:00:00Z".into();
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

fn beh(ok: bool) -> CapturedBehaviour {
    CapturedBehaviour {
        prompt_id: "refund_decision".into(),
        content: if ok {
            "Refunds over £500 require manager approval".into()
        } else {
            "approve".into()
        },
        latency_ms: 10,
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
    BaselineSnapshot {
        schema_version: 1,
        id: "baseline-0001".into(),
        version: 1,
        application_id: "customer-support".into(),
        contract_version: contract.version,
        contract_hash: contract.content_hash(),
        model: "openai:gpt-current".into(),
        provider_model_id: "gpt-current".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
        behaviours,
        prompt_hashes: BTreeMap::new(),
        qualification_thresholds: QualificationThresholds::default(),
        runtime: BTreeMap::new(),
    }
}

fn assert_no_poison(html: &str) {
    assert!(!html.contains("NaN"), "HTML must not contain NaN");
    assert!(!html.contains("Infinity"), "HTML must not contain Infinity");
    assert!(
        !html.contains("fonts.googleapis.com"),
        "qualification HTML must not depend on Google Fonts CDN"
    );
    assert!(
        !html.contains("cdn."),
        "qualification HTML must not load external CDN assets"
    );
    assert!(
        !html.contains("<script src="),
        "qualification HTML must not load external scripts"
    );
}

fn assert_no_contradictory_safe(html: &str) {
    // Top-level decision banner must not claim SAFE while also STALE/INCOMPLETE aggregate.
    let has_top_safe =
        html.contains("decision-banner safe") || html.contains("class=\"decision-banner safe\"");
    let has_top_stale = html.contains("decision-banner stale");
    let has_top_incomplete = html.contains("decision-banner incomplete");
    if has_top_safe {
        assert!(
            !has_top_stale && !has_top_incomplete,
            "top-level SAFE must not coincide with STALE/INCOMPLETE banners"
        );
    }
}

#[test]
fn html_current_pass_shows_safe_and_valid() {
    let c = claim_contract(1);
    let baseline = baseline_for(&c);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let q = qualify_candidate(&c, &baseline, "openai:gpt-safe", &cand, "qual-0001");
    assert_eq!(q.decision, QualificationDecision::Pass);

    let html = render_application_report(&c, Some(&baseline), &[q], None).expect("render");
    assert!(html.contains("SAFE TO MIGRATE"));
    assert!(html.contains("VALID"));
    assert!(html.contains("Customer Support"));
    assert!(html.contains("openai:gpt-safe"));
    assert_no_poison(&html);
    assert_no_contradictory_safe(&html);
    // Drift report markers must not be required here; ensure Arsenic product chrome.
    assert!(html.contains("ARSENIC · Qualification"));
    assert!(html.contains("Migration decision"));
}

#[test]
fn html_current_block_shows_migration_blocked() {
    let c = claim_contract(1);
    let baseline = baseline_for(&c);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(false));
    let q = qualify_candidate(&c, &baseline, "openai:gpt-bad", &cand, "qual-0001");
    assert_eq!(q.decision, QualificationDecision::Block);

    let html = render_application_report(&c, Some(&baseline), &[q], None).expect("render");
    assert!(html.contains("MIGRATION BLOCKED"));
    assert!(html.contains("BLOCK"));
    assert_no_poison(&html);
    assert_no_contradictory_safe(&html);
    assert!(!html.contains("decision-banner safe"));
}

#[test]
fn html_stale_historical_pass_requires_requalify() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let q = qualify_candidate(&c1, &baseline, "openai:gpt-safe", &cand, "qual-0001");
    assert_eq!(q.decision, QualificationDecision::Pass);

    let c2 = claim_contract(2);
    let html = render_application_report(&c2, Some(&baseline), &[q], None).expect("render");

    assert!(html.contains("recorded"));
    assert!(html.contains("PASS"));
    assert!(html.contains("STALE"));
    assert!(html.contains("REQUALIFY REQUIRED"));
    assert!(html.contains("Historical qualification"));
    // Must not present top-level SAFE for stale-only evidence.
    assert!(!html.contains("decision-banner safe"));
    assert!(html.contains("STALE — REQUALIFY REQUIRED") || html.contains("REQUALIFY REQUIRED"));
    assert_no_poison(&html);
    assert_no_contradictory_safe(&html);
}

#[test]
fn html_incomplete_execution_shows_reasons() {
    let c = claim_contract(1);
    let baseline = baseline_for(&c);
    let mut cand = BTreeMap::new();
    let mut b = beh(true);
    b.execution_error = Some(ExecutionErrorKind::Timeout);
    b.execution_error_message = Some("provider timed out".into());
    cand.insert("refund_decision".into(), b);
    let q = qualify_candidate(&c, &baseline, "openai:gpt-timeout", &cand, "qual-0001");

    let html = render_application_report(&c, Some(&baseline), &[q], None).expect("render");
    assert!(html.contains("QUALIFICATION INCOMPLETE"));
    assert!(
        html.contains("provider/runtime execution errors")
            || html.contains("TIMEOUT")
            || html.contains("execution")
    );
    assert!(html.contains("could not obtain sufficient evidence") || html.contains("INCOMPLETE"));
    assert!(!html.contains("decision-banner safe"));
    assert_no_poison(&html);
    assert_no_contradictory_safe(&html);
}

#[test]
fn html_validated_current_patch_marked_current() {
    let c = claim_contract(1);
    let baseline = baseline_for(&c);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let mut q = qualify_candidate(&c, &baseline, "openai:gpt-repairable", &cand, "qual-0001");
    q.decision = QualificationDecision::PassWithPatch;
    q.validated_patch = Some(ValidatedPatch {
        qualification_id: "qual-0001".into(),
        model: "openai:gpt-repairable".into(),
        contract_hash: c.content_hash(),
        strategies: vec!["claim".into()],
        prompt_diff: "--- original\n+++ patched\n+must include claim".into(),
        original_prompt: "refund?".into(),
        patched_prompt: "refund? must include claim".into(),
        revalidated: true,
    });

    let html = render_application_report(&c, Some(&baseline), &[q], None).expect("render");
    assert!(html.contains("SAFE TO MIGRATE") || html.contains("PASS_WITH_PATCH"));
    assert!(html.contains("applies to current contract/baseline"));
    assert!(html.contains("Prompt diff") || html.contains("must include claim"));
    assert!(!html.contains("no longer valid for the current contract/baseline"));
    assert_no_poison(&html);
}

#[test]
fn html_stale_patch_not_presented_as_current() {
    let c1 = claim_contract(1);
    let baseline = baseline_for(&c1);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let mut q = qualify_candidate(&c1, &baseline, "openai:gpt-repairable", &cand, "qual-0001");
    q.decision = QualificationDecision::PassWithPatch;
    q.validated_patch = Some(ValidatedPatch {
        qualification_id: "qual-0001".into(),
        model: "openai:gpt-repairable".into(),
        contract_hash: c1.content_hash(),
        strategies: vec![],
        prompt_diff: "--- a\n+++ b\n+stale patch body".into(),
        original_prompt: "x".into(),
        patched_prompt: "y".into(),
        revalidated: true,
    });

    let c2 = claim_contract(2);
    let html = render_application_report(&c2, Some(&baseline), &[q], None).expect("render");
    assert!(html.contains("no longer valid for the current contract/baseline"));
    assert!(!html.contains("applies to current contract/baseline"));
    assert!(!html.contains("decision-banner safe"));
    assert_no_poison(&html);
    assert_no_contradictory_safe(&html);
}

#[test]
fn html_surfaces_hash_provenance() {
    let c = claim_contract(1);
    let baseline = baseline_for(&c);
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let q = qualify_candidate(&c, &baseline, "openai:gpt-safe", &cand, "qual-0001");
    let html = render_application_report(&c, Some(&baseline), &[q], None).expect("render");
    assert!(html.contains("Qual contract hash"));
    assert!(html.contains("Current contract hash"));
    assert!(html.contains("Qual baseline hash"));
    assert!(html.contains("Current baseline hash"));
    assert!(html.contains("Qualification ID"));
    assert!(html.contains("assess_qualification"));
}

#[test]
fn drift_report_still_renders_fingerprint_radar() {
    // Regression guard: this work must not disturb Drift Report rendering.
    use arsenic_core::DriftReport;
    use arsenic_report::ReportRenderer;

    let fixture = include_str!("../fixtures/report_openai_upgrade.json");
    let report: DriftReport = serde_json::from_str(fixture).expect("parse fixture");
    let html = ReportRenderer::render_html(&report).expect("drift html");
    assert!(html.contains("Behavioural fingerprint"));
    assert!(html.contains("class=\"fp-baseline\""));
    assert!(html.contains("class=\"fp-candidate\""));
    assert!(html.contains("Retention table"));
    assert!(html.contains("Behavioural changelog"));
}
