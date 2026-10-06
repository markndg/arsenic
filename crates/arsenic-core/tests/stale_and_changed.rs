//! Stale qualification history preservation + --changed fingerprint tests.

use std::collections::BTreeMap;

use arsenic_core::{
    candidate_input_fingerprint, content_hash, qualify_candidate, ApplicationContract,
    BaselineSnapshot, CapturedBehaviour, ContractExpectation, ContractItem, ContractItemKind,
    ContractSeverity, QualificationDecision,
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

#[test]
fn stale_is_derived_without_mutating_original_decision() {
    let contract_v1 = claim_contract(1);
    let mut behaviours = BTreeMap::new();
    behaviours.insert("refund_decision".into(), beh(true));
    let baseline = BaselineSnapshot {
        schema_version: 1,
        id: "baseline-0001".into(),
        version: 1,
        application_id: "app".into(),
        contract_version: 1,
        contract_hash: contract_v1.content_hash(),
        model: "openai:prod".into(),
        provider_model_id: "prod".into(),
        created_at: "t".into(),
        behaviours,
        prompt_hashes: BTreeMap::new(),
        qualification_thresholds: Default::default(),
        runtime: BTreeMap::new(),
    };
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh(true));
    let q = qualify_candidate(&contract_v1, &baseline, "openai:gpt-x", &cand, "qual-1");
    assert_eq!(q.decision, QualificationDecision::Pass);
    assert_eq!(q.original_decision, Some(QualificationDecision::Pass));

    let contract_v2 = claim_contract(2);
    // Derived stale — historical decision untouched.
    assert_eq!(
        q.effective_decision(&contract_v2.content_hash(), &baseline.content_hash()),
        QualificationDecision::Stale
    );
    assert_eq!(q.decision, QualificationDecision::Pass);
    assert_eq!(q.contract_hash, contract_v1.content_hash());
}

#[test]
fn changed_fingerprint_reacts_to_contract_and_config() {
    let mut prompts = BTreeMap::new();
    prompts.insert("refund_decision".into(), content_hash("refund?"));
    let a = candidate_input_fingerprint(
        "c1",
        "b1",
        "openai:gpt-x",
        &prompts,
        None,
        0.0,
        Some(256),
        Some("https://api"),
        "th1",
    );
    let b = candidate_input_fingerprint(
        "c1",
        "b1",
        "openai:gpt-x",
        &prompts,
        None,
        0.0,
        Some(256),
        Some("https://api"),
        "th1",
    );
    assert_eq!(a, b);
    let c = candidate_input_fingerprint(
        "c2",
        "b1",
        "openai:gpt-x",
        &prompts,
        None,
        0.0,
        Some(256),
        Some("https://api"),
        "th1",
    );
    assert_ne!(a, c);
    let d = candidate_input_fingerprint(
        "c1",
        "b1",
        "openai:gpt-x",
        &prompts,
        None,
        0.2,
        Some(256),
        Some("https://api"),
        "th1",
    );
    assert_ne!(a, d);
}
