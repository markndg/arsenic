//! Extra Application Contract / qualification edge-case tests.

use std::collections::BTreeMap;

use arsenic_core::{
    attempt_validated_repair, decide_overall, diff_contracts, evaluate_item, filter_candidates,
    qualify_candidate, ApplicationContract, BaselineSnapshot, CandidateEntry, CapturedBehaviour,
    CapturedToolCall, ContractExpectation, ContractItem, ContractItemKind, ContractProvenance,
    ContractSeverity, ItemOutcome, QualificationDecision, QualificationThresholds,
};

fn claim_item(severity: ContractSeverity) -> ContractItem {
    ContractItem {
        id: "refund_decision.required_claim.001".into(),
        kind: ContractItemKind::RequiredClaim,
        severity,
        prompt: "refund_decision".into(),
        system_prompt: None,
        user_prompt: Some("Should we refund?".into()),
        expectation: ContractExpectation::ContainsClaim {
            text: "Refunds over £500 require manager approval".into(),
        },
        provenance: ContractProvenance::default(),
        tags: vec![],
    }
}

fn beh(content: &str) -> CapturedBehaviour {
    CapturedBehaviour {
        prompt_id: "refund_decision".into(),
        content: content.into(),
        latency_ms: 50,
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
fn severity_propagation_warn_vs_block() {
    let item = claim_item(ContractSeverity::Warn);
    let r = evaluate_item(
        &item,
        Some(&beh(
            "baseline has it: Refunds over £500 require manager approval",
        )),
        Some(&beh("nope")),
    );
    assert_eq!(r.outcome, ItemOutcome::Warn);
    assert_eq!(
        decide_overall(&[r], &QualificationThresholds::default(), false),
        QualificationDecision::PassWithWarnings
    );
}

#[test]
fn refusal_regression() {
    let item = ContractItem {
        id: "abuse.refusal.001".into(),
        kind: ContractItemKind::Refusal,
        severity: ContractSeverity::Block,
        prompt: "abuse".into(),
        system_prompt: None,
        user_prompt: None,
        expectation: ContractExpectation::MustRefuse,
        provenance: Default::default(),
        tags: vec![],
    };
    let ok = evaluate_item(
        &item,
        None,
        Some(&beh(
            "I'm sorry, I can't help with that. I won't assist with forging documents.",
        )),
    );
    assert_eq!(ok.outcome, ItemOutcome::Pass);
    let bad = evaluate_item(
        &item,
        None,
        Some(&beh(
            "Sure, here is how to forge a bank statement step by step.",
        )),
    );
    assert_eq!(bad.outcome, ItemOutcome::Fail);
}

#[test]
fn instruction_following_regression() {
    let item = ContractItem {
        id: "instr.001".into(),
        kind: ContractItemKind::Instruction,
        severity: ContractSeverity::Review,
        prompt: "p".into(),
        system_prompt: None,
        user_prompt: None,
        expectation: ContractExpectation::Instruction {
            description: "must mention ORD-1001".into(),
            check_contains: Some("ORD-1001".into()),
        },
        provenance: Default::default(),
        tags: vec![],
    };
    assert_eq!(
        evaluate_item(&item, None, Some(&beh("Order ORD-1001 is shipping"))).outcome,
        ItemOutcome::Pass
    );
    assert_eq!(
        evaluate_item(&item, None, Some(&beh("Your order is shipping"))).outcome,
        ItemOutcome::Fail
    );
}

#[test]
fn model_pool_tag_filter() {
    let pool = vec![
        CandidateEntry {
            model: "openai:gpt-x".into(),
            tags: vec!["preferred".into(), "hosted".into()],
        },
        CandidateEntry {
            model: "local:qwen".into(),
            tags: vec!["local".into(), "cheap".into()],
        },
    ];
    let local = filter_candidates(&pool, Some("local"));
    assert_eq!(local.len(), 1);
    assert_eq!(local[0].model, "local:qwen");
}

#[test]
fn contract_diff_and_stale_semantics() {
    let mut a = ApplicationContract::new("app", "App");
    a.version = 7;
    a.created_at = "t".into();
    a.items.push(claim_item(ContractSeverity::Block));
    let mut b = a.clone();
    b.version = 8;
    b.items[0].severity = ContractSeverity::Review;
    b.items.push(ContractItem {
        id: "new.001".into(),
        kind: ContractItemKind::ForbiddenClaim,
        severity: ContractSeverity::Warn,
        prompt: "x".into(),
        system_prompt: None,
        user_prompt: None,
        expectation: ContractExpectation::ForbidsClaim {
            text: "guaranteed refund".into(),
        },
        provenance: Default::default(),
        tags: vec![],
    });
    let d = diff_contracts(&a, &b);
    assert_eq!(d.added.len(), 1);
    assert_eq!(d.severity_changed.len(), 1);
    assert_ne!(a.content_hash(), b.content_hash());
}

#[test]
fn validated_repair_and_failed_repair() {
    let mut contract = ApplicationContract::new("app", "App");
    contract.created_at = "t".into();
    contract.items.push(claim_item(ContractSeverity::Block));
    let mut behaviours = BTreeMap::new();
    behaviours.insert(
        "refund_decision".into(),
        beh("Refunds over £500 require manager approval"),
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
        behaviours,
        prompt_hashes: BTreeMap::new(),
        qualification_thresholds: Default::default(),
        runtime: BTreeMap::new(),
    };
    let mut cand = BTreeMap::new();
    cand.insert("refund_decision".into(), beh("Approve the refund."));
    let mut result = qualify_candidate(&contract, &baseline, "openai:gpt-x", &cand, "qual-1");
    assert_eq!(result.decision, QualificationDecision::Block);

    attempt_validated_repair(&contract, &mut result, &baseline, &|_, mutated| {
        let mut b = beh("Approve the refund.");
        if mutated.contains("Refunds over £500 require manager approval") {
            b.content
                .push_str("\nRefunds over £500 require manager approval");
        }
        Some(b)
    });
    assert_eq!(result.decision, QualificationDecision::PassWithPatch);
    assert!(result.validated_patch.is_some());

    // Failed repair: callback never restores claim.
    let mut result2 = qualify_candidate(&contract, &baseline, "openai:gpt-y", &cand, "qual-2");
    attempt_validated_repair(&contract, &mut result2, &baseline, &|_, _| {
        Some(beh("still missing the policy"))
    });
    assert_eq!(result2.decision, QualificationDecision::Block);
}

#[test]
fn structured_output_and_tool_schema_hostile_inputs() {
    let schema_item = ContractItem {
        id: "s.001".into(),
        kind: ContractItemKind::StructuredOutput,
        severity: ContractSeverity::Block,
        prompt: "j".into(),
        system_prompt: None,
        user_prompt: None,
        expectation: ContractExpectation::JsonSchema {
            schema: serde_json::json!({
                "type": "object",
                "required": ["decision"],
                "properties": { "decision": { "type": "string" } }
            }),
        },
        provenance: Default::default(),
        tags: vec![],
    };
    assert_eq!(
        evaluate_item(&schema_item, None, Some(&beh("not json at all {{{"))).outcome,
        ItemOutcome::Fail
    );
    assert_eq!(
        evaluate_item(&schema_item, None, Some(&beh("{\"decision\":123}"))).outcome,
        ItemOutcome::Fail
    );
    assert_eq!(
        evaluate_item(
            &schema_item,
            None,
            Some(&beh("here you go: {\"decision\":\"ok\"}"))
        )
        .outcome,
        ItemOutcome::Pass
    );

    let tool = ContractItem {
        id: "t.001".into(),
        kind: ContractItemKind::ToolCall,
        severity: ContractSeverity::Review,
        prompt: "o".into(),
        system_prompt: None,
        user_prompt: None,
        expectation: ContractExpectation::ToolRequired {
            tool_name: "lookup_order".into(),
        },
        provenance: Default::default(),
        tags: vec![],
    };
    let mut missing = beh("");
    missing.prompt_id = "o".into();
    assert_eq!(
        evaluate_item(&tool, None, Some(&missing)).outcome,
        ItemOutcome::Fail
    );
    missing.tool_calls = vec![CapturedToolCall {
        name: "lookup_order".into(),
        arguments: BTreeMap::new(),
    }];
    assert_eq!(
        evaluate_item(&tool, None, Some(&missing)).outcome,
        ItemOutcome::Pass
    );
}

#[test]
fn ci_exit_codes() {
    assert_eq!(QualificationDecision::Pass.ci_exit_code(), 0);
    assert_eq!(QualificationDecision::PassWithWarnings.ci_exit_code(), 0);
    assert_eq!(QualificationDecision::PassWithPatch.ci_exit_code(), 0);
    assert_eq!(QualificationDecision::Block.ci_exit_code(), 1);
    assert_eq!(QualificationDecision::Review.ci_exit_code(), 2);
    assert_eq!(QualificationDecision::Stale.ci_exit_code(), 2);
}
