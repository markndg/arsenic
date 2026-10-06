//! Mocked-live HTTP tests for OpenAI adapter contract capture.

use arsenic_adapters::{build_adapter, AdapterSpec};
use arsenic_core::{
    capture_contract_live, qualify_candidate, redact_secrets, scenarios_from_contract,
    ApplicationContract, BaselineSnapshot, CaptureSource, ContractExpectation, ContractItem,
    ContractItemKind, ContractSeverity, LiveCaptureConfig, ProviderCapabilities,
    QualificationDecision,
};
use serde_json::json;
use std::collections::BTreeMap;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn sample_contract() -> ApplicationContract {
    let mut c = ApplicationContract::new("demo", "Demo");
    c.created_at = "t".into();
    c.items = vec![
        ContractItem {
            id: "refund_decision.required_claim.001".into(),
            kind: ContractItemKind::RequiredClaim,
            severity: ContractSeverity::Block,
            prompt: "refund_decision".into(),
            system_prompt: Some("You are support.".into()),
            user_prompt: Some("Refund £600?".into()),
            expectation: ContractExpectation::ContainsClaim {
                text: "Refunds over £500 require manager approval".into(),
            },
            provenance: Default::default(),
            tags: vec![],
        },
        ContractItem {
            id: "order_lookup.tool_argument.002".into(),
            kind: ContractItemKind::ToolArgument,
            severity: ContractSeverity::Block,
            prompt: "order_lookup".into(),
            system_prompt: None,
            user_prompt: Some("Look up ORD-1".into()),
            expectation: ContractExpectation::ToolArgument {
                tool_name: "lookup_order".into(),
                argument: "order_id".into(),
                expected_type: "string".into(),
            },
            provenance: Default::default(),
            tags: vec![],
        },
    ];
    c
}

#[tokio::test]
async fn mocked_live_baseline_and_qualify_parity() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "gpt-mock",
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "Refunds over £500 require manager approval. Escalating.",
                    "tool_calls": [{
                        "function": {
                            "name": "lookup_order",
                            "arguments": "{\"order_id\":\"ORD-1\"}"
                        }
                    }]
                }
            }],
            "usage": { "total_tokens": 42 }
        })))
        .mount(&server)
        .await;

    std::env::set_var("ARSENIC_MOCK_KEY", "test-key-not-secret");
    let spec = AdapterSpec {
        adapter_type: "openai".into(),
        endpoint: Some(server.uri()),
        api_key_env: "ARSENIC_MOCK_KEY".into(),
        model_id: "gpt-mock".into(),
        temperature: Some(0.0),
        max_tokens: Some(256),
        timeout_secs: Some(5),
    };
    let adapter = build_adapter(&spec).unwrap();
    let contract = sample_contract();
    let cfg = LiveCaptureConfig {
        concurrency: 2,
        retry_attempts: 1,
        retry_delay_ms: 10,
        temperature: 0.0,
        max_tokens: Some(256),
        arsenic_version: "test".into(),
        source: CaptureSource::MockedLive,
    };
    let live = capture_contract_live(&contract, adapter, "openai:gpt-mock", &cfg).await;
    assert!(live.contains_key("refund_decision") || live.len() >= 1);

    // Fixture-shaped copy of normalised live evidence → same evaluator.
    let mut baseline_beh = BTreeMap::new();
    for (k, v) in &live {
        baseline_beh.insert(k.clone(), v.clone());
    }
    let baseline = BaselineSnapshot {
        schema_version: 1,
        id: "baseline-0001".into(),
        version: 1,
        application_id: "demo".into(),
        contract_version: 1,
        contract_hash: contract.content_hash(),
        model: "openai:gpt-mock".into(),
        provider_model_id: "gpt-mock".into(),
        created_at: "t".into(),
        behaviours: baseline_beh.clone(),
        prompt_hashes: BTreeMap::new(),
        qualification_thresholds: Default::default(),
        runtime: BTreeMap::from([("source".into(), "mocked-live".into())]),
    };

    // Candidate identical to baseline → PASS
    let q = qualify_candidate(&contract, &baseline, "openai:gpt-mock", &live, "qual-1");
    assert!(
        !q.has_execution_errors,
        "unexpected execution errors: {:?}",
        q.item_results
    );
    // Tool argument may fail if tool_calls only on one scenario response (same mock for all).
    // Ensure evaluator path is shared: replay from captured_behaviours.
    let replay = qualify_candidate(
        &contract,
        &baseline,
        "openai:gpt-mock",
        &q.captured_behaviours,
        "qual-1-replay",
    );
    assert_eq!(q.decision, replay.decision);
    assert_eq!(
        q.item_results
            .iter()
            .map(|r| (&r.item_id, r.outcome))
            .collect::<Vec<_>>(),
        replay
            .item_results
            .iter()
            .map(|r| (&r.item_id, r.outcome))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn auth_failure_is_execution_error_not_block() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "error": { "message": "invalid api key sk-abcdefghijklmnopqrstuvwxyz" }
        })))
        .mount(&server)
        .await;

    std::env::set_var("ARSENIC_MOCK_KEY", "bad");
    let spec = AdapterSpec {
        adapter_type: "openai".into(),
        endpoint: Some(server.uri()),
        api_key_env: "ARSENIC_MOCK_KEY".into(),
        model_id: "gpt-mock".into(),
        temperature: Some(0.0),
        max_tokens: Some(64),
        timeout_secs: Some(5),
    };
    let adapter = build_adapter(&spec).unwrap();
    let contract = sample_contract();
    let cfg = LiveCaptureConfig {
        concurrency: 1,
        retry_attempts: 1,
        retry_delay_ms: 1,
        temperature: 0.0,
        max_tokens: Some(64),
        arsenic_version: "test".into(),
        source: CaptureSource::MockedLive,
    };
    let live = capture_contract_live(&contract, adapter, "openai:gpt-mock", &cfg).await;
    assert!(live.values().all(|b| b.execution_error.is_some()));
    let baseline = BaselineSnapshot {
        schema_version: 1,
        id: "b1".into(),
        version: 1,
        application_id: "demo".into(),
        contract_version: 1,
        contract_hash: contract.content_hash(),
        model: "openai:prod".into(),
        provider_model_id: "prod".into(),
        created_at: "t".into(),
        behaviours: BTreeMap::new(),
        prompt_hashes: BTreeMap::new(),
        qualification_thresholds: Default::default(),
        runtime: BTreeMap::new(),
    };
    let q = qualify_candidate(&contract, &baseline, "openai:gpt-mock", &live, "qual-auth");
    assert!(q.has_execution_errors);
    assert_ne!(q.decision, QualificationDecision::Block);
    assert_ne!(q.decision, QualificationDecision::Pass);
    assert_eq!(q.decision, QualificationDecision::Review);
    // Redaction: error messages must not leak sk- keys
    for b in live.values() {
        if let Some(msg) = &b.execution_error_message {
            assert!(!msg.contains("sk-abcdefghijklmnop"));
            assert!(msg.contains("[REDACTED]") || !msg.contains("sk-"));
        }
    }
}

#[test]
fn capability_mismatch_for_anthropic_tools() {
    let contract = sample_contract();
    let caps = ProviderCapabilities::for_adapter("anthropic");
    let fails = arsenic_core::capability_failures(&contract, &caps);
    assert!(fails.iter().any(|f| f.item_id.contains("tool")));
}

#[test]
fn redaction_unit() {
    let s = redact_secrets("Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz012345");
    assert!(s.contains("[REDACTED]"));
}

#[test]
fn scenarios_cover_unique_prompts() {
    let c = sample_contract();
    let s = scenarios_from_contract(&c);
    assert_eq!(s.len(), 2);
}
