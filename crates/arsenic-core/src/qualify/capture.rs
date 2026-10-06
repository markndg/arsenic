//! Live/fixture capture normalisation for Application Contracts.
//!
//! Provider responses are normalised once into [`CapturedBehaviour`]; evaluation
//! never branches on live vs fixture after this point.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::adapter::ModelAdapter;
use crate::contract::{ApplicationContract, ContractExpectation, ContractItem, ContractItemKind};
use crate::types::{FinishReason, ModelResponse, Probe, ProbeCategory, ProbeSource};
use uuid::Uuid;

use super::{
    CapturedBehaviour, CapturedToolCall, ContractItemResult, EvidenceSnippet, ItemOutcome,
};

/// Provider/runtime failure — distinct from behavioural FAIL.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExecutionErrorKind {
    ExecutionError,
    Timeout,
    RateLimited,
    AuthError,
    InvalidResponse,
    CapabilityUnsupported,
}

impl ExecutionErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExecutionError => "EXECUTION_ERROR",
            Self::Timeout => "TIMEOUT",
            Self::RateLimited => "RATE_LIMITED",
            Self::AuthError => "AUTH_ERROR",
            Self::InvalidResponse => "INVALID_RESPONSE",
            Self::CapabilityUnsupported => "CAPABILITY_UNSUPPORTED",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CaptureSource {
    #[default]
    Fixture,
    Live,
    Replay,
    MockedLive,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ExecutionMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub returned_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arsenic_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executed_at: Option<String>,
    #[serde(default)]
    pub source: CaptureSource,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProviderCapabilities {
    pub tools: bool,
    pub structured_output: bool,
    pub system_prompts: bool,
    pub usage_reporting: bool,
    pub seed: bool,
}

impl ProviderCapabilities {
    pub fn for_adapter(adapter_type: &str) -> Self {
        match adapter_type {
            "openai" | "ollama" => Self {
                tools: true,
                structured_output: true,
                system_prompts: true,
                usage_reporting: true,
                seed: false,
            },
            "anthropic" => Self {
                tools: false, // not wired in Arsenic adapters yet
                structured_output: false,
                system_prompts: true,
                usage_reporting: true,
                seed: false,
            },
            "google" => Self {
                tools: false,
                structured_output: false,
                system_prompts: true,
                usage_reporting: true,
                seed: false,
            },
            _ => Self::default(),
        }
    }
}

/// Redact secrets from provider JSON / error strings before persistence.
pub fn redact_secrets(text: &str) -> String {
    let mut out = text.to_string();
    if let Ok(re) = regex::Regex::new(r"(?i)bearer\s+[a-z0-9\-._~+/]+=*") {
        out = re.replace_all(&out, "Bearer [REDACTED]").to_string();
    }
    if let Ok(re) = regex::Regex::new(r"sk-[a-zA-Z0-9]{10,}") {
        out = re.replace_all(&out, "[REDACTED]").to_string();
    }
    if let Ok(re) = regex::Regex::new(r"AIza[0-9A-Za-z\-_]{20,}") {
        out = re.replace_all(&out, "[REDACTED]").to_string();
    }
    if let Ok(re) = regex::Regex::new(r"(?i)(api[_-]?key|authorization|x-api-key)\s*[:=]\s*\S+") {
        out = re.replace_all(&out, "$1: [REDACTED]").to_string();
    }
    out
}

pub fn redact_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                let kl = k.to_lowercase();
                if kl.contains("key")
                    || kl.contains("authorization")
                    || kl.contains("secret")
                    || kl.contains("token") && !kl.contains("token_count") && !kl.contains("tokens")
                {
                    out.insert(k.clone(), serde_json::json!("[REDACTED]"));
                } else {
                    out.insert(k.clone(), redact_json(v));
                }
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(redact_json).collect())
        }
        serde_json::Value::String(s) => serde_json::Value::String(redact_secrets(s)),
        other => other.clone(),
    }
}

pub fn classify_provider_error(err: &str) -> ExecutionErrorKind {
    let e = err.to_lowercase();
    if e.contains("401") || e.contains("403") || e.contains("missing env") || e.contains("auth") {
        ExecutionErrorKind::AuthError
    } else if e.contains("429") || e.contains("rate limit") || e.contains("rate_limit") {
        ExecutionErrorKind::RateLimited
    } else if e.contains("timeout") || e.contains("timed out") || e.contains("deadline") {
        ExecutionErrorKind::Timeout
    } else if e.contains("invalid") || e.contains("decode") || e.contains("parse") {
        ExecutionErrorKind::InvalidResponse
    } else {
        ExecutionErrorKind::ExecutionError
    }
}

pub fn content_hash(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    format!("{:x}", h.finalize())
}

pub fn json_hash(v: &serde_json::Value) -> String {
    content_hash(&serde_json::to_string(v).unwrap_or_default())
}

/// Extract tool calls from OpenAI-style raw response bodies.
pub fn extract_tool_calls(raw: &serde_json::Value) -> Vec<CapturedToolCall> {
    let mut out = Vec::new();
    let Some(arr) = raw
        .pointer("/choices/0/message/tool_calls")
        .and_then(|v| v.as_array())
    else {
        return out;
    };
    for tc in arr {
        let name = tc
            .pointer("/function/name")
            .or_else(|| tc.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if name.is_empty() {
            continue;
        }
        let mut arguments = BTreeMap::new();
        if let Some(args) = tc
            .pointer("/function/arguments")
            .or_else(|| tc.get("arguments"))
        {
            match args {
                serde_json::Value::String(s) => {
                    if let Ok(obj) = serde_json::from_str::<serde_json::Value>(s) {
                        if let Some(map) = obj.as_object() {
                            for (k, v) in map {
                                arguments.insert(k.clone(), v.clone());
                            }
                        }
                    }
                }
                serde_json::Value::Object(map) => {
                    for (k, v) in map {
                        arguments.insert(k.clone(), v.clone());
                    }
                }
                _ => {}
            }
        }
        out.push(CapturedToolCall { name, arguments });
    }
    out
}

/// Rough cost estimate from token count (USD). Marked as estimate in meta.
pub fn estimate_cost_usd(adapter: &str, total_tokens: usize) -> Option<f64> {
    if total_tokens == 0 {
        return None;
    }
    // Deterministic placeholder rates — clearly estimates only.
    let per_1k = match adapter {
        "openai" => 0.002,
        "anthropic" => 0.003,
        "google" => 0.001,
        _ => 0.002,
    };
    Some((total_tokens as f64 / 1000.0) * per_1k)
}

pub fn behaviour_from_response(
    prompt_id: &str,
    resp: &ModelResponse,
    meta: ExecutionMeta,
    adapter_name: &str,
) -> CapturedBehaviour {
    let tool_calls = extract_tool_calls(&resp.raw);
    let raw = redact_json(&resp.raw);
    CapturedBehaviour {
        prompt_id: prompt_id.to_string(),
        content: resp.content.clone(),
        latency_ms: resp.latency_ms,
        cost_usd: estimate_cost_usd(adapter_name, resp.token_count),
        tool_calls,
        finish_reason: Some(format!("{:?}", resp.finish_reason).to_lowercase()),
        fingerprint: None,
        token_count: Some(resp.token_count),
        raw: Some(raw),
        execution_error: if matches!(resp.finish_reason, FinishReason::Error) {
            Some(ExecutionErrorKind::ExecutionError)
        } else {
            None
        },
        execution_error_message: None,
        execution_meta: meta,
    }
}

pub fn behaviour_from_error(
    prompt_id: &str,
    err: &str,
    mut meta: ExecutionMeta,
) -> CapturedBehaviour {
    let kind = classify_provider_error(err);
    meta.executed_at = Some(chrono::Utc::now().to_rfc3339());
    CapturedBehaviour {
        prompt_id: prompt_id.to_string(),
        content: String::new(),
        latency_ms: 0,
        cost_usd: None,
        tool_calls: vec![],
        finish_reason: Some("error".into()),
        fingerprint: None,
        token_count: None,
        raw: None,
        execution_error: Some(kind),
        execution_error_message: Some(redact_secrets(err)),
        execution_meta: meta,
    }
}

/// Build one Probe per unique contract prompt (merge tools/schemas from items).
pub fn scenarios_from_contract(
    contract: &ApplicationContract,
) -> Vec<(String, Probe, Vec<String>)> {
    let mut by_prompt: BTreeMap<String, (Probe, Vec<String>)> = BTreeMap::new();
    for item in &contract.items {
        let entry = by_prompt.entry(item.prompt.clone()).or_insert_with(|| {
            (
                Probe {
                    id: Uuid::new_v4(),
                    name: item.prompt.clone(),
                    category: category_for_item(item),
                    prompt: item
                        .user_prompt
                        .clone()
                        .unwrap_or_else(|| item.prompt.clone()),
                    system_prompt: item.system_prompt.clone(),
                    known_answer: None,
                    expected_schema: None,
                    instructions: vec![],
                    tags: item.tags.clone(),
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
                },
                vec![],
            )
        });
        entry.1.push(item.id.clone());
        if entry.0.system_prompt.is_none() {
            entry.0.system_prompt = item.system_prompt.clone();
        }
        if entry.0.prompt.is_empty() || entry.0.prompt == item.prompt {
            if let Some(u) = &item.user_prompt {
                entry.0.prompt = u.clone();
            }
        }
        match &item.expectation {
            ContractExpectation::JsonSchema { schema } => {
                entry.0.expected_schema = Some(schema.clone());
            }
            ContractExpectation::ToolRequired { tool_name }
            | ContractExpectation::ToolArgument { tool_name, .. } => {
                let tools = entry.0.tools.get_or_insert_with(|| serde_json::json!([]));
                if let Some(arr) = tools.as_array_mut() {
                    let exists = arr.iter().any(|t| {
                        t.pointer("/function/name")
                            .or_else(|| t.get("name"))
                            .and_then(|v| v.as_str())
                            == Some(tool_name.as_str())
                    });
                    if !exists {
                        arr.push(serde_json::json!({
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "description": format!("Contract tool {tool_name}"),
                                "parameters": { "type": "object", "properties": {} }
                            }
                        }));
                    }
                }
            }
            _ => {}
        }
    }
    by_prompt
        .into_iter()
        .map(|(pid, (probe, ids))| (pid, probe, ids))
        .collect()
}

fn category_for_item(item: &ContractItem) -> ProbeCategory {
    match item.kind {
        ContractItemKind::RequiredClaim | ContractItemKind::ForbiddenClaim => {
            ProbeCategory::Factual
        }
        ContractItemKind::StructuredOutput => ProbeCategory::Schema,
        ContractItemKind::ToolCall | ContractItemKind::ToolArgument => ProbeCategory::Schema,
        ContractItemKind::Refusal => ProbeCategory::Refusal,
        ContractItemKind::Instruction => ProbeCategory::Instruction,
        ContractItemKind::Presentation | ContractItemKind::OutputFormat => {
            ProbeCategory::Morphology
        }
        _ => ProbeCategory::Semantic,
    }
}

/// Capability preflight: returns synthetic Fail results for unsupported required items.
pub fn capability_failures(
    contract: &ApplicationContract,
    caps: &ProviderCapabilities,
) -> Vec<ContractItemResult> {
    let mut out = Vec::new();
    for item in &contract.items {
        let (unsupported, reason) = match &item.expectation {
            ContractExpectation::ToolRequired { .. } | ContractExpectation::ToolArgument { .. }
                if !caps.tools =>
            {
                (
                    true,
                    "candidate model/provider does not support required tool calling",
                )
            }
            ContractExpectation::JsonSchema { .. } if !caps.structured_output => (
                true,
                "candidate model/provider does not support required structured output",
            ),
            _ => (false, ""),
        };
        if unsupported {
            out.push(ContractItemResult {
                item_id: item.id.clone(),
                kind: item.kind.clone(),
                severity: item.severity,
                outcome: ItemOutcome::Fail,
                reason: reason.into(),
                regression_type: Some("capability_mismatch".into()),
                baseline_evidence: vec![],
                candidate_evidence: vec![EvidenceSnippet {
                    label: "Capability".into(),
                    text: reason.into(),
                }],
                comparator: Some("capability".into()),
            });
        }
    }
    out
}

pub struct LiveCaptureConfig {
    pub concurrency: usize,
    pub retry_attempts: usize,
    pub retry_delay_ms: u64,
    pub temperature: f64,
    pub max_tokens: Option<usize>,
    pub arsenic_version: String,
    pub source: CaptureSource,
}

impl Default for LiveCaptureConfig {
    fn default() -> Self {
        Self {
            concurrency: 4,
            retry_attempts: 3,
            retry_delay_ms: 1000,
            temperature: 0.0,
            max_tokens: None,
            arsenic_version: env!("CARGO_PKG_VERSION").to_string(),
            source: CaptureSource::Live,
        }
    }
}

/// Execute contract scenarios via an existing [`ModelAdapter`].
pub async fn capture_contract_live(
    contract: &ApplicationContract,
    adapter: Arc<dyn ModelAdapter>,
    model_spec: &str,
    cfg: &LiveCaptureConfig,
) -> BTreeMap<String, CapturedBehaviour> {
    use futures::stream::{self, StreamExt};

    let scenarios = scenarios_from_contract(contract);
    let concurrency = cfg.concurrency.max(1);
    let adapter_name = adapter.adapter_name().to_string();
    let endpoint = adapter.endpoint().to_string();
    let model_id = adapter.model_id().to_string();

    let results = stream::iter(scenarios)
        .map(|(prompt_id, probe, _item_ids)| {
            let adapter = Arc::clone(&adapter);
            let adapter_name = adapter_name.clone();
            let endpoint = endpoint.clone();
            let model_id = model_id.clone();
            let model_spec = model_spec.to_string();
            let cfg_retries = cfg.retry_attempts.max(1);
            let cfg_delay = cfg.retry_delay_ms;
            let version = cfg.arsenic_version.clone();
            let source = cfg.source.clone();
            let temperature = cfg.temperature;
            let max_tokens = cfg.max_tokens;
            async move {
                let meta = ExecutionMeta {
                    provider: Some(adapter_name.clone()),
                    requested_model: Some(model_spec.clone()),
                    returned_model: Some(model_id.clone()),
                    endpoint: Some(endpoint),
                    temperature: Some(temperature),
                    max_tokens,
                    seed: None,
                    system_prompt_hash: probe.system_prompt.as_ref().map(|s| content_hash(s)),
                    tools_hash: probe.tools.as_ref().map(json_hash),
                    arsenic_version: Some(version),
                    executed_at: Some(chrono::Utc::now().to_rfc3339()),
                    source,
                };
                let mut last_err = None;
                for attempt in 0..cfg_retries {
                    match adapter.complete(&probe).await {
                        Ok(resp) => {
                            return (
                                prompt_id.clone(),
                                behaviour_from_response(&prompt_id, &resp, meta, &adapter_name),
                            );
                        }
                        Err(e) => {
                            last_err = Some(e.to_string());
                            let kind = classify_provider_error(&e.to_string());
                            if matches!(
                                kind,
                                ExecutionErrorKind::AuthError | ExecutionErrorKind::InvalidResponse
                            ) {
                                break;
                            }
                            if attempt + 1 < cfg_retries {
                                tokio::time::sleep(std::time::Duration::from_millis(
                                    cfg_delay.saturating_mul(1u64 << attempt.min(4)),
                                ))
                                .await;
                            }
                        }
                    }
                }
                (
                    prompt_id.clone(),
                    behaviour_from_error(
                        &prompt_id,
                        &last_err.unwrap_or_else(|| "unknown error".into()),
                        meta,
                    ),
                )
            }
        })
        .buffer_unordered(concurrency)
        .collect::<Vec<_>>()
        .await;

    results.into_iter().collect()
}

/// Fingerprint of inputs that force a candidate re-run under `--changed`.
#[allow(clippy::too_many_arguments)]
pub fn candidate_input_fingerprint(
    contract_hash: &str,
    baseline_hash: &str,
    model: &str,
    prompt_hashes: &BTreeMap<String, String>,
    tools_hash: Option<&str>,
    temperature: f64,
    max_tokens: Option<usize>,
    endpoint: Option<&str>,
    thresholds_hash: &str,
) -> String {
    let payload = serde_json::json!({
        "contract_hash": contract_hash,
        "baseline_hash": baseline_hash,
        "model": model,
        "prompt_hashes": prompt_hashes,
        "tools_hash": tools_hash,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "endpoint": endpoint,
        "thresholds_hash": thresholds_hash,
    });
    json_hash(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_api_keys() {
        let s = redact_secrets("Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz");
        assert!(s.contains("[REDACTED]"));
        assert!(!s.contains("sk-abcdefghijklmnop"));
    }

    #[test]
    fn classify_auth_and_rate() {
        assert_eq!(
            classify_provider_error("missing env OPENAI_API_KEY"),
            ExecutionErrorKind::AuthError
        );
        assert_eq!(
            classify_provider_error("OpenAI error 429: rate limit"),
            ExecutionErrorKind::RateLimited
        );
        assert_eq!(
            classify_provider_error("timeout waiting for response"),
            ExecutionErrorKind::Timeout
        );
    }

    #[test]
    fn extracts_openai_tool_calls() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "lookup_order",
                            "arguments": "{\"order_id\":\"ORD-1\"}"
                        }
                    }]
                }
            }]
        });
        let calls = extract_tool_calls(&raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup_order");
        assert_eq!(
            calls[0].arguments.get("order_id"),
            Some(&serde_json::json!("ORD-1"))
        );
    }
}
