use std::time::{Duration, Instant};

use arsenic_core::{FinishReason, ModelAdapter, ModelResponse, Probe};
use async_trait::async_trait;
use serde_json::json;

const DEFAULT_MAX_TOKENS: usize = 2048;

pub struct AnthropicAdapter {
    pub client: reqwest::Client,
    pub api_key: String,
    pub model_id: String,
    pub temperature: f64,
    pub max_tokens: usize,
    pub timeout_secs: u64,
}

impl AnthropicAdapter {
    pub fn from_spec(spec: &super::AdapterSpec) -> anyhow::Result<Self> {
        let api_key = std::env::var(&spec.api_key_env)
            .map_err(|_| anyhow::anyhow!("missing env {}", spec.api_key_env))?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(spec.timeout_secs.unwrap_or(30)))
            .build()?;
        Ok(Self {
            client,
            api_key,
            model_id: spec.model_id.clone(),
            temperature: spec.temperature.unwrap_or(0.0),
            max_tokens: spec.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            timeout_secs: spec.timeout_secs.unwrap_or(30),
        })
    }

    fn endpoint() -> &'static str {
        "https://api.anthropic.com/v1/messages"
    }
}

#[async_trait]
impl ModelAdapter for AnthropicAdapter {
    async fn complete(&self, probe: &Probe) -> anyhow::Result<ModelResponse> {
        let mut body = json!({
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role":"user","content": probe.prompt}],
        });
        if let Some(sys) = &probe.system_prompt {
            body.as_object_mut()
                .expect("object")
                .insert("system".into(), json!(sys));
        }
        let start = Instant::now();
        let resp = self
            .client
            .post(Self::endpoint())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;
        let latency_ms = start.elapsed().as_millis() as u64;
        let status = resp.status();
        let raw: serde_json::Value = resp.json().await.unwrap_or(json!({}));
        if !status.is_success() {
            anyhow::bail!("Anthropic error {}: {}", status, raw);
        }
        let mut text = String::new();
        if let Some(arr) = raw.get("content").and_then(|c| c.as_array()) {
            for block in arr {
                if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                    if let Some(t) = block.get("text").and_then(|x| x.as_str()) {
                        text.push_str(t);
                    }
                }
            }
        }
        let stop = raw
            .get("stop_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let finish = match stop {
            "end_turn" => FinishReason::Stop,
            "max_tokens" => FinishReason::Length,
            "refusal" => FinishReason::Refusal,
            _ => FinishReason::Unknown,
        };
        let in_tok = raw
            .pointer("/usage/input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let out_tok = raw
            .pointer("/usage/output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let token_count = (in_tok + out_tok) as usize;
        Ok(ModelResponse {
            probe_id: probe.id,
            model_label: String::new(),
            model_id: self.model_id.clone(),
            content: text,
            token_count,
            latency_ms,
            finish_reason: finish,
            timestamp: chrono::Utc::now(),
            raw,
        })
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn adapter_name(&self) -> &str {
        "anthropic"
    }

    fn endpoint(&self) -> &str {
        Self::endpoint()
    }
}
