use std::time::{Duration, Instant};

use arsenic_core::{FinishReason, ModelAdapter, ModelResponse, Probe};
use async_trait::async_trait;
use serde_json::json;

const DEFAULT_MAX_TOKENS: usize = 2048;

pub struct OpenAIAdapter {
    pub client: reqwest::Client,
    pub endpoint: String,
    pub api_key: String,
    pub model_id: String,
    pub temperature: f64,
    pub max_tokens: usize,
    pub timeout_secs: u64,
}

impl OpenAIAdapter {
    pub fn from_spec(spec: &super::AdapterSpec) -> anyhow::Result<Self> {
        let api_key = std::env::var(&spec.api_key_env)
            .map_err(|_| anyhow::anyhow!("missing env {}", spec.api_key_env))?;
        let endpoint = spec
            .endpoint
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(spec.timeout_secs.unwrap_or(30)))
            .build()?;
        Ok(Self {
            client,
            endpoint,
            api_key,
            model_id: spec.model_id.clone(),
            temperature: spec.temperature.unwrap_or(0.0),
            max_tokens: spec.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            timeout_secs: spec.timeout_secs.unwrap_or(30),
        })
    }
}

#[async_trait]
impl ModelAdapter for OpenAIAdapter {
    async fn complete(&self, probe: &Probe) -> anyhow::Result<ModelResponse> {
        let url = format!(
            "{}/chat/completions",
            self.endpoint.trim_end_matches('/')
        );
        let mut messages = Vec::new();
        if let Some(sys) = &probe.system_prompt {
            messages.push(json!({"role":"system","content":sys}));
        }
        messages.push(json!({"role":"user","content":probe.prompt}));
        let body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        });
        let start = Instant::now();
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;
        let latency_ms = start.elapsed().as_millis() as u64;
        let status = resp.status();
        let raw: serde_json::Value = resp.json().await.unwrap_or(json!({}));
        if !status.is_success() {
            anyhow::bail!("OpenAI error {}: {}", status, raw);
        }
        let content = raw
            .pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let finish_raw = raw
            .pointer("/choices/0/finish_reason")
            .and_then(|v| v.as_str());
        let finish = match finish_raw {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::Length,
            Some("content_filter") => FinishReason::Refusal,
            _ => FinishReason::Unknown,
        };
        let token_count = raw
            .pointer("/usage/total_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        Ok(ModelResponse {
            probe_id: probe.id,
            model_label: String::new(),
            model_id: self.model_id.clone(),
            content,
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
        "openai"
    }

    fn endpoint(&self) -> &str {
        &self.endpoint
    }
}
