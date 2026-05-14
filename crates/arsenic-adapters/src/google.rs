use std::time::{Duration, Instant};

use arsenic_core::{FinishReason, ModelAdapter, ModelResponse, Probe};
use async_trait::async_trait;
use serde_json::json;

const DEFAULT_MAX_TOKENS: usize = 2048;

pub struct GoogleAdapter {
    pub client: reqwest::Client,
    pub api_key: String,
    pub model_id: String,
    pub temperature: f64,
    pub max_tokens: usize,
    pub timeout_secs: u64,
}

impl GoogleAdapter {
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

    fn url(&self) -> String {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model_id, self.api_key
        )
    }
}

#[async_trait]
impl ModelAdapter for GoogleAdapter {
    async fn complete(&self, probe: &Probe) -> anyhow::Result<ModelResponse> {
        let user_text = if let Some(sys) = &probe.system_prompt {
            format!("System instructions:\n{sys}\n\nUser:\n{}", probe.prompt)
        } else {
            probe.prompt.clone()
        };
        let body = json!({
            "contents": [{"role":"user","parts": [{"text": user_text}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            }
        });
        let start = Instant::now();
        let resp = self.client.post(self.url()).json(&body).send().await?;
        let latency_ms = start.elapsed().as_millis() as u64;
        let status = resp.status();
        let raw: serde_json::Value = resp.json().await.unwrap_or(json!({}));
        if !status.is_success() {
            anyhow::bail!("Google error {}: {}", status, raw);
        }
        let mut text = String::new();
        if let Some(parts) = raw
            .pointer("/candidates/0/content/parts")
            .and_then(|p| p.as_array())
        {
            for p in parts {
                if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                    text.push_str(t);
                }
            }
        }
        let finish = raw
            .pointer("/candidates/0/finishReason")
            .and_then(|v| v.as_str())
            .map(|r| match r {
                "STOP" => FinishReason::Stop,
                "MAX_TOKENS" => FinishReason::Length,
                "SAFETY" | "RECITATION" | "OTHER" => FinishReason::Refusal,
                _ => FinishReason::Unknown,
            })
            .unwrap_or(FinishReason::Unknown);
        let prompt_tok = raw
            .pointer("/usageMetadata/promptTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cand_tok = raw
            .pointer("/usageMetadata/candidatesTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let token_count = (prompt_tok + cand_tok) as usize;
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
        "google"
    }

    fn endpoint(&self) -> &str {
        "https://generativelanguage.googleapis.com"
    }
}
