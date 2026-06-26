use std::time::{Duration, Instant};

use arsenic_core::{FinishReason, ModelAdapter, ModelResponse, Probe};
use async_trait::async_trait;
use serde_json::json;

const DEFAULT_MAX_TOKENS: usize = 2048;

/// OpenAI reasoning / GPT-5 chat models reject `max_tokens` and require
/// `max_completion_tokens` instead. gpt-4.x and gpt-4o still accept `max_tokens`.
fn uses_max_completion_tokens(model_id: &str) -> bool {
    let id = model_id.to_ascii_lowercase();
    id.starts_with("gpt-5") || id.starts_with("o1") || id.starts_with("o3") || id.starts_with("o4")
}

fn completion_limit_field(model_id: &str, limit: usize) -> (&'static str, usize) {
    if uses_max_completion_tokens(model_id) {
        ("max_completion_tokens", limit)
    } else {
        ("max_tokens", limit)
    }
}

fn build_chat_body(
    model_id: &str,
    messages: &[serde_json::Value],
    temperature: f64,
    max_tokens: usize,
) -> serde_json::Value {
    let (limit_key, limit) = completion_limit_field(model_id, max_tokens);
    json!({
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        limit_key: limit,
    })
}

fn is_max_tokens_unsupported(err: &serde_json::Value) -> bool {
    err.get("error")
        .and_then(|e| e.get("param"))
        .and_then(|p| p.as_str())
        == Some("max_tokens")
        || err
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .is_some_and(|m| m.contains("max_completion_tokens"))
}

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
        let url = format!("{}/chat/completions", self.endpoint.trim_end_matches('/'));
        let mut messages = Vec::new();
        if let Some(sys) = &probe.system_prompt {
            messages.push(json!({"role":"system","content":sys}));
        }
        messages.push(json!({"role":"user","content":probe.prompt}));
        let body = build_chat_body(&self.model_id, &messages, self.temperature, self.max_tokens);
        let start = Instant::now();
        let (status, raw) = self.post_chat_completion(&url, &body).await?;

        // Fallback: unknown future model sent max_tokens but API wants the other field.
        if !status.is_success()
            && is_max_tokens_unsupported(&raw)
            && body.get("max_tokens").is_some()
        {
            let fallback = json!({
                "model": self.model_id,
                "messages": messages,
                "temperature": self.temperature,
                "max_completion_tokens": self.max_tokens,
            });
            let retry_start = Instant::now();
            let (status2, raw2) = self.post_chat_completion(&url, &fallback).await?;
            if !status2.is_success() {
                anyhow::bail!("OpenAI error {}: {}", status2, raw2);
            }
            return self.parse_chat_response(probe, raw2, retry_start.elapsed().as_millis() as u64);
        }

        let latency_ms = start.elapsed().as_millis() as u64;
        if !status.is_success() {
            anyhow::bail!("OpenAI error {}: {}", status, raw);
        }
        self.parse_chat_response(probe, raw, latency_ms)
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

impl OpenAIAdapter {
    async fn post_chat_completion(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> anyhow::Result<(reqwest::StatusCode, serde_json::Value)> {
        let resp = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        let raw: serde_json::Value = resp.json().await.unwrap_or(json!({}));
        Ok((status, raw))
    }

    fn parse_chat_response(
        &self,
        probe: &Probe,
        raw: serde_json::Value,
        latency_ms: u64,
    ) -> anyhow::Result<ModelResponse> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt5_family_uses_max_completion_tokens() {
        assert!(uses_max_completion_tokens("gpt-5.4-mini"));
        assert!(uses_max_completion_tokens("gpt-5-mini"));
        assert!(uses_max_completion_tokens("gpt-5.5"));
    }

    #[test]
    fn gpt4_family_uses_max_tokens() {
        assert!(!uses_max_completion_tokens("gpt-4.1-mini"));
        assert!(!uses_max_completion_tokens("gpt-4o-mini"));
    }

    #[test]
    fn o_series_uses_max_completion_tokens() {
        assert!(uses_max_completion_tokens("o1"));
        assert!(uses_max_completion_tokens("o3-mini"));
        assert!(uses_max_completion_tokens("o4-mini"));
    }

    #[test]
    fn build_chat_body_selects_correct_limit_key() {
        let msgs = vec![json!({"role":"user","content":"hi"})];
        let legacy = build_chat_body("gpt-4.1-mini", &msgs, 0.0, 512);
        assert!(legacy.get("max_tokens").is_some());
        assert!(legacy.get("max_completion_tokens").is_none());

        let modern = build_chat_body("gpt-5.4-mini", &msgs, 0.0, 512);
        assert!(modern.get("max_completion_tokens").is_some());
        assert!(modern.get("max_tokens").is_none());
    }

    #[test]
    fn detects_max_tokens_unsupported_error_shape() {
        let err = json!({
            "error": {
                "param": "max_tokens",
                "message": "Use max_completion_tokens instead"
            }
        });
        assert!(is_max_tokens_unsupported(&err));
    }
}
