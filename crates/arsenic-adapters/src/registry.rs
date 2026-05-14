use serde::Deserialize;

use crate::{AnthropicAdapter, GoogleAdapter, OpenAIAdapter};

#[derive(Debug, Deserialize, Clone)]
pub struct AdapterSpec {
    pub adapter_type: String,
    pub endpoint: Option<String>,
    pub api_key_env: String,
    pub model_id: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<usize>,
    pub timeout_secs: Option<u64>,
}

pub fn build_adapter(spec: &AdapterSpec) -> anyhow::Result<std::sync::Arc<dyn arsenic_core::ModelAdapter>> {
    match spec.adapter_type.as_str() {
        "openai" | "ollama" => Ok(std::sync::Arc::new(OpenAIAdapter::from_spec(spec)?)),
        "anthropic" => Ok(std::sync::Arc::new(AnthropicAdapter::from_spec(spec)?)),
        "google" => Ok(std::sync::Arc::new(GoogleAdapter::from_spec(spec)?)),
        _ => Err(anyhow::anyhow!("Unknown adapter type: {}", spec.adapter_type)),
    }
}
