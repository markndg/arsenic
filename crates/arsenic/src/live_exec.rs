//! Shared helpers to build provider adapters for Application Contract live runs.

use anyhow::{bail, Context, Result};
use arsenic_adapters::{build_adapter, AdapterSpec};
use arsenic_core::ModelAdapter;
use std::sync::Arc;

pub fn default_key_env(adapter_type: &str) -> &'static str {
    match adapter_type {
        "openai" | "ollama" => "OPENAI_API_KEY",
        "anthropic" => "ANTHROPIC_API_KEY",
        "google" => "GOOGLE_API_KEY",
        _ => "ARSENIC_API_KEY",
    }
}

pub fn parse_provider_model(spec: &str) -> Result<(String, String)> {
    let (adapter, model) = match spec.split_once(':') {
        Some((a, m)) => (a.to_lowercase(), m.to_string()),
        None => bail!(
            "model must look like provider:model (e.g. openai:gpt-4.1-mini), got `{spec}`"
        ),
    };
    let adapter = if adapter == "ollama" {
        "openai".to_string()
    } else {
        adapter
    };
    Ok((adapter, model))
}

pub fn build_contract_adapter(
    model_spec: &str,
    endpoint: Option<String>,
    key_env: Option<String>,
    temperature: f64,
    max_tokens: Option<usize>,
    timeout_secs: u64,
) -> Result<Arc<dyn ModelAdapter>> {
    let (adapter_type, model_id) = parse_provider_model(model_spec)?;
    let api_key_env = key_env.unwrap_or_else(|| default_key_env(&adapter_type).to_string());
    if std::env::var(&api_key_env).is_err() {
        bail!(
            "missing credential environment variable `{api_key_env}` for `{model_spec}`.\n\
             Set it to your API key, or pass --key-env <VAR>.\n\
             Defaults: openai/ollama→OPENAI_API_KEY, anthropic→ANTHROPIC_API_KEY, google→GOOGLE_API_KEY."
        );
    }
    let spec = AdapterSpec {
        adapter_type,
        endpoint,
        api_key_env,
        model_id,
        temperature: Some(temperature),
        max_tokens,
        timeout_secs: Some(timeout_secs),
    };
    build_adapter(&spec).with_context(|| format!("build adapter for {model_spec}"))
}
