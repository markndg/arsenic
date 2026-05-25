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

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_KEY_ENV: &str = "ARSENIC_TEST_KEY";

    fn ensure_test_key() {
        // SAFETY: writing a process-wide env var. Tests within one binary run
        // single-threaded by default for the registry module (no other tests
        // here mutate this var concurrently).
        unsafe { std::env::set_var(TEST_KEY_ENV, "test-key") };
    }

    fn spec_for(adapter_type: &str) -> AdapterSpec {
        ensure_test_key();
        AdapterSpec {
            adapter_type: adapter_type.into(),
            endpoint: None,
            api_key_env: TEST_KEY_ENV.into(),
            model_id: "test-model".into(),
            temperature: Some(0.0),
            max_tokens: None,
            timeout_secs: Some(30),
        }
    }

    fn assert_built(spec: &AdapterSpec, expected_name: &str) {
        match build_adapter(spec) {
            Ok(a) => assert_eq!(a.adapter_name(), expected_name),
            Err(e) => panic!("expected {expected_name} build to succeed, got: {e:?}"),
        }
    }

    fn assert_build_fails_with(spec: &AdapterSpec, substring: &str) {
        match build_adapter(spec) {
            Ok(_) => panic!("expected build_adapter to fail for spec {:?}", spec),
            Err(e) => {
                let msg = format!("{e:?}");
                assert!(
                    msg.to_lowercase().contains(&substring.to_lowercase()),
                    "expected error to mention `{substring}`, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn dispatches_openai_to_openai_adapter() {
        assert_built(&spec_for("openai"), "openai");
    }

    #[test]
    fn ollama_alias_uses_openai_adapter() {
        // The "ollama" string is sugar for the OpenAI-compatible adapter so that
        // local Ollama endpoints can be addressed naturally as "ollama:llama3:8b".
        assert_built(&spec_for("ollama"), "openai");
    }

    #[test]
    fn dispatches_anthropic() {
        assert_built(&spec_for("anthropic"), "anthropic");
    }

    #[test]
    fn dispatches_google() {
        assert_built(&spec_for("google"), "google");
    }

    #[test]
    fn unknown_adapter_type_errors_clearly() {
        let mut spec = spec_for("openai");
        spec.adapter_type = "made-up".into();
        assert_build_fails_with(&spec, "made-up");
        assert_build_fails_with(&spec, "Unknown");
    }

    #[test]
    fn missing_api_key_env_var_is_an_error() {
        let mut spec = spec_for("openai");
        spec.api_key_env = "ARSENIC_DEFINITELY_NOT_SET_XYZ_2026".into();
        // SAFETY: see ensure_test_key().
        unsafe { std::env::remove_var(&spec.api_key_env) };
        assert_build_fails_with(&spec, "env");
    }
}
