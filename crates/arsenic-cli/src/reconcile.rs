//! `arsenic reconcile` — single-prompt certification against a target model endpoint.

use std::path::PathBuf;

use anyhow::{bail, Context};
use arsenic_adapters::{build_adapter, AdapterSpec};
use arsenic_core::{
    build_reconcile_probe, run_reconcile, synthetic_model_response, ComparisonEngine,
    DEFAULT_MAX_STRATEGIES, ModelInfo, RiskThresholds,
};
use arsenic_report::{render_reconcile_html, render_reconcile_json};
use futures_util::try_join;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReconcileInputMode {
    /// Generate baseline and target responses from both endpoints.
    GenerateBoth,
    /// Baseline response from file; target from file or endpoint.
    FromFiles,
    /// Both responses inline (scripting / tests).
    Inline,
}

pub struct ReconcileArgs {
    pub prompt: Option<String>,
    pub prompt_file: Option<PathBuf>,
    pub v1: Option<String>,
    pub v2: String,
    pub v1_endpoint: Option<String>,
    pub v2_endpoint: String,
    pub v1_key_env: Option<String>,
    pub v2_key_env: String,
    pub v1_response: Option<PathBuf>,
    pub v2_response: Option<PathBuf>,
    pub v1_response_inline: Option<String>,
    pub v2_response_inline: Option<String>,
    pub system_prompt: Option<String>,
    pub system_prompt_file: Option<PathBuf>,
    pub max_strategies: usize,
    pub timeout_secs: u64,
    pub output: Option<PathBuf>,
    pub json: Option<PathBuf>,
    pub no_semantic: bool,
    pub v1_label: String,
    pub v2_label: String,
    pub temperature: f64,
}

pub fn validate_reconcile_args(args: &ReconcileArgs) -> anyhow::Result<ReconcileInputMode> {
    if args.prompt.is_none() && args.prompt_file.is_none() {
        bail!("provide --prompt or --prompt-file");
    }
    if args.prompt.is_some() && args.prompt_file.is_some() {
        bail!("use only one of --prompt or --prompt-file");
    }
    if args.system_prompt.is_some() && args.system_prompt_file.is_some() {
        bail!("use only one of --system-prompt or --system-prompt-file");
    }

    let has_inline = args.v1_response_inline.is_some() || args.v2_response_inline.is_some();
    let has_file = args.v1_response.is_some() || args.v2_response.is_some();
    let has_v1_gen = args.v1.is_some();

    if has_inline {
        if has_file {
            bail!("Mode 3 (inline): do not combine --v1-response / --v2-response with inline flags");
        }
        if has_v1_gen || args.v1_endpoint.is_some() || args.v1_key_env.is_some() {
            bail!("Mode 3 (inline): do not set --v1, --v1-endpoint, or --v1-key-env");
        }
        if args.v1_response_inline.is_none() || args.v2_response_inline.is_none() {
            bail!("Mode 3 (inline): require both --v1-response-inline and --v2-response-inline");
        }
        return Ok(ReconcileInputMode::Inline);
    }

    if has_file || args.v1_response.is_some() {
        if has_v1_gen || args.v1_endpoint.is_some() || args.v1_key_env.is_some() {
            bail!("Mode 2 (files): do not set --v1, --v1-endpoint, or --v1-key-env");
        }
        if args.v1_response.is_none() {
            bail!("Mode 2 (files): require --v1-response");
        }
        return Ok(ReconcileInputMode::FromFiles);
    }

    if !has_v1_gen {
        bail!("Mode 1 (generate): require --v1 with --v1-endpoint and --v1-key-env");
    }
    if args.v1_endpoint.is_none() || args.v1_key_env.is_none() {
        bail!("Mode 1 (generate): require --v1-endpoint and --v1-key-env");
    }
    Ok(ReconcileInputMode::GenerateBoth)
}

pub async fn run_reconcile_command(args: ReconcileArgs) -> anyhow::Result<()> {
    let mode = validate_reconcile_args(&args)?;

    let prompt = if let Some(p) = args.prompt {
        p
    } else {
        let path = args.prompt_file.context("--prompt-file")?;
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?
    };

    let system_prompt = match (args.system_prompt, args.system_prompt_file) {
        (Some(s), None) => Some(s),
        (None, Some(path)) => Some(
            std::fs::read_to_string(&path)
                .with_context(|| format!("read system prompt {}", path.display()))?,
        ),
        (None, None) => None,
        _ => unreachable!(),
    };

    let v2_spec = parse_target_spec(
        &args.v2,
        Some(args.v2_endpoint.clone()),
        Some(args.v2_key_env.clone()),
        args.temperature,
        args.timeout_secs,
    )?;
    let target_adapter = build_adapter(&v2_spec)?;

    let probe = build_reconcile_probe(prompt, system_prompt);
    let probe_id = probe.id;

    let (v1_model, v1_response, v2_response) = match mode {
        ReconcileInputMode::GenerateBoth => {
            let v1_spec = parse_target_spec(
                args.v1.as_ref().context("--v1")?,
                args.v1_endpoint.clone(),
                args.v1_key_env.clone(),
                args.temperature,
                args.timeout_secs,
            )?;
            let baseline_adapter = build_adapter(&v1_spec)?;
            let (r1, r2) = try_join!(
                baseline_adapter.complete(&probe),
                target_adapter.complete(&probe)
            )?;
            (
                model_info(&args.v1_label, &v1_spec),
                r1,
                r2,
            )
        }
        ReconcileInputMode::FromFiles => {
            let v1_path = args.v1_response.as_ref().context("--v1-response")?;
            let v1_text = std::fs::read_to_string(v1_path)
                .with_context(|| format!("read {}", v1_path.display()))?;
            let v2_text = if let Some(p) = &args.v2_response {
                std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?
            } else {
                target_adapter
                    .complete(&probe)
                    .await
                    .context("target model completion for initial response")?
                    .content
            };
            let v1 = synthetic_model_response(
                probe_id,
                &args.v1_label,
                "supplied",
                &v1_text,
            );
            let v2 = synthetic_model_response(
                probe_id,
                &args.v2_label,
                &v2_spec.model_id,
                &v2_text,
            );
            (
                ModelInfo {
                    label: args.v1_label.clone(),
                    model_id: "supplied".into(),
                    adapter: "file".into(),
                    endpoint: String::new(),
                },
                v1,
                v2,
            )
        }
        ReconcileInputMode::Inline => {
            let v1_text = args.v1_response_inline.context("--v1-response-inline")?;
            let v2_text = args.v2_response_inline.context("--v2-response-inline")?;
            let v1 = synthetic_model_response(probe_id, &args.v1_label, "inline", &v1_text);
            let v2 = synthetic_model_response(
                probe_id,
                &args.v2_label,
                &v2_spec.model_id,
                &v2_text,
            );
            (
                ModelInfo {
                    label: args.v1_label.clone(),
                    model_id: "inline".into(),
                    adapter: "inline".into(),
                    endpoint: String::new(),
                },
                v1,
                v2,
            )
        }
    };

    let engine = ComparisonEngine::new(
        !args.no_semantic,
        0.85,
        RiskThresholds::default(),
        false,
    );

    let max_strategies = if args.max_strategies == 0 {
        DEFAULT_MAX_STRATEGIES
    } else {
        args.max_strategies
    };

    let result = run_reconcile(
        &engine,
        probe,
        v1_response,
        v2_response,
        v1_model,
        model_info(&args.v2_label, &v2_spec),
        target_adapter.as_ref(),
        max_strategies,
    )
    .await?;

    if result.certified {
        println!(
            "Certified — validation risk {:?}",
            result.validation_risk
        );
    } else {
        println!("Manual review suggested");
    }

    if let Some(p) = args.output {
        let html = render_reconcile_html(&result)?;
        std::fs::write(&p, html).with_context(|| format!("write {}", p.display()))?;
        println!("Wrote HTML {}", p.display());
    }
    if let Some(p) = args.json {
        let j = render_reconcile_json(&result)?;
        std::fs::write(&p, j).with_context(|| format!("write {}", p.display()))?;
        println!("Wrote JSON {}", p.display());
    }

    Ok(())
}

fn model_info(label: &str, spec: &AdapterSpec) -> ModelInfo {
    ModelInfo {
        label: label.to_string(),
        model_id: spec.model_id.clone(),
        adapter: spec.adapter_type.clone(),
        endpoint: spec.endpoint.clone().unwrap_or_default(),
    }
}

fn parse_target_spec(
    model: &str,
    endpoint: Option<String>,
    api_key_env: Option<String>,
    temperature: f64,
    timeout_secs: u64,
) -> anyhow::Result<AdapterSpec> {
    let (adapter_type, model_id) = parse_model_spec(model)?;
    Ok(AdapterSpec {
        adapter_type,
        endpoint,
        api_key_env: api_key_env.context("--v2-key-env required")?,
        model_id,
        temperature: Some(temperature),
        max_tokens: None,
        timeout_secs: Some(timeout_secs),
    })
}

fn parse_model_spec(s: &str) -> anyhow::Result<(String, String)> {
    let mut parts = s.splitn(2, ':');
    let a = parts.next().context("empty model spec")?.trim();
    let m = parts.next().context("model spec must be adapter:model_id")?.trim();
    let adapter = match a.to_lowercase().as_str() {
        "ollama" => "openai".to_string(),
        other => other.to_string(),
    };
    Ok((adapter, m.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> ReconcileArgs {
        ReconcileArgs {
            prompt: Some("test".into()),
            prompt_file: None,
            v1: None,
            v2: "openai:model".into(),
            v1_endpoint: None,
            v2_endpoint: "http://localhost".into(),
            v1_key_env: None,
            v2_key_env: "KEY".into(),
            v1_response: None,
            v2_response: None,
            v1_response_inline: None,
            v2_response_inline: None,
            system_prompt: None,
            system_prompt_file: None,
            max_strategies: 5,
            timeout_secs: 30,
            output: None,
            json: None,
            no_semantic: false,
            v1_label: "baseline".into(),
            v2_label: "target".into(),
            temperature: 0.0,
        }
    }

    #[test]
    fn mode1_requires_baseline_endpoint_fields() {
        let mut a = base_args();
        a.v1 = Some("openai:m1".into());
        a.v1_endpoint = Some("http://localhost".into());
        a.v1_key_env = Some("K".into());
        assert_eq!(validate_reconcile_args(&a).unwrap(), ReconcileInputMode::GenerateBoth);
    }

    #[test]
    fn mode2_rejects_baseline_endpoint() {
        let mut a = base_args();
        a.v1_response = Some(PathBuf::from("v1.txt"));
        a.v1 = Some("openai:m1".into());
        assert!(validate_reconcile_args(&a).is_err());
    }

    #[test]
    fn mode3_requires_both_inline_responses() {
        let mut a = base_args();
        a.v1_response_inline = Some("a".into());
        assert!(validate_reconcile_args(&a).is_err());
        a.v2_response_inline = Some("b".into());
        assert_eq!(validate_reconcile_args(&a).unwrap(), ReconcileInputMode::Inline);
    }

    #[test]
    fn mode3_rejects_mixed_file_and_inline() {
        let mut a = base_args();
        a.v1_response_inline = Some("a".into());
        a.v2_response_inline = Some("b".into());
        a.v1_response = Some(PathBuf::from("v1.txt"));
        assert!(validate_reconcile_args(&a).is_err());
    }
}
