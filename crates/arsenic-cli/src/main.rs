use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use arsenic_adapters::{build_adapter, AdapterSpec};
use arsenic_core::{
    ComparisonEngine, DriftReport, ModelInfo, ProbeCategory, ProbeRunner, RiskThresholds,
};
use arsenic_probes::ProbeLoader;
use arsenic_report::ReportRenderer;
use clap::{Parser, Subcommand};
use colored::Colorize;
use indicatif::ProgressBar;
use serde::Deserialize;
use uuid::Uuid;

mod model_download;
mod mutation_validate;
mod reconcile;

#[derive(Parser)]
#[command(name = "arsenic", version, about = "ARSENIC — migration safety and behavioural drift")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run probe suite against two model endpoints
    Compare {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        v1: Option<String>,
        #[arg(long)]
        v2: Option<String>,
        #[arg(long)]
        v1_endpoint: Option<String>,
        #[arg(long)]
        v2_endpoint: Option<String>,
        #[arg(long)]
        v1_key_env: Option<String>,
        #[arg(long)]
        v2_key_env: Option<String>,
        #[arg(long)]
        standard_suite: Option<String>,
        #[arg(long)]
        user_corpus: Option<PathBuf>,
        #[arg(long)]
        suite_path: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        json: Option<PathBuf>,
        #[arg(long, default_value_t = 10)]
        concurrency: usize,
        #[arg(long, default_value_t = 3)]
        retry_attempts: usize,
        #[arg(long, default_value_t = 1000)]
        retry_delay_ms: u64,
        #[arg(long, default_value = "0.85")]
        semantic_threshold: String,
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
        #[arg(long)]
        no_semantic: bool,
        #[arg(long, default_value = "v1")]
        v1_label: String,
        #[arg(long, default_value = "v2")]
        v2_label: String,
        /// v2: completions per probe per endpoint for consistency scoring (1 disables extra runs).
        #[arg(long, default_value_t = 3)]
        consistency_runs: usize,
        /// Request timeout in seconds for model API calls (default: 30). Increase for slow local models.
        #[arg(long, default_value_t = 30)]
        timeout_secs: u64,
        /// v2: after compare, try rule-based prompt mutations against v2 and record results.
        #[arg(long)]
        mutate: bool,
    },
    /// Inspect probe suites
    Probe {
        #[command(subcommand)]
        sub: ProbeCmd,
    },
    /// Render reports from saved JSON
    Report {
        #[command(subcommand)]
        sub: ReportCmd,
    },
    /// Validate a user probe corpus directory or file
    Validate {
        path: PathBuf,
    },
    /// Model assets (placeholder for air-gapped download instructions)
    Models {
        #[command(subcommand)]
        sub: ModelsCmd,
    },
    /// Reconcile a single prompt: analyse drift and validate a prompt patch for the target model
    Reconcile {
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        #[arg(long)]
        v1: Option<String>,
        #[arg(long)]
        v2: String,
        #[arg(long)]
        v1_endpoint: Option<String>,
        #[arg(long)]
        v2_endpoint: String,
        #[arg(long)]
        v1_key_env: Option<String>,
        #[arg(long)]
        v2_key_env: String,
        #[arg(long)]
        v1_response: Option<PathBuf>,
        #[arg(long)]
        v2_response: Option<PathBuf>,
        #[arg(long)]
        v1_response_inline: Option<String>,
        #[arg(long)]
        v2_response_inline: Option<String>,
        #[arg(long)]
        system_prompt: Option<String>,
        #[arg(long)]
        system_prompt_file: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        max_strategies: usize,
        #[arg(long, default_value_t = 30)]
        timeout_secs: u64,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        json: Option<PathBuf>,
        #[arg(long)]
        no_semantic: bool,
        #[arg(long, default_value = "baseline")]
        v1_label: String,
        #[arg(long, default_value = "target")]
        v2_label: String,
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
    },
}

#[derive(Subcommand)]
enum ProbeCmd {
    List {
        #[arg(long)]
        suite_path: Option<PathBuf>,
        #[arg(long)]
        category: Option<String>,
    },
    Show {
        name: String,
        #[arg(long)]
        suite_path: Option<PathBuf>,
    },
    Validate {
        path: PathBuf,
    },
}

#[derive(Subcommand)]
enum ReportCmd {
    Render {
        input: PathBuf,
        #[arg(long)]
        format: String,
        #[arg(long)]
        output: PathBuf,
    },
    Summary {
        input: PathBuf,
        /// Print `summary` object keys and valence-related fields to stderr (for debugging consumers).
        #[arg(long)]
        debug_summary: bool,
    },
}

#[derive(Subcommand)]
enum ModelsCmd {
    Download {
        model: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Debug, Deserialize)]
struct ArsenicConfig {
    v1: AdapterConfigSection,
    v2: AdapterConfigSection,
    #[serde(default)]
    run: RunConfigSection,
    #[serde(default)]
    output: OutputConfigSection,
}

#[derive(Debug, Deserialize, Default)]
struct RunConfigSection {
    standard_suite: Option<String>,
    user_corpus: Option<PathBuf>,
    concurrency: Option<usize>,
    retry_attempts: Option<usize>,
    semantic_threshold: Option<f64>,
    consistency_runs: Option<usize>,
    timeout_secs: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
struct OutputConfigSection {
    html: Option<PathBuf>,
    json: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct AdapterConfigSection {
    adapter: String,
    endpoint: Option<String>,
    api_key_env: String,
    model_id: String,
    temperature: Option<f64>,
    max_tokens: Option<usize>,
}

fn default_suite_path() -> PathBuf {
    if let Ok(p) = std::env::var("ARSENIC_SUITE_PATH") {
        return PathBuf::from(p);
    }
    PathBuf::from("probe-suite/standard")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("ARSENIC_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Compare {
            config,
            v1,
            v2,
            v1_endpoint,
            v2_endpoint,
            v1_key_env,
            v2_key_env,
            standard_suite,
            user_corpus,
            suite_path,
            output,
            json,
            concurrency,
            retry_attempts,
            retry_delay_ms,
            semantic_threshold,
            temperature,
            no_semantic,
            v1_label,
            v2_label,
            consistency_runs,
            timeout_secs,
            mutate,
        } => {
            let mut cfg_opt: Option<ArsenicConfig> = None;
            if let Some(p) = &config {
                let text = std::fs::read_to_string(p)
                    .with_context(|| format!("read config {}", p.display()))?;
                cfg_opt = Some(toml::from_str(&text).context("parse arsenic.toml")?);
            }
            let timeout_secs_effective = cfg_opt
                .as_ref()
                .and_then(|c| c.run.timeout_secs)
                .unwrap_or(timeout_secs);

            let (spec1, spec2, suite_str, corpus, conc, sem_thr, out_html, out_json) =
                merge_compare_config(
                    cfg_opt.as_ref(),
                    v1,
                    v2,
                    v1_endpoint,
                    v2_endpoint,
                    v1_key_env,
                    v2_key_env,
                    standard_suite,
                    user_corpus,
                    concurrency,
                    &semantic_threshold,
                    output,
                    json,
                    temperature,
                    timeout_secs_effective,
                )?;

            let consistency_runs_effective = cfg_opt
                .as_ref()
                .and_then(|c| c.run.consistency_runs)
                .unwrap_or(consistency_runs)
                .max(1);

            let suite_dir = suite_path.unwrap_or_else(default_suite_path);
            let probes = load_probes_for_suite(&suite_dir, &suite_str, corpus.as_ref())?;

            if probes.is_empty() {
                anyhow::bail!("no probes selected; check --standard-suite and --user-corpus");
            }

            let a1 = build_adapter(&spec1)?;
            let a2 = build_adapter(&spec2)?;
            let v2_for_mutate = Arc::clone(&a2);

            let runner = ProbeRunner {
                v1_adapter: a1,
                v2_adapter: a2,
                v1_label: v1_label.clone(),
                v2_label: v2_label.clone(),
                concurrency: conc,
                retry_attempts: cfg_opt
                    .as_ref()
                    .and_then(|c| c.run.retry_attempts)
                    .unwrap_or(retry_attempts),
                retry_delay_ms,
                consistency_runs: consistency_runs_effective,
            };

            let pb = ProgressBar::new_spinner();
            pb.set_message("running probes…");
            let pairs = runner.run(probes).await?;
            pb.finish_and_clear();

            let pair_by_id: HashMap<_, _> = pairs.iter().map(|p| (p.probe.id, p.clone())).collect();

            let risk = RiskThresholds::default();
            let engine = ComparisonEngine::new(!no_semantic, sem_thr, risk);
            let mut report = engine.compare(
                Uuid::new_v4(),
                pairs,
                ModelInfo {
                    label: v1_label.clone(),
                    model_id: spec1.model_id.clone(),
                    adapter: spec1.adapter_type.clone(),
                    endpoint: spec1.endpoint.clone().unwrap_or_default(),
                },
                ModelInfo {
                    label: v2_label.clone(),
                    model_id: spec2.model_id.clone(),
                    adapter: spec2.adapter_type.clone(),
                    endpoint: spec2.endpoint.clone().unwrap_or_default(),
                },
            )?;
            report.sync_valence_from_probe_results();

            if mutate {
                let mutation_results = mutation_validate::validate_mutations_for_report(
                    &engine,
                    &report.probe_results,
                    &pair_by_id,
                    v2_for_mutate.as_ref(),
                )
                .await?;
                ComparisonEngine::attach_mutations(&mut report, mutation_results);
            }

            let overall = engine.compute_overall_risk(&report.probe_results);
            println!(
                "{}",
                format!("Overall risk: {:?}", overall).bright_white().bold()
            );

            let need_summary = out_html.is_none() && out_json.is_none();

            if let Some(p) = out_html {
                let html = ReportRenderer::render_html(&report)?;
                std::fs::write(&p, html).with_context(|| format!("write {}", p.display()))?;
                println!("Wrote HTML {}", p.display());
            }
            if let Some(p) = out_json {
                let j = ReportRenderer::render_json(&report)?;
                std::fs::write(&p, j).with_context(|| format!("write {}", p.display()))?;
                println!("Wrote JSON {}", p.display());
            }
            if need_summary {
                println!("{}", ReportRenderer::render_summary_line(&report));
            }
        }
        Commands::Probe { sub } => match sub {
            ProbeCmd::List {
                suite_path,
                category,
            } => {
                let dir = suite_path.unwrap_or_else(default_suite_path);
                let mut probes = if let Some(c) = category {
                    let cat = parse_category(&c).context("invalid category")?;
                    ProbeLoader::load_standard_categories(&dir, &[cat])?
                } else {
                    ProbeLoader::load_standard_suite(&dir)?
                };
                probes.sort_by(|a, b| a.name.cmp(&b.name));
                for p in probes {
                    println!("{} [{}] {:?}", p.name, format!("{:?}", p.category).dimmed(), p.tags);
                }
            }
            ProbeCmd::Show { name, suite_path } => {
                let dir = suite_path.unwrap_or_else(default_suite_path);
                let probes = ProbeLoader::load_standard_suite(&dir)?;
                let found = probes.into_iter().find(|p| p.name == name);
                if let Some(p) = found {
                    println!("{}", serde_json::to_string_pretty(&p)?);
                } else {
                    anyhow::bail!("probe not found: {name}");
                }
            }
            ProbeCmd::Validate { path } => {
                validate_corpus(&path)?;
                println!("{}", "OK".green());
            }
        },
        Commands::Report { sub } => match sub {
            ReportCmd::Render {
                input,
                format,
                output,
            } => {
                let report: DriftReport = load_report_json(&input)?;
                let bytes = match format.as_str() {
                    "html" => ReportRenderer::render_html(&report)?,
                    "md" | "markdown" => ReportRenderer::render_markdown(&report)?,
                    "json" => ReportRenderer::render_json(&report)?,
                    _ => anyhow::bail!("unknown format {format}"),
                };
                std::fs::write(&output, bytes).with_context(|| format!("write {}", output.display()))?;
                println!("Wrote {}", output.display());
            }
            ReportCmd::Summary {
                input,
                debug_summary,
            } => {
                let report: DriftReport = load_report_json(&input)?;
                let summary_json = ReportRenderer::summary_json(&report)?;
                if debug_summary {
                    if let Some(s) = summary_json.get("summary").and_then(|x| x.as_object()) {
                        let mut keys: Vec<_> = s.keys().cloned().collect();
                        keys.sort();
                        eprintln!("summary keys (sorted): {keys:?}");
                        for k in [
                            "total_probes",
                            "probe_regressions",
                            "regressions",
                            "probe_improvements",
                            "improvements",
                            "probe_neutral",
                            "neutral",
                        ] {
                            eprintln!("  {k}: {:?}", s.get(k));
                        }
                    } else {
                        eprintln!("debug_summary: no `summary` object in output");
                    }
                }
                println!("{}", serde_json::to_string_pretty(&summary_json)?);
            }
        },
        Commands::Validate { path } => {
            validate_corpus(&path)?;
            println!("{}", "OK".green());
        }
        Commands::Reconcile {
            prompt,
            prompt_file,
            v1,
            v2,
            v1_endpoint,
            v2_endpoint,
            v1_key_env,
            v2_key_env,
            v1_response,
            v2_response,
            v1_response_inline,
            v2_response_inline,
            system_prompt,
            system_prompt_file,
            max_strategies,
            timeout_secs,
            output,
            json,
            no_semantic,
            v1_label,
            v2_label,
            temperature,
        } => {
            reconcile::run_reconcile_command(reconcile::ReconcileArgs {
                prompt,
                prompt_file,
                v1,
                v2,
                v1_endpoint,
                v2_endpoint,
                v1_key_env,
                v2_key_env,
                v1_response,
                v2_response,
                v1_response_inline,
                v2_response_inline,
                system_prompt,
                system_prompt_file,
                max_strategies,
                timeout_secs,
                output,
                json,
                no_semantic,
                v1_label,
                v2_label,
                temperature,
            })
            .await?;
        }
        Commands::Models { sub } => match sub {
            ModelsCmd::Download { model, output } => {
                let out_base = output.unwrap_or_else(|| {
                    std::env::var_os("HOME")
                        .map(PathBuf::from)
                        .map(|h| h.join(".arsenic/models"))
                        .unwrap_or_else(|| PathBuf::from(".arsenic/models"))
                });
                let slug = model.trim().trim_matches('/').replace('/', "__");
                let dest = out_base.join(&slug);
                let manifest = model_download::download_hf_model_files(&model, &dest).await?;
                println!(
                    "Downloaded {} files for repo {} into {}",
                    manifest.len(),
                    model_download::resolve_hf_repo(&model),
                    dest.display()
                );
                for (name, sha) in &manifest {
                    println!("  {name}  sha256={sha}");
                }
            }
        },
    }
    Ok(())
}

fn load_report_json(path: &Path) -> anyhow::Result<DriftReport> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(serde_json::from_str(&text).context("parse drift JSON")?)
}

fn validate_corpus(path: &Path) -> anyhow::Result<()> {
    ProbeLoader::load_user_corpus(path).map(|_| ())
}

fn parse_category(s: &str) -> anyhow::Result<ProbeCategory> {
    match s.to_lowercase().as_str() {
        "morphology" => Ok(ProbeCategory::Morphology),
        "tone" => Ok(ProbeCategory::Tone),
        "factual" => Ok(ProbeCategory::Factual),
        "schema" => Ok(ProbeCategory::Schema),
        "instruction" => Ok(ProbeCategory::Instruction),
        "refusal" => Ok(ProbeCategory::Refusal),
        "semantic" => Ok(ProbeCategory::Semantic),
        _ => anyhow::bail!("unknown category {s}"),
    }
}

fn merge_compare_config(
    cfg: Option<&ArsenicConfig>,
    v1: Option<String>,
    v2: Option<String>,
    v1_endpoint: Option<String>,
    v2_endpoint: Option<String>,
    v1_key_env: Option<String>,
    v2_key_env: Option<String>,
    standard_suite: Option<String>,
    user_corpus: Option<PathBuf>,
    concurrency_cli: usize,
    semantic_threshold: &str,
    output_cli: Option<PathBuf>,
    json_cli: Option<PathBuf>,
    temperature_cli: f64,
    timeout_secs: u64,
) -> anyhow::Result<(
    AdapterSpec,
    AdapterSpec,
    String,
    Option<PathBuf>,
    usize,
    f64,
    Option<PathBuf>,
    Option<PathBuf>,
)> {
    let spec1 = parse_model_cli_or_cfg(
        cfg.map(|c| &c.v1),
        v1,
        v1_endpoint,
        v1_key_env,
        temperature_cli,
        timeout_secs,
    )?;
    let spec2 = parse_model_cli_or_cfg(
        cfg.map(|c| &c.v2),
        v2,
        v2_endpoint,
        v2_key_env,
        temperature_cli,
        timeout_secs,
    )?;
    let suite_str = standard_suite
        .or_else(|| cfg.and_then(|c| c.run.standard_suite.clone()))
        .unwrap_or_else(|| "full".to_string());
    let corpus = user_corpus.or_else(|| cfg.and_then(|c| c.run.user_corpus.clone()));
    let conc = cfg
        .and_then(|c| c.run.concurrency)
        .unwrap_or(concurrency_cli);
    let sem_thr = cfg
        .and_then(|c| c.run.semantic_threshold)
        .unwrap_or_else(|| semantic_threshold.parse().unwrap_or(0.85));
    let out_html = output_cli.or_else(|| cfg.and_then(|c| c.output.html.clone()));
    let out_json = json_cli.or_else(|| cfg.and_then(|c| c.output.json.clone()));
    Ok((spec1, spec2, suite_str, corpus, conc, sem_thr, out_html, out_json))
}

fn parse_model_cli_or_cfg(
    section: Option<&AdapterConfigSection>,
    cli: Option<String>,
    endpoint_cli: Option<String>,
    key_env_cli: Option<String>,
    temperature_cli: f64,
    timeout_secs: u64,
) -> anyhow::Result<AdapterSpec> {
    if let Some(s) = cli {
        let (adapter_type, model_id) = parse_model_spec(&s)?;
        return Ok(AdapterSpec {
            adapter_type,
            endpoint: endpoint_cli,
            api_key_env: key_env_cli.context("--v1-key-env / --v2-key-env required with --v1/--v2")?,
            model_id,
            temperature: Some(temperature_cli),
            max_tokens: None,
            timeout_secs: Some(timeout_secs),
        });
    }
    let sec = section.context("provide --v1/--v2 or a config file with [v1]/[v2]")?;
    Ok(AdapterSpec {
        adapter_type: sec.adapter.clone(),
        endpoint: sec.endpoint.clone().or(endpoint_cli),
        api_key_env: sec.api_key_env.clone(),
        model_id: sec.model_id.clone(),
        temperature: sec.temperature.or(Some(temperature_cli)),
        max_tokens: sec.max_tokens,
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

fn load_probes_for_suite(
    suite_dir: &Path,
    suite_sel: &str,
    user_corpus: Option<&PathBuf>,
) -> anyhow::Result<Vec<arsenic_core::Probe>> {
    let mut out = Vec::new();
    if suite_sel.trim().is_empty() || suite_sel == "none" {
        // user only
    } else if suite_sel == "full" {
        out.extend(ProbeLoader::load_standard_suite(suite_dir)?);
    } else {
        let cats: Vec<ProbeCategory> = suite_sel
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| parse_category(s))
            .collect::<anyhow::Result<_>>()?;
        out.extend(ProbeLoader::load_standard_categories(suite_dir, &cats)?);
    }
    if let Some(p) = user_corpus {
        out.extend(ProbeLoader::load_user_corpus(p)?);
    }
    Ok(out)
}
