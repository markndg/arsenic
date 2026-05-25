//! `arsenic baseline` subcommand implementations.
//!
//! Captures, inspects, verifies, freezes, and diffs cached v1 response sets so
//! later `compare` runs can replay them without re-billing the live API.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use arsenic_adapters::{build_adapter, AdapterSpec, BaselineIdentity, CacheMode, CachingAdapter};
use arsenic_core::{
    cache::{
        corpus_fingerprint, BaselineCache, BaselineManifest, BaselineModel, BaselineProbeEntry,
    },
    ModelAdapter,
};
use chrono::Utc;
use colored::Colorize;
use futures::stream::{FuturesUnordered, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use tokio::sync::Semaphore;

const DEFAULT_CACHE_SUBDIR: &str = ".arsenic/baselines";

/// Resolve where baselines live: explicit `--cache-dir` wins, else walk up
/// from the current dir looking for `.arsenic/baselines/`, else fall back
/// to `<cwd>/.arsenic/baselines/`.
pub fn resolve_cache_dir(cli: Option<&PathBuf>) -> PathBuf {
    if let Some(p) = cli {
        return p.clone();
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut walk = cwd.clone();
    loop {
        let candidate = walk.join(DEFAULT_CACHE_SUBDIR);
        if candidate.exists() {
            return candidate;
        }
        if !walk.pop() {
            break;
        }
    }
    cwd.join(DEFAULT_CACHE_SUBDIR)
}

#[allow(clippy::too_many_arguments)]
pub struct CreateArgs {
    pub name: String,
    pub model: String,
    pub endpoint: Option<String>,
    pub key_env: String,
    pub standard_suite: Option<String>,
    pub user_corpus: Option<PathBuf>,
    pub user_corpus_only: bool,
    pub suite_path: Option<PathBuf>,
    pub consistency_runs: usize,
    pub timeout_secs: u64,
    pub concurrency: usize,
    pub retry_attempts: usize,
    pub retry_delay_ms: u64,
    pub temperature: f64,
    pub cache_dir: Option<PathBuf>,
    pub notes: Option<String>,
    pub force: bool,
}

pub async fn run_create(args: CreateArgs) -> anyhow::Result<()> {
    let cache_dir = resolve_cache_dir(args.cache_dir.as_ref());
    let cache_root = cache_dir.join(&args.name);
    let cache = BaselineCache::new(cache_root.clone());

    if cache.exists() && !args.force {
        anyhow::bail!(
            "baseline {} already exists at {}. Use --force to overwrite or pick a new name.",
            args.name,
            cache_root.display()
        );
    }
    if cache.exists() && args.force {
        eprintln!(
            "{} overwriting existing baseline at {}",
            "warn:".yellow().bold(),
            cache_root.display()
        );
        let _ = std::fs::remove_dir_all(&cache_root);
    }

    let (adapter_type, model_id) = parse_model_spec(&args.model)?;
    let spec = AdapterSpec {
        adapter_type: adapter_type.clone(),
        endpoint: args.endpoint.clone(),
        api_key_env: args.key_env.clone(),
        model_id: model_id.clone(),
        temperature: Some(args.temperature),
        max_tokens: None,
        timeout_secs: Some(args.timeout_secs),
    };
    let live = build_adapter(&spec)?;
    let resolved_endpoint = live.endpoint().to_string();

    let suite_dir = args.suite_path.unwrap_or_else(crate::default_suite_path);
    let probes = crate::load_probes_for_suite(
        &suite_dir,
        args.standard_suite.as_deref().unwrap_or("full"),
        args.user_corpus.as_ref(),
        args.user_corpus_only,
    )?;
    if probes.is_empty() {
        anyhow::bail!(
            "no probes selected; check --standard-suite, --user-corpus, and --user-corpus-only"
        );
    }
    let fingerprint = corpus_fingerprint(&probes);

    let initial_manifest = BaselineManifest {
        name: args.name.clone(),
        created_at: Utc::now(),
        arsenic_version: env!("CARGO_PKG_VERSION").into(),
        model: BaselineModel {
            adapter_type: adapter_type.clone(),
            endpoint: resolved_endpoint.clone(),
            model_id: model_id.clone(),
            temperature: args.temperature,
            max_tokens: None,
        },
        corpus_fingerprint: fingerprint.clone(),
        probes: Vec::new(),
        consistency_runs: args.consistency_runs.max(1),
        locked: false,
        notes: args.notes.clone(),
        created_by: std::env::var("USER").ok(),
    };
    cache.write_manifest(&initial_manifest)?;

    let cache_arc = Arc::new(cache.clone());
    let identity = BaselineIdentity {
        adapter_type,
        endpoint: resolved_endpoint,
        model_id: model_id.clone(),
        temperature: args.temperature,
        max_tokens: None,
    };
    let wrap = Arc::new(CachingAdapter::new(
        Some(live),
        Arc::clone(&cache_arc),
        CacheMode::WriteOnly,
        identity.clone(),
    ));

    let pb = ProgressBar::new((probes.len() * args.consistency_runs.max(1)) as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{bar:40.cyan/blue} {pos}/{len} {msg} (elapsed {elapsed})",
        )
        .unwrap(),
    );
    pb.set_message("capturing baseline…");

    let sem = Arc::new(Semaphore::new(args.concurrency.max(1)));
    let mut futs = FuturesUnordered::new();
    for probe in &probes {
        for run_idx in 0..args.consistency_runs.max(1) {
            let permit = Arc::clone(&sem);
            let wrap = Arc::clone(&wrap);
            let pb = pb.clone();
            let attempts = args.retry_attempts;
            let delay = args.retry_delay_ms;
            let probe_cloned = probe.clone();
            futs.push(tokio::spawn(async move {
                let _g = permit.acquire().await.expect("semaphore");
                let res = complete_with_retry(&*wrap, &probe_cloned, attempts, delay).await;
                pb.inc(1);
                (probe_cloned.name.clone(), run_idx, res)
            }));
        }
    }

    let mut errors: Vec<(String, usize, String)> = Vec::new();
    while let Some(joined) = futs.next().await {
        match joined {
            Ok((name, run_idx, Err(e))) => errors.push((name, run_idx, format!("{e:?}"))),
            Ok((_, _, Ok(_))) => {}
            Err(e) => errors.push(("<join>".into(), 0, format!("{e:?}"))),
        }
    }
    pb.finish_with_message("capture complete");

    // Update manifest with the probes we successfully captured.
    let mut entries: Vec<BaselineProbeEntry> = Vec::with_capacity(probes.len());
    for probe in &probes {
        let key = arsenic_core::cache::CacheKey::new(
            &identity.adapter_type,
            &identity.endpoint,
            &identity.model_id,
            identity.temperature,
            identity.max_tokens,
            probe,
        );
        let key_hash = key.hash();
        if cache.read_one(&key_hash)?.is_some() {
            entries.push(BaselineProbeEntry {
                name: probe.name.clone(),
                key_hash,
            });
        }
    }
    let mut manifest = initial_manifest;
    manifest.probes = entries;
    cache.write_manifest(&manifest)?;

    println!(
        "{} {} ({} probes, {} consistency runs each)",
        "Captured baseline".green().bold(),
        args.name.bold(),
        manifest.probes.len(),
        manifest.consistency_runs
    );
    println!("  location: {}", cache_root.display());
    println!("  corpus fingerprint: {fingerprint}");
    println!(
        "  {} editing prompts, system prompts, or sampling settings invalidates this cache. Editing analysis rules (known_answer, instructions, custom_assertions, tags) re-grades the existing capture without new API calls.",
        "note:".dimmed()
    );
    if !errors.is_empty() {
        println!(
            "  {} {} probe(s) had errors and were not cached:",
            "warn:".yellow().bold(),
            errors.len()
        );
        for (n, run_idx, e) in errors.iter().take(10) {
            println!("    - {n} (run {run_idx}): {e}");
        }
        if errors.len() > 10 {
            println!("    … and {} more", errors.len() - 10);
        }
    }
    Ok(())
}

async fn complete_with_retry(
    adapter: &dyn ModelAdapter,
    probe: &arsenic_core::Probe,
    attempts: usize,
    delay_ms: u64,
) -> anyhow::Result<arsenic_core::ModelResponse> {
    let max = attempts.max(1);
    let mut last: Option<anyhow::Error> = None;
    for attempt in 0..max {
        match adapter.complete(probe).await {
            Ok(r) => return Ok(r),
            Err(e) => {
                last = Some(e);
                if attempt + 1 < max {
                    let backoff = delay_ms.saturating_mul(1u64 << attempt.min(6));
                    tokio::time::sleep(Duration::from_millis(backoff)).await;
                }
            }
        }
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("unknown completion failure")))
}

pub fn run_list(cache_dir: Option<&PathBuf>) -> anyhow::Result<()> {
    let root = resolve_cache_dir(cache_dir);
    let entries = BaselineCache::list_baselines(&root)?;
    if entries.is_empty() {
        println!("No baselines under {}", root.display());
        return Ok(());
    }
    println!(
        "{:<30} {:<22} {:<24} {:<7} {:<8}",
        "NAME", "MODEL", "CREATED", "PROBES", "STATUS"
    );
    for path in entries {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let cache = BaselineCache::new(path.clone());
        match cache.read_manifest() {
            Ok(m) => {
                let status = if m.locked {
                    "frozen".yellow().to_string()
                } else {
                    "open".green().to_string()
                };
                println!(
                    "{:<30} {:<22} {:<24} {:<7} {:<8}",
                    truncate(&name, 30),
                    truncate(&m.model.model_id, 22),
                    m.created_at.format("%Y-%m-%d %H:%M UTC"),
                    m.probe_count(),
                    status
                );
            }
            Err(e) => {
                println!("{:<30} <unreadable manifest: {}>", truncate(&name, 30), e);
            }
        }
    }
    Ok(())
}

pub fn run_show(name: &str, cache_dir: Option<&PathBuf>) -> anyhow::Result<()> {
    let root = resolve_cache_dir(cache_dir);
    let path = root.join(name);
    let cache = BaselineCache::new(path.clone());
    if !cache.exists() {
        anyhow::bail!("baseline {} not found at {}", name, path.display());
    }
    let m = cache.read_manifest()?;
    println!("Baseline: {}", m.name.bold());
    println!("  Location: {}", path.display());
    println!("  Created: {}", m.created_at.format("%Y-%m-%d %H:%M:%S UTC"));
    println!("  arsenic version: {}", m.arsenic_version);
    println!(
        "  Model: {} ({} @ {})",
        m.model.model_id, m.model.adapter_type, m.model.endpoint
    );
    println!(
        "  Sampling: temperature={}, max_tokens={:?}",
        m.model.temperature, m.model.max_tokens
    );
    println!("  Consistency runs per probe: {}", m.consistency_runs);
    println!("  Corpus fingerprint: {}", m.corpus_fingerprint);
    println!(
        "  Status: {}",
        if m.locked {
            "FROZEN".yellow().to_string()
        } else {
            "open".green().to_string()
        }
    );
    if let Some(notes) = &m.notes {
        println!("  Notes: {notes}");
    }
    if let Some(creator) = &m.created_by {
        println!("  Created by: {creator}");
    }
    println!("  Probes ({}):", m.probes.len());
    for entry in &m.probes {
        println!("    - {} ({})", entry.name, &entry.key_hash[..16]);
    }
    Ok(())
}

pub fn run_verify(name: &str, cache_dir: Option<&PathBuf>) -> anyhow::Result<()> {
    let root = resolve_cache_dir(cache_dir);
    let path = root.join(name);
    let cache = BaselineCache::new(path.clone());
    if !cache.exists() {
        anyhow::bail!("baseline {} not found at {}", name, path.display());
    }
    let report = cache.verify()?;
    println!(
        "Verified {}: {} ok / {} total",
        name.bold(),
        report.ok,
        report.total
    );
    if !report.mismatched.is_empty() {
        println!("  {} mismatched files:", "FAIL".red().bold());
        for p in &report.mismatched {
            println!("    - {p}");
        }
    }
    if !report.unreadable.is_empty() {
        println!("  {} unreadable files:", "FAIL".red().bold());
        for p in &report.unreadable {
            println!("    - {p}");
        }
    }
    if report.is_clean() {
        println!("{}", "Baseline is clean.".green().bold());
        Ok(())
    } else {
        anyhow::bail!("verification failed");
    }
}

pub fn run_remove(name: &str, cache_dir: Option<&PathBuf>, yes: bool) -> anyhow::Result<()> {
    let root = resolve_cache_dir(cache_dir);
    let path = root.join(name);
    if !path.exists() {
        anyhow::bail!("baseline {} not found at {}", name, path.display());
    }
    let cache = BaselineCache::new(path.clone());
    if cache.is_locked() && !yes {
        anyhow::bail!(
            "baseline {} is frozen; pass --yes to confirm removal (or `arsenic baseline unfreeze {}` first)",
            name,
            name
        );
    }
    if !yes {
        eprintln!(
            "{} this will permanently delete {} (use --yes to proceed)",
            "warn:".yellow().bold(),
            path.display()
        );
        anyhow::bail!("removal not confirmed");
    }
    std::fs::remove_dir_all(&path).with_context(|| format!("remove {}", path.display()))?;
    println!("Removed baseline {} ({}).", name.bold(), path.display());
    Ok(())
}

pub fn run_freeze(name: &str, cache_dir: Option<&PathBuf>) -> anyhow::Result<()> {
    let root = resolve_cache_dir(cache_dir);
    let path = root.join(name);
    let cache = BaselineCache::new(path.clone());
    if !cache.exists() {
        anyhow::bail!("baseline {} not found at {}", name, path.display());
    }
    cache.lock()?;
    println!("Baseline {} is now {}.", name.bold(), "FROZEN".yellow().bold());
    Ok(())
}

pub fn run_unfreeze(name: &str, cache_dir: Option<&PathBuf>) -> anyhow::Result<()> {
    let root = resolve_cache_dir(cache_dir);
    let path = root.join(name);
    let cache = BaselineCache::new(path.clone());
    if !cache.exists() {
        anyhow::bail!("baseline {} not found at {}", name, path.display());
    }
    cache.unlock()?;
    println!("Baseline {} unfrozen.", name.bold());
    Ok(())
}

pub fn run_diff(a: &str, b: &str, cache_dir: Option<&PathBuf>) -> anyhow::Result<()> {
    let root = resolve_cache_dir(cache_dir);
    let cache_a = BaselineCache::new(root.join(a));
    let cache_b = BaselineCache::new(root.join(b));
    if !cache_a.exists() {
        anyhow::bail!("baseline {a} not found");
    }
    if !cache_b.exists() {
        anyhow::bail!("baseline {b} not found");
    }
    let ma = cache_a.read_manifest()?;
    let mb = cache_b.read_manifest()?;

    println!("{} vs {}", a.bold(), b.bold());
    println!("  {}: {}", a, ma.model.model_id);
    println!("  {}: {}", b, mb.model.model_id);
    println!(
        "  corpus fingerprint match: {}",
        if ma.corpus_fingerprint == mb.corpus_fingerprint {
            "yes".green().to_string()
        } else {
            "NO".red().to_string()
        }
    );

    use std::collections::HashSet;
    let set_a: HashSet<&str> = ma.probes.iter().map(|p| p.name.as_str()).collect();
    let set_b: HashSet<&str> = mb.probes.iter().map(|p| p.name.as_str()).collect();
    let only_a: Vec<&&str> = set_a.difference(&set_b).collect();
    let only_b: Vec<&&str> = set_b.difference(&set_a).collect();
    let common: Vec<&&str> = set_a.intersection(&set_b).collect();
    println!(
        "  probes: {} in both, {} only in {a}, {} only in {b}",
        common.len(),
        only_a.len(),
        only_b.len()
    );
    if !only_a.is_empty() {
        println!("  only in {a}:");
        for n in &only_a {
            println!("    - {n}");
        }
    }
    if !only_b.is_empty() {
        println!("  only in {b}:");
        for n in &only_b {
            println!("    - {n}");
        }
    }

    let mut differing = 0usize;
    for name in &common {
        let entry_a = ma.probes.iter().find(|p| &p.name.as_str() == *name).unwrap();
        let entry_b = mb.probes.iter().find(|p| &p.name.as_str() == *name).unwrap();
        let resp_a = cache_a.read_one(&entry_a.key_hash)?;
        let resp_b = cache_b.read_one(&entry_b.key_hash)?;
        let first_a = resp_a.as_ref().and_then(|r| r.runs.first()).map(|r| r.content.as_str()).unwrap_or("");
        let first_b = resp_b.as_ref().and_then(|r| r.runs.first()).map(|r| r.content.as_str()).unwrap_or("");
        if first_a != first_b {
            differing += 1;
        }
    }
    println!("  first-run content differs on {differing}/{} probes", common.len());
    Ok(())
}

pub fn run_timeline(model: Option<&str>, cache_dir: Option<&PathBuf>) -> anyhow::Result<()> {
    let root = resolve_cache_dir(cache_dir);
    let mut rows: Vec<(chrono::DateTime<chrono::Utc>, String, BaselineManifest)> = Vec::new();
    for path in BaselineCache::list_baselines(&root)? {
        let cache = BaselineCache::new(path.clone());
        let m = match cache.read_manifest() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if let Some(filter) = model {
            if !m.model.model_id.contains(filter) {
                continue;
            }
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        rows.push((m.created_at, name, m));
    }
    rows.sort_by_key(|r| r.0);
    if rows.is_empty() {
        println!("No baselines match{}.", if model.is_some() { " filter" } else { "" });
        return Ok(());
    }
    println!(
        "{:<24} {:<30} {:<22} {:<7}",
        "CREATED", "NAME", "MODEL", "PROBES"
    );
    for (ts, name, m) in rows {
        println!(
            "{:<24} {:<30} {:<22} {:<7}",
            ts.format("%Y-%m-%d %H:%M UTC"),
            truncate(&name, 30),
            truncate(&m.model.model_id, 22),
            m.probe_count()
        );
    }
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn parse_model_spec(s: &str) -> anyhow::Result<(String, String)> {
    let mut parts = s.splitn(2, ':');
    let a = parts.next().context("empty model spec")?.trim();
    let m = parts
        .next()
        .context("model spec must be adapter:model_id")?
        .trim();
    let adapter = match a.to_lowercase().as_str() {
        "ollama" => "openai".to_string(),
        other => other.to_string(),
    };
    Ok((adapter, m.to_string()))
}
