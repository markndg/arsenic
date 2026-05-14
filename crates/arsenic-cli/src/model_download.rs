//! Download public model files from Hugging Face `resolve/main` (no token).

use std::path::{Path, PathBuf};

use anyhow::Context;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::StatusCode;
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub fn resolve_hf_repo(model: &str) -> String {
    let m = model.trim();
    if m.contains('/') {
        m.to_string()
    } else {
        format!("BAAI/{m}")
    }
}

/// Returns `(relative_path, sha256_hex)` for each file written.
pub async fn download_hf_model_files(model: &str, dest_dir: &Path) -> anyhow::Result<Vec<(String, String)>> {
    fs::create_dir_all(dest_dir)
        .await
        .with_context(|| format!("mkdir {}", dest_dir.display()))?;

    let repo = resolve_hf_repo(model);
    let client = reqwest::Client::builder()
        .user_agent(concat!("arsenic/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("build HTTP client")?;

    let base = format!("https://huggingface.co/{repo}/resolve/main");
    let candidates = [
        "config.json",
        "config_sentence_transformers.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "modules.json",
        "vocab.txt",
        "special_tokens_map.json",
        "model.safetensors",
    ];

    let mut manifest = Vec::new();
    for file in candidates {
        let url = format!("{base}/{file}");
        let dest = dest_dir.join(file);
        match download_one(&client, &url, &dest, file).await {
            Ok(sha) => manifest.push((file.to_string(), sha)),
            Err(e) => {
                if file == "model.safetensors" {
                    return Err(e).with_context(|| format!("required file {file}"));
                }
                tracing::debug!(%file, error = %e, "skipped");
            }
        }
    }

    anyhow::ensure!(
        !manifest.is_empty(),
        "no files downloaded for repo {repo}; check model id and network"
    );
    Ok(manifest)
}

async fn download_one(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    label: &str,
) -> anyhow::Result<String> {
    let resp = client.get(url).send().await?;
    if resp.status() == StatusCode::NOT_FOUND {
        anyhow::bail!("404");
    }
    let resp = resp.error_for_status().with_context(|| format!("GET {url}"))?;

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg} {bytes}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.set_message(label.to_string());

    let mut stream = resp.bytes_stream();
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .await
            .with_context(|| format!("mkdir {}", parent.display()))?;
    }
    let mut file = fs::File::create(dest)
        .await
        .with_context(|| format!("create {}", dest.display()))?;
    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read body")?;
        hasher.update(&chunk);
        file.write_all(&chunk).await.context("write file")?;
        downloaded += chunk.len() as u64;
        pb.set_message(format!("{label} ({})", format_bytes(downloaded)));
        pb.tick();
    }
    pb.finish_with_message(format!("{label} {}", format_bytes(downloaded)));

    let hash = format!("{:x}", hasher.finalize());
    let fname = dest
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("artifact");
    let checksum_name = format!("{fname}.sha256");
    let checksum_path = dest
        .parent()
        .map(|p| p.join(&checksum_name))
        .unwrap_or_else(|| PathBuf::from(checksum_name));
    let _ = fs::write(&checksum_path, format!("{hash}\n")).await;
    Ok(hash)
}

fn format_bytes(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2} GB", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2} MB", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1} kB", n as f64 / 1_000.0)
    } else {
        format!("{n} B")
    }
}
