//! Baseline response cache.
//!
//! Captures and replays `ModelResponse` outputs keyed by the *request-affecting*
//! inputs to the model.
//!
//! # Invalidation rule
//!
//! **Changing analysis expectations re-analyses cached outputs. Changing
//! prompts, model, or runtime settings invalidates cached outputs.**
//!
//! The cache key is a SHA-256 over the canonical JSON of a [`CacheKey`]
//! containing:
//!
//! - adapter type, endpoint, model id
//! - temperature, max tokens
//! - system prompt, user prompt
//! - `expected_schema` (only when it's sent to the model, e.g. via
//!   structured-output / response-format APIs)
//!
//! Anything that *doesn't* affect what the model was asked is deliberately
//! excluded from the key, even though it lives on [`crate::types::Probe`].
//! That includes:
//!
//! - `tags`
//! - `known_answer`
//! - `custom_assertions`
//! - `instructions` (rubric: `MaxWords`, `MustNotContain`, etc.)
//! - `refusal_expectation`
//! - `expected_verbosity`, `expected_tone`
//! - `mutation_hint`
//! - probe `id` and `name`
//!
//! The model never sees those fields — they drive scoring *after* the
//! response. Editing them re-grades the existing baseline against new rules,
//! which is usually what you want. Editing the prompt, system prompt, or
//! sampling settings is a different question being asked of a different
//! system, and gets a fresh capture.
//!
//! If you suspect the model itself changed under a fixed alias (e.g. a
//! provider rolled a new revision of `gpt-4o-mini`), force re-capture by
//! removing and re-creating the baseline, or pass `--baseline NAME` together
//! with a live `--v1` spec to fill in any missing probes via cache-warming.
//!
//! # Layout
//!
//! On disk a baseline lives under `<cache_dir>/<name>/`:
//!
//! ```text
//! <name>/
//!   manifest.json
//!   probes/
//!     <sha[..2]>/<sha>.json   ← one file per (probe × model × sampling) tuple
//!   lock                       ← present iff baseline is frozen
//! ```
//!
//! Multiple runs (consistency runs) of the same probe accumulate into the
//! `runs[]` array of the same file. Failed responses (`FinishReason::Error`)
//! are never written.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::types::{FinishReason, ModelResponse, Probe};

/// Bump only on incompatible on-disk format changes; provide a migration
/// path in `BaselineCache::verify` and `read_one` for older versions.
pub const CACHE_SCHEMA_VERSION: u32 = 1;

/// Inputs hashed to form the cache key.
///
/// **Invalidation rule**: anything that affects what the model is asked goes
/// in here (and changing it busts the cache); anything that only affects how
/// ARSENIC analyses the response is excluded (and changing it re-analyses
/// the existing baseline). See the module-level docs for the full list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheKey {
    pub cache_schema_version: u32,
    pub adapter_type: String,
    pub endpoint: String,
    pub model_id: String,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub expected_schema: Option<serde_json::Value>,
}

impl CacheKey {
    pub fn new(
        adapter_type: &str,
        endpoint: &str,
        model_id: &str,
        temperature: f64,
        max_tokens: Option<usize>,
        probe: &Probe,
    ) -> Self {
        Self {
            cache_schema_version: CACHE_SCHEMA_VERSION,
            adapter_type: adapter_type.to_string(),
            endpoint: endpoint.to_string(),
            model_id: model_id.to_string(),
            temperature: Some(temperature),
            max_tokens,
            system_prompt: probe.system_prompt.clone(),
            prompt: probe.prompt.clone(),
            expected_schema: probe.expected_schema.clone(),
        }
    }

    /// Stable SHA-256 of the canonical-JSON serialisation of the key fields.
    pub fn hash(&self) -> String {
        let value = serde_json::to_value(self).expect("CacheKey serialises");
        let canonical = canonical_json(&value);
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Deterministic JSON: object keys sorted, no whitespace. Stable across
/// trivial reformatting of the key struct.
fn canonical_json(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys
                .into_iter()
                .map(|k| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).expect("string key"),
                        canonical_json(map.get(k).expect("key present"))
                    )
                })
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        serde_json::Value::Array(arr) => {
            let parts: Vec<String> = arr.iter().map(canonical_json).collect();
            format!("[{}]", parts.join(","))
        }
        other => serde_json::to_string(other).expect("scalar serialises"),
    }
}

/// A single captured invocation of the model for one key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedRun {
    pub content: String,
    pub token_count: usize,
    pub latency_ms: u64,
    pub finish_reason: FinishReason,
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub raw: serde_json::Value,
}

impl CachedRun {
    pub fn from_response(r: &ModelResponse) -> Self {
        Self {
            content: r.content.clone(),
            token_count: r.token_count,
            latency_ms: r.latency_ms,
            finish_reason: r.finish_reason.clone(),
            timestamp: r.timestamp,
            raw: r.raw.clone(),
        }
    }

    pub fn to_response(&self, probe_id: uuid::Uuid, label: &str, model_id: &str) -> ModelResponse {
        ModelResponse {
            probe_id,
            model_label: label.to_string(),
            model_id: model_id.to_string(),
            content: self.content.clone(),
            token_count: self.token_count,
            latency_ms: self.latency_ms,
            finish_reason: self.finish_reason.clone(),
            timestamp: self.timestamp,
            raw: self.raw.clone(),
        }
    }
}

/// On-disk file format for one cached probe key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    pub cache_schema_version: u32,
    pub key: CacheKey,
    pub key_hash: String,
    pub probe_name_at_capture: String,
    pub runs: Vec<CachedRun>,
    pub adapter_version: String,
    pub arsenic_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineModel {
    pub adapter_type: String,
    pub endpoint: String,
    pub model_id: String,
    pub temperature: f64,
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineProbeEntry {
    pub name: String,
    pub key_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineManifest {
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub arsenic_version: String,
    pub model: BaselineModel,
    pub corpus_fingerprint: String,
    pub probes: Vec<BaselineProbeEntry>,
    pub consistency_runs: usize,
    #[serde(default)]
    pub locked: bool,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub created_by: Option<String>,
}

impl BaselineManifest {
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }
}

/// On-disk baseline cache rooted at `<cache_dir>/<name>/`.
#[derive(Debug, Clone)]
pub struct BaselineCache {
    root: PathBuf,
}

impl BaselineCache {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.json")
    }

    pub fn exists(&self) -> bool {
        self.manifest_path().exists()
    }

    pub fn read_manifest(&self) -> Result<BaselineManifest> {
        let path = self.manifest_path();
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).context("parse baseline manifest")
    }

    pub fn write_manifest(&self, manifest: &BaselineManifest) -> Result<()> {
        std::fs::create_dir_all(&self.root)?;
        let text = serde_json::to_string_pretty(manifest)?;
        std::fs::write(self.manifest_path(), text)?;
        Ok(())
    }

    pub fn lock_path(&self) -> PathBuf {
        self.root.join("lock")
    }

    pub fn is_locked(&self) -> bool {
        self.lock_path().exists()
    }

    pub fn lock(&self) -> Result<()> {
        std::fs::write(self.lock_path(), b"locked\n")?;
        if self.manifest_path().exists() {
            let mut m = self.read_manifest()?;
            m.locked = true;
            self.write_manifest(&m)?;
        }
        Ok(())
    }

    pub fn unlock(&self) -> Result<()> {
        let lp = self.lock_path();
        if lp.exists() {
            std::fs::remove_file(lp)?;
        }
        if self.manifest_path().exists() {
            let mut m = self.read_manifest()?;
            m.locked = false;
            self.write_manifest(&m)?;
        }
        Ok(())
    }

    pub fn probes_dir(&self) -> PathBuf {
        self.root.join("probes")
    }

    pub fn probe_path(&self, key_hash: &str) -> PathBuf {
        let shard = if key_hash.len() >= 2 {
            &key_hash[..2]
        } else {
            "00"
        };
        self.probes_dir().join(shard).join(format!("{key_hash}.json"))
    }

    pub fn read_one(&self, key_hash: &str) -> Result<Option<CachedResponse>> {
        let path = self.probe_path(key_hash);
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let parsed: CachedResponse =
            serde_json::from_str(&text).context("parse cached response")?;
        Ok(Some(parsed))
    }

    /// Append a successful run to the cache file for `key`. Errors are never
    /// persisted: a `FinishReason::Error` response is silently ignored.
    pub fn append_run(
        &self,
        key: &CacheKey,
        response: &ModelResponse,
        probe: &Probe,
    ) -> Result<()> {
        if matches!(response.finish_reason, FinishReason::Error) {
            return Ok(());
        }
        if self.is_locked() {
            anyhow::bail!(
                "baseline at {} is locked; unfreeze before writing",
                self.root.display()
            );
        }
        let key_hash = key.hash();
        let path = self.probe_path(&key_hash);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut entry = match self.read_one(&key_hash)? {
            Some(e) => e,
            None => CachedResponse {
                cache_schema_version: CACHE_SCHEMA_VERSION,
                key: key.clone(),
                key_hash: key_hash.clone(),
                probe_name_at_capture: probe.name.clone(),
                runs: Vec::new(),
                adapter_version: arsenic_adapter_version_string(),
                arsenic_version: env!("CARGO_PKG_VERSION").to_string(),
            },
        };
        entry.runs.push(CachedRun::from_response(response));
        let text = serde_json::to_string_pretty(&entry)?;
        std::fs::write(&path, text)?;
        Ok(())
    }

    /// Re-hash every cached file's `key` and confirm it matches both the file
    /// stem and the stored `key_hash` field. Surfaces corruption / tampering.
    pub fn verify(&self) -> Result<VerifyReport> {
        let mut report = VerifyReport::default();
        let probes_dir = self.probes_dir();
        if !probes_dir.exists() {
            return Ok(report);
        }
        for shard in std::fs::read_dir(&probes_dir)? {
            let shard = shard?;
            if !shard.path().is_dir() {
                continue;
            }
            for entry in std::fs::read_dir(shard.path())? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) != Some("json") {
                    continue;
                }
                report.total += 1;
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                let text = match std::fs::read_to_string(&path) {
                    Ok(t) => t,
                    Err(_) => {
                        report.unreadable.push(path.display().to_string());
                        continue;
                    }
                };
                let cached: CachedResponse = match serde_json::from_str(&text) {
                    Ok(c) => c,
                    Err(_) => {
                        report.unreadable.push(path.display().to_string());
                        continue;
                    }
                };
                let recomputed = cached.key.hash();
                if recomputed != stem || recomputed != cached.key_hash {
                    report
                        .mismatched
                        .push(format!("{} (recomputed={recomputed})", path.display()));
                } else {
                    report.ok += 1;
                }
            }
        }
        Ok(report)
    }

    /// All baselines under `parent_dir` (anything containing `manifest.json`).
    pub fn list_baselines(parent_dir: &Path) -> Result<Vec<PathBuf>> {
        let mut out = Vec::new();
        if !parent_dir.exists() {
            return Ok(out);
        }
        for entry in std::fs::read_dir(parent_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() && path.join("manifest.json").exists() {
                out.push(path);
            }
        }
        out.sort();
        Ok(out)
    }
}

/// Returned by [`BaselineCache::verify`]: number of cached files inspected,
/// number that re-hashed cleanly, and any mismatched / unreadable filenames.
#[derive(Debug, Default)]
pub struct VerifyReport {
    pub total: usize,
    pub ok: usize,
    pub mismatched: Vec<String>,
    pub unreadable: Vec<String>,
}

impl VerifyReport {
    pub fn is_clean(&self) -> bool {
        self.mismatched.is_empty() && self.unreadable.is_empty()
    }
}

/// Stable fingerprint of the request-affecting content of a probe corpus.
/// Used by the manifest to record which corpus the baseline was captured against.
pub fn corpus_fingerprint(probes: &[Probe]) -> String {
    let mut entries: Vec<(&str, &str, Option<&str>)> = probes
        .iter()
        .map(|p| (p.name.as_str(), p.prompt.as_str(), p.system_prompt.as_deref()))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    let mut hasher = Sha256::new();
    for (name, prompt, system) in entries {
        hasher.update(name.as_bytes());
        hasher.update(b"\x1f");
        hasher.update(prompt.as_bytes());
        hasher.update(b"\x1f");
        if let Some(s) = system {
            hasher.update(s.as_bytes());
        }
        hasher.update(b"\x1e");
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn arsenic_adapter_version_string() -> String {
    format!("arsenic {}", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FinishReason, ProbeCategory, ProbeSource, RefusalExpectation};
    use uuid::Uuid;

    fn tmp_root(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "arsenic-cache-test-{label}-{}",
            Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn mk_probe(name: &str, prompt: &str) -> Probe {
        Probe {
            id: Uuid::new_v4(),
            name: name.to_string(),
            category: ProbeCategory::Factual,
            prompt: prompt.to_string(),
            system_prompt: None,
            known_answer: None,
            expected_schema: None,
            instructions: vec![],
            tags: vec![],
            source: ProbeSource::Standard,
            expected_verbosity: None,
            expected_tone: None,
            refusal_expectation: None,
            mutation_hint: None,
            custom_assertions: vec![],
        }
    }

    fn mk_resp(probe_id: Uuid, content: &str) -> ModelResponse {
        ModelResponse {
            probe_id,
            model_label: "v1".into(),
            model_id: "gpt-4o-mini".into(),
            content: content.into(),
            token_count: content.split_whitespace().count(),
            latency_ms: 100,
            finish_reason: FinishReason::Stop,
            timestamp: Utc::now(),
            raw: serde_json::json!({}),
        }
    }

    fn key_for(probe: &Probe) -> CacheKey {
        CacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            0.0,
            None,
            probe,
        )
    }

    #[test]
    fn hash_changes_with_prompt() {
        let p1 = mk_probe("a", "What is 2+2?");
        let p2 = mk_probe("a", "What is 3+3?");
        assert_ne!(key_for(&p1).hash(), key_for(&p2).hash());
    }

    #[test]
    fn hash_changes_with_model() {
        let p = mk_probe("a", "What is 2+2?");
        let k1 = CacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            0.0,
            None,
            &p,
        );
        let k2 = CacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "gpt-4.1-mini",
            0.0,
            None,
            &p,
        );
        assert_ne!(k1.hash(), k2.hash());
    }

    #[test]
    fn hash_changes_with_temperature() {
        let p = mk_probe("a", "What is 2+2?");
        let k1 = CacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            0.0,
            None,
            &p,
        );
        let k2 = CacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            0.7,
            None,
            &p,
        );
        assert_ne!(k1.hash(), k2.hash());
    }

    #[test]
    fn hash_changes_with_endpoint() {
        let p = mk_probe("a", "What is 2+2?");
        let k1 = CacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            0.0,
            None,
            &p,
        );
        let k2 = CacheKey::new(
            "openai",
            "http://localhost:11434/v1",
            "gpt-4o-mini",
            0.0,
            None,
            &p,
        );
        assert_ne!(k1.hash(), k2.hash());
    }

    #[test]
    fn hash_changes_with_system_prompt() {
        let mut p1 = mk_probe("a", "Same prompt.");
        let mut p2 = mk_probe("a", "Same prompt.");
        p1.system_prompt = Some("You are a helpful assistant.".into());
        p2.system_prompt = Some("You are a sarcastic assistant.".into());
        assert_ne!(key_for(&p1).hash(), key_for(&p2).hash());
    }

    #[test]
    fn hash_changes_with_expected_schema() {
        let mut p1 = mk_probe("a", "Same prompt.");
        let mut p2 = mk_probe("a", "Same prompt.");
        p1.expected_schema = Some(serde_json::json!({"type": "object", "required": ["x"]}));
        p2.expected_schema = Some(serde_json::json!({"type": "object", "required": ["y"]}));
        assert_ne!(key_for(&p1).hash(), key_for(&p2).hash());
    }

    #[test]
    fn hash_stable_across_uuid_regeneration() {
        let mut p1 = mk_probe("a", "Same prompt.");
        let mut p2 = mk_probe("a", "Same prompt.");
        p1.id = Uuid::new_v4();
        p2.id = Uuid::new_v4();
        assert_ne!(p1.id, p2.id);
        assert_eq!(key_for(&p1).hash(), key_for(&p2).hash());
    }

    #[test]
    fn hash_unaffected_by_analyser_only_fields() {
        let mut p1 = mk_probe("a", "Same prompt.");
        let mut p2 = mk_probe("a", "Same prompt.");
        p1.tags = vec!["one".into()];
        p2.tags = vec!["two".into(), "three".into()];
        p1.known_answer = Some("answer one".into());
        p2.known_answer = Some("answer two".into());
        p1.mutation_hint = Some("hint one".into());
        p2.mutation_hint = Some("hint two".into());
        p1.refusal_expectation = Some(RefusalExpectation::ShouldAnswer);
        p2.refusal_expectation = Some(RefusalExpectation::ShouldRefuse);
        assert_eq!(
            key_for(&p1).hash(),
            key_for(&p2).hash(),
            "analyser-only fields must not affect the cache key"
        );
    }

    #[test]
    fn hash_unaffected_by_probe_name() {
        let p1 = mk_probe("alpha", "Same prompt.");
        let p2 = mk_probe("beta", "Same prompt.");
        assert_eq!(key_for(&p1).hash(), key_for(&p2).hash());
    }

    #[test]
    fn round_trip_write_read() {
        let root = tmp_root("roundtrip");
        let cache = BaselineCache::new(root.join("b"));
        let p = mk_probe("test", "What is 2+2?");
        let k = key_for(&p);
        let r = mk_resp(p.id, "4");
        cache.append_run(&k, &r, &p).unwrap();

        let read = cache.read_one(&k.hash()).unwrap().expect("present");
        assert_eq!(read.runs.len(), 1);
        assert_eq!(read.runs[0].content, "4");
        assert_eq!(read.probe_name_at_capture, "test");
    }

    #[test]
    fn multiple_runs_accumulate() {
        let root = tmp_root("multi");
        let cache = BaselineCache::new(root.join("b"));
        let p = mk_probe("test", "Same prompt.");
        let k = key_for(&p);
        for i in 0..3 {
            cache.append_run(&k, &mk_resp(p.id, &format!("run {i}")), &p).unwrap();
        }
        let read = cache.read_one(&k.hash()).unwrap().expect("present");
        assert_eq!(read.runs.len(), 3);
        assert_eq!(read.runs[0].content, "run 0");
        assert_eq!(read.runs[2].content, "run 2");
    }

    #[test]
    fn errors_are_not_cached() {
        let root = tmp_root("err");
        let cache = BaselineCache::new(root.join("b"));
        let p = mk_probe("test", "prompt");
        let k = key_for(&p);
        let mut r = mk_resp(p.id, "ERROR: timeout");
        r.finish_reason = FinishReason::Error;
        cache.append_run(&k, &r, &p).unwrap();
        assert!(cache.read_one(&k.hash()).unwrap().is_none());
    }

    #[test]
    fn locked_baseline_rejects_writes() {
        let root = tmp_root("lock");
        let cache = BaselineCache::new(root.join("b"));
        let manifest = BaselineManifest {
            name: "test".into(),
            created_at: Utc::now(),
            arsenic_version: env!("CARGO_PKG_VERSION").into(),
            model: BaselineModel {
                adapter_type: "openai".into(),
                endpoint: "https://api.openai.com/v1".into(),
                model_id: "gpt-4o-mini".into(),
                temperature: 0.0,
                max_tokens: None,
            },
            corpus_fingerprint: "sha256:abc".into(),
            probes: vec![],
            consistency_runs: 1,
            locked: false,
            notes: None,
            created_by: None,
        };
        cache.write_manifest(&manifest).unwrap();
        cache.lock().unwrap();
        assert!(cache.is_locked());

        let p = mk_probe("test", "p");
        let k = key_for(&p);
        let r = mk_resp(p.id, "x");
        let err = cache.append_run(&k, &r, &p);
        assert!(err.is_err());

        cache.unlock().unwrap();
        assert!(!cache.is_locked());
        cache.append_run(&k, &r, &p).unwrap();
    }

    #[test]
    fn verify_detects_tampered_files() {
        let root = tmp_root("tamper");
        let cache = BaselineCache::new(root.join("b"));
        let p = mk_probe("test", "prompt");
        let k = key_for(&p);
        let r = mk_resp(p.id, "x");
        cache.append_run(&k, &r, &p).unwrap();

        let path = cache.probe_path(&k.hash());
        let mut cached: CachedResponse =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        cached.key.prompt = "TAMPERED".into();
        std::fs::write(&path, serde_json::to_string(&cached).unwrap()).unwrap();

        let report = cache.verify().unwrap();
        assert_eq!(report.total, 1);
        assert_eq!(report.mismatched.len(), 1);
        assert!(!report.is_clean());
    }

    #[test]
    fn verify_clean_for_well_formed_cache() {
        let root = tmp_root("clean");
        let cache = BaselineCache::new(root.join("b"));
        let p = mk_probe("test", "prompt");
        let k = key_for(&p);
        let r = mk_resp(p.id, "ok");
        cache.append_run(&k, &r, &p).unwrap();
        let report = cache.verify().unwrap();
        assert_eq!(report.total, 1);
        assert_eq!(report.ok, 1);
        assert!(report.is_clean());
    }

    #[test]
    fn list_baselines_returns_only_dirs_with_manifest() {
        let root = tmp_root("listing");
        let a = root.join("a");
        let b = root.join("b");
        let c = root.join("c-no-manifest");
        let cache_a = BaselineCache::new(a.clone());
        let cache_b = BaselineCache::new(b.clone());
        std::fs::create_dir_all(&c).unwrap();
        let manifest = BaselineManifest {
            name: "x".into(),
            created_at: Utc::now(),
            arsenic_version: env!("CARGO_PKG_VERSION").into(),
            model: BaselineModel {
                adapter_type: "openai".into(),
                endpoint: "https://api.openai.com/v1".into(),
                model_id: "gpt-4o-mini".into(),
                temperature: 0.0,
                max_tokens: None,
            },
            corpus_fingerprint: "sha256:abc".into(),
            probes: vec![],
            consistency_runs: 1,
            locked: false,
            notes: None,
            created_by: None,
        };
        cache_a.write_manifest(&manifest).unwrap();
        cache_b.write_manifest(&manifest).unwrap();

        let found = BaselineCache::list_baselines(&root).unwrap();
        assert_eq!(found.len(), 2);
        assert!(found.iter().any(|p| p == &a));
        assert!(found.iter().any(|p| p == &b));
    }

    #[test]
    fn corpus_fingerprint_stable_across_order() {
        let p1 = mk_probe("a", "A");
        let p2 = mk_probe("b", "B");
        let f1 = corpus_fingerprint(&[p1.clone(), p2.clone()]);
        let f2 = corpus_fingerprint(&[p2, p1]);
        assert_eq!(f1, f2);
    }

    #[test]
    fn corpus_fingerprint_changes_with_prompt_edit() {
        let p1 = mk_probe("a", "A");
        let p1_edited = mk_probe("a", "A edited");
        assert_ne!(
            corpus_fingerprint(std::slice::from_ref(&p1)),
            corpus_fingerprint(&[p1_edited])
        );
    }

    #[test]
    fn hash_handles_unicode_emoji_and_rtl_stably() {
        // Bidi text + emoji + combining accents must not break canonical JSON.
        let p1 = mk_probe("a", "café 🥐 — مرحبا — naïve résumé");
        let p2 = mk_probe("a", "café 🥐 — مرحبا — naïve résumé");
        assert_eq!(
            key_for(&p1).hash(),
            key_for(&p2).hash(),
            "identical unicode prompts must produce identical hashes"
        );
        // And a different code-point sequence (NFC vs NFD-style) MUST hash differently
        // (we hash bytes, not normalised forms — this is documented behaviour).
        let p3 = mk_probe("a", "cafe 🥐 — مرحبا — naive resume");
        assert_ne!(key_for(&p1).hash(), key_for(&p3).hash());
    }

    #[test]
    fn hash_handles_very_long_prompts_without_truncation() {
        // 200k chars — exceeds anything a model would actually accept, but the
        // cache key must compute deterministically without panic or truncation.
        let big = "a".repeat(200_000);
        let p1 = mk_probe("p", &big);
        let p2 = mk_probe("p", &big);
        let h1 = key_for(&p1).hash();
        let h2 = key_for(&p2).hash();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64, "sha-256 hex should always be 64 chars");

        // Append one char → different hash.
        let bigger = format!("{big}b");
        let p3 = mk_probe("p", &bigger);
        assert_ne!(h1, key_for(&p3).hash());
    }

    #[test]
    fn expected_schema_field_order_does_not_affect_hash() {
        // Same logical JSON schema, different physical key ordering. Canonical
        // JSON sorts object keys, so the cache key must collapse them.
        let mut p1 = mk_probe("p", "Same prompt.");
        let mut p2 = mk_probe("p", "Same prompt.");
        p1.expected_schema = Some(serde_json::json!({
            "type": "object",
            "required": ["name", "age"],
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}}
        }));
        // Same content, reordered top-level keys + reordered nested keys.
        p2.expected_schema = Some(serde_json::json!({
            "properties": {"age": {"type": "number"}, "name": {"type": "string"}},
            "required": ["name", "age"],
            "type": "object"
        }));
        assert_eq!(
            key_for(&p1).hash(),
            key_for(&p2).hash(),
            "canonical JSON must be order-independent for object keys"
        );
    }

    #[test]
    fn cache_file_with_future_schema_version_is_still_parseable_but_distinguishable() {
        // We don't currently reject future versions, but the loaded struct
        // exposes the on-disk `cache_schema_version` so callers can detect them.
        // If this changes (e.g. add strict version checking), this test should
        // be updated to assert the new behaviour rather than silently passing.
        let root = tmp_root("future-version");
        let cache = BaselineCache::new(root.join("b"));
        let p = mk_probe("p", "hi");
        let k = key_for(&p);
        let r = mk_resp(p.id, "ok");
        cache.append_run(&k, &r, &p).unwrap();

        // Tamper: bump the version on disk.
        let path = cache.probe_path(&k.hash());
        let mut cached: CachedResponse =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        cached.cache_schema_version = 9999;
        std::fs::write(&path, serde_json::to_string(&cached).unwrap()).unwrap();

        let reread = cache.read_one(&k.hash()).unwrap().expect("present");
        assert_eq!(
            reread.cache_schema_version, 9999,
            "current loader surfaces the on-disk version verbatim"
        );
    }

    #[test]
    fn write_only_to_locked_baseline_via_append_run_is_rejected() {
        // append_run() is the lowest-level write path; ensure it refuses to
        // write to a locked baseline even when called directly (CachingAdapter
        // is the typical caller, but library users may bypass it).
        let root = tmp_root("locked-direct");
        let cache = BaselineCache::new(root.join("b"));
        let manifest = BaselineManifest {
            name: "x".into(),
            created_at: Utc::now(),
            arsenic_version: env!("CARGO_PKG_VERSION").into(),
            model: BaselineModel {
                adapter_type: "openai".into(),
                endpoint: "https://api.openai.com/v1".into(),
                model_id: "gpt-4o-mini".into(),
                temperature: 0.0,
                max_tokens: None,
            },
            corpus_fingerprint: "sha256:0".into(),
            probes: vec![],
            consistency_runs: 1,
            locked: false,
            notes: None,
            created_by: None,
        };
        cache.write_manifest(&manifest).unwrap();
        cache.lock().unwrap();

        let p = mk_probe("p", "hi");
        let err = cache.append_run(&key_for(&p), &mk_resp(p.id, "ok"), &p);
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("locked"), "got: {msg}");
    }

    #[test]
    fn empty_prompt_still_hashes_deterministically() {
        let p1 = mk_probe("p", "");
        let p2 = mk_probe("p", "");
        assert_eq!(key_for(&p1).hash(), key_for(&p2).hash());
        // Empty != whitespace.
        let p3 = mk_probe("p", " ");
        assert_ne!(key_for(&p1).hash(), key_for(&p3).hash());
    }
}
