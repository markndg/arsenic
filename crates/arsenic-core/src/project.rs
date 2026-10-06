//! Project layout for Application Contracts: `.arsenic/`, `arsenic.lock`, config.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::contract::discovery::{default_contract_path, load_contract_file, write_contract_file};
use crate::contract::ApplicationContract;
use crate::qualify::{BaselineSnapshot, QualificationResult, ValidatedPatch};

pub const PROJECT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateEntry {
    pub model: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ArsenicProjectConfig {
    #[serde(default)]
    pub application: Option<String>,
    #[serde(default)]
    pub production_model: Option<String>,
    #[serde(default)]
    pub candidates: Vec<CandidateEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LockQualified {
    pub model: String,
    pub qualification: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LockBlocked {
    pub model: String,
    pub qualification: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LockProduction {
    pub model: String,
    pub baseline: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArsenicLock {
    pub application: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub production: Option<LockProduction>,
    #[serde(default)]
    pub qualified: Vec<LockQualified>,
    #[serde(default)]
    pub blocked: Vec<LockBlocked>,
    #[serde(default)]
    pub review: Vec<LockQualified>,
    #[serde(default)]
    pub contract_hash: Option<String>,
    #[serde(default)]
    pub baseline_hash: Option<String>,
}

impl ArsenicLock {
    pub fn new(application: impl Into<String>) -> Self {
        Self {
            application: application.into(),
            production: None,
            qualified: vec![],
            blocked: vec![],
            review: vec![],
            contract_hash: None,
            baseline_hash: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectPaths {
    pub root: PathBuf,
}

impl ProjectPaths {
    pub fn discover(start: &Path) -> Result<Self> {
        let mut cur = start.canonicalize().unwrap_or_else(|_| start.to_path_buf());
        loop {
            if cur.join("arsenic.toml").exists() || cur.join(".arsenic").exists() {
                return Ok(Self { root: cur });
            }
            if !cur.pop() {
                bail!("No arsenic project found (looked for arsenic.toml or .arsenic/)");
            }
        }
    }

    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn arsenic_dir(&self) -> PathBuf {
        self.root.join(".arsenic")
    }
    pub fn contract_path(&self) -> PathBuf {
        default_contract_path(&self.root)
    }
    pub fn discovery_path(&self) -> PathBuf {
        self.arsenic_dir().join("discovery.json")
    }
    pub fn baselines_dir(&self) -> PathBuf {
        self.arsenic_dir().join("baselines")
    }
    pub fn qualifications_dir(&self) -> PathBuf {
        self.arsenic_dir().join("qualifications")
    }
    pub fn patches_dir(&self) -> PathBuf {
        self.arsenic_dir().join("patches")
    }
    pub fn lock_path(&self) -> PathBuf {
        self.root.join("arsenic.lock")
    }
    pub fn config_path(&self) -> PathBuf {
        self.root.join("arsenic.toml")
    }
    pub fn ensure_layout(&self) -> Result<()> {
        for d in [
            self.arsenic_dir(),
            self.arsenic_dir().join("contract"),
            self.baselines_dir(),
            self.qualifications_dir(),
            self.patches_dir(),
        ] {
            fs::create_dir_all(&d)?;
        }
        Ok(())
    }
}

pub fn load_config(paths: &ProjectPaths) -> Result<ArsenicProjectConfig> {
    let p = paths.config_path();
    if !p.exists() {
        return Ok(ArsenicProjectConfig::default());
    }
    let text = fs::read_to_string(&p)?;
    Ok(toml::from_str(&text)?)
}

pub fn save_config(paths: &ProjectPaths, cfg: &ArsenicProjectConfig) -> Result<()> {
    let text = toml::to_string_pretty(cfg)?;
    fs::write(paths.config_path(), text)?;
    Ok(())
}

pub fn load_lock(paths: &ProjectPaths) -> Result<Option<ArsenicLock>> {
    let p = paths.lock_path();
    if !p.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&p)?;
    // Prefer JSON lock for stable diffs; also accept TOML-ish JSON.
    if let Ok(lock) = serde_json::from_str::<ArsenicLock>(&text) {
        return Ok(Some(lock));
    }
    Ok(Some(toml::from_str(&text)?))
}

pub fn save_lock(paths: &ProjectPaths, lock: &ArsenicLock) -> Result<()> {
    // Human-editable YAML-like via TOML.
    let text = toml::to_string_pretty(lock)?;
    fs::write(paths.lock_path(), text)?;
    Ok(())
}

pub fn load_contract(paths: &ProjectPaths) -> Result<ApplicationContract> {
    load_contract_file(&paths.contract_path())
}

pub fn save_contract(paths: &ProjectPaths, contract: &ApplicationContract) -> Result<()> {
    write_contract_file(&paths.contract_path(), contract)
}

pub fn next_baseline_id(paths: &ProjectPaths) -> Result<String> {
    let dir = paths.baselines_dir();
    let mut max = 0u32;
    if dir.exists() {
        for e in fs::read_dir(&dir)?.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if let Some(n) = name
                .strip_prefix("baseline-")
                .and_then(|s| s.strip_suffix(".json"))
                .and_then(|s| s.parse().ok())
            {
                max = max.max(n);
            }
        }
    }
    Ok(format!("baseline-{:04}", max + 1))
}

pub fn save_baseline(paths: &ProjectPaths, snap: &BaselineSnapshot) -> Result<PathBuf> {
    paths.ensure_layout()?;
    let path = paths.baselines_dir().join(format!("{}.json", snap.id));
    if path.exists() {
        bail!("Baseline {} already exists (baselines are immutable)", snap.id);
    }
    fs::write(&path, serde_json::to_string_pretty(snap)?)?;
    Ok(path)
}

pub fn load_baseline(paths: &ProjectPaths, id: &str) -> Result<BaselineSnapshot> {
    let path = paths.baselines_dir().join(format!("{id}.json"));
    let text = fs::read_to_string(&path)
        .with_context(|| format!("baseline not found: {}", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

pub fn load_current_baseline(paths: &ProjectPaths) -> Result<BaselineSnapshot> {
    let lock = load_lock(paths)?.context("no arsenic.lock — run arsenic baseline first")?;
    let prod = lock.production.context("no production baseline in arsenic.lock")?;
    load_baseline(paths, &prod.baseline)
}

pub fn next_qualification_id(paths: &ProjectPaths) -> Result<String> {
    let dir = paths.qualifications_dir();
    let mut max = 0u32;
    if dir.exists() {
        for e in fs::read_dir(&dir)?.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if let Some(n) = name
                .strip_prefix("qual-")
                .and_then(|s| s.strip_suffix(".json"))
                .and_then(|s| s.parse().ok())
            {
                max = max.max(n);
            }
        }
    }
    Ok(format!("qual-{:04}", max + 1))
}

pub fn save_qualification(paths: &ProjectPaths, result: &QualificationResult) -> Result<PathBuf> {
    paths.ensure_layout()?;
    let path = paths
        .qualifications_dir()
        .join(format!("{}.json", result.id));
    // Never overwrite historical evidence.
    if path.exists() {
        bail!("Qualification {} already exists", result.id);
    }
    fs::write(&path, serde_json::to_string_pretty(result)?)?;
    Ok(path)
}

pub fn load_qualification(paths: &ProjectPaths, id: &str) -> Result<QualificationResult> {
    let path = paths.qualifications_dir().join(format!("{id}.json"));
    let text = fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&text)?)
}

pub fn list_qualifications(paths: &ProjectPaths) -> Result<Vec<QualificationResult>> {
    let mut out = Vec::new();
    let dir = paths.qualifications_dir();
    if !dir.exists() {
        return Ok(out);
    }
    let mut files: Vec<_> = fs::read_dir(&dir)?.flatten().map(|e| e.path()).collect();
    files.sort();
    for f in files {
        if f.extension().and_then(|e| e.to_str()) == Some("json") {
            if let Ok(text) = fs::read_to_string(&f) {
                if let Ok(q) = serde_json::from_str::<QualificationResult>(&text) {
                    out.push(q);
                }
            }
        }
    }
    Ok(out)
}

/// Mark qualifications stale when contract or baseline hash diverges.
///
/// Does **not** mutate historical qualification JSON. Returns count that would
/// be considered stale under current hashes (for CLI messaging).
pub fn count_stale_qualifications(
    paths: &ProjectPaths,
    contract_hash: &str,
    baseline_hash: &str,
) -> Result<usize> {
    let mut n = 0;
    for q in list_qualifications(paths)? {
        if q.effective_decision(contract_hash, baseline_hash) == crate::qualify::QualificationDecision::Stale
        {
            n += 1;
        }
    }
    Ok(n)
}

/// Deprecated alias — retained for callers; no longer rewrites history.
pub fn mark_stale_qualifications(
    paths: &ProjectPaths,
    contract_hash: &str,
    baseline_hash: &str,
) -> Result<usize> {
    count_stale_qualifications(paths, contract_hash, baseline_hash)
}

pub fn save_patch(paths: &ProjectPaths, patch: &ValidatedPatch) -> Result<PathBuf> {
    paths.ensure_layout()?;
    let path = paths
        .patches_dir()
        .join(format!("{}.json", patch.qualification_id));
    fs::write(&path, serde_json::to_string_pretty(patch)?)?;
    Ok(path)
}

pub fn load_patch(paths: &ProjectPaths, qualification_id: &str) -> Result<ValidatedPatch> {
    let path = paths
        .patches_dir()
        .join(format!("{qualification_id}.json"));
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

pub fn update_lock_from_qualification(
    lock: &mut ArsenicLock,
    result: &QualificationResult,
) {
    // Remove prior entries for this model.
    lock.qualified.retain(|q| q.model != result.candidate_model);
    lock.blocked.retain(|q| q.model != result.candidate_model);
    lock.review.retain(|q| q.model != result.candidate_model);

    let entry = LockQualified {
        model: result.candidate_model.clone(),
        qualification: result.id.clone(),
    };
    match result.decision {
        crate::qualify::QualificationDecision::Pass
        | crate::qualify::QualificationDecision::PassWithWarnings
        | crate::qualify::QualificationDecision::PassWithPatch => {
            lock.qualified.push(entry);
        }
        crate::qualify::QualificationDecision::Block => {
            lock.blocked.push(LockBlocked {
                model: result.candidate_model.clone(),
                qualification: result.id.clone(),
            });
        }
        crate::qualify::QualificationDecision::Review
        | crate::qualify::QualificationDecision::Stale => {
            lock.review.push(entry);
        }
    }
}

pub fn filter_candidates<'a>(
    candidates: &'a [CandidateEntry],
    tag: Option<&str>,
) -> Vec<&'a CandidateEntry> {
    candidates
        .iter()
        .filter(|c| match tag {
            None => true,
            Some(t) => c.tags.iter().any(|x| x == t),
        })
        .collect()
}

/// Parse `provider:model` loosely.
pub fn parse_model_spec(spec: &str) -> (String, String) {
    if let Some((p, m)) = spec.split_once(':') {
        (p.to_string(), m.to_string())
    } else {
        ("unknown".into(), spec.to_string())
    }
}

pub fn write_fixture_behaviours(
    path: &Path,
    behaviours: &BTreeMap<String, crate::qualify::CapturedBehaviour>,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(behaviours)?)?;
    Ok(())
}

pub fn load_fixture_behaviours(
    path: &Path,
) -> Result<BTreeMap<String, crate::qualify::CapturedBehaviour>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}
