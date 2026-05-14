use std::fs;
use std::path::Path;

use anyhow::Context;
use arsenic_core::{
    CustomAssertion, ExpectedTonePreference, ExpectedVerbosity, Probe, ProbeCategory, ProbeInstruction,
    ProbeSource, RefusalExpectation,
};
use serde::Deserialize;
use uuid::Uuid;
use walkdir::WalkDir;

pub struct ProbeLoader;

#[derive(Debug, Deserialize)]
struct ProbeFile {
    probes: Vec<TomlProbe>,
}

#[derive(Debug, Deserialize)]
struct TomlProbe {
    name: String,
    category: String,
    prompt: String,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    known_answer: Option<String>,
    #[serde(default)]
    expected_schema: Option<serde_json::Value>,
    #[serde(default)]
    instructions: Vec<ProbeInstruction>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    expected_verbosity: Option<ExpectedVerbosity>,
    #[serde(default)]
    expected_tone: Option<ExpectedTonePreference>,
    #[serde(default)]
    refusal_expectation: Option<RefusalExpectation>,
    #[serde(default)]
    mutation_hint: Option<String>,
    #[serde(default)]
    custom_assertions: Vec<CustomAssertion>,
}

impl ProbeLoader {
    pub fn load_standard_suite(suite_path: &Path) -> anyhow::Result<Vec<Probe>> {
        let mut out = Vec::new();
        for entry in WalkDir::new(suite_path).into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            if entry.path().extension().and_then(|s| s.to_str()) != Some("toml") {
                continue;
            }
            out.extend(load_file(entry.path(), ProbeSource::Standard)?);
        }
        Ok(out)
    }

    pub fn load_user_corpus(corpus_path: &Path) -> anyhow::Result<Vec<Probe>> {
        let mut out = Vec::new();
        if corpus_path.is_file() {
            return load_file(corpus_path, ProbeSource::UserDefined);
        }
        for entry in WalkDir::new(corpus_path).into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            if entry.path().extension().and_then(|s| s.to_str()) != Some("toml") {
                continue;
            }
            out.extend(load_file(entry.path(), ProbeSource::UserDefined)?);
        }
        Ok(out)
    }

    pub fn load_standard_categories(
        suite_path: &Path,
        categories: &[ProbeCategory],
    ) -> anyhow::Result<Vec<Probe>> {
        let want: std::collections::HashSet<_> = categories.iter().collect();
        let all = Self::load_standard_suite(suite_path)?;
        Ok(all
            .into_iter()
            .filter(|p| want.contains(&p.category))
            .collect())
    }
}

fn load_file(path: &Path, source: ProbeSource) -> anyhow::Result<Vec<Probe>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let parsed: ProbeFile = toml::from_str(&text)
        .with_context(|| format!("parse probe TOML {}", path.display()))?;
    let mut out = Vec::new();
    for p in parsed.probes {
        out.push(toml_to_probe(p, source.clone(), path)?);
    }
    Ok(out)
}

fn toml_to_probe(p: TomlProbe, source: ProbeSource, path: &Path) -> anyhow::Result<Probe> {
    let category = parse_category(&p.category)
        .with_context(|| format!("unknown category '{}' in {}", p.category, path.display()))?;
    Ok(Probe {
        id: Uuid::new_v4(),
        name: p.name,
        category,
        prompt: p.prompt,
        system_prompt: p.system_prompt,
        known_answer: p.known_answer,
        expected_schema: p.expected_schema,
        instructions: p.instructions,
        tags: p.tags,
        expected_verbosity: p.expected_verbosity,
        expected_tone: p.expected_tone,
        refusal_expectation: p.refusal_expectation,
        mutation_hint: p.mutation_hint,
        custom_assertions: p.custom_assertions,
        source,
    })
}

fn parse_category(s: &str) -> Option<ProbeCategory> {
    match s.to_lowercase().as_str() {
        "morphology" => Some(ProbeCategory::Morphology),
        "tone" => Some(ProbeCategory::Tone),
        "factual" => Some(ProbeCategory::Factual),
        "schema" => Some(ProbeCategory::Schema),
        "instruction" => Some(ProbeCategory::Instruction),
        "refusal" => Some(ProbeCategory::Refusal),
        "semantic" => Some(ProbeCategory::Semantic),
        _ => None,
    }
}
