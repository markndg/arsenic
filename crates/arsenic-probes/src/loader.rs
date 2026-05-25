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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn tmp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "arsenic-loader-test-{label}-{}",
            Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_toml(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        path
    }

    const HAPPY_PATH_TOML: &str = r#"
[[probes]]
name = "factual_capital"
category = "Factual"
prompt = "What is the capital of France?"
known_answer = "Paris"
tags = ["geography", "deterministic"]

[[probes]]
name = "tone_check"
category = "Tone"
prompt = "Tell me about quantum mechanics."
system_prompt = "Respond formally."
expected_tone = "Formal"
expected_verbosity = "Concise"
refusal_expectation = "ShouldAnswer"
"#;

    #[test]
    fn happy_path_parses_all_fields() {
        let dir = tmp_dir("happy");
        write_toml(&dir, "set.toml", HAPPY_PATH_TOML);
        let probes = ProbeLoader::load_user_corpus(&dir).unwrap();
        assert_eq!(probes.len(), 2);

        let factual = probes.iter().find(|p| p.name == "factual_capital").unwrap();
        assert!(matches!(factual.category, ProbeCategory::Factual));
        assert_eq!(factual.known_answer.as_deref(), Some("Paris"));
        assert_eq!(factual.tags, vec!["geography", "deterministic"]);
        assert!(matches!(factual.source, ProbeSource::UserDefined));

        let tone = probes.iter().find(|p| p.name == "tone_check").unwrap();
        assert_eq!(tone.system_prompt.as_deref(), Some("Respond formally."));
        assert!(tone.expected_tone.is_some());
        assert!(tone.expected_verbosity.is_some());
        assert!(tone.refusal_expectation.is_some());
    }

    #[test]
    fn unknown_category_is_a_clear_error() {
        let dir = tmp_dir("badcat");
        write_toml(
            &dir,
            "bad.toml",
            r#"
[[probes]]
name = "x"
category = "NotARealDimension"
prompt = "Hi."
"#,
        );
        let err = ProbeLoader::load_user_corpus(&dir).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("NotARealDimension"),
            "error should mention the offending category, got: {msg}"
        );
    }

    #[test]
    fn category_parsing_is_case_insensitive() {
        let dir = tmp_dir("case");
        write_toml(
            &dir,
            "set.toml",
            r#"
[[probes]]
name = "p1"
category = "factual"
prompt = "lowercased"

[[probes]]
name = "p2"
category = "FACTUAL"
prompt = "uppercased"

[[probes]]
name = "p3"
category = "Factual"
prompt = "titlecased"
"#,
        );
        let probes = ProbeLoader::load_user_corpus(&dir).unwrap();
        assert_eq!(probes.len(), 3);
        assert!(probes
            .iter()
            .all(|p| matches!(p.category, ProbeCategory::Factual)));
    }

    #[test]
    fn missing_required_field_fails_to_parse() {
        let dir = tmp_dir("missing");
        write_toml(
            &dir,
            "bad.toml",
            r#"
[[probes]]
name = "x"
category = "Factual"
# prompt missing!
"#,
        );
        let err = ProbeLoader::load_user_corpus(&dir).unwrap_err();
        let msg = format!("{err:?}").to_lowercase();
        assert!(msg.contains("prompt") || msg.contains("missing"), "got: {msg}");
    }

    #[test]
    fn load_single_file_directly() {
        let dir = tmp_dir("file");
        let p = write_toml(&dir, "single.toml", HAPPY_PATH_TOML);
        let probes = ProbeLoader::load_user_corpus(&p).unwrap();
        assert_eq!(probes.len(), 2);
        assert!(probes.iter().all(|p| matches!(p.source, ProbeSource::UserDefined)));
    }

    #[test]
    fn standard_suite_loader_tags_source_as_standard() {
        let dir = tmp_dir("std");
        write_toml(
            &dir,
            "set.toml",
            r#"
[[probes]]
name = "x"
category = "Factual"
prompt = "Hi."
"#,
        );
        let probes = ProbeLoader::load_standard_suite(&dir).unwrap();
        assert_eq!(probes.len(), 1);
        assert!(matches!(probes[0].source, ProbeSource::Standard));
    }

    #[test]
    fn load_standard_categories_filters_correctly() {
        let dir = tmp_dir("filter");
        write_toml(
            &dir,
            "set.toml",
            r#"
[[probes]]
name = "f1"
category = "Factual"
prompt = "Q1"

[[probes]]
name = "t1"
category = "Tone"
prompt = "Q2"

[[probes]]
name = "f2"
category = "Factual"
prompt = "Q3"
"#,
        );
        let only_factual =
            ProbeLoader::load_standard_categories(&dir, &[ProbeCategory::Factual]).unwrap();
        assert_eq!(only_factual.len(), 2);
        assert!(only_factual
            .iter()
            .all(|p| matches!(p.category, ProbeCategory::Factual)));
    }

    #[test]
    fn non_toml_files_in_directory_are_ignored() {
        let dir = tmp_dir("mixed");
        write_toml(&dir, "set.toml", HAPPY_PATH_TOML);
        std::fs::write(dir.join("README.md"), "# notes").unwrap();
        std::fs::write(dir.join("ignore.txt"), "ignore me").unwrap();
        let probes = ProbeLoader::load_user_corpus(&dir).unwrap();
        assert_eq!(probes.len(), 2);
    }

    #[test]
    fn each_probe_gets_a_unique_uuid() {
        let dir = tmp_dir("uuid");
        write_toml(&dir, "set.toml", HAPPY_PATH_TOML);
        let probes = ProbeLoader::load_user_corpus(&dir).unwrap();
        assert_eq!(probes.len(), 2);
        assert_ne!(probes[0].id, probes[1].id);
    }
}
