//! Discover candidate contract artefacts from a project tree.
//!
//! Discovery proposes items with confidence and never silently invents
//! critical behavioural expectations (`auto_accepted = false` for block/review).

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};

use super::{
    ApplicationContract, ContractExpectation, ContractItem, ContractItemKind, ContractProvenance,
    ContractProvenanceSource, ContractSeverity,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryHit {
    pub kind: String,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_start: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_end: Option<u32>,
    pub parser: String,
    pub confidence: f64,
    pub auto_accepted: bool,
    pub summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposed_item: Option<ContractItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiscoveryReport {
    pub root: String,
    pub hits: Vec<DiscoveryHit>,
    pub prompts: usize,
    pub structured_schemas: usize,
    pub tool_schemas: usize,
    pub system_prompts: usize,
    pub existing_suites: usize,
}

/// Walk `root` and collect candidate AI-application artefacts.
pub fn discover_project(root: &Path) -> Result<DiscoveryReport> {
    let mut report = DiscoveryReport {
        root: root.display().to_string(),
        ..Default::default()
    };

    walk_dir(root, root, &mut report, 0)?;
    report.prompts = report
        .hits
        .iter()
        .filter(|h| h.kind == "prompt" || h.kind == "system_prompt")
        .count();
    report.system_prompts = report.hits.iter().filter(|h| h.kind == "system_prompt").count();
    report.structured_schemas = report
        .hits
        .iter()
        .filter(|h| h.kind == "structured_output")
        .count();
    report.tool_schemas = report.hits.iter().filter(|h| h.kind == "tool_schema").count();
    report.existing_suites = report
        .hits
        .iter()
        .filter(|h| h.kind == "behavioural_suite")
        .count();
    Ok(report)
}

fn walk_dir(root: &Path, dir: &Path, report: &mut DiscoveryReport, depth: usize) -> Result<()> {
    if depth > 8 {
        return Ok(());
    }
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.')
            || name == "target"
            || name == "node_modules"
            || name == "dist"
            || name == "build"
        {
            continue;
        }
        if path.is_dir() {
            walk_dir(root, &path, report, depth + 1)?;
            continue;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        match ext.as_str() {
            "txt" | "md" | "prompt" => inspect_prompt_file(root, &path, report)?,
            "json" => inspect_json(root, &path, report)?,
            "yaml" | "yml" => inspect_yaml_like(root, &path, report)?,
            "toml" => inspect_toml_suite(root, &path, report)?,
            "py" => inspect_python(root, &path, report)?,
            "ts" | "tsx" | "js" | "jsx" => inspect_js(root, &path, report)?,
            _ => {}
        }
    }
    Ok(())
}

fn rel(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn inspect_prompt_file(root: &Path, path: &Path, report: &mut DiscoveryReport) -> Result<()> {
    let text = fs::read_to_string(path).unwrap_or_default();
    if text.trim().is_empty() || text.len() > 100_000 {
        return Ok(());
    }
    let lower_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();
    let is_system = lower_name.contains("system") || text.to_lowercase().starts_with("you are ");
    let kind = if is_system { "system_prompt" } else { "prompt" };
    let prompt_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("prompt")
        .to_string();
    let item = ContractItem {
        id: format!("{prompt_id}.output_format.001"),
        kind: ContractItemKind::OutputFormat,
        severity: ContractSeverity::Info,
        prompt: prompt_id.clone(),
        system_prompt: if is_system {
            Some(text.clone())
        } else {
            None
        },
        user_prompt: if is_system { None } else { Some(text.clone()) },
        expectation: ContractExpectation::OutputFormat {
            description: format!("Preserve behaviour of discovered {kind}"),
        },
        provenance: ContractProvenance {
            source: ContractProvenanceSource::Discovery,
            file: Some(rel(root, path)),
            line_start: Some(1),
            line_end: Some(text.lines().count() as u32),
            parser: Some("plain_prompt".into()),
            confidence: 0.55,
            auto_accepted: false,
            note: Some("Discovered prompt file — confirm before treating as contract".into()),
        },
        tags: vec!["discovered".into()],
    };
    report.hits.push(DiscoveryHit {
        kind: kind.into(),
        path: rel(root, path),
        line_start: Some(1),
        line_end: Some(text.lines().count() as u32),
        parser: "plain_prompt".into(),
        confidence: 0.55,
        auto_accepted: false,
        summary: format!("Prompt artefact ({})", path.display()),
        proposed_item: Some(item),
    });
    Ok(())
}

fn inspect_json(root: &Path, path: &Path, report: &mut DiscoveryReport) -> Result<()> {
    let text = fs::read_to_string(path).unwrap_or_default();
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Ok(());
    };
    let path_s = rel(root, path);
    if looks_like_json_schema(&value) {
        let id_base = stem(path);
        let item = ContractItem {
            id: format!("{id_base}.structured_output.001"),
            kind: ContractItemKind::StructuredOutput,
            severity: ContractSeverity::Block,
            prompt: id_base.clone(),
            system_prompt: None,
            user_prompt: None,
            expectation: ContractExpectation::JsonSchema {
                schema: value.clone(),
            },
            provenance: ContractProvenance {
                source: ContractProvenanceSource::Discovery,
                file: Some(path_s.clone()),
                line_start: None,
                line_end: None,
                parser: Some("json_schema".into()),
                confidence: 0.8,
                auto_accepted: false,
                note: Some("Structured schema discovered — confirm severity".into()),
            },
            tags: vec!["discovered".into()],
        };
        report.hits.push(DiscoveryHit {
            kind: "structured_output".into(),
            path: path_s.clone(),
            line_start: None,
            line_end: None,
            parser: "json_schema".into(),
            confidence: 0.8,
            auto_accepted: false,
            summary: "JSON Schema / structured output".into(),
            proposed_item: Some(item),
        });
    }
    if let Some(tools) = value.get("tools").and_then(|t| t.as_array()) {
        for (i, tool) in tools.iter().enumerate() {
            let name = tool
                .pointer("/function/name")
                .or_else(|| tool.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("tool")
                .to_string();
            let item = ContractItem {
                id: format!("{}.tool_call.{:03}", stem(path), i + 1),
                kind: ContractItemKind::ToolCall,
                severity: ContractSeverity::Review,
                prompt: stem(path),
                system_prompt: None,
                user_prompt: None,
                expectation: ContractExpectation::ToolRequired {
                    tool_name: name.clone(),
                },
                provenance: ContractProvenance {
                    source: ContractProvenanceSource::Discovery,
                    file: Some(path_s.clone()),
                    line_start: None,
                    line_end: None,
                    parser: Some("openai_tools_json".into()),
                    confidence: 0.75,
                    auto_accepted: false,
                    note: None,
                },
                tags: vec!["discovered".into()],
            };
            report.hits.push(DiscoveryHit {
                kind: "tool_schema".into(),
                path: path_s.clone(),
                line_start: None,
                line_end: None,
                parser: "openai_tools_json".into(),
                confidence: 0.75,
                auto_accepted: false,
                summary: format!("Tool schema: {name}"),
                proposed_item: Some(item),
            });
        }
    }
    if value.get("messages").is_some() || value.get("system").is_some() {
        report.hits.push(DiscoveryHit {
            kind: "prompt".into(),
            path: path_s,
            line_start: None,
            line_end: None,
            parser: "openai_messages_json".into(),
            confidence: 0.6,
            auto_accepted: false,
            summary: "OpenAI-style message structure".into(),
            proposed_item: None,
        });
    }
    Ok(())
}

fn looks_like_json_schema(v: &serde_json::Value) -> bool {
    v.get("type").is_some()
        && (v.get("properties").is_some() || v.get("$schema").is_some() || v.get("required").is_some())
}

fn inspect_yaml_like(root: &Path, path: &Path, report: &mut DiscoveryReport) -> Result<()> {
    let text = fs::read_to_string(path).unwrap_or_default();
    if text.contains("system:") || text.contains("prompt:") || text.contains("messages:") {
        report.hits.push(DiscoveryHit {
            kind: "prompt".into(),
            path: rel(root, path),
            line_start: None,
            line_end: None,
            parser: "yaml_prompt".into(),
            confidence: 0.5,
            auto_accepted: false,
            summary: "YAML prompt-like definition".into(),
            proposed_item: None,
        });
    }
    Ok(())
}

fn inspect_toml_suite(root: &Path, path: &Path, report: &mut DiscoveryReport) -> Result<()> {
    let text = fs::read_to_string(path).unwrap_or_default();
    if text.contains("[[probe]]") || text.contains("[probe]") || text.contains("category") {
        report.hits.push(DiscoveryHit {
            kind: "behavioural_suite".into(),
            path: rel(root, path),
            line_start: None,
            line_end: None,
            parser: "arsenic_suite_toml".into(),
            confidence: 0.9,
            auto_accepted: false,
            summary: "Existing Arsenic probe suite".into(),
            proposed_item: None,
        });
    }
    Ok(())
}

fn inspect_python(root: &Path, path: &Path, report: &mut DiscoveryReport) -> Result<()> {
    let text = fs::read_to_string(path).unwrap_or_default();
    let sys_re = Regex::new(r#"(?i)(system_prompt|SYSTEM_PROMPT)\s*=\s*[\"']{1,3}"#).unwrap();
    for (i, line) in text.lines().enumerate() {
        if sys_re.is_match(line) {
            report.hits.push(DiscoveryHit {
                kind: "system_prompt".into(),
                path: rel(root, path),
                line_start: Some((i + 1) as u32),
                line_end: Some((i + 1) as u32),
                parser: "python_source".into(),
                confidence: 0.65,
                auto_accepted: false,
                summary: "System prompt assignment in Python".into(),
                proposed_item: None,
            });
        }
        if line.contains("tools=") || line.contains("\"tools\"") || line.contains("function_call")
        {
            report.hits.push(DiscoveryHit {
                kind: "tool_schema".into(),
                path: rel(root, path),
                line_start: Some((i + 1) as u32),
                line_end: Some((i + 1) as u32),
                parser: "python_source".into(),
                confidence: 0.45,
                auto_accepted: false,
                summary: "Possible tool definition in Python".into(),
                proposed_item: None,
            });
            break;
        }
    }
    Ok(())
}

fn inspect_js(root: &Path, path: &Path, report: &mut DiscoveryReport) -> Result<()> {
    let text = fs::read_to_string(path).unwrap_or_default();
    for (i, line) in text.lines().enumerate() {
        let l = line.to_lowercase();
        if l.contains("system:") || l.contains("role: \"system\"") || l.contains("role: 'system'")
        {
            report.hits.push(DiscoveryHit {
                kind: "system_prompt".into(),
                path: rel(root, path),
                line_start: Some((i + 1) as u32),
                line_end: Some((i + 1) as u32),
                parser: "js_ts_source".into(),
                confidence: 0.6,
                auto_accepted: false,
                summary: "System message in JS/TS".into(),
                proposed_item: None,
            });
        }
        if l.contains("parameters:") && (l.contains("type") || text.contains("\"type\": \"object\""))
        {
            report.hits.push(DiscoveryHit {
                kind: "tool_schema".into(),
                path: rel(root, path),
                line_start: Some((i + 1) as u32),
                line_end: Some((i + 1) as u32),
                parser: "js_ts_source".into(),
                confidence: 0.5,
                auto_accepted: false,
                summary: "Possible tool parameters schema".into(),
                proposed_item: None,
            });
            break;
        }
    }
    Ok(())
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("item")
        .replace([' ', '-'], "_")
}

/// Build a draft contract from accepted discovery hits (only auto_accepted or explicitly accepted ids).
pub fn contract_from_discovery(
    application_id: &str,
    application_name: &str,
    report: &DiscoveryReport,
    accept_all_proposals: bool,
) -> ApplicationContract {
    let mut contract = ApplicationContract::new(application_id, application_name);
    contract.source_project = Some(report.root.clone());
    for hit in &report.hits {
        if let Some(mut item) = hit.proposed_item.clone() {
            if accept_all_proposals {
                // Still never auto-accept block severity without confirmation flag clarity:
                // accept_all_proposals is an explicit user action.
                item.provenance.auto_accepted = true;
                contract.items.push(item);
            } else if hit.auto_accepted {
                contract.items.push(item);
            }
        }
    }
    contract
}

/// Merge an explicit hand-authored contract file (JSON) over discovery.
pub fn load_contract_file(path: &Path) -> Result<ApplicationContract> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

pub fn write_contract_file(path: &Path, contract: &ApplicationContract) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(contract)?;
    fs::write(path, text)?;
    Ok(())
}

pub fn default_contract_path(project_root: &Path) -> PathBuf {
    project_root.join(".arsenic").join("contract").join("contract.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_prompt_and_schema() {
        let dir = std::env::temp_dir().join(format!(
            "arsenic-discovery-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let prompts = dir.join("prompts");
        fs::create_dir_all(&prompts).unwrap();
        fs::write(prompts.join("system.txt"), "You are a support agent.\n").unwrap();
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "decision": { "type": "string" } },
            "required": ["decision"]
        });
        fs::write(
            dir.join("refund_schema.json"),
            serde_json::to_string_pretty(&schema).unwrap(),
        )
        .unwrap();
        let report = discover_project(&dir).unwrap();
        assert!(report.system_prompts >= 1);
        assert!(report.structured_schemas >= 1);
        assert!(report.hits.iter().all(|h| !h.auto_accepted || h.confidence >= 0.9));
        let _ = fs::remove_dir_all(&dir);
    }
}
