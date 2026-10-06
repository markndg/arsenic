//! Application contracts — behavioural expectations independent of any model.
//!
//! Contracts sit above the comparison engine. Each [`ContractItem`] is a stable,
//! severity-tagged requirement with provenance. Discovery may propose items;
//! critical expectations are never silently invented.

pub mod discovery;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const CONTRACT_SCHEMA_VERSION: u32 = 1;

/// Severity of a contract requirement.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ContractSeverity {
    Info,
    Warn,
    Review,
    Block,
}

impl Default for ContractSeverity {
    fn default() -> Self {
        Self::Review
    }
}

/// How a contract item was obtained.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContractProvenanceSource {
    Manual,
    Discovery,
    BaselineObservation,
    ExistingSuite,
    Imported,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContractProvenance {
    pub source: ContractProvenanceSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_start: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_end: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parser: Option<String>,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub auto_accepted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl Default for ContractProvenance {
    fn default() -> Self {
        Self {
            source: ContractProvenanceSource::Manual,
            file: None,
            line_start: None,
            line_end: None,
            parser: None,
            confidence: 1.0,
            auto_accepted: true,
            note: None,
        }
    }
}

/// Kind of behavioural expectation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContractItemKind {
    RequiredClaim,
    ForbiddenClaim,
    StructuredOutput,
    ToolCall,
    ToolArgument,
    Refusal,
    Instruction,
    OutputFormat,
    Semantic,
    Latency,
    Cost,
    Presentation,
}

/// Expectation payload for a contract item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContractExpectation {
    ContainsClaim {
        text: String,
    },
    ForbidsClaim {
        text: String,
    },
    JsonSchema {
        schema: serde_json::Value,
    },
    ToolRequired {
        tool_name: String,
    },
    ToolArgument {
        tool_name: String,
        argument: String,
        expected_type: String,
    },
    MustRefuse,
    MustAnswer,
    Instruction {
        description: String,
        #[serde(default)]
        check_contains: Option<String>,
    },
    OutputFormat {
        description: String,
    },
    SemanticSimilarity {
        min_cosine: f64,
    },
    LatencyBudget {
        max_ms: u64,
    },
    CostBudget {
        max_relative_delta_pct: f64,
    },
}

/// One stable behavioural requirement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContractItem {
    pub id: String,
    pub kind: ContractItemKind,
    pub severity: ContractSeverity,
    /// Logical prompt / feature name this item applies to.
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_prompt: Option<String>,
    pub expectation: ContractExpectation,
    pub provenance: ContractProvenance,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// First-class application behavioural contract.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApplicationContract {
    pub schema_version: u32,
    pub application_id: String,
    pub application_name: String,
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_project: Option<String>,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    pub items: Vec<ContractItem>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl ApplicationContract {
    pub fn new(application_id: impl Into<String>, application_name: impl Into<String>) -> Self {
        Self {
            schema_version: CONTRACT_SCHEMA_VERSION,
            application_id: application_id.into(),
            application_name: application_name.into(),
            version: 1,
            source_project: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: None,
            items: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Deterministic content hash over schema, application id, version, and items.
    pub fn content_hash(&self) -> String {
        let mut hasher = Sha256::new();
        let payload = serde_json::json!({
            "schema_version": self.schema_version,
            "application_id": self.application_id,
            "version": self.version,
            "items": self.items,
        });
        hasher.update(serde_json::to_vec(&payload).unwrap_or_default());
        format!("{:x}", hasher.finalize())
    }

    pub fn item(&self, id: &str) -> Option<&ContractItem> {
        self.items.iter().find(|i| i.id == id)
    }

    pub fn blocking_items(&self) -> impl Iterator<Item = &ContractItem> {
        self.items
            .iter()
            .filter(|i| matches!(i.severity, ContractSeverity::Block))
    }
}

/// Diff between two contract versions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ContractDiff {
    pub from_version: u32,
    pub to_version: u32,
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub severity_changed: Vec<SeverityChange>,
    pub modified: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SeverityChange {
    pub id: String,
    pub from: ContractSeverity,
    pub to: ContractSeverity,
}

/// Compare two contracts by item id.
pub fn diff_contracts(from: &ApplicationContract, to: &ApplicationContract) -> ContractDiff {
    let mut diff = ContractDiff {
        from_version: from.version,
        to_version: to.version,
        ..Default::default()
    };
    let from_ids: BTreeMap<_, _> = from.items.iter().map(|i| (i.id.as_str(), i)).collect();
    let to_ids: BTreeMap<_, _> = to.items.iter().map(|i| (i.id.as_str(), i)).collect();

    for (id, item) in &to_ids {
        match from_ids.get(id) {
            None => diff.added.push((*id).to_string()),
            Some(prev) => {
                if prev.severity != item.severity {
                    diff.severity_changed.push(SeverityChange {
                        id: (*id).to_string(),
                        from: prev.severity,
                        to: item.severity,
                    });
                }
                if prev.expectation != item.expectation
                    || prev.prompt != item.prompt
                    || prev.kind != item.kind
                {
                    diff.modified.push((*id).to_string());
                }
            }
        }
    }
    for id in from_ids.keys() {
        if !to_ids.contains_key(id) {
            diff.removed.push((*id).to_string());
        }
    }
    diff.added.sort();
    diff.removed.sort();
    diff.modified.sort();
    diff.severity_changed.sort_by(|a, b| a.id.cmp(&b.id));
    diff
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_item(id: &str, severity: ContractSeverity) -> ContractItem {
        ContractItem {
            id: id.into(),
            kind: ContractItemKind::RequiredClaim,
            severity,
            prompt: "refund_decision".into(),
            system_prompt: None,
            user_prompt: Some("Should we refund?".into()),
            expectation: ContractExpectation::ContainsClaim {
                text: "Refunds over £500 require manager approval".into(),
            },
            provenance: ContractProvenance::default(),
            tags: vec![],
        }
    }

    #[test]
    fn content_hash_is_stable() {
        let mut a = ApplicationContract::new("customer-support", "Customer Support");
        a.created_at = "fixed".into();
        a.items.push(sample_item("refund.001", ContractSeverity::Block));
        let mut b = a.clone();
        assert_eq!(a.content_hash(), b.content_hash());
        b.items[0].severity = ContractSeverity::Warn;
        assert_ne!(a.content_hash(), b.content_hash());
    }

    #[test]
    fn diff_detects_add_remove_severity() {
        let mut v1 = ApplicationContract::new("app", "App");
        v1.version = 1;
        v1.items.push(sample_item("a", ContractSeverity::Block));
        v1.items.push(sample_item("b", ContractSeverity::Review));
        let mut v2 = v1.clone();
        v2.version = 2;
        v2.items.retain(|i| i.id != "b");
        v2.items.push(sample_item("c", ContractSeverity::Warn));
        v2.items[0].severity = ContractSeverity::Review;
        let d = diff_contracts(&v1, &v2);
        assert_eq!(d.added, vec!["c".to_string()]);
        assert_eq!(d.removed, vec!["b".to_string()]);
        assert_eq!(d.severity_changed.len(), 1);
        assert_eq!(d.severity_changed[0].id, "a");
    }
}
