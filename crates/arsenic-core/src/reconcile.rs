//! Types for the `reconcile` single-prompt certification flow.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::{
    AnchorDrift, DriftDirection, ModelInfo, MutationStrategy, ProbeResult, RiskLevel,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconcileResult {
    pub run_id: Uuid,
    pub generated_at: DateTime<Utc>,
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub v1_model: ModelInfo,
    pub v2_model: ModelInfo,
    pub v1_response: String,
    pub v2_response: String,
    pub delta: ProbeResult,
    pub signals: Vec<ReconcileSignal>,
    pub attempts: Vec<ReconcileAttempt>,
    pub certified: bool,
    pub certified_prompt: Option<String>,
    pub strategies_applied: Vec<MutationStrategy>,
    pub validation_risk: Option<RiskLevel>,
    pub validation_response: Option<String>,
    pub requires_manual_review: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconcileSignal {
    pub dimension: ReconcileDimension,
    pub magnitude: f64,
    pub direction: DriftDirection,
    pub detail: SignalDetail,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum ReconcileDimension {
    Claim,
    Morphology,
    Tone,
    Factual,
    Schema,
    Refusal,
    Semantic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalDetail {
    DroppedClaims {
        anchors: Vec<String>,
        count: usize,
    },
    MorphologyDelta {
        token_delta_pct: f64,
        response_type_changed: bool,
        v2_shorter: bool,
    },
    ToneDelta {
        formality_delta: f64,
        assertiveness_delta: f64,
        v2_less_formal: bool,
        v2_over_hedged: bool,
    },
    RefusalNew,
    SchemaInvalid {
        missing_fields: Vec<String>,
    },
    FactualRegression {
        v1_answer: String,
        v2_answer: String,
    },
    SemanticDrift {
        similarity: f64,
    },
    AnchorDrift {
        drifted: Vec<AnchorDrift>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconcileAttempt {
    pub attempt_number: usize,
    pub mutated_prompt: String,
    pub strategies: Vec<MutationStrategy>,
    pub v2_response: String,
    pub risk_after: RiskLevel,
    pub improved: bool,
}
