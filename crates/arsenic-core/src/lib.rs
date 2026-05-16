//! ARSENIC core: probe/response types, drift dimensions, comparison engine, and probe runner.

pub mod adapter;
pub mod category_infer;
pub mod claim;
pub mod comparison;
pub mod embedding;
pub mod error;
pub mod morphology;
pub mod mutation;
pub mod reconcile;
pub mod reconcile_engine;
pub mod refusal;
pub mod runner;
pub mod semantic;
pub mod tone;
pub mod types;

pub use adapter::ModelAdapter;
pub use claim::{ClaimExtractor, ClaimMatcher};
pub use comparison::{
    compute_latency_summary, compute_migration_profile, compute_probe_risk, dimension_severity,
    ComparisonEngine, RiskThresholds,
};
pub use embedding::{embed_batch_hash, hash_embed, weighted_sentence_similarity};
pub use error::ArsenicError;
pub use morphology::MorphologyAnalyser;
pub use category_infer::infer_probe_category;
pub use mutation::{apply_mutations, propose_strategies};
pub use reconcile::{ReconcileAttempt, ReconcileDimension, ReconcileResult, ReconcileSignal, SignalDetail};
pub use reconcile_engine::{
    build_reconcile_probe, expand_strategies_for_attempts, extract_coverage_topics, rank_signals,
    run_reconcile, signals_to_strategies, synthetic_model_response, DEFAULT_MAX_STRATEGIES,
};
pub use refusal::RefusalDetector;
pub use runner::ProbeRunner;
pub use semantic::SemanticAnalyser;
pub use tone::ToneAnalyser;
pub use types::*;
