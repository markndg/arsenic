//! ARSENIC core: probe/response types, drift dimensions, comparison engine, and probe runner.

pub mod adapter;
pub mod claim;
pub mod comparison;
pub mod embedding;
pub mod error;
pub mod morphology;
pub mod mutation;
pub mod refusal;
pub mod runner;
pub mod semantic;
pub mod tone;
pub mod types;

pub use adapter::ModelAdapter;
pub use claim::{ClaimExtractor, ClaimMatcher};
pub use comparison::{ComparisonEngine, RiskThresholds};
pub use embedding::{embed_batch_hash, hash_embed, weighted_sentence_similarity};
pub use error::ArsenicError;
pub use morphology::MorphologyAnalyser;
pub use mutation::{apply_mutations, propose_strategies};
pub use refusal::RefusalDetector;
pub use runner::ProbeRunner;
pub use semantic::SemanticAnalyser;
pub use tone::ToneAnalyser;
pub use types::*;
