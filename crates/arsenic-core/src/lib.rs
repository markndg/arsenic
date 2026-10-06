//! ARSENIC core: probe/response types, drift dimensions, comparison engine, and probe runner.

pub mod adapter;
pub mod cache;
pub mod category_infer;
pub mod claim;
pub mod code_equivalence;
pub mod comparison;
pub mod contract;
pub mod embedding;
pub mod error;
pub mod fingerprint;
pub mod impact;
pub mod materiality;
pub mod morphology;
pub mod mutation;
pub mod project;
pub mod qualify;
pub mod reconcile;
pub mod reconcile_engine;
pub mod refusal;
pub mod runner;
pub mod semantic;
pub mod tone;
pub mod types;

pub use contract::discovery::{
    contract_from_discovery, discover_project, load_contract_file, write_contract_file,
    DiscoveryHit, DiscoveryReport,
};
pub use contract::{
    diff_contracts, ApplicationContract, ContractDiff, ContractExpectation, ContractItem,
    ContractItemKind, ContractProvenance, ContractProvenanceSource, ContractSeverity,
    SeverityChange, CONTRACT_SCHEMA_VERSION,
};
pub use project::{
    count_stale_qualifications, filter_candidates, load_baseline, load_config, load_contract,
    load_current_baseline, load_fixture_behaviours, load_lock, load_patch, load_qualification,
    list_qualifications, mark_stale_qualifications, next_baseline_id, next_qualification_id,
    parse_model_spec, save_baseline, save_config, save_contract, save_lock, save_patch,
    save_qualification, update_lock_from_qualification, write_fixture_behaviours, ArsenicLock,
    ArsenicProjectConfig, CandidateEntry, LockBlocked, LockProduction, LockQualified, ProjectPaths,
};
pub use qualify::{
    attempt_validated_repair, behaviour_from_error, behaviour_from_response, build_impact,
    candidate_input_fingerprint, capability_failures, capture_contract_live,
    classify_provider_error, compatibility_summary_pct, content_hash, decide_overall,
    evaluate_item, extract_tool_calls, json_hash, qualify_candidate, redact_json, redact_secrets,
    scenarios_from_contract, BaselineSnapshot, CapturedBehaviour, CapturedToolCall, CaptureSource,
    ContractItemResult, CostLatencyDelta, EvidenceSnippet, ExecutionErrorKind, ExecutionMeta,
    ImpactReport, ItemOutcome, LiveCaptureConfig, ProviderCapabilities, QualificationDecision,
    QualificationResult, QualificationThresholds, RepairAttempt, ValidatedPatch,
    QUALIFICATION_SCHEMA_VERSION,
};
pub use cache::{
    corpus_fingerprint, BaselineCache, BaselineManifest, BaselineModel, BaselineProbeEntry,
    CacheKey, CachedResponse, CachedRun, VerifyReport, CACHE_SCHEMA_VERSION,
};
pub use category_infer::infer_probe_category;
pub use code_equivalence::{
    compare_code_equivalence, extract_code_body, is_code_context, is_code_probe,
    looks_like_code_content, non_code_prose, prompt_requires_code_only, CodeEquivalence,
};
pub use comparison::{
    compute_latency_summary, compute_migration_profile, compute_probe_risk, dimension_severity,
    ComparisonEngine, RiskThresholds,
};
pub use embedding::{embed_batch_hash, hash_embed, weighted_sentence_similarity};
pub use error::ArsenicError;
pub use fingerprint::{
    build_fingerprint_changelog, build_fingerprint_svg, compute_behaviour_fingerprint,
    consistency_retention, fingerprint_summary_mismatches, retention_score, validate_fingerprint,
    validate_fingerprint_rollups, variance_to_repeatability, BehaviourFingerprint,
    FingerprintAggregationKind, FingerprintAxis, FingerprintChangelog, FingerprintConfidence,
    FingerprintInterpretation, FingerprintSvgModel, FingerprintSvgSpoke, OmittedAxisReason,
    OmittedFingerprintAxis, RollupValidationError, FINGERPRINT_VERSION, MIN_AXES_FOR_RADAR,
};
pub use impact::{assess_probe_impact, dimension_impact_label, ProbeImpactAssessment};
pub use materiality::{
    assess_consistency_materiality, claim_materially_changed, consistency_materially_changed,
    factual_materially_changed, instruction_materially_changed, morphology_crosses_material_band,
    morphology_materially_changed, refusal_materially_changed, schema_materially_changed,
    semantic_materially_changed, tone_materially_changed, ConsistencyMateriality,
    FingerprintObservation, CONSISTENCY_HIGH_IMPACT_RETENTION_BELOW,
    DEFAULT_MORPHOLOGY_TOKEN_DELTA_AMBER,
};
pub use morphology::MorphologyAnalyser;
pub use mutation::{apply_mutations, propose_strategies};
pub use reconcile::{
    ReconcileAttempt, ReconcileDimension, ReconcileResult, ReconcileSignal, SignalDetail,
};
pub use reconcile_engine::{
    build_reconcile_probe, expand_strategies_for_attempts, extract_coverage_topics, rank_signals,
    run_reconcile, signals_to_strategies, synthetic_model_response, DEFAULT_MAX_STRATEGIES,
};
pub use refusal::RefusalDetector;
pub use runner::ProbeRunner;
pub use semantic::SemanticAnalyser;
pub use tone::ToneAnalyser;
pub use types::*;
pub use adapter::ModelAdapter;
