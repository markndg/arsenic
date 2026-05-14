use thiserror::Error;

#[derive(Debug, Error)]
pub enum ArsenicError {
    #[error("Adapter error for model {model_id}: {source}")]
    AdapterError {
        model_id: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("Probe {probe_name} failed after {attempts} attempts")]
    ProbeExhausted { probe_name: String, attempts: usize },

    #[error("Invalid probe corpus at {path}: {reason}")]
    InvalidCorpus { path: String, reason: String },

    #[error("Embedding model not found at {path}")]
    EmbeddingModelNotFound { path: String },

    #[error("Report rendering failed: {source}")]
    ReportError {
        #[source]
        source: anyhow::Error,
    },
}
