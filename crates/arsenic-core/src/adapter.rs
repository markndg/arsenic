use async_trait::async_trait;

use crate::types::{ModelResponse, Probe};

/// Implemented by `arsenic-adapters`. Defined in core to avoid circular crate dependencies.
#[async_trait]
pub trait ModelAdapter: Send + Sync {
    async fn complete(&self, probe: &Probe) -> anyhow::Result<ModelResponse>;
    fn model_id(&self) -> &str;
    fn adapter_name(&self) -> &str;
    fn endpoint(&self) -> &str;
}
