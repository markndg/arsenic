//! HTTP adapters for OpenAI-compatible, Anthropic, and Google APIs.

mod anthropic;
mod google;
mod openai;
mod registry;

pub use anthropic::AnthropicAdapter;
pub use google::GoogleAdapter;
pub use openai::OpenAIAdapter;
pub use registry::{build_adapter, AdapterSpec};
