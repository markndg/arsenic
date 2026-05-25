//! HTTP adapters for OpenAI-compatible, Anthropic, and Google APIs, plus a
//! cache-aware [`CachingAdapter`] wrapper for baseline replay.

mod anthropic;
mod caching;
mod google;
mod openai;
mod registry;

pub use anthropic::AnthropicAdapter;
pub use caching::{BaselineIdentity, CacheMode, CachingAdapter};
pub use google::GoogleAdapter;
pub use openai::OpenAIAdapter;
pub use registry::{build_adapter, AdapterSpec};
