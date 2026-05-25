//! Cache-aware wrapper around any [`ModelAdapter`].
//!
//! [`CachingAdapter`] takes an optional `inner` live adapter plus a
//! [`BaselineCache`] and a [`CacheMode`]. Per-key call counters track which
//! cached run to return when a probe is invoked multiple times (consistency
//! runs). Errors are never cached; the wrapper degrades gracefully on retry.
//!
//! Composition is intentional: the runner sees a plain `dyn ModelAdapter` and
//! is unchanged. Caching is invisible at the call site.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use arsenic_core::{
    cache::{BaselineCache, CacheKey},
    types::{FinishReason, ModelResponse, Probe},
    ModelAdapter,
};
use async_trait::async_trait;

/// Behaviour of a [`CachingAdapter`] at the time it handles a probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    /// Cache hit → return. Cache miss → live call. Successful results are
    /// appended to the cache. Errors are not cached.
    ReadWrite,
    /// Cache hit → return. Cache miss → error. No live calls.
    ReadOnly,
    /// Always make a live call. Append successful results to the cache.
    WriteOnly,
    /// Always make a live call. Do not touch the cache.
    Bypass,
}

/// Identity fields used to compute the cache key and to satisfy
/// `ModelAdapter::{model_id, endpoint, adapter_name}`. In [`CacheMode::ReadOnly`]
/// these are typically lifted from the baseline manifest so the wrapper can
/// answer identity queries without a working inner adapter.
#[derive(Debug, Clone)]
pub struct BaselineIdentity {
    pub adapter_type: String,
    pub endpoint: String,
    pub model_id: String,
    pub temperature: f64,
    pub max_tokens: Option<usize>,
}

pub struct CachingAdapter {
    /// Live adapter for cache misses. Must be `Some` for any mode that may
    /// dispatch a live call (`ReadWrite`, `WriteOnly`, `Bypass`).
    inner: Option<Arc<dyn ModelAdapter>>,
    cache: Arc<BaselineCache>,
    mode: CacheMode,
    identity: BaselineIdentity,
    /// Per-key call counter: the *n*th successful `complete` for a given key
    /// returns `runs[n - 1]`. Cache misses and errors do not increment.
    counters: Mutex<HashMap<String, usize>>,
}

impl CachingAdapter {
    pub fn new(
        inner: Option<Arc<dyn ModelAdapter>>,
        cache: Arc<BaselineCache>,
        mode: CacheMode,
        identity: BaselineIdentity,
    ) -> Self {
        Self {
            inner,
            cache,
            mode,
            identity,
            counters: Mutex::new(HashMap::new()),
        }
    }

    fn build_key(&self, probe: &Probe) -> CacheKey {
        CacheKey::new(
            &self.identity.adapter_type,
            &self.identity.endpoint,
            &self.identity.model_id,
            self.identity.temperature,
            self.identity.max_tokens,
            probe,
        )
    }

    fn pending_index(&self, key_hash: &str) -> usize {
        let guard = self.counters.lock().expect("counter mutex");
        *guard.get(key_hash).unwrap_or(&0)
    }

    fn commit(&self, key_hash: &str) {
        let mut guard = self.counters.lock().expect("counter mutex");
        let entry = guard.entry(key_hash.to_string()).or_insert(0);
        *entry += 1;
    }

    async fn complete_live(&self, probe: &Probe) -> Result<ModelResponse> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| anyhow!("cache mode requires a live adapter but none was provided"))?;
        inner.complete(probe).await
    }
}

#[async_trait]
impl ModelAdapter for CachingAdapter {
    async fn complete(&self, probe: &Probe) -> Result<ModelResponse> {
        let key = self.build_key(probe);
        let key_hash = key.hash();

        match self.mode {
            CacheMode::Bypass => self.complete_live(probe).await,

            CacheMode::ReadOnly => {
                let pending = self.pending_index(&key_hash);
                let cached = self
                    .cache
                    .read_one(&key_hash)
                    .with_context(|| format!("read cache for probe {}", probe.name))?;
                match cached {
                    Some(c) if pending < c.runs.len() => {
                        let resp = c.runs[pending].to_response(
                            probe.id,
                            "v1",
                            &self.identity.model_id,
                        );
                        self.commit(&key_hash);
                        Ok(resp)
                    }
                    Some(c) => Err(anyhow!(
                        "baseline has {} cached run(s) for probe {}, but the runner requested run #{}",
                        c.runs.len(),
                        probe.name,
                        pending + 1
                    )),
                    None => Err(anyhow!(
                        "cache miss for probe {} (key {})",
                        probe.name,
                        &key_hash[..16]
                    )),
                }
            }

            CacheMode::ReadWrite => {
                let pending = self.pending_index(&key_hash);
                if let Some(c) = self
                    .cache
                    .read_one(&key_hash)
                    .with_context(|| format!("read cache for probe {}", probe.name))?
                {
                    if pending < c.runs.len() {
                        let resp =
                            c.runs[pending].to_response(probe.id, "v1", &self.identity.model_id);
                        self.commit(&key_hash);
                        return Ok(resp);
                    }
                }
                // Cache miss (or runs exhausted): live call, then append.
                let live = self.complete_live(probe).await?;
                if !matches!(live.finish_reason, FinishReason::Error) {
                    if !self.cache.is_locked() {
                        self.cache.append_run(&key, &live, probe)?;
                    }
                    self.commit(&key_hash);
                }
                Ok(live)
            }

            CacheMode::WriteOnly => {
                let live = self.complete_live(probe).await?;
                if !matches!(live.finish_reason, FinishReason::Error) {
                    if !self.cache.is_locked() {
                        self.cache.append_run(&key, &live, probe)?;
                    }
                    self.commit(&key_hash);
                }
                Ok(live)
            }
        }
    }

    fn model_id(&self) -> &str {
        &self.identity.model_id
    }

    fn adapter_name(&self) -> &str {
        &self.identity.adapter_type
    }

    fn endpoint(&self) -> &str {
        &self.identity.endpoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arsenic_core::types::{ProbeCategory, ProbeSource};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use uuid::Uuid;

    fn tmp_root(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "arsenic-caching-test-{label}-{}",
            Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn mk_probe(name: &str, prompt: &str) -> Probe {
        Probe {
            id: Uuid::new_v4(),
            name: name.into(),
            category: ProbeCategory::Factual,
            prompt: prompt.into(),
            system_prompt: None,
            known_answer: None,
            expected_schema: None,
            instructions: vec![],
            tags: vec![],
            source: ProbeSource::Standard,
            expected_verbosity: None,
            expected_tone: None,
            refusal_expectation: None,
            mutation_hint: None,
            custom_assertions: vec![],
        }
    }

    fn identity() -> BaselineIdentity {
        BaselineIdentity {
            adapter_type: "openai".into(),
            endpoint: "https://api.openai.com/v1".into(),
            model_id: "gpt-4o-mini".into(),
            temperature: 0.0,
            max_tokens: None,
        }
    }

    /// Stub adapter that returns a counter-stamped response and tracks calls.
    struct StubAdapter {
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    #[async_trait]
    impl ModelAdapter for StubAdapter {
        async fn complete(&self, probe: &Probe) -> Result<ModelResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(anyhow!("stub failure"));
            }
            let n = self.calls.load(Ordering::SeqCst);
            Ok(ModelResponse {
                probe_id: probe.id,
                model_label: "stub".into(),
                model_id: "stub-model".into(),
                content: format!("response-{n}"),
                token_count: 1,
                latency_ms: 10,
                finish_reason: FinishReason::Stop,
                timestamp: chrono::Utc::now(),
                raw: serde_json::json!({}),
            })
        }
        fn model_id(&self) -> &str {
            "stub-model"
        }
        fn adapter_name(&self) -> &str {
            "stub"
        }
        fn endpoint(&self) -> &str {
            "stub://"
        }
    }

    #[tokio::test]
    async fn write_only_records_to_cache_and_calls_live() {
        let root = tmp_root("wo");
        let cache = Arc::new(BaselineCache::new(root.join("b")));
        let calls = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(StubAdapter {
            calls: Arc::clone(&calls),
            fail: false,
        });
        let wrap = CachingAdapter::new(Some(inner), Arc::clone(&cache), CacheMode::WriteOnly, identity());
        let p = mk_probe("p", "hi");

        let r = wrap.complete(&p).await.unwrap();
        assert_eq!(r.content, "response-1");
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let key = CacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            0.0,
            None,
            &p,
        );
        let read = cache.read_one(&key.hash()).unwrap().expect("present");
        assert_eq!(read.runs.len(), 1);
        assert_eq!(read.runs[0].content, "response-1");
    }

    #[tokio::test]
    async fn read_only_returns_cached_run_without_live_call() {
        let root = tmp_root("ro");
        let cache = Arc::new(BaselineCache::new(root.join("b")));
        let p = mk_probe("p", "hi");
        let key = CacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            0.0,
            None,
            &p,
        );
        let pre = ModelResponse {
            probe_id: p.id,
            model_label: "v1".into(),
            model_id: "gpt-4o-mini".into(),
            content: "cached".into(),
            token_count: 1,
            latency_ms: 10,
            finish_reason: FinishReason::Stop,
            timestamp: chrono::Utc::now(),
            raw: serde_json::json!({}),
        };
        cache.append_run(&key, &pre, &p).unwrap();

        // Inner adapter explicitly None; any live call would panic with the
        // helpful error in complete_live.
        let wrap =
            CachingAdapter::new(None, Arc::clone(&cache), CacheMode::ReadOnly, identity());
        let r = wrap.complete(&p).await.unwrap();
        assert_eq!(r.content, "cached");
    }

    #[tokio::test]
    async fn read_only_misses_are_an_error() {
        let root = tmp_root("ro-miss");
        let cache = Arc::new(BaselineCache::new(root.join("b")));
        let wrap =
            CachingAdapter::new(None, Arc::clone(&cache), CacheMode::ReadOnly, identity());
        let p = mk_probe("p", "uncached");
        let err = wrap.complete(&p).await;
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("cache miss"), "got: {msg}");
    }

    #[tokio::test]
    async fn read_only_consistency_runs_walk_through_cached_array() {
        let root = tmp_root("ro-runs");
        let cache = Arc::new(BaselineCache::new(root.join("b")));
        let p = mk_probe("p", "hi");
        let key = CacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            0.0,
            None,
            &p,
        );
        for i in 0..3 {
            let r = ModelResponse {
                probe_id: p.id,
                model_label: "v1".into(),
                model_id: "gpt-4o-mini".into(),
                content: format!("run-{i}"),
                token_count: 1,
                latency_ms: 10,
                finish_reason: FinishReason::Stop,
                timestamp: chrono::Utc::now(),
                raw: serde_json::json!({}),
            };
            cache.append_run(&key, &r, &p).unwrap();
        }
        let wrap =
            CachingAdapter::new(None, Arc::clone(&cache), CacheMode::ReadOnly, identity());
        for i in 0..3 {
            let r = wrap.complete(&p).await.unwrap();
            assert_eq!(r.content, format!("run-{i}"));
        }
        // 4th call exhausts the cached runs.
        let err = wrap.complete(&p).await;
        assert!(err.is_err());
        assert!(format!("{}", err.unwrap_err()).contains("cached run"));
    }

    #[tokio::test]
    async fn read_only_failed_reads_do_not_advance_counter() {
        let root = tmp_root("ro-retry");
        let cache = Arc::new(BaselineCache::new(root.join("b")));
        let wrap =
            CachingAdapter::new(None, Arc::clone(&cache), CacheMode::ReadOnly, identity());
        let p = mk_probe("p", "miss");
        // Three failed reads in a row (e.g. retry loop) must not bump the
        // counter past 0.
        for _ in 0..3 {
            let _ = wrap.complete(&p).await;
        }
        let key = CacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            0.0,
            None,
            &p,
        );
        // After populating runs[0], the next call must succeed (counter is 0,
        // not 3).
        let resp = ModelResponse {
            probe_id: p.id,
            model_label: "v1".into(),
            model_id: "gpt-4o-mini".into(),
            content: "fresh".into(),
            token_count: 1,
            latency_ms: 10,
            finish_reason: FinishReason::Stop,
            timestamp: chrono::Utc::now(),
            raw: serde_json::json!({}),
        };
        cache.append_run(&key, &resp, &p).unwrap();
        let ok = wrap.complete(&p).await.unwrap();
        assert_eq!(ok.content, "fresh");
    }

    #[tokio::test]
    async fn read_write_hits_cache_after_first_warm_up() {
        let root = tmp_root("rw");
        let cache = Arc::new(BaselineCache::new(root.join("b")));
        let calls = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(StubAdapter {
            calls: Arc::clone(&calls),
            fail: false,
        });
        let wrap = CachingAdapter::new(
            Some(inner),
            Arc::clone(&cache),
            CacheMode::ReadWrite,
            identity(),
        );
        let p = mk_probe("p", "hi");

        // First call: cache miss, live call, write.
        let r1 = wrap.complete(&p).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let r1_content = r1.content.clone();

        // Fresh wrapper sharing the same on-disk cache.
        let calls2 = Arc::new(AtomicUsize::new(0));
        let inner2 = Arc::new(StubAdapter {
            calls: Arc::clone(&calls2),
            fail: false,
        });
        let wrap2 = CachingAdapter::new(
            Some(inner2),
            Arc::clone(&cache),
            CacheMode::ReadWrite,
            identity(),
        );
        let r2 = wrap2.complete(&p).await.unwrap();
        assert_eq!(
            calls2.load(Ordering::SeqCst),
            0,
            "second invocation must hit the cache"
        );
        assert_eq!(r2.content, r1_content);
    }

    #[tokio::test]
    async fn live_failure_is_not_cached() {
        let root = tmp_root("fail");
        let cache = Arc::new(BaselineCache::new(root.join("b")));
        let calls = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(StubAdapter {
            calls: Arc::clone(&calls),
            fail: true,
        });
        let wrap = CachingAdapter::new(
            Some(inner),
            Arc::clone(&cache),
            CacheMode::WriteOnly,
            identity(),
        );
        let p = mk_probe("p", "hi");
        let err = wrap.complete(&p).await;
        assert!(err.is_err());

        let key = CacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            0.0,
            None,
            &p,
        );
        assert!(cache.read_one(&key.hash()).unwrap().is_none());
    }

    #[tokio::test]
    async fn write_only_on_locked_cache_still_returns_live_but_does_not_write() {
        let root = tmp_root("wo-locked");
        let cache = Arc::new(BaselineCache::new(root.join("b")));
        // Pre-create a manifest so lock() has something to update.
        let manifest = arsenic_core::cache::BaselineManifest {
            name: "b".into(),
            created_at: chrono::Utc::now(),
            arsenic_version: "test".into(),
            model: arsenic_core::cache::BaselineModel {
                adapter_type: "openai".into(),
                endpoint: "https://api.openai.com/v1".into(),
                model_id: "gpt-4o-mini".into(),
                temperature: 0.0,
                max_tokens: None,
            },
            corpus_fingerprint: "sha256:0".into(),
            probes: vec![],
            consistency_runs: 1,
            locked: false,
            notes: None,
            created_by: None,
        };
        cache.write_manifest(&manifest).unwrap();
        cache.lock().unwrap();

        let calls = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(StubAdapter {
            calls: Arc::clone(&calls),
            fail: false,
        });
        let wrap = CachingAdapter::new(
            Some(inner),
            Arc::clone(&cache),
            CacheMode::WriteOnly,
            identity(),
        );
        let p = mk_probe("p", "hi");

        // Live call still succeeds (lock only gates writes, not reads/calls).
        let r = wrap.complete(&p).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(r.content.starts_with("response-"));

        // But nothing was cached, because the baseline is locked.
        let key = CacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            0.0,
            None,
            &p,
        );
        assert!(
            cache.read_one(&key.hash()).unwrap().is_none(),
            "locked baseline must not gain new entries"
        );
    }

    #[tokio::test]
    async fn bypass_never_writes_to_cache() {
        let root = tmp_root("bypass");
        let cache = Arc::new(BaselineCache::new(root.join("b")));
        let calls = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(StubAdapter {
            calls: Arc::clone(&calls),
            fail: false,
        });
        let wrap = CachingAdapter::new(
            Some(inner),
            Arc::clone(&cache),
            CacheMode::Bypass,
            identity(),
        );
        let p = mk_probe("p", "hi");
        let _ = wrap.complete(&p).await.unwrap();
        let key = CacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            0.0,
            None,
            &p,
        );
        assert!(cache.read_one(&key.hash()).unwrap().is_none());
    }
}
