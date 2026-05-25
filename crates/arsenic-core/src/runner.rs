use std::sync::Arc;
use std::time::Duration;

use futures::stream::{FuturesUnordered, StreamExt};
use tokio::sync::Semaphore;

use crate::adapter::ModelAdapter;
use crate::types::{FinishReason, ModelResponse, Probe, ResponsePair};
use uuid::Uuid;

pub struct ProbeRunner {
    pub v1_adapter: Arc<dyn ModelAdapter>,
    pub v2_adapter: Arc<dyn ModelAdapter>,
    pub v1_label: String,
    pub v2_label: String,
    pub concurrency: usize,
    pub retry_attempts: usize,
    pub retry_delay_ms: u64,
    /// v2: number of identical completions per probe per endpoint (1 = v1 behaviour, no extra runs).
    pub consistency_runs: usize,
}

impl ProbeRunner {
    pub async fn run(&self, probes: Vec<Probe>) -> anyhow::Result<Vec<ResponsePair>> {
        let sem = Arc::new(Semaphore::new(self.concurrency.max(1)));
        let v1 = Arc::clone(&self.v1_adapter);
        let v2 = Arc::clone(&self.v2_adapter);
        let v1_label = self.v1_label.clone();
        let v2_label = self.v2_label.clone();
        let attempts = self.retry_attempts.max(1);
        let delay = self.retry_delay_ms;
        let n_runs = self.consistency_runs.max(1);

        let mut unordered = FuturesUnordered::new();
        for probe in probes {
            let sem = Arc::clone(&sem);
            let v1 = Arc::clone(&v1);
            let v2 = Arc::clone(&v2);
            let v1l = v1_label.clone();
            let v2l = v2_label.clone();
            unordered.push(async move {
                let _permit = sem.acquire_owned().await.expect("semaphore");
                let mut v1_runs = Vec::with_capacity(n_runs);
                let mut v2_runs = Vec::with_capacity(n_runs);
                for _ in 0..n_runs {
                    v1_runs.push(complete_with_retry(&*v1, &probe, &v1l, attempts, delay).await);
                    v2_runs.push(complete_with_retry(&*v2, &probe, &v2l, attempts, delay).await);
                }
                let v1 = v1_runs[0].clone();
                let v2 = v2_runs[0].clone();
                let (v1_runs, v2_runs) = if n_runs <= 1 {
                    (Vec::new(), Vec::new())
                } else {
                    (v1_runs, v2_runs)
                };
                ResponsePair {
                    probe,
                    v1,
                    v2,
                    v1_runs,
                    v2_runs,
                }
            });
        }
        let mut pairs: Vec<ResponsePair> = Vec::new();
        while let Some(pair) = unordered.next().await {
            pairs.push(pair);
        }
        pairs.sort_by(|a, b| a.probe.name.cmp(&b.probe.name));
        Ok(pairs)
    }
}

async fn complete_with_retry(
    adapter: &dyn ModelAdapter,
    probe: &Probe,
    label: &str,
    attempts: usize,
    delay_ms: u64,
) -> ModelResponse {
    let mut last_err = String::new();
    for attempt in 0..attempts {
        match adapter.complete(probe).await {
            Ok(mut r) => {
                r.model_label = label.to_string();
                return r;
            }
            Err(e) => {
                last_err = e.to_string();
                let backoff = delay_ms * (1u64 << attempt);
                tokio::time::sleep(Duration::from_millis(backoff.min(30_000))).await;
            }
        }
    }
    error_response(probe.id, label, adapter, &last_err)
}

fn error_response(probe_id: Uuid, label: &str, adapter: &dyn ModelAdapter, err: &str) -> ModelResponse {
    ModelResponse {
        probe_id,
        model_label: label.to_string(),
        model_id: adapter.model_id().to_string(),
        content: format!("ERROR: {err}"),
        token_count: 0,
        latency_ms: 0,
        finish_reason: FinishReason::Error,
        timestamp: chrono::Utc::now(),
        raw: serde_json::json!({ "error": err }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ProbeCategory, ProbeSource};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Adapter that succeeds on the Nth call (1-indexed) and errors before that.
    struct FlakyAdapter {
        calls: Arc<AtomicUsize>,
        succeed_on: usize,
    }

    #[async_trait]
    impl ModelAdapter for FlakyAdapter {
        async fn complete(&self, probe: &Probe) -> anyhow::Result<ModelResponse> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if n < self.succeed_on {
                return Err(anyhow::anyhow!("transient flake on attempt {n}"));
            }
            Ok(ModelResponse {
                probe_id: probe.id,
                // Intentionally a "wrong" label here so we can prove the runner overwrites it.
                model_label: "internal".into(),
                model_id: "flaky-model".into(),
                content: format!("ok on attempt {n}"),
                token_count: 3,
                latency_ms: 10,
                finish_reason: FinishReason::Stop,
                timestamp: chrono::Utc::now(),
                raw: serde_json::json!({}),
            })
        }
        fn model_id(&self) -> &str {
            "flaky-model"
        }
        fn adapter_name(&self) -> &str {
            "flaky"
        }
        fn endpoint(&self) -> &str {
            "flaky://"
        }
    }

    fn mk_probe() -> Probe {
        Probe {
            id: Uuid::new_v4(),
            name: "t".into(),
            category: ProbeCategory::Factual,
            prompt: "p".into(),
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

    #[tokio::test]
    async fn complete_with_retry_succeeds_after_transient_failures() {
        let calls = Arc::new(AtomicUsize::new(0));
        let adapter = FlakyAdapter {
            calls: Arc::clone(&calls),
            succeed_on: 3,
        };
        let probe = mk_probe();
        // delay_ms 0 to keep the test fast.
        let r = complete_with_retry(&adapter, &probe, "v1", 5, 0).await;
        assert!(!matches!(r.finish_reason, FinishReason::Error));
        assert_eq!(r.model_label, "v1", "label must be rewritten by the runner");
        assert!(r.content.contains("ok on attempt 3"));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn complete_with_retry_returns_synthetic_error_after_exhaustion() {
        let calls = Arc::new(AtomicUsize::new(0));
        let adapter = FlakyAdapter {
            calls: Arc::clone(&calls),
            succeed_on: 999, // never succeeds within attempts
        };
        let probe = mk_probe();
        let r = complete_with_retry(&adapter, &probe, "v2", 3, 0).await;
        assert!(matches!(r.finish_reason, FinishReason::Error));
        assert_eq!(r.model_id, "flaky-model", "error must record adapter model id");
        assert_eq!(r.model_label, "v2");
        assert!(r.content.starts_with("ERROR:"), "got: {}", r.content);
        assert!(
            r.raw.get("error").is_some(),
            "synthetic error must include the cause in `raw.error`"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 3, "attempts must be exhausted");
    }

    #[tokio::test]
    async fn complete_with_retry_zero_attempts_floored_to_one() {
        // The runner's `attempts.max(1)` clamps zero up to one. This test
        // proves a `0` attempts argument doesn't loop infinitely or skip.
        let calls = Arc::new(AtomicUsize::new(0));
        let adapter = FlakyAdapter {
            calls: Arc::clone(&calls),
            succeed_on: 1,
        };
        let probe = mk_probe();
        // Call complete_with_retry with attempts=1 (the runner's effective floor).
        let r = complete_with_retry(&adapter, &probe, "v1", 1, 0).await;
        assert!(!matches!(r.finish_reason, FinishReason::Error));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
