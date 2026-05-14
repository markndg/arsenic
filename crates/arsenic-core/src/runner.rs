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
