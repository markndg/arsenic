use std::collections::HashMap;
use std::time::Duration;

use anyhow::Context;
use arsenic_core::{
    apply_mutations, propose_strategies, ComparisonEngine, ModelAdapter, ModelResponse,
    MutationResult, MutationStrategy, Probe, ProbeResult, ResponsePair, RiskLevel,
};
use uuid::Uuid;

fn risk_rank(r: &RiskLevel) -> u8 {
    match r {
        RiskLevel::Green => 0,
        RiskLevel::Amber => 1,
        RiskLevel::Red => 2,
    }
}

fn risk_improved(before: &RiskLevel, after: &RiskLevel) -> bool {
    risk_rank(after) < risk_rank(before)
}

/// Call the adapter with bounded exponential backoff retries. Returns `Err`
/// only after every attempt has failed, so callers can decide whether to skip
/// the probe rather than abort the whole run.
async fn complete_with_retry(
    adapter: &dyn ModelAdapter,
    probe: &Probe,
    attempts: usize,
    delay_ms: u64,
) -> Result<ModelResponse, String> {
    let attempts = attempts.max(1);
    let mut last_err = String::new();
    for attempt in 0..attempts {
        match adapter.complete(probe).await {
            Ok(r) => return Ok(r),
            Err(e) => {
                last_err = e.to_string();
                if attempt + 1 < attempts {
                    let backoff = delay_ms.saturating_mul(1u64 << attempt).min(30_000);
                    tokio::time::sleep(Duration::from_millis(backoff)).await;
                }
            }
        }
    }
    Err(last_err)
}

pub async fn validate_mutations_for_report(
    engine: &ComparisonEngine,
    baseline: &[ProbeResult],
    pairs_by_id: &HashMap<Uuid, ResponsePair>,
    v2: &dyn ModelAdapter,
    retry_attempts: usize,
    retry_delay_ms: u64,
) -> anyhow::Result<Vec<MutationResult>> {
    let mut out = Vec::new();
    for pr in baseline {
        let inst_reg = pr
            .dimensions
            .instruction
            .as_ref()
            .map(|i| i.regressions.as_slice())
            .unwrap_or(&[]);
        let strategies = propose_strategies(
            &pr.probe,
            &pr.dimensions.morphology,
            &pr.dimensions.tone,
            &pr.dimensions.refusal,
            &pr.dimensions.claim,
            inst_reg,
        );
        if strategies.is_empty() {
            continue;
        }

        let pair = pairs_by_id
            .get(&pr.probe.id)
            .with_context(|| format!("missing response pair for probe {}", pr.probe.name))?;

        let baseline_risk = &pr.overall_risk;
        let mut validated = false;
        let mut validation_risk = baseline_risk.clone();
        let mut strategies_applied: Vec<MutationStrategy> = Vec::new();
        let mut notes = String::new();
        let mut last_mutated = pr.probe.prompt.clone();
        let mut failed = 0usize;
        let mut transient_failure: Option<String> = None;

        // `failed` and `i` intentionally diverge: an early break (validated /
        // transient_failure) does not increment `failed`.
        #[allow(clippy::explicit_counter_loop)]
        for i in 0..strategies.len() {
            if failed >= 3 {
                notes.push_str("Stopped after 3 unsuccessful cumulative attempts.");
                break;
            }
            let trial: Vec<_> = strategies[..=i].to_vec();
            last_mutated = apply_mutations(&pr.probe, &trial);
            strategies_applied = trial;
            let mut probe2 = pr.probe.clone();
            probe2.prompt = last_mutated.clone();

            let v2_resp =
                match complete_with_retry(v2, &probe2, retry_attempts, retry_delay_ms).await {
                    Ok(r) => r,
                    Err(e) => {
                        transient_failure = Some(e);
                        break;
                    }
                };

            let trial_pair = ResponsePair {
                probe: pr.probe.clone(),
                v1: pair.v1.clone(),
                v2: v2_resp,
                v1_runs: Vec::new(),
                v2_runs: Vec::new(),
            };
            let new_pr = engine
                .compare_one(trial_pair)
                .with_context(|| format!("compare mutated pair {}", pr.probe.name))?;
            validation_risk = new_pr.overall_risk.clone();
            if risk_improved(baseline_risk, &validation_risk) {
                validated = true;
                notes.push_str("Cumulative strategies reduced overall risk.");
                break;
            }
            failed += 1;
        }

        if let Some(err) = transient_failure {
            if !notes.is_empty() {
                notes.push(' ');
            }
            notes.push_str(&format!(
                "v2 request failed after {} attempt(s) during mutation validation: {}. Marked for manual review.",
                retry_attempts.max(1),
                err
            ));
        }

        out.push(MutationResult {
            probe_name: pr.probe.name.clone(),
            original_prompt: pr.probe.prompt.clone(),
            mutated_prompt: last_mutated,
            strategies_applied,
            validated,
            validation_risk,
            requires_manual_review: !validated,
            notes,
        });
    }
    Ok(out)
}
