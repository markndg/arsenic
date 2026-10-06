//! Application contract / qualification HTML report.

use arsenic_core::{
    ApplicationContract, ArsenicLock, BaselineSnapshot, QualificationDecision, QualificationResult,
};
use tera::{Context as TeraContext, Tera};

const APPLICATION_TEMPLATE: &str = include_str!("../../../report-templates/application.html.tera");

pub fn render_application_report(
    contract: &ApplicationContract,
    baseline: Option<&BaselineSnapshot>,
    quals: &[QualificationResult],
    lock: Option<&ArsenicLock>,
) -> anyhow::Result<String> {
    let mut tera = Tera::default();
    tera.add_raw_template("application.html", APPLICATION_TEMPLATE)?;

    // Build matrix rows: kind -> model -> outcome
    let mut models: Vec<String> = quals
        .iter()
        .filter(|q| !q.stale)
        .map(|q| q.candidate_model.clone())
        .collect();
    models.sort();
    models.dedup();

    let kinds = [
        ("claims", "required_claim"),
        ("tool arguments", "tool_argument"),
        ("structure", "structured_output"),
        ("instructions", "instruction"),
        ("refusals", "refusal"),
        ("presentation", "presentation"),
    ];

    let mut matrix: Vec<serde_json::Value> = Vec::new();
    for (label, reg) in kinds {
        let mut row = serde_json::Map::new();
        row.insert("label".into(), serde_json::json!(label));
        let mut cells = Vec::new();
        for m in &models {
            let q = quals.iter().rev().find(|q| &q.candidate_model == m && !q.stale);
            let cell = match q {
                None => serde_json::json!({"status": "—", "detail": null}),
                Some(q) => {
                    let relevant: Vec<_> = q
                        .item_results
                        .iter()
                        .filter(|r| r.regression_type.as_deref() == Some(reg) || {
                            // also match by kind name loosely
                            format!("{:?}", r.kind).to_lowercase().contains(&reg.replace('_', ""))
                        })
                        .collect();
                    if relevant.is_empty() {
                        // cost row special-cased below
                        serde_json::json!({"status": "—", "detail": null})
                    } else if relevant.iter().any(|r| {
                        matches!(r.outcome, arsenic_core::ItemOutcome::Fail)
                            && matches!(r.severity, arsenic_core::ContractSeverity::Block)
                    }) {
                        serde_json::json!({"status": "BLOCK", "detail": relevant[0]})
                    } else if relevant.iter().any(|r| {
                        matches!(
                            r.outcome,
                            arsenic_core::ItemOutcome::Fail | arsenic_core::ItemOutcome::Review
                        )
                    }) {
                        serde_json::json!({"status": "REVIEW", "detail": relevant[0]})
                    } else if relevant
                        .iter()
                        .any(|r| matches!(r.outcome, arsenic_core::ItemOutcome::Warn))
                    {
                        serde_json::json!({"status": "WARN", "detail": relevant[0]})
                    } else {
                        serde_json::json!({"status": "PASS", "detail": relevant[0]})
                    }
                }
            };
            cells.push(cell);
        }
        row.insert("cells".into(), serde_json::json!(cells));
        matrix.push(serde_json::Value::Object(row));
    }

    // Cost row
    {
        let mut row = serde_json::Map::new();
        row.insert("label".into(), serde_json::json!("cost"));
        let mut cells = Vec::new();
        for m in &models {
            let q = quals.iter().rev().find(|q| &q.candidate_model == m && !q.stale);
            let status = q
                .and_then(|q| q.cost_latency.cost_delta_pct)
                .map(|p| format!("{p:+.0}%"))
                .unwrap_or_else(|| "n/a".into());
            cells.push(serde_json::json!({"status": status, "detail": null}));
        }
        row.insert("cells".into(), serde_json::json!(cells));
        matrix.push(serde_json::Value::Object(row));
    }

    let migration = if quals.iter().any(|q| q.decision == QualificationDecision::Block && !q.stale) {
        "MIGRATION BLOCKED"
    } else if quals
        .iter()
        .any(|q| matches!(q.decision, QualificationDecision::Review) && !q.stale)
    {
        "REVIEW REQUIRED"
    } else if models.is_empty() {
        "NO CANDIDATES"
    } else {
        "SAFE TO MIGRATE"
    };

    let mut ctx = TeraContext::new();
    ctx.insert("contract", contract);
    ctx.insert("baseline", &baseline);
    ctx.insert("lock", &lock);
    ctx.insert("qualifications", quals);
    ctx.insert("models", &models);
    ctx.insert("matrix", &matrix);
    ctx.insert("migration", migration);
    ctx.insert(
        "requirement_count",
        &contract.items.len(),
    );

    Ok(tera.render("application.html", &ctx)?)
}
