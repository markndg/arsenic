//! Application contract / qualification HTML report.
//!
//! Migration recommendations are derived exclusively via
//! [`arsenic_core::assess_all`] / [`arsenic_core::aggregate_migration`].

use arsenic_core::{
    aggregate_migration, assess_all, assess_qualification, ApplicationContract, ArsenicLock,
    BaselineSnapshot, EvidenceValidity, MigrationRecommendation, QualificationResult,
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

    let contract_hash = contract.content_hash();
    let baseline_hash = baseline.map(|b| b.content_hash()).unwrap_or_default();
    let effectives = assess_all(quals, &contract_hash, &baseline_hash);
    let migration = aggregate_migration(&effectives);
    let migration_text = migration.as_str();

    // Current (non-stale) candidates for the matrix; stale still listed separately.
    let mut models: Vec<String> = effectives
        .iter()
        .filter(|e| !e.is_stale)
        .map(|e| e.candidate_model.clone())
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
            let q = quals
                .iter()
                .filter(|q| &q.candidate_model == m)
                .max_by(|a, b| {
                    a.created_at
                        .cmp(&b.created_at)
                        .then_with(|| a.id.cmp(&b.id))
                });
            let cell = match q {
                None => serde_json::json!({"status": "—", "detail": null}),
                Some(q) => {
                    let eff = assess_qualification(q, &contract_hash, &baseline_hash);
                    if eff.is_stale {
                        serde_json::json!({"status": "STALE", "detail": null})
                    } else if eff.evidence_validity == EvidenceValidity::Incomplete {
                        serde_json::json!({"status": "INCOMPLETE", "detail": null})
                    } else {
                        let relevant: Vec<_> = q
                            .item_results
                            .iter()
                            .filter(|r| {
                                r.regression_type.as_deref() == Some(reg)
                                    || format!("{:?}", r.kind)
                                        .to_lowercase()
                                        .contains(&reg.replace('_', ""))
                            })
                            .collect();
                        if relevant.is_empty() {
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
            let q = quals
                .iter()
                .filter(|q| &q.candidate_model == m)
                .max_by(|a, b| {
                    a.created_at
                        .cmp(&b.created_at)
                        .then_with(|| a.id.cmp(&b.id))
                });
            let status = q
                .and_then(|q| {
                    let eff = assess_qualification(q, &contract_hash, &baseline_hash);
                    if eff.is_stale || eff.evidence_validity == EvidenceValidity::Incomplete {
                        None
                    } else {
                        q.cost_latency.cost_delta_pct
                    }
                })
                .map(|p| format!("{p:+.0}%"))
                .unwrap_or_else(|| "n/a".into());
            cells.push(serde_json::json!({"status": status, "detail": null}));
        }
        row.insert("cells".into(), serde_json::json!(cells));
        matrix.push(serde_json::Value::Object(row));
    }

    // Enrich qualifications for the template with effective fields.
    let quals_view: Vec<serde_json::Value> = {
        let mut sorted: Vec<_> = quals.iter().collect();
        sorted.sort_by(|a, b| {
            a.created_at
                .cmp(&b.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });
        sorted
            .into_iter()
            .map(|q| {
                let eff = assess_qualification(q, &contract_hash, &baseline_hash);
                let mut v = serde_json::to_value(q).unwrap_or(serde_json::json!({}));
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("is_stale".into(), serde_json::json!(eff.is_stale));
                    obj.insert(
                        "evidence_validity".into(),
                        serde_json::json!(eff.evidence_validity),
                    );
                    obj.insert(
                        "effective_decision".into(),
                        serde_json::json!(eff.effective_decision),
                    );
                    obj.insert(
                        "migration_recommendation".into(),
                        serde_json::json!(eff.migration_recommendation.as_str()),
                    );
                }
                v
            })
            .collect()
    };

    let migration_class = match migration {
        MigrationRecommendation::SafeToMigrate => "safe",
        MigrationRecommendation::MigrationBlocked => "blocked",
        MigrationRecommendation::Incomplete => "incomplete",
        MigrationRecommendation::Stale => "stale",
        _ => "review",
    };

    let mut ctx = TeraContext::new();
    ctx.insert("contract", contract);
    ctx.insert("baseline", &baseline);
    ctx.insert("lock", &lock);
    ctx.insert("qualifications", &quals_view);
    ctx.insert("effectives", &effectives);
    ctx.insert("models", &models);
    ctx.insert("matrix", &matrix);
    ctx.insert("migration", migration_text);
    ctx.insert("migration_class", migration_class);
    ctx.insert("requirement_count", &contract.items.len());

    Ok(tera.render("application.html", &ctx)?)
}
