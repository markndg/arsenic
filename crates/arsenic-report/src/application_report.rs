//! Application contract / qualification HTML report.
//!
//! Migration recommendations are derived exclusively via
//! [`arsenic_core::assess_all`] / [`arsenic_core::aggregate_migration`].
//! The template must never reconstruct migration safety independently.

use arsenic_core::{
    aggregate_migration, assess_all, assess_qualification, ApplicationContract, ArsenicLock,
    BaselineSnapshot, EvidenceValidity, ItemOutcome, MigrationRecommendation, QualificationResult,
};
use tera::{Context as TeraContext, Tera};

const APPLICATION_TEMPLATE: &str = include_str!("../../../report-templates/application.html.tera");

fn abbrev_hash(h: &str) -> String {
    if h.is_empty() {
        return "—".into();
    }
    let take = h.chars().take(12).collect::<String>();
    if h.len() > 12 {
        format!("{take}…")
    } else {
        take
    }
}

fn overall_evidence_label(
    migration: MigrationRecommendation,
    effectives: &[arsenic_core::EffectiveQualification],
) -> &'static str {
    match migration {
        MigrationRecommendation::Incomplete => "INCOMPLETE",
        MigrationRecommendation::Stale | MigrationRecommendation::NoEvidence => {
            if effectives.iter().any(|e| e.is_stale) {
                "STALE"
            } else {
                "—"
            }
        }
        _ => {
            if effectives
                .iter()
                .any(|e| e.evidence_validity == EvidenceValidity::Incomplete)
            {
                "INCOMPLETE"
            } else if effectives.iter().all(|e| e.is_stale) && !effectives.is_empty() {
                "STALE"
            } else if effectives.iter().any(|e| e.is_stale)
                && effectives
                    .iter()
                    .any(|e| e.evidence_validity == EvidenceValidity::Valid)
            {
                "MIXED"
            } else {
                "VALID"
            }
        }
    }
}

fn migration_class(migration: MigrationRecommendation) -> &'static str {
    match migration {
        MigrationRecommendation::SafeToMigrate => "safe",
        MigrationRecommendation::MigrationBlocked => "blocked",
        MigrationRecommendation::Incomplete => "incomplete",
        MigrationRecommendation::Stale => "stale",
        MigrationRecommendation::NoEvidence => "none",
        MigrationRecommendation::ReviewRequired => "review",
    }
}

fn outcome_status(outcome: ItemOutcome, severity: arsenic_core::ContractSeverity) -> &'static str {
    match outcome {
        ItemOutcome::Pass => "PASS",
        ItemOutcome::Warn => "WARN",
        ItemOutcome::Review => "REVIEW",
        ItemOutcome::Fail if matches!(severity, arsenic_core::ContractSeverity::Block) => "BLOCK",
        ItemOutcome::Fail => "REVIEW",
    }
}

/// Build the candidate × requirement-kind matrix (latest qualification per model).
fn build_matrix(
    quals: &[QualificationResult],
    models: &[String],
    contract_hash: &str,
    baseline_hash: &str,
) -> Vec<serde_json::Value> {
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
        for m in models {
            let q = quals
                .iter()
                .filter(|q| &q.candidate_model == m)
                .max_by(|a, b| {
                    a.created_at
                        .cmp(&b.created_at)
                        .then_with(|| a.id.cmp(&b.id))
                });
            let cell = match q {
                None => serde_json::json!({
                    "status": "—",
                    "detail": null,
                    "title": "No qualification evidence for this candidate"
                }),
                Some(q) => {
                    let eff = assess_qualification(q, contract_hash, baseline_hash);
                    if eff.is_stale {
                        serde_json::json!({
                            "status": "STALE",
                            "detail": null,
                            "title": eff.stale_reasons.join("; ")
                        })
                    } else if eff.evidence_validity == EvidenceValidity::Incomplete {
                        serde_json::json!({
                            "status": "INCOMPLETE",
                            "detail": null,
                            "title": eff.incomplete_reasons.join("; ")
                        })
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
                            serde_json::json!({
                                "status": "—",
                                "detail": null,
                                "title": "No requirement of this kind"
                            })
                        } else if relevant.iter().any(|r| {
                            matches!(r.outcome, ItemOutcome::Fail)
                                && matches!(r.severity, arsenic_core::ContractSeverity::Block)
                        }) {
                            let r = relevant
                                .iter()
                                .find(|r| {
                                    matches!(r.outcome, ItemOutcome::Fail)
                                        && matches!(
                                            r.severity,
                                            arsenic_core::ContractSeverity::Block
                                        )
                                })
                                .unwrap();
                            serde_json::json!({
                                "status": "BLOCK",
                                "detail": r,
                                "title": r.reason
                            })
                        } else if relevant
                            .iter()
                            .any(|r| matches!(r.outcome, ItemOutcome::Fail | ItemOutcome::Review))
                        {
                            let r = relevant
                                .iter()
                                .find(|r| {
                                    matches!(r.outcome, ItemOutcome::Fail | ItemOutcome::Review)
                                })
                                .unwrap();
                            serde_json::json!({
                                "status": "REVIEW",
                                "detail": r,
                                "title": r.reason
                            })
                        } else if relevant
                            .iter()
                            .any(|r| matches!(r.outcome, ItemOutcome::Warn))
                        {
                            let r = relevant
                                .iter()
                                .find(|r| matches!(r.outcome, ItemOutcome::Warn))
                                .unwrap();
                            serde_json::json!({
                                "status": "WARN",
                                "detail": r,
                                "title": r.reason
                            })
                        } else {
                            serde_json::json!({
                                "status": "PASS",
                                "detail": relevant[0],
                                "title": "Requirement satisfied"
                            })
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
        row.insert("label".into(), serde_json::json!("cost Δ"));
        let mut cells = Vec::new();
        for m in models {
            let q = quals
                .iter()
                .filter(|q| &q.candidate_model == m)
                .max_by(|a, b| {
                    a.created_at
                        .cmp(&b.created_at)
                        .then_with(|| a.id.cmp(&b.id))
                });
            let (status, title) = match q {
                None => ("—".into(), "No evidence".into()),
                Some(q) => {
                    let eff = assess_qualification(q, contract_hash, baseline_hash);
                    if eff.is_stale {
                        ("STALE".into(), eff.stale_reasons.join("; "))
                    } else if eff.evidence_validity == EvidenceValidity::Incomplete {
                        ("INCOMPLETE".into(), eff.incomplete_reasons.join("; "))
                    } else {
                        let s = q
                            .cost_latency
                            .cost_delta_pct
                            .map(|p| format!("{p:+.0}%"))
                            .unwrap_or_else(|| "n/a".into());
                        (s, "Estimated cost delta vs production baseline".into())
                    }
                }
            };
            cells.push(serde_json::json!({
                "status": status,
                "detail": null,
                "title": title
            }));
        }
        row.insert("cells".into(), serde_json::json!(cells));
        matrix.push(serde_json::Value::Object(row));
    }

    matrix
}

fn enrich_qualification(
    q: &QualificationResult,
    contract_hash: &str,
    baseline_hash: &str,
    baseline_model: Option<&str>,
    contract_version: u32,
) -> serde_json::Value {
    let eff = assess_qualification(q, contract_hash, baseline_hash);
    let mut v = serde_json::to_value(q).unwrap_or(serde_json::json!({}));
    let Some(obj) = v.as_object_mut() else {
        return v;
    };

    let is_historical = eff.is_stale
        || eff.evidence_validity == EvidenceValidity::Incomplete
        || !eff.migration_recommendation.allows_safe_migrate();

    let certifies_migration = eff.is_currently_passing();

    obj.insert("is_stale".into(), serde_json::json!(eff.is_stale));
    obj.insert(
        "evidence_validity".into(),
        serde_json::json!(eff.evidence_validity.as_str()),
    );
    obj.insert(
        "recorded_decision".into(),
        serde_json::json!(eff.recorded_decision.as_str()),
    );
    obj.insert(
        "effective_decision".into(),
        serde_json::json!(eff.effective_decision.as_str()),
    );
    obj.insert(
        "migration_recommendation".into(),
        serde_json::json!(eff.migration_recommendation.as_str()),
    );
    obj.insert(
        "migration_class".into(),
        serde_json::json!(migration_class(eff.migration_recommendation)),
    );
    obj.insert("stale_reasons".into(), serde_json::json!(eff.stale_reasons));
    obj.insert(
        "incomplete_reasons".into(),
        serde_json::json!(eff.incomplete_reasons),
    );
    obj.insert(
        "patch_applies_to_current".into(),
        serde_json::json!(eff.patch_applies_to_current),
    );
    obj.insert(
        "has_validated_patch".into(),
        serde_json::json!(eff.has_validated_patch),
    );
    obj.insert(
        "contract_hash_short".into(),
        serde_json::json!(abbrev_hash(&eff.contract_hash)),
    );
    obj.insert(
        "baseline_hash_short".into(),
        serde_json::json!(abbrev_hash(&eff.baseline_hash)),
    );
    obj.insert(
        "current_contract_hash_short".into(),
        serde_json::json!(abbrev_hash(&eff.current_contract_hash)),
    );
    obj.insert(
        "current_baseline_hash_short".into(),
        serde_json::json!(abbrev_hash(&eff.current_baseline_hash)),
    );
    obj.insert(
        "input_fingerprint_short".into(),
        serde_json::json!(eff
            .input_fingerprint
            .as_deref()
            .map(abbrev_hash)
            .unwrap_or_else(|| "—".into())),
    );
    obj.insert("is_historical".into(), serde_json::json!(is_historical));
    obj.insert(
        "certifies_migration".into(),
        serde_json::json!(certifies_migration),
    );
    obj.insert(
        "baseline_model".into(),
        serde_json::json!(baseline_model.unwrap_or("—")),
    );
    obj.insert(
        "contract_version".into(),
        serde_json::json!(contract_version),
    );
    obj.insert("blockers".into(), serde_json::json!(eff.blockers));
    obj.insert("review_items".into(), serde_json::json!(eff.review_items));

    // Classify item results for template presentation.
    let mut item_views = Vec::new();
    for r in &q.item_results {
        let status = outcome_status(r.outcome, r.severity);
        let evidence_quality = r.regression_type.as_deref() == Some("execution_error");
        item_views.push(serde_json::json!({
            "item_id": r.item_id,
            "kind": r.kind,
            "severity": format!("{:?}", r.severity).to_lowercase(),
            "outcome": format!("{:?}", r.outcome).to_lowercase(),
            "status": status,
            "reason": r.reason,
            "regression_type": r.regression_type,
            "is_execution_failure": evidence_quality,
            "baseline_evidence": r.baseline_evidence,
            "candidate_evidence": r.candidate_evidence,
            "comparator": r.comparator,
        }));
    }
    obj.insert("item_views".into(), serde_json::json!(item_views));

    v
}

fn executive_counts(
    contract: &ApplicationContract,
    effectives: &[arsenic_core::EffectiveQualification],
    quals_latest: &[&QualificationResult],
) -> serde_json::Value {
    let mut passed = 0usize;
    let mut warnings = 0usize;
    let mut review = 0usize;
    let mut blockers = 0usize;
    for q in quals_latest {
        // Only count behavioural item outcomes from current (non-stale, non-incomplete) quals.
        let eff = effectives.iter().find(|e| e.qualification_id == q.id);
        if eff
            .map(|e| e.is_stale || e.evidence_validity == EvidenceValidity::Incomplete)
            .unwrap_or(true)
        {
            continue;
        }
        for r in &q.item_results {
            if r.regression_type.as_deref() == Some("execution_error") {
                continue;
            }
            match r.outcome {
                ItemOutcome::Pass => passed += 1,
                ItemOutcome::Warn => warnings += 1,
                ItemOutcome::Review => review += 1,
                ItemOutcome::Fail
                    if matches!(r.severity, arsenic_core::ContractSeverity::Block) =>
                {
                    blockers += 1;
                }
                ItemOutcome::Fail => review += 1,
            }
        }
    }

    let stale_quals = effectives.iter().filter(|e| e.is_stale).count();
    let incomplete_quals = effectives
        .iter()
        .filter(|e| e.evidence_validity == EvidenceValidity::Incomplete)
        .count();
    let validated_repairs = effectives
        .iter()
        .filter(|e| e.has_validated_patch && e.patch_applies_to_current)
        .count();
    let stale_repairs = effectives
        .iter()
        .filter(|e| e.has_validated_patch && !e.patch_applies_to_current)
        .count();

    serde_json::json!({
        "requirements": contract.items.len(),
        "passed": passed,
        "warnings": warnings,
        "review_items": review,
        "blockers": blockers,
        "stale_qualifications": stale_quals,
        "incomplete_qualifications": incomplete_quals,
        "validated_repairs": validated_repairs,
        "stale_repairs": stale_repairs,
        "candidates": effectives.len(),
    })
}

fn contract_requirement_rows(
    contract: &ApplicationContract,
    quals: &[QualificationResult],
    models: &[String],
    contract_hash: &str,
    baseline_hash: &str,
) -> Vec<serde_json::Value> {
    let mut rows = Vec::new();
    for item in &contract.items {
        let mut results = Vec::new();
        for m in models {
            let q = quals
                .iter()
                .filter(|q| &q.candidate_model == m)
                .max_by(|a, b| {
                    a.created_at
                        .cmp(&b.created_at)
                        .then_with(|| a.id.cmp(&b.id))
                });
            let cell = match q {
                None => serde_json::json!({
                    "model": m,
                    "status": "—",
                    "reason": "No qualification"
                }),
                Some(q) => {
                    let eff = assess_qualification(q, contract_hash, baseline_hash);
                    if eff.is_stale {
                        serde_json::json!({
                            "model": m,
                            "status": "STALE",
                            "reason": eff.stale_reasons.first().cloned().unwrap_or_default()
                        })
                    } else if eff.evidence_validity == EvidenceValidity::Incomplete {
                        serde_json::json!({
                            "model": m,
                            "status": "INCOMPLETE",
                            "reason": eff.incomplete_reasons.first().cloned().unwrap_or_default()
                        })
                    } else if let Some(r) = q.item_results.iter().find(|r| r.item_id == item.id) {
                        serde_json::json!({
                            "model": m,
                            "status": outcome_status(r.outcome, r.severity),
                            "reason": r.reason,
                            "is_execution_failure": r.regression_type.as_deref() == Some("execution_error"),
                        })
                    } else {
                        serde_json::json!({
                            "model": m,
                            "status": "—",
                            "reason": "Not evaluated"
                        })
                    }
                }
            };
            results.push(cell);
        }
        rows.push(serde_json::json!({
            "id": item.id,
            "kind": format!("{:?}", item.kind),
            "severity": format!("{:?}", item.severity).to_lowercase(),
            "prompt": item.prompt,
            "provenance_source": format!("{:?}", item.provenance.source).to_lowercase(),
            "provenance_file": item.provenance.file,
            "results": results,
        }));
    }
    rows
}

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

    // Latest per candidate — include stale/incomplete so matrix can show them.
    let mut models: Vec<String> = effectives
        .iter()
        .map(|e| e.candidate_model.clone())
        .collect();
    models.sort();
    models.dedup();

    let latest_refs: Vec<&QualificationResult> = models
        .iter()
        .filter_map(|m| {
            quals
                .iter()
                .filter(|q| &q.candidate_model == m)
                .max_by(|a, b| {
                    a.created_at
                        .cmp(&b.created_at)
                        .then_with(|| a.id.cmp(&b.id))
                })
        })
        .collect();

    let matrix = build_matrix(quals, &models, &contract_hash, &baseline_hash);

    let baseline_model = baseline.map(|b| b.model.as_str());
    let mut quals_view: Vec<serde_json::Value> = {
        let mut sorted: Vec<_> = quals.iter().collect();
        sorted.sort_by(|a, b| {
            // Current certifying first, then by candidate, then newest first.
            let ea = assess_qualification(a, &contract_hash, &baseline_hash);
            let eb = assess_qualification(b, &contract_hash, &baseline_hash);
            eb.is_currently_passing()
                .cmp(&ea.is_currently_passing())
                .then_with(|| a.candidate_model.cmp(&b.candidate_model))
                .then_with(|| b.created_at.cmp(&a.created_at))
                .then_with(|| b.id.cmp(&a.id))
        });
        sorted
            .into_iter()
            .map(|q| {
                enrich_qualification(
                    q,
                    &contract_hash,
                    &baseline_hash,
                    baseline_model,
                    contract.version,
                )
            })
            .collect()
    };

    // Stable secondary sort key already applied; ensure deterministic id order for ties.
    quals_view.sort_by(|a, b| {
        let ca = a
            .get("certifies_migration")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let cb = b
            .get("certifies_migration")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        cb.cmp(&ca).then_with(|| {
            let ma = a
                .get("candidate_model")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let mb = b
                .get("candidate_model")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            ma.cmp(mb)
        })
    });

    let candidates_label = match models.as_slice() {
        [] => "—".to_string(),
        [one] => one.clone(),
        many => format!("{} candidates", many.len()),
    };

    let evidence_label = overall_evidence_label(migration, &effectives);
    let exec = executive_counts(contract, &effectives, &latest_refs);
    let requirements =
        contract_requirement_rows(contract, quals, &models, &contract_hash, &baseline_hash);

    let decision_explainer = match migration {
        MigrationRecommendation::SafeToMigrate => {
            "Current, complete evidence supports migrating this application to the candidate model(s)."
        }
        MigrationRecommendation::MigrationBlocked => {
            "Blocking contract requirements failed. Do not migrate until blockers are resolved and re-qualified."
        }
        MigrationRecommendation::ReviewRequired => {
            "Unresolved review findings remain. Migration needs human review before approval."
        }
        MigrationRecommendation::Incomplete => {
            "Arsenic could not obtain sufficient evidence to decide. This is not a behavioural PASS or BLOCK."
        }
        MigrationRecommendation::Stale => {
            "Available qualifications no longer match the current contract and/or baseline. Requalify before migrating."
        }
        MigrationRecommendation::NoEvidence => {
            "No qualification evidence is available for this application."
        }
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
    ctx.insert("migration_class", migration_class(migration));
    ctx.insert("decision_explainer", decision_explainer);
    ctx.insert("requirement_count", &contract.items.len());
    ctx.insert("candidates_label", &candidates_label);
    ctx.insert("evidence_label", evidence_label);
    ctx.insert("exec", &exec);
    ctx.insert("requirements", &requirements);
    ctx.insert("contract_hash", &contract_hash);
    ctx.insert("baseline_hash", &baseline_hash);
    ctx.insert("contract_hash_short", &abbrev_hash(&contract_hash));
    ctx.insert("baseline_hash_short", &abbrev_hash(&baseline_hash));

    Ok(tera.render("application.html", &ctx)?)
}
