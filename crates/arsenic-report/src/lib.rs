//! HTML / JSON / Markdown report rendering.

mod reconcile_report;

use anyhow::Context;
use arsenic_core::{
    build_fingerprint_changelog, build_fingerprint_svg, validate_fingerprint, DriftReport,
};
use serde_json::{json, Value};
use tera::{Context as TeraContext, Tera};

pub use reconcile_report::{reconcile_json_value, render_reconcile_html, render_reconcile_json};

pub struct ReportRenderer;

/// Syncs valence from `probe_results`, serializes the report, then mirrors drift valence into
/// `summary.regressions` / `improvements` / `neutral` (same values as `probe_*`).
///
/// External scripts often expect the short names; Rust and the HTML templates use `probe_*`.
/// Also attaches derived `fingerprint_svg` and `fingerprint_changelog` view models for HTML.
pub fn drift_report_json_value(report: &DriftReport) -> anyhow::Result<Value> {
    let mut report = report.clone();
    // Source of truth: probe results → summaries + fingerprint (one recompute).
    // Stale derived fields in saved JSON never override probe results.
    // A second recompute cannot resolve a persistent semantic mismatch.
    report.rebuild_rollups();
    report.sync_valence_from_probe_results();
    report.upgrade_path.sync_review_aliases();

    match arsenic_core::validate_fingerprint_rollups(
        &report.behaviour_fingerprint,
        &report.dimension_summaries,
    ) {
        Ok(()) => {}
        Err(errors) => {
            let diagnostics: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            eprintln!(
                "ARSENIC rollup validation failed after recompute from probe results ({} issue{}). Marking behavioural fingerprint unavailable rather than rendering contradictory derived data:\n  - {}",
                diagnostics.len(),
                if diagnostics.len() == 1 { "" } else { "s" },
                diagnostics.join("\n  - ")
            );
            report.behaviour_fingerprint.validation_diagnostics = diagnostics;
            report.behaviour_fingerprint.radar_available = false;
        }
    }

    let svg = if report.behaviour_fingerprint.radar_available
        && validate_fingerprint(&report.behaviour_fingerprint).is_ok()
    {
        build_fingerprint_svg(&report.behaviour_fingerprint)
    } else {
        None
    };
    let changelog = if report
        .behaviour_fingerprint
        .validation_diagnostics
        .is_empty()
    {
        build_fingerprint_changelog(&report.behaviour_fingerprint, &report.latency_summary)
    } else {
        arsenic_core::FingerprintChangelog {
            high_impact: vec![],
            stable_contracts: vec![],
            telemetry: vec![],
            text_summary: format!(
                "Behavioural fingerprint unavailable: {}",
                report
                    .behaviour_fingerprint
                    .validation_diagnostics
                    .join("; ")
            ),
        }
    };

    let mut v = serde_json::to_value(&report).context("serialize drift report")?;
    if let Some(summary) = v.get_mut("summary") {
        mirror_summary_valence_aliases(summary);
    }
    if let Some(obj) = v.as_object_mut() {
        obj.insert(
            "fingerprint_svg".into(),
            serde_json::to_value(svg).unwrap_or(Value::Null),
        );
        obj.insert(
            "fingerprint_changelog".into(),
            serde_json::to_value(changelog).unwrap_or(Value::Null),
        );
        // Probe name lookup for evidence links (id → name), kept small.
        let mut id_to_name = serde_json::Map::new();
        for pr in &report.probe_results {
            id_to_name.insert(pr.probe.id.to_string(), json!(pr.probe.name));
        }
        obj.insert("probe_id_names".into(), Value::Object(id_to_name.clone()));
        // Pre-serialized for progressive-enhancement JS (no score calculation in browser).
        // Escape `<` so untrusted probe text cannot close a <script> tag when marked |safe.
        obj.insert(
            "fingerprint_axes_json".into(),
            Value::String(json_for_inline_script(&report.behaviour_fingerprint.axes)?),
        );
        obj.insert(
            "probe_id_names_json".into(),
            Value::String(json_for_inline_script(&id_to_name)?),
        );
        let latency_change_label = if report.latency_summary.delta_pct <= -10.0 {
            format!("{:.1}% faster", report.latency_summary.delta_pct.abs())
        } else if report.latency_summary.delta_pct >= 10.0 {
            format!("{:.1}% slower", report.latency_summary.delta_pct)
        } else {
            "unchanged".into()
        };
        obj.insert("latency_change_label".into(), json!(latency_change_label));
    }
    Ok(v)
}

/// Serialize JSON for embedding in `<script type="application/json">` with `|safe`.
/// Escapes `<` to `\u003c` so model/probe text cannot break out of the script element.
fn json_for_inline_script<T: serde::Serialize>(value: &T) -> anyhow::Result<String> {
    let s = serde_json::to_string(value).context("serialize fingerprint view json")?;
    Ok(s.replace('<', "\\u003c"))
}

/// Copies `probe_regressions` / `probe_improvements` / `probe_neutral` into legacy summary keys.
pub fn mirror_summary_valence_aliases(summary: &mut Value) {
    let Some(m) = summary.as_object_mut() else {
        return;
    };
    for (from, to) in [
        ("probe_regressions", "regressions"),
        ("probe_improvements", "improvements"),
        ("probe_neutral", "neutral"),
    ] {
        if let Some(v) = m.get(from).cloned() {
            m.insert(to.to_string(), v);
        }
    }
}

impl ReportRenderer {
    pub fn render_html(report: &DriftReport) -> anyhow::Result<String> {
        let mut tera = Tera::default();
        // Path is relative to this file: crates/arsenic-report/src/lib.rs → repo root/report-templates/
        let tpl = include_str!("../../../report-templates/report.html.tera");
        tera.add_raw_template("report.html", tpl)
            .context("parse HTML template")?;
        let ctx = TeraContext::from_value(drift_report_json_value(report)?)?;
        tera.render("report.html", &ctx).context("render HTML")
    }

    pub fn render_json(report: &DriftReport) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(&drift_report_json_value(
            report,
        )?)?)
    }

    pub fn render_markdown(report: &DriftReport) -> anyhow::Result<String> {
        let mut tera = Tera::default();
        // Same path convention as `render_html` (relative to this source file).
        let tpl = include_str!("../../../report-templates/report.md.tera");
        tera.add_raw_template("report.md", tpl)
            .context("parse Markdown template")?;
        let ctx = TeraContext::from_value(drift_report_json_value(report)?)?;
        tera.render("report.md", &ctx).context("render Markdown")
    }

    /// Minimal stdout summary without Tera.
    pub fn render_summary_line(report: &DriftReport) -> String {
        let mut report = report.clone();
        report.rebuild_rollups();
        report.sync_valence_from_probe_results();
        format!(
            "run={} overall={:?} probes={} green={} amber={} red={} blocking={} review={} presentation={} telemetry={} regressions={} improvements={} neutral={} safe_to_upgrade={}",
            report.run_id,
            report.overall_risk,
            report.summary.total_probes,
            report.summary.probes_green,
            report.summary.probes_amber,
            report.summary.probes_red,
            report.summary.blocking_regressions,
            report.summary.review_items,
            report.summary.presentation_drift,
            report.summary.telemetry_drift,
            report.summary.probe_regressions,
            report.summary.probe_improvements,
            report.summary.probe_neutral,
            report.summary.safe_to_upgrade
        )
    }

    /// Compact JSON for CLI / tooling; `summary` includes both `probe_*` and `regressions` / `improvements` / `neutral`.
    pub fn summary_json(report: &DriftReport) -> anyhow::Result<Value> {
        let full = drift_report_json_value(report)?;
        Ok(json!({
            "run_id": full.get("run_id").cloned().unwrap_or(Value::Null),
            "overall_risk": full.get("overall_risk").cloned().unwrap_or(Value::Null),
            "summary": full.get("summary").cloned().unwrap_or(Value::Null),
            "v1": full.get("v1_model").cloned().unwrap_or(Value::Null),
            "v2": full.get("v2_model").cloned().unwrap_or(Value::Null),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arsenic_core::DriftReport;

    const FIXTURE_REPORT: &str = include_str!("../fixtures/report_openai_upgrade.json");

    #[test]
    fn mirror_summary_valence_aliases_fills_short_keys() {
        let report: DriftReport = serde_json::from_str(FIXTURE_REPORT)
            .expect("parse fixtures/report_openai_upgrade.json");
        let v = drift_report_json_value(&report).expect("export");
        let s = v.get("summary").expect("summary");
        assert_eq!(s.get("probe_regressions"), s.get("regressions"));
        assert_eq!(s.get("probe_improvements"), s.get("improvements"));
        assert_eq!(s.get("probe_neutral"), s.get("neutral"));
    }

    #[test]
    fn legacy_json_export_is_additive() {
        let report: DriftReport = serde_json::from_str(FIXTURE_REPORT)
            .expect("parse fixtures/report_openai_upgrade.json");
        let v = drift_report_json_value(&report).expect("export");

        let summary = v.get("summary").expect("summary");
        assert!(summary.get("probes_red").is_some());
        assert!(summary.get("blocking_regressions").is_some());
        assert!(summary.get("review_items").is_some());
        assert!(summary.get("presentation_drift").is_some());
        assert!(summary.get("telemetry_drift").is_some());
        assert_eq!(summary.get("regressions"), summary.get("probe_regressions"));

        let probes = v
            .get("probe_results")
            .and_then(|p| p.as_array())
            .expect("probes");
        assert!(!probes.is_empty());
        assert!(probes[0].get("drift_impact").is_some());
        assert!(probes[0].get("overall_risk").is_some());
    }

    #[test]
    fn legacy_fixture_renders_executive_impact_summary() {
        let report: DriftReport = serde_json::from_str(FIXTURE_REPORT)
            .expect("parse fixtures/report_openai_upgrade.json");
        let html = ReportRenderer::render_html(&report).expect("render HTML");
        assert!(html.contains("Blocking regressions:"));
        assert!(html.contains("Review items:"));
        assert!(html.contains("Presentation drift:"));
        assert!(html.contains("Telemetry drift:"));
    }

    #[test]
    fn legacy_fixture_derives_fingerprint_and_renders_radar() {
        let report: DriftReport = serde_json::from_str(FIXTURE_REPORT)
            .expect("parse fixtures/report_openai_upgrade.json");
        let v = drift_report_json_value(&report).expect("export");
        let fp = v.get("behaviour_fingerprint").expect("fingerprint field");
        assert!(fp
            .get("axes")
            .and_then(|a| a.as_array())
            .is_some_and(|a| !a.is_empty()));
        assert!(v.get("fingerprint_svg").is_some());
        assert!(v.get("fingerprint_changelog").is_some());

        let html = ReportRenderer::render_html(&report).expect("render HTML");
        assert!(html.contains("Behavioural fingerprint"));
        assert!(html.contains("fp-svg-title"));
        assert!(html.contains("fp-svg-desc"));
        assert!(html.contains("class=\"fp-baseline\""));
        assert!(html.contains("class=\"fp-candidate\""));
        assert!(html.contains("class=\"fp-grid\""));
        assert!(html.contains("Retention table"));
        assert!(html.contains("Behavioural changelog"));
        assert!(html.contains("Dimensions outside the polygon"));
        assert!(!html.contains("chart.js"));
        assert!(!html.contains("cdn."));
        assert!(!html.contains("NaN"));
        assert!(!html.contains("Infinity"));
        assert!(html.contains("id=\"fp-axes-data\""));
        assert!(html.contains("probe-"));
    }

    #[test]
    fn fingerprint_json_round_trip() {
        let report: DriftReport = serde_json::from_str(FIXTURE_REPORT)
            .expect("parse fixtures/report_openai_upgrade.json");
        let json = ReportRenderer::render_json(&report).expect("json");
        let again: DriftReport = serde_json::from_str(&json).expect("re-parse");
        assert!(!again.behaviour_fingerprint.axes.is_empty());
        assert_eq!(
            again.behaviour_fingerprint.version,
            arsenic_core::FINGERPRINT_VERSION
        );
    }

    /// Machine-readable fixture audit: retention arithmetic and omitted-axis reasons.
    #[test]
    fn fixture_fingerprint_audit_table() {
        let mut report: DriftReport = serde_json::from_str(FIXTURE_REPORT)
            .expect("parse fixtures/report_openai_upgrade.json");
        report.rebuild_rollups();
        arsenic_core::validate_fingerprint(&report.behaviour_fingerprint)
            .expect("fingerprint validates");

        let mut table = Vec::new();
        for axis in &report.behaviour_fingerprint.axes {
            let sum =
                axis.unchanged_probes + axis.regressions + axis.improvements + axis.neutral_changes;
            assert_eq!(
                sum, axis.applicable_probes,
                "axis {}: unchanged+reg+imp+neu ({sum}) != applicable ({})",
                axis.id, axis.applicable_probes
            );
            table.push(json!({
                "axis": axis.id,
                "label": axis.label,
                "applicable": axis.applicable_probes,
                "unchanged": axis.unchanged_probes,
                "regressions": axis.regressions,
                "improvements": axis.improvements,
                "neutral_changes": axis.neutral_changes,
                "retention": axis.score,
                "risk": format!("{:?}", axis.risk),
            }));
        }

        let omitted: Vec<_> = report
            .behaviour_fingerprint
            .omitted_axes
            .iter()
            .map(|o| {
                json!({
                    "axis": o.id,
                    "label": o.label,
                    "reason": o.reason,
                    "detail": o.detail,
                })
            })
            .collect();

        assert!(
            omitted
                .iter()
                .any(|o| o.get("axis") == Some(&json!("schema"))),
            "Schema must be explicitly omitted on this fixture"
        );

        let audit = json!({
            "axes": table,
            "omitted_axes": omitted,
        });
        let pretty = serde_json::to_string_pretty(&audit).expect("serialize audit");
        assert!(pretty.contains("\"applicable\""));
        // Surface for `cargo test -- --nocapture` / audit review.
        eprintln!("FINGERPRINT_FIXTURE_AUDIT\n{pretty}");

        let html = ReportRenderer::render_html(&report).expect("html");
        assert!(html.contains("Dimensions outside the polygon"));
        assert!(html.contains("expected schema"));
        assert!(html.contains("Consistency retention"));
        assert!(html.contains("Compatibility retention"));
        assert!(html.contains("Deployment risk / severity"));
        assert!(html.contains("dimension-specific materiality rules"));
        assert!(!html.contains("including 0 regressions"));
    }

    #[test]
    fn real_report_fingerprint_reconciles_with_summaries() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../report.json");
        let Ok(raw) = std::fs::read_to_string(path) else {
            // Optional local artifact from a live compare; skip if absent.
            return;
        };
        let mut report: DriftReport = serde_json::from_str(&raw).expect("parse report.json");
        report.rebuild_rollups();
        let mismatches = arsenic_core::fingerprint_summary_mismatches(
            &report.behaviour_fingerprint,
            &report.dimension_summaries,
        );
        assert!(
            mismatches.is_empty(),
            "fingerprint vs summary mismatches: {mismatches:?}"
        );
        let morph = report
            .behaviour_fingerprint
            .axes
            .iter()
            .find(|a| a.id == "morphology")
            .expect("morphology");
        assert_eq!(
            morph.changed_probes,
            report.dimension_summaries.morphology.probes_affected
        );
        let tone = report
            .behaviour_fingerprint
            .axes
            .iter()
            .find(|a| a.id == "tone")
            .expect("tone");
        assert_eq!(
            tone.changed_probes,
            report.dimension_summaries.tone.probes_affected
        );
        let cons = report
            .behaviour_fingerprint
            .axes
            .iter()
            .find(|a| a.id == "consistency")
            .expect("consistency");
        // Fingerprint uses drift (band crossing), not absolute inconsistency.
        assert_eq!(
            cons.changed_probes,
            report
                .dimension_summaries
                .consistency
                .materially_changed_probes
        );
        assert_eq!(cons.changed_probes, 1); // open_ended_recommendation only
        assert_eq!(report.dimension_summaries.consistency.probes_affected, 5); // absolute inconsistency still 5
        assert!((cons.score - 99.05).abs() < 0.01);
        assert_eq!(cons.baseline_inconsistent_probes, Some(5));
        assert_eq!(cons.candidate_inconsistent_probes, Some(4));
        assert_eq!(
            cons.aggregation_kind,
            arsenic_core::FingerprintAggregationKind::AggregateSimilarity
        );
        arsenic_core::validate_fingerprint_rollups(
            &report.behaviour_fingerprint,
            &report.dimension_summaries,
        )
        .expect("rollups reconcile");

        let html = ReportRenderer::render_html(&report).expect("html");
        assert!(html.contains("effectively stable") || html.contains("Aggregate consistency"));
        assert!(html.contains("consistency-drift rule") || html.contains("absolute consistency"));
        assert!(!html.contains("Consistency retention fell to 99%"));
        assert!(html.contains("Review items"));
    }

    #[test]
    fn blocking_panel_empty_shows_banner_only() {
        use arsenic_core::RiskLevel;

        let mut report: DriftReport = serde_json::from_str(FIXTURE_REPORT)
            .expect("parse fixtures/report_openai_upgrade.json");
        report.rebuild_rollups();
        for pr in &mut report.probe_results {
            pr.overall_risk = RiskLevel::Green;
            pr.drift_impact = arsenic_core::DriftImpact::Informational;
        }
        report.summary.probes_red = 0;
        report.summary.blocking_regressions = 0;
        report.summary.safe_to_upgrade = true;

        let html = ReportRenderer::render_html(&report).expect("render HTML");
        let start = html
            .find("<h2>Blocking regressions</h2>")
            .expect("blocking regressions section");
        let end = html[start..]
            .find("<h2>All probe results</h2>")
            .map(|i| start + i)
            .expect("all probe results section");
        let section = &html[start..end];

        assert!(section.contains("No blocking regressions detected."));
        assert!(!section.contains("badge Red"));
    }
}
