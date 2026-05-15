//! HTML / JSON / Markdown report rendering.

use anyhow::Context;
use arsenic_core::DriftReport;
use serde_json::{json, Value};
use tera::{Context as TeraContext, Tera};

pub struct ReportRenderer;

/// Syncs valence from `probe_results`, serializes the report, then mirrors drift valence into
/// `summary.regressions` / `improvements` / `neutral` (same values as `probe_*`).
///
/// External scripts often expect the short names; Rust and the HTML templates use `probe_*`.
pub fn drift_report_json_value(report: &DriftReport) -> anyhow::Result<Value> {
    let mut report = report.clone();
    report.sync_valence_from_probe_results();
    let mut v = serde_json::to_value(&report).context("serialize drift report")?;
    if let Some(summary) = v.get_mut("summary") {
        mirror_summary_valence_aliases(summary);
    }
    Ok(v)
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
        Ok(serde_json::to_string_pretty(&drift_report_json_value(report)?)?)
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
        report.sync_valence_from_probe_results();
        format!(
            "run={} overall={:?} probes={} green={} amber={} red={} regressions={} improvements={} neutral={} safe_to_upgrade={}",
            report.run_id,
            report.overall_risk,
            report.summary.total_probes,
            report.summary.probes_green,
            report.summary.probes_amber,
            report.summary.probes_red,
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

    #[test]
    fn mirror_summary_valence_aliases_fills_short_keys() {
        let json = include_str!("../../../report_llama_upgrade.json");
        let report: DriftReport = serde_json::from_str(json).expect("parse report_llama_upgrade.json");
        let v = drift_report_json_value(&report).expect("export");
        let s = v.get("summary").expect("summary");
        assert_eq!(s.get("probe_regressions"), s.get("regressions"));
        assert_eq!(s.get("probe_improvements"), s.get("improvements"));
        assert_eq!(s.get("probe_neutral"), s.get("neutral"));
    }
}
