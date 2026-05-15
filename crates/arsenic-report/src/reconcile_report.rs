//! HTML and JSON rendering for reconcile results.

use anyhow::Context;
use arsenic_core::ReconcileResult;
use serde_json::Value;
use tera::{Context as TeraContext, Tera};

pub fn reconcile_json_value(result: &ReconcileResult) -> anyhow::Result<Value> {
    serde_json::to_value(result).context("serialize reconcile result")
}

pub fn render_reconcile_html(result: &ReconcileResult) -> anyhow::Result<String> {
    let mut tera = Tera::default();
    let tpl = include_str!("../../../report-templates/reconcile.html.tera");
    tera.add_raw_template("reconcile.html", tpl)
        .context("parse reconcile HTML template")?;
    let ctx = TeraContext::from_value(reconcile_json_value(result)?)?;
    tera.render("reconcile.html", &ctx)
        .context("render reconcile HTML")
}

pub fn render_reconcile_json(result: &ReconcileResult) -> anyhow::Result<String> {
    Ok(serde_json::to_string_pretty(result)?)
}
