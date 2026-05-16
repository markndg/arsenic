fn main() {
    println!("cargo:rerun-if-changed=../../report-templates/report.html.tera");
    println!("cargo:rerun-if-changed=../../report-templates/report.md.tera");
    println!("cargo:rerun-if-changed=../../report-templates/reconcile.html.tera");
}
