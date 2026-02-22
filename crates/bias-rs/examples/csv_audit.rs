use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use bias_rs::{AuditConfig, CsvReadOptions, audit_dataset, read_csv};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join(format!(
        "bias-rs-example-{}.csv",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::write(
        &path,
        "gender,region,tenure\nwoman,north,2\nwoman,north,3\nwoman,south,4\nman,south,8\nman,south,9\nman,south,10\n",
    )?;

    let dataset = read_csv(&path, CsvReadOptions::default())?;
    let config = AuditConfig::builder()
        .sensitive_column("gender")
        .min_group_size(2)
        .build();
    let report = audit_dataset(&dataset, &config)?;

    println!("rows: {}", report.dataset.row_count);
    println!("findings: {}", report.findings.len());
    for finding in &report.findings {
        println!("- {}", finding.message);
    }
    let _ = fs::remove_file(&path);

    Ok(())
}
