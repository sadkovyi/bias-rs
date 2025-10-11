use crate::{AuditConfig, AuditReport, BiasError, ColumnProfile, Dataset, DatasetSummary};
use crate::{report::AuditReportConfig, Severity};

/// Runs a dataset audit.
pub fn audit_dataset(dataset: &Dataset, config: &AuditConfig) -> Result<AuditReport, BiasError> {
    if config.sensitive_columns.is_empty() {
        return Err(BiasError::InvalidConfig(
            "at least one sensitive column is required".to_string(),
        ));
    }

    let missing_columns: Vec<_> = config
        .sensitive_columns
        .iter()
        .filter(|column| !dataset.has_column(column))
        .cloned()
        .collect();
    if let Some(column) = missing_columns.first() {
        return Err(BiasError::MissingColumn(column.clone()));
    }

    Ok(AuditReport {
        dataset: DatasetSummary {
            row_count: dataset.row_count(),
            column_count: dataset.column_count(),
            batch_count: dataset.batches().len(),
            columns: dataset
                .schema()
                .fields()
                .iter()
                .map(|field| ColumnProfile {
                    name: field.name().to_string(),
                    data_type: field.data_type().to_string(),
                    nullable: field.is_nullable(),
                })
                .collect(),
        },
        config: AuditReportConfig {
            sensitive_columns: config.sensitive_columns.clone(),
            grouping_mode: config.grouping_mode,
            alpha: config.alpha,
            multiple_testing: config.multiple_testing,
        },
        detector_runs: Vec::new(),
        group_summaries: Vec::new(),
        skipped: Vec::new(),
        findings: vec![crate::Finding {
            detector: crate::DetectorKind::Representation,
            grouping: "not-run".to_string(),
            sensitive_columns: config.sensitive_columns.clone(),
            target_column: None,
            group: None,
            severity: Severity::Info,
            message: "audit engine is not wired yet".to_string(),
            p_value: None,
            corrected_p_value: None,
            effect_size: None,
            metrics: Default::default(),
        }],
    })
}
