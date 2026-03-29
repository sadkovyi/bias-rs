use std::collections::BTreeMap;

use crate::config::{DetectorConfig, DetectorKind, GroupingMode, MultipleTestingCorrection};

/// High-level report returned by the audit engine.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditReport {
    pub dataset: DatasetSummary,
    pub config: AuditReportConfig,
    pub detector_runs: Vec<DetectorRun>,
    pub group_summaries: Vec<GroupSummary>,
    pub skipped: Vec<SkippedAnalysis>,
    pub findings: Vec<Finding>,
}

/// Snapshot of the effective audit configuration.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditReportConfig {
    pub sensitive_columns: Vec<String>,
    pub grouping_mode: GroupingMode,
    pub alpha: f64,
    pub multiple_testing: MultipleTestingCorrection,
    pub detectors: Vec<DetectorConfig>,
}

/// Basic dataset metadata included in the report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DatasetSummary {
    pub row_count: usize,
    pub column_count: usize,
    pub batch_count: usize,
    pub columns: Vec<ColumnProfile>,
}

/// Per-column metadata.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ColumnProfile {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
}

/// Metadata about a detector execution.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DetectorRun {
    pub detector: DetectorKind,
    pub grouping: String,
    pub analyzed_columns: usize,
    pub finding_count: usize,
}

/// Per-group row distribution summary.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GroupSummary {
    pub grouping: String,
    pub sensitive_columns: Vec<String>,
    pub group: String,
    pub row_count: usize,
    pub proportion: f64,
    pub expected_proportion: Option<f64>,
}

/// Severity assigned to a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

/// Single finding emitted by a detector.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Finding {
    pub detector: DetectorKind,
    pub grouping: String,
    pub sensitive_columns: Vec<String>,
    pub target_column: Option<String>,
    pub group: Option<String>,
    pub severity: Severity,
    pub message: String,
    pub p_value: Option<f64>,
    pub corrected_p_value: Option<f64>,
    pub effect_size: Option<f64>,
    pub metrics: BTreeMap<String, f64>,
}

/// Analysis that was skipped and why.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SkippedAnalysis {
    pub detector: DetectorKind,
    pub grouping: String,
    pub target_column: Option<String>,
    pub reason: String,
}
