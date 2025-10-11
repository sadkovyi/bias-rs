//! Dataset bias detection for tabular data.

mod audit;
mod config;
mod dataset;
mod error;
mod report;

pub use audit::audit_dataset;
pub use config::{
    AuditConfig, AuditConfigBuilder, CategoricalAssociationConfig, ColumnSelection, DetectorConfig,
    DetectorKind, GroupingMode, MissingnessConfig, MultipleTestingCorrection,
    NumericDistributionConfig, ReferenceDistribution, RepresentationConfig,
};
pub use dataset::Dataset;
pub use error::BiasError;
pub use report::{
    AuditReport, ColumnProfile, DatasetSummary, DetectorRun, Finding, GroupSummary, Severity,
    SkippedAnalysis,
};

/// Returns the crate version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
