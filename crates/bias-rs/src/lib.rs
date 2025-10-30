//! Dataset bias detection for tabular data.

mod audit;
mod config;
mod dataset;
mod error;
pub mod io;
mod report;
mod stats;

pub use audit::audit_dataset;
pub use config::{
    AuditConfig, AuditConfigBuilder, CategoricalAssociationConfig, ColumnSelection, DetectorConfig,
    DetectorKind, GroupingMode, MissingnessConfig, MultipleTestingCorrection,
    NumericDistributionConfig, ReferenceDistribution, RepresentationConfig,
};
pub use dataset::Dataset;
pub use error::BiasError;
pub use io::csv::{CsvReadOptions, read_csv};
pub use report::{
    AuditReport, ColumnProfile, DatasetSummary, DetectorRun, Finding, GroupSummary, Severity,
    SkippedAnalysis,
};

/// Returns the crate version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
