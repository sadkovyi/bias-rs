use thiserror::Error;

/// Errors returned by the bias-rs library.
#[derive(Debug, Error)]
pub enum BiasError {
    #[error("column `{0}` was not found")]
    MissingColumn(String),
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("unsupported column type for `{column}`: {data_type}")]
    UnsupportedColumnType { column: String, data_type: String },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("csv error: {0}")]
    Csv(String),
    #[error("{0}")]
    Message(String),
}
