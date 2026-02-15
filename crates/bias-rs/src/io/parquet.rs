use std::fs::File;
use std::path::Path;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::{BiasError, Dataset};

/// Options for Parquet ingestion.
#[derive(Debug, Clone, Copy)]
pub struct ParquetReadOptions {
    pub batch_size: usize,
}

impl Default for ParquetReadOptions {
    fn default() -> Self {
        Self { batch_size: 4096 }
    }
}

/// Reads a Parquet file into an Arrow-backed dataset.
pub fn read_parquet(
    path: impl AsRef<Path>,
    options: ParquetReadOptions,
) -> Result<Dataset, BiasError> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| BiasError::Parquet(error.to_string()))?;
    let schema = builder.schema().clone();
    let reader = builder
        .with_batch_size(options.batch_size.max(1))
        .build()
        .map_err(|error| BiasError::Parquet(error.to_string()))?;
    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(|error| BiasError::Parquet(error.to_string()))?);
    }

    Dataset::new(schema, batches)
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use arrow_array::{Float64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use tempfile::NamedTempFile;

    use super::{ParquetReadOptions, read_parquet};

    #[test]
    fn reads_parquet_files() {
        let file = NamedTempFile::new().expect("temp file");
        let schema = std::sync::Arc::new(Schema::new(vec![
            Field::new("group", DataType::Utf8, true),
            Field::new("score", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                std::sync::Arc::new(StringArray::from(vec![Some("a"), Some("b")])) as _,
                std::sync::Arc::new(Float64Array::from(vec![Some(1.0), Some(2.5)])) as _,
            ],
        )
        .expect("record batch");

        let writer_file = File::create(file.path()).expect("writer file");
        let mut writer = ArrowWriter::try_new(writer_file, schema, None).expect("writer");
        writer.write(&batch).expect("write batch");
        writer.close().expect("close writer");

        let dataset = read_parquet(file.path(), ParquetReadOptions::default()).expect("dataset");
        assert_eq!(dataset.row_count(), 2);
        assert_eq!(dataset.column_type("group"), Some(&DataType::Utf8));
        assert_eq!(dataset.column_type("score"), Some(&DataType::Float64));
    }
}
