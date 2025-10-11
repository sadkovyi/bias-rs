use arrow_array::RecordBatch;
use arrow_schema::{DataType, SchemaRef};

use crate::error::BiasError;

/// Arrow-backed dataset container used by the audit engine.
#[derive(Debug, Clone)]
pub struct Dataset {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

impl Dataset {
    /// Creates a dataset from a schema and record batches.
    pub fn new(schema: SchemaRef, batches: Vec<RecordBatch>) -> Result<Self, BiasError> {
        for batch in &batches {
            if batch.schema() != schema {
                return Err(BiasError::InvalidConfig(
                    "record batch schema does not match dataset schema".to_string(),
                ));
            }
        }

        Ok(Self { schema, batches })
    }

    /// Returns the Arrow schema.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Returns the record batches.
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// Returns the number of rows across all batches.
    pub fn row_count(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }

    /// Returns the number of columns in the schema.
    pub fn column_count(&self) -> usize {
        self.schema.fields().len()
    }

    /// Returns the names of the columns in schema order.
    pub fn column_names(&self) -> Vec<&str> {
        self.schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect()
    }

    /// Returns the data type for a named column.
    pub fn column_type(&self, name: &str) -> Option<&DataType> {
        self.schema.field_with_name(name).ok().map(|field| field.data_type())
    }

    /// Returns true when the column exists.
    pub fn has_column(&self, name: &str) -> bool {
        self.schema.field_with_name(name).is_ok()
    }
}
