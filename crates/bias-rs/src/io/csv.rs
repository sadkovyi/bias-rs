use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{BooleanBuilder, Float64Builder, Int64Builder, StringBuilder};
use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

use crate::{BiasError, Dataset};

/// Options for CSV ingestion.
#[derive(Debug, Clone)]
pub struct CsvReadOptions {
    pub has_headers: bool,
    pub delimiter: u8,
    pub batch_size: usize,
    pub schema_overrides: BTreeMap<String, DataType>,
    pub null_values: Vec<String>,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        Self {
            has_headers: true,
            delimiter: b',',
            batch_size: 4096,
            schema_overrides: BTreeMap::new(),
            null_values: vec![
                String::new(),
                "NA".to_string(),
                "N/A".to_string(),
                "NULL".to_string(),
                "null".to_string(),
            ],
        }
    }
}

/// Reads a CSV file into an Arrow-backed dataset.
pub fn read_csv(path: impl AsRef<Path>, options: CsvReadOptions) -> Result<Dataset, BiasError> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(options.has_headers)
        .delimiter(options.delimiter)
        .flexible(true)
        .from_path(path)
        .map_err(|error| BiasError::Csv(error.to_string()))?;

    let mut rows = Vec::<Vec<Option<String>>>::new();
    let header_names = if options.has_headers {
        let headers = reader
            .headers()
            .map_err(|error| BiasError::Csv(error.to_string()))?;
        headers.iter().map(ToOwned::to_owned).collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    for record in reader.records() {
        let record = record.map_err(|error| BiasError::Csv(error.to_string()))?;
        rows.push(
            record
                .iter()
                .map(|value| normalize_value(value, &options.null_values))
                .collect(),
        );
    }

    let column_count = if options.has_headers {
        header_names.len()
    } else {
        rows.first()
            .map(Vec::len)
            .ok_or_else(|| BiasError::Csv("csv file has no rows".to_string()))?
    };

    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != column_count {
            return Err(BiasError::Csv(format!(
                "row {} has {} columns but expected {}",
                row_index + 1,
                row.len(),
                column_count
            )));
        }
    }

    let column_names = if options.has_headers {
        header_names
    } else {
        (0..column_count)
            .map(|index| format!("column_{}", index + 1))
            .collect()
    };

    let inferred_types = infer_schema(&column_names, &rows, &options.schema_overrides)?;
    let schema = Arc::new(Schema::new(
        column_names
            .iter()
            .zip(inferred_types.iter())
            .map(|(name, data_type)| Field::new(name, data_type.clone(), true))
            .collect::<Vec<_>>(),
    ));

    let batch_size = options.batch_size.max(1);
    let mut batches = Vec::new();
    for chunk in rows.chunks(batch_size) {
        let arrays = column_names
            .iter()
            .enumerate()
            .map(|(column_index, name)| {
                build_array(name, &inferred_types[column_index], chunk, column_index)
            })
            .collect::<Result<Vec<_>, _>>()?;
        batches.push(
            RecordBatch::try_new(schema.clone(), arrays)
                .map_err(|error| BiasError::Message(error.to_string()))?,
        );
    }

    Dataset::new(schema, batches)
}

fn normalize_value(raw: &str, null_values: &[String]) -> Option<String> {
    null_values
        .iter()
        .any(|candidate| candidate == raw)
        .then_some(())
        .map(|_| None)
        .unwrap_or_else(|| Some(raw.to_string()))
}

fn infer_schema(
    column_names: &[String],
    rows: &[Vec<Option<String>>],
    overrides: &BTreeMap<String, DataType>,
) -> Result<Vec<DataType>, BiasError> {
    column_names
        .iter()
        .enumerate()
        .map(|(column_index, column_name)| {
            if let Some(data_type) = overrides.get(column_name) {
                return Ok(data_type.clone());
            }

            let inferred = rows.iter().fold(InferredType::Unknown, |current, row| {
                match row[column_index].as_deref() {
                    Some(value) => current.promote(value),
                    None => current,
                }
            });
            Ok(inferred.to_arrow())
        })
        .collect()
}

fn build_array(
    column_name: &str,
    data_type: &DataType,
    rows: &[Vec<Option<String>>],
    column_index: usize,
) -> Result<ArrayRef, BiasError> {
    match data_type {
        DataType::Boolean => {
            let mut builder = BooleanBuilder::with_capacity(rows.len());
            for row in rows {
                match row[column_index].as_deref() {
                    Some(value) => builder.append_value(parse_bool(column_name, value)?),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(BooleanArray::from(builder.finish())) as ArrayRef)
        }
        DataType::Int64 => {
            let mut builder = Int64Builder::with_capacity(rows.len());
            for row in rows {
                match row[column_index].as_deref() {
                    Some(value) => builder.append_value(parse_i64(column_name, value)?),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(Int64Array::from(builder.finish())) as ArrayRef)
        }
        DataType::Float64 => {
            let mut builder = Float64Builder::with_capacity(rows.len());
            for row in rows {
                match row[column_index].as_deref() {
                    Some(value) => builder.append_value(parse_f64(column_name, value)?),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(Float64Array::from(builder.finish())) as ArrayRef)
        }
        DataType::Utf8 => {
            let mut builder = StringBuilder::with_capacity(rows.len(), rows.len() * 8);
            for row in rows {
                match row[column_index].as_deref() {
                    Some(value) => builder.append_value(value),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(StringArray::from(builder.finish())) as ArrayRef)
        }
        other => Err(BiasError::UnsupportedColumnType {
            column: column_name.to_string(),
            data_type: other.to_string(),
        }),
    }
}

fn parse_bool(column_name: &str, value: &str) -> Result<bool, BiasError> {
    match value.to_ascii_lowercase().as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(BiasError::Csv(format!(
            "column `{column_name}` could not parse `{value}` as boolean"
        ))),
    }
}

fn parse_i64(column_name: &str, value: &str) -> Result<i64, BiasError> {
    value.parse::<i64>().map_err(|_| {
        BiasError::Csv(format!(
            "column `{column_name}` could not parse `{value}` as integer"
        ))
    })
}

fn parse_f64(column_name: &str, value: &str) -> Result<f64, BiasError> {
    value.parse::<f64>().map_err(|_| {
        BiasError::Csv(format!(
            "column `{column_name}` could not parse `{value}` as float"
        ))
    })
}

#[derive(Debug, Clone, Copy)]
enum InferredType {
    Unknown,
    Boolean,
    Integer,
    Float,
    Utf8,
}

impl InferredType {
    fn promote(self, value: &str) -> Self {
        let next = if value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("false") {
            Self::Boolean
        } else if value.parse::<i64>().is_ok() {
            Self::Integer
        } else if value.parse::<f64>().is_ok() {
            Self::Float
        } else {
            Self::Utf8
        };

        match (self, next) {
            (Self::Unknown, current) => current,
            (Self::Utf8, _) | (_, Self::Utf8) => Self::Utf8,
            (Self::Float, _) | (_, Self::Float) => Self::Float,
            (Self::Integer, Self::Integer) => Self::Integer,
            (Self::Boolean, Self::Boolean) => Self::Boolean,
            _ => Self::Utf8,
        }
    }

    fn to_arrow(self) -> DataType {
        match self {
            Self::Unknown | Self::Utf8 => DataType::Utf8,
            Self::Boolean => DataType::Boolean,
            Self::Integer => DataType::Int64,
            Self::Float => DataType::Float64,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::Write;

    use arrow_array::{Array, Float64Array, Int64Array, StringArray};
    use arrow_schema::DataType;
    use tempfile::NamedTempFile;

    use super::{CsvReadOptions, read_csv};

    #[test]
    fn reads_headers_and_mixed_types() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(
            file,
            "group,age,score,active\nA,21,0.5,true\nB,34,1.2,false"
        )
        .expect("write csv");

        let dataset = read_csv(file.path(), CsvReadOptions::default()).expect("dataset");
        assert_eq!(dataset.column_names(), vec!["group", "age", "score", "active"]);
        assert_eq!(dataset.row_count(), 2);
        assert_eq!(dataset.column_type("age"), Some(&DataType::Int64));
        assert_eq!(dataset.column_type("score"), Some(&DataType::Float64));
        assert_eq!(dataset.column_type("active"), Some(&DataType::Boolean));
    }

    #[test]
    fn respects_null_values_and_quotes() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(file, "city,note\n\"New York\",\"\"\nParis,\"kept\"").expect("write csv");

        let dataset = read_csv(file.path(), CsvReadOptions::default()).expect("dataset");
        let batch = &dataset.batches()[0];
        let city = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("city column");
        let note = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("note column");
        assert_eq!(city.value(0), "New York");
        assert!(note.is_null(0));
        assert_eq!(note.value(1), "kept");
    }

    #[test]
    fn supports_schema_overrides() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(file, "zip\n00123").expect("write csv");

        let mut options = CsvReadOptions::default();
        options
            .schema_overrides
            .insert("zip".to_string(), DataType::Utf8);
        let dataset = read_csv(file.path(), options).expect("dataset");
        let batch = &dataset.batches()[0];
        let zip = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("zip column");
        assert_eq!(zip.value(0), "00123");
    }

    #[test]
    fn batches_rows_when_requested() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(file, "group,age\nA,20\nB,30\nC,40").expect("write csv");

        let options = CsvReadOptions {
            batch_size: 2,
            ..CsvReadOptions::default()
        };
        let dataset = read_csv(file.path(), options).expect("dataset");
        assert_eq!(dataset.batches().len(), 2);

        let first_batch_age = dataset.batches()[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("age column");
        assert_eq!(first_batch_age.value(0), 20);

        let second_batch_age = dataset.batches()[1]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("age column");
        assert_eq!(second_batch_age.value(0), 40);
    }

    #[test]
    fn reports_row_width_errors() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(file, "group,age\nA,20\nB").expect("write csv");

        let error = read_csv(file.path(), CsvReadOptions::default()).expect_err("csv error");
        let message = error.to_string();
        assert!(message.contains("row 2"));
    }

    #[test]
    fn infers_float_columns() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(file, "score\n1\n2.5").expect("write csv");

        let dataset = read_csv(file.path(), CsvReadOptions::default()).expect("dataset");
        let batch = &dataset.batches()[0];
        let score = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("score column");
        assert_eq!(score.value(0), 1.0);
        assert_eq!(score.value(1), 2.5);
    }

    #[test]
    fn supports_headerless_files() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(file, "A,20\nB,30").expect("write csv");

        let options = CsvReadOptions {
            has_headers: false,
            ..CsvReadOptions::default()
        };
        let dataset = read_csv(file.path(), options).expect("dataset");
        assert_eq!(dataset.column_names(), vec!["column_1", "column_2"]);
    }

    #[test]
    fn keeps_utf8_when_inference_is_mixed() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(file, "value\n10\nnorth").expect("write csv");

        let dataset = read_csv(file.path(), CsvReadOptions::default()).expect("dataset");
        assert_eq!(dataset.column_type("value"), Some(&DataType::Utf8));
    }

    #[test]
    fn accepts_custom_null_values() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(file, "score\nmissing\n1.5").expect("write csv");

        let options = CsvReadOptions {
            null_values: vec!["missing".to_string()],
            ..CsvReadOptions::default()
        };
        let dataset = read_csv(file.path(), options).expect("dataset");
        let batch = &dataset.batches()[0];
        let score = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("score column");
        assert!(score.is_null(0));
        assert_eq!(score.value(1), 1.5);
    }

    #[test]
    fn allows_explicit_type_map() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(file, "flag\ntrue").expect("write csv");

        let mut options = CsvReadOptions::default();
        let mut overrides = BTreeMap::new();
        overrides.insert("flag".to_string(), DataType::Utf8);
        options.schema_overrides = overrides;

        let dataset = read_csv(file.path(), options).expect("dataset");
        assert_eq!(dataset.column_type("flag"), Some(&DataType::Utf8));
    }
}
