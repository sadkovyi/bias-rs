use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, LargeStringArray, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::DataType;

use crate::{AuditConfig, BiasError, Dataset, GroupingMode};

macro_rules! push_display_values {
    ($array:expr, $values:expr) => {{
        for index in 0..$array.len() {
            if $array.is_null(index) {
                $values.push(None);
            } else {
                $values.push(Some($array.value(index).to_string()));
            }
        }
    }};
}

macro_rules! push_numeric_values {
    ($array:expr, $values:expr) => {{
        for index in 0..$array.len() {
            if $array.is_null(index) {
                $values.push(None);
            } else {
                $values.push(Some($array.value(index) as f64));
            }
        }
    }};
}

#[derive(Debug, Clone)]
pub(crate) struct GroupingSpec {
    pub label: String,
    pub columns: Vec<String>,
}

pub(crate) fn grouping_specs(config: &AuditConfig) -> Vec<GroupingSpec> {
    let per_column = config.sensitive_columns.iter().map(|column| GroupingSpec {
        label: column.clone(),
        columns: vec![column.clone()],
    });
    let intersectional = (config.sensitive_columns.len() > 1).then(|| GroupingSpec {
        label: config.sensitive_columns.join("+"),
        columns: config.sensitive_columns.clone(),
    });

    match config.grouping_mode {
        GroupingMode::PerSensitiveColumn => per_column.collect(),
        GroupingMode::Intersectional => intersectional.into_iter().collect(),
        GroupingMode::Both => {
            let mut groups = per_column.collect::<Vec<_>>();
            groups.extend(intersectional);
            groups
        }
    }
}

pub(crate) fn build_group_keys(
    dataset: &Dataset,
    grouping: &GroupingSpec,
) -> Result<Vec<String>, BiasError> {
    let columns = grouping
        .columns
        .iter()
        .map(|column| collect_stringish_column(dataset, column))
        .collect::<Result<Vec<_>, _>>()?;

    let row_count = dataset.row_count();
    let mut keys = Vec::with_capacity(row_count);
    for row_index in 0..row_count {
        if grouping.columns.len() == 1 {
            keys.push(columns[0][row_index].clone().unwrap_or_else(|| "<missing>".to_string()));
        } else {
            let parts = grouping
                .columns
                .iter()
                .zip(columns.iter())
                .map(|(column, values)| {
                    format!(
                        "{column}={}",
                        values[row_index]
                            .as_deref()
                            .unwrap_or("<missing>")
                    )
                })
                .collect::<Vec<_>>();
            keys.push(parts.join("|"));
        }
    }

    Ok(keys)
}

pub(crate) fn collect_stringish_column(
    dataset: &Dataset,
    column_name: &str,
) -> Result<Vec<Option<String>>, BiasError> {
    let column_index = dataset
        .schema()
        .index_of(column_name)
        .map_err(|_| BiasError::MissingColumn(column_name.to_string()))?;
    let data_type = dataset
        .column_type(column_name)
        .ok_or_else(|| BiasError::MissingColumn(column_name.to_string()))?;

    let mut values = Vec::with_capacity(dataset.row_count());
    for batch in dataset.batches() {
        let array = batch.column(column_index);
        extend_string_values(array, data_type, &mut values, column_name)?;
    }
    Ok(values)
}

pub(crate) fn collect_numeric_column(
    dataset: &Dataset,
    column_name: &str,
) -> Result<Vec<Option<f64>>, BiasError> {
    let column_index = dataset
        .schema()
        .index_of(column_name)
        .map_err(|_| BiasError::MissingColumn(column_name.to_string()))?;
    let data_type = dataset
        .column_type(column_name)
        .ok_or_else(|| BiasError::MissingColumn(column_name.to_string()))?;

    let mut values = Vec::with_capacity(dataset.row_count());
    for batch in dataset.batches() {
        let array = batch.column(column_index);
        extend_numeric_values(array, data_type, &mut values, column_name)?;
    }
    Ok(values)
}

pub(crate) fn collect_null_flags(
    dataset: &Dataset,
    column_name: &str,
) -> Result<Vec<bool>, BiasError> {
    let column_index = dataset
        .schema()
        .index_of(column_name)
        .map_err(|_| BiasError::MissingColumn(column_name.to_string()))?;
    let mut flags = Vec::with_capacity(dataset.row_count());
    for batch in dataset.batches() {
        let array = batch.column(column_index);
        for index in 0..array.len() {
            flags.push(array.is_null(index));
        }
    }
    Ok(flags)
}

pub(crate) fn analysis_columns(
    dataset: &Dataset,
    config: &AuditConfig,
) -> Result<Vec<String>, BiasError> {
    let mut columns = match &config.analysis_columns {
        crate::ColumnSelection::All => dataset
            .column_names()
            .into_iter()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>(),
        crate::ColumnSelection::Named(columns) => columns.clone(),
    };
    for column in &columns {
        if !dataset.has_column(column) {
            return Err(BiasError::MissingColumn(column.clone()));
        }
    }

    columns.retain(|column| !config.sensitive_columns.iter().any(|sensitive| sensitive == column));
    Ok(columns)
}

pub(crate) fn is_numeric_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Float32
            | DataType::Float64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

fn extend_string_values(
    array: &ArrayRef,
    data_type: &DataType,
    values: &mut Vec<Option<String>>,
    column_name: &str,
) -> Result<(), BiasError> {
    match data_type {
        DataType::Utf8 => push_utf8_array(
            array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values,
        ),
        DataType::LargeUtf8 => push_large_utf8_array(
            array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values,
        ),
        DataType::Boolean => push_display_values!(
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::Int8 => push_display_values!(
            array
                .as_any()
                .downcast_ref::<Int8Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::Int16 => push_display_values!(
            array
                .as_any()
                .downcast_ref::<Int16Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::Int32 => push_display_values!(
            array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::Int64 => push_display_values!(
            array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::UInt8 => push_display_values!(
            array
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::UInt16 => push_display_values!(
            array
                .as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::UInt32 => push_display_values!(
            array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::UInt64 => push_display_values!(
            array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::Float32 => push_display_values!(
            array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::Float64 => push_display_values!(
            array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        _ => return Err(unsupported(column_name, data_type)),
    }

    Ok(())
}

fn extend_numeric_values(
    array: &ArrayRef,
    data_type: &DataType,
    values: &mut Vec<Option<f64>>,
    column_name: &str,
) -> Result<(), BiasError> {
    match data_type {
        DataType::Int8 => push_numeric_values!(
            array
                .as_any()
                .downcast_ref::<Int8Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::Int16 => push_numeric_values!(
            array
                .as_any()
                .downcast_ref::<Int16Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::Int32 => push_numeric_values!(
            array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::Int64 => push_numeric_values!(
            array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::UInt8 => push_numeric_values!(
            array
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::UInt16 => push_numeric_values!(
            array
                .as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::UInt32 => push_numeric_values!(
            array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::UInt64 => push_numeric_values!(
            array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::Float32 => push_numeric_values!(
            array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        DataType::Float64 => push_numeric_values!(
            array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| unsupported(column_name, data_type))?,
            values
        ),
        _ => return Err(unsupported(column_name, data_type)),
    }

    Ok(())
}

fn push_utf8_array(array: &StringArray, values: &mut Vec<Option<String>>) {
    for index in 0..array.len() {
        if array.is_null(index) {
            values.push(None);
        } else {
            values.push(Some(array.value(index).to_string()));
        }
    }
}

fn push_large_utf8_array(array: &LargeStringArray, values: &mut Vec<Option<String>>) {
    for index in 0..array.len() {
        if array.is_null(index) {
            values.push(None);
        } else {
            values.push(Some(array.value(index).to_string()));
        }
    }
}

fn unsupported(column_name: &str, data_type: &DataType) -> BiasError {
    BiasError::UnsupportedColumnType {
        column: column_name.to_string(),
        data_type: data_type.to_string(),
    }
}
