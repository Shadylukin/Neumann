// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{ScalarValue, TensorData, TensorValue};

#[derive(Arbitrary, Debug, Clone)]
struct FuzzColumn {
    name: String,
    col_type: FuzzColumnType,
    nullable: bool,
}

#[derive(Arbitrary, Debug, Clone, Copy)]
enum FuzzColumnType {
    Int,
    Float,
    String,
    Bool,
}

#[derive(Arbitrary, Debug)]
struct SchemaInput {
    test_case: TestCase,
}

#[derive(Arbitrary, Debug)]
enum TestCase {
    ValidRoundtrip { columns: Vec<FuzzColumn> },
    ParseColumnString { column_str: String },
    ParseTypeString { type_str: String },
    MissingColumnMeta { columns: Vec<FuzzColumn>, missing_idx: u8 },
    MalformedTypeString { columns: Vec<FuzzColumn> },
}

fn is_valid_column_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.starts_with('_')
}

fuzz_target!(|input: SchemaInput| {
    match input.test_case {
        TestCase::ValidRoundtrip { columns } => {
            // Filter valid column names, limit count, and ensure uniqueness
            let mut seen_names = std::collections::HashSet::new();
            let columns: Vec<FuzzColumn> = columns
                .into_iter()
                .take(50)
                .filter(|c| is_valid_column_name(&c.name))
                .filter(|c| seen_names.insert(c.name.clone()))
                .collect();

            if columns.is_empty() {
                return;
            }

            // Serialize to TensorData
            let tensor = schema_to_tensor(&columns);

            // Parse back
            let result = tensor_to_schema(&tensor);
            assert!(result.is_ok(), "Valid schema should parse");

            let parsed = result.unwrap();
            assert_eq!(parsed.len(), columns.len(), "Column count mismatch");

            for (orig, parsed) in columns.iter().zip(parsed.iter()) {
                assert_eq!(orig.name, parsed.name);
                assert_eq!(type_to_str(orig.col_type), type_to_str(parsed.col_type));
                assert_eq!(orig.nullable, parsed.nullable);
            }
        }
        TestCase::ParseColumnString { column_str } => {
            let column_str: String = column_str.chars().take(1000).collect();
            // Split and validate each name
            for name in column_str.split(',') {
                let _ = validate_column_name(name);
            }
        }
        TestCase::ParseTypeString { type_str } => {
            let type_str: String = type_str.chars().take(100).collect();
            let _ = parse_type_string(&type_str);
        }
        TestCase::MissingColumnMeta { columns, missing_idx } => {
            let mut seen_names = std::collections::HashSet::new();
            let columns: Vec<FuzzColumn> = columns
                .into_iter()
                .take(20)
                .filter(|c| is_valid_column_name(&c.name))
                .filter(|c| seen_names.insert(c.name.clone()))
                .collect();

            if columns.is_empty() {
                return;
            }

            let missing_idx = (missing_idx as usize) % columns.len();
            let tensor = schema_to_tensor_missing(&columns, missing_idx);

            // Should fail with missing column metadata
            let result = tensor_to_schema(&tensor);
            assert!(result.is_err());
        }
        TestCase::MalformedTypeString { columns } => {
            let mut seen_names = std::collections::HashSet::new();
            let columns: Vec<FuzzColumn> = columns
                .into_iter()
                .take(10)
                .filter(|c| is_valid_column_name(&c.name))
                .filter(|c| seen_names.insert(c.name.clone()))
                .collect();

            if columns.is_empty() {
                return;
            }

            let tensor = schema_to_tensor_malformed(&columns);
            let result = tensor_to_schema(&tensor);
            assert!(result.is_err());
        }
    }
});

fn type_to_str(t: FuzzColumnType) -> &'static str {
    match t {
        FuzzColumnType::Int => "int",
        FuzzColumnType::Float => "float",
        FuzzColumnType::String => "string",
        FuzzColumnType::Bool => "bool",
    }
}

fn schema_to_tensor(columns: &[FuzzColumn]) -> TensorData {
    let mut tensor = TensorData::new();
    tensor.set(
        "_type",
        TensorValue::Scalar(ScalarValue::String("table".into())),
    );

    let column_names: Vec<&str> = columns.iter().map(|c| c.name.as_str()).collect();
    tensor.set(
        "_columns",
        TensorValue::Scalar(ScalarValue::String(column_names.join(","))),
    );

    for col in columns {
        let nullable_str = if col.nullable { "null" } else { "notnull" };
        let type_str = format!("{}:{}", type_to_str(col.col_type), nullable_str);
        tensor.set(
            format!("_col:{}", col.name),
            TensorValue::Scalar(ScalarValue::String(type_str)),
        );
    }
    tensor
}

fn schema_to_tensor_missing(columns: &[FuzzColumn], skip_idx: usize) -> TensorData {
    let mut tensor = TensorData::new();
    tensor.set(
        "_type",
        TensorValue::Scalar(ScalarValue::String("table".into())),
    );

    let column_names: Vec<&str> = columns.iter().map(|c| c.name.as_str()).collect();
    tensor.set(
        "_columns",
        TensorValue::Scalar(ScalarValue::String(column_names.join(","))),
    );

    for (i, col) in columns.iter().enumerate() {
        if i == skip_idx {
            continue; // Skip this column's metadata
        }
        let nullable_str = if col.nullable { "null" } else { "notnull" };
        let type_str = format!("{}:{}", type_to_str(col.col_type), nullable_str);
        tensor.set(
            format!("_col:{}", col.name),
            TensorValue::Scalar(ScalarValue::String(type_str)),
        );
    }
    tensor
}

fn schema_to_tensor_malformed(columns: &[FuzzColumn]) -> TensorData {
    let mut tensor = TensorData::new();
    tensor.set(
        "_type",
        TensorValue::Scalar(ScalarValue::String("table".into())),
    );

    let column_names: Vec<&str> = columns.iter().map(|c| c.name.as_str()).collect();
    tensor.set(
        "_columns",
        TensorValue::Scalar(ScalarValue::String(column_names.join(","))),
    );

    for col in columns {
        // Malformed: missing colon separator
        tensor.set(
            format!("_col:{}", col.name),
            TensorValue::Scalar(ScalarValue::String("invalidformat".into())),
        );
    }
    tensor
}

fn tensor_to_schema(tensor: &TensorData) -> Result<Vec<FuzzColumn>, &'static str> {
    let columns_str = match tensor.get("_columns") {
        Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
        _ => return Err("missing _columns"),
    };

    let mut result = Vec::new();
    for col_name in columns_str.split(',') {
        if col_name.is_empty() {
            continue;
        }
        let col_key = format!("_col:{col_name}");
        let type_str = match tensor.get(&col_key) {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
            _ => return Err("missing column metadata"),
        };

        let (col_type, nullable) = parse_type_string(&type_str)?;
        result.push(FuzzColumn {
            name: col_name.to_string(),
            col_type,
            nullable,
        });
    }
    Ok(result)
}

fn parse_type_string(s: &str) -> Result<(FuzzColumnType, bool), &'static str> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err("invalid type format");
    }
    let col_type = match parts[0] {
        "int" => FuzzColumnType::Int,
        "float" => FuzzColumnType::Float,
        "string" => FuzzColumnType::String,
        "bool" => FuzzColumnType::Bool,
        _ => return Err("unknown type"),
    };
    let nullable = parts[1] == "null";
    Ok((col_type, nullable))
}

fn validate_column_name(name: &str) -> Result<(), &'static str> {
    if name.is_empty() {
        return Err("empty name");
    }
    if name.len() > 255 {
        return Err("name too long");
    }
    if name.starts_with('_') {
        return Err("reserved prefix");
    }
    if name.contains(':') || name.contains(',') {
        return Err("invalid characters");
    }
    Ok(())
}
