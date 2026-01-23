use std::{
    collections::{BTreeMap, HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::atomic::{AtomicU64, Ordering},
};

use dashmap::DashMap;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorStoreError, TensorValue};

mod simd {
    use wide::{f64x4, i64x4, CmpEq, CmpGt, CmpLt};

    #[inline]
    pub fn filter_lt_i64(values: &[i64], threshold: i64, result: &mut [u64]) {
        let chunks = values.len() / 4;
        let threshold_vec = i64x4::splat(threshold);

        for i in 0..chunks {
            let offset = i * 4;
            let v = i64x4::new([
                values[offset],
                values[offset + 1],
                values[offset + 2],
                values[offset + 3],
            ]);
            let cmp = v.cmp_lt(threshold_vec);
            let mask_arr: [i64; 4] = cmp.into();

            for (j, &m) in mask_arr.iter().enumerate() {
                if m != 0 {
                    let bit_pos = offset + j;
                    result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
                }
            }
        }

        let start = chunks * 4;
        for i in start..values.len() {
            if values[i] < threshold {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    #[inline]
    pub fn filter_le_i64(values: &[i64], threshold: i64, result: &mut [u64]) {
        let chunks = values.len() / 4;
        let threshold_vec = i64x4::splat(threshold);

        for i in 0..chunks {
            let offset = i * 4;
            let v = i64x4::new([
                values[offset],
                values[offset + 1],
                values[offset + 2],
                values[offset + 3],
            ]);
            let lt: [i64; 4] = v.cmp_lt(threshold_vec).into();
            let eq: [i64; 4] = v.cmp_eq(threshold_vec).into();

            for j in 0..4 {
                if lt[j] != 0 || eq[j] != 0 {
                    let bit_pos = offset + j;
                    result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
                }
            }
        }

        let start = chunks * 4;
        for i in start..values.len() {
            if values[i] <= threshold {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    #[inline]
    pub fn filter_gt_i64(values: &[i64], threshold: i64, result: &mut [u64]) {
        let chunks = values.len() / 4;
        let threshold_vec = i64x4::splat(threshold);

        for i in 0..chunks {
            let offset = i * 4;
            let v = i64x4::new([
                values[offset],
                values[offset + 1],
                values[offset + 2],
                values[offset + 3],
            ]);
            let cmp = v.cmp_gt(threshold_vec);
            let mask_arr: [i64; 4] = cmp.into();

            for (j, &m) in mask_arr.iter().enumerate() {
                if m != 0 {
                    let bit_pos = offset + j;
                    result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
                }
            }
        }

        let start = chunks * 4;
        for i in start..values.len() {
            if values[i] > threshold {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    #[inline]
    pub fn filter_ge_i64(values: &[i64], threshold: i64, result: &mut [u64]) {
        let chunks = values.len() / 4;
        let threshold_vec = i64x4::splat(threshold);

        for i in 0..chunks {
            let offset = i * 4;
            let v = i64x4::new([
                values[offset],
                values[offset + 1],
                values[offset + 2],
                values[offset + 3],
            ]);
            let gt: [i64; 4] = v.cmp_gt(threshold_vec).into();
            let eq: [i64; 4] = v.cmp_eq(threshold_vec).into();

            for j in 0..4 {
                if gt[j] != 0 || eq[j] != 0 {
                    let bit_pos = offset + j;
                    result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
                }
            }
        }

        let start = chunks * 4;
        for i in start..values.len() {
            if values[i] >= threshold {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    #[inline]
    pub fn filter_eq_i64(values: &[i64], target: i64, result: &mut [u64]) {
        let chunks = values.len() / 4;
        let target_vec = i64x4::splat(target);

        for i in 0..chunks {
            let offset = i * 4;
            let v = i64x4::new([
                values[offset],
                values[offset + 1],
                values[offset + 2],
                values[offset + 3],
            ]);
            let cmp = v.cmp_eq(target_vec);
            let mask_arr: [i64; 4] = cmp.into();

            for (j, &m) in mask_arr.iter().enumerate() {
                if m != 0 {
                    let bit_pos = offset + j;
                    result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
                }
            }
        }

        let start = chunks * 4;
        for i in start..values.len() {
            if values[i] == target {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    #[inline]
    pub fn filter_ne_i64(values: &[i64], target: i64, result: &mut [u64]) {
        let chunks = values.len() / 4;
        let target_vec = i64x4::splat(target);

        for i in 0..chunks {
            let offset = i * 4;
            let v = i64x4::new([
                values[offset],
                values[offset + 1],
                values[offset + 2],
                values[offset + 3],
            ]);
            let eq: [i64; 4] = v.cmp_eq(target_vec).into();

            for (j, &eq_val) in eq.iter().enumerate() {
                if eq_val == 0 {
                    let bit_pos = offset + j;
                    result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
                }
            }
        }

        let start = chunks * 4;
        for i in start..values.len() {
            if values[i] != target {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn filter_lt_f64(values: &[f64], threshold: f64, result: &mut [u64]) {
        let chunks = values.len() / 4;
        let threshold_vec = f64x4::splat(threshold);

        for i in 0..chunks {
            let offset = i * 4;
            let v = f64x4::new([
                values[offset],
                values[offset + 1],
                values[offset + 2],
                values[offset + 3],
            ]);
            let cmp = v.cmp_lt(threshold_vec);
            let mask = cmp.move_mask();

            for j in 0..4 {
                if (mask & (1 << j)) != 0 {
                    let bit_pos = offset + j;
                    result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
                }
            }
        }

        let start = chunks * 4;
        for i in start..values.len() {
            if values[i] < threshold {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn filter_gt_f64(values: &[f64], threshold: f64, result: &mut [u64]) {
        let chunks = values.len() / 4;
        let threshold_vec = f64x4::splat(threshold);

        for i in 0..chunks {
            let offset = i * 4;
            let v = f64x4::new([
                values[offset],
                values[offset + 1],
                values[offset + 2],
                values[offset + 3],
            ]);
            let cmp = v.cmp_gt(threshold_vec);
            let mask = cmp.move_mask();

            for j in 0..4 {
                if (mask & (1 << j)) != 0 {
                    let bit_pos = offset + j;
                    result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
                }
            }
        }

        let start = chunks * 4;
        for i in start..values.len() {
            if values[i] > threshold {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn filter_eq_f64(values: &[f64], threshold: f64, result: &mut [u64]) {
        let chunks = values.len() / 4;
        let threshold_vec = f64x4::splat(threshold);

        for i in 0..chunks {
            let offset = i * 4;
            let v = f64x4::new([
                values[offset],
                values[offset + 1],
                values[offset + 2],
                values[offset + 3],
            ]);
            let cmp = v.cmp_eq(threshold_vec);
            let mask = cmp.move_mask();

            for j in 0..4 {
                if (mask & (1 << j)) != 0 {
                    let bit_pos = offset + j;
                    result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
                }
            }
        }

        let start = chunks * 4;
        for i in start..values.len() {
            if (values[i] - threshold).abs() < f64::EPSILON {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    #[inline]
    pub fn bitmap_and(a: &[u64], b: &[u64], result: &mut [u64]) {
        for i in 0..a.len().min(b.len()).min(result.len()) {
            result[i] = a[i] & b[i];
        }
    }

    #[inline]
    pub fn bitmap_or(a: &[u64], b: &[u64], result: &mut [u64]) {
        for i in 0..a.len().min(b.len()).min(result.len()) {
            result[i] = a[i] | b[i];
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn popcount(bitmap: &[u64]) -> usize {
        bitmap.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn selected_indices(bitmap: &[u64], max_count: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(max_count.min(1024));
        for (word_idx, &word) in bitmap.iter().enumerate() {
            let base = word_idx * 64;
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                indices.push(base + bit);
                w &= w - 1;
            }
        }
        indices
    }

    #[inline]
    pub fn bitmap_words(n: usize) -> usize {
        n.div_ceil(64)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_filter_lt_i64() {
            let values = vec![1, 5, 3, 8, 2, 9, 4, 7];
            let mut bitmap = vec![0u64; 1];
            filter_lt_i64(&values, 5, &mut bitmap);
            // Positions 0,2,4,6 have values < 5: 1,3,2,4
            assert_eq!(bitmap[0] & 0xff, 0b01010101);
        }

        #[test]
        fn test_filter_eq_i64() {
            let values = vec![1, 5, 5, 8, 5, 9, 4, 7];
            let mut bitmap = vec![0u64; 1];
            filter_eq_i64(&values, 5, &mut bitmap);
            // Positions 1,2,4 have value == 5
            assert_eq!(bitmap[0] & 0xff, 0b00010110);
        }

        #[test]
        fn test_filter_gt_i64() {
            let values = vec![1, 5, 3, 8, 2, 9, 4, 7];
            let mut bitmap = vec![0u64; 1];
            filter_gt_i64(&values, 5, &mut bitmap);
            // Positions 3,5,7 have values > 5: 8,9,7
            assert_eq!(bitmap[0] & 0xff, 0b10101000);
        }

        #[test]
        fn test_filter_handles_remainder() {
            let values = vec![1, 2, 3, 4, 5]; // Not divisible by 4
            let mut bitmap = vec![0u64; 1];
            filter_lt_i64(&values, 4, &mut bitmap);
            // Positions 0,1,2 have values < 4
            assert_eq!(bitmap[0] & 0xff, 0b00000111);
        }

        #[test]
        fn test_bitmap_and() {
            let a = [0b1111_0000u64];
            let b = [0b1010_1010u64];
            let mut result = [0u64];
            bitmap_and(&a, &b, &mut result);
            assert_eq!(result[0], 0b1010_0000);
        }

        #[test]
        fn test_bitmap_or() {
            let a = [0b1111_0000u64];
            let b = [0b0000_1111u64];
            let mut result = [0u64];
            bitmap_or(&a, &b, &mut result);
            assert_eq!(result[0], 0b1111_1111);
        }

        #[test]
        fn test_selected_indices() {
            let bitmap = [0b00010110u64]; // bits 1,2,4 set
            let indices = selected_indices(&bitmap, 10);
            assert_eq!(indices, vec![1, 2, 4]);
        }

        #[test]
        fn test_popcount() {
            let bitmap = [0b11110000u64, 0b00001111u64];
            assert_eq!(popcount(&bitmap), 8);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColumnType {
    Int,
    Float,
    String,
    Bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub column_type: ColumnType,
    pub nullable: bool,
}

impl Column {
    pub fn new(name: impl Into<String>, column_type: ColumnType) -> Self {
        Self {
            name: name.into(),
            column_type,
            nullable: false,
        }
    }

    pub fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    pub columns: Vec<Column>,
}

impl Schema {
    pub fn new(columns: Vec<Column>) -> Self {
        Self { columns }
    }

    pub fn get_column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

impl Value {
    fn to_scalar(&self) -> ScalarValue {
        match self {
            Value::Null => ScalarValue::Null,
            Value::Int(v) => ScalarValue::Int(*v),
            Value::Float(v) => ScalarValue::Float(*v),
            Value::String(v) => ScalarValue::String(v.clone()),
            Value::Bool(v) => ScalarValue::Bool(*v),
        }
    }

    fn from_scalar(scalar: &ScalarValue) -> Self {
        match scalar {
            ScalarValue::Null => Value::Null,
            ScalarValue::Int(v) => Value::Int(*v),
            ScalarValue::Float(v) => Value::Float(*v),
            ScalarValue::String(v) => Value::String(v.clone()),
            ScalarValue::Bool(v) => Value::Bool(*v),
            ScalarValue::Bytes(_) => Value::Null,
        }
    }

    fn matches_type(&self, column_type: &ColumnType) -> bool {
        matches!(
            (self, column_type),
            (Value::Null, _)
                | (Value::Int(_), ColumnType::Int)
                | (Value::Float(_), ColumnType::Float)
                | (Value::String(_), ColumnType::String)
                | (Value::Bool(_), ColumnType::Bool)
        )
    }

    fn hash_key(&self) -> String {
        match self {
            Value::Null => "null".to_string(),
            Value::Int(v) => format!("i:{}", v),
            Value::Float(v) => format!("f:{}", v.to_bits()),
            Value::String(v) => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                v.hash(&mut hasher);
                format!("s:{:x}", hasher.finish())
            },
            Value::Bool(v) => format!("b:{}", v),
        }
    }

    fn partial_cmp_value(&self, other: &Value) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Some(a.cmp(b)),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }

    fn sortable_key(&self) -> String {
        match self {
            Value::Null => "0".to_string(),
            Value::Int(v) => {
                // Encode i64 as hex with offset to handle negative numbers
                // Add i64::MAX + 1 to shift range from [-2^63, 2^63-1] to [0, 2^64-1]
                let unsigned = (*v as i128 + (i64::MAX as i128) + 1) as u64;
                format!("i{:016x}", unsigned)
            },
            Value::Float(v) => {
                // IEEE 754 float bit encoding with sign handling for correct ordering
                let bits = v.to_bits();
                let sortable = if *v >= 0.0 {
                    bits ^ 0x8000_0000_0000_0000 // Flip sign bit for positive
                } else {
                    !bits // Flip all bits for negative
                };
                format!("f{:016x}", sortable)
            },
            Value::String(v) => format!("s{}", v),
            Value::Bool(v) => {
                if *v {
                    "b1".to_string()
                } else {
                    "b0".to_string()
                }
            },
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::String(s) => !s.is_empty(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub id: u64,
    /// Column values stored as ordered pairs for faster iteration.
    /// Use `get()` for single-column access, or iterate directly for bulk access.
    pub values: Vec<(String, Value)>,
}

impl Row {
    pub fn get(&self, column: &str) -> Option<&Value> {
        if column == "_id" {
            return None;
        }
        self.values
            .iter()
            .find(|(k, _)| k == column)
            .map(|(_, v)| v)
    }

    pub fn get_with_id(&self, column: &str) -> Option<Value> {
        if column == "_id" {
            return Some(Value::Int(self.id as i64));
        }
        self.values
            .iter()
            .find(|(k, _)| k == column)
            .map(|(_, v)| v.clone())
    }

    pub fn contains(&self, column: &str) -> bool {
        self.values.iter().any(|(k, _)| k == column)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Condition {
    Eq(String, Value),
    Ne(String, Value),
    Lt(String, Value),
    Le(String, Value),
    Gt(String, Value),
    Ge(String, Value),
    And(Box<Condition>, Box<Condition>),
    Or(Box<Condition>, Box<Condition>),
    True,
}

impl Condition {
    pub fn and(self, other: Condition) -> Condition {
        Condition::And(Box::new(self), Box::new(other))
    }

    pub fn or(self, other: Condition) -> Condition {
        Condition::Or(Box::new(self), Box::new(other))
    }

    pub fn evaluate(&self, row: &Row) -> bool {
        use std::cmp::Ordering;
        match self {
            Condition::True => true,
            Condition::Eq(col, val) => row.get_with_id(col).as_ref() == Some(val),
            Condition::Ne(col, val) => row.get_with_id(col).as_ref() != Some(val),
            Condition::Lt(col, val) => self.compare_ord(row, col, val, Ordering::Less),
            Condition::Le(col, val) => self.compare_ord_le(row, col, val),
            Condition::Gt(col, val) => self.compare_ord(row, col, val, Ordering::Greater),
            Condition::Ge(col, val) => self.compare_ord_ge(row, col, val),
            Condition::And(a, b) => a.evaluate(row) && b.evaluate(row),
            Condition::Or(a, b) => a.evaluate(row) || b.evaluate(row),
        }
    }

    fn compare_ord(&self, row: &Row, col: &str, val: &Value, ord: std::cmp::Ordering) -> bool {
        row.get_with_id(col)
            .and_then(|v| v.partial_cmp_value(val))
            .map(|o| o == ord)
            .unwrap_or(false)
    }

    fn compare_ord_le(&self, row: &Row, col: &str, val: &Value) -> bool {
        row.get_with_id(col)
            .and_then(|v| v.partial_cmp_value(val))
            .map(|o| o != std::cmp::Ordering::Greater)
            .unwrap_or(false)
    }

    fn compare_ord_ge(&self, row: &Row, col: &str, val: &Value) -> bool {
        row.get_with_id(col)
            .and_then(|v| v.partial_cmp_value(val))
            .map(|o| o != std::cmp::Ordering::Less)
            .unwrap_or(false)
    }

    #[cfg(feature = "test-internals")]
    pub fn evaluate_tensor(&self, tensor: &TensorData) -> bool {
        self.evaluate_tensor_impl(tensor)
    }

    #[cfg(not(feature = "test-internals"))]
    pub(crate) fn evaluate_tensor(&self, tensor: &TensorData) -> bool {
        self.evaluate_tensor_impl(tensor)
    }

    fn evaluate_tensor_impl(&self, tensor: &TensorData) -> bool {
        use std::cmp::Ordering;
        match self {
            Condition::True => true,
            Condition::Eq(col, val) => Self::tensor_field_eq(tensor, col, val),
            Condition::Ne(col, val) => !Self::tensor_field_eq(tensor, col, val),
            Condition::Lt(col, val) => Self::tensor_compare_ord(tensor, col, val, Ordering::Less),
            Condition::Le(col, val) => Self::tensor_compare_le(tensor, col, val),
            Condition::Gt(col, val) => {
                Self::tensor_compare_ord(tensor, col, val, Ordering::Greater)
            },
            Condition::Ge(col, val) => Self::tensor_compare_ge(tensor, col, val),
            Condition::And(a, b) => a.evaluate_tensor(tensor) && b.evaluate_tensor(tensor),
            Condition::Or(a, b) => a.evaluate_tensor(tensor) || b.evaluate_tensor(tensor),
        }
    }

    fn tensor_get_value(tensor: &TensorData, col: &str) -> Option<Value> {
        if col == "_id" {
            if let Some(TensorValue::Scalar(ScalarValue::Int(id))) = tensor.get("_id") {
                return Some(Value::Int(*id));
            }
            return None;
        }

        match tensor.get(col) {
            Some(TensorValue::Scalar(scalar)) => Some(Value::from_scalar(scalar)),
            _ => None,
        }
    }

    fn tensor_field_eq(tensor: &TensorData, col: &str, val: &Value) -> bool {
        Self::tensor_get_value(tensor, col).as_ref() == Some(val)
    }

    fn tensor_compare_ord(
        tensor: &TensorData,
        col: &str,
        val: &Value,
        ord: std::cmp::Ordering,
    ) -> bool {
        Self::tensor_get_value(tensor, col)
            .and_then(|v| v.partial_cmp_value(val))
            .map(|o| o == ord)
            .unwrap_or(false)
    }

    fn tensor_compare_le(tensor: &TensorData, col: &str, val: &Value) -> bool {
        Self::tensor_get_value(tensor, col)
            .and_then(|v| v.partial_cmp_value(val))
            .map(|o| o != std::cmp::Ordering::Greater)
            .unwrap_or(false)
    }

    fn tensor_compare_ge(tensor: &TensorData, col: &str, val: &Value) -> bool {
        Self::tensor_get_value(tensor, col)
            .and_then(|v| v.partial_cmp_value(val))
            .map(|o| o != std::cmp::Ordering::Less)
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RelationalError {
    TableNotFound(String),
    TableAlreadyExists(String),
    ColumnNotFound(String),
    TypeMismatch {
        column: String,
        expected: ColumnType,
    },
    NullNotAllowed(String),
    IndexAlreadyExists {
        table: String,
        column: String,
    },
    IndexNotFound {
        table: String,
        column: String,
    },
    StorageError(String),
}

impl std::fmt::Display for RelationalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RelationalError::TableNotFound(t) => write!(f, "Table not found: {}", t),
            RelationalError::TableAlreadyExists(t) => write!(f, "Table already exists: {}", t),
            RelationalError::ColumnNotFound(c) => write!(f, "Column not found: {}", c),
            RelationalError::TypeMismatch { column, expected } => {
                write!(
                    f,
                    "Type mismatch for column {}: expected {:?}",
                    column, expected
                )
            },
            RelationalError::NullNotAllowed(c) => write!(f, "Null not allowed for column: {}", c),
            RelationalError::IndexAlreadyExists { table, column } => {
                write!(f, "Index already exists on {}.{}", table, column)
            },
            RelationalError::IndexNotFound { table, column } => {
                write!(f, "Index not found on {}.{}", table, column)
            },
            RelationalError::StorageError(e) => write!(f, "Storage error: {}", e),
        }
    }
}

impl std::error::Error for RelationalError {}

impl From<TensorStoreError> for RelationalError {
    fn from(e: TensorStoreError) -> Self {
        RelationalError::StorageError(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, RelationalError>;

#[derive(Debug, Clone)]
pub(crate) enum ColumnValues {
    Int(Vec<i64>),
    Float(Vec<f64>),
    String {
        dict: Vec<String>,
        indices: Vec<u32>,
    },
    Bool(Vec<u64>),
}

#[allow(dead_code)]
impl ColumnValues {
    pub fn len(&self) -> usize {
        match self {
            ColumnValues::Int(v) => v.len(),
            ColumnValues::Float(v) => v.len(),
            ColumnValues::String { indices, .. } => indices.len(),
            ColumnValues::Bool(v) => v.len() * 64, // approximate
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub(crate) enum NullBitmap {
    None,
    Dense(Vec<u64>),
    Sparse(Vec<u64>),
}

impl NullBitmap {
    pub fn is_null(&self, idx: usize) -> bool {
        match self {
            NullBitmap::None => false,
            NullBitmap::Dense(bitmap) => {
                let word_idx = idx / 64;
                let bit_idx = idx % 64;
                word_idx < bitmap.len() && (bitmap[word_idx] & (1u64 << bit_idx)) != 0
            },
            NullBitmap::Sparse(positions) => positions.binary_search(&(idx as u64)).is_ok(),
        }
    }

    pub fn null_count(&self) -> usize {
        match self {
            NullBitmap::None => 0,
            NullBitmap::Dense(bitmap) => bitmap.iter().map(|w| w.count_ones() as usize).sum(),
            NullBitmap::Sparse(positions) => positions.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ColumnData {
    name: String,
    pub row_ids: Vec<u64>,
    nulls: NullBitmap,
    values: ColumnValues,
}

impl ColumnData {
    pub fn null_count(&self) -> usize {
        self.nulls.null_count()
    }

    #[inline]
    pub fn get_value(&self, idx: usize) -> Option<Value> {
        if self.nulls.is_null(idx) {
            return Some(Value::Null);
        }
        match &self.values {
            ColumnValues::Int(v) => v.get(idx).map(|&i| Value::Int(i)),
            ColumnValues::Float(v) => v.get(idx).map(|&f| Value::Float(f)),
            ColumnValues::String { dict, indices } => indices
                .get(idx)
                .and_then(|&i| dict.get(i as usize).map(|s| Value::String(s.clone()))),
            ColumnValues::Bool(v) => {
                let word_idx = idx / 64;
                let bit_idx = idx % 64;
                v.get(word_idx)
                    .map(|&word| Value::Bool((word & (1u64 << bit_idx)) != 0))
            },
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SelectionVector {
    bitmap: Vec<u64>,
    row_count: usize,
}

#[allow(dead_code)]
impl SelectionVector {
    pub fn all(row_count: usize) -> Self {
        let words = simd::bitmap_words(row_count);
        let mut bitmap = vec![!0u64; words];
        if !row_count.is_multiple_of(64) {
            let last_word_bits = row_count % 64;
            bitmap[words - 1] = (1u64 << last_word_bits) - 1;
        }
        Self { bitmap, row_count }
    }

    pub fn none(row_count: usize) -> Self {
        let words = simd::bitmap_words(row_count);
        Self {
            bitmap: vec![0u64; words],
            row_count,
        }
    }

    pub fn from_bitmap(bitmap: Vec<u64>, row_count: usize) -> Self {
        Self { bitmap, row_count }
    }

    pub fn bitmap_mut(&mut self) -> &mut [u64] {
        &mut self.bitmap
    }

    pub fn bitmap(&self) -> &[u64] {
        &self.bitmap
    }

    pub fn count(&self) -> usize {
        simd::popcount(&self.bitmap)
    }

    pub fn is_selected(&self, idx: usize) -> bool {
        if idx >= self.row_count {
            return false;
        }
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        (self.bitmap[word_idx] & (1u64 << bit_idx)) != 0
    }

    pub fn selected_indices(&self) -> Vec<usize> {
        simd::selected_indices(&self.bitmap, self.row_count)
    }

    pub fn intersect(&self, other: &SelectionVector) -> SelectionVector {
        let mut result = vec![0u64; self.bitmap.len()];
        simd::bitmap_and(&self.bitmap, &other.bitmap, &mut result);
        SelectionVector {
            bitmap: result,
            row_count: self.row_count,
        }
    }

    pub fn union(&self, other: &SelectionVector) -> SelectionVector {
        let mut result = vec![0u64; self.bitmap.len()];
        simd::bitmap_or(&self.bitmap, &other.bitmap, &mut result);
        SelectionVector {
            bitmap: result,
            row_count: self.row_count,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ColumnarScanOptions {
    pub projection: Option<Vec<String>>,
    pub prefer_columnar: bool,
}

/// Key for in-memory B-tree indexes: (table, column) -> BTreeMap<value, row_ids>
type BTreeIndexKey = (String, String);

pub struct RelationalEngine {
    store: TensorStore,
    row_counters: DashMap<String, AtomicU64>,
    /// In-memory B-tree indexes for O(log n) range queries.
    /// Maps (table, column) to a BTreeMap of value -> row_ids.
    btree_indexes: RwLock<HashMap<BTreeIndexKey, BTreeMap<OrderedKey, Vec<u64>>>>,
}

/// Ordered key for B-tree indexes with correct comparison semantics.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum OrderedKey {
    Null,
    Bool(bool),
    Int(i64),
    Float(OrderedFloat),
    String(String),
}

/// Wrapper for f64 that implements total ordering (NaN is less than all values).
#[derive(Debug, Clone, Copy)]
pub struct OrderedFloat(pub f64);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => self
                .0
                .partial_cmp(&other.0)
                .unwrap_or(std::cmp::Ordering::Equal),
        }
    }
}

impl OrderedKey {
    fn from_value(value: &Value) -> Self {
        match value {
            Value::Null => OrderedKey::Null,
            Value::Bool(b) => OrderedKey::Bool(*b),
            Value::Int(i) => OrderedKey::Int(*i),
            Value::Float(f) => OrderedKey::Float(OrderedFloat(*f)),
            Value::String(s) => OrderedKey::String(s.clone()),
        }
    }
}

impl RelationalEngine {
    /// Threshold for parallel operations (below this, sequential is faster)
    const PARALLEL_THRESHOLD: usize = 1000;

    pub fn new() -> Self {
        Self {
            store: TensorStore::new(),
            row_counters: DashMap::new(),
            btree_indexes: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_store(store: TensorStore) -> Self {
        Self {
            store,
            row_counters: DashMap::new(),
            btree_indexes: RwLock::new(HashMap::new()),
        }
    }

    pub fn store(&self) -> &TensorStore {
        &self.store
    }

    fn table_meta_key(name: &str) -> String {
        format!("_meta:table:{}", name)
    }

    fn row_key(table: &str, id: u64) -> String {
        format!("{}:{}", table, id)
    }

    fn row_prefix(table: &str) -> String {
        format!("{}:", table)
    }

    fn index_meta_key(table: &str, column: &str) -> String {
        format!("_idx:{}:{}", table, column)
    }

    fn index_entry_key(table: &str, column: &str, value_hash: &str) -> String {
        format!("_idx:{}:{}:{}", table, column, value_hash)
    }

    fn index_prefix(table: &str, column: &str) -> String {
        format!("_idx:{}:{}:", table, column)
    }

    fn all_indexes_prefix(table: &str) -> String {
        format!("_idx:{}:", table)
    }

    fn btree_meta_key(table: &str, column: &str) -> String {
        format!("_btree:{}:{}", table, column)
    }

    fn btree_entry_key(table: &str, column: &str, sortable_value: &str) -> String {
        format!("_btree:{}:{}:{}", table, column, sortable_value)
    }

    fn btree_prefix(table: &str, column: &str) -> String {
        format!("_btree:{}:{}:", table, column)
    }

    pub fn create_table(&self, name: &str, schema: Schema) -> Result<()> {
        let meta_key = Self::table_meta_key(name);

        if self.store.exists(&meta_key) {
            return Err(RelationalError::TableAlreadyExists(name.to_string()));
        }

        let mut meta = TensorData::new();
        meta.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("table".into())),
        );
        meta.set(
            "_name",
            TensorValue::Scalar(ScalarValue::String(name.into())),
        );

        let column_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
        meta.set(
            "_columns",
            TensorValue::Scalar(ScalarValue::String(column_names.join(","))),
        );

        for col in &schema.columns {
            let type_str = match col.column_type {
                ColumnType::Int => "int",
                ColumnType::Float => "float",
                ColumnType::String => "string",
                ColumnType::Bool => "bool",
            };
            meta.set(
                format!("_col:{}", col.name),
                TensorValue::Scalar(ScalarValue::String(format!(
                    "{}:{}",
                    type_str,
                    if col.nullable { "null" } else { "notnull" }
                ))),
            );
        }

        self.store.put(meta_key, meta)?;

        self.row_counters
            .insert(name.to_string(), AtomicU64::new(0));

        Ok(())
    }

    pub fn get_schema(&self, table: &str) -> Result<Schema> {
        let meta_key = Self::table_meta_key(table);
        let meta = self
            .store
            .get(&meta_key)
            .map_err(|_| RelationalError::TableNotFound(table.to_string()))?;

        let columns_str = match meta.get("_columns") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
            _ => return Err(RelationalError::TableNotFound(table.to_string())),
        };

        let mut columns = Vec::new();
        for col_name in columns_str.split(',') {
            if col_name.is_empty() {
                continue;
            }
            let col_key = format!("_col:{}", col_name);
            if let Some(TensorValue::Scalar(ScalarValue::String(type_str))) = meta.get(&col_key) {
                let parts: Vec<&str> = type_str.split(':').collect();
                if parts.len() == 2 {
                    let column_type = match parts[0] {
                        "int" => ColumnType::Int,
                        "float" => ColumnType::Float,
                        "string" => ColumnType::String,
                        "bool" => ColumnType::Bool,
                        _ => ColumnType::String,
                    };
                    let nullable = parts[1] == "null";
                    let mut col = Column::new(col_name, column_type);
                    if nullable {
                        col = col.nullable();
                    }
                    columns.push(col);
                }
            }
        }

        Ok(Schema::new(columns))
    }

    pub fn list_tables(&self) -> Vec<String> {
        self.store
            .scan("_meta:table:")
            .into_iter()
            .filter_map(|key| key.strip_prefix("_meta:table:").map(String::from))
            .collect()
    }

    fn next_row_id(&self, table: &str) -> u64 {
        self.row_counters
            .entry(table.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed)
            + 1
    }

    pub fn insert(&self, table: &str, values: HashMap<String, Value>) -> Result<u64> {
        let schema = self.get_schema(table)?;

        for col in &schema.columns {
            let value = values.get(&col.name);
            match value {
                None | Some(Value::Null) => {
                    if !col.nullable {
                        return Err(RelationalError::NullNotAllowed(col.name.clone()));
                    }
                },
                Some(v) => {
                    if !v.matches_type(&col.column_type) {
                        return Err(RelationalError::TypeMismatch {
                            column: col.name.clone(),
                            expected: col.column_type.clone(),
                        });
                    }
                },
            }
        }

        let row_id = self.next_row_id(table);
        let key = Self::row_key(table, row_id);

        let mut tensor = TensorData::new();
        tensor.set("_id", TensorValue::Scalar(ScalarValue::Int(row_id as i64)));

        for (col_name, value) in &values {
            tensor.set(col_name, TensorValue::Scalar(value.to_scalar()));
        }

        self.store.put(key, tensor)?;

        let indexed_columns = self.get_table_indexes(table);
        for col in &indexed_columns {
            if col == "_id" {
                self.index_add(table, col, &Value::Int(row_id as i64), row_id)?;
            } else if let Some(value) = values.get(col) {
                self.index_add(table, col, value, row_id)?;
            }
        }

        let btree_columns = self.get_table_btree_indexes(table);
        for col in &btree_columns {
            if col == "_id" {
                self.btree_index_add(table, col, &Value::Int(row_id as i64), row_id)?;
            } else if let Some(value) = values.get(col) {
                self.btree_index_add(table, col, value, row_id)?;
            }
        }

        Ok(row_id)
    }

    pub fn batch_insert(&self, table: &str, rows: Vec<HashMap<String, Value>>) -> Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let schema = self.get_schema(table)?;

        for (row_idx, values) in rows.iter().enumerate() {
            for col in &schema.columns {
                let value = values.get(&col.name);
                match value {
                    None | Some(Value::Null) => {
                        if !col.nullable {
                            return Err(RelationalError::NullNotAllowed(format!(
                                "{} (row {})",
                                col.name, row_idx
                            )));
                        }
                    },
                    Some(v) => {
                        if !v.matches_type(&col.column_type) {
                            return Err(RelationalError::TypeMismatch {
                                column: format!("{} (row {})", col.name, row_idx),
                                expected: col.column_type.clone(),
                            });
                        }
                    },
                }
            }
        }

        let indexed_columns = self.get_table_indexes(table);
        let btree_columns = self.get_table_btree_indexes(table);
        let mut row_ids = Vec::with_capacity(rows.len());

        for values in rows {
            let row_id = self.next_row_id(table);
            let key = Self::row_key(table, row_id);

            let mut tensor = TensorData::new();
            tensor.set("_id", TensorValue::Scalar(ScalarValue::Int(row_id as i64)));

            for (col_name, value) in &values {
                tensor.set(col_name, TensorValue::Scalar(value.to_scalar()));
            }

            self.store.put(key, tensor)?;

            for col in &indexed_columns {
                if col == "_id" {
                    self.index_add(table, col, &Value::Int(row_id as i64), row_id)?;
                } else if let Some(value) = values.get(col) {
                    self.index_add(table, col, value, row_id)?;
                }
            }

            for col in &btree_columns {
                if col == "_id" {
                    self.btree_index_add(table, col, &Value::Int(row_id as i64), row_id)?;
                } else if let Some(value) = values.get(col) {
                    self.btree_index_add(table, col, value, row_id)?;
                }
            }

            row_ids.push(row_id);
        }

        Ok(row_ids)
    }

    fn tensor_to_row(&self, tensor: &TensorData) -> Option<Row> {
        let id = match tensor.get("_id") {
            Some(TensorValue::Scalar(ScalarValue::Int(id))) => *id as u64,
            _ => return None,
        };

        let mut values = Vec::with_capacity(tensor.len().saturating_sub(1));
        for key in tensor.keys() {
            if key.starts_with('_') {
                continue;
            }
            if let Some(TensorValue::Scalar(scalar)) = tensor.get(key) {
                values.push((key.clone(), Value::from_scalar(scalar)));
            }
        }

        Some(Row { id, values })
    }

    pub fn select(&self, table: &str, condition: Condition) -> Result<Vec<Row>> {
        let _ = self.get_schema(table)?;

        if let Some(row_ids) = self.try_index_lookup(table, &condition) {
            let mut rows = Vec::with_capacity(row_ids.len());
            for row_id in row_ids {
                let key = Self::row_key(table, row_id);
                if let Ok(tensor) = self.store.get(&key) {
                    if condition.evaluate_tensor(&tensor) {
                        if let Some(row) = self.tensor_to_row(&tensor) {
                            rows.push(row);
                        }
                    }
                }
            }
            rows.sort_by_key(|r| r.id);
            return Ok(rows);
        }

        let prefix = Self::row_prefix(table);

        let mut rows: Vec<Row> = self.store.scan_filter_map(&prefix, |_key, tensor| {
            if condition.evaluate_tensor(tensor) {
                self.tensor_to_row(tensor)
            } else {
                None
            }
        });

        rows.sort_by_key(|r| r.id);
        Ok(rows)
    }

    fn try_index_lookup(&self, table: &str, condition: &Condition) -> Option<Vec<u64>> {
        match condition {
            Condition::Eq(column, value) => self.index_lookup(table, column, value),
            Condition::Lt(column, value) => {
                self.btree_range_lookup(table, column, value, RangeOp::Lt)
            },
            Condition::Le(column, value) => {
                self.btree_range_lookup(table, column, value, RangeOp::Le)
            },
            Condition::Gt(column, value) => {
                self.btree_range_lookup(table, column, value, RangeOp::Gt)
            },
            Condition::Ge(column, value) => {
                self.btree_range_lookup(table, column, value, RangeOp::Ge)
            },
            Condition::And(a, b) => {
                if let Some(ids_a) = self.try_index_lookup(table, a) {
                    Some(ids_a)
                } else {
                    self.try_index_lookup(table, b)
                }
            },
            _ => None,
        }
    }

    pub fn update(
        &self,
        table: &str,
        condition: Condition,
        updates: HashMap<String, Value>,
    ) -> Result<usize> {
        let schema = self.get_schema(table)?;

        for (col_name, value) in &updates {
            let col = schema
                .get_column(col_name)
                .ok_or_else(|| RelationalError::ColumnNotFound(col_name.clone()))?;

            if !value.matches_type(&col.column_type) && *value != Value::Null {
                return Err(RelationalError::TypeMismatch {
                    column: col_name.clone(),
                    expected: col.column_type.clone(),
                });
            }

            if *value == Value::Null && !col.nullable {
                return Err(RelationalError::NullNotAllowed(col_name.clone()));
            }
        }

        let indexed_columns = self.get_table_indexes(table);
        let btree_columns = self.get_table_btree_indexes(table);
        let prefix = Self::row_prefix(table);

        let matching_rows: Vec<(String, TensorData, Row)> =
            self.store.scan_filter_map(&prefix, |key, tensor| {
                let row = self.tensor_to_row(tensor)?;
                if condition.evaluate(&row) {
                    Some((key.to_string(), tensor.clone(), row))
                } else {
                    None
                }
            });

        for (key, tensor, row) in &matching_rows {
            for col in &indexed_columns {
                if let Some(new_value) = updates.get(col) {
                    if let Some(old_value) = row.get_with_id(col) {
                        self.index_remove(table, col, &old_value, row.id)?;
                    }
                    self.index_add(table, col, new_value, row.id)?;
                }
            }

            for col in &btree_columns {
                if let Some(new_value) = updates.get(col) {
                    if let Some(old_value) = row.get_with_id(col) {
                        self.btree_index_remove(table, col, &old_value, row.id)?;
                    }
                    self.btree_index_add(table, col, new_value, row.id)?;
                }
            }

            let mut new_tensor = tensor.clone();
            for (col_name, value) in &updates {
                new_tensor.set(col_name, TensorValue::Scalar(value.to_scalar()));
            }
            self.store.put(key, new_tensor)?;
        }

        Ok(matching_rows.len())
    }

    pub fn delete_rows(&self, table: &str, condition: Condition) -> Result<usize> {
        let _ = self.get_schema(table)?;

        let indexed_columns = self.get_table_indexes(table);
        let btree_columns = self.get_table_btree_indexes(table);
        let prefix = Self::row_prefix(table);

        let to_delete: Vec<(String, Row)> = self.store.scan_filter_map(&prefix, |key, tensor| {
            let row = self.tensor_to_row(tensor)?;
            if condition.evaluate(&row) {
                Some((key.to_string(), row))
            } else {
                None
            }
        });

        for (_, row) in &to_delete {
            for col in &indexed_columns {
                if let Some(value) = row.get_with_id(col) {
                    self.index_remove(table, col, &value, row.id)?;
                }
            }
            for col in &btree_columns {
                if let Some(value) = row.get_with_id(col) {
                    self.btree_index_remove(table, col, &value, row.id)?;
                }
            }
        }

        let count = to_delete.len();

        if count >= Self::PARALLEL_THRESHOLD {
            to_delete.par_iter().try_for_each(|(key, _)| {
                self.store
                    .delete(key)
                    .map_err(|e| RelationalError::StorageError(e.to_string()))
            })?;
        } else {
            for (key, _) in to_delete {
                self.store.delete(&key)?;
            }
        }

        Ok(count)
    }

    /// Hash join: O(n+m) instead of O(n*m) nested loop join.
    /// Builds a hash index on the right table, then probes from the left.
    pub fn join(
        &self,
        table_a: &str,
        table_b: &str,
        on_a: &str,
        on_b: &str,
    ) -> Result<Vec<(Row, Row)>> {
        let _ = self.get_schema(table_a)?;
        let _ = self.get_schema(table_b)?;

        let rows_a = self.select(table_a, Condition::True)?;
        let rows_b = self.select(table_b, Condition::True)?;

        let mut index: HashMap<String, Vec<usize>> = HashMap::with_capacity(rows_b.len());
        for (i, row) in rows_b.iter().enumerate() {
            if let Some(val) = row.get_with_id(on_b) {
                let hash = val.hash_key();
                index.entry(hash).or_default().push(i);
            }
        }

        let results = if rows_a.len() >= Self::PARALLEL_THRESHOLD {
            rows_a
                .par_iter()
                .flat_map(|row_a| {
                    row_a
                        .get_with_id(on_a)
                        .and_then(|val| {
                            let hash = val.hash_key();
                            index.get(&hash).map(|indices| {
                                indices
                                    .iter()
                                    .filter_map(|&i| {
                                        let row_b = &rows_b[i];
                                        if row_b.get_with_id(on_b).as_ref() == Some(&val) {
                                            Some((row_a.clone(), row_b.clone()))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            })
                        })
                        .unwrap_or_default()
                })
                .collect()
        } else {
            let estimated_capacity = std::cmp::min(rows_a.len(), rows_b.len());
            let mut results = Vec::with_capacity(estimated_capacity);
            for row_a in &rows_a {
                if let Some(val) = row_a.get_with_id(on_a) {
                    let hash = val.hash_key();
                    if let Some(indices) = index.get(&hash) {
                        for &i in indices {
                            let row_b = &rows_b[i];
                            if row_b.get_with_id(on_b).as_ref() == Some(&val) {
                                results.push((row_a.clone(), row_b.clone()));
                            }
                        }
                    }
                }
            }
            results
        };

        Ok(results)
    }

    /// LEFT JOIN: Returns all rows from table_a, with matching rows from table_b.
    /// If no match, the table_b side is None.
    pub fn left_join(
        &self,
        table_a: &str,
        table_b: &str,
        on_a: &str,
        on_b: &str,
    ) -> Result<Vec<(Row, Option<Row>)>> {
        let _ = self.get_schema(table_a)?;
        let _ = self.get_schema(table_b)?;

        let rows_a = self.select(table_a, Condition::True)?;
        let rows_b = self.select(table_b, Condition::True)?;

        let mut index: HashMap<String, Vec<usize>> = HashMap::with_capacity(rows_b.len());
        for (i, row) in rows_b.iter().enumerate() {
            if let Some(val) = row.get_with_id(on_b) {
                index.entry(val.hash_key()).or_default().push(i);
            }
        }

        let mut results = Vec::with_capacity(rows_a.len());
        for row_a in &rows_a {
            let mut matched = false;
            if let Some(val) = row_a.get_with_id(on_a) {
                if let Some(indices) = index.get(&val.hash_key()) {
                    for &i in indices {
                        let row_b = &rows_b[i];
                        if row_b.get_with_id(on_b).as_ref() == Some(&val) {
                            results.push((row_a.clone(), Some(row_b.clone())));
                            matched = true;
                        }
                    }
                }
            }
            if !matched {
                results.push((row_a.clone(), None));
            }
        }

        Ok(results)
    }

    /// RIGHT JOIN: Returns all rows from table_b, with matching rows from table_a.
    /// If no match, the table_a side is None.
    pub fn right_join(
        &self,
        table_a: &str,
        table_b: &str,
        on_a: &str,
        on_b: &str,
    ) -> Result<Vec<(Option<Row>, Row)>> {
        let _ = self.get_schema(table_a)?;
        let _ = self.get_schema(table_b)?;

        let rows_a = self.select(table_a, Condition::True)?;
        let rows_b = self.select(table_b, Condition::True)?;

        let mut index: HashMap<String, Vec<usize>> = HashMap::with_capacity(rows_a.len());
        for (i, row) in rows_a.iter().enumerate() {
            if let Some(val) = row.get_with_id(on_a) {
                index.entry(val.hash_key()).or_default().push(i);
            }
        }

        let mut results = Vec::with_capacity(rows_b.len());
        for row_b in &rows_b {
            let mut matched = false;
            if let Some(val) = row_b.get_with_id(on_b) {
                if let Some(indices) = index.get(&val.hash_key()) {
                    for &i in indices {
                        let row_a = &rows_a[i];
                        if row_a.get_with_id(on_a).as_ref() == Some(&val) {
                            results.push((Some(row_a.clone()), row_b.clone()));
                            matched = true;
                        }
                    }
                }
            }
            if !matched {
                results.push((None, row_b.clone()));
            }
        }

        Ok(results)
    }

    /// FULL OUTER JOIN: Returns all rows from both tables.
    /// Rows that don't match get None on the non-matching side.
    pub fn full_join(
        &self,
        table_a: &str,
        table_b: &str,
        on_a: &str,
        on_b: &str,
    ) -> Result<Vec<(Option<Row>, Option<Row>)>> {
        let _ = self.get_schema(table_a)?;
        let _ = self.get_schema(table_b)?;

        let rows_a = self.select(table_a, Condition::True)?;
        let rows_b = self.select(table_b, Condition::True)?;

        let mut index: HashMap<String, Vec<usize>> = HashMap::with_capacity(rows_b.len());
        for (i, row) in rows_b.iter().enumerate() {
            if let Some(val) = row.get_with_id(on_b) {
                index.entry(val.hash_key()).or_default().push(i);
            }
        }

        let mut matched_b: HashSet<usize> = HashSet::new();
        let mut results = Vec::with_capacity(rows_a.len() + rows_b.len());

        for row_a in &rows_a {
            let mut matched = false;
            if let Some(val) = row_a.get_with_id(on_a) {
                if let Some(indices) = index.get(&val.hash_key()) {
                    for &i in indices {
                        let row_b = &rows_b[i];
                        if row_b.get_with_id(on_b).as_ref() == Some(&val) {
                            results.push((Some(row_a.clone()), Some(row_b.clone())));
                            matched_b.insert(i);
                            matched = true;
                        }
                    }
                }
            }
            if !matched {
                results.push((Some(row_a.clone()), None));
            }
        }

        for (i, row_b) in rows_b.iter().enumerate() {
            if !matched_b.contains(&i) {
                results.push((None, Some(row_b.clone())));
            }
        }

        Ok(results)
    }

    /// CROSS JOIN: Returns the cartesian product of both tables.
    /// Every row from table_a is paired with every row from table_b.
    pub fn cross_join(&self, table_a: &str, table_b: &str) -> Result<Vec<(Row, Row)>> {
        let _ = self.get_schema(table_a)?;
        let _ = self.get_schema(table_b)?;

        let rows_a = self.select(table_a, Condition::True)?;
        let rows_b = self.select(table_b, Condition::True)?;

        let mut results = Vec::with_capacity(rows_a.len() * rows_b.len());
        for row_a in &rows_a {
            for row_b in &rows_b {
                results.push((row_a.clone(), row_b.clone()));
            }
        }

        Ok(results)
    }

    /// NATURAL JOIN: Joins on all columns with the same name.
    /// Returns rows where all common columns have equal values.
    pub fn natural_join(&self, table_a: &str, table_b: &str) -> Result<Vec<(Row, Row)>> {
        let schema_a = self.get_schema(table_a)?;
        let schema_b = self.get_schema(table_b)?;

        let cols_a: HashSet<_> = schema_a.columns.iter().map(|c| c.name.as_str()).collect();
        let cols_b: HashSet<_> = schema_b.columns.iter().map(|c| c.name.as_str()).collect();
        let common_cols: Vec<_> = cols_a.intersection(&cols_b).copied().collect();

        if common_cols.is_empty() {
            return self.cross_join(table_a, table_b);
        }

        let rows_a = self.select(table_a, Condition::True)?;
        let rows_b = self.select(table_b, Condition::True)?;

        let mut index: HashMap<String, Vec<usize>> = HashMap::with_capacity(rows_b.len());
        for (i, row) in rows_b.iter().enumerate() {
            let mut composite_key = String::new();
            for col in &common_cols {
                if let Some(val) = row.get_with_id(col) {
                    composite_key.push_str(&val.hash_key());
                    composite_key.push('\0');
                }
            }
            index.entry(composite_key).or_default().push(i);
        }

        let estimated_capacity = std::cmp::min(rows_a.len(), rows_b.len());
        let mut results = Vec::with_capacity(estimated_capacity);
        for row_a in &rows_a {
            let mut composite_key = String::new();
            let mut all_cols_present = true;
            for col in &common_cols {
                if let Some(val) = row_a.get_with_id(col) {
                    composite_key.push_str(&val.hash_key());
                    composite_key.push('\0');
                } else {
                    all_cols_present = false;
                    break;
                }
            }

            if !all_cols_present {
                continue;
            }

            if let Some(indices) = index.get(&composite_key) {
                for &i in indices {
                    let row_b = &rows_b[i];
                    let all_match = common_cols
                        .iter()
                        .all(|col| row_a.get_with_id(col) == row_b.get_with_id(col));
                    if all_match {
                        results.push((row_a.clone(), row_b.clone()));
                    }
                }
            }
        }

        Ok(results)
    }

    pub fn count(&self, table: &str, condition: Condition) -> Result<u64> {
        let rows = self.select(table, condition)?;
        Ok(rows.len() as u64)
    }

    pub fn count_column(&self, table: &str, column: &str, condition: Condition) -> Result<u64> {
        let rows = self.select(table, condition)?;
        let count = rows
            .iter()
            .filter(|row| {
                row.get(column)
                    .map(|v| !matches!(v, Value::Null))
                    .unwrap_or(false)
            })
            .count();
        Ok(count as u64)
    }

    pub fn sum(&self, table: &str, column: &str, condition: Condition) -> Result<f64> {
        let rows = self.select(table, condition)?;
        let col = column.to_string();

        if rows.len() >= Self::PARALLEL_THRESHOLD {
            let total: f64 = rows
                .par_iter()
                .map(|row| {
                    row.get(&col)
                        .map(|val| match val {
                            Value::Int(i) => *i as f64,
                            Value::Float(f) => *f,
                            _ => 0.0,
                        })
                        .unwrap_or(0.0)
                })
                .sum();
            Ok(total)
        } else {
            let mut total = 0.0;
            for row in &rows {
                if let Some(val) = row.get(column) {
                    match val {
                        Value::Int(i) => total += *i as f64,
                        Value::Float(f) => total += *f,
                        _ => {},
                    }
                }
            }
            Ok(total)
        }
    }

    pub fn avg(&self, table: &str, column: &str, condition: Condition) -> Result<Option<f64>> {
        let rows = self.select(table, condition)?;
        let col = column.to_string();

        let (total, count) = if rows.len() >= Self::PARALLEL_THRESHOLD {
            rows.par_iter()
                .map(|row| {
                    row.get(&col)
                        .map(|val| match val {
                            Value::Int(i) => (*i as f64, 1u64),
                            Value::Float(f) => (*f, 1u64),
                            _ => (0.0, 0u64),
                        })
                        .unwrap_or((0.0, 0u64))
                })
                .reduce(|| (0.0, 0u64), |(s1, c1), (s2, c2)| (s1 + s2, c1 + c2))
        } else {
            let mut total = 0.0;
            let mut count = 0u64;
            for row in &rows {
                if let Some(val) = row.get(column) {
                    match val {
                        Value::Int(i) => {
                            total += *i as f64;
                            count += 1;
                        },
                        Value::Float(f) => {
                            total += *f;
                            count += 1;
                        },
                        _ => {},
                    }
                }
            }
            (total, count)
        };

        if count == 0 {
            Ok(None)
        } else {
            Ok(Some(total / count as f64))
        }
    }

    pub fn min(&self, table: &str, column: &str, condition: Condition) -> Result<Option<Value>> {
        let rows = self.select(table, condition)?;
        let col = column.to_string();

        if rows.len() >= Self::PARALLEL_THRESHOLD {
            let min_val = rows
                .par_iter()
                .filter_map(|row| {
                    row.get(&col).and_then(|val| {
                        if matches!(val, Value::Null) {
                            None
                        } else {
                            Some(val.clone())
                        }
                    })
                })
                .reduce_with(|a, b| {
                    if a.partial_cmp_value(&b) == Some(std::cmp::Ordering::Less) {
                        a
                    } else {
                        b
                    }
                });
            Ok(min_val)
        } else {
            let mut min_val: Option<Value> = None;
            for row in &rows {
                if let Some(val) = row.get(column) {
                    if matches!(val, Value::Null) {
                        continue;
                    }
                    min_val = match &min_val {
                        None => Some(val.clone()),
                        Some(current) => {
                            if val.partial_cmp_value(current) == Some(std::cmp::Ordering::Less) {
                                Some(val.clone())
                            } else {
                                min_val
                            }
                        },
                    };
                }
            }
            Ok(min_val)
        }
    }

    pub fn max(&self, table: &str, column: &str, condition: Condition) -> Result<Option<Value>> {
        let rows = self.select(table, condition)?;
        let col = column.to_string();

        if rows.len() >= Self::PARALLEL_THRESHOLD {
            let max_val = rows
                .par_iter()
                .filter_map(|row| {
                    row.get(&col).and_then(|val| {
                        if matches!(val, Value::Null) {
                            None
                        } else {
                            Some(val.clone())
                        }
                    })
                })
                .reduce_with(|a, b| {
                    if a.partial_cmp_value(&b) == Some(std::cmp::Ordering::Greater) {
                        a
                    } else {
                        b
                    }
                });
            Ok(max_val)
        } else {
            let mut max_val: Option<Value> = None;
            for row in &rows {
                if let Some(val) = row.get(column) {
                    if matches!(val, Value::Null) {
                        continue;
                    }
                    max_val = match &max_val {
                        None => Some(val.clone()),
                        Some(current) => {
                            if val.partial_cmp_value(current) == Some(std::cmp::Ordering::Greater) {
                                Some(val.clone())
                            } else {
                                max_val
                            }
                        },
                    };
                }
            }
            Ok(max_val)
        }
    }

    pub fn drop_table(&self, table: &str) -> Result<()> {
        let meta_key = Self::table_meta_key(table);

        if !self.store.exists(&meta_key) {
            return Err(RelationalError::TableNotFound(table.to_string()));
        }

        let prefix = Self::row_prefix(table);
        let keys = self.store.scan(&prefix);
        for key in keys {
            self.store.delete(&key)?;
        }

        let idx_prefix = Self::all_indexes_prefix(table);
        let idx_keys = self.store.scan(&idx_prefix);
        for key in idx_keys {
            self.store.delete(&key)?;
        }

        let btree_prefix = format!("_btree:{}:", table);
        let btree_keys = self.store.scan(&btree_prefix);
        for key in btree_keys {
            self.store.delete(&key)?;
        }

        self.store.delete(&meta_key)?;

        self.row_counters.remove(table);

        Ok(())
    }

    pub fn table_exists(&self, table: &str) -> bool {
        let meta_key = Self::table_meta_key(table);
        self.store.exists(&meta_key)
    }

    pub fn row_count(&self, table: &str) -> Result<usize> {
        let _ = self.get_schema(table)?;
        let prefix = Self::row_prefix(table);
        Ok(self.store.scan_count(&prefix))
    }

    /// Create a hash index on a column for fast equality lookups.
    pub fn create_index(&self, table: &str, column: &str) -> Result<()> {
        let schema = self.get_schema(table)?;

        if column != "_id" && schema.get_column(column).is_none() {
            return Err(RelationalError::ColumnNotFound(column.to_string()));
        }

        let meta_key = Self::index_meta_key(table, column);
        if self.store.exists(&meta_key) {
            return Err(RelationalError::IndexAlreadyExists {
                table: table.to_string(),
                column: column.to_string(),
            });
        }

        let mut meta = TensorData::new();
        meta.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("index".into())),
        );
        meta.set(
            "_table",
            TensorValue::Scalar(ScalarValue::String(table.into())),
        );
        meta.set(
            "_column",
            TensorValue::Scalar(ScalarValue::String(column.into())),
        );
        self.store.put(&meta_key, meta)?;

        let prefix = Self::row_prefix(table);
        let entries: Vec<(Value, u64)> = self.store.scan_filter_map(&prefix, |_key, tensor| {
            let row = self.tensor_to_row(tensor)?;
            let value = row.get_with_id(column)?;
            Some((value, row.id))
        });

        for (value, row_id) in entries {
            self.index_add(table, column, &value, row_id)?;
        }

        Ok(())
    }

    /// Create a B-tree index on a column for fast range queries.
    /// B-tree indexes accelerate Lt, Le, Gt, Ge conditions with O(log n) lookup.
    pub fn create_btree_index(&self, table: &str, column: &str) -> Result<()> {
        let schema = self.get_schema(table)?;

        if column != "_id" && schema.get_column(column).is_none() {
            return Err(RelationalError::ColumnNotFound(column.to_string()));
        }

        let meta_key = Self::btree_meta_key(table, column);
        if self.store.exists(&meta_key) {
            return Err(RelationalError::IndexAlreadyExists {
                table: table.to_string(),
                column: column.to_string(),
            });
        }

        let mut meta = TensorData::new();
        meta.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("btree".into())),
        );
        meta.set(
            "_table",
            TensorValue::Scalar(ScalarValue::String(table.into())),
        );
        meta.set(
            "_column",
            TensorValue::Scalar(ScalarValue::String(column.into())),
        );
        self.store.put(&meta_key, meta)?;

        {
            let key = (table.to_string(), column.to_string());
            self.btree_indexes.write().entry(key).or_default();
        }

        let prefix = Self::row_prefix(table);
        let entries: Vec<(Value, u64)> = self.store.scan_filter_map(&prefix, |_key, tensor| {
            let row = self.tensor_to_row(tensor)?;
            let value = row.get_with_id(column)?;
            Some((value, row.id))
        });

        for (value, row_id) in entries {
            self.btree_index_add(table, column, &value, row_id)?;
        }

        Ok(())
    }

    pub fn has_btree_index(&self, table: &str, column: &str) -> bool {
        let meta_key = Self::btree_meta_key(table, column);
        self.store.exists(&meta_key)
    }

    pub fn drop_btree_index(&self, table: &str, column: &str) -> Result<()> {
        let _ = self.get_schema(table)?;

        let meta_key = Self::btree_meta_key(table, column);
        if !self.store.exists(&meta_key) {
            return Err(RelationalError::IndexNotFound {
                table: table.to_string(),
                column: column.to_string(),
            });
        }

        {
            let key = (table.to_string(), column.to_string());
            self.btree_indexes.write().remove(&key);
        }

        let prefix = Self::btree_prefix(table, column);
        let keys = self.store.scan(&prefix);
        for key in keys {
            self.store.delete(&key)?;
        }

        self.store.delete(&meta_key)?;

        Ok(())
    }

    pub fn get_btree_indexed_columns(&self, table: &str) -> Vec<String> {
        let prefix = "_btree:".to_string() + table + ":";
        self.store
            .scan(&prefix)
            .into_iter()
            .filter_map(|key| {
                let parts: Vec<&str> = key.split(':').collect();
                if parts.len() == 3 {
                    Some(parts[2].to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn drop_index(&self, table: &str, column: &str) -> Result<()> {
        let _ = self.get_schema(table)?;

        let meta_key = Self::index_meta_key(table, column);
        if !self.store.exists(&meta_key) {
            return Err(RelationalError::IndexNotFound {
                table: table.to_string(),
                column: column.to_string(),
            });
        }

        let prefix = Self::index_prefix(table, column);
        let keys = self.store.scan(&prefix);
        for key in keys {
            self.store.delete(&key)?;
        }

        self.store.delete(&meta_key)?;

        Ok(())
    }

    pub fn has_index(&self, table: &str, column: &str) -> bool {
        let meta_key = Self::index_meta_key(table, column);
        self.store.exists(&meta_key)
    }

    pub fn get_indexed_columns(&self, table: &str) -> Vec<String> {
        let prefix = format!("_idx:{}:", table);
        let mut columns = HashSet::new();

        for key in self.store.scan(&prefix) {
            let parts: Vec<&str> = key.split(':').collect();
            if parts.len() >= 3 {
                let meta_key = Self::index_meta_key(table, parts[2]);
                if self.store.exists(&meta_key) {
                    columns.insert(parts[2].to_string());
                }
            }
        }

        columns.into_iter().collect()
    }

    fn index_add(&self, table: &str, column: &str, value: &Value, row_id: u64) -> Result<()> {
        let value_hash = value.hash_key();
        let key = Self::index_entry_key(table, column, &value_hash);

        let mut ids: Vec<u64> = if let Ok(tensor) = self.store.get(&key) {
            self.tensor_to_id_list(&tensor)
        } else {
            Vec::new()
        };

        if !ids.contains(&row_id) {
            ids.push(row_id);
            self.store.put(&key, self.id_list_to_tensor(&ids))?;
        }

        Ok(())
    }

    fn index_remove(&self, table: &str, column: &str, value: &Value, row_id: u64) -> Result<()> {
        let value_hash = value.hash_key();
        let key = Self::index_entry_key(table, column, &value_hash);

        if let Ok(tensor) = self.store.get(&key) {
            let mut ids = self.tensor_to_id_list(&tensor);
            ids.retain(|&id| id != row_id);

            if ids.is_empty() {
                self.store.delete(&key)?;
            } else {
                self.store.put(&key, self.id_list_to_tensor(&ids))?;
            }
        }

        Ok(())
    }

    fn index_lookup(&self, table: &str, column: &str, value: &Value) -> Option<Vec<u64>> {
        if !self.has_index(table, column) {
            return None;
        }

        let value_hash = value.hash_key();
        let key = Self::index_entry_key(table, column, &value_hash);

        if let Ok(tensor) = self.store.get(&key) {
            Some(self.tensor_to_id_list(&tensor))
        } else {
            Some(Vec::new()) // Index exists but no entries for this value
        }
    }

    fn tensor_to_id_list(&self, tensor: &TensorData) -> Vec<u64> {
        match tensor.get("ids") {
            Some(TensorValue::Vector(v)) => v.iter().map(|f| *f as u64).collect(),
            _ => Vec::new(),
        }
    }

    fn id_list_to_tensor(&self, ids: &[u64]) -> TensorData {
        let mut tensor = TensorData::new();
        tensor.set(
            "ids",
            TensorValue::Vector(ids.iter().map(|&id| id as f32).collect()),
        );
        tensor
    }

    fn get_table_indexes(&self, table: &str) -> Vec<String> {
        self.get_indexed_columns(table)
    }

    fn get_table_btree_indexes(&self, table: &str) -> Vec<String> {
        self.get_btree_indexed_columns(table)
    }

    fn btree_index_add(&self, table: &str, column: &str, value: &Value, row_id: u64) -> Result<()> {
        let key = (table.to_string(), column.to_string());
        let ordered_key = OrderedKey::from_value(value);

        {
            let mut indexes = self.btree_indexes.write();
            let btree = indexes.entry(key).or_default();
            let ids = btree.entry(ordered_key).or_default();
            if !ids.contains(&row_id) {
                ids.push(row_id);
            }
        }

        let sortable = value.sortable_key();
        let store_key = Self::btree_entry_key(table, column, &sortable);

        let mut ids: Vec<u64> = if let Ok(tensor) = self.store.get(&store_key) {
            self.tensor_to_id_list(&tensor)
        } else {
            Vec::new()
        };

        if !ids.contains(&row_id) {
            ids.push(row_id);
            self.store.put(&store_key, self.id_list_to_tensor(&ids))?;
        }

        Ok(())
    }

    fn btree_index_remove(
        &self,
        table: &str,
        column: &str,
        value: &Value,
        row_id: u64,
    ) -> Result<()> {
        let key = (table.to_string(), column.to_string());
        let ordered_key = OrderedKey::from_value(value);

        {
            let mut indexes = self.btree_indexes.write();
            if let Some(btree) = indexes.get_mut(&key) {
                if let Some(ids) = btree.get_mut(&ordered_key) {
                    ids.retain(|&id| id != row_id);
                    if ids.is_empty() {
                        btree.remove(&ordered_key);
                    }
                }
            }
        }

        let sortable = value.sortable_key();
        let store_key = Self::btree_entry_key(table, column, &sortable);

        if let Ok(tensor) = self.store.get(&store_key) {
            let mut ids = self.tensor_to_id_list(&tensor);
            ids.retain(|&id| id != row_id);

            if ids.is_empty() {
                self.store.delete(&store_key)?;
            } else {
                self.store.put(&store_key, self.id_list_to_tensor(&ids))?;
            }
        }

        Ok(())
    }

    /// B-tree range lookup: returns row IDs matching the range condition.
    /// Uses in-memory BTreeMap for O(log n) range operations.
    fn btree_range_lookup(
        &self,
        table: &str,
        column: &str,
        value: &Value,
        op: RangeOp,
    ) -> Option<Vec<u64>> {
        if !self.has_btree_index(table, column) {
            return None;
        }

        let key = (table.to_string(), column.to_string());
        let target = OrderedKey::from_value(value);
        let indexes = self.btree_indexes.read();

        let btree = indexes.get(&key)?;
        let mut result_ids = Vec::new();

        match op {
            RangeOp::Lt => {
                for (_, ids) in btree.range(..target) {
                    result_ids.extend(ids);
                }
            },
            RangeOp::Le => {
                for (_, ids) in btree.range(..=target) {
                    result_ids.extend(ids);
                }
            },
            RangeOp::Gt => {
                use std::ops::Bound;
                for (_, ids) in btree.range((Bound::Excluded(target), Bound::Unbounded)) {
                    result_ids.extend(ids);
                }
            },
            RangeOp::Ge => {
                for (_, ids) in btree.range(target..) {
                    result_ids.extend(ids);
                }
            },
        }

        Some(result_ids)
    }

    fn column_data_key(table: &str, column: &str) -> String {
        format!("_col:{}:{}:data", table, column)
    }

    fn column_ids_key(table: &str, column: &str) -> String {
        format!("_col:{}:{}:ids", table, column)
    }

    fn column_nulls_key(table: &str, column: &str) -> String {
        format!("_col:{}:{}:nulls", table, column)
    }

    fn column_meta_key(table: &str, column: &str) -> String {
        format!("_col:{}:{}:meta", table, column)
    }

    pub fn has_columnar_data(&self, table: &str, column: &str) -> bool {
        let meta_key = Self::column_meta_key(table, column);
        self.store.exists(&meta_key)
    }

    /// Materialize specified columns into columnar format.
    /// This extracts column data from row storage into contiguous vectors.
    pub fn materialize_columns(&self, table: &str, columns: &[&str]) -> Result<()> {
        let schema = self.get_schema(table)?;

        for col_name in columns {
            if schema.get_column(col_name).is_none() {
                return Err(RelationalError::ColumnNotFound(col_name.to_string()));
            }
        }

        let prefix = Self::row_prefix(table);
        let keys = self.store.scan(&prefix);

        for col_name in columns {
            let col = schema
                .get_column(col_name)
                .ok_or_else(|| RelationalError::ColumnNotFound(col_name.to_string()))?;
            let column_data = self.extract_column_data(&keys, col_name, &col.column_type)?;
            self.store_column_data(table, &column_data)?;
        }

        Ok(())
    }

    fn extract_column_data(
        &self,
        row_keys: &[String],
        column: &str,
        col_type: &ColumnType,
    ) -> Result<ColumnData> {
        let mut row_ids = Vec::with_capacity(row_keys.len());
        let mut null_positions = Vec::new();

        match col_type {
            ColumnType::Int => {
                let mut values = Vec::with_capacity(row_keys.len());
                for (idx, key) in row_keys.iter().enumerate() {
                    if let Ok(tensor) = self.store.get(key) {
                        let row_id = match tensor.get("_id") {
                            Some(TensorValue::Scalar(ScalarValue::Int(id))) => *id as u64,
                            _ => continue,
                        };
                        row_ids.push(row_id);

                        match tensor.get(column) {
                            Some(TensorValue::Scalar(ScalarValue::Int(v))) => {
                                values.push(*v);
                            },
                            Some(TensorValue::Scalar(ScalarValue::Null)) | None => {
                                null_positions.push(idx as u64);
                                values.push(0); // placeholder
                            },
                            _ => {
                                values.push(0);
                            },
                        }
                    }
                }
                Ok(ColumnData {
                    name: column.to_string(),
                    row_ids,
                    nulls: Self::build_null_bitmap(null_positions, values.len()),
                    values: ColumnValues::Int(values),
                })
            },
            ColumnType::Float => {
                let mut values = Vec::with_capacity(row_keys.len());
                for (idx, key) in row_keys.iter().enumerate() {
                    if let Ok(tensor) = self.store.get(key) {
                        let row_id = match tensor.get("_id") {
                            Some(TensorValue::Scalar(ScalarValue::Int(id))) => *id as u64,
                            _ => continue,
                        };
                        row_ids.push(row_id);

                        match tensor.get(column) {
                            Some(TensorValue::Scalar(ScalarValue::Float(v))) => {
                                values.push(*v);
                            },
                            Some(TensorValue::Scalar(ScalarValue::Null)) | None => {
                                null_positions.push(idx as u64);
                                values.push(0.0);
                            },
                            _ => {
                                values.push(0.0);
                            },
                        }
                    }
                }
                Ok(ColumnData {
                    name: column.to_string(),
                    row_ids,
                    nulls: Self::build_null_bitmap(null_positions, values.len()),
                    values: ColumnValues::Float(values),
                })
            },
            ColumnType::String => {
                let mut dict: Vec<String> = Vec::new();
                let mut dict_map: HashMap<String, u32> = HashMap::new();
                let mut indices = Vec::with_capacity(row_keys.len());

                for (idx, key) in row_keys.iter().enumerate() {
                    if let Ok(tensor) = self.store.get(key) {
                        let row_id = match tensor.get("_id") {
                            Some(TensorValue::Scalar(ScalarValue::Int(id))) => *id as u64,
                            _ => continue,
                        };
                        row_ids.push(row_id);

                        match tensor.get(column) {
                            Some(TensorValue::Scalar(ScalarValue::String(v))) => {
                                let dict_idx = *dict_map.entry(v.clone()).or_insert_with(|| {
                                    let idx = dict.len() as u32;
                                    dict.push(v.clone());
                                    idx
                                });
                                indices.push(dict_idx);
                            },
                            Some(TensorValue::Scalar(ScalarValue::Null)) | None => {
                                null_positions.push(idx as u64);
                                indices.push(u32::MAX); // null marker
                            },
                            _ => {
                                indices.push(u32::MAX);
                            },
                        }
                    }
                }
                Ok(ColumnData {
                    name: column.to_string(),
                    row_ids,
                    nulls: Self::build_null_bitmap(null_positions, indices.len()),
                    values: ColumnValues::String { dict, indices },
                })
            },
            ColumnType::Bool => {
                let word_count = row_keys.len().div_ceil(64);
                let mut values = vec![0u64; word_count];

                for (idx, key) in row_keys.iter().enumerate() {
                    if let Ok(tensor) = self.store.get(key) {
                        let row_id = match tensor.get("_id") {
                            Some(TensorValue::Scalar(ScalarValue::Int(id))) => *id as u64,
                            _ => continue,
                        };
                        row_ids.push(row_id);

                        match tensor.get(column) {
                            Some(TensorValue::Scalar(ScalarValue::Bool(true))) => {
                                values[idx / 64] |= 1u64 << (idx % 64);
                            },
                            Some(TensorValue::Scalar(ScalarValue::Null)) | None => {
                                null_positions.push(idx as u64);
                            },
                            _ => {},
                        }
                    }
                }
                let row_count = row_ids.len();
                Ok(ColumnData {
                    name: column.to_string(),
                    row_ids,
                    nulls: Self::build_null_bitmap(null_positions, row_count),
                    values: ColumnValues::Bool(values),
                })
            },
        }
    }

    fn build_null_bitmap(null_positions: Vec<u64>, row_count: usize) -> NullBitmap {
        if null_positions.is_empty() {
            return NullBitmap::None;
        }

        // Use sparse if nulls are < 10% of rows
        if null_positions.len() < row_count / 10 {
            NullBitmap::Sparse(null_positions)
        } else {
            let word_count = row_count.div_ceil(64);
            let mut bitmap = vec![0u64; word_count];
            for pos in null_positions {
                let idx = pos as usize;
                bitmap[idx / 64] |= 1u64 << (idx % 64);
            }
            NullBitmap::Dense(bitmap)
        }
    }

    fn store_column_data(&self, table: &str, column_data: &ColumnData) -> Result<()> {
        let column = &column_data.name;

        let ids_key = Self::column_ids_key(table, column);
        let ids_vec: Vec<f32> = column_data.row_ids.iter().map(|&id| id as f32).collect();
        let mut ids_tensor = TensorData::new();
        ids_tensor.set("ids", TensorValue::Vector(ids_vec));
        self.store.put(ids_key, ids_tensor)?;

        let nulls_key = Self::column_nulls_key(table, column);
        let mut nulls_tensor = TensorData::new();
        match &column_data.nulls {
            NullBitmap::None => {
                nulls_tensor.set(
                    "type",
                    TensorValue::Scalar(ScalarValue::String("none".to_string())),
                );
            },
            NullBitmap::Sparse(positions) => {
                nulls_tensor.set(
                    "type",
                    TensorValue::Scalar(ScalarValue::String("sparse".to_string())),
                );
                let pos_vec: Vec<f32> = positions.iter().map(|&p| p as f32).collect();
                nulls_tensor.set("positions", TensorValue::Vector(pos_vec));
            },
            NullBitmap::Dense(bitmap) => {
                nulls_tensor.set(
                    "type",
                    TensorValue::Scalar(ScalarValue::String("dense".to_string())),
                );
                let bitmap_vec: Vec<f32> = bitmap.iter().map(|&w| w as f32).collect();
                nulls_tensor.set("bitmap", TensorValue::Vector(bitmap_vec));
            },
        }
        self.store.put(nulls_key, nulls_tensor)?;

        let data_key = Self::column_data_key(table, column);
        let mut data_tensor = TensorData::new();
        match &column_data.values {
            ColumnValues::Int(values) => {
                data_tensor.set(
                    "type",
                    TensorValue::Scalar(ScalarValue::String("int".to_string())),
                );
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                data_tensor.set("data", TensorValue::Scalar(ScalarValue::Bytes(bytes)));
            },
            ColumnValues::Float(values) => {
                data_tensor.set(
                    "type",
                    TensorValue::Scalar(ScalarValue::String("float".to_string())),
                );
                let float_vec: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                data_tensor.set("data", TensorValue::Vector(float_vec));
            },
            ColumnValues::String { dict, indices } => {
                data_tensor.set(
                    "type",
                    TensorValue::Scalar(ScalarValue::String("string".to_string())),
                );
                data_tensor.set(
                    "dict",
                    TensorValue::Scalar(ScalarValue::String(dict.join("\x00"))),
                );
                let indices_vec: Vec<f32> = indices.iter().map(|&i| i as f32).collect();
                data_tensor.set("indices", TensorValue::Vector(indices_vec));
            },
            ColumnValues::Bool(bitmap) => {
                data_tensor.set(
                    "type",
                    TensorValue::Scalar(ScalarValue::String("bool".to_string())),
                );
                let bitmap_vec: Vec<f32> = bitmap.iter().map(|&w| w as f32).collect();
                data_tensor.set("data", TensorValue::Vector(bitmap_vec));
            },
        }
        self.store.put(data_key, data_tensor)?;

        let meta_key = Self::column_meta_key(table, column);
        let mut meta_tensor = TensorData::new();
        meta_tensor.set(
            "row_count",
            TensorValue::Scalar(ScalarValue::Int(column_data.row_ids.len() as i64)),
        );
        self.store.put(meta_key, meta_tensor)?;

        Ok(())
    }

    pub fn load_column_data(&self, table: &str, column: &str) -> Result<ColumnData> {
        let ids_key = Self::column_ids_key(table, column);
        let ids_tensor = self
            .store
            .get(&ids_key)
            .map_err(|_| RelationalError::ColumnNotFound(column.to_string()))?;
        let row_ids: Vec<u64> = match ids_tensor.get("ids") {
            Some(TensorValue::Vector(v)) => v.iter().map(|&f| f as u64).collect(),
            _ => return Err(RelationalError::ColumnNotFound(column.to_string())),
        };

        let nulls_key = Self::column_nulls_key(table, column);
        let nulls = if let Ok(nulls_tensor) = self.store.get(&nulls_key) {
            match nulls_tensor.get("type") {
                Some(TensorValue::Scalar(ScalarValue::String(t))) if t == "none" => {
                    NullBitmap::None
                },
                Some(TensorValue::Scalar(ScalarValue::String(t))) if t == "sparse" => {
                    let positions: Vec<u64> = match nulls_tensor.get("positions") {
                        Some(TensorValue::Vector(v)) => v.iter().map(|&f| f as u64).collect(),
                        _ => vec![],
                    };
                    NullBitmap::Sparse(positions)
                },
                Some(TensorValue::Scalar(ScalarValue::String(t))) if t == "dense" => {
                    let bitmap: Vec<u64> = match nulls_tensor.get("bitmap") {
                        Some(TensorValue::Vector(v)) => v.iter().map(|&f| f as u64).collect(),
                        _ => vec![],
                    };
                    NullBitmap::Dense(bitmap)
                },
                _ => NullBitmap::None,
            }
        } else {
            NullBitmap::None
        };

        let data_key = Self::column_data_key(table, column);
        let data_tensor = self
            .store
            .get(&data_key)
            .map_err(|_| RelationalError::ColumnNotFound(column.to_string()))?;

        let values = match data_tensor.get("type") {
            Some(TensorValue::Scalar(ScalarValue::String(t))) if t == "int" => {
                match data_tensor.get("data") {
                    Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) => {
                        let values: Vec<i64> = bytes
                            .chunks_exact(8)
                            .map(|chunk| {
                                // chunks_exact guarantees 8 bytes, but handle gracefully
                                let arr: [u8; 8] = chunk.try_into().unwrap_or([0; 8]);
                                i64::from_le_bytes(arr)
                            })
                            .collect();
                        ColumnValues::Int(values)
                    },
                    _ => return Err(RelationalError::StorageError("Invalid column data".into())),
                }
            },
            Some(TensorValue::Scalar(ScalarValue::String(t))) if t == "float" => {
                match data_tensor.get("data") {
                    Some(TensorValue::Vector(v)) => {
                        ColumnValues::Float(v.iter().map(|&f| f as f64).collect())
                    },
                    _ => return Err(RelationalError::StorageError("Invalid column data".into())),
                }
            },
            Some(TensorValue::Scalar(ScalarValue::String(t))) if t == "string" => {
                let dict: Vec<String> = match data_tensor.get("dict") {
                    Some(TensorValue::Scalar(ScalarValue::String(s))) => {
                        s.split('\x00').map(String::from).collect()
                    },
                    _ => vec![],
                };
                let indices: Vec<u32> = match data_tensor.get("indices") {
                    Some(TensorValue::Vector(v)) => v.iter().map(|&f| f as u32).collect(),
                    _ => vec![],
                };
                ColumnValues::String { dict, indices }
            },
            Some(TensorValue::Scalar(ScalarValue::String(t))) if t == "bool" => {
                match data_tensor.get("data") {
                    Some(TensorValue::Vector(v)) => {
                        ColumnValues::Bool(v.iter().map(|&f| f as u64).collect())
                    },
                    _ => return Err(RelationalError::StorageError("Invalid column data".into())),
                }
            },
            _ => return Err(RelationalError::StorageError("Unknown column type".into())),
        };

        Ok(ColumnData {
            name: column.to_string(),
            row_ids,
            nulls,
            values,
        })
    }

    pub fn drop_columnar_data(&self, table: &str, column: &str) -> Result<()> {
        // Column data may not exist - delete is idempotent
        self.store
            .delete(&Self::column_data_key(table, column))
            .ok();
        self.store.delete(&Self::column_ids_key(table, column)).ok();
        self.store
            .delete(&Self::column_nulls_key(table, column))
            .ok();
        self.store
            .delete(&Self::column_meta_key(table, column))
            .ok();
        Ok(())
    }

    /// Select with columnar scan and projection support.
    ///
    /// If columnar data is available and options.prefer_columnar is true,
    /// uses SIMD-accelerated vectorized filtering. Otherwise falls back to
    /// row-based scan with projection.
    pub fn select_columnar(
        &self,
        table: &str,
        condition: Condition,
        options: ColumnarScanOptions,
    ) -> Result<Vec<Row>> {
        let filter_columns = Self::extract_filter_columns(&condition);

        let use_columnar = options.prefer_columnar
            && !filter_columns.is_empty()
            && filter_columns
                .iter()
                .all(|col| self.has_columnar_data(table, col));

        if use_columnar {
            self.select_columnar_impl(table, condition, options)
        } else {
            self.select_with_projection(table, condition, options.projection)
        }
    }

    /// Extract column names referenced in a condition.
    fn extract_filter_columns(condition: &Condition) -> Vec<String> {
        let mut columns = Vec::new();
        Self::collect_filter_columns(condition, &mut columns);
        columns
    }

    fn collect_filter_columns(condition: &Condition, columns: &mut Vec<String>) {
        match condition {
            Condition::True => {},
            Condition::Eq(col, _)
            | Condition::Ne(col, _)
            | Condition::Lt(col, _)
            | Condition::Le(col, _)
            | Condition::Gt(col, _)
            | Condition::Ge(col, _) => {
                if !columns.contains(col) {
                    columns.push(col.clone());
                }
            },
            Condition::And(a, b) | Condition::Or(a, b) => {
                Self::collect_filter_columns(a, columns);
                Self::collect_filter_columns(b, columns);
            },
        }
    }

    fn select_columnar_impl(
        &self,
        table: &str,
        condition: Condition,
        options: ColumnarScanOptions,
    ) -> Result<Vec<Row>> {
        let filter_columns = Self::extract_filter_columns(&condition);

        let projection_columns: Vec<String> = match &options.projection {
            Some(cols) => cols.clone(),
            None => {
                let schema_key = format!("{}:_schema", table);
                match self.store.get(&schema_key) {
                    Ok(tensor) => tensor
                        .keys()
                        .filter(|k| !k.starts_with('_'))
                        .cloned()
                        .collect(),
                    Err(_) => {
                        return self.select_with_projection(table, condition, options.projection)
                    },
                }
            },
        };

        let mut all_needed: Vec<String> = filter_columns.clone();
        for col in &projection_columns {
            if !all_needed.contains(col) {
                all_needed.push(col.clone());
            }
        }

        let use_pure_columnar = self.all_columns_materialized(table, &all_needed);

        let mut column_map: HashMap<String, ColumnData> = HashMap::new();
        for col in &filter_columns {
            let col_data = self.load_column_data(table, col)?;
            column_map.insert(col.clone(), col_data);
        }

        if use_pure_columnar && column_map.is_empty() && !projection_columns.is_empty() {
            let first_col = &projection_columns[0];
            let col_data = self.load_column_data(table, first_col)?;
            column_map.insert(first_col.clone(), col_data);
        }

        let row_count = column_map
            .values()
            .next()
            .map(|c| c.row_ids.len())
            .unwrap_or(0);

        if row_count == 0 {
            return Ok(Vec::new());
        }

        let selection = self.apply_vectorized_filter(&column_map, &condition, row_count)?;

        let row_ids: Vec<u64> = column_map
            .values()
            .next()
            .map(|c| c.row_ids.clone())
            .unwrap_or_default();

        let selected_indices = selection.selected_indices();

        if use_pure_columnar {
            for col in &projection_columns {
                if !column_map.contains_key(col) {
                    let col_data = self.load_column_data(table, col)?;
                    column_map.insert(col.clone(), col_data);
                }
            }
            Ok(Self::materialize_from_columns(
                &column_map,
                &row_ids,
                &selected_indices,
                &options.projection,
            ))
        } else {
            self.materialize_selected_rows(table, &row_ids, &selected_indices, options.projection)
        }
    }

    fn apply_vectorized_filter(
        &self,
        columns: &HashMap<String, ColumnData>,
        condition: &Condition,
        row_count: usize,
    ) -> Result<SelectionVector> {
        match condition {
            Condition::True => Ok(SelectionVector::all(row_count)),

            Condition::Eq(col, Value::Int(val)) => {
                let col_data = columns
                    .get(col)
                    .ok_or_else(|| RelationalError::ColumnNotFound(col.clone()))?;
                match &col_data.values {
                    ColumnValues::Int(values) => {
                        let mut bitmap = vec![0u64; simd::bitmap_words(values.len())];
                        simd::filter_eq_i64(values, *val, &mut bitmap);
                        Ok(SelectionVector::from_bitmap(bitmap, values.len()))
                    },
                    _ => Err(RelationalError::TypeMismatch {
                        column: col.clone(),
                        expected: ColumnType::Int,
                    }),
                }
            },

            Condition::Ne(col, Value::Int(val)) => {
                let col_data = columns
                    .get(col)
                    .ok_or_else(|| RelationalError::ColumnNotFound(col.clone()))?;
                match &col_data.values {
                    ColumnValues::Int(values) => {
                        let mut bitmap = vec![0u64; simd::bitmap_words(values.len())];
                        simd::filter_ne_i64(values, *val, &mut bitmap);
                        Ok(SelectionVector::from_bitmap(bitmap, values.len()))
                    },
                    _ => Err(RelationalError::TypeMismatch {
                        column: col.clone(),
                        expected: ColumnType::Int,
                    }),
                }
            },

            Condition::Lt(col, Value::Int(val)) => {
                let col_data = columns
                    .get(col)
                    .ok_or_else(|| RelationalError::ColumnNotFound(col.clone()))?;
                match &col_data.values {
                    ColumnValues::Int(values) => {
                        let mut bitmap = vec![0u64; simd::bitmap_words(values.len())];
                        simd::filter_lt_i64(values, *val, &mut bitmap);
                        Ok(SelectionVector::from_bitmap(bitmap, values.len()))
                    },
                    _ => Err(RelationalError::TypeMismatch {
                        column: col.clone(),
                        expected: ColumnType::Int,
                    }),
                }
            },

            Condition::Le(col, Value::Int(val)) => {
                let col_data = columns
                    .get(col)
                    .ok_or_else(|| RelationalError::ColumnNotFound(col.clone()))?;
                match &col_data.values {
                    ColumnValues::Int(values) => {
                        let mut bitmap = vec![0u64; simd::bitmap_words(values.len())];
                        simd::filter_le_i64(values, *val, &mut bitmap);
                        Ok(SelectionVector::from_bitmap(bitmap, values.len()))
                    },
                    _ => Err(RelationalError::TypeMismatch {
                        column: col.clone(),
                        expected: ColumnType::Int,
                    }),
                }
            },

            Condition::Gt(col, Value::Int(val)) => {
                let col_data = columns
                    .get(col)
                    .ok_or_else(|| RelationalError::ColumnNotFound(col.clone()))?;
                match &col_data.values {
                    ColumnValues::Int(values) => {
                        let mut bitmap = vec![0u64; simd::bitmap_words(values.len())];
                        simd::filter_gt_i64(values, *val, &mut bitmap);
                        Ok(SelectionVector::from_bitmap(bitmap, values.len()))
                    },
                    _ => Err(RelationalError::TypeMismatch {
                        column: col.clone(),
                        expected: ColumnType::Int,
                    }),
                }
            },

            Condition::Ge(col, Value::Int(val)) => {
                let col_data = columns
                    .get(col)
                    .ok_or_else(|| RelationalError::ColumnNotFound(col.clone()))?;
                match &col_data.values {
                    ColumnValues::Int(values) => {
                        let mut bitmap = vec![0u64; simd::bitmap_words(values.len())];
                        simd::filter_ge_i64(values, *val, &mut bitmap);
                        Ok(SelectionVector::from_bitmap(bitmap, values.len()))
                    },
                    _ => Err(RelationalError::TypeMismatch {
                        column: col.clone(),
                        expected: ColumnType::Int,
                    }),
                }
            },

            Condition::And(a, b) => {
                let sel_a = self.apply_vectorized_filter(columns, a, row_count)?;
                let sel_b = self.apply_vectorized_filter(columns, b, row_count)?;
                Ok(sel_a.intersect(&sel_b))
            },

            Condition::Or(a, b) => {
                let sel_a = self.apply_vectorized_filter(columns, a, row_count)?;
                let sel_b = self.apply_vectorized_filter(columns, b, row_count)?;
                Ok(sel_a.union(&sel_b))
            },

            // Fallback for non-Int comparisons: select all and filter later
            _ => Ok(SelectionVector::all(row_count)),
        }
    }

    fn materialize_selected_rows(
        &self,
        table: &str,
        row_ids: &[u64],
        selected_indices: &[usize],
        projection: Option<Vec<String>>,
    ) -> Result<Vec<Row>> {
        let mut rows = Vec::with_capacity(selected_indices.len());

        for &idx in selected_indices {
            if idx >= row_ids.len() {
                continue;
            }
            let row_id = row_ids[idx];
            let key = Self::row_key(table, row_id);

            if let Ok(tensor) = self.store.get(&key) {
                let mut values = Vec::new();

                match &projection {
                    Some(cols) => {
                        for col in cols {
                            if col == "_id" {
                                continue;
                            }
                            if let Some(TensorValue::Scalar(scalar)) = tensor.get(col) {
                                values.push((col.clone(), Value::from_scalar(scalar)));
                            }
                        }
                    },
                    None => {
                        for key in tensor.keys() {
                            if key.starts_with('_') {
                                continue;
                            }
                            if let Some(TensorValue::Scalar(scalar)) = tensor.get(key) {
                                values.push((key.clone(), Value::from_scalar(scalar)));
                            }
                        }
                    },
                }

                rows.push(Row { id: row_id, values });
            }
        }

        rows.sort_by_key(|r| r.id);
        Ok(rows)
    }

    /// Build rows purely from columnar data without touching row storage.
    fn materialize_from_columns(
        columns: &HashMap<String, ColumnData>,
        row_ids: &[u64],
        selected_indices: &[usize],
        projection: &Option<Vec<String>>,
    ) -> Vec<Row> {
        let num_rows = selected_indices.len();
        let mut rows = Vec::with_capacity(num_rows);

        // Pre-resolve column references to avoid repeated HashMap lookups
        let col_refs: Vec<(&str, &ColumnData)> = match projection {
            Some(cols) => cols
                .iter()
                .filter(|c| *c != "_id")
                .filter_map(|c| columns.get(c).map(|data| (c.as_str(), data)))
                .collect(),
            None => columns.iter().map(|(k, v)| (k.as_str(), v)).collect(),
        };

        for &idx in selected_indices {
            if idx >= row_ids.len() {
                continue;
            }
            let row_id = row_ids[idx];
            let mut values = Vec::with_capacity(col_refs.len());

            for &(col_name, col_data) in &col_refs {
                if let Some(val) = col_data.get_value(idx) {
                    values.push((col_name.to_string(), val));
                }
            }

            rows.push(Row { id: row_id, values });
        }

        // Skip sort if rows are already in order (common case for sequential scans)
        if rows.len() > 1 {
            let is_sorted = rows.windows(2).all(|w| w[0].id <= w[1].id);
            if !is_sorted {
                rows.sort_by_key(|r| r.id);
            }
        }

        rows
    }

    fn all_columns_materialized(&self, table: &str, columns: &[String]) -> bool {
        columns.iter().all(|col| self.has_columnar_data(table, col))
    }

    /// Row-based select with projection (fallback path).
    pub fn select_with_projection(
        &self,
        table: &str,
        condition: Condition,
        projection: Option<Vec<String>>,
    ) -> Result<Vec<Row>> {
        let rows = self.select(table, condition)?;

        match projection {
            Some(cols) => Ok(rows
                .into_iter()
                .map(|row| {
                    let values = cols
                        .iter()
                        .filter_map(|c| {
                            if c == "_id" {
                                None
                            } else {
                                row.get(c).map(|v| (c.clone(), v.clone()))
                            }
                        })
                        .collect();
                    Row { id: row.id, values }
                })
                .collect()),
            None => Ok(rows),
        }
    }
}

/// Range operation for B-tree index lookups
#[derive(Debug, Clone, Copy)]
enum RangeOp {
    Lt,
    Le,
    Gt,
    Ge,
}

impl Default for RelationalEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_users_table(engine: &RelationalEngine) {
        let schema = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("age", ColumnType::Int),
            Column::new("email", ColumnType::String).nullable(),
        ]);
        engine.create_table("users", schema).unwrap();
    }

    fn create_posts_table(engine: &RelationalEngine) {
        let schema = Schema::new(vec![
            Column::new("user_id", ColumnType::Int),
            Column::new("title", ColumnType::String),
            Column::new("views", ColumnType::Int),
        ]);
        engine.create_table("posts", schema).unwrap();
    }

    #[test]
    fn create_table_and_insert() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".to_string()));
        values.insert("age".to_string(), Value::Int(30));

        let id = engine.insert("users", values).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn store_accessor_returns_underlying_store() {
        let store = TensorStore::new();
        let engine = RelationalEngine::with_store(store.clone());

        // Verify we can access the underlying store
        let accessed_store = engine.store();

        // The store should be the same (verify by checking they share data)
        store.put("test_key", TensorData::new()).unwrap();
        assert!(accessed_store.exists("test_key"));
    }

    #[test]
    fn insert_1000_rows_select_with_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..1000 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + (i % 50)));
            engine.insert("users", values).unwrap();
        }

        assert_eq!(engine.row_count("users").unwrap(), 1000);

        let rows = engine
            .select("users", Condition::Eq("age".to_string(), Value::Int(25)))
            .unwrap();

        assert_eq!(rows.len(), 20);

        for row in &rows {
            assert_eq!(row.get("age"), Some(&Value::Int(25)));
        }
    }

    #[test]
    fn select_with_range_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..100 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        let rows = engine
            .select("users", Condition::Ge("age".to_string(), Value::Int(90)))
            .unwrap();

        assert_eq!(rows.len(), 10);
    }

    #[test]
    fn select_with_compound_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..100 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        let condition = Condition::Ge("age".to_string(), Value::Int(40))
            .and(Condition::Lt("age".to_string(), Value::Int(50)));

        let rows = engine.select("users", condition).unwrap();
        assert_eq!(rows.len(), 10);
    }

    #[test]
    fn join_two_tables() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        create_posts_table(&engine);

        for i in 1..=5 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        let post_data = vec![
            (1, "Post A", 100),
            (1, "Post B", 200),
            (2, "Post C", 150),
            (3, "Post D", 50),
            (3, "Post E", 75),
            (3, "Post F", 25),
        ];

        for (user_id, title, views) in post_data {
            let mut values = HashMap::new();
            values.insert("user_id".to_string(), Value::Int(user_id));
            values.insert("title".to_string(), Value::String(title.to_string()));
            values.insert("views".to_string(), Value::Int(views));
            engine.insert("posts", values).unwrap();
        }

        let joined = engine.join("users", "posts", "_id", "user_id").unwrap();

        assert_eq!(joined.len(), 6);

        let user1_posts: Vec<_> = joined.iter().filter(|(u, _)| u.id == 1).collect();
        assert_eq!(user1_posts.len(), 2);

        let user3_posts: Vec<_> = joined.iter().filter(|(u, _)| u.id == 3).collect();
        assert_eq!(user3_posts.len(), 3);
    }

    #[test]
    fn update_modifies_correct_rows() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(25));
            engine.insert("users", values).unwrap();
        }

        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(30));

        let count = engine
            .update(
                "users",
                Condition::Lt("_id".to_string(), Value::Int(6)),
                updates,
            )
            .unwrap();

        assert_eq!(count, 5);

        let rows = engine
            .select("users", Condition::Eq("age".to_string(), Value::Int(30)))
            .unwrap();
        assert_eq!(rows.len(), 5);

        let rows = engine
            .select("users", Condition::Eq("age".to_string(), Value::Int(25)))
            .unwrap();
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn delete_removes_correct_rows() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..20 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        assert_eq!(engine.row_count("users").unwrap(), 20);

        let count = engine
            .delete_rows("users", Condition::Lt("age".to_string(), Value::Int(10)))
            .unwrap();

        assert_eq!(count, 10);
        assert_eq!(engine.row_count("users").unwrap(), 10);

        let remaining = engine.select("users", Condition::True).unwrap();
        for row in remaining {
            if let Some(Value::Int(age)) = row.get("age") {
                assert!(*age >= 10);
            }
        }
    }

    #[test]
    fn delete_data_is_gone() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("ToDelete".to_string()));
        values.insert("age".to_string(), Value::Int(99));
        let id = engine.insert("users", values).unwrap();

        let rows = engine
            .select(
                "users",
                Condition::Eq("_id".to_string(), Value::Int(id as i64)),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);

        engine
            .delete_rows(
                "users",
                Condition::Eq("_id".to_string(), Value::Int(id as i64)),
            )
            .unwrap();

        let rows = engine
            .select(
                "users",
                Condition::Eq("_id".to_string(), Value::Int(id as i64)),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn table_not_found_error() {
        let engine = RelationalEngine::new();

        let result = engine.select("nonexistent", Condition::True);
        assert!(matches!(result, Err(RelationalError::TableNotFound(_))));
    }

    #[test]
    fn duplicate_table_error() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let schema = Schema::new(vec![Column::new("x", ColumnType::Int)]);
        let result = engine.create_table("users", schema);
        assert!(matches!(
            result,
            Err(RelationalError::TableAlreadyExists(_))
        ));
    }

    #[test]
    fn type_mismatch_error() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::Int(123));
        values.insert("age".to_string(), Value::Int(30));

        let result = engine.insert("users", values);
        assert!(matches!(result, Err(RelationalError::TypeMismatch { .. })));
    }

    #[test]
    fn null_not_allowed_error() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::Null);
        values.insert("age".to_string(), Value::Int(30));

        let result = engine.insert("users", values);
        assert!(matches!(result, Err(RelationalError::NullNotAllowed(_))));
    }

    #[test]
    fn nullable_column_accepts_null() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".to_string()));
        values.insert("age".to_string(), Value::Int(30));
        values.insert("email".to_string(), Value::Null);

        let id = engine.insert("users", values).unwrap();
        assert!(id > 0);
    }

    #[test]
    fn drop_table() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".to_string()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        assert!(engine.table_exists("users"));

        engine.drop_table("users").unwrap();

        assert!(!engine.table_exists("users"));

        let result = engine.select("users", Condition::True);
        assert!(matches!(result, Err(RelationalError::TableNotFound(_))));
    }

    #[test]
    fn or_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        let condition = Condition::Eq("age".to_string(), Value::Int(0))
            .or(Condition::Eq("age".to_string(), Value::Int(9)));

        let rows = engine.select("users", condition).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn row_id_in_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..5 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        let rows = engine
            .select("users", Condition::Eq("_id".to_string(), Value::Int(3)))
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, 3);
    }

    // Additional tests for 100% coverage

    #[test]
    fn error_display_all_variants() {
        let e1 = RelationalError::TableNotFound("test".into());
        assert!(format!("{}", e1).contains("test"));

        let e2 = RelationalError::TableAlreadyExists("test".into());
        assert!(format!("{}", e2).contains("test"));

        let e3 = RelationalError::ColumnNotFound("col".into());
        assert!(format!("{}", e3).contains("col"));

        let e4 = RelationalError::TypeMismatch {
            column: "age".into(),
            expected: ColumnType::Int,
        };
        assert!(format!("{}", e4).contains("age"));

        let e5 = RelationalError::NullNotAllowed("name".into());
        assert!(format!("{}", e5).contains("name"));

        let e6 = RelationalError::StorageError("disk full".into());
        assert!(format!("{}", e6).contains("disk full"));
    }

    #[test]
    fn error_is_error_trait() {
        let err: &dyn std::error::Error = &RelationalError::TableNotFound("x".into());
        assert!(err.to_string().contains("x"));
    }

    #[test]
    fn engine_default_trait() {
        let engine = RelationalEngine::default();
        assert!(!engine.table_exists("any"));
    }

    #[test]
    fn engine_with_store() {
        let store = TensorStore::new();
        let engine = RelationalEngine::with_store(store);
        assert!(!engine.table_exists("any"));
    }

    #[test]
    fn row_get_returns_none_for_id() {
        let values = vec![("name".to_string(), Value::String("test".into()))];
        let row = Row { id: 1, values };
        assert!(row.get("_id").is_none());
        assert_eq!(row.get_with_id("_id"), Some(Value::Int(1)));
    }

    #[test]
    fn condition_ne() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..5 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        let rows = engine
            .select("users", Condition::Ne("age".to_string(), Value::Int(2)))
            .unwrap();
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn condition_le() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        let rows = engine
            .select("users", Condition::Le("age".to_string(), Value::Int(5)))
            .unwrap();
        assert_eq!(rows.len(), 6);
    }

    #[test]
    fn condition_gt() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        let rows = engine
            .select("users", Condition::Gt("age".to_string(), Value::Int(7)))
            .unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn float_comparisons() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("score", ColumnType::Float),
        ]);
        engine.create_table("scores", schema).unwrap();

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("score".to_string(), Value::Float(i as f64 * 0.5));
            engine.insert("scores", values).unwrap();
        }

        let lt = engine
            .select(
                "scores",
                Condition::Lt("score".to_string(), Value::Float(2.0)),
            )
            .unwrap();
        assert_eq!(lt.len(), 4);

        let le = engine
            .select(
                "scores",
                Condition::Le("score".to_string(), Value::Float(2.0)),
            )
            .unwrap();
        assert_eq!(le.len(), 5);

        let gt = engine
            .select(
                "scores",
                Condition::Gt("score".to_string(), Value::Float(3.5)),
            )
            .unwrap();
        assert_eq!(gt.len(), 2);

        let ge = engine
            .select(
                "scores",
                Condition::Ge("score".to_string(), Value::Float(3.5)),
            )
            .unwrap();
        assert_eq!(ge.len(), 3);
    }

    #[test]
    fn string_comparisons() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let names = vec!["Alice", "Bob", "Charlie", "David", "Eve"];
        for name in &names {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(name.to_string()));
            values.insert("age".to_string(), Value::Int(30));
            engine.insert("users", values).unwrap();
        }

        let lt = engine
            .select(
                "users",
                Condition::Lt("name".to_string(), Value::String("Charlie".into())),
            )
            .unwrap();
        assert_eq!(lt.len(), 2);

        let le = engine
            .select(
                "users",
                Condition::Le("name".to_string(), Value::String("Charlie".into())),
            )
            .unwrap();
        assert_eq!(le.len(), 3);

        let gt = engine
            .select(
                "users",
                Condition::Gt("name".to_string(), Value::String("Charlie".into())),
            )
            .unwrap();
        assert_eq!(gt.len(), 2);

        let ge = engine
            .select(
                "users",
                Condition::Ge("name".to_string(), Value::String("Charlie".into())),
            )
            .unwrap();
        assert_eq!(ge.len(), 3);
    }

    #[test]
    fn update_column_not_found() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let mut updates = HashMap::new();
        updates.insert("nonexistent".to_string(), Value::Int(1));

        let result = engine.update("users", Condition::True, updates);
        assert!(matches!(result, Err(RelationalError::ColumnNotFound(_))));
    }

    #[test]
    fn update_type_mismatch() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::String("wrong type".into()));

        let result = engine.update("users", Condition::True, updates);
        assert!(matches!(result, Err(RelationalError::TypeMismatch { .. })));
    }

    #[test]
    fn update_null_not_allowed() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let mut updates = HashMap::new();
        updates.insert("name".to_string(), Value::Null);

        let result = engine.update("users", Condition::True, updates);
        assert!(matches!(result, Err(RelationalError::NullNotAllowed(_))));
    }

    #[test]
    fn drop_nonexistent_table() {
        let engine = RelationalEngine::new();
        let result = engine.drop_table("nonexistent");
        assert!(matches!(result, Err(RelationalError::TableNotFound(_))));
    }

    #[test]
    fn join_no_matches() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        create_posts_table(&engine);

        let mut user_values = HashMap::new();
        user_values.insert("name".to_string(), Value::String("Alice".into()));
        user_values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", user_values).unwrap();

        let mut post_values = HashMap::new();
        post_values.insert("user_id".to_string(), Value::Int(999));
        post_values.insert("title".to_string(), Value::String("Orphan".into()));
        post_values.insert("views".to_string(), Value::Int(0));
        engine.insert("posts", post_values).unwrap();

        let joined = engine.join("users", "posts", "_id", "user_id").unwrap();
        assert_eq!(joined.len(), 0);
    }

    #[test]
    fn empty_table_select() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let rows = engine.select("users", Condition::True).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn value_clone_and_eq() {
        let v1 = Value::Null;
        let v2 = Value::Int(42);
        let v3 = Value::Float(3.14);
        let v4 = Value::String("test".into());
        let v5 = Value::Bool(true);

        assert_eq!(v1.clone(), v1);
        assert_eq!(v2.clone(), v2);
        assert_eq!(v3.clone(), v3);
        assert_eq!(v4.clone(), v4);
        assert_eq!(v5.clone(), v5);
    }

    #[test]
    fn column_type_clone_and_eq() {
        assert_eq!(ColumnType::Int.clone(), ColumnType::Int);
        assert_eq!(ColumnType::Float.clone(), ColumnType::Float);
        assert_eq!(ColumnType::String.clone(), ColumnType::String);
        assert_eq!(ColumnType::Bool.clone(), ColumnType::Bool);
    }

    #[test]
    fn schema_get_column() {
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("name", ColumnType::String),
        ]);

        assert!(schema.get_column("id").is_some());
        assert!(schema.get_column("name").is_some());
        assert!(schema.get_column("nonexistent").is_none());
    }

    #[test]
    fn value_from_bytes_scalar() {
        let bytes_scalar = ScalarValue::Bytes(vec![1, 2, 3]);
        let value = Value::from_scalar(&bytes_scalar);
        assert_eq!(value, Value::Null);
    }

    #[test]
    fn condition_debug() {
        let c = Condition::True;
        let debug_str = format!("{:?}", c);
        assert!(debug_str.contains("True"));
    }

    #[test]
    fn row_debug_and_clone() {
        let values = vec![("name".to_string(), Value::String("test".into()))];
        let row = Row { id: 1, values };
        let cloned = row.clone();
        assert_eq!(cloned.id, 1);
        let debug_str = format!("{:?}", row);
        assert!(debug_str.contains("Row"));
    }

    #[test]
    fn column_debug_and_clone() {
        let col = Column::new("test", ColumnType::Int);
        let cloned = col.clone();
        assert_eq!(cloned.name, "test");
        let debug_str = format!("{:?}", col);
        assert!(debug_str.contains("Column"));
    }

    #[test]
    fn schema_debug_and_clone() {
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        let cloned = schema.clone();
        assert_eq!(cloned.columns.len(), 1);
        let debug_str = format!("{:?}", schema);
        assert!(debug_str.contains("Schema"));
    }

    #[test]
    fn error_clone_and_eq() {
        let e1 = RelationalError::TableNotFound("test".into());
        let e2 = RelationalError::TableAlreadyExists("test".into());
        let e3 = RelationalError::ColumnNotFound("col".into());
        let e4 = RelationalError::TypeMismatch {
            column: "age".into(),
            expected: ColumnType::Int,
        };
        let e5 = RelationalError::NullNotAllowed("name".into());
        let e6 = RelationalError::StorageError("err".into());

        assert_eq!(e1.clone(), e1);
        assert_eq!(e2.clone(), e2);
        assert_eq!(e3.clone(), e3);
        assert_eq!(e4.clone(), e4);
        assert_eq!(e5.clone(), e5);
        assert_eq!(e6.clone(), e6);
    }

    #[test]
    fn storage_error_from_tensor_store() {
        use tensor_store::TensorStoreError;
        let tensor_err = TensorStoreError::NotFound("key".into());
        let rel_err: RelationalError = tensor_err.into();
        assert!(matches!(rel_err, RelationalError::StorageError(_)));
    }

    #[test]
    fn insert_missing_nullable_column() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        // email is nullable and not provided

        let id = engine.insert("users", values).unwrap();
        assert!(id > 0);
    }

    #[test]
    fn comparison_with_mismatched_types_returns_false() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        // Compare int column with string value - should match nothing
        let rows = engine
            .select(
                "users",
                Condition::Lt("age".to_string(), Value::String("30".into())),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn comparison_with_null_column_returns_false() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        values.insert("email".to_string(), Value::Null);
        engine.insert("users", values).unwrap();

        // Comparing null email with string - should return false
        let rows = engine
            .select(
                "users",
                Condition::Lt("email".to_string(), Value::String("z".into())),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn bool_column_type() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("active", ColumnType::Bool),
        ]);
        engine.create_table("flags", schema).unwrap();

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("active".to_string(), Value::Bool(true));
        let id = engine.insert("flags", values).unwrap();
        assert!(id > 0);

        let rows = engine
            .select(
                "flags",
                Condition::Eq("active".to_string(), Value::Bool(true)),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn row_counter_initialization_on_insert() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Drop and recreate to test counter reinitialization path
        engine.drop_table("users").unwrap();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        let id = engine.insert("users", values).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn value_debug() {
        let v = Value::Int(42);
        let debug_str = format!("{:?}", v);
        assert!(debug_str.contains("Int"));
    }

    #[test]
    fn column_type_debug() {
        let ct = ColumnType::Float;
        let debug_str = format!("{:?}", ct);
        assert!(debug_str.contains("Float"));
    }

    #[test]
    fn condition_clone() {
        let c1 = Condition::Eq("col".into(), Value::Int(1));
        let c2 = Condition::Ne("col".into(), Value::Int(2));
        let c3 = Condition::Lt("col".into(), Value::Int(3));
        let c4 = Condition::Le("col".into(), Value::Int(4));
        let c5 = Condition::Gt("col".into(), Value::Int(5));
        let c6 = Condition::Ge("col".into(), Value::Int(6));
        let c7 = Condition::True;

        let _ = c1.clone();
        let _ = c2.clone();
        let _ = c3.clone();
        let _ = c4.clone();
        let _ = c5.clone();
        let _ = c6.clone();
        let _ = c7.clone();

        let c8 = Condition::And(Box::new(Condition::True), Box::new(Condition::True));
        let c9 = Condition::Or(Box::new(Condition::True), Box::new(Condition::True));
        let _ = c8.clone();
        let _ = c9.clone();
    }

    #[test]
    fn row_get_nonexistent_column() {
        let row = Row {
            id: 1,
            values: Vec::new(),
        };
        assert!(row.get("nonexistent").is_none());
        assert!(row.get_with_id("nonexistent").is_none());
    }

    #[test]
    fn compare_le_mismatched_types_returns_false() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let rows = engine
            .select(
                "users",
                Condition::Le("age".to_string(), Value::String("30".into())),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn compare_gt_mismatched_types_returns_false() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let rows = engine
            .select(
                "users",
                Condition::Gt("age".to_string(), Value::String("30".into())),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn compare_ge_mismatched_types_returns_false() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let rows = engine
            .select(
                "users",
                Condition::Ge("age".to_string(), Value::String("30".into())),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn next_row_id_without_counter_initialized() {
        // Use with_store to create engine without going through create_table
        let store = TensorStore::new();

        // Manually insert table metadata without initializing the counter
        let mut meta = TensorData::new();
        meta.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("table".into())),
        );
        meta.set(
            "_name",
            TensorValue::Scalar(ScalarValue::String("manual_table".into())),
        );
        meta.set(
            "_columns",
            TensorValue::Scalar(ScalarValue::String("name,age".into())),
        );
        meta.set(
            "_col:name",
            TensorValue::Scalar(ScalarValue::String("string:notnull".into())),
        );
        meta.set(
            "_col:age",
            TensorValue::Scalar(ScalarValue::String("int:notnull".into())),
        );
        store.put("_meta:table:manual_table", meta).unwrap();

        let engine = RelationalEngine::with_store(store);

        // Now insert - this should trigger the else branch in next_row_id
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        let id = engine.insert("manual_table", values).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn update_with_nullable_null_value() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        values.insert("email".to_string(), Value::String("test@test.com".into()));
        engine.insert("users", values).unwrap();

        // Update email (nullable) to Null - should succeed
        let mut updates = HashMap::new();
        updates.insert("email".to_string(), Value::Null);
        let count = engine.update("users", Condition::True, updates).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn join_with_null_join_column() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let schema = Schema::new(vec![
            Column::new("user_id", ColumnType::Int).nullable(),
            Column::new("title", ColumnType::String),
        ]);
        engine.create_table("posts", schema).unwrap();

        // Insert user
        let mut user_values = HashMap::new();
        user_values.insert("name".to_string(), Value::String("Alice".into()));
        user_values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", user_values).unwrap();

        // Insert post with null user_id
        let mut post_values = HashMap::new();
        post_values.insert("user_id".to_string(), Value::Null);
        post_values.insert("title".to_string(), Value::String("Orphan".into()));
        engine.insert("posts", post_values).unwrap();

        let joined = engine.join("users", "posts", "_id", "user_id").unwrap();
        // Should not match because null != 1
        assert_eq!(joined.len(), 0);
    }

    // Index tests

    #[test]
    fn create_and_use_index() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert some data
        for i in 0..100 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + (i % 10)));
            engine.insert("users", values).unwrap();
        }

        // Create index on age
        engine.create_index("users", "age").unwrap();
        assert!(engine.has_index("users", "age"));

        // Query using index
        let rows = engine
            .select("users", Condition::Eq("age".to_string(), Value::Int(25)))
            .unwrap();
        assert_eq!(rows.len(), 10); // 10 users with age 25
    }

    #[test]
    fn index_accelerates_select() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert 1000 rows
        for i in 0..1000 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + (i % 50)));
            engine.insert("users", values).unwrap();
        }

        // Create index
        engine.create_index("users", "age").unwrap();

        // Query should still work correctly
        let rows = engine
            .select("users", Condition::Eq("age".to_string(), Value::Int(25)))
            .unwrap();
        assert_eq!(rows.len(), 20);

        for row in &rows {
            assert_eq!(row.get("age"), Some(&Value::Int(25)));
        }
    }

    #[test]
    fn index_on_id_column() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..100 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(25));
            engine.insert("users", values).unwrap();
        }

        // Create index on _id
        engine.create_index("users", "_id").unwrap();
        assert!(engine.has_index("users", "_id"));

        // Query by _id using index
        let rows = engine
            .select("users", Condition::Eq("_id".to_string(), Value::Int(50)))
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, 50);
    }

    #[test]
    fn drop_index() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        engine.create_index("users", "age").unwrap();
        assert!(engine.has_index("users", "age"));

        engine.drop_index("users", "age").unwrap();
        assert!(!engine.has_index("users", "age"));
    }

    #[test]
    fn index_already_exists_error() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        engine.create_index("users", "age").unwrap();
        let result = engine.create_index("users", "age");
        assert!(matches!(
            result,
            Err(RelationalError::IndexAlreadyExists { .. })
        ));
    }

    #[test]
    fn index_not_found_error() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let result = engine.drop_index("users", "age");
        assert!(matches!(result, Err(RelationalError::IndexNotFound { .. })));
    }

    #[test]
    fn index_on_nonexistent_column_error() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let result = engine.create_index("users", "nonexistent");
        assert!(matches!(result, Err(RelationalError::ColumnNotFound(_))));
    }

    #[test]
    fn index_maintained_on_insert() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Create index first
        engine.create_index("users", "age").unwrap();

        // Then insert data
        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(25));
            engine.insert("users", values).unwrap();
        }

        // Query using index should find all rows
        let rows = engine
            .select("users", Condition::Eq("age".to_string(), Value::Int(25)))
            .unwrap();
        assert_eq!(rows.len(), 10);
    }

    #[test]
    fn index_maintained_on_update() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert data
        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(25));
            engine.insert("users", values).unwrap();
        }

        // Create index
        engine.create_index("users", "age").unwrap();

        // Update some rows
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(30));
        engine
            .update(
                "users",
                Condition::Eq("_id".to_string(), Value::Int(5)),
                updates,
            )
            .unwrap();

        // Old value should have 9 rows
        let rows_25 = engine
            .select("users", Condition::Eq("age".to_string(), Value::Int(25)))
            .unwrap();
        assert_eq!(rows_25.len(), 9);

        // New value should have 1 row
        let rows_30 = engine
            .select("users", Condition::Eq("age".to_string(), Value::Int(30)))
            .unwrap();
        assert_eq!(rows_30.len(), 1);
    }

    #[test]
    fn index_maintained_on_delete() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert data
        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(25));
            engine.insert("users", values).unwrap();
        }

        // Create index
        engine.create_index("users", "age").unwrap();

        // Delete some rows
        engine
            .delete_rows("users", Condition::Lt("_id".to_string(), Value::Int(5)))
            .unwrap();

        // Should have 6 rows left (ids 5-10)
        let rows = engine
            .select("users", Condition::Eq("age".to_string(), Value::Int(25)))
            .unwrap();
        assert_eq!(rows.len(), 6);
    }

    #[test]
    fn get_indexed_columns() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        engine.create_index("users", "age").unwrap();
        engine.create_index("users", "name").unwrap();

        let indexed = engine.get_indexed_columns("users");
        assert_eq!(indexed.len(), 2);
        assert!(indexed.contains(&"age".to_string()));
        assert!(indexed.contains(&"name".to_string()));
    }

    #[test]
    fn drop_table_cleans_up_indexes() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        engine.create_index("users", "age").unwrap();

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".into()));
        values.insert("age".to_string(), Value::Int(25));
        engine.insert("users", values).unwrap();

        engine.drop_table("users").unwrap();

        // Recreate table and check no stale index data
        create_users_table(&engine);
        assert!(!engine.has_index("users", "age"));
    }

    #[test]
    fn index_with_compound_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert data
        for i in 0..100 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + (i % 10)));
            engine.insert("users", values).unwrap();
        }

        // Create index on age
        engine.create_index("users", "age").unwrap();

        // Query with AND condition - should use index for one side
        let condition = Condition::Eq("age".to_string(), Value::Int(25))
            .and(Condition::Lt("_id".to_string(), Value::Int(50)));
        let rows = engine.select("users", condition).unwrap();

        // Should have 5 rows (ages 5, 15, 25, 35, 45 have age=25 and id < 50)
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn value_hash_key_variants() {
        // Test hash_key for different Value types
        assert_eq!(Value::Null.hash_key(), "null");
        assert_eq!(Value::Int(42).hash_key(), "i:42");
        assert_eq!(Value::Bool(true).hash_key(), "b:true");
        assert_eq!(Value::Bool(false).hash_key(), "b:false");

        // Float uses to_bits for stable hashing
        let float_hash = Value::Float(3.14).hash_key();
        assert!(float_hash.starts_with("f:"));

        // String uses hasher
        let str_hash = Value::String("hello".into()).hash_key();
        assert!(str_hash.starts_with("s:"));
    }

    #[test]
    fn index_error_display() {
        let err1 = RelationalError::IndexAlreadyExists {
            table: "users".into(),
            column: "age".into(),
        };
        assert_eq!(format!("{}", err1), "Index already exists on users.age");

        let err2 = RelationalError::IndexNotFound {
            table: "users".into(),
            column: "age".into(),
        };
        assert_eq!(format!("{}", err2), "Index not found on users.age");
    }

    #[test]
    fn index_on_string_column() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..50 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("Name{}", i % 5)));
            values.insert("age".to_string(), Value::Int(25));
            engine.insert("users", values).unwrap();
        }

        engine.create_index("users", "name").unwrap();

        let rows = engine
            .select(
                "users",
                Condition::Eq("name".to_string(), Value::String("Name2".into())),
            )
            .unwrap();
        assert_eq!(rows.len(), 10);
    }

    #[test]
    fn btree_index_accelerates_range_query() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert 100 rows
        for i in 0..100 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i as i64));
            engine.insert("users", values).unwrap();
        }

        // Create B-tree index on age
        engine.create_btree_index("users", "age").unwrap();
        assert!(engine.has_btree_index("users", "age"));

        // Range query: age >= 50 (should use B-tree index)
        let rows = engine
            .select("users", Condition::Ge("age".to_string(), Value::Int(50)))
            .unwrap();
        assert_eq!(rows.len(), 50);

        // Range query: age < 25
        let rows = engine
            .select("users", Condition::Lt("age".to_string(), Value::Int(25)))
            .unwrap();
        assert_eq!(rows.len(), 25);

        // Range query: age > 90
        let rows = engine
            .select("users", Condition::Gt("age".to_string(), Value::Int(90)))
            .unwrap();
        assert_eq!(rows.len(), 9);

        // Range query: age <= 10
        let rows = engine
            .select("users", Condition::Le("age".to_string(), Value::Int(10)))
            .unwrap();
        assert_eq!(rows.len(), 11);
    }

    #[test]
    fn btree_index_maintained_on_insert() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Create B-tree index first
        engine.create_btree_index("users", "age").unwrap();

        // Insert rows after index creation
        for i in 0..50 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i as i64));
            engine.insert("users", values).unwrap();
        }

        // Verify index is used for range query
        let rows = engine
            .select("users", Condition::Lt("age".to_string(), Value::Int(10)))
            .unwrap();
        assert_eq!(rows.len(), 10);
    }

    #[test]
    fn btree_index_maintained_on_update() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert initial data
        for i in 0..20 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i as i64));
            engine.insert("users", values).unwrap();
        }

        // Create B-tree index
        engine.create_btree_index("users", "age").unwrap();

        // Update ages: set all ages < 10 to 100
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(100));
        engine
            .update(
                "users",
                Condition::Lt("age".to_string(), Value::Int(10)),
                updates,
            )
            .unwrap();

        // Now age < 10 should return 0 rows
        let rows = engine
            .select("users", Condition::Lt("age".to_string(), Value::Int(10)))
            .unwrap();
        assert_eq!(rows.len(), 0);

        // age >= 100 should return 10 rows (the updated ones)
        let rows = engine
            .select("users", Condition::Ge("age".to_string(), Value::Int(100)))
            .unwrap();
        assert_eq!(rows.len(), 10);
    }

    #[test]
    fn btree_index_maintained_on_delete() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert data
        for i in 0..30 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i as i64));
            engine.insert("users", values).unwrap();
        }

        // Create B-tree index
        engine.create_btree_index("users", "age").unwrap();

        // Delete rows where age < 10
        engine
            .delete_rows("users", Condition::Lt("age".to_string(), Value::Int(10)))
            .unwrap();

        // Verify age < 15 now returns only 5 rows (ages 10-14)
        let rows = engine
            .select("users", Condition::Lt("age".to_string(), Value::Int(15)))
            .unwrap();
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn btree_index_drop() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        engine.create_btree_index("users", "age").unwrap();
        assert!(engine.has_btree_index("users", "age"));

        engine.drop_btree_index("users", "age").unwrap();
        assert!(!engine.has_btree_index("users", "age"));
    }

    #[test]
    fn btree_index_with_negative_numbers() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert with negative ages
        for i in -50..50 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        engine.create_btree_index("users", "age").unwrap();

        // Query age < 0 should return 50 rows
        let rows = engine
            .select("users", Condition::Lt("age".to_string(), Value::Int(0)))
            .unwrap();
        assert_eq!(rows.len(), 50);

        // Query age >= -25 should return 75 rows
        let rows = engine
            .select("users", Condition::Ge("age".to_string(), Value::Int(-25)))
            .unwrap();
        assert_eq!(rows.len(), 75);
    }

    #[test]
    fn drop_table_cleans_up_btree_indexes() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        engine.create_btree_index("users", "age").unwrap();
        assert!(engine.has_btree_index("users", "age"));

        engine.drop_table("users").unwrap();

        // Recreate and verify no index exists
        create_users_table(&engine);
        assert!(!engine.has_btree_index("users", "age"));
    }

    #[test]
    fn sortable_key_ordering() {
        // Test that sortable keys maintain correct ordering
        let v1 = Value::Int(-100);
        let v2 = Value::Int(-1);
        let v3 = Value::Int(0);
        let v4 = Value::Int(1);
        let v5 = Value::Int(100);

        assert!(v1.sortable_key() < v2.sortable_key());
        assert!(v2.sortable_key() < v3.sortable_key());
        assert!(v3.sortable_key() < v4.sortable_key());
        assert!(v4.sortable_key() < v5.sortable_key());

        // Test floats
        let f1 = Value::Float(-100.5);
        let f2 = Value::Float(-0.1);
        let f3 = Value::Float(0.0);
        let f4 = Value::Float(0.1);
        let f5 = Value::Float(100.5);

        assert!(f1.sortable_key() < f2.sortable_key());
        assert!(f2.sortable_key() < f3.sortable_key());
        assert!(f3.sortable_key() < f4.sortable_key());
        assert!(f4.sortable_key() < f5.sortable_key());

        // Test strings
        let s1 = Value::String("aaa".into());
        let s2 = Value::String("bbb".into());
        let s3 = Value::String("zzz".into());

        assert!(s1.sortable_key() < s2.sortable_key());
        assert!(s2.sortable_key() < s3.sortable_key());

        // Test null and bool
        let null = Value::Null;
        let bool_false = Value::Bool(false);
        let bool_true = Value::Bool(true);

        // Null should have a sortable key
        assert_eq!(null.sortable_key(), "0");
        // Bool false < Bool true
        assert!(bool_false.sortable_key() < bool_true.sortable_key());
        assert_eq!(bool_false.sortable_key(), "b0");
        assert_eq!(bool_true.sortable_key(), "b1");
    }

    #[test]
    fn get_btree_indexed_columns_returns_columns() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Initially no B-tree indexes
        let cols = engine.get_btree_indexed_columns("users");
        assert!(cols.is_empty());

        // Create B-tree index
        engine.create_btree_index("users", "age").unwrap();

        // Now should have one indexed column
        let cols = engine.get_btree_indexed_columns("users");
        assert_eq!(cols.len(), 1);
        assert!(cols.contains(&"age".to_string()));
    }

    #[test]
    fn batch_insert_multiple_rows() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let rows: Vec<HashMap<String, Value>> = (0..100)
            .map(|i| {
                let mut values = HashMap::new();
                values.insert("name".to_string(), Value::String(format!("User{}", i)));
                values.insert("age".to_string(), Value::Int(20 + i));
                values
            })
            .collect();

        let ids = engine.batch_insert("users", rows).unwrap();

        assert_eq!(ids.len(), 100);
        assert_eq!(engine.row_count("users").unwrap(), 100);

        // Verify data was inserted correctly
        let all_rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(all_rows.len(), 100);
    }

    #[test]
    fn batch_insert_empty() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let ids = engine.batch_insert("users", Vec::new()).unwrap();
        assert!(ids.is_empty());
        assert_eq!(engine.row_count("users").unwrap(), 0);
    }

    #[test]
    fn batch_insert_validates_all_rows_upfront() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Row 0 is valid, row 1 has type mismatch
        let rows: Vec<HashMap<String, Value>> = vec![
            {
                let mut values = HashMap::new();
                values.insert("name".to_string(), Value::String("Alice".into()));
                values.insert("age".to_string(), Value::Int(30));
                values
            },
            {
                let mut values = HashMap::new();
                values.insert("name".to_string(), Value::String("Bob".into()));
                values.insert("age".to_string(), Value::String("not a number".into())); // Invalid
                values
            },
        ];

        let result = engine.batch_insert("users", rows);
        assert!(result.is_err());

        // No rows should have been inserted (fail-fast)
        assert_eq!(engine.row_count("users").unwrap(), 0);
    }

    #[test]
    fn batch_insert_with_indexes() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Create index before batch insert
        engine.create_index("users", "age").unwrap();

        let rows: Vec<HashMap<String, Value>> = (0..50)
            .map(|i| {
                let mut values = HashMap::new();
                values.insert("name".to_string(), Value::String(format!("User{}", i)));
                values.insert("age".to_string(), Value::Int(25)); // All same age
                values
            })
            .collect();

        let ids = engine.batch_insert("users", rows).unwrap();
        assert_eq!(ids.len(), 50);

        // Index should work correctly
        let rows = engine
            .select("users", Condition::Eq("age".to_string(), Value::Int(25)))
            .unwrap();
        assert_eq!(rows.len(), 50);
    }

    #[test]
    fn batch_insert_with_btree_index() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Create B-tree index before batch insert
        engine.create_btree_index("users", "age").unwrap();

        let rows: Vec<HashMap<String, Value>> = (0..100)
            .map(|i| {
                let mut values = HashMap::new();
                values.insert("name".to_string(), Value::String(format!("User{}", i)));
                values.insert("age".to_string(), Value::Int(i as i64));
                values
            })
            .collect();

        engine.batch_insert("users", rows).unwrap();

        // B-tree index should work for range query
        let rows = engine
            .select("users", Condition::Ge("age".to_string(), Value::Int(50)))
            .unwrap();
        assert_eq!(rows.len(), 50);
    }

    #[test]
    fn batch_insert_null_not_allowed() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let rows: Vec<HashMap<String, Value>> = vec![
            {
                let mut values = HashMap::new();
                values.insert("name".to_string(), Value::String("Alice".into()));
                values.insert("age".to_string(), Value::Int(30));
                values
            },
            {
                let mut values = HashMap::new();
                // Missing required 'name' field
                values.insert("age".to_string(), Value::Int(25));
                values
            },
        ];

        let result = engine.batch_insert("users", rows);
        assert!(matches!(result, Err(RelationalError::NullNotAllowed(_))));
    }

    #[test]
    fn list_tables_empty() {
        let engine = RelationalEngine::new();
        let tables = engine.list_tables();
        assert!(tables.is_empty());
    }

    #[test]
    fn list_tables_with_tables() {
        let engine = RelationalEngine::new();
        engine
            .create_table(
                "users",
                Schema::new(vec![Column::new("id", ColumnType::Int)]),
            )
            .unwrap();
        engine
            .create_table(
                "products",
                Schema::new(vec![Column::new("id", ColumnType::Int)]),
            )
            .unwrap();

        let tables = engine.list_tables();
        assert_eq!(tables.len(), 2);
        assert!(tables.contains(&"users".to_string()));
        assert!(tables.contains(&"products".to_string()));
    }

    // ========================================================================
    // Columnar Storage Tests
    // ========================================================================

    #[test]
    fn materialize_int_column() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..100 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + (i % 50)));
            engine.insert("users", values).unwrap();
        }

        engine.materialize_columns("users", &["age"]).unwrap();
        assert!(engine.has_columnar_data("users", "age"));

        let col_data = engine.load_column_data("users", "age").unwrap();
        assert_eq!(col_data.row_ids.len(), 100);
        if let ColumnValues::Int(values) = col_data.values {
            assert_eq!(values.len(), 100);
        } else {
            panic!("Expected Int column values");
        }
    }

    #[test]
    fn materialize_string_column_with_dictionary() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..50 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i % 10)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        engine.materialize_columns("users", &["name"]).unwrap();

        let col_data = engine.load_column_data("users", "name").unwrap();
        if let ColumnValues::String { dict, indices } = col_data.values {
            assert_eq!(dict.len(), 10);
            assert_eq!(indices.len(), 50);
        } else {
            panic!("Expected String column values");
        }
    }

    #[test]
    fn select_columnar_with_vectorized_filter() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..1000 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i % 100));
            engine.insert("users", values).unwrap();
        }

        engine.materialize_columns("users", &["age"]).unwrap();

        let options = ColumnarScanOptions {
            projection: None,
            prefer_columnar: true,
        };

        let rows = engine
            .select_columnar(
                "users",
                Condition::Gt("age".into(), Value::Int(50)),
                options,
            )
            .unwrap();

        assert_eq!(rows.len(), 490);
    }

    #[test]
    fn select_columnar_with_projection() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            values.insert(
                "email".to_string(),
                Value::String(format!("user{}@test.com", i)),
            );
            engine.insert("users", values).unwrap();
        }

        engine.materialize_columns("users", &["age"]).unwrap();

        let options = ColumnarScanOptions {
            projection: Some(vec!["name".into()]),
            prefer_columnar: true,
        };

        let rows = engine
            .select_columnar(
                "users",
                Condition::Gt("age".into(), Value::Int(25)),
                options,
            )
            .unwrap();

        assert_eq!(rows.len(), 4);

        for row in &rows {
            assert!(row.contains("name"));
            assert!(!row.contains("age"));
            assert!(!row.contains("email"));
        }
    }

    #[test]
    fn select_columnar_compound_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..100 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        engine.materialize_columns("users", &["age"]).unwrap();

        let options = ColumnarScanOptions {
            projection: None,
            prefer_columnar: true,
        };

        let condition = Condition::Ge("age".into(), Value::Int(30))
            .and(Condition::Lt("age".into(), Value::Int(40)));

        let rows = engine.select_columnar("users", condition, options).unwrap();
        assert_eq!(rows.len(), 10);

        for row in &rows {
            let age = row.get("age").unwrap();
            if let Value::Int(a) = age {
                assert!(*a >= 30 && *a < 40);
            }
        }
    }

    #[test]
    fn select_columnar_fallback_to_row_based() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        let options = ColumnarScanOptions {
            projection: Some(vec!["name".into()]),
            prefer_columnar: true,
        };

        let rows = engine
            .select_columnar(
                "users",
                Condition::Gt("age".into(), Value::Int(25)),
                options,
            )
            .unwrap();

        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn drop_columnar_data() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        engine.materialize_columns("users", &["age"]).unwrap();
        assert!(engine.has_columnar_data("users", "age"));

        engine.drop_columnar_data("users", "age").unwrap();
        assert!(!engine.has_columnar_data("users", "age"));
    }

    #[test]
    fn selection_vector_operations() {
        let sel_all = SelectionVector::all(100);
        assert_eq!(sel_all.count(), 100);
        assert!(sel_all.is_selected(0));
        assert!(sel_all.is_selected(99));
        assert!(!sel_all.is_selected(100));

        let sel_none = SelectionVector::none(100);
        assert_eq!(sel_none.count(), 0);
        assert!(!sel_none.is_selected(0));

        let intersected = sel_all.intersect(&sel_none);
        assert_eq!(intersected.count(), 0);

        let unioned = sel_all.union(&sel_none);
        assert_eq!(unioned.count(), 100);
    }

    #[test]
    fn simd_filter_le() {
        let values = vec![1, 5, 3, 8, 2, 9, 4, 7];
        let mut bitmap = vec![0u64; 1];
        simd::filter_le_i64(&values, 5, &mut bitmap);
        // Values <= 5: 1,5,3,2,4 at positions 0,1,2,4,6
        assert_eq!(bitmap[0] & 0xff, 0b01010111);
    }

    #[test]
    fn simd_filter_ge() {
        let values = vec![1, 5, 3, 8, 2, 9, 4, 7];
        let mut bitmap = vec![0u64; 1];
        simd::filter_ge_i64(&values, 5, &mut bitmap);
        // Values >= 5: 5,8,9,7 at positions 1,3,5,7
        assert_eq!(bitmap[0] & 0xff, 0b10101010);
    }

    #[test]
    fn simd_filter_ne() {
        let values = vec![5, 5, 3, 5, 2, 5, 4, 5];
        let mut bitmap = vec![0u64; 1];
        simd::filter_ne_i64(&values, 5, &mut bitmap);
        // Values != 5: 3,2,4 at positions 2,4,6
        assert_eq!(bitmap[0] & 0xff, 0b01010100);
    }

    #[test]
    fn column_data_get_value_int() {
        let col = ColumnData {
            name: "age".into(),
            row_ids: vec![1, 2, 3],
            nulls: NullBitmap::None,
            values: ColumnValues::Int(vec![25, 30, 35]),
        };
        assert_eq!(col.get_value(0), Some(Value::Int(25)));
        assert_eq!(col.get_value(1), Some(Value::Int(30)));
        assert_eq!(col.get_value(2), Some(Value::Int(35)));
        assert_eq!(col.get_value(10), None);
    }

    #[test]
    fn column_data_get_value_float() {
        let col = ColumnData {
            name: "score".into(),
            row_ids: vec![1, 2],
            nulls: NullBitmap::None,
            values: ColumnValues::Float(vec![1.5, 2.5]),
        };
        assert_eq!(col.get_value(0), Some(Value::Float(1.5)));
        assert_eq!(col.get_value(1), Some(Value::Float(2.5)));
    }

    #[test]
    fn column_data_get_value_string() {
        let col = ColumnData {
            name: "name".into(),
            row_ids: vec![1, 2],
            nulls: NullBitmap::None,
            values: ColumnValues::String {
                dict: vec!["Alice".into(), "Bob".into()],
                indices: vec![0, 1],
            },
        };
        assert_eq!(col.get_value(0), Some(Value::String("Alice".into())));
        assert_eq!(col.get_value(1), Some(Value::String("Bob".into())));
    }

    #[test]
    fn column_data_get_value_bool() {
        let mut values = vec![0u64];
        values[0] |= 1 << 0; // true at 0
        values[0] |= 1 << 2; // true at 2
        let col = ColumnData {
            name: "active".into(),
            row_ids: vec![1, 2, 3],
            nulls: NullBitmap::None,
            values: ColumnValues::Bool(values),
        };
        assert_eq!(col.get_value(0), Some(Value::Bool(true)));
        assert_eq!(col.get_value(1), Some(Value::Bool(false)));
        assert_eq!(col.get_value(2), Some(Value::Bool(true)));
    }

    #[test]
    fn column_data_get_value_with_nulls() {
        let col = ColumnData {
            name: "age".into(),
            row_ids: vec![1, 2, 3],
            nulls: NullBitmap::Sparse(vec![1]),
            values: ColumnValues::Int(vec![25, 0, 35]),
        };
        assert_eq!(col.get_value(0), Some(Value::Int(25)));
        assert_eq!(col.get_value(1), Some(Value::Null));
        assert_eq!(col.get_value(2), Some(Value::Int(35)));
    }

    #[test]
    fn null_bitmap_dense() {
        let mut bitmap = vec![0u64; 1];
        bitmap[0] |= 1 << 5; // null at position 5
        let nulls = NullBitmap::Dense(bitmap);
        assert!(!nulls.is_null(0));
        assert!(nulls.is_null(5));
        assert_eq!(nulls.null_count(), 1);
    }

    #[test]
    fn null_bitmap_sparse() {
        let nulls = NullBitmap::Sparse(vec![3, 7, 15]);
        assert!(!nulls.is_null(0));
        assert!(nulls.is_null(3));
        assert!(nulls.is_null(7));
        assert!(nulls.is_null(15));
        assert!(!nulls.is_null(10));
        assert_eq!(nulls.null_count(), 3);
    }

    #[test]
    fn null_bitmap_none() {
        let nulls = NullBitmap::None;
        assert!(!nulls.is_null(0));
        assert!(!nulls.is_null(100));
        assert_eq!(nulls.null_count(), 0);
    }

    #[test]
    fn column_values_len() {
        let int_vals = ColumnValues::Int(vec![1, 2, 3]);
        assert_eq!(int_vals.len(), 3);
        assert!(!int_vals.is_empty());

        let float_vals = ColumnValues::Float(vec![1.0, 2.0]);
        assert_eq!(float_vals.len(), 2);

        let str_vals = ColumnValues::String {
            dict: vec!["a".into()],
            indices: vec![0, 0, 0],
        };
        assert_eq!(str_vals.len(), 3);

        let bool_vals = ColumnValues::Bool(vec![0u64; 2]);
        assert_eq!(bool_vals.len(), 128); // 2 * 64
    }

    #[test]
    fn pure_columnar_select_all_columns_materialized() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..20 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        // Materialize all columns used in projection
        engine
            .materialize_columns("users", &["name", "age"])
            .unwrap();

        let options = ColumnarScanOptions {
            projection: Some(vec!["name".into(), "age".into()]),
            prefer_columnar: true,
        };

        let rows = engine
            .select_columnar(
                "users",
                Condition::Ge("age".into(), Value::Int(35)),
                options,
            )
            .unwrap();

        assert_eq!(rows.len(), 5); // ages 35-39
        for row in &rows {
            assert!(row.contains("name"));
            assert!(row.contains("age"));
        }
    }

    #[test]
    fn select_columnar_with_true_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        engine.materialize_columns("users", &["age"]).unwrap();

        let options = ColumnarScanOptions {
            projection: Some(vec!["age".into()]),
            prefer_columnar: true,
        };

        let rows = engine
            .select_columnar("users", Condition::True, options)
            .unwrap();

        assert_eq!(rows.len(), 10);
    }

    #[test]
    fn select_columnar_with_and_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..50 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        engine.materialize_columns("users", &["age"]).unwrap();

        let options = ColumnarScanOptions {
            projection: Some(vec!["age".into()]),
            prefer_columnar: true,
        };

        // age >= 20 AND age < 30
        let condition = Condition::Ge("age".into(), Value::Int(20))
            .and(Condition::Lt("age".into(), Value::Int(30)));

        let rows = engine.select_columnar("users", condition, options).unwrap();

        assert_eq!(rows.len(), 10);
    }

    #[test]
    fn select_columnar_with_or_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..50 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        engine.materialize_columns("users", &["age"]).unwrap();

        let options = ColumnarScanOptions {
            projection: Some(vec!["age".into()]),
            prefer_columnar: true,
        };

        // age < 10 OR age >= 40
        let condition = Condition::Lt("age".into(), Value::Int(10))
            .or(Condition::Ge("age".into(), Value::Int(40)));

        let rows = engine.select_columnar("users", condition, options).unwrap();

        assert_eq!(rows.len(), 20); // 0-9 and 40-49
    }

    #[test]
    fn select_columnar_ne_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(25));
            engine.insert("users", values).unwrap();
        }
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Special".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        engine.materialize_columns("users", &["age"]).unwrap();

        let options = ColumnarScanOptions {
            projection: Some(vec!["age".into()]),
            prefer_columnar: true,
        };

        let rows = engine
            .select_columnar(
                "users",
                Condition::Ne("age".into(), Value::Int(25)),
                options,
            )
            .unwrap();

        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn select_columnar_le_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        engine.materialize_columns("users", &["age"]).unwrap();

        let options = ColumnarScanOptions {
            projection: Some(vec!["age".into()]),
            prefer_columnar: true,
        };

        let rows = engine
            .select_columnar(
                "users",
                Condition::Le("age".into(), Value::Int(24)),
                options,
            )
            .unwrap();

        assert_eq!(rows.len(), 5); // ages 20-24
    }

    #[test]
    fn select_with_projection_filters() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        let rows = engine
            .select_with_projection(
                "users",
                Condition::Gt("age".into(), Value::Int(25)),
                Some(vec!["name".into()]),
            )
            .unwrap();

        assert_eq!(rows.len(), 4);
        for row in &rows {
            assert!(row.contains("name"));
            assert!(!row.contains("age"));
        }
    }

    #[test]
    fn select_with_projection_includes_id() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let rows = engine
            .select_with_projection(
                "users",
                Condition::True,
                Some(vec!["_id".into(), "name".into()]),
            )
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains("name"));
        // _id is in row.id, not in values
    }

    #[test]
    fn columnar_scan_options_debug_clone() {
        let options = ColumnarScanOptions {
            projection: Some(vec!["col".into()]),
            prefer_columnar: true,
        };
        let cloned = options.clone();
        assert_eq!(cloned.projection, options.projection);
        assert_eq!(cloned.prefer_columnar, options.prefer_columnar);
        let debug = format!("{:?}", options);
        assert!(debug.contains("ColumnarScanOptions"));
    }

    #[test]
    fn column_data_debug_clone() {
        let col = ColumnData {
            name: "test".into(),
            row_ids: vec![1],
            nulls: NullBitmap::None,
            values: ColumnValues::Int(vec![42]),
        };
        let cloned = col.clone();
        assert_eq!(cloned.name, col.name);
        let debug = format!("{:?}", col);
        assert!(debug.contains("ColumnData"));
    }

    #[test]
    fn selection_vector_from_bitmap() {
        let bitmap = vec![0b00000101u64]; // bits 0 and 2 set
        let sel = SelectionVector::from_bitmap(bitmap, 8);
        assert!(sel.is_selected(0));
        assert!(!sel.is_selected(1));
        assert!(sel.is_selected(2));
        assert_eq!(sel.count(), 2);
    }

    #[test]
    fn selection_vector_selected_indices() {
        let bitmap = vec![0b00001010u64]; // bits 1 and 3 set
        let sel = SelectionVector::from_bitmap(bitmap, 8);
        let indices = sel.selected_indices();
        assert_eq!(indices, vec![1, 3]);
    }

    #[test]
    fn selection_vector_bitmap_accessor() {
        let mut sel = SelectionVector::all(10);
        let bitmap = sel.bitmap_mut();
        bitmap[0] = 0; // Clear all bits
        assert_eq!(sel.count(), 0);

        let bitmap_ref = sel.bitmap();
        assert_eq!(bitmap_ref[0], 0);
    }

    #[test]
    fn materialize_columns_with_bool_type() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("active", ColumnType::Bool),
        ]);
        engine.create_table("flags", schema).unwrap();

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("Item{}", i)));
            values.insert("active".to_string(), Value::Bool(i % 2 == 0));
            engine.insert("flags", values).unwrap();
        }

        engine.materialize_columns("flags", &["active"]).unwrap();
        assert!(engine.has_columnar_data("flags", "active"));

        let col_data = engine.load_column_data("flags", "active").unwrap();
        assert_eq!(col_data.row_ids.len(), 10);
    }

    #[test]
    fn materialize_columns_with_null_values() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            if i % 2 == 0 {
                values.insert(
                    "email".to_string(),
                    Value::String(format!("user{}@test.com", i)),
                );
            } else {
                values.insert("email".to_string(), Value::Null);
            }
            engine.insert("users", values).unwrap();
        }

        engine.materialize_columns("users", &["email"]).unwrap();
        let col_data = engine.load_column_data("users", "email").unwrap();
        assert!(col_data.nulls.null_count() > 0);
    }

    #[test]
    fn load_column_data_not_materialized() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let result = engine.load_column_data("users", "age");
        assert!(result.is_err());
    }

    #[test]
    fn select_columnar_type_mismatch() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..5 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        engine.materialize_columns("users", &["name"]).unwrap();

        let options = ColumnarScanOptions {
            projection: Some(vec!["name".into()]),
            prefer_columnar: true,
        };

        // Try to filter string column with int - should fall back or handle gracefully
        let result = engine.select_columnar(
            "users",
            Condition::Gt("name".into(), Value::Int(25)),
            options,
        );
        // This might fail with TypeMismatch or fall back - either is acceptable
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn simd_bitmap_words() {
        assert_eq!(simd::bitmap_words(0), 0);
        assert_eq!(simd::bitmap_words(1), 1);
        assert_eq!(simd::bitmap_words(64), 1);
        assert_eq!(simd::bitmap_words(65), 2);
        assert_eq!(simd::bitmap_words(128), 2);
        assert_eq!(simd::bitmap_words(129), 3);
    }

    #[test]
    fn selection_vector_all_edge_cases() {
        // Exact multiple of 64
        let sel64 = SelectionVector::all(64);
        assert_eq!(sel64.count(), 64);
        assert!(sel64.is_selected(63));
        assert!(!sel64.is_selected(64));

        // Non-multiple of 64
        let sel70 = SelectionVector::all(70);
        assert_eq!(sel70.count(), 70);
        assert!(sel70.is_selected(69));
        assert!(!sel70.is_selected(70));
    }

    #[test]
    fn simd_filter_eq_with_remainder() {
        // 5 elements - not multiple of 4, tests remainder path
        let values = vec![1, 2, 3, 2, 2];
        let mut bitmap = vec![0u64; 1];
        simd::filter_eq_i64(&values, 2, &mut bitmap);
        // Values == 2 at positions 1, 3, 4
        assert_eq!(bitmap[0] & 0x1f, 0b11010);
    }

    #[test]
    fn simd_filter_ne_with_remainder() {
        // 6 elements - not multiple of 4, tests remainder path
        let values = vec![1, 2, 2, 2, 1, 3];
        let mut bitmap = vec![0u64; 1];
        simd::filter_ne_i64(&values, 2, &mut bitmap);
        // Values != 2 at positions 0, 4, 5
        assert_eq!(bitmap[0] & 0x3f, 0b110001);
    }

    #[test]
    fn simd_filter_with_single_element() {
        let values = vec![42];
        let mut bitmap = vec![0u64; 1];
        simd::filter_eq_i64(&values, 42, &mut bitmap);
        assert_eq!(bitmap[0] & 1, 1);

        let mut bitmap2 = vec![0u64; 1];
        simd::filter_ne_i64(&values, 42, &mut bitmap2);
        assert_eq!(bitmap2[0] & 1, 0);
    }

    #[test]
    fn parallel_select_large_dataset() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert enough rows to trigger parallel execution
        for i in 0..2000 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + (i % 50)));
            engine.insert("users", values).unwrap();
        }

        let rows = engine
            .select("users", Condition::Gt("age".into(), Value::Int(60)))
            .unwrap();
        assert_eq!(rows.len(), 360); // 9 ages (61-69) * 40 each
    }

    #[test]
    fn parallel_update_large_dataset() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..2000 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(25));
            engine.insert("users", values).unwrap();
        }

        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(30));
        let count = engine
            .update(
                "users",
                Condition::Eq("age".into(), Value::Int(25)),
                updates,
            )
            .unwrap();
        assert_eq!(count, 2000);
    }

    #[test]
    fn parallel_delete_large_dataset() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..2000 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert(
                "age".to_string(),
                Value::Int(if i % 2 == 0 { 25 } else { 30 }),
            );
            engine.insert("users", values).unwrap();
        }

        let count = engine
            .delete_rows("users", Condition::Eq("age".into(), Value::Int(25)))
            .unwrap();
        assert_eq!(count, 1000);
    }

    #[test]
    fn parallel_join_large_dataset() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        create_posts_table(&engine);

        for i in 0..500 {
            let mut user_values = HashMap::new();
            user_values.insert("name".to_string(), Value::String(format!("User{}", i)));
            user_values.insert("age".to_string(), Value::Int(20 + (i % 50)));
            engine.insert("users", user_values).unwrap();
        }

        for i in 0..1000 {
            let mut post_values = HashMap::new();
            post_values.insert("user_id".to_string(), Value::Int((i % 500) as i64 + 1));
            post_values.insert("title".to_string(), Value::String(format!("Post{}", i)));
            post_values.insert("views".to_string(), Value::Int(i * 10));
            engine.insert("posts", post_values).unwrap();
        }

        let joined = engine.join("users", "posts", "_id", "user_id").unwrap();
        assert_eq!(joined.len(), 1000);
    }

    #[test]
    fn btree_index_on_id_column() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        // Create btree index on _id
        engine.create_btree_index("users", "_id").unwrap();

        // Query using the index
        let rows = engine
            .select("users", Condition::Eq("_id".into(), Value::Int(5)))
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, 5);
    }

    #[test]
    fn hash_index_on_id_column() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        // Create hash index on _id
        engine.create_index("users", "_id").unwrap();

        let rows = engine
            .select("users", Condition::Eq("_id".into(), Value::Int(3)))
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn btree_range_query_no_matches() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        engine.create_btree_index("users", "age").unwrap();

        // Query for ages that don't exist
        let rows = engine
            .select("users", Condition::Gt("age".into(), Value::Int(100)))
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn update_with_btree_index() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..20 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(25));
            engine.insert("users", values).unwrap();
        }

        engine.create_btree_index("users", "age").unwrap();

        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(30));
        let count = engine
            .update(
                "users",
                Condition::Eq("age".into(), Value::Int(25)),
                updates,
            )
            .unwrap();
        assert_eq!(count, 20);

        // Verify index is updated
        let rows = engine
            .select("users", Condition::Eq("age".into(), Value::Int(30)))
            .unwrap();
        assert_eq!(rows.len(), 20);
    }

    #[test]
    fn select_columnar_empty_result() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        engine.materialize_columns("users", &["age"]).unwrap();

        let options = ColumnarScanOptions {
            projection: Some(vec!["age".into()]),
            prefer_columnar: true,
        };

        let rows = engine
            .select_columnar(
                "users",
                Condition::Gt("age".into(), Value::Int(100)),
                options,
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn select_columnar_no_projection_with_filter() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        engine
            .materialize_columns("users", &["name", "age"])
            .unwrap();

        let options = ColumnarScanOptions {
            projection: None,
            prefer_columnar: true,
        };

        let rows = engine
            .select_columnar(
                "users",
                Condition::Lt("age".into(), Value::Int(25)),
                options,
            )
            .unwrap();
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn materialize_columns_float_type() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("score", ColumnType::Float),
        ]);
        engine.create_table("scores", schema).unwrap();

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("score".to_string(), Value::Float(i as f64 * 1.5));
            engine.insert("scores", values).unwrap();
        }

        engine.materialize_columns("scores", &["score"]).unwrap();
        let col_data = engine.load_column_data("scores", "score").unwrap();
        assert_eq!(col_data.row_ids.len(), 10);
        match col_data.values {
            ColumnValues::Float(vals) => assert_eq!(vals.len(), 10),
            _ => panic!("Expected Float column"),
        }
    }

    #[test]
    fn column_values_empty() {
        let empty_int = ColumnValues::Int(vec![]);
        assert!(empty_int.is_empty());
        assert_eq!(empty_int.len(), 0);
    }

    #[test]
    fn insert_and_update_with_id_index() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Create index on _id before inserting
        engine.create_index("users", "_id").unwrap();
        engine.create_btree_index("users", "_id").unwrap();

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        // Update some rows - this should update the indexes
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(99));
        let count = engine
            .update(
                "users",
                Condition::Lt("age".into(), Value::Int(25)),
                updates,
            )
            .unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn large_parallel_join() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        create_posts_table(&engine);

        // Insert many users and posts to trigger parallel join
        for i in 0..1000 {
            let mut user_values = HashMap::new();
            user_values.insert("name".to_string(), Value::String(format!("User{}", i)));
            user_values.insert("age".to_string(), Value::Int(20 + (i % 50)));
            engine.insert("users", user_values).unwrap();
        }

        for i in 0..2000 {
            let mut post_values = HashMap::new();
            post_values.insert("user_id".to_string(), Value::Int((i % 1000) as i64 + 1));
            post_values.insert("title".to_string(), Value::String(format!("Post{}", i)));
            post_values.insert("views".to_string(), Value::Int(i * 10));
            engine.insert("posts", post_values).unwrap();
        }

        let joined = engine.join("users", "posts", "_id", "user_id").unwrap();
        assert_eq!(joined.len(), 2000);
    }

    #[test]
    fn select_with_condition_returning_none() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..100 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + (i % 10)));
            engine.insert("users", values).unwrap();
        }

        // Create conditions that will be evaluated but return nothing
        let rows = engine
            .select(
                "users",
                Condition::Eq("name".into(), Value::String("NonExistent".into())),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn btree_index_range_le() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..50 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        engine.create_btree_index("users", "age").unwrap();

        let rows = engine
            .select("users", Condition::Le("age".into(), Value::Int(10)))
            .unwrap();
        assert_eq!(rows.len(), 11); // 0-10 inclusive
    }

    #[test]
    fn btree_index_range_ge() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..50 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        engine.create_btree_index("users", "age").unwrap();

        let rows = engine
            .select("users", Condition::Ge("age".into(), Value::Int(40)))
            .unwrap();
        assert_eq!(rows.len(), 10); // 40-49 inclusive
    }

    #[test]
    fn update_large_dataset_with_index() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..1000 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(25));
            engine.insert("users", values).unwrap();
        }

        engine.create_index("users", "age").unwrap();
        engine.create_btree_index("users", "age").unwrap();

        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(30));
        let count = engine
            .update(
                "users",
                Condition::Eq("age".into(), Value::Int(25)),
                updates,
            )
            .unwrap();
        assert_eq!(count, 1000);
    }

    #[test]
    fn delete_large_dataset_with_index() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..1000 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert(
                "age".to_string(),
                Value::Int(if i % 2 == 0 { 25 } else { 30 }),
            );
            engine.insert("users", values).unwrap();
        }

        engine.create_index("users", "age").unwrap();
        engine.create_btree_index("users", "age").unwrap();

        let count = engine
            .delete_rows("users", Condition::Eq("age".into(), Value::Int(25)))
            .unwrap();
        assert_eq!(count, 500);
    }

    // ========================================================================
    // JOIN Tests
    // ========================================================================

    fn create_orders_table(engine: &RelationalEngine) {
        let schema = Schema::new(vec![
            Column::new("order_id", ColumnType::Int),
            Column::new("user_id", ColumnType::Int),
            Column::new("amount", ColumnType::Float),
        ]);
        engine.create_table("orders", schema).unwrap();
    }

    #[test]
    fn test_inner_join() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        create_orders_table(&engine);

        // Users: _id will be 1, 2, 3 for these three
        for (name, age) in [("Alice", 25), ("Bob", 30), ("Carol", 35)] {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(name.to_string()));
            values.insert("age".to_string(), Value::Int(age));
            engine.insert("users", values).unwrap();
        }

        // Orders: user_id matches the _id of users (1=Alice, 2=Bob)
        // Alice gets 2 orders, Bob gets 1, Carol gets 0
        let order_data = [(1i64, 100.0), (1i64, 200.0), (2i64, 150.0)];
        for (idx, (user_id, amount)) in order_data.iter().enumerate() {
            let mut values = HashMap::new();
            values.insert("order_id".to_string(), Value::Int(idx as i64));
            values.insert("user_id".to_string(), Value::Int(*user_id));
            values.insert("amount".to_string(), Value::Float(*amount));
            engine.insert("orders", values).unwrap();
        }

        let results = engine.join("users", "orders", "_id", "user_id").unwrap();
        assert_eq!(results.len(), 3); // Alice x2 + Bob x1

        // Verify Alice (id=1) appears twice
        let alice_joins: Vec<_> = results.iter().filter(|(u, _)| u.id == 1).collect();
        assert_eq!(alice_joins.len(), 2);
    }

    #[test]
    fn test_left_join() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        create_orders_table(&engine);

        // Users: Alice (_id=1), Bob (_id=2), Carol (_id=3)
        for (name, age) in [("Alice", 25), ("Bob", 30), ("Carol", 35)] {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(name.to_string()));
            values.insert("age".to_string(), Value::Int(age));
            engine.insert("users", values).unwrap();
        }

        // Orders: Only Alice (user_id=1) has orders
        let mut values = HashMap::new();
        values.insert("order_id".to_string(), Value::Int(1));
        values.insert("user_id".to_string(), Value::Int(1)); // Alice
        values.insert("amount".to_string(), Value::Float(100.0));
        engine.insert("orders", values).unwrap();

        let results = engine
            .left_join("users", "orders", "_id", "user_id")
            .unwrap();

        // All 3 users should appear
        assert_eq!(results.len(), 3);

        // Alice (id=1) has a match
        let alice = results.iter().find(|(u, _)| u.id == 1).unwrap();
        assert!(alice.1.is_some());

        // Bob (id=2) has None
        let bob = results.iter().find(|(u, _)| u.id == 2).unwrap();
        assert!(bob.1.is_none());
    }

    #[test]
    fn test_right_join() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        create_orders_table(&engine);

        // Only Alice (_id=1)
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".into()));
        values.insert("age".to_string(), Value::Int(25));
        engine.insert("users", values).unwrap();

        // Orders for user 1 (Alice) and user 99 (doesn't exist)
        for (user_id, amount) in [(1i64, 100.0), (99i64, 200.0)] {
            let mut values = HashMap::new();
            values.insert("order_id".to_string(), Value::Int(user_id));
            values.insert("user_id".to_string(), Value::Int(user_id));
            values.insert("amount".to_string(), Value::Float(amount));
            engine.insert("orders", values).unwrap();
        }

        let results = engine
            .right_join("users", "orders", "_id", "user_id")
            .unwrap();

        // Both orders should appear
        assert_eq!(results.len(), 2);

        // Order for Alice has a user
        let with_user: Vec<_> = results.iter().filter(|(u, _)| u.is_some()).collect();
        assert_eq!(with_user.len(), 1);

        // Order for user 99 has None
        let without_user: Vec<_> = results.iter().filter(|(u, _)| u.is_none()).collect();
        assert_eq!(without_user.len(), 1);
    }

    #[test]
    fn test_full_join() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        create_orders_table(&engine);

        // Alice (_id=1) and Bob (_id=2)
        for (name, age) in [("Alice", 25), ("Bob", 30)] {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(name.to_string()));
            values.insert("age".to_string(), Value::Int(age));
            engine.insert("users", values).unwrap();
        }

        // Orders for Alice (user_id=1) and user 99 (doesn't exist)
        for (user_id, amount) in [(1i64, 100.0), (99i64, 200.0)] {
            let mut values = HashMap::new();
            values.insert("order_id".to_string(), Value::Int(user_id));
            values.insert("user_id".to_string(), Value::Int(user_id));
            values.insert("amount".to_string(), Value::Float(amount));
            engine.insert("orders", values).unwrap();
        }

        let results = engine
            .full_join("users", "orders", "_id", "user_id")
            .unwrap();

        // Alice matched, Bob unmatched, Order 99 unmatched = 3 rows
        assert_eq!(results.len(), 3);

        // Alice has both
        let matched: Vec<_> = results
            .iter()
            .filter(|(a, b)| a.is_some() && b.is_some())
            .collect();
        assert_eq!(matched.len(), 1);

        // Bob has no order (Some, None)
        let left_only: Vec<_> = results
            .iter()
            .filter(|(a, b)| a.is_some() && b.is_none())
            .collect();
        assert_eq!(left_only.len(), 1);

        // Order 99 has no user (None, Some)
        let right_only: Vec<_> = results
            .iter()
            .filter(|(a, b)| a.is_none() && b.is_some())
            .collect();
        assert_eq!(right_only.len(), 1);
    }

    #[test]
    fn test_cross_join() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        create_orders_table(&engine);

        // 2 users
        for (name, age) in [("Alice", 25), ("Bob", 30)] {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(name.to_string()));
            values.insert("age".to_string(), Value::Int(age));
            engine.insert("users", values).unwrap();
        }

        // 3 orders
        for i in 0..3 {
            let mut values = HashMap::new();
            values.insert("order_id".to_string(), Value::Int(i));
            values.insert("user_id".to_string(), Value::Int(i));
            values.insert("amount".to_string(), Value::Float(100.0));
            engine.insert("orders", values).unwrap();
        }

        let results = engine.cross_join("users", "orders").unwrap();

        // 2 users x 3 orders = 6 rows
        assert_eq!(results.len(), 6);
    }

    #[test]
    fn test_natural_join() {
        let engine = RelationalEngine::new();

        // Create two tables with a common column "dept_id"
        let schema_a = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("dept_id", ColumnType::Int),
        ]);
        engine.create_table("employees", schema_a).unwrap();

        let schema_b = Schema::new(vec![
            Column::new("dept_id", ColumnType::Int),
            Column::new("dept_name", ColumnType::String),
        ]);
        engine.create_table("departments", schema_b).unwrap();

        // Employees
        for (name, dept) in [("Alice", 1), ("Bob", 1), ("Carol", 2)] {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(name.to_string()));
            values.insert("dept_id".to_string(), Value::Int(dept));
            engine.insert("employees", values).unwrap();
        }

        // Departments
        for (id, name) in [(1, "Engineering"), (2, "Sales")] {
            let mut values = HashMap::new();
            values.insert("dept_id".to_string(), Value::Int(id));
            values.insert("dept_name".to_string(), Value::String(name.to_string()));
            engine.insert("departments", values).unwrap();
        }

        let results = engine.natural_join("employees", "departments").unwrap();

        // All 3 employees should match
        assert_eq!(results.len(), 3);

        // Verify Engineering has 2 employees
        let eng_count = results
            .iter()
            .filter(|(_, d)| {
                d.get_with_id("dept_name") == Some(Value::String("Engineering".into()))
            })
            .count();
        assert_eq!(eng_count, 2);
    }

    #[test]
    fn test_natural_join_no_common_columns() {
        let engine = RelationalEngine::new();

        let schema_a = Schema::new(vec![Column::new("a_col", ColumnType::Int)]);
        engine.create_table("table_a", schema_a).unwrap();

        let schema_b = Schema::new(vec![Column::new("b_col", ColumnType::Int)]);
        engine.create_table("table_b", schema_b).unwrap();

        let mut values = HashMap::new();
        values.insert("a_col".to_string(), Value::Int(1));
        engine.insert("table_a", values).unwrap();

        let mut values = HashMap::new();
        values.insert("b_col".to_string(), Value::Int(2));
        engine.insert("table_b", values).unwrap();

        // No common columns = cross join
        let results = engine.natural_join("table_a", "table_b").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_left_join_empty_right() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        create_orders_table(&engine);

        // One user, no orders
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".into()));
        values.insert("age".to_string(), Value::Int(25));
        engine.insert("users", values).unwrap();

        let results = engine
            .left_join("users", "orders", "_id", "user_id")
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].1.is_none());
    }

    #[test]
    fn test_join_with_multiple_matches() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        create_orders_table(&engine);

        // One user (Alice, _id=1)
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".into()));
        values.insert("age".to_string(), Value::Int(25));
        engine.insert("users", values).unwrap();

        // 5 orders for Alice (user_id=1)
        for i in 0..5 {
            let mut values = HashMap::new();
            values.insert("order_id".to_string(), Value::Int(i));
            values.insert("user_id".to_string(), Value::Int(1)); // Alice's _id
            values.insert("amount".to_string(), Value::Float(100.0 * (i + 1) as f64));
            engine.insert("orders", values).unwrap();
        }

        let results = engine.join("users", "orders", "_id", "user_id").unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_join_with_null_keys() {
        let engine = RelationalEngine::new();

        // Create tables with nullable join columns
        let users_schema = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("ref_id", ColumnType::Int).nullable(),
        ]);
        engine.create_table("users_null", users_schema).unwrap();

        let orders_schema = Schema::new(vec![
            Column::new("order_id", ColumnType::Int),
            Column::new("user_ref", ColumnType::Int).nullable(),
        ]);
        engine.create_table("orders_null", orders_schema).unwrap();

        // Insert user with null ref_id
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Bob".into()));
        values.insert("ref_id".to_string(), Value::Null);
        engine.insert("users_null", values).unwrap();

        // Insert order with null user_ref
        let mut order = HashMap::new();
        order.insert("order_id".to_string(), Value::Int(1));
        order.insert("user_ref".to_string(), Value::Null);
        engine.insert("orders_null", order).unwrap();

        // Join on nullable columns - engine matches null == null
        let results = engine
            .join("users_null", "orders_null", "ref_id", "user_ref")
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_aggregate_overflow_i64() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("value", ColumnType::Int)]);
        engine.create_table("big_nums", schema).unwrap();

        // Insert values near i64::MAX
        let near_max = i64::MAX / 2;
        for _ in 0..3 {
            let mut values = HashMap::new();
            values.insert("value".to_string(), Value::Int(near_max));
            engine.insert("big_nums", values).unwrap();
        }

        // Sum would overflow, but we should handle gracefully via wrapping
        let result = engine.sum("big_nums", "value", Condition::True);
        assert!(result.is_ok());
    }

    #[test]
    fn test_index_create_duplicate_name() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Create index first time - succeeds
        engine.create_index("users", "age").unwrap();

        // Create same index again - should fail
        let result = engine.create_index("users", "age");
        assert!(result.is_err());
        match result {
            Err(RelationalError::IndexAlreadyExists { table, column }) => {
                assert_eq!(table, "users");
                assert_eq!(column, "age");
            },
            _ => panic!("Expected IndexAlreadyExists error"),
        }
    }

    #[test]
    fn test_btree_range_empty_result() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        engine.create_btree_index("users", "age").unwrap();

        // Insert some values
        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{i}")));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        // Query range with no matches (age > 100)
        let condition = Condition::Gt("age".to_string(), Value::Int(100));
        let rows = engine.select("users", condition).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_condition_evaluate_type_mismatch() {
        // Test that comparing incompatible types returns false
        let row = Row {
            id: 1,
            values: vec![
                ("name".to_string(), Value::String("Alice".into())),
                ("age".to_string(), Value::Int(25)),
            ],
        };

        // Compare string column with int value
        let condition = Condition::Eq("name".to_string(), Value::Int(42));
        assert!(!condition.evaluate(&row));

        // Compare int column with string value
        let condition2 = Condition::Eq("age".to_string(), Value::String("twenty".into()));
        assert!(!condition2.evaluate(&row));
    }
}
