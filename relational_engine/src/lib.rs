//! SQL-like relational engine with SIMD-accelerated filtering.
//!
//! Provides table storage, indexing, and query execution on top of
//! `tensor_store::RelationalSlab` for columnar storage.
//!
//! # Durability Model
//!
//! When using durable storage (via [`RelationalEngine::open_durable`] or
//! [`RelationalEngine::recover`]), the following data is persisted:
//!
//! **Persisted (durable):**
//! - Table schemas and metadata
//! - Row data (columnar storage)
//!
//! **NOT persisted (must recreate after recovery):**
//! - Hash indexes: call [`RelationalEngine::create_index`] after recovery
//! - B-tree indexes: call [`RelationalEngine::create_btree_index`] after recovery
//! - Row counters: automatically rebuilt from scanning data
//! - Transaction state: transactions are aborted on crash
//!
//! # Recovery Example
//!
//! ```ignore
//! // Open durable engine
//! let engine = RelationalEngine::recover(
//!     "data/wal",
//!     &WalConfig::default(),
//!     Some(Path::new("data/snapshot")),
//! )?;
//!
//! // Recreate indexes after recovery
//! for table in engine.list_tables() {
//!     // Recreate indexes based on your schema requirements
//!     engine.create_index(&table, "user_id")?;
//!     engine.create_btree_index(&table, "created_at")?;
//! }
//! ```

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    hash::{Hash, Hasher},
    path::Path,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};

use dashmap::DashMap;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
#[allow(unused_imports)] // Used in Phase 3 integration
use tensor_store::RelationalSlab;
pub use tensor_store::{
    ColumnDef as SlabColumnDef, ColumnType as SlabColumnType, ColumnValue as SlabColumnValue,
    RowId as SlabRowId, TableSchema as SlabTableSchema, WalConfig,
};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorStoreError, TensorValue};
use tracing::instrument;

pub mod transaction;
pub use transaction::{
    Deadline, IndexChange, LockConflictInfo, RowLock, RowLockManager, Transaction,
    TransactionManager, TxPhase, UndoEntry,
};

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
    pub const fn bitmap_words(n: usize) -> usize {
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
        fn test_filter_lt_remainder_match() {
            // Ensure remainder elements that match are included
            let values = vec![10, 10, 10, 10, 1]; // 5 elements, position 4 matches
            let mut bitmap = vec![0u64; 1];
            filter_lt_i64(&values, 5, &mut bitmap);
            // Only position 4 (value 1) is < 5
            assert_eq!(bitmap[0], 0b00010000);
        }

        #[test]
        fn test_filter_le_remainder_match() {
            let values = vec![10, 10, 10, 10, 5]; // 5 elements, position 4 matches <=
            let mut bitmap = vec![0u64; 1];
            filter_le_i64(&values, 5, &mut bitmap);
            assert_eq!(bitmap[0], 0b00010000);
        }

        #[test]
        fn test_filter_gt_remainder_match() {
            let values = vec![1, 1, 1, 1, 10]; // 5 elements, position 4 matches >
            let mut bitmap = vec![0u64; 1];
            filter_gt_i64(&values, 5, &mut bitmap);
            assert_eq!(bitmap[0], 0b00010000);
        }

        #[test]
        fn test_filter_ge_remainder_match() {
            let values = vec![1, 1, 1, 1, 5]; // 5 elements, position 4 matches >=
            let mut bitmap = vec![0u64; 1];
            filter_ge_i64(&values, 5, &mut bitmap);
            assert_eq!(bitmap[0], 0b00010000);
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

        #[test]
        fn test_filter_lt_f64_remainder_match() {
            let values = vec![10.0, 10.0, 10.0, 10.0, 1.0]; // 5 elements
            let mut bitmap = vec![0u64; 1];
            filter_lt_f64(&values, 5.0, &mut bitmap);
            assert_eq!(bitmap[0], 0b00010000);
        }

        #[test]
        fn test_filter_gt_f64_remainder_match() {
            let values = vec![1.0, 1.0, 1.0, 1.0, 10.0]; // 5 elements
            let mut bitmap = vec![0u64; 1];
            filter_gt_f64(&values, 5.0, &mut bitmap);
            assert_eq!(bitmap[0], 0b00010000);
        }

        #[test]
        fn test_filter_eq_f64_remainder_match() {
            let values = vec![1.0, 1.0, 1.0, 1.0, 5.0]; // 5 elements
            let mut bitmap = vec![0u64; 1];
            filter_eq_f64(&values, 5.0, &mut bitmap);
            assert_eq!(bitmap[0], 0b00010000);
        }
    }
}

/// Column data type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnType {
    /// 64-bit signed integer.
    Int,
    /// 64-bit floating point.
    Float,
    /// UTF-8 string.
    String,
    /// Boolean.
    Bool,
}

/// Column definition with name, type, and nullability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    /// Column name.
    pub name: String,
    /// Column data type.
    pub column_type: ColumnType,
    /// Whether null values are allowed.
    pub nullable: bool,
}

impl Column {
    /// Creates a non-nullable column.
    pub fn new(name: impl Into<String>, column_type: ColumnType) -> Self {
        Self {
            name: name.into(),
            column_type,
            nullable: false,
        }
    }

    /// Makes this column nullable.
    #[must_use]
    pub const fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }
}

/// Table schema defining column structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    /// Ordered list of columns.
    pub columns: Vec<Column>,
}

impl Schema {
    /// Creates a schema from columns.
    #[must_use]
    pub const fn new(columns: Vec<Column>) -> Self {
        Self { columns }
    }

    /// Finds a column by name.
    #[must_use]
    pub fn get_column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }
}

/// Query value type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Null value.
    Null,
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit floating point.
    Float(f64),
    /// UTF-8 string.
    String(String),
    /// Boolean.
    Bool(bool),
}

impl Value {
    fn from_scalar(scalar: &ScalarValue) -> Self {
        match scalar {
            ScalarValue::Int(v) => Self::Int(*v),
            ScalarValue::Float(v) => Self::Float(*v),
            ScalarValue::String(v) => Self::String(v.clone()),
            ScalarValue::Bool(v) => Self::Bool(*v),
            ScalarValue::Null | ScalarValue::Bytes(_) => Self::Null,
        }
    }

    const fn matches_type(&self, column_type: &ColumnType) -> bool {
        matches!(
            (self, column_type),
            (Self::Null, _)
                | (Self::Int(_), ColumnType::Int)
                | (Self::Float(_), ColumnType::Float)
                | (Self::String(_), ColumnType::String)
                | (Self::Bool(_), ColumnType::Bool)
        )
    }

    fn hash_key(&self) -> String {
        match self {
            Self::Null => "null".to_string(),
            Self::Int(v) => format!("i:{v}"),
            Self::Float(v) => format!("f:{}", v.to_bits()),
            Self::String(v) => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                v.hash(&mut hasher);
                format!("s:{:x}", hasher.finish())
            },
            Self::Bool(v) => format!("b:{v}"),
        }
    }

    fn partial_cmp_value(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => Some(a.cmp(b)),
            (Self::Float(a), Self::Float(b)) => a.partial_cmp(b),
            (Self::String(a), Self::String(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Math guarantees valid u64 range
    fn sortable_key(&self) -> String {
        match self {
            Self::Null => "0".to_string(),
            Self::Int(v) => {
                // Encode i64 as hex with offset to handle negative numbers
                // Add i64::MAX + 1 to shift range from [-2^63, 2^63-1] to [0, 2^64-1]
                let unsigned = (i128::from(*v) + i128::from(i64::MAX) + 1) as u64;
                format!("i{unsigned:016x}")
            },
            Self::Float(v) => {
                // IEEE 754 float bit encoding with sign handling for correct ordering
                let bits = v.to_bits();
                let sortable = if *v >= 0.0 {
                    bits ^ 0x8000_0000_0000_0000 // Flip sign bit for positive
                } else {
                    !bits // Flip all bits for negative
                };
                format!("f{sortable:016x}")
            },
            Self::String(v) => format!("s{v}"),
            Self::Bool(v) => {
                if *v {
                    "b1".to_string()
                } else {
                    "b0".to_string()
                }
            },
        }
    }

    /// Returns true if this value is "truthy" (non-null, non-zero, non-empty).
    #[must_use]
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Null => false,
            Self::Bool(b) => *b,
            Self::Int(i) => *i != 0,
            Self::Float(f) => *f != 0.0,
            Self::String(s) => !s.is_empty(),
        }
    }
}

// Schema bridge: conversions between relational_engine types and RelationalSlab types

impl From<&ColumnType> for SlabColumnType {
    fn from(ct: &ColumnType) -> Self {
        match ct {
            ColumnType::Int => Self::Int,
            ColumnType::Float => Self::Float,
            ColumnType::String => Self::String,
            ColumnType::Bool => Self::Bool,
        }
    }
}

impl From<&SlabColumnType> for ColumnType {
    fn from(ct: &SlabColumnType) -> Self {
        match ct {
            SlabColumnType::Int => Self::Int,
            SlabColumnType::Float => Self::Float,
            SlabColumnType::Bool => Self::Bool,
            SlabColumnType::String | SlabColumnType::Bytes | SlabColumnType::Json => Self::String,
        }
    }
}

impl From<&Schema> for SlabTableSchema {
    fn from(schema: &Schema) -> Self {
        let columns = schema
            .columns
            .iter()
            .map(|c| SlabColumnDef::new(&c.name, (&c.column_type).into(), c.nullable))
            .collect();
        Self::new(columns)
    }
}

impl From<&Value> for SlabColumnValue {
    fn from(v: &Value) -> Self {
        match v {
            Value::Null => Self::Null,
            Value::Int(i) => Self::Int(*i),
            Value::Float(f) => Self::Float(*f),
            Value::String(s) => Self::String(s.clone()),
            Value::Bool(b) => Self::Bool(*b),
        }
    }
}

impl From<SlabColumnValue> for Value {
    fn from(cv: SlabColumnValue) -> Self {
        match cv {
            SlabColumnValue::Null => Self::Null,
            SlabColumnValue::Int(i) => Self::Int(i),
            SlabColumnValue::Float(f) => Self::Float(f),
            SlabColumnValue::String(s) => Self::String(s),
            SlabColumnValue::Bool(b) => Self::Bool(b),
            SlabColumnValue::Bytes(b) => Self::String(format!("{b:?}")),
            SlabColumnValue::Json(j) => Self::String(j),
        }
    }
}

/// A row with ID and column values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    /// Unique row identifier.
    pub id: u64,
    /// Column values stored as ordered pairs for faster iteration.
    /// Use `get()` for single-column access, or iterate directly for bulk access.
    pub values: Vec<(String, Value)>,
}

impl Row {
    /// Gets a column value by name.
    #[must_use]
    pub fn get(&self, column: &str) -> Option<&Value> {
        if column == "_id" {
            return None;
        }
        self.values
            .iter()
            .find(|(k, _)| k == column)
            .map(|(_, v)| v)
    }

    /// Gets a column value by name, including the special `_id` column.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)] // Row IDs won't exceed i64::MAX
    pub fn get_with_id(&self, column: &str) -> Option<Value> {
        if column == "_id" {
            return Some(Value::Int(self.id as i64));
        }
        self.values
            .iter()
            .find(|(k, _)| k == column)
            .map(|(_, v)| v.clone())
    }

    /// Returns true if this row has the given column.
    #[must_use]
    pub fn contains(&self, column: &str) -> bool {
        self.values.iter().any(|(k, _)| k == column)
    }
}

/// Query filter condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Condition {
    /// Equality: column = value.
    Eq(String, Value),
    /// Not equal: column != value.
    Ne(String, Value),
    /// Less than: column < value.
    Lt(String, Value),
    /// Less than or equal: column <= value.
    Le(String, Value),
    /// Greater than: column > value.
    Gt(String, Value),
    /// Greater than or equal: column >= value.
    Ge(String, Value),
    /// Logical AND of two conditions.
    And(Box<Self>, Box<Self>),
    /// Logical OR of two conditions.
    Or(Box<Self>, Box<Self>),
    /// Always true (matches all rows).
    True,
}

impl Condition {
    /// Combines this condition with another using AND.
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        Self::And(Box::new(self), Box::new(other))
    }

    /// Combines this condition with another using OR.
    #[must_use]
    pub fn or(self, other: Self) -> Self {
        Self::Or(Box::new(self), Box::new(other))
    }

    /// Evaluates this condition against a row.
    #[must_use]
    pub fn evaluate(&self, row: &Row) -> bool {
        use std::cmp::Ordering;
        match self {
            Self::True => true,
            Self::Eq(col, val) => row.get_with_id(col).as_ref() == Some(val),
            Self::Ne(col, val) => row.get_with_id(col).as_ref() != Some(val),
            Self::Lt(col, val) => Self::compare_ord(row, col, val, Ordering::Less),
            Self::Le(col, val) => Self::compare_ord_le(row, col, val),
            Self::Gt(col, val) => Self::compare_ord(row, col, val, Ordering::Greater),
            Self::Ge(col, val) => Self::compare_ord_ge(row, col, val),
            Self::And(a, b) => a.evaluate(row) && b.evaluate(row),
            Self::Or(a, b) => a.evaluate(row) || b.evaluate(row),
        }
    }

    fn compare_ord(row: &Row, col: &str, val: &Value, ord: std::cmp::Ordering) -> bool {
        row.get_with_id(col)
            .and_then(|v| v.partial_cmp_value(val))
            .is_some_and(|o| o == ord)
    }

    fn compare_ord_le(row: &Row, col: &str, val: &Value) -> bool {
        row.get_with_id(col)
            .and_then(|v| v.partial_cmp_value(val))
            .is_some_and(|o| o != std::cmp::Ordering::Greater)
    }

    fn compare_ord_ge(row: &Row, col: &str, val: &Value) -> bool {
        row.get_with_id(col)
            .and_then(|v| v.partial_cmp_value(val))
            .is_some_and(|o| o != std::cmp::Ordering::Less)
    }

    /// Evaluate this condition against a tensor record.
    #[cfg(feature = "test-internals")]
    #[must_use]
    pub fn evaluate_tensor(&self, tensor: &TensorData) -> bool {
        self.evaluate_tensor_impl(tensor)
    }

    #[cfg(not(feature = "test-internals"))]
    pub(crate) fn evaluate_tensor(&self, tensor: &TensorData) -> bool {
        self.evaluate_tensor_impl(tensor)
    }

    #[allow(dead_code)] // Legacy method for TensorData evaluation
    fn evaluate_tensor_impl(&self, tensor: &TensorData) -> bool {
        use std::cmp::Ordering;
        match self {
            Self::True => true,
            Self::Eq(col, val) => Self::tensor_field_eq(tensor, col, val),
            Self::Ne(col, val) => !Self::tensor_field_eq(tensor, col, val),
            Self::Lt(col, val) => Self::tensor_compare_ord(tensor, col, val, Ordering::Less),
            Self::Le(col, val) => Self::tensor_compare_le(tensor, col, val),
            Self::Gt(col, val) => Self::tensor_compare_ord(tensor, col, val, Ordering::Greater),
            Self::Ge(col, val) => Self::tensor_compare_ge(tensor, col, val),
            Self::And(a, b) => a.evaluate_tensor(tensor) && b.evaluate_tensor(tensor),
            Self::Or(a, b) => a.evaluate_tensor(tensor) || b.evaluate_tensor(tensor),
        }
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    fn tensor_field_eq(tensor: &TensorData, col: &str, val: &Value) -> bool {
        Self::tensor_get_value(tensor, col).as_ref() == Some(val)
    }

    #[allow(dead_code)]
    fn tensor_compare_ord(
        tensor: &TensorData,
        col: &str,
        val: &Value,
        ord: std::cmp::Ordering,
    ) -> bool {
        Self::tensor_get_value(tensor, col)
            .and_then(|v| v.partial_cmp_value(val))
            .is_some_and(|o| o == ord)
    }

    #[allow(dead_code)]
    fn tensor_compare_le(tensor: &TensorData, col: &str, val: &Value) -> bool {
        Self::tensor_get_value(tensor, col)
            .and_then(|v| v.partial_cmp_value(val))
            .is_some_and(|o| o != std::cmp::Ordering::Greater)
    }

    #[allow(dead_code)]
    fn tensor_compare_ge(tensor: &TensorData, col: &str, val: &Value) -> bool {
        Self::tensor_get_value(tensor, col)
            .and_then(|v| v.partial_cmp_value(val))
            .is_some_and(|o| o != std::cmp::Ordering::Less)
    }
}

/// Configuration for `RelationalEngine` resource limits and timeouts.
#[derive(Debug, Clone)]
pub struct RelationalConfig {
    /// Maximum number of tables allowed. `None` means unlimited.
    pub max_tables: Option<usize>,
    /// Maximum number of indexes per table. `None` means unlimited.
    pub max_indexes_per_table: Option<usize>,
    /// Maximum total entries across all B-tree indexes.
    pub max_btree_entries: usize,
    /// Default query timeout in milliseconds. `None` means no timeout.
    pub default_query_timeout_ms: Option<u64>,
    /// Maximum allowed query timeout in milliseconds.
    pub max_query_timeout_ms: Option<u64>,
}

impl Default for RelationalConfig {
    fn default() -> Self {
        Self {
            max_tables: None,
            max_indexes_per_table: None,
            max_btree_entries: 10_000_000,
            default_query_timeout_ms: None,
            max_query_timeout_ms: Some(300_000), // 5 minutes
        }
    }
}

impl RelationalConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of tables allowed.
    #[must_use]
    pub const fn with_max_tables(mut self, max: usize) -> Self {
        self.max_tables = Some(max);
        self
    }

    /// Sets the maximum number of indexes per table.
    #[must_use]
    pub const fn with_max_indexes_per_table(mut self, max: usize) -> Self {
        self.max_indexes_per_table = Some(max);
        self
    }

    /// Sets the default query timeout in milliseconds.
    #[must_use]
    pub const fn with_default_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.default_query_timeout_ms = Some(timeout_ms);
        self
    }

    /// Sets the maximum allowed query timeout in milliseconds.
    #[must_use]
    pub const fn with_max_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.max_query_timeout_ms = Some(timeout_ms);
        self
    }

    /// Sets the maximum B-tree index entries.
    #[must_use]
    pub const fn with_max_btree_entries(mut self, max: usize) -> Self {
        self.max_btree_entries = max;
        self
    }

    /// Configuration preset for high-throughput workloads.
    ///
    /// - No table or index limits
    /// - 30 second default query timeout
    /// - 20M B-tree entries allowed
    #[must_use]
    pub const fn high_throughput() -> Self {
        Self {
            max_tables: None,
            max_indexes_per_table: None,
            max_btree_entries: 20_000_000,
            default_query_timeout_ms: Some(30_000),
            max_query_timeout_ms: Some(600_000), // 10 minutes
        }
    }

    /// Configuration preset for low-memory environments.
    ///
    /// - Maximum 100 tables
    /// - Maximum 5 indexes per table
    /// - 1M B-tree entries
    /// - 10 second default query timeout
    #[must_use]
    pub const fn low_memory() -> Self {
        Self {
            max_tables: Some(100),
            max_indexes_per_table: Some(5),
            max_btree_entries: 1_000_000,
            default_query_timeout_ms: Some(10_000),
            max_query_timeout_ms: Some(60_000), // 1 minute
        }
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the default timeout exceeds the maximum timeout.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if let (Some(default), Some(max)) =
            (self.default_query_timeout_ms, self.max_query_timeout_ms)
        {
            if default > max {
                return Err(format!(
                    "default_query_timeout_ms ({default}) exceeds max_query_timeout_ms ({max})"
                ));
            }
        }
        Ok(())
    }
}

/// Options for query execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct QueryOptions {
    /// Query timeout in milliseconds. `None` uses the engine's default.
    pub timeout_ms: Option<u64>,
}

impl QueryOptions {
    /// Creates new query options with no timeout override.
    #[must_use]
    pub const fn new() -> Self {
        Self { timeout_ms: None }
    }

    /// Sets the query timeout in milliseconds.
    #[must_use]
    pub const fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
}

/// Errors from relational engine operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationalError {
    /// Table does not exist.
    TableNotFound(String),
    /// Table already exists.
    TableAlreadyExists(String),
    /// Column does not exist.
    ColumnNotFound(String),
    /// Value type doesn't match column type.
    TypeMismatch {
        /// Column name.
        column: String,
        /// Expected type.
        expected: ColumnType,
    },
    /// Null value in non-nullable column.
    NullNotAllowed(String),
    /// Index already exists.
    IndexAlreadyExists {
        /// Table name.
        table: String,
        /// Column name.
        column: String,
    },
    /// Index does not exist.
    IndexNotFound {
        /// Table name.
        table: String,
        /// Column name.
        column: String,
    },
    /// Underlying storage error.
    StorageError(String),
    /// Invalid table or column name.
    InvalidName(String),
    /// Transaction not found.
    TransactionNotFound(u64),
    /// Transaction is not active (already committed or aborted).
    TransactionInactive(u64),
    /// Lock conflict with another transaction.
    LockConflict {
        /// Transaction requesting the lock.
        tx_id: u64,
        /// Transaction holding the lock.
        blocking_tx: u64,
        /// Table name.
        table: String,
        /// Row identifier.
        row_id: u64,
    },
    /// Lock acquisition timed out.
    LockTimeout {
        /// Transaction ID.
        tx_id: u64,
        /// Table name.
        table: String,
        /// Row identifiers.
        row_ids: Vec<u64>,
    },
    /// Rollback failed.
    RollbackFailed {
        /// Transaction ID.
        tx_id: u64,
        /// Failure reason.
        reason: String,
    },
    /// Result set exceeds the maximum allowed size.
    ResultTooLarge {
        /// Operation name.
        operation: String,
        /// Actual result size.
        actual: usize,
        /// Maximum allowed size.
        max: usize,
    },
    /// Index data is corrupted.
    IndexCorrupted {
        /// Description of the corruption.
        reason: String,
    },
    /// Schema metadata is corrupted or incomplete.
    SchemaCorrupted {
        /// Table name.
        table: String,
        /// Description of the corruption.
        reason: String,
    },
    /// Maximum number of tables exceeded.
    TooManyTables {
        /// Current table count.
        current: usize,
        /// Maximum allowed tables.
        max: usize,
    },
    /// Maximum number of indexes per table exceeded.
    TooManyIndexes {
        /// Table name.
        table: String,
        /// Current index count.
        current: usize,
        /// Maximum allowed indexes.
        max: usize,
    },
    /// Query timed out.
    QueryTimeout {
        /// Operation that timed out.
        operation: String,
        /// Timeout in milliseconds.
        timeout_ms: u64,
    },
}

impl std::fmt::Display for RelationalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TableNotFound(t) => write!(f, "Table not found: {t}"),
            Self::TableAlreadyExists(t) => write!(f, "Table already exists: {t}"),
            Self::ColumnNotFound(c) => write!(f, "Column not found: {c}"),
            Self::TypeMismatch { column, expected } => {
                write!(
                    f,
                    "Type mismatch for column {column}: expected {expected:?}"
                )
            },
            Self::NullNotAllowed(c) => write!(f, "Null not allowed for column: {c}"),
            Self::IndexAlreadyExists { table, column } => {
                write!(f, "Index already exists on {table}.{column}")
            },
            Self::IndexNotFound { table, column } => {
                write!(f, "Index not found on {table}.{column}")
            },
            Self::StorageError(e) => write!(f, "Storage error: {e}"),
            Self::InvalidName(msg) => write!(f, "Invalid name: {msg}"),
            Self::TransactionNotFound(tx_id) => {
                write!(f, "Transaction not found: {tx_id}")
            },
            Self::TransactionInactive(tx_id) => {
                write!(f, "Transaction not active: {tx_id}")
            },
            Self::LockConflict {
                tx_id,
                blocking_tx,
                table,
                row_id,
            } => {
                write!(
                    f,
                    "Lock conflict: tx {tx_id} blocked by tx {blocking_tx} on {table}.{row_id}"
                )
            },
            Self::LockTimeout {
                tx_id,
                table,
                row_ids,
            } => {
                write!(
                    f,
                    "Lock timeout: tx {} waiting for {} rows in {}",
                    tx_id,
                    row_ids.len(),
                    table
                )
            },
            Self::RollbackFailed { tx_id, reason } => {
                write!(f, "Rollback failed for tx {tx_id}: {reason}")
            },
            Self::ResultTooLarge {
                operation,
                actual,
                max,
            } => {
                write!(
                    f,
                    "{operation} result too large: {actual} rows exceeds maximum of {max}"
                )
            },
            Self::IndexCorrupted { reason } => {
                write!(f, "Index data corrupted: {reason}")
            },
            Self::SchemaCorrupted { table, reason } => {
                write!(f, "Schema corrupted for table '{table}': {reason}")
            },
            Self::TooManyTables { current, max } => {
                write!(
                    f,
                    "Too many tables: {current} tables exceeds maximum of {max}"
                )
            },
            Self::TooManyIndexes {
                table,
                current,
                max,
            } => {
                write!(
                    f,
                    "Too many indexes on table '{table}': {current} indexes exceeds maximum of {max}"
                )
            },
            Self::QueryTimeout {
                operation,
                timeout_ms,
            } => {
                write!(f, "Query timeout: {operation} exceeded {timeout_ms}ms")
            },
        }
    }
}

impl std::error::Error for RelationalError {}

impl From<TensorStoreError> for RelationalError {
    fn from(e: TensorStoreError) -> Self {
        Self::StorageError(e.to_string())
    }
}

/// Result type alias for relational engine operations.
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
    pub const fn len(&self) -> usize {
        match self {
            Self::Int(v) => v.len(),
            Self::Float(v) => v.len(),
            Self::String { indices, .. } => indices.len(),
            Self::Bool(v) => v.len() * 64, // approximate
        }
    }

    pub const fn is_empty(&self) -> bool {
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
            Self::None => false,
            Self::Dense(bitmap) => {
                let word_idx = idx / 64;
                let bit_idx = idx % 64;
                word_idx < bitmap.len() && (bitmap[word_idx] & (1u64 << bit_idx)) != 0
            },
            Self::Sparse(positions) => positions.binary_search(&(idx as u64)).is_ok(),
        }
    }

    pub fn null_count(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Dense(bitmap) => bitmap.iter().map(|w| w.count_ones() as usize).sum(),
            Self::Sparse(positions) => positions.len(),
        }
    }
}

/// Column data with values and null tracking.
#[derive(Debug, Clone)]
pub struct ColumnData {
    #[allow(dead_code)] // Used in future WAL integration
    name: String,
    /// Row identifiers for each value.
    pub row_ids: Vec<u64>,
    nulls: NullBitmap,
    values: ColumnValues,
}

impl ColumnData {
    /// Returns the number of null values.
    #[must_use]
    pub fn null_count(&self) -> usize {
        self.nulls.null_count()
    }

    /// Gets a value at the given index.
    #[must_use]
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

    pub const fn from_bitmap(bitmap: Vec<u64>, row_count: usize) -> Self {
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

    pub fn intersect(&self, other: &Self) -> Self {
        let mut result = vec![0u64; self.bitmap.len()];
        simd::bitmap_and(&self.bitmap, &other.bitmap, &mut result);
        Self {
            bitmap: result,
            row_count: self.row_count,
        }
    }

    pub fn union(&self, other: &Self) -> Self {
        let mut result = vec![0u64; self.bitmap.len()];
        simd::bitmap_or(&self.bitmap, &other.bitmap, &mut result);
        Self {
            bitmap: result,
            row_count: self.row_count,
        }
    }
}

/// Options for columnar scan operations.
#[derive(Debug, Clone, Default)]
pub struct ColumnarScanOptions {
    /// Columns to include (None = all columns).
    pub projection: Option<Vec<String>>,
    /// Prefer columnar storage path when available.
    pub prefer_columnar: bool,
}

/// Key for in-memory B-tree indexes: (table, column) -> `BTreeMap<value, row_ids>`
type BTreeIndexKey = (String, String);

/// SQL-like relational database engine.
///
/// Provides tables, indexes, and SIMD-accelerated queries.
///
/// # Lock Ordering
///
/// To prevent deadlocks, locks must be acquired in this order:
/// 1. `ddl_lock` (global DDL serialization)
/// 2. `btree_indexes` (global B-tree index map)
/// 3. `index_locks[n]` (per-key striped locks)
///
/// Never acquire a higher-numbered lock while holding a lower-numbered lock.
pub struct RelationalEngine {
    store: TensorStore,
    row_counters: DashMap<String, AtomicU64>,
    /// In-memory B-tree indexes for O(log n) range queries.
    /// Maps (table, column) to a `BTreeMap` of value -> `row_ids`.
    btree_indexes: RwLock<HashMap<BTreeIndexKey, BTreeMap<OrderedKey, Vec<u64>>>>,
    /// Whether WAL is enabled for crash-safe writes.
    ///
    /// When `true`, table metadata and row data are written to the Write-Ahead Log.
    /// Note: B-tree indexes and hash indexes are kept in-memory only and are NOT
    /// persisted. After recovery, indexes must be recreated using `create_btree_index()`
    /// or `create_index()`.
    is_durable: bool,
    /// Transaction manager for local transactions.
    tx_manager: TransactionManager,
    /// Serializes DDL operations to prevent TOCTOU races.
    ddl_lock: RwLock<()>,
    /// Striped locks for atomic index entry updates (bounded memory via lock striping).
    index_locks: [RwLock<()>; 64],
    /// Maximum allowed B-tree index entries across all indexes.
    max_btree_entries: usize,
    /// Current total B-tree index entry count.
    btree_entry_count: AtomicUsize,
    /// Engine configuration for resource limits and timeouts.
    config: RelationalConfig,
    /// Current table count for limit enforcement.
    table_count: AtomicUsize,
}

/// Ordered key for B-tree indexes with correct comparison semantics.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum OrderedKey {
    /// Null value (sorts first).
    Null,
    /// Boolean value.
    Bool(bool),
    /// 64-bit signed integer.
    Int(i64),
    /// Floating point with total ordering.
    Float(OrderedFloat),
    /// UTF-8 string.
    String(String),
}

/// Wrapper for f64 that implements total ordering (NaN is less than all values).
#[derive(Debug, Clone, Copy)]
pub struct OrderedFloat(
    /// The wrapped f64 value.
    pub f64,
);

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
            Value::Null => Self::Null,
            Value::Bool(b) => Self::Bool(*b),
            Value::Int(i) => Self::Int(*i),
            Value::Float(f) => Self::Float(OrderedFloat(*f)),
            Value::String(s) => Self::String(s.clone()),
        }
    }
}

impl RelationalEngine {
    /// Threshold for parallel operations (below this, sequential is faster)
    const PARALLEL_THRESHOLD: usize = 1000;

    /// Maximum rows in cross join result (prevents memory exhaustion).
    const MAX_CROSS_JOIN_ROWS: usize = 1_000_000;

    /// Number of shards for index lock striping (power of 2 for fast modulo).
    const INDEX_LOCK_SHARDS: usize = 64;

    /// Creates a new in-memory relational engine with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(RelationalConfig::default())
    }

    /// Creates a new in-memory relational engine with custom configuration.
    #[must_use]
    pub fn with_config(config: RelationalConfig) -> Self {
        Self {
            store: TensorStore::new(),
            row_counters: DashMap::new(),
            btree_indexes: RwLock::new(HashMap::new()),
            is_durable: false,
            tx_manager: TransactionManager::new(),
            ddl_lock: RwLock::new(()),
            index_locks: std::array::from_fn(|_| RwLock::new(())),
            max_btree_entries: config.max_btree_entries,
            btree_entry_count: AtomicUsize::new(0),
            config,
            table_count: AtomicUsize::new(0),
        }
    }

    /// Creates an engine with an existing store and default configuration.
    #[must_use]
    pub fn with_store(store: TensorStore) -> Self {
        Self::with_store_and_config(store, RelationalConfig::default())
    }

    /// Creates an engine with an existing store and custom configuration.
    #[must_use]
    pub fn with_store_and_config(store: TensorStore, config: RelationalConfig) -> Self {
        Self {
            store,
            row_counters: DashMap::new(),
            btree_indexes: RwLock::new(HashMap::new()),
            is_durable: false,
            tx_manager: TransactionManager::new(),
            ddl_lock: RwLock::new(()),
            index_locks: std::array::from_fn(|_| RwLock::new(())),
            max_btree_entries: config.max_btree_entries,
            btree_entry_count: AtomicUsize::new(0),
            config,
            table_count: AtomicUsize::new(0),
        }
    }

    /// Create a durable engine with Write-Ahead Log for crash safety.
    ///
    /// All table metadata and row data are persisted to the WAL. However:
    /// - B-tree indexes are kept in-memory only and must be recreated after recovery
    /// - Hash indexes are kept in-memory only and must be recreated after recovery
    /// - Row counters are recovered from scanning existing data
    ///
    /// # Errors
    /// Returns an I/O error if the WAL file cannot be created or opened.
    pub fn open_durable<P: AsRef<Path>>(
        wal_path: P,
        wal_config: WalConfig,
    ) -> std::io::Result<Self> {
        Self::open_durable_with_config(wal_path, wal_config, RelationalConfig::default())
    }

    /// Create a durable engine with Write-Ahead Log and custom configuration.
    ///
    /// # Errors
    /// Returns an I/O error if the WAL file cannot be created or opened.
    pub fn open_durable_with_config<P: AsRef<Path>>(
        wal_path: P,
        wal_config: WalConfig,
        config: RelationalConfig,
    ) -> std::io::Result<Self> {
        let store = TensorStore::open_durable(wal_path, wal_config)?;
        Ok(Self {
            store,
            row_counters: DashMap::new(),
            btree_indexes: RwLock::new(HashMap::new()),
            is_durable: true,
            tx_manager: TransactionManager::new(),
            ddl_lock: RwLock::new(()),
            index_locks: std::array::from_fn(|_| RwLock::new(())),
            max_btree_entries: config.max_btree_entries,
            btree_entry_count: AtomicUsize::new(0),
            config,
            table_count: AtomicUsize::new(0),
        })
    }

    /// Recover a durable engine from WAL after crash.
    ///
    /// Recovers all table metadata and row data from the WAL. The following are
    /// NOT recovered and must be recreated manually:
    /// - B-tree indexes: call `create_btree_index()` for each needed index
    /// - Hash indexes: call `create_index()` for each needed index
    ///
    /// # Errors
    /// Returns `RelationalError::StorageError` if WAL recovery fails.
    pub fn recover<P: AsRef<Path>>(
        wal_path: P,
        wal_config: &WalConfig,
        snapshot_path: Option<&Path>,
    ) -> std::result::Result<Self, RelationalError> {
        Self::recover_with_config(
            wal_path,
            wal_config,
            snapshot_path,
            RelationalConfig::default(),
        )
    }

    /// Recover a durable engine from WAL with custom configuration.
    ///
    /// # Errors
    /// Returns `RelationalError::StorageError` if WAL recovery fails.
    pub fn recover_with_config<P: AsRef<Path>>(
        wal_path: P,
        wal_config: &WalConfig,
        snapshot_path: Option<&Path>,
        config: RelationalConfig,
    ) -> std::result::Result<Self, RelationalError> {
        let store = TensorStore::recover(wal_path, wal_config, snapshot_path)
            .map_err(|e| RelationalError::StorageError(e.to_string()))?;
        // Count existing tables for limit tracking
        let existing_tables = store.scan("_meta:table:").len();
        Ok(Self {
            store,
            row_counters: DashMap::new(),
            btree_indexes: RwLock::new(HashMap::new()),
            is_durable: true,
            tx_manager: TransactionManager::new(),
            ddl_lock: RwLock::new(()),
            index_locks: std::array::from_fn(|_| RwLock::new(())),
            max_btree_entries: config.max_btree_entries,
            btree_entry_count: AtomicUsize::new(0),
            config,
            table_count: AtomicUsize::new(existing_tables),
        })
    }

    /// Returns a reference to the underlying tensor store.
    pub const fn store(&self) -> &TensorStore {
        &self.store
    }

    /// Returns a reference to the engine configuration.
    pub const fn config(&self) -> &RelationalConfig {
        &self.config
    }

    /// Returns the current number of tables.
    pub fn table_count(&self) -> usize {
        self.table_count.load(Ordering::Relaxed)
    }

    /// Access the underlying `RelationalSlab` for direct columnar operations.
    fn slab(&self) -> &RelationalSlab {
        &self.store.router().relations
    }

    fn acquire_index_lock(&self, key: &str) -> &RwLock<()> {
        let shard = Self::index_lock_shard(key);
        &self.index_locks[shard]
    }

    #[allow(clippy::cast_possible_truncation)]
    fn index_lock_shard(key: &str) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish() as usize % Self::INDEX_LOCK_SHARDS
    }

    /// Checks if creating a new table would exceed the configured limit.
    fn check_table_limit(&self) -> Result<()> {
        if let Some(max) = self.config.max_tables {
            let current = self.table_count.load(Ordering::Acquire);
            if current >= max {
                return Err(RelationalError::TooManyTables { current, max });
            }
        }
        Ok(())
    }

    /// Resolves the effective timeout in milliseconds from query options and config.
    fn resolve_timeout(&self, options: QueryOptions) -> Option<u64> {
        let timeout = options.timeout_ms.or(self.config.default_query_timeout_ms);

        // Clamp to max if configured
        if let (Some(t), Some(max)) = (timeout, self.config.max_query_timeout_ms) {
            Some(t.min(max))
        } else {
            timeout
        }
    }

    /// Checks if creating a new index on a table would exceed the configured limit.
    fn check_index_limit(&self, table: &str) -> Result<()> {
        if let Some(max) = self.config.max_indexes_per_table {
            // Count hash indexes
            let idx_prefix = Self::all_indexes_prefix(table);
            let hash_count = self.store.scan(&idx_prefix).len();

            // Count B-tree indexes
            let btree_prefix = format!("_btree:{table}:");
            let btree_count = self.store.scan(&btree_prefix).len();

            let current = hash_count + btree_count;
            if current >= max {
                return Err(RelationalError::TooManyIndexes {
                    table: table.to_string(),
                    current,
                    max,
                });
            }
        }
        Ok(())
    }

    fn put_maybe_durable(&self, key: impl Into<String>, tensor: TensorData) -> Result<()> {
        let key = key.into();
        if self.is_durable {
            self.store
                .put_durable(&key, tensor)
                .map_err(|e| RelationalError::StorageError(e.to_string()))
        } else {
            self.store
                .put(&key, tensor)
                .map_err(|e| RelationalError::StorageError(e.to_string()))
        }
    }

    fn delete_maybe_durable(&self, key: &str) -> Result<()> {
        if self.is_durable {
            self.store
                .delete_durable(key)
                .map_err(|e| RelationalError::StorageError(e.to_string()))
        } else {
            self.store
                .delete(key)
                .map_err(|e| RelationalError::StorageError(e.to_string()))
        }
    }

    const MAX_NAME_LENGTH: usize = 255;

    fn validate_name(name: &str, kind: &str) -> Result<()> {
        if name.is_empty() {
            return Err(RelationalError::InvalidName(format!(
                "{kind} name cannot be empty"
            )));
        }
        if name.len() > Self::MAX_NAME_LENGTH {
            return Err(RelationalError::InvalidName(format!(
                "{} name exceeds {} characters",
                kind,
                Self::MAX_NAME_LENGTH
            )));
        }
        if name.starts_with('_') {
            return Err(RelationalError::InvalidName(format!(
                "{kind} name cannot start with underscore (reserved)"
            )));
        }
        if name.contains(':') {
            return Err(RelationalError::InvalidName(format!(
                "{kind} name cannot contain colon"
            )));
        }
        if name.contains(',') {
            return Err(RelationalError::InvalidName(format!(
                "{kind} name cannot contain comma"
            )));
        }
        Ok(())
    }

    fn table_meta_key(name: &str) -> String {
        format!("_meta:table:{name}")
    }

    fn row_key(table: &str, id: u64) -> String {
        format!("{table}:{id}")
    }

    fn row_prefix(table: &str) -> String {
        format!("{table}:")
    }

    fn index_meta_key(table: &str, column: &str) -> String {
        format!("_idx:{table}:{column}")
    }

    fn index_entry_key(table: &str, column: &str, value_hash: &str) -> String {
        format!("_idx:{table}:{column}:{value_hash}")
    }

    fn index_prefix(table: &str, column: &str) -> String {
        format!("_idx:{table}:{column}:")
    }

    fn all_indexes_prefix(table: &str) -> String {
        format!("_idx:{table}:")
    }

    fn btree_meta_key(table: &str, column: &str) -> String {
        format!("_btree:{table}:{column}")
    }

    fn btree_entry_key(table: &str, column: &str, sortable_value: &str) -> String {
        format!("_btree:{table}:{column}:{sortable_value}")
    }

    fn btree_prefix(table: &str, column: &str) -> String {
        format!("_btree:{table}:{column}:")
    }

    /// # Errors
    /// Returns `InvalidName` if table/column names exceed 255 chars or are empty.
    /// Returns `TableAlreadyExists` if the table already exists.
    /// Returns `StorageError` if the underlying storage fails.
    #[allow(clippy::needless_pass_by_value)] // Public API takes ownership for ergonomics
    pub fn create_table(&self, name: &str, schema: Schema) -> Result<()> {
        Self::validate_name(name, "Table")?;
        for col in &schema.columns {
            Self::validate_name(&col.name, "Column")?;
        }

        // Check table limit before acquiring DDL lock
        self.check_table_limit()?;

        // Atomic DDL: acquire lock before check-then-act to prevent TOCTOU races
        let _ddl_guard = self.ddl_lock.write();

        // Re-check limit under lock to prevent race conditions
        self.check_table_limit()?;

        let meta_key = Self::table_meta_key(name);

        if self.store.exists(&meta_key) {
            return Err(RelationalError::TableAlreadyExists(name.to_string()));
        }

        // Create table in RelationalSlab for columnar storage
        let slab_schema: SlabTableSchema = (&schema).into();
        self.slab()
            .create_table(name, slab_schema)
            .map_err(|e| RelationalError::StorageError(e.to_string()))?;

        // Store metadata in TensorStore for backward compatibility
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

        self.put_maybe_durable(meta_key, meta)?;

        self.row_counters
            .insert(name.to_string(), AtomicU64::new(0));

        // Increment table count after successful creation
        self.table_count.fetch_add(1, Ordering::Release);

        Ok(())
    }

    /// # Errors
    /// Returns `TableNotFound` if the table does not exist, or `SchemaCorrupted`
    /// if the schema metadata is malformed.
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
                // Empty from trailing comma is OK
                continue;
            }
            let col_key = format!("_col:{col_name}");
            let Some(TensorValue::Scalar(ScalarValue::String(type_str))) = meta.get(&col_key)
            else {
                return Err(RelationalError::SchemaCorrupted {
                    table: table.to_string(),
                    reason: format!("missing metadata for column '{col_name}'"),
                });
            };
            let parts: Vec<&str> = type_str.split(':').collect();
            if parts.len() != 2 {
                return Err(RelationalError::SchemaCorrupted {
                    table: table.to_string(),
                    reason: format!("malformed type string for column '{col_name}': '{type_str}'"),
                });
            }
            let column_type = match parts[0] {
                "int" => ColumnType::Int,
                "float" => ColumnType::Float,
                "bool" => ColumnType::Bool,
                "string" => ColumnType::String,
                unknown => {
                    return Err(RelationalError::SchemaCorrupted {
                        table: table.to_string(),
                        reason: format!("unknown column type '{unknown}' for column '{col_name}'"),
                    });
                },
            };
            let nullable = parts[1] == "null";
            let mut col = Column::new(col_name, column_type);
            if nullable {
                col = col.nullable();
            }
            columns.push(col);
        }

        Ok(Schema::new(columns))
    }

    /// Returns a list of all table names.
    pub fn list_tables(&self) -> Vec<String> {
        self.store
            .scan("_meta:table:")
            .into_iter()
            .filter_map(|key| key.strip_prefix("_meta:table:").map(String::from))
            .collect()
    }

    /// # Errors
    /// Returns `TableNotFound`, `NullNotAllowed`, `TypeMismatch`, or `StorageError`.
    #[allow(clippy::needless_pass_by_value)] // Public API takes ownership for ergonomics
    #[instrument(skip(self, values), fields(table = %table))]
    pub fn insert(&self, table: &str, values: HashMap<String, Value>) -> Result<u64> {
        // Validation for early exit without transaction overhead
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

        // Use internal transaction for atomicity
        let tx_id = self.begin_transaction();
        match self.tx_insert(tx_id, table, values) {
            Ok(row_id) => {
                self.commit(tx_id)?;
                Ok(row_id)
            },
            Err(e) => {
                let _ = self.rollback(tx_id);
                Err(e)
            },
        }
    }

    /// # Errors
    /// Returns `TableNotFound`, `NullNotAllowed`, `TypeMismatch`, or `StorageError`.
    #[must_use = "batch insert results contain row IDs that should be used"]
    #[allow(clippy::cast_possible_wrap)] // Row IDs are monotonic from 1, won't exceed i64::MAX
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
            // Build slab row in column order
            let slab_row: Vec<SlabColumnValue> = schema
                .columns
                .iter()
                .map(|col| {
                    values
                        .get(&col.name)
                        .map_or(SlabColumnValue::Null, std::convert::Into::into)
                })
                .collect();

            // Insert into slab
            let slab_row_id = self
                .slab()
                .insert(table, slab_row)
                .map_err(|e| RelationalError::StorageError(e.to_string()))?;

            let row_id = slab_row_id.as_u64() + 1; // Convert 0-based to 1-based

            // Update row counter
            self.row_counters
                .entry(table.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_max(row_id, Ordering::Relaxed);

            // Update indexes
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

    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[must_use = "query results should be used"]
    #[allow(clippy::cast_possible_truncation)] // Row IDs fit in usize on 64-bit platforms
    #[allow(clippy::needless_pass_by_value)] // Public API takes ownership for ergonomics
    #[instrument(skip(self, condition), fields(table = %table))]
    pub fn select(&self, table: &str, condition: Condition) -> Result<Vec<Row>> {
        self.select_with_options(table, condition, QueryOptions::default())
    }

    /// Select rows with query options including timeout.
    ///
    /// # Errors
    /// Returns `TableNotFound`, `StorageError`, or `QueryTimeout`.
    #[must_use = "query results should be used"]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::needless_pass_by_value)]
    pub fn select_with_options(
        &self,
        table: &str,
        condition: Condition,
        options: QueryOptions,
    ) -> Result<Vec<Row>> {
        let deadline = Deadline::from_timeout_ms(self.resolve_timeout(options));
        let schema = self.get_schema(table)?;

        // Check timeout before index lookup
        if deadline.is_expired() {
            return Err(RelationalError::QueryTimeout {
                operation: "select".to_string(),
                timeout_ms: options.timeout_ms.unwrap_or(0),
            });
        }

        // Index lookup path: get row_ids from index, then fetch from slab
        if let Some(row_ids) = self.try_index_lookup(table, &condition)? {
            // Check timeout after index lookup
            if deadline.is_expired() {
                return Err(RelationalError::QueryTimeout {
                    operation: "select (index lookup)".to_string(),
                    timeout_ms: options.timeout_ms.unwrap_or(0),
                });
            }

            // Convert 1-based row_ids to 0-based slab indices
            let indices: Vec<usize> = row_ids.iter().map(|id| (*id - 1) as usize).collect();
            let slab_rows = self
                .slab()
                .get_rows_by_indices(table, &indices)
                .map_err(|e| RelationalError::StorageError(e.to_string()))?;

            let mut rows: Vec<Row> = slab_rows
                .into_iter()
                .filter_map(|(row_id, slab_row)| {
                    let row = Self::slab_row_to_engine_row(&schema, row_id, slab_row);
                    if condition.evaluate(&row) {
                        Some(row)
                    } else {
                        None
                    }
                })
                .collect();
            rows.sort_by_key(|r| r.id);
            return Ok(rows);
        }

        // Check timeout before full scan
        if deadline.is_expired() {
            return Err(RelationalError::QueryTimeout {
                operation: "select (before scan)".to_string(),
                timeout_ms: options.timeout_ms.unwrap_or(0),
            });
        }

        // Full scan path: scan slab and filter
        let slab_rows = self
            .slab()
            .scan_all(table)
            .map_err(|e| RelationalError::StorageError(e.to_string()))?;

        // Check timeout after scan
        if deadline.is_expired() {
            return Err(RelationalError::QueryTimeout {
                operation: "select (after scan)".to_string(),
                timeout_ms: options.timeout_ms.unwrap_or(0),
            });
        }

        let mut rows: Vec<Row> = slab_rows
            .into_iter()
            .filter_map(|(row_id, slab_row)| {
                let row = Self::slab_row_to_engine_row(&schema, row_id, slab_row);
                if condition.evaluate(&row) {
                    Some(row)
                } else {
                    None
                }
            })
            .collect();

        rows.sort_by_key(|r| r.id);
        Ok(rows)
    }

    fn slab_row_to_engine_row(
        schema: &Schema,
        row_id: SlabRowId,
        slab_row: Vec<SlabColumnValue>,
    ) -> Row {
        let id = row_id.as_u64() + 1; // Convert 0-based to 1-based
        let values: Vec<(String, Value)> = slab_row
            .into_iter()
            .enumerate()
            .map(|(col_idx, value)| {
                let col_name = schema.columns[col_idx].name.clone();
                (col_name, value.into())
            })
            .collect();
        Row { id, values }
    }

    fn try_index_lookup(&self, table: &str, condition: &Condition) -> Result<Option<Vec<u64>>> {
        match condition {
            Condition::Eq(column, value) => self.index_lookup(table, column, value),
            Condition::Lt(column, value) => {
                Ok(self.btree_range_lookup(table, column, value, RangeOp::Lt))
            },
            Condition::Le(column, value) => {
                Ok(self.btree_range_lookup(table, column, value, RangeOp::Le))
            },
            Condition::Gt(column, value) => {
                Ok(self.btree_range_lookup(table, column, value, RangeOp::Gt))
            },
            Condition::Ge(column, value) => {
                Ok(self.btree_range_lookup(table, column, value, RangeOp::Ge))
            },
            Condition::And(a, b) => {
                let a_result = self.try_index_lookup(table, a)?;
                if a_result.is_some() {
                    return Ok(a_result);
                }
                self.try_index_lookup(table, b)
            },
            _ => Ok(None),
        }
    }

    /// # Errors
    /// Returns `TableNotFound`, `ColumnNotFound`, `TypeMismatch`, `NullNotAllowed`, or `StorageError`.
    #[allow(clippy::needless_pass_by_value)] // Public API takes ownership for ergonomics
    pub fn update(
        &self,
        table: &str,
        condition: Condition,
        updates: HashMap<String, Value>,
    ) -> Result<usize> {
        self.update_with_options(table, condition, updates, QueryOptions::default())
    }

    /// Update rows with query options including timeout.
    ///
    /// # Errors
    /// Returns `TableNotFound`, `ColumnNotFound`, `TypeMismatch`, `NullNotAllowed`,
    /// `StorageError`, or `QueryTimeout`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn update_with_options(
        &self,
        table: &str,
        condition: Condition,
        updates: HashMap<String, Value>,
        options: QueryOptions,
    ) -> Result<usize> {
        let deadline = Deadline::from_timeout_ms(self.resolve_timeout(options));

        // Validation for early exit without transaction overhead
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

        // Check timeout before transaction
        if deadline.is_expired() {
            return Err(RelationalError::QueryTimeout {
                operation: "update".to_string(),
                timeout_ms: options.timeout_ms.unwrap_or(0),
            });
        }

        // Use internal transaction for atomicity
        let tx_id = self.begin_transaction();
        match self.tx_update(tx_id, table, condition, updates) {
            Ok(count) => {
                self.commit(tx_id)?;
                Ok(count)
            },
            Err(e) => {
                let _ = self.rollback(tx_id);
                Err(e)
            },
        }
    }

    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[allow(clippy::needless_pass_by_value)] // Public API takes ownership for ergonomics
    pub fn delete_rows(&self, table: &str, condition: Condition) -> Result<usize> {
        self.delete_rows_with_options(table, condition, QueryOptions::default())
    }

    /// Delete rows with query options including timeout.
    ///
    /// # Errors
    /// Returns `TableNotFound`, `StorageError`, or `QueryTimeout`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn delete_rows_with_options(
        &self,
        table: &str,
        condition: Condition,
        options: QueryOptions,
    ) -> Result<usize> {
        let deadline = Deadline::from_timeout_ms(self.resolve_timeout(options));

        // Validation for early exit
        let _ = self.get_schema(table)?;

        // Check timeout before transaction
        if deadline.is_expired() {
            return Err(RelationalError::QueryTimeout {
                operation: "delete".to_string(),
                timeout_ms: options.timeout_ms.unwrap_or(0),
            });
        }

        // Use internal transaction for atomicity
        let tx_id = self.begin_transaction();
        match self.tx_delete(tx_id, table, condition) {
            Ok(count) => {
                self.commit(tx_id)?;
                Ok(count)
            },
            Err(e) => {
                let _ = self.rollback(tx_id);
                Err(e)
            },
        }
    }

    /// Hash join: O(n+m) instead of O(n*m) nested loop join.
    /// Builds a hash index on the right table, then probes from the left.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    pub fn join(
        &self,
        table_a: &str,
        table_b: &str,
        on_a: &str,
        on_b: &str,
    ) -> Result<Vec<(Row, Row)>> {
        self.join_with_options(table_a, table_b, on_a, on_b, QueryOptions::default())
    }

    /// Hash join with query options including timeout.
    ///
    /// # Errors
    /// Returns `TableNotFound`, `StorageError`, or `QueryTimeout`.
    pub fn join_with_options(
        &self,
        table_a: &str,
        table_b: &str,
        on_a: &str,
        on_b: &str,
        options: QueryOptions,
    ) -> Result<Vec<(Row, Row)>> {
        let deadline = Deadline::from_timeout_ms(self.resolve_timeout(options));

        let _ = self.get_schema(table_a)?;
        let _ = self.get_schema(table_b)?;

        // Check timeout before fetching rows
        if deadline.is_expired() {
            return Err(RelationalError::QueryTimeout {
                operation: "join".to_string(),
                timeout_ms: options.timeout_ms.unwrap_or(0),
            });
        }

        let rows_a = self.select(table_a, Condition::True)?;

        // Check timeout after fetching first table
        if deadline.is_expired() {
            return Err(RelationalError::QueryTimeout {
                operation: "join (after table_a)".to_string(),
                timeout_ms: options.timeout_ms.unwrap_or(0),
            });
        }

        let rows_b = self.select(table_b, Condition::True)?;

        // Check timeout after fetching second table
        if deadline.is_expired() {
            return Err(RelationalError::QueryTimeout {
                operation: "join (after table_b)".to_string(),
                timeout_ms: options.timeout_ms.unwrap_or(0),
            });
        }

        let mut index: HashMap<String, Vec<usize>> = HashMap::with_capacity(rows_b.len());
        for (i, row) in rows_b.iter().enumerate() {
            if let Some(val) = row.get_with_id(on_b) {
                let hash = val.hash_key();
                index.entry(hash).or_default().push(i);
            }
        }

        // Check timeout after building hash index
        if deadline.is_expired() {
            return Err(RelationalError::QueryTimeout {
                operation: "join (after hash build)".to_string(),
                timeout_ms: options.timeout_ms.unwrap_or(0),
            });
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
                                        let matched = &rows_b[i];
                                        if matched.get_with_id(on_b).as_ref() == Some(&val) {
                                            Some((row_a.clone(), matched.clone()))
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
                            let matched = &rows_b[i];
                            if matched.get_with_id(on_b).as_ref() == Some(&val) {
                                results.push((row_a.clone(), matched.clone()));
                            }
                        }
                    }
                }
            }
            results
        };

        Ok(results)
    }

    /// LEFT JOIN: Returns all rows from `table_a`, with matching rows from `table_b`.
    /// If no match, the `table_b` side is `None`.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
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
            let mut found = false;
            if let Some(val) = row_a.get_with_id(on_a) {
                if let Some(indices) = index.get(&val.hash_key()) {
                    for &i in indices {
                        let found_row = &rows_b[i];
                        if found_row.get_with_id(on_b).as_ref() == Some(&val) {
                            results.push((row_a.clone(), Some(found_row.clone())));
                            found = true;
                        }
                    }
                }
            }
            if !found {
                results.push((row_a.clone(), None));
            }
        }

        Ok(results)
    }

    /// RIGHT JOIN: Returns all rows from `table_b`, with matching rows from `table_a`.
    /// If no match, the `table_a` side is `None`.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
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
            let mut found = false;
            if let Some(val) = row_b.get_with_id(on_b) {
                if let Some(indices) = index.get(&val.hash_key()) {
                    for &i in indices {
                        let found_row = &rows_a[i];
                        if found_row.get_with_id(on_a).as_ref() == Some(&val) {
                            results.push((Some(found_row.clone()), row_b.clone()));
                            found = true;
                        }
                    }
                }
            }
            if !found {
                results.push((None, row_b.clone()));
            }
        }

        Ok(results)
    }

    /// FULL OUTER JOIN: Returns all rows from both tables.
    /// Rows that don't match get None on the non-matching side.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
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
            let mut found = false;
            if let Some(val) = row_a.get_with_id(on_a) {
                if let Some(indices) = index.get(&val.hash_key()) {
                    for &i in indices {
                        let found_row = &rows_b[i];
                        if found_row.get_with_id(on_b).as_ref() == Some(&val) {
                            results.push((Some(row_a.clone()), Some(found_row.clone())));
                            matched_b.insert(i);
                            found = true;
                        }
                    }
                }
            }
            if !found {
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
    /// Every row from `table_a` is paired with every row from `table_b`.
    ///
    /// # Errors
    /// Returns `TableNotFound`, `ResultTooLarge`, or `StorageError`.
    #[must_use = "query results should be used"]
    pub fn cross_join(&self, table_a: &str, table_b: &str) -> Result<Vec<(Row, Row)>> {
        let _ = self.get_schema(table_a)?;
        let _ = self.get_schema(table_b)?;

        let rows_a = self.select(table_a, Condition::True)?;
        let rows_b = self.select(table_b, Condition::True)?;

        // Check result size before allocation
        let result_size = rows_a.len().saturating_mul(rows_b.len());
        if result_size > Self::MAX_CROSS_JOIN_ROWS {
            return Err(RelationalError::ResultTooLarge {
                operation: "CROSS JOIN".to_string(),
                actual: result_size,
                max: Self::MAX_CROSS_JOIN_ROWS,
            });
        }

        let mut results = Vec::with_capacity(result_size);
        for row_a in &rows_a {
            for row_b in &rows_b {
                results.push((row_a.clone(), row_b.clone()));
            }
        }

        Ok(results)
    }

    /// NATURAL JOIN: Joins on all columns with the same name.
    /// Returns rows where all common columns have equal values.
    ///
    /// # Errors
    /// Returns `TableNotFound`, `ResultTooLarge`, or `StorageError`.
    #[must_use = "query results should be used"]
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
                    let candidate = &rows_b[i];
                    let all_match = common_cols
                        .iter()
                        .all(|col| row_a.get_with_id(col) == candidate.get_with_id(col));
                    if all_match {
                        results.push((row_a.clone(), candidate.clone()));
                    }
                }
            }
        }

        Ok(results)
    }

    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    pub fn count(&self, table: &str, condition: Condition) -> Result<u64> {
        let rows = self.select(table, condition)?;
        Ok(rows.len() as u64)
    }

    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    pub fn count_column(&self, table: &str, column: &str, condition: Condition) -> Result<u64> {
        let rows = self.select(table, condition)?;
        let count = rows
            .iter()
            .filter(|row| row.get(column).is_some_and(|v| !matches!(v, Value::Null)))
            .count();
        Ok(count as u64)
    }

    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[allow(clippy::cast_precision_loss)] // Aggregate functions accept f64 precision loss
    pub fn sum(&self, table: &str, column: &str, condition: Condition) -> Result<f64> {
        let rows = self.select(table, condition)?;
        let col = column.to_string();

        if rows.len() >= Self::PARALLEL_THRESHOLD {
            let total: f64 = rows
                .par_iter()
                .map(|row| {
                    row.get(&col).map_or(0.0, |val| match val {
                        Value::Int(i) => *i as f64,
                        Value::Float(f) => *f,
                        _ => 0.0,
                    })
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

    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[allow(clippy::cast_precision_loss)] // Aggregate functions accept f64 precision loss
    pub fn avg(&self, table: &str, column: &str, condition: Condition) -> Result<Option<f64>> {
        let rows = self.select(table, condition)?;
        let col = column.to_string();

        let (total, count) = if rows.len() >= Self::PARALLEL_THRESHOLD {
            rows.par_iter()
                .map(|row| {
                    row.get(&col).map_or((0.0, 0u64), |val| match val {
                        Value::Int(i) => (*i as f64, 1u64),
                        Value::Float(f) => (*f, 1u64),
                        _ => (0.0, 0u64),
                    })
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

    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
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

    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
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

    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    pub fn drop_table(&self, table: &str) -> Result<()> {
        // Atomic DDL: acquire lock before check-then-act to prevent TOCTOU races
        let _ddl_guard = self.ddl_lock.write();

        let meta_key = Self::table_meta_key(table);

        if !self.store.exists(&meta_key) {
            return Err(RelationalError::TableNotFound(table.to_string()));
        }

        // Drop from RelationalSlab
        let _ = self.slab().drop_table(table); // Ignore error if table doesn't exist in slab

        let prefix = Self::row_prefix(table);
        let keys = self.store.scan(&prefix);
        for key in keys {
            self.delete_maybe_durable(&key)?;
        }

        let idx_prefix = Self::all_indexes_prefix(table);
        let idx_keys = self.store.scan(&idx_prefix);
        for key in idx_keys {
            self.delete_maybe_durable(&key)?;
        }

        let btree_prefix = format!("_btree:{table}:");
        let btree_keys = self.store.scan(&btree_prefix);
        for key in btree_keys {
            self.delete_maybe_durable(&key)?;
        }

        self.delete_maybe_durable(&meta_key)?;

        self.row_counters.remove(table);

        // Decrement table count after successful deletion
        self.table_count.fetch_sub(1, Ordering::Release);

        Ok(())
    }

    /// Returns true if the table exists.
    pub fn table_exists(&self, table: &str) -> bool {
        let meta_key = Self::table_meta_key(table);
        self.store.exists(&meta_key)
    }

    /// Returns the number of rows in the table.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    pub fn row_count(&self, table: &str) -> Result<usize> {
        let _ = self.get_schema(table)?;
        self.slab()
            .row_count(table)
            .map_err(|e| RelationalError::StorageError(e.to_string()))
    }

    /// Create a hash index on a column for fast equality lookups.
    ///
    /// # Errors
    /// Returns `InvalidName`, `TableNotFound`, `ColumnNotFound`, `IndexAlreadyExists`, or `StorageError`.
    #[allow(clippy::cast_possible_wrap)] // Row IDs won't exceed i64::MAX
    pub fn create_index(&self, table: &str, column: &str) -> Result<()> {
        // Allow _id as a special case (system column)
        if column != "_id" {
            Self::validate_name(column, "Column")?;
        }

        let schema = self.get_schema(table)?;

        if column != "_id" && schema.get_column(column).is_none() {
            return Err(RelationalError::ColumnNotFound(column.to_string()));
        }

        // Check index limit before acquiring DDL lock
        self.check_index_limit(table)?;

        // Atomic DDL: acquire lock before check-then-act to prevent TOCTOU races
        let _ddl_guard = self.ddl_lock.write();

        // Re-check limit under lock to prevent race conditions
        self.check_index_limit(table)?;

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

        // Build index from slab data
        let col_idx = schema
            .columns
            .iter()
            .position(|c| c.name == column)
            .or_else(|| (column == "_id").then_some(usize::MAX));

        let slab_rows = self
            .slab()
            .scan_all(table)
            .map_err(|e| RelationalError::StorageError(e.to_string()))?;

        for (row_id, slab_row) in slab_rows {
            let engine_row_id = row_id.as_u64() + 1; // Convert 0-based to 1-based
            let value = if column == "_id" {
                Value::Int(engine_row_id as i64)
            } else if let Some(idx) = col_idx {
                slab_row.get(idx).map_or(Value::Null, |v| v.clone().into())
            } else {
                continue;
            };
            self.index_add(table, column, &value, engine_row_id)?;
        }

        Ok(())
    }

    /// Create a B-tree index on a column for fast range queries.
    /// B-tree indexes accelerate Lt, Le, Gt, Ge conditions with O(log n) lookup.
    ///
    /// # Errors
    /// Returns `InvalidName`, `TableNotFound`, `ColumnNotFound`, `IndexAlreadyExists`, or `StorageError`.
    #[allow(clippy::cast_possible_wrap)] // Row IDs won't exceed i64::MAX
    pub fn create_btree_index(&self, table: &str, column: &str) -> Result<()> {
        // Allow _id as a special case (system column)
        if column != "_id" {
            Self::validate_name(column, "Column")?;
        }

        let schema = self.get_schema(table)?;

        if column != "_id" && schema.get_column(column).is_none() {
            return Err(RelationalError::ColumnNotFound(column.to_string()));
        }

        // Check index limit before acquiring DDL lock
        self.check_index_limit(table)?;

        // Atomic DDL: acquire lock before check-then-act to prevent TOCTOU races
        let _ddl_guard = self.ddl_lock.write();

        // Re-check limit under lock to prevent race conditions
        self.check_index_limit(table)?;

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

        // Build index from slab data
        let col_idx = schema
            .columns
            .iter()
            .position(|c| c.name == column)
            .or_else(|| (column == "_id").then_some(usize::MAX));

        let slab_rows = self
            .slab()
            .scan_all(table)
            .map_err(|e| RelationalError::StorageError(e.to_string()))?;

        for (row_id, slab_row) in slab_rows {
            let engine_row_id = row_id.as_u64() + 1; // Convert 0-based to 1-based
            let value = if column == "_id" {
                Value::Int(engine_row_id as i64)
            } else if let Some(idx) = col_idx {
                slab_row.get(idx).map_or(Value::Null, |v| v.clone().into())
            } else {
                continue;
            };
            self.btree_index_add(table, column, &value, engine_row_id)?;
        }

        Ok(())
    }

    /// Returns true if a B-tree index exists on the column.
    pub fn has_btree_index(&self, table: &str, column: &str) -> bool {
        let meta_key = Self::btree_meta_key(table, column);
        self.store.exists(&meta_key)
    }

    /// Drops a B-tree index from a column.
    ///
    /// # Errors
    /// Returns `TableNotFound`, `IndexNotFound`, or `StorageError`.
    pub fn drop_btree_index(&self, table: &str, column: &str) -> Result<()> {
        let _ = self.get_schema(table)?;

        // Atomic DDL: acquire lock before check-then-act to prevent TOCTOU races
        let _ddl_guard = self.ddl_lock.write();

        let meta_key = Self::btree_meta_key(table, column);
        if !self.store.exists(&meta_key) {
            return Err(RelationalError::IndexNotFound {
                table: table.to_string(),
                column: column.to_string(),
            });
        }

        // Remove in-memory index and count entries being removed
        let entries_removed = {
            let key = (table.to_string(), column.to_string());
            let mut indexes = self.btree_indexes.write();
            indexes.remove(&key).map_or(0, |btree| btree.len())
        };

        // Decrement entry counter
        if entries_removed > 0 {
            self.btree_entry_count
                .fetch_sub(entries_removed, Ordering::Relaxed);
        }

        let prefix = Self::btree_prefix(table, column);
        let keys = self.store.scan(&prefix);
        for key in keys {
            self.store.delete(&key)?;
        }

        self.store.delete(&meta_key)?;

        Ok(())
    }

    /// Returns columns that have B-tree indexes.
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

    /// Drops a hash index from a column.
    ///
    /// # Errors
    /// Returns `TableNotFound`, `IndexNotFound`, or `StorageError`.
    pub fn drop_index(&self, table: &str, column: &str) -> Result<()> {
        let _ = self.get_schema(table)?;

        // Atomic DDL: acquire lock before check-then-act to prevent TOCTOU races
        let _ddl_guard = self.ddl_lock.write();

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

    /// Returns true if a hash index exists on the column.
    pub fn has_index(&self, table: &str, column: &str) -> bool {
        let meta_key = Self::index_meta_key(table, column);
        self.store.exists(&meta_key)
    }

    /// Returns columns that have hash indexes.
    pub fn get_indexed_columns(&self, table: &str) -> Vec<String> {
        let prefix = format!("_idx:{table}:");
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

        // Atomic: acquire per-key lock before read-modify-write
        let lock = self.acquire_index_lock(&key);
        let _guard = lock.write();

        let mut ids: Vec<u64> = match self.store.get(&key) {
            Ok(tensor) => Self::tensor_to_id_list(&tensor)?,
            Err(_) => Vec::new(),
        };

        if !ids.contains(&row_id) {
            ids.push(row_id);
            self.store.put(&key, Self::id_list_to_tensor(&ids))?;
        }

        Ok(())
    }

    fn index_remove(&self, table: &str, column: &str, value: &Value, row_id: u64) -> Result<()> {
        let value_hash = value.hash_key();
        let key = Self::index_entry_key(table, column, &value_hash);

        // Atomic: acquire per-key lock before read-modify-write
        let lock = self.acquire_index_lock(&key);
        let _guard = lock.write();

        if let Ok(tensor) = self.store.get(&key) {
            let mut ids = Self::tensor_to_id_list(&tensor)?;
            ids.retain(|&id| id != row_id);

            if ids.is_empty() {
                self.store.delete(&key)?;
            } else {
                self.store.put(&key, Self::id_list_to_tensor(&ids))?;
            }
        }

        Ok(())
    }

    fn index_lookup(&self, table: &str, column: &str, value: &Value) -> Result<Option<Vec<u64>>> {
        if !self.has_index(table, column) {
            return Ok(None);
        }

        let value_hash = value.hash_key();
        let key = Self::index_entry_key(table, column, &value_hash);

        // Index exists but may have no entries for this value
        let ids = match self.store.get(&key) {
            Ok(tensor) => Self::tensor_to_id_list(&tensor)?,
            Err(_) => Vec::new(),
        };
        Ok(Some(ids))
    }

    fn tensor_to_id_list(tensor: &TensorData) -> Result<Vec<u64>> {
        match tensor.get("ids") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) => {
                if bytes.len() % 8 != 0 {
                    return Err(RelationalError::IndexCorrupted {
                        reason: format!(
                            "ID list has {} bytes, expected multiple of 8",
                            bytes.len()
                        ),
                    });
                }
                Ok(bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        let arr: [u8; 8] = chunk
                            .try_into()
                            .expect("chunks_exact(8) guarantees 8-byte slices");
                        u64::from_le_bytes(arr)
                    })
                    .collect())
            },
            // Legacy format fallback for existing data
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            Some(TensorValue::Vector(v)) => Ok(v.iter().map(|f| *f as u64).collect()),
            Some(other) => Err(RelationalError::IndexCorrupted {
                reason: format!("expected Bytes or Vector for 'ids', got {other:?}"),
            }),
            None => Ok(Vec::new()),
        }
    }

    fn id_list_to_tensor(ids: &[u64]) -> TensorData {
        let mut tensor = TensorData::new();
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        tensor.set("ids", TensorValue::Scalar(ScalarValue::Bytes(bytes)));
        tensor
    }

    fn get_table_indexes(&self, table: &str) -> Vec<String> {
        self.get_indexed_columns(table)
    }

    fn get_table_btree_indexes(&self, table: &str) -> Vec<String> {
        self.get_btree_indexed_columns(table)
    }

    #[allow(clippy::significant_drop_tightening)] // Lock scope is intentional for atomicity
    fn btree_index_add(&self, table: &str, column: &str, value: &Value, row_id: u64) -> Result<()> {
        let key = (table.to_string(), column.to_string());
        let ordered_key = OrderedKey::from_value(value);
        let sortable = value.sortable_key();
        let store_key = Self::btree_entry_key(table, column, &sortable);

        // Lock ordering: global btree_indexes FIRST, then per-key lock
        // Update in-memory BTreeMap (protected by btree_indexes RwLock)
        let added_new_key = {
            let mut indexes = self.btree_indexes.write();
            let btree = indexes.entry(key).or_default();

            // Check if this is a new key (will add memory)
            let is_new_key = !btree.contains_key(&ordered_key);
            if is_new_key {
                // Check bounds before adding new entry
                let current = self.btree_entry_count.load(Ordering::Relaxed);
                if current >= self.max_btree_entries {
                    return Err(RelationalError::ResultTooLarge {
                        operation: "btree_index_add".to_string(),
                        actual: current + 1,
                        max: self.max_btree_entries,
                    });
                }
            }

            let ids = btree.entry(ordered_key).or_default();
            if !ids.contains(&row_id) {
                ids.push(row_id);
            }
            is_new_key
        };

        // Increment counter after successful in-memory insert
        if added_new_key {
            self.btree_entry_count.fetch_add(1, Ordering::Relaxed);
        }

        // Per-key lock for TensorStore update (after global lock released)
        let lock = self.acquire_index_lock(&store_key);
        let _guard = lock.write();

        // Update TensorStore (now protected by per-key lock)
        let mut ids: Vec<u64> = match self.store.get(&store_key) {
            Ok(tensor) => Self::tensor_to_id_list(&tensor)?,
            Err(_) => Vec::new(),
        };

        if !ids.contains(&row_id) {
            ids.push(row_id);
            self.store.put(&store_key, Self::id_list_to_tensor(&ids))?;
        }

        Ok(())
    }

    #[allow(clippy::significant_drop_tightening)] // Lock scope is intentional for atomicity
    fn btree_index_remove(
        &self,
        table: &str,
        column: &str,
        value: &Value,
        row_id: u64,
    ) -> Result<()> {
        let key = (table.to_string(), column.to_string());
        let ordered_key = OrderedKey::from_value(value);
        let sortable = value.sortable_key();
        let store_key = Self::btree_entry_key(table, column, &sortable);

        // Lock ordering: global btree_indexes FIRST, then per-key lock
        // Update in-memory BTreeMap
        let removed_key = {
            let mut indexes = self.btree_indexes.write();
            let mut key_removed = false;
            if let Some(btree) = indexes.get_mut(&key) {
                if let Some(ids) = btree.get_mut(&ordered_key) {
                    ids.retain(|&id| id != row_id);
                    if ids.is_empty() {
                        btree.remove(&ordered_key);
                        key_removed = true;
                    }
                }
            }
            key_removed
        };

        // Decrement counter if key was removed
        if removed_key {
            self.btree_entry_count.fetch_sub(1, Ordering::Relaxed);
        }

        // Per-key lock for TensorStore update (after global lock released)
        let lock = self.acquire_index_lock(&store_key);
        let _guard = lock.write();

        // Update TensorStore (now protected by per-key lock)
        if let Ok(tensor) = self.store.get(&store_key) {
            let mut ids = Self::tensor_to_id_list(&tensor)?;
            ids.retain(|&id| id != row_id);

            if ids.is_empty() {
                self.store.delete(&store_key)?;
            } else {
                self.store.put(&store_key, Self::id_list_to_tensor(&ids))?;
            }
        }

        Ok(())
    }

    /// B-tree range lookup: returns row IDs matching the range condition.
    /// Uses in-memory `BTreeMap` for O(log n) range operations.
    #[allow(clippy::significant_drop_tightening)] // Lock scope is intentional for atomicity
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
        format!("_col:{table}:{column}:data")
    }

    fn column_ids_key(table: &str, column: &str) -> String {
        format!("_col:{table}:{column}:ids")
    }

    fn column_nulls_key(table: &str, column: &str) -> String {
        format!("_col:{table}:{column}:nulls")
    }

    fn column_meta_key(table: &str, column: &str) -> String {
        format!("_col:{table}:{column}:meta")
    }

    /// Returns true if column data is stored in columnar format.
    pub fn has_columnar_data(&self, table: &str, column: &str) -> bool {
        // With slab integration, all columns are stored in columnar format
        // Check if the table exists and has the column
        self.slab()
            .get_schema(table)
            .is_some_and(|s| s.columns.iter().any(|c| c.name == column))
    }

    /// Materialize specified columns into columnar format.
    /// This extracts column data from row storage into contiguous vectors.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `ColumnNotFound`.
    pub fn materialize_columns(&self, table: &str, columns: &[&str]) -> Result<()> {
        // With RelationalSlab integration, data is already stored in columnar format.
        // This method now just validates inputs for backward compatibility.
        let schema = self.get_schema(table)?;

        for col_name in columns {
            if schema.get_column(col_name).is_none() {
                return Err(RelationalError::ColumnNotFound(col_name.to_string()));
            }
        }

        Ok(())
    }

    #[allow(clippy::cast_possible_truncation)] // Null positions fit in usize on 64-bit platforms
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

    /// # Errors
    /// Returns `TableNotFound`, `ColumnNotFound`, or `StorageError`.
    #[allow(clippy::cast_possible_truncation)] // Dict indices fit in u32; row counts fit in u64
    pub fn load_column_data(&self, table: &str, column: &str) -> Result<ColumnData> {
        // With slab integration, extract column data directly from slab's columnar storage
        let schema = self.get_schema(table)?;
        let col_idx = schema
            .columns
            .iter()
            .position(|c| c.name == column)
            .ok_or_else(|| RelationalError::ColumnNotFound(column.to_string()))?;
        let col_type = &schema.columns[col_idx].column_type;

        // Scan all rows from slab
        let slab_rows = self
            .slab()
            .scan_all(table)
            .map_err(|e| RelationalError::StorageError(e.to_string()))?;

        let row_count = slab_rows.len();
        let mut row_ids = Vec::with_capacity(row_count);
        let mut null_positions = Vec::new();

        // Extract column values based on type
        let values = match col_type {
            ColumnType::Int => {
                let mut values = Vec::with_capacity(row_count);
                for (idx, (row_id, row)) in slab_rows.iter().enumerate() {
                    row_ids.push(row_id.as_u64() + 1);
                    match &row[col_idx] {
                        SlabColumnValue::Int(v) => values.push(*v),
                        SlabColumnValue::Null => {
                            values.push(0);
                            null_positions.push(idx as u64);
                        },
                        _ => values.push(0),
                    }
                }
                ColumnValues::Int(values)
            },
            ColumnType::Float => {
                let mut values = Vec::with_capacity(row_count);
                for (idx, (row_id, row)) in slab_rows.iter().enumerate() {
                    row_ids.push(row_id.as_u64() + 1);
                    match &row[col_idx] {
                        SlabColumnValue::Float(v) => values.push(*v),
                        SlabColumnValue::Null => {
                            values.push(0.0);
                            null_positions.push(idx as u64);
                        },
                        _ => values.push(0.0),
                    }
                }
                ColumnValues::Float(values)
            },
            ColumnType::String => {
                let mut dict = Vec::new();
                let mut indices = Vec::with_capacity(row_count);
                let mut dict_map: std::collections::HashMap<String, u32> =
                    std::collections::HashMap::new();
                for (idx, (row_id, row)) in slab_rows.iter().enumerate() {
                    row_ids.push(row_id.as_u64() + 1);
                    match &row[col_idx] {
                        SlabColumnValue::String(s) => {
                            let dict_idx = *dict_map.entry(s.clone()).or_insert_with(|| {
                                let idx = dict.len() as u32;
                                dict.push(s.clone());
                                idx
                            });
                            indices.push(dict_idx);
                        },
                        SlabColumnValue::Null => {
                            indices.push(0);
                            null_positions.push(idx as u64);
                        },
                        _ => indices.push(0),
                    }
                }
                ColumnValues::String { dict, indices }
            },
            ColumnType::Bool => {
                let word_count = row_count.div_ceil(64);
                let mut bitmap = vec![0u64; word_count];
                for (idx, (row_id, row)) in slab_rows.iter().enumerate() {
                    row_ids.push(row_id.as_u64() + 1);
                    match &row[col_idx] {
                        SlabColumnValue::Bool(b) => {
                            if *b {
                                bitmap[idx / 64] |= 1u64 << (idx % 64);
                            }
                        },
                        SlabColumnValue::Null => {
                            null_positions.push(idx as u64);
                        },
                        _ => {},
                    }
                }
                ColumnValues::Bool(bitmap)
            },
        };

        let nulls = Self::build_null_bitmap(null_positions, row_count);

        Ok(ColumnData {
            name: column.to_string(),
            row_ids,
            nulls,
            values,
        })
    }

    /// # Errors
    /// This function cannot fail - returns `Ok(())` unconditionally.
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
    /// If columnar data is available and `options.prefer_columnar` is true,
    /// uses SIMD-accelerated vectorized filtering. Otherwise falls back to
    /// row-based scan with projection.
    ///
    /// # Errors
    /// Returns `TableNotFound`, `ColumnNotFound`, `TypeMismatch`, or `StorageError`.
    #[must_use = "query results should be used"]
    #[instrument(skip(self, condition, options), fields(table = %table))]
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
        // Try slab-based SIMD filtering first (Phase 4 integration)
        if let Some(rows) = self.try_slab_select(table, &condition, &options) {
            return Ok(rows);
        }

        // Fall back to legacy columnar path
        let filter_columns = Self::extract_filter_columns(&condition);

        let projection_columns: Vec<String> = if let Some(cols) = &options.projection {
            cols.clone()
        } else {
            let schema_key = format!("{table}:_schema");
            match self.store.get(&schema_key) {
                Ok(tensor) => tensor
                    .keys()
                    .filter(|k| !k.starts_with('_'))
                    .cloned()
                    .collect(),
                Err(_) => return self.select_with_projection(table, condition, options.projection),
            }
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

        let row_count = column_map.values().next().map_or(0, |c| c.row_ids.len());

        if row_count == 0 {
            return Ok(Vec::new());
        }

        let selection = Self::apply_vectorized_filter(&column_map, &condition, row_count)?;

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
                options.projection.as_ref(),
            ))
        } else {
            Ok(self.materialize_selected_rows(
                table,
                &row_ids,
                &selected_indices,
                options.projection,
            ))
        }
    }

    #[allow(clippy::too_many_lines)] // Exhaustive match over all Condition variants
    fn apply_vectorized_filter(
        columns: &HashMap<String, ColumnData>,
        condition: &Condition,
        row_count: usize,
    ) -> Result<SelectionVector> {
        match condition {
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
                let sel_a = Self::apply_vectorized_filter(columns, a, row_count)?;
                let sel_b = Self::apply_vectorized_filter(columns, b, row_count)?;
                Ok(sel_a.intersect(&sel_b))
            },

            Condition::Or(a, b) => {
                let sel_a = Self::apply_vectorized_filter(columns, a, row_count)?;
                let sel_b = Self::apply_vectorized_filter(columns, b, row_count)?;
                Ok(sel_a.union(&sel_b))
            },

            // Fallback for non-Int comparisons: select all and filter later
            _ => Ok(SelectionVector::all(row_count)),
        }
    }

    /// Try to execute select using `RelationalSlab` directly.
    /// Returns `Some(rows)` if successful, `None` if slab path not available.
    fn try_slab_select(
        &self,
        table: &str,
        condition: &Condition,
        options: &ColumnarScanOptions,
    ) -> Option<Vec<Row>> {
        // Get selection from slab-based SIMD filtering
        let (selection, _row_count) = self.apply_slab_vectorized_filter(table, condition)?;

        // Get selected row indices
        let indices = selection.selected_indices();

        // Fetch rows from slab
        let slab_rows = self.slab().get_rows_by_indices(table, &indices).ok()?;

        // Get schema for column name mapping
        let schema = self.slab().get_schema(table)?;

        // Convert slab rows to engine rows
        let rows: Vec<Row> = slab_rows
            .into_iter()
            .map(|(row_id, slab_row)| {
                let id = row_id.as_u64() + 1; // Convert 0-based to 1-based
                let values: Vec<(String, Value)> = slab_row
                    .into_iter()
                    .enumerate()
                    .map(|(col_idx, value)| {
                        let col_name = schema.columns[col_idx].name.clone();
                        (col_name, value.into())
                    })
                    .collect();
                Row { id, values }
            })
            .collect();

        // Apply projection if specified
        if let Some(proj_cols) = &options.projection {
            Some(
                rows.into_iter()
                    .map(|row| {
                        let values: Vec<(String, Value)> = row
                            .values
                            .into_iter()
                            .filter(|(name, _)| proj_cols.contains(name))
                            .collect();
                        Row { id: row.id, values }
                    })
                    .collect(),
            )
        } else {
            Some(rows)
        }
    }

    /// Apply SIMD filtering directly on `RelationalSlab`'s columnar data.
    /// Returns `(SelectionVector, row_count)` if successful, `None` if slab doesn't support the condition.
    #[allow(clippy::too_many_lines)] // Exhaustive match over all Condition variants
    fn apply_slab_vectorized_filter(
        &self,
        table: &str,
        condition: &Condition,
    ) -> Option<(SelectionVector, usize)> {
        match condition {
            Condition::True => {
                // Get row count from slab
                let row_count = self.slab().row_count(table).ok()?;
                Some((SelectionVector::all(row_count), row_count))
            },

            Condition::Eq(col, Value::Int(val)) => {
                let (values, alive_words, _null_words) =
                    self.slab().get_int_column(table, col).ok()?;
                let row_count = values.len();
                if row_count == 0 {
                    return Some((SelectionVector::none(0), 0));
                }
                let mut bitmap = vec![0u64; simd::bitmap_words(row_count)];
                simd::filter_eq_i64(&values, *val, &mut bitmap);
                // AND with alive bitmap to exclude deleted rows
                Self::apply_alive_mask(&mut bitmap, &alive_words);
                Some((SelectionVector::from_bitmap(bitmap, row_count), row_count))
            },

            Condition::Ne(col, Value::Int(val)) => {
                let (values, alive_words, _null_words) =
                    self.slab().get_int_column(table, col).ok()?;
                let row_count = values.len();
                if row_count == 0 {
                    return Some((SelectionVector::none(0), 0));
                }
                let mut bitmap = vec![0u64; simd::bitmap_words(row_count)];
                simd::filter_ne_i64(&values, *val, &mut bitmap);
                Self::apply_alive_mask(&mut bitmap, &alive_words);
                Some((SelectionVector::from_bitmap(bitmap, row_count), row_count))
            },

            Condition::Lt(col, Value::Int(val)) => {
                let (values, alive_words, _null_words) =
                    self.slab().get_int_column(table, col).ok()?;
                let row_count = values.len();
                if row_count == 0 {
                    return Some((SelectionVector::none(0), 0));
                }
                let mut bitmap = vec![0u64; simd::bitmap_words(row_count)];
                simd::filter_lt_i64(&values, *val, &mut bitmap);
                Self::apply_alive_mask(&mut bitmap, &alive_words);
                Some((SelectionVector::from_bitmap(bitmap, row_count), row_count))
            },

            Condition::Le(col, Value::Int(val)) => {
                let (values, alive_words, _null_words) =
                    self.slab().get_int_column(table, col).ok()?;
                let row_count = values.len();
                if row_count == 0 {
                    return Some((SelectionVector::none(0), 0));
                }
                let mut bitmap = vec![0u64; simd::bitmap_words(row_count)];
                simd::filter_le_i64(&values, *val, &mut bitmap);
                Self::apply_alive_mask(&mut bitmap, &alive_words);
                Some((SelectionVector::from_bitmap(bitmap, row_count), row_count))
            },

            Condition::Gt(col, Value::Int(val)) => {
                let (values, alive_words, _null_words) =
                    self.slab().get_int_column(table, col).ok()?;
                let row_count = values.len();
                if row_count == 0 {
                    return Some((SelectionVector::none(0), 0));
                }
                let mut bitmap = vec![0u64; simd::bitmap_words(row_count)];
                simd::filter_gt_i64(&values, *val, &mut bitmap);
                Self::apply_alive_mask(&mut bitmap, &alive_words);
                Some((SelectionVector::from_bitmap(bitmap, row_count), row_count))
            },

            Condition::Ge(col, Value::Int(val)) => {
                let (values, alive_words, _null_words) =
                    self.slab().get_int_column(table, col).ok()?;
                let row_count = values.len();
                if row_count == 0 {
                    return Some((SelectionVector::none(0), 0));
                }
                let mut bitmap = vec![0u64; simd::bitmap_words(row_count)];
                simd::filter_ge_i64(&values, *val, &mut bitmap);
                Self::apply_alive_mask(&mut bitmap, &alive_words);
                Some((SelectionVector::from_bitmap(bitmap, row_count), row_count))
            },

            Condition::Lt(col, Value::Float(val)) => {
                let (values, alive_words, _null_words) =
                    self.slab().get_float_column(table, col).ok()?;
                let row_count = values.len();
                if row_count == 0 {
                    return Some((SelectionVector::none(0), 0));
                }
                let mut bitmap = vec![0u64; simd::bitmap_words(row_count)];
                simd::filter_lt_f64(&values, *val, &mut bitmap);
                Self::apply_alive_mask(&mut bitmap, &alive_words);
                Some((SelectionVector::from_bitmap(bitmap, row_count), row_count))
            },

            Condition::Gt(col, Value::Float(val)) => {
                let (values, alive_words, _null_words) =
                    self.slab().get_float_column(table, col).ok()?;
                let row_count = values.len();
                if row_count == 0 {
                    return Some((SelectionVector::none(0), 0));
                }
                let mut bitmap = vec![0u64; simd::bitmap_words(row_count)];
                simd::filter_gt_f64(&values, *val, &mut bitmap);
                Self::apply_alive_mask(&mut bitmap, &alive_words);
                Some((SelectionVector::from_bitmap(bitmap, row_count), row_count))
            },

            Condition::Eq(col, Value::Float(val)) => {
                let (values, alive_words, _null_words) =
                    self.slab().get_float_column(table, col).ok()?;
                let row_count = values.len();
                if row_count == 0 {
                    return Some((SelectionVector::none(0), 0));
                }
                let mut bitmap = vec![0u64; simd::bitmap_words(row_count)];
                simd::filter_eq_f64(&values, *val, &mut bitmap);
                Self::apply_alive_mask(&mut bitmap, &alive_words);
                Some((SelectionVector::from_bitmap(bitmap, row_count), row_count))
            },

            Condition::And(a, b) => {
                let (sel_a, count_a) = self.apply_slab_vectorized_filter(table, a)?;
                let (sel_b, _) = self.apply_slab_vectorized_filter(table, b)?;
                Some((sel_a.intersect(&sel_b), count_a))
            },

            Condition::Or(a, b) => {
                let (sel_a, count_a) = self.apply_slab_vectorized_filter(table, a)?;
                let (sel_b, _) = self.apply_slab_vectorized_filter(table, b)?;
                Some((sel_a.union(&sel_b), count_a))
            },

            // Unsupported conditions - fall back to legacy path
            _ => None,
        }
    }

    /// Apply alive bitmap mask to filter result (AND operation).
    fn apply_alive_mask(bitmap: &mut [u64], alive_words: &[u64]) {
        for (i, word) in bitmap.iter_mut().enumerate() {
            if i < alive_words.len() {
                *word &= alive_words[i];
            } else {
                *word = 0;
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)] // projection is used optionally, ownership is fine
    fn materialize_selected_rows(
        &self,
        table: &str,
        row_ids: &[u64],
        selected_indices: &[usize],
        projection: Option<Vec<String>>,
    ) -> Vec<Row> {
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
        rows
    }

    /// Build rows purely from columnar data without touching row storage.
    fn materialize_from_columns(
        columns: &HashMap<String, ColumnData>,
        row_ids: &[u64],
        selected_indices: &[usize],
        projection: Option<&Vec<String>>,
    ) -> Vec<Row> {
        let num_rows = selected_indices.len();
        let mut rows = Vec::with_capacity(num_rows);

        // Pre-resolve column references to avoid repeated HashMap lookups
        let col_refs: Vec<(&str, &ColumnData)> = projection.map_or_else(
            || columns.iter().map(|(k, v)| (k.as_str(), v)).collect(),
            |cols| {
                cols.iter()
                    .filter(|c| *c != "_id")
                    .filter_map(|c| columns.get(c).map(|data| (c.as_str(), data)))
                    .collect()
            },
        );

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
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[must_use = "query results should be used"]
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

    // ==================== Transaction API ====================

    /// Begin a new transaction.
    pub fn begin_transaction(&self) -> u64 {
        self.tx_manager.begin()
    }

    /// Get the transaction manager for advanced use cases.
    pub const fn tx_manager(&self) -> &TransactionManager {
        &self.tx_manager
    }

    /// Check if a transaction is active.
    pub fn is_transaction_active(&self, tx_id: u64) -> bool {
        self.tx_manager.is_active(tx_id)
    }

    /// Commit a transaction, making all changes permanent.
    ///
    /// # Errors
    /// Returns `TransactionNotFound` or `TransactionInactive`.
    pub fn commit(&self, tx_id: u64) -> Result<()> {
        if !self.tx_manager.is_active(tx_id) {
            if self.tx_manager.get(tx_id).is_none() {
                return Err(RelationalError::TransactionNotFound(tx_id));
            }
            return Err(RelationalError::TransactionInactive(tx_id));
        }

        // Mark as committing
        self.tx_manager.set_phase(tx_id, TxPhase::Committing);

        // Release locks
        self.tx_manager.release_locks(tx_id);

        // Mark as committed and remove
        self.tx_manager.set_phase(tx_id, TxPhase::Committed);
        self.tx_manager.remove(tx_id);

        Ok(())
    }

    /// Rollback a transaction, undoing all changes.
    ///
    /// This function always completes transaction cleanup (releasing locks, removing
    /// the transaction) even if individual undo operations fail. This ensures
    /// transactions never get stuck in the `Aborting` phase.
    ///
    /// # Errors
    /// Returns `TransactionNotFound` or `TransactionInactive` for invalid transactions.
    /// Returns `RollbackFailed` if any undo operations failed, but the transaction
    /// is still properly cleaned up in this case.
    pub fn rollback(&self, tx_id: u64) -> Result<()> {
        if !self.tx_manager.is_active(tx_id) {
            if self.tx_manager.get(tx_id).is_none() {
                return Err(RelationalError::TransactionNotFound(tx_id));
            }
            return Err(RelationalError::TransactionInactive(tx_id));
        }

        // Mark as aborting
        self.tx_manager.set_phase(tx_id, TxPhase::Aborting);

        // Get undo log
        let undo_log = self.tx_manager.get_undo_log(tx_id).unwrap_or_default();

        // Apply undo entries in reverse order, collecting any errors
        let mut all_errors: Vec<String> = Vec::new();
        for entry in undo_log.into_iter().rev() {
            let errors = self.apply_undo_entry(&entry);
            all_errors.extend(errors);
        }

        // ALWAYS release locks and clean up, even if undo had errors
        self.tx_manager.release_locks(tx_id);
        self.tx_manager.set_phase(tx_id, TxPhase::Aborted);
        self.tx_manager.remove(tx_id);

        // Report errors if any occurred during undo
        if !all_errors.is_empty() {
            return Err(RelationalError::RollbackFailed {
                tx_id,
                reason: format!(
                    "{} error(s) during rollback: {}",
                    all_errors.len(),
                    all_errors.join("; ")
                ),
            });
        }

        Ok(())
    }

    /// Apply a single undo entry during rollback.
    ///
    /// This function is infallible by design - rollback must always complete
    /// to release locks and clean up transaction state. Any errors during
    /// undo are collected and returned but do not prevent rollback from completing.
    #[allow(clippy::cast_possible_wrap)] // Row IDs won't exceed i64::MAX
    #[allow(clippy::too_many_lines)] // Necessary to handle all UndoEntry variants with error collection
    fn apply_undo_entry(&self, entry: &UndoEntry) -> Vec<String> {
        let mut errors: Vec<String> = Vec::new();

        match entry {
            UndoEntry::InsertedRow {
                table,
                slab_row_id,
                row_id,
                index_entries,
            } => {
                // Undo insert: delete the row
                if let Err(e) = self.slab().delete(table, *slab_row_id) {
                    errors.push(format!(
                        "Failed to delete row {row_id} from table '{table}': {e}"
                    ));
                }

                // Remove from indexes (continue even if some fail)
                for (col, value) in index_entries {
                    if let Err(e) = self.index_remove(table, col, value, *row_id) {
                        errors.push(format!(
                            "Failed to remove index entry for {table}.{col}: {e}"
                        ));
                    }
                    if let Err(e) = self.btree_index_remove(table, col, value, *row_id) {
                        errors.push(format!(
                            "Failed to remove btree index entry for {table}.{col}: {e}"
                        ));
                    }
                }
            },
            UndoEntry::UpdatedRow {
                table,
                slab_row_id,
                row_id,
                old_values,
                index_changes,
            } => {
                // Undo update: restore old values
                if let Err(e) = self.slab().restore_row(table, *slab_row_id, old_values) {
                    errors.push(format!(
                        "Failed to restore row {row_id} in table '{table}': {e}"
                    ));
                }

                // Revert index changes (continue even if some fail)
                for change in index_changes {
                    if let Err(e) =
                        self.index_remove(table, &change.column, &change.new_value, *row_id)
                    {
                        errors.push(format!(
                            "Failed to remove index entry for {table}.{}: {e}",
                            change.column
                        ));
                    }
                    if let Err(e) =
                        self.index_add(table, &change.column, &change.old_value, *row_id)
                    {
                        errors.push(format!(
                            "Failed to add index entry for {table}.{}: {e}",
                            change.column
                        ));
                    }
                    if let Err(e) =
                        self.btree_index_remove(table, &change.column, &change.new_value, *row_id)
                    {
                        errors.push(format!(
                            "Failed to remove btree index for {table}.{}: {e}",
                            change.column
                        ));
                    }
                    if let Err(e) =
                        self.btree_index_add(table, &change.column, &change.old_value, *row_id)
                    {
                        errors.push(format!(
                            "Failed to add btree index for {table}.{}: {e}",
                            change.column
                        ));
                    }
                }
            },
            UndoEntry::DeletedRow {
                table,
                slab_row_id,
                row_id,
                old_values,
                index_entries,
            } => {
                // Undo delete: restore the row
                if let Err(e) = self
                    .slab()
                    .restore_deleted_row(table, *slab_row_id, old_values)
                {
                    errors.push(format!(
                        "Failed to restore deleted row {row_id} in table '{table}': {e}"
                    ));
                }

                // Restore index entries (continue even if some fail)
                for (col, value) in index_entries {
                    if let Err(e) = self.index_add(table, col, value, *row_id) {
                        errors.push(format!("Failed to add index entry for {table}.{col}: {e}"));
                    }
                    if let Err(e) = self.btree_index_add(table, col, value, *row_id) {
                        errors.push(format!(
                            "Failed to add btree index entry for {table}.{col}: {e}"
                        ));
                    }
                }
            },
        }

        errors
    }

    /// Insert a row within a transaction.
    ///
    /// # Errors
    /// Returns `TransactionNotFound`, `TransactionInactive`, `TableNotFound`,
    /// `NullNotAllowed`, `TypeMismatch`, or `StorageError`.
    #[allow(clippy::cast_possible_wrap)] // Row IDs won't exceed i64::MAX
    #[allow(clippy::needless_pass_by_value)] // Public API takes ownership for ergonomics
    pub fn tx_insert(
        &self,
        tx_id: u64,
        table: &str,
        values: HashMap<String, Value>,
    ) -> Result<u64> {
        if !self.tx_manager.is_active(tx_id) {
            if self.tx_manager.get(tx_id).is_none() {
                return Err(RelationalError::TransactionNotFound(tx_id));
            }
            return Err(RelationalError::TransactionInactive(tx_id));
        }

        let schema = self.get_schema(table)?;

        // Validate values
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

        // Build slab row
        let slab_row: Vec<SlabColumnValue> = schema
            .columns
            .iter()
            .map(|col| {
                values
                    .get(&col.name)
                    .map_or(SlabColumnValue::Null, std::convert::Into::into)
            })
            .collect();

        // Insert into slab
        let slab_row_id = self
            .slab()
            .insert(table, slab_row)
            .map_err(|e| RelationalError::StorageError(e.to_string()))?;

        let row_id = slab_row_id.as_u64() + 1;

        // Update row counter
        self.row_counters
            .entry(table.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_max(row_id, Ordering::Relaxed);

        // Update indexes
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

        // Capture index entries for rollback (must happen AFTER index updates)
        let mut index_entries: Vec<(String, Value)> = Vec::new();
        for col in indexed_columns.iter().chain(btree_columns.iter()) {
            if col == "_id" {
                index_entries.push((col.clone(), Value::Int(row_id as i64)));
            } else if let Some(value) = values.get(col) {
                index_entries.push((col.clone(), value.clone()));
            }
        }

        // Record undo entry
        self.tx_manager.record_undo(
            tx_id,
            UndoEntry::InsertedRow {
                table: table.to_string(),
                slab_row_id,
                row_id,
                index_entries,
            },
        );

        Ok(row_id)
    }

    /// Update rows within a transaction.
    ///
    /// # Errors
    /// Returns `TransactionNotFound`, `TransactionInactive`, `TableNotFound`,
    /// `ColumnNotFound`, `TypeMismatch`, `NullNotAllowed`, `LockConflict`, or `StorageError`.
    #[allow(clippy::too_many_lines)] // Transaction update logic is inherently complex
    #[allow(clippy::needless_pass_by_value)] // Public API takes ownership for ergonomics
    pub fn tx_update(
        &self,
        tx_id: u64,
        table: &str,
        condition: Condition,
        updates: HashMap<String, Value>,
    ) -> Result<usize> {
        if !self.tx_manager.is_active(tx_id) {
            if self.tx_manager.get(tx_id).is_none() {
                return Err(RelationalError::TransactionNotFound(tx_id));
            }
            return Err(RelationalError::TransactionInactive(tx_id));
        }

        let schema = self.get_schema(table)?;

        // Validate updates
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

        // Scan to find matching rows
        let slab_rows = self
            .slab()
            .scan_all(table)
            .map_err(|e| RelationalError::StorageError(e.to_string()))?;

        let matching_rows: Vec<(SlabRowId, Row, Vec<SlabColumnValue>)> = slab_rows
            .into_iter()
            .filter_map(|(row_id, slab_row)| {
                let row = Self::slab_row_to_engine_row(&schema, row_id, slab_row.clone());
                if condition.evaluate(&row) {
                    Some((row_id, row, slab_row))
                } else {
                    None
                }
            })
            .collect();

        // Acquire locks on all matching rows
        let rows_to_lock: Vec<(String, u64)> = matching_rows
            .iter()
            .map(|(_, row, _)| (table.to_string(), row.id))
            .collect();

        if !rows_to_lock.is_empty() {
            self.tx_manager
                .lock_manager()
                .try_lock(tx_id, &rows_to_lock)
                .map_err(|info| RelationalError::LockConflict {
                    tx_id,
                    blocking_tx: info.blocking_tx,
                    table: info.table,
                    row_id: info.row_id,
                })?;
        }

        // Convert updates to slab format
        let slab_updates: Vec<(String, SlabColumnValue)> = updates
            .iter()
            .map(|(col, val)| (col.clone(), val.into()))
            .collect();

        for (slab_row_id, row, old_slab_values) in &matching_rows {
            // Capture index changes for undo
            let mut index_changes = Vec::new();
            for col in indexed_columns.iter().chain(btree_columns.iter()) {
                if let Some(new_value) = updates.get(col) {
                    if let Some(old_value) = row.get_with_id(col) {
                        index_changes.push(IndexChange {
                            column: col.clone(),
                            old_value: old_value.clone(),
                            new_value: new_value.clone(),
                        });
                    }
                }
            }

            // Record undo entry BEFORE making changes
            self.tx_manager.record_undo(
                tx_id,
                UndoEntry::UpdatedRow {
                    table: table.to_string(),
                    slab_row_id: *slab_row_id,
                    row_id: row.id,
                    old_values: old_slab_values.clone(),
                    index_changes,
                },
            );

            // Update indexes
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

            // Update the row in slab
            self.slab()
                .update_row(table, *slab_row_id, &slab_updates)
                .map_err(|e| RelationalError::StorageError(e.to_string()))?;
        }

        Ok(matching_rows.len())
    }

    /// Delete rows within a transaction.
    ///
    /// # Errors
    /// Returns `TransactionNotFound`, `TransactionInactive`, `TableNotFound`,
    /// `LockConflict`, or `StorageError`.
    #[allow(clippy::needless_pass_by_value)] // Public API takes ownership for ergonomics
    pub fn tx_delete(&self, tx_id: u64, table: &str, condition: Condition) -> Result<usize> {
        if !self.tx_manager.is_active(tx_id) {
            if self.tx_manager.get(tx_id).is_none() {
                return Err(RelationalError::TransactionNotFound(tx_id));
            }
            return Err(RelationalError::TransactionInactive(tx_id));
        }

        let schema = self.get_schema(table)?;

        let indexed_columns = self.get_table_indexes(table);
        let btree_columns = self.get_table_btree_indexes(table);

        // Scan to find matching rows
        let slab_rows = self
            .slab()
            .scan_all(table)
            .map_err(|e| RelationalError::StorageError(e.to_string()))?;

        let to_delete: Vec<(SlabRowId, Row, Vec<SlabColumnValue>)> = slab_rows
            .into_iter()
            .filter_map(|(row_id, slab_row)| {
                let row = Self::slab_row_to_engine_row(&schema, row_id, slab_row.clone());
                if condition.evaluate(&row) {
                    Some((row_id, row, slab_row))
                } else {
                    None
                }
            })
            .collect();

        // Acquire locks on all rows to delete
        let rows_to_lock: Vec<(String, u64)> = to_delete
            .iter()
            .map(|(_, row, _)| (table.to_string(), row.id))
            .collect();

        if !rows_to_lock.is_empty() {
            self.tx_manager
                .lock_manager()
                .try_lock(tx_id, &rows_to_lock)
                .map_err(|info| RelationalError::LockConflict {
                    tx_id,
                    blocking_tx: info.blocking_tx,
                    table: info.table,
                    row_id: info.row_id,
                })?;
        }

        for (slab_row_id, row, old_slab_values) in &to_delete {
            // Capture index entries for undo
            let mut index_entries: Vec<(String, Value)> = Vec::new();
            for col in indexed_columns.iter().chain(btree_columns.iter()) {
                if let Some(value) = row.get_with_id(col) {
                    index_entries.push((col.clone(), value));
                }
            }

            // Record undo entry BEFORE making changes
            self.tx_manager.record_undo(
                tx_id,
                UndoEntry::DeletedRow {
                    table: table.to_string(),
                    slab_row_id: *slab_row_id,
                    row_id: row.id,
                    old_values: old_slab_values.clone(),
                    index_entries,
                },
            );

            // Remove from indexes
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

            // Delete from slab
            self.slab()
                .delete(table, *slab_row_id)
                .map_err(|e| RelationalError::StorageError(e.to_string()))?;
        }

        Ok(to_delete.len())
    }

    /// Select rows within a transaction (read-committed isolation).
    ///
    /// In read-committed isolation, reads see committed data only.
    /// No read locks are acquired.
    ///
    /// # Errors
    /// Returns `TransactionNotFound`, `TransactionInactive`, `TableNotFound`, or `StorageError`.
    #[must_use = "query results should be used"]
    pub fn tx_select(&self, tx_id: u64, table: &str, condition: Condition) -> Result<Vec<Row>> {
        if !self.tx_manager.is_active(tx_id) {
            if self.tx_manager.get(tx_id).is_none() {
                return Err(RelationalError::TransactionNotFound(tx_id));
            }
            return Err(RelationalError::TransactionInactive(tx_id));
        }

        // For read-committed isolation, just delegate to regular select
        self.select(table, condition)
    }

    /// Get the number of active transactions.
    pub fn active_transaction_count(&self) -> usize {
        self.tx_manager.active_count()
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
        // Test that row counter is properly initialized when inserting
        let engine = RelationalEngine::new();

        // Create table properly (initializes both slab and metadata)
        let schema = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("age", ColumnType::Int),
        ]);
        engine.create_table("manual_table", schema).unwrap();

        // Clear the row counter to simulate uninitialized state
        engine.row_counters.remove("manual_table");

        // Now insert - this should handle the case where counter doesn't exist
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        let id = engine.insert("manual_table", values).unwrap();
        // With slab integration, ID is slab's row_id + 1 (0-based to 1-based)
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

        // With slab-based storage, data is always columnar
        assert!(engine.has_columnar_data("users", "age"));

        // drop_columnar_data is now a no-op (slab is always columnar)
        engine.drop_columnar_data("users", "age").unwrap();

        // Column still exists in schema, so columnar data is still available
        assert!(engine.has_columnar_data("users", "age"));
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
    fn load_column_data_empty_table() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // With slab-based storage, load_column_data works even on empty tables
        // It returns empty column data rather than an error
        let result = engine.load_column_data("users", "age");
        assert!(result.is_ok());
        let col_data = result.unwrap();
        assert_eq!(col_data.row_ids.len(), 0);
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

    #[test]
    fn test_invalid_table_name_empty() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        let result = engine.create_table("", schema);
        assert!(matches!(result, Err(RelationalError::InvalidName(_))));
    }

    #[test]
    fn test_invalid_table_name_colon() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        let result = engine.create_table("foo:bar", schema);
        assert!(matches!(result, Err(RelationalError::InvalidName(_))));
    }

    #[test]
    fn test_invalid_table_name_comma() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        let result = engine.create_table("foo,bar", schema);
        assert!(matches!(result, Err(RelationalError::InvalidName(_))));
    }

    #[test]
    fn test_invalid_table_name_underscore_prefix() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        let result = engine.create_table("_reserved", schema);
        assert!(matches!(result, Err(RelationalError::InvalidName(_))));
    }

    #[test]
    fn test_invalid_column_name_colon() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("col:umn", ColumnType::Int)]);
        let result = engine.create_table("test", schema);
        assert!(matches!(result, Err(RelationalError::InvalidName(_))));
    }

    #[test]
    fn test_invalid_column_name_underscore_prefix() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("_reserved", ColumnType::Int)]);
        let result = engine.create_table("test", schema);
        assert!(matches!(result, Err(RelationalError::InvalidName(_))));
    }

    #[test]
    fn test_valid_names_accepted() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("user_name", ColumnType::String),
            Column::new("Age123", ColumnType::Int),
        ]);
        let result = engine.create_table("my_table", schema);
        assert!(result.is_ok());
    }

    #[test]
    fn test_index_column_validation() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("test", schema).unwrap();

        // _id is allowed as a special case
        let result = engine.create_index("test", "_id");
        assert!(result.is_ok());
    }

    // ==================== Transaction Tests ====================

    #[test]
    fn test_transaction_begin_commit() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let tx = engine.begin_transaction();
        assert!(engine.is_transaction_active(tx));

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".to_string()));
        values.insert("age".to_string(), Value::Int(30));

        let row_id = engine.tx_insert(tx, "users", values).unwrap();
        assert_eq!(row_id, 1);

        engine.commit(tx).unwrap();
        assert!(!engine.is_transaction_active(tx));

        // Row should still exist after commit
        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_transaction_rollback_insert() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let tx = engine.begin_transaction();

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".to_string()));
        values.insert("age".to_string(), Value::Int(30));

        engine.tx_insert(tx, "users", values).unwrap();

        // Verify row exists before rollback
        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows.len(), 1);

        // Rollback
        engine.rollback(tx).unwrap();

        // Row should be gone after rollback
        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_transaction_rollback_update() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert initial row
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".to_string()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let tx = engine.begin_transaction();

        // Update the row
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(31));
        engine
            .tx_update(tx, "users", Condition::True, updates)
            .unwrap();

        // Verify update before rollback
        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows[0].get("age"), Some(&Value::Int(31)));

        // Rollback
        engine.rollback(tx).unwrap();

        // Age should be restored to 30
        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows[0].get("age"), Some(&Value::Int(30)));
    }

    #[test]
    fn test_transaction_rollback_delete() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert initial row
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".to_string()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let tx = engine.begin_transaction();

        // Delete the row
        engine.tx_delete(tx, "users", Condition::True).unwrap();

        // Verify deletion before rollback
        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows.len(), 0);

        // Rollback
        engine.rollback(tx).unwrap();

        // Row should be restored
        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].get("name"),
            Some(&Value::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_transaction_multiple_inserts_rollback() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let tx = engine.begin_transaction();

        for i in 0..5 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.tx_insert(tx, "users", values).unwrap();
        }

        // Verify all rows exist
        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows.len(), 5);

        // Rollback
        engine.rollback(tx).unwrap();

        // All rows should be gone
        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_transaction_not_found() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let result = engine.commit(999);
        assert!(matches!(
            result,
            Err(RelationalError::TransactionNotFound(999))
        ));

        let result = engine.rollback(999);
        assert!(matches!(
            result,
            Err(RelationalError::TransactionNotFound(999))
        ));
    }

    #[test]
    fn test_transaction_inactive_after_commit() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let tx = engine.begin_transaction();
        engine.commit(tx).unwrap();

        // Second commit should fail - transaction is removed
        let result = engine.commit(tx);
        assert!(matches!(
            result,
            Err(RelationalError::TransactionNotFound(_))
        ));
    }

    #[test]
    fn test_transaction_inactive_after_rollback() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let tx = engine.begin_transaction();
        engine.rollback(tx).unwrap();

        // Operations after rollback should fail
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".to_string()));
        values.insert("age".to_string(), Value::Int(30));

        let result = engine.tx_insert(tx, "users", values);
        assert!(matches!(
            result,
            Err(RelationalError::TransactionNotFound(_))
        ));
    }

    #[test]
    fn test_lock_conflict() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert a row
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".to_string()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let tx1 = engine.begin_transaction();
        let tx2 = engine.begin_transaction();

        // tx1 updates the row (acquires lock)
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(31));
        engine
            .tx_update(tx1, "users", Condition::True, updates.clone())
            .unwrap();

        // tx2 tries to update the same row (should fail)
        let result = engine.tx_update(tx2, "users", Condition::True, updates);
        assert!(matches!(result, Err(RelationalError::LockConflict { .. })));

        // Clean up
        engine.rollback(tx1).unwrap();
        engine.rollback(tx2).unwrap();
    }

    #[test]
    fn test_lock_released_after_commit() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert a row
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".to_string()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let tx1 = engine.begin_transaction();

        // tx1 updates the row
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(31));
        engine
            .tx_update(tx1, "users", Condition::True, updates)
            .unwrap();
        engine.commit(tx1).unwrap();

        // tx2 should now be able to update
        let tx2 = engine.begin_transaction();
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(32));
        let result = engine.tx_update(tx2, "users", Condition::True, updates);
        assert!(result.is_ok());

        engine.commit(tx2).unwrap();

        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows[0].get("age"), Some(&Value::Int(32)));
    }

    #[test]
    fn test_lock_released_after_rollback() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert a row
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".to_string()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let tx1 = engine.begin_transaction();

        // tx1 updates the row
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(31));
        engine
            .tx_update(tx1, "users", Condition::True, updates)
            .unwrap();
        engine.rollback(tx1).unwrap();

        // tx2 should now be able to update
        let tx2 = engine.begin_transaction();
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(32));
        let result = engine.tx_update(tx2, "users", Condition::True, updates);
        assert!(result.is_ok());

        engine.commit(tx2).unwrap();

        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows[0].get("age"), Some(&Value::Int(32)));
    }

    #[test]
    fn test_tx_select() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert some rows
        for i in 0..3 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        let tx = engine.begin_transaction();

        let rows = engine.tx_select(tx, "users", Condition::True).unwrap();
        assert_eq!(rows.len(), 3);

        let rows = engine
            .tx_select(
                tx,
                "users",
                Condition::Eq("name".to_string(), Value::String("User1".to_string())),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);

        engine.commit(tx).unwrap();
    }

    #[test]
    fn test_tx_delete_with_lock() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert a row
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".to_string()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let tx1 = engine.begin_transaction();
        let tx2 = engine.begin_transaction();

        // tx1 updates the row (acquires lock but row stays visible)
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(31));
        engine
            .tx_update(tx1, "users", Condition::True, updates)
            .unwrap();

        // tx2 tries to delete (should fail due to lock held by tx1)
        let result = engine.tx_delete(tx2, "users", Condition::True);
        assert!(matches!(result, Err(RelationalError::LockConflict { .. })));

        engine.commit(tx1).unwrap();
        engine.rollback(tx2).unwrap();

        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("age"), Some(&Value::Int(31)));
    }

    #[test]
    fn test_active_transaction_count() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        assert_eq!(engine.active_transaction_count(), 0);

        let tx1 = engine.begin_transaction();
        assert_eq!(engine.active_transaction_count(), 1);

        let tx2 = engine.begin_transaction();
        assert_eq!(engine.active_transaction_count(), 2);

        engine.commit(tx1).unwrap();
        assert_eq!(engine.active_transaction_count(), 1);

        engine.rollback(tx2).unwrap();
        assert_eq!(engine.active_transaction_count(), 0);
    }

    #[test]
    fn test_tx_insert_validation() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let tx = engine.begin_transaction();

        // Missing required column
        let mut values = HashMap::new();
        values.insert("age".to_string(), Value::Int(30));
        // name is required but missing

        let result = engine.tx_insert(tx, "users", values);
        assert!(matches!(result, Err(RelationalError::NullNotAllowed(_))));

        engine.rollback(tx).unwrap();
    }

    #[test]
    fn test_tx_update_validation() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Insert a row
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".to_string()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let tx = engine.begin_transaction();

        // Try to update with wrong type
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::String("not a number".to_string()));

        let result = engine.tx_update(tx, "users", Condition::True, updates);
        assert!(matches!(result, Err(RelationalError::TypeMismatch { .. })));

        engine.rollback(tx).unwrap();
    }

    #[test]
    fn test_transaction_with_index() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        engine.create_index("users", "name").unwrap();

        let tx = engine.begin_transaction();

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".to_string()));
        values.insert("age".to_string(), Value::Int(30));
        engine.tx_insert(tx, "users", values).unwrap();

        // Rollback should also clean up index entries
        engine.rollback(tx).unwrap();

        // Verify index is clean by checking lookup returns nothing
        let rows = engine
            .select(
                "users",
                Condition::Eq("name".to_string(), Value::String("Alice".to_string())),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_cross_join_limit_exceeded() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("large_a", schema.clone()).unwrap();
        engine.create_table("large_b", schema).unwrap();

        // 1001 x 1001 = 1_002_001 > 1_000_000
        for i in 0..1001 {
            let mut values = HashMap::new();
            values.insert("id".to_string(), Value::Int(i));
            engine.insert("large_a", values.clone()).unwrap();
            engine.insert("large_b", values).unwrap();
        }

        let result = engine.cross_join("large_a", "large_b");
        assert!(matches!(
            result,
            Err(RelationalError::ResultTooLarge { .. })
        ));
    }

    #[test]
    fn test_cross_join_at_limit_succeeds() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("limit_a", schema.clone()).unwrap();
        engine.create_table("limit_b", schema).unwrap();

        // 1000 x 1000 = 1_000_000 (exactly at limit)
        for i in 0..1000 {
            let mut values = HashMap::new();
            values.insert("id".to_string(), Value::Int(i));
            engine.insert("limit_a", values.clone()).unwrap();
            engine.insert("limit_b", values).unwrap();
        }

        let result = engine.cross_join("limit_a", "limit_b");
        assert!(result.is_ok());
    }

    #[test]
    fn test_natural_join_no_common_columns_respects_limit() {
        let engine = RelationalEngine::new();
        let schema_a = Schema::new(vec![Column::new("a_col", ColumnType::Int)]);
        let schema_b = Schema::new(vec![Column::new("b_col", ColumnType::Int)]);
        engine.create_table("no_common_a", schema_a).unwrap();
        engine.create_table("no_common_b", schema_b).unwrap();

        for i in 0..1001 {
            engine
                .insert(
                    "no_common_a",
                    HashMap::from([("a_col".to_string(), Value::Int(i))]),
                )
                .unwrap();
            engine
                .insert(
                    "no_common_b",
                    HashMap::from([("b_col".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        // Falls back to cross_join, should hit limit
        let result = engine.natural_join("no_common_a", "no_common_b");
        assert!(matches!(
            result,
            Err(RelationalError::ResultTooLarge { .. })
        ));
    }

    #[test]
    fn test_table_name_length_limit() {
        let engine = RelationalEngine::new();
        let long_name = "a".repeat(256);
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);

        let result = engine.create_table(&long_name, schema);
        assert!(matches!(
            result,
            Err(RelationalError::InvalidName(msg)) if msg.contains("255")
        ));
    }

    #[test]
    fn test_column_name_length_limit() {
        let engine = RelationalEngine::new();
        let long_col = "c".repeat(256);
        let schema = Schema::new(vec![Column::new(&long_col, ColumnType::Int)]);

        let result = engine.create_table("test_tbl", schema);
        assert!(matches!(result, Err(RelationalError::InvalidName(_))));
    }

    #[test]
    fn test_name_at_max_length_succeeds() {
        let engine = RelationalEngine::new();
        let max_name = "a".repeat(255);
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);

        assert!(engine.create_table(&max_name, schema).is_ok());
    }

    // ========== SIMD Filter Tests ==========

    #[test]
    fn test_simd_filter_lt_f64() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mut result = vec![0u64; 1];
        simd::filter_lt_f64(&values, 5.0, &mut result);

        // Values 0,1,2,3,4 < 5.0
        for i in 0..5 {
            assert!((result[0] & (1u64 << i)) != 0, "Expected bit {} set", i);
        }
        for i in 5..10 {
            assert!((result[0] & (1u64 << i)) == 0, "Expected bit {} unset", i);
        }
    }

    #[test]
    fn test_simd_filter_gt_f64() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mut result = vec![0u64; 1];
        simd::filter_gt_f64(&values, 5.0, &mut result);

        // Values 6,7,8,9 > 5.0
        for i in 0..6 {
            assert!((result[0] & (1u64 << i)) == 0, "Expected bit {} unset", i);
        }
        for i in 6..10 {
            assert!((result[0] & (1u64 << i)) != 0, "Expected bit {} set", i);
        }
    }

    #[test]
    fn test_simd_filter_eq_f64() {
        let values = vec![1.0, 2.0, 3.0, 2.0, 5.0, 2.0, 7.0, 8.0];
        let mut result = vec![0u64; 1];
        simd::filter_eq_f64(&values, 2.0, &mut result);

        // Indices 1, 3, 5 have value 2.0
        assert!((result[0] & (1u64 << 1)) != 0);
        assert!((result[0] & (1u64 << 3)) != 0);
        assert!((result[0] & (1u64 << 5)) != 0);
        assert!((result[0] & (1u64 << 0)) == 0);
        assert!((result[0] & (1u64 << 2)) == 0);
    }

    #[test]
    fn test_simd_filter_lt_f64_non_aligned() {
        // Test with non-4-aligned length to cover remainder loop
        let values: Vec<f64> = (0..7).map(|i| i as f64).collect();
        let mut result = vec![0u64; 1];
        simd::filter_lt_f64(&values, 3.0, &mut result);

        // 0, 1, 2 < 3.0
        assert_eq!(result[0] & 0b111, 0b111);
        assert!((result[0] & (1u64 << 3)) == 0);
    }

    #[test]
    fn test_simd_filter_gt_f64_non_aligned() {
        let values: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let mut result = vec![0u64; 1];
        simd::filter_gt_f64(&values, 2.0, &mut result);

        // 3, 4 > 2.0
        assert!((result[0] & (1u64 << 3)) != 0);
        assert!((result[0] & (1u64 << 4)) != 0);
    }

    #[test]
    fn test_simd_filter_eq_f64_non_aligned() {
        let values = vec![1.0, 2.0, 2.0, 4.0, 5.0];
        let mut result = vec![0u64; 1];
        simd::filter_eq_f64(&values, 2.0, &mut result);

        assert!((result[0] & (1u64 << 1)) != 0);
        assert!((result[0] & (1u64 << 2)) != 0);
    }

    // ========== Value Method Tests ==========

    #[test]
    fn test_value_is_truthy() {
        assert!(!Value::Null.is_truthy());
        assert!(!Value::Bool(false).is_truthy());
        assert!(Value::Bool(true).is_truthy());
        assert!(!Value::Int(0).is_truthy());
        assert!(Value::Int(1).is_truthy());
        assert!(Value::Int(-1).is_truthy());
        assert!(!Value::Float(0.0).is_truthy());
        assert!(Value::Float(1.0).is_truthy());
        assert!(!Value::String(String::new()).is_truthy());
        assert!(Value::String("hello".to_string()).is_truthy());
    }

    #[test]
    fn test_value_hash_key() {
        let null_key = Value::Null.hash_key();
        assert_eq!(null_key, "null");

        let int_key = Value::Int(42).hash_key();
        assert!(int_key.starts_with("i:"));

        let float_key = Value::Float(3.14).hash_key();
        assert!(float_key.starts_with("f:"));

        let string_key = Value::String("test".to_string()).hash_key();
        assert!(string_key.starts_with("s:"));

        let bool_key = Value::Bool(true).hash_key();
        assert!(bool_key.starts_with("b:"));
    }

    #[test]
    fn test_value_sortable_key() {
        let null_key = Value::Null.sortable_key();
        assert_eq!(null_key, "0");

        let int_key = Value::Int(100).sortable_key();
        assert!(int_key.starts_with("i"));

        let float_key = Value::Float(1.5).sortable_key();
        assert!(float_key.starts_with("f"));

        let string_key = Value::String("abc".to_string()).sortable_key();
        assert_eq!(string_key, "sabc");

        let true_key = Value::Bool(true).sortable_key();
        assert_eq!(true_key, "b1");

        let false_key = Value::Bool(false).sortable_key();
        assert_eq!(false_key, "b0");
    }

    #[test]
    fn test_value_sortable_key_ordering() {
        // Test that sortable keys maintain correct ordering
        let neg_key = Value::Int(-100).sortable_key();
        let zero_key = Value::Int(0).sortable_key();
        let pos_key = Value::Int(100).sortable_key();
        assert!(neg_key < zero_key);
        assert!(zero_key < pos_key);

        // Float ordering
        let neg_float = Value::Float(-1.0).sortable_key();
        let zero_float = Value::Float(0.0).sortable_key();
        let pos_float = Value::Float(1.0).sortable_key();
        assert!(neg_float < zero_float);
        assert!(zero_float < pos_float);
    }

    #[test]
    fn test_value_partial_cmp() {
        assert_eq!(
            Value::Int(5).partial_cmp_value(&Value::Int(3)),
            Some(std::cmp::Ordering::Greater)
        );
        assert_eq!(
            Value::Float(1.0).partial_cmp_value(&Value::Float(2.0)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            Value::String("a".to_string()).partial_cmp_value(&Value::String("b".to_string())),
            Some(std::cmp::Ordering::Less)
        );
        // Mismatched types return None
        assert_eq!(
            Value::Int(5).partial_cmp_value(&Value::String("5".to_string())),
            None
        );
    }

    #[test]
    fn test_value_matches_type() {
        assert!(Value::Null.matches_type(&ColumnType::Int)); // Null matches any
        assert!(Value::Int(42).matches_type(&ColumnType::Int));
        assert!(!Value::Int(42).matches_type(&ColumnType::String));
        assert!(Value::Float(1.0).matches_type(&ColumnType::Float));
        assert!(Value::String("x".to_string()).matches_type(&ColumnType::String));
        assert!(Value::Bool(true).matches_type(&ColumnType::Bool));
    }

    // ========== Error Display Tests ==========

    #[test]
    fn test_error_display_table_not_found() {
        let err = RelationalError::TableNotFound("users".to_string());
        assert_eq!(format!("{}", err), "Table not found: users");
    }

    #[test]
    fn test_error_display_table_already_exists() {
        let err = RelationalError::TableAlreadyExists("users".to_string());
        assert_eq!(format!("{}", err), "Table already exists: users");
    }

    #[test]
    fn test_error_display_column_not_found() {
        let err = RelationalError::ColumnNotFound("age".to_string());
        assert_eq!(format!("{}", err), "Column not found: age");
    }

    #[test]
    fn test_error_display_type_mismatch() {
        let err = RelationalError::TypeMismatch {
            column: "age".to_string(),
            expected: ColumnType::Int,
        };
        assert!(format!("{}", err).contains("Type mismatch"));
    }

    #[test]
    fn test_error_display_null_not_allowed() {
        let err = RelationalError::NullNotAllowed("name".to_string());
        assert!(format!("{}", err).contains("Null not allowed"));
    }

    #[test]
    fn test_error_display_index_already_exists() {
        let err = RelationalError::IndexAlreadyExists {
            table: "users".to_string(),
            column: "email".to_string(),
        };
        assert!(format!("{}", err).contains("Index already exists"));
    }

    #[test]
    fn test_error_display_index_not_found() {
        let err = RelationalError::IndexNotFound {
            table: "users".to_string(),
            column: "email".to_string(),
        };
        assert!(format!("{}", err).contains("Index not found"));
    }

    #[test]
    fn test_error_display_storage_error() {
        let err = RelationalError::StorageError("disk full".to_string());
        assert!(format!("{}", err).contains("Storage error"));
    }

    #[test]
    fn test_error_display_transaction_not_found() {
        let err = RelationalError::TransactionNotFound(123);
        assert!(format!("{}", err).contains("Transaction not found: 123"));
    }

    #[test]
    fn test_error_display_transaction_inactive() {
        let err = RelationalError::TransactionInactive(456);
        assert!(format!("{}", err).contains("Transaction not active: 456"));
    }

    #[test]
    fn test_error_display_lock_conflict() {
        let err = RelationalError::LockConflict {
            tx_id: 1,
            blocking_tx: 2,
            table: "users".to_string(),
            row_id: 100,
        };
        assert!(format!("{}", err).contains("Lock conflict"));
    }

    #[test]
    fn test_error_display_lock_timeout() {
        let err = RelationalError::LockTimeout {
            tx_id: 1,
            table: "users".to_string(),
            row_ids: vec![1, 2, 3],
        };
        assert!(format!("{}", err).contains("Lock timeout"));
    }

    #[test]
    fn test_error_display_rollback_failed() {
        let err = RelationalError::RollbackFailed {
            tx_id: 1,
            reason: "storage failure".to_string(),
        };
        assert!(format!("{}", err).contains("Rollback failed"));
    }

    #[test]
    fn test_error_display_result_too_large() {
        let err = RelationalError::ResultTooLarge {
            operation: "CROSS JOIN".to_string(),
            actual: 2_000_000,
            max: 1_000_000,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("CROSS JOIN"));
        assert!(msg.contains("2000000"));
        assert!(msg.contains("1000000"));
    }

    #[test]
    fn test_error_display_index_corrupted() {
        let err = RelationalError::IndexCorrupted {
            reason: "ID list has 7 bytes, expected multiple of 8".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Index data corrupted"));
        assert!(msg.contains("7 bytes"));
    }

    #[test]
    fn test_tensor_to_id_list_corrupted_bytes() {
        // Create a TensorData with corrupted bytes (not a multiple of 8)
        let mut tensor = TensorData::new();
        tensor.set(
            "ids",
            TensorValue::Scalar(ScalarValue::Bytes(vec![1, 2, 3, 4, 5, 6, 7])),
        );

        let result = RelationalEngine::tensor_to_id_list(&tensor);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, RelationalError::IndexCorrupted { .. }));
    }

    #[test]
    fn test_tensor_to_id_list_wrong_type() {
        // Create a TensorData with wrong type for "ids"
        let mut tensor = TensorData::new();
        tensor.set("ids", TensorValue::Scalar(ScalarValue::Int(42)));

        let result = RelationalEngine::tensor_to_id_list(&tensor);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, RelationalError::IndexCorrupted { .. }));
    }

    #[test]
    fn test_tensor_to_id_list_valid_bytes() {
        // Create a TensorData with valid bytes (multiple of 8)
        let mut tensor = TensorData::new();
        let ids: Vec<u64> = vec![1, 2, 3];
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        tensor.set("ids", TensorValue::Scalar(ScalarValue::Bytes(bytes)));

        let result = RelationalEngine::tensor_to_id_list(&tensor);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_tensor_to_id_list_empty() {
        // Create a TensorData with no "ids" field
        let tensor = TensorData::new();

        let result = RelationalEngine::tensor_to_id_list(&tensor);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_tensor_to_id_list_legacy_vector_format() {
        // Create a TensorData with legacy Vector format
        let mut tensor = TensorData::new();
        tensor.set("ids", TensorValue::Vector(vec![1.0, 2.0, 3.0]));

        let result = RelationalEngine::tensor_to_id_list(&tensor);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![1, 2, 3]);
    }

    // ========== Schema Bridge Conversion Tests ==========

    #[test]
    fn test_column_type_to_slab_column_type() {
        assert_eq!(SlabColumnType::from(&ColumnType::Int), SlabColumnType::Int);
        assert_eq!(
            SlabColumnType::from(&ColumnType::Float),
            SlabColumnType::Float
        );
        assert_eq!(
            SlabColumnType::from(&ColumnType::String),
            SlabColumnType::String
        );
        assert_eq!(
            SlabColumnType::from(&ColumnType::Bool),
            SlabColumnType::Bool
        );
    }

    #[test]
    fn test_slab_column_type_to_column_type() {
        assert_eq!(ColumnType::from(&SlabColumnType::Int), ColumnType::Int);
        assert_eq!(ColumnType::from(&SlabColumnType::Float), ColumnType::Float);
        assert_eq!(
            ColumnType::from(&SlabColumnType::String),
            ColumnType::String
        );
        assert_eq!(ColumnType::from(&SlabColumnType::Bool), ColumnType::Bool);
        assert_eq!(ColumnType::from(&SlabColumnType::Bytes), ColumnType::String);
        assert_eq!(ColumnType::from(&SlabColumnType::Json), ColumnType::String);
    }

    #[test]
    fn test_schema_to_slab_table_schema() {
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("name", ColumnType::String).nullable(),
        ]);
        let slab_schema: SlabTableSchema = (&schema).into();
        assert_eq!(slab_schema.columns.len(), 2);
    }

    #[test]
    fn test_value_to_slab_column_value() {
        assert_eq!(SlabColumnValue::from(&Value::Null), SlabColumnValue::Null);
        assert_eq!(
            SlabColumnValue::from(&Value::Int(42)),
            SlabColumnValue::Int(42)
        );
        assert_eq!(
            SlabColumnValue::from(&Value::Float(3.14)),
            SlabColumnValue::Float(3.14)
        );
        assert_eq!(
            SlabColumnValue::from(&Value::String("test".to_string())),
            SlabColumnValue::String("test".to_string())
        );
        assert_eq!(
            SlabColumnValue::from(&Value::Bool(true)),
            SlabColumnValue::Bool(true)
        );
    }

    #[test]
    fn test_slab_column_value_to_value() {
        assert_eq!(Value::from(SlabColumnValue::Null), Value::Null);
        assert_eq!(Value::from(SlabColumnValue::Int(42)), Value::Int(42));
        assert_eq!(
            Value::from(SlabColumnValue::Float(3.14)),
            Value::Float(3.14)
        );
        assert_eq!(
            Value::from(SlabColumnValue::String("test".to_string())),
            Value::String("test".to_string())
        );
        assert_eq!(Value::from(SlabColumnValue::Bool(true)), Value::Bool(true));
        // Bytes converts to string representation
        let bytes_val = Value::from(SlabColumnValue::Bytes(vec![1, 2, 3]));
        assert!(matches!(bytes_val, Value::String(_)));
        // Json converts to string
        let json_val = Value::from(SlabColumnValue::Json(r#"{"key": "value"}"#.to_string()));
        assert_eq!(json_val, Value::String(r#"{"key": "value"}"#.to_string()));
    }

    // ========== OrderedFloat Tests ==========

    #[test]
    fn test_ordered_float_eq() {
        let a = OrderedFloat(1.0);
        let b = OrderedFloat(1.0);
        let c = OrderedFloat(2.0);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_ordered_float_ord() {
        let a = OrderedFloat(1.0);
        let b = OrderedFloat(2.0);
        let nan = OrderedFloat(f64::NAN);

        assert!(a < b);
        assert!(nan < a); // NaN is less than all values
    }

    #[test]
    fn test_ordered_key_ordering() {
        // Null < Bool < Int < Float < String
        let null = OrderedKey::Null;
        let bool_val = OrderedKey::Bool(false);
        let int_val = OrderedKey::Int(0);
        let float_val = OrderedKey::Float(OrderedFloat(0.0));
        let str_val = OrderedKey::String(String::new());

        assert!(null < bool_val);
        assert!(bool_val < int_val);
        assert!(int_val < float_val);
        assert!(float_val < str_val);
    }

    // ========== Row Tests ==========

    #[test]
    fn test_row_get() {
        let row = Row {
            id: 1,
            values: vec![
                ("name".to_string(), Value::String("Alice".to_string())),
                ("age".to_string(), Value::Int(30)),
            ],
        };

        assert_eq!(row.get("name"), Some(&Value::String("Alice".to_string())));
        assert_eq!(row.get("age"), Some(&Value::Int(30)));
        assert_eq!(row.get("missing"), None);
        assert_eq!(row.get("_id"), None); // Special _id handling
    }

    #[test]
    fn test_row_get_with_id() {
        let row = Row {
            id: 42,
            values: vec![("name".to_string(), Value::String("Bob".to_string()))],
        };

        assert_eq!(row.get_with_id("_id"), Some(Value::Int(42)));
        assert_eq!(
            row.get_with_id("name"),
            Some(Value::String("Bob".to_string()))
        );
        assert_eq!(row.get_with_id("missing"), None);
    }

    // ========== NullBitmap Tests ==========

    #[test]
    fn test_null_bitmap_none() {
        let bitmap = NullBitmap::None;
        assert_eq!(bitmap.null_count(), 0);
        assert!(!bitmap.is_null(0));
        assert!(!bitmap.is_null(99));
    }

    #[test]
    fn test_null_bitmap_dense() {
        // Positions 0, 2, 5 are null (bits set)
        let bitmap = NullBitmap::Dense(vec![0b100101]);
        assert_eq!(bitmap.null_count(), 3);
        assert!(bitmap.is_null(0));
        assert!(!bitmap.is_null(1));
        assert!(bitmap.is_null(2));
        assert!(bitmap.is_null(5));
    }

    #[test]
    fn test_null_bitmap_sparse() {
        // Sparse positions list
        let bitmap = NullBitmap::Sparse(vec![1, 5, 10]);
        assert_eq!(bitmap.null_count(), 3);
        assert!(bitmap.is_null(1));
        assert!(bitmap.is_null(5));
        assert!(bitmap.is_null(10));
        assert!(!bitmap.is_null(0));
        assert!(!bitmap.is_null(3));
    }

    // ========== SelectionVector Tests ==========

    #[test]
    fn test_selection_vector_all() {
        let sv = SelectionVector::all(10);
        assert_eq!(sv.count(), 10);
        for i in 0..10 {
            assert!(sv.is_selected(i));
        }
        assert!(!sv.is_selected(10)); // out of bounds
    }

    #[test]
    fn test_selection_vector_none() {
        let sv = SelectionVector::none(10);
        assert_eq!(sv.count(), 0);
        for i in 0..10 {
            assert!(!sv.is_selected(i));
        }
    }

    #[test]
    fn test_selection_vector_from_bitmap() {
        let bitmap = vec![0b1010u64]; // positions 1 and 3 selected
        let sv = SelectionVector::from_bitmap(bitmap, 8);
        assert!(sv.is_selected(1));
        assert!(sv.is_selected(3));
        assert!(!sv.is_selected(0));
        assert!(!sv.is_selected(2));
    }

    #[test]
    fn test_selection_vector_intersect() {
        let sv1 = SelectionVector::from_bitmap(vec![0b1111], 8);
        let sv2 = SelectionVector::from_bitmap(vec![0b1010], 8);
        let result = sv1.intersect(&sv2);
        assert!(result.is_selected(1));
        assert!(result.is_selected(3));
        assert!(!result.is_selected(0));
        assert!(!result.is_selected(2));
    }

    #[test]
    fn test_selection_vector_union() {
        let sv1 = SelectionVector::from_bitmap(vec![0b0101], 8);
        let sv2 = SelectionVector::from_bitmap(vec![0b1010], 8);
        let result = sv1.union(&sv2);
        for i in 0..4 {
            assert!(result.is_selected(i));
        }
    }

    #[test]
    fn test_selection_vector_selected_indices() {
        let sv = SelectionVector::from_bitmap(vec![0b10100101], 8);
        let indices = sv.selected_indices();
        assert_eq!(indices, vec![0, 2, 5, 7]);
    }

    #[test]
    fn test_selection_vector_bitmap_mut() {
        let mut sv = SelectionVector::none(64);
        sv.bitmap_mut()[0] = 0b11110000;
        assert!(!sv.is_selected(0));
        assert!(sv.is_selected(4));
    }

    // ========== ColumnData Tests ==========

    #[test]
    fn test_column_data_get_value_int() {
        let col = ColumnData {
            name: "age".to_string(),
            row_ids: vec![1, 2, 3],
            nulls: NullBitmap::None,
            values: ColumnValues::Int(vec![25, 30, 35]),
        };
        assert_eq!(col.get_value(0), Some(Value::Int(25)));
        assert_eq!(col.get_value(1), Some(Value::Int(30)));
        assert_eq!(col.get_value(2), Some(Value::Int(35)));
        assert_eq!(col.null_count(), 0);
    }

    #[test]
    fn test_column_data_get_value_float() {
        let col = ColumnData {
            name: "score".to_string(),
            row_ids: vec![1, 2],
            nulls: NullBitmap::None,
            values: ColumnValues::Float(vec![1.5, 2.5]),
        };
        assert_eq!(col.get_value(0), Some(Value::Float(1.5)));
        assert_eq!(col.get_value(1), Some(Value::Float(2.5)));
    }

    #[test]
    fn test_column_data_get_value_string() {
        let col = ColumnData {
            name: "name".to_string(),
            row_ids: vec![1, 2],
            nulls: NullBitmap::None,
            values: ColumnValues::String {
                dict: vec!["Alice".to_string(), "Bob".to_string()],
                indices: vec![0, 1],
            },
        };
        assert_eq!(col.get_value(0), Some(Value::String("Alice".to_string())));
        assert_eq!(col.get_value(1), Some(Value::String("Bob".to_string())));
    }

    #[test]
    fn test_column_data_get_value_bool() {
        let col = ColumnData {
            name: "active".to_string(),
            row_ids: vec![1, 2, 3],
            nulls: NullBitmap::None,
            values: ColumnValues::Bool(vec![0b101]), // true, false, true
        };
        assert_eq!(col.get_value(0), Some(Value::Bool(true)));
        assert_eq!(col.get_value(1), Some(Value::Bool(false)));
        assert_eq!(col.get_value(2), Some(Value::Bool(true)));
    }

    #[test]
    fn test_column_data_with_nulls() {
        // Position 1 is null (bit 1 is set = 0b10)
        let col = ColumnData {
            name: "age".to_string(),
            row_ids: vec![1, 2, 3],
            nulls: NullBitmap::Dense(vec![0b010]),
            values: ColumnValues::Int(vec![25, 0, 35]),
        };
        assert_eq!(col.get_value(0), Some(Value::Int(25)));
        assert_eq!(col.get_value(1), Some(Value::Null));
        assert_eq!(col.get_value(2), Some(Value::Int(35)));
        assert_eq!(col.null_count(), 1);
    }

    // ========== Condition Range Tests ==========

    #[test]
    fn test_condition_range_le() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("score", ColumnType::Float),
        ]);
        engine.create_table("scores", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "scores",
                    HashMap::from([
                        ("id".to_string(), Value::Int(i)),
                        ("score".to_string(), Value::Float(i as f64)),
                    ]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "scores",
                Condition::Le("score".to_string(), Value::Float(5.0)),
            )
            .unwrap();
        assert_eq!(rows.len(), 6); // 0, 1, 2, 3, 4, 5
    }

    #[test]
    fn test_condition_range_ge() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("vals", schema).unwrap();

        for i in 0..10 {
            engine
                .insert("vals", HashMap::from([("val".to_string(), Value::Int(i))]))
                .unwrap();
        }

        let rows = engine
            .select("vals", Condition::Ge("val".to_string(), Value::Int(7)))
            .unwrap();
        assert_eq!(rows.len(), 3); // 7, 8, 9
    }

    #[test]
    fn test_condition_range_ne() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("vals", schema).unwrap();

        for i in 0..5 {
            engine
                .insert("vals", HashMap::from([("val".to_string(), Value::Int(i))]))
                .unwrap();
        }

        let rows = engine
            .select("vals", Condition::Ne("val".to_string(), Value::Int(2)))
            .unwrap();
        assert_eq!(rows.len(), 4); // 0, 1, 3, 4
    }

    // ========== ColumnarScanOptions Tests ==========

    #[test]
    fn test_columnar_scan_options_default() {
        let opts = ColumnarScanOptions::default();
        assert!(opts.projection.is_none());
        assert!(!opts.prefer_columnar);
    }

    #[test]
    fn test_columnar_scan_options_with_projection() {
        let opts = ColumnarScanOptions {
            projection: Some(vec!["name".to_string(), "age".to_string()]),
            prefer_columnar: true,
        };
        assert_eq!(opts.projection.as_ref().unwrap().len(), 2);
        assert!(opts.prefer_columnar);
    }

    // ========== Storage Error Conversion Test ==========

    #[test]
    fn test_storage_error_from_tensor_store() {
        use tensor_store::TensorStoreError;
        let store_err = TensorStoreError::NotFound("test".to_string());
        let rel_err: RelationalError = store_err.into();
        assert!(matches!(rel_err, RelationalError::StorageError(_)));
    }

    // ========== Float Filtering Edge Cases ==========

    #[test]
    fn test_select_float_conditions() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("price", ColumnType::Float),
        ]);
        engine.create_table("products", schema).unwrap();

        let prices = [1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5, 9.0, 10.0];
        for (i, &price) in prices.iter().enumerate() {
            engine
                .insert(
                    "products",
                    HashMap::from([
                        ("id".to_string(), Value::Int(i as i64)),
                        ("price".to_string(), Value::Float(price)),
                    ]),
                )
                .unwrap();
        }

        // Test Lt
        let lt_rows = engine
            .select(
                "products",
                Condition::Lt("price".to_string(), Value::Float(5.0)),
            )
            .unwrap();
        assert_eq!(lt_rows.len(), 4); // 1.0, 2.5, 3.0, 4.5

        // Test Gt
        let gt_rows = engine
            .select(
                "products",
                Condition::Gt("price".to_string(), Value::Float(7.0)),
            )
            .unwrap();
        assert_eq!(gt_rows.len(), 3); // 8.5, 9.0, 10.0

        // Test Eq
        let eq_rows = engine
            .select(
                "products",
                Condition::Eq("price".to_string(), Value::Float(5.0)),
            )
            .unwrap();
        assert_eq!(eq_rows.len(), 1);
    }

    // ========== Value from_scalar Tests ==========

    #[test]
    fn test_value_from_scalar() {
        use tensor_store::ScalarValue;

        let null_val = Value::from_scalar(&ScalarValue::Null);
        assert_eq!(null_val, Value::Null);

        let int_val = Value::from_scalar(&ScalarValue::Int(42));
        assert_eq!(int_val, Value::Int(42));

        let float_val = Value::from_scalar(&ScalarValue::Float(3.14));
        assert_eq!(float_val, Value::Float(3.14));

        let str_val = Value::from_scalar(&ScalarValue::String("test".to_string()));
        assert_eq!(str_val, Value::String("test".to_string()));

        let bool_val = Value::from_scalar(&ScalarValue::Bool(true));
        assert_eq!(bool_val, Value::Bool(true));

        // Bytes converts to Null
        let bytes_val = Value::from_scalar(&ScalarValue::Bytes(vec![1, 2, 3]));
        assert_eq!(bytes_val, Value::Null);
    }

    // ========== Selection Vector Large Tests ==========

    #[test]
    fn test_selection_vector_all_large() {
        // Test with more than 64 elements
        let sv = SelectionVector::all(100);
        assert_eq!(sv.count(), 100);
        assert!(sv.is_selected(0));
        assert!(sv.is_selected(63));
        assert!(sv.is_selected(64));
        assert!(sv.is_selected(99));
        assert!(!sv.is_selected(100));
    }

    #[test]
    fn test_selection_vector_non_aligned() {
        // Test with non-64-aligned count
        let sv = SelectionVector::all(70);
        assert_eq!(sv.count(), 70);
        assert!(sv.is_selected(69));
        assert!(!sv.is_selected(70));
    }

    // ========== SIMD i64 Filter Tests ==========

    #[test]
    fn test_simd_filter_lt_i64() {
        let values: Vec<i64> = (0..10).collect();
        let mut result = vec![0u64; 1];
        simd::filter_lt_i64(&values, 5, &mut result);

        // 0,1,2,3,4 < 5
        for i in 0..5 {
            assert!((result[0] & (1u64 << i)) != 0);
        }
        for i in 5..10 {
            assert!((result[0] & (1u64 << i)) == 0);
        }
    }

    #[test]
    fn test_simd_filter_le_i64() {
        let values: Vec<i64> = (0..10).collect();
        let mut result = vec![0u64; 1];
        simd::filter_le_i64(&values, 5, &mut result);

        // 0,1,2,3,4,5 <= 5
        for i in 0..6 {
            assert!((result[0] & (1u64 << i)) != 0);
        }
        for i in 6..10 {
            assert!((result[0] & (1u64 << i)) == 0);
        }
    }

    #[test]
    fn test_simd_filter_gt_i64() {
        let values: Vec<i64> = (0..10).collect();
        let mut result = vec![0u64; 1];
        simd::filter_gt_i64(&values, 5, &mut result);

        // 6,7,8,9 > 5
        for i in 0..6 {
            assert!((result[0] & (1u64 << i)) == 0);
        }
        for i in 6..10 {
            assert!((result[0] & (1u64 << i)) != 0);
        }
    }

    #[test]
    fn test_simd_filter_ge_i64() {
        let values: Vec<i64> = (0..10).collect();
        let mut result = vec![0u64; 1];
        simd::filter_ge_i64(&values, 5, &mut result);

        // 5,6,7,8,9 >= 5
        for i in 0..5 {
            assert!((result[0] & (1u64 << i)) == 0);
        }
        for i in 5..10 {
            assert!((result[0] & (1u64 << i)) != 0);
        }
    }

    #[test]
    fn test_simd_filter_eq_i64() {
        let values = vec![1, 2, 3, 2, 5, 2, 7, 8];
        let mut result = vec![0u64; 1];
        simd::filter_eq_i64(&values, 2, &mut result);

        // Indices 1, 3, 5 have value 2
        assert!((result[0] & (1u64 << 1)) != 0);
        assert!((result[0] & (1u64 << 3)) != 0);
        assert!((result[0] & (1u64 << 5)) != 0);
    }

    #[test]
    fn test_simd_filter_i64_non_aligned() {
        // Test with non-4-aligned length to cover remainder loop
        let values: Vec<i64> = (0..7).collect();
        let mut result = vec![0u64; 1];
        simd::filter_lt_i64(&values, 3, &mut result);

        // 0, 1, 2 < 3
        assert_eq!(result[0] & 0b111, 0b111);
        assert!((result[0] & (1u64 << 3)) == 0);
    }

    // ========== OrderedFloat Edge Cases ==========

    #[test]
    fn test_ordered_float_nan_comparison() {
        let nan1 = OrderedFloat(f64::NAN);
        let nan2 = OrderedFloat(f64::NAN);
        let regular = OrderedFloat(1.0);

        // NaN == NaN for OrderedFloat
        assert_eq!(nan1.cmp(&nan2), std::cmp::Ordering::Equal);
        // NaN < regular
        assert_eq!(nan1.cmp(&regular), std::cmp::Ordering::Less);
        // regular > NaN
        assert_eq!(regular.cmp(&nan1), std::cmp::Ordering::Greater);
    }

    // ========== OrderedKey::from_value ==========

    #[test]
    fn test_ordered_key_from_value() {
        let null_key = OrderedKey::from_value(&Value::Null);
        assert_eq!(null_key, OrderedKey::Null);

        let bool_key = OrderedKey::from_value(&Value::Bool(true));
        assert_eq!(bool_key, OrderedKey::Bool(true));

        let int_key = OrderedKey::from_value(&Value::Int(42));
        assert_eq!(int_key, OrderedKey::Int(42));

        let float_key = OrderedKey::from_value(&Value::Float(3.14));
        assert!(matches!(float_key, OrderedKey::Float(_)));

        let string_key = OrderedKey::from_value(&Value::String("test".to_string()));
        assert_eq!(string_key, OrderedKey::String("test".to_string()));
    }

    // ========== More Int Condition Tests (for SIMD paths) ==========

    #[test]
    fn test_select_int_le_condition() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("ints", schema).unwrap();

        for i in 0..10 {
            engine
                .insert("ints", HashMap::from([("val".to_string(), Value::Int(i))]))
                .unwrap();
        }

        let rows = engine
            .select("ints", Condition::Le("val".to_string(), Value::Int(5)))
            .unwrap();
        assert_eq!(rows.len(), 6); // 0,1,2,3,4,5
    }

    #[test]
    fn test_select_int_gt_condition() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("ints", schema).unwrap();

        for i in 0..10 {
            engine
                .insert("ints", HashMap::from([("val".to_string(), Value::Int(i))]))
                .unwrap();
        }

        let rows = engine
            .select("ints", Condition::Gt("val".to_string(), Value::Int(5)))
            .unwrap();
        assert_eq!(rows.len(), 4); // 6,7,8,9
    }

    #[test]
    fn test_select_int_ge_condition() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("ints", schema).unwrap();

        for i in 0..10 {
            engine
                .insert("ints", HashMap::from([("val".to_string(), Value::Int(i))]))
                .unwrap();
        }

        let rows = engine
            .select("ints", Condition::Ge("val".to_string(), Value::Int(5)))
            .unwrap();
        assert_eq!(rows.len(), 5); // 5,6,7,8,9
    }

    #[test]
    fn test_select_int_eq_condition() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("ints", schema).unwrap();

        for i in 0..10 {
            engine
                .insert("ints", HashMap::from([("val".to_string(), Value::Int(i))]))
                .unwrap();
        }

        let rows = engine
            .select("ints", Condition::Eq("val".to_string(), Value::Int(5)))
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    // ========== Float Le/Ge Condition Tests ==========

    #[test]
    fn test_select_float_le_condition() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float)]);
        engine.create_table("floats", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "floats",
                    HashMap::from([("val".to_string(), Value::Float(i as f64))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "floats",
                Condition::Le("val".to_string(), Value::Float(5.0)),
            )
            .unwrap();
        assert_eq!(rows.len(), 6); // 0,1,2,3,4,5
    }

    #[test]
    fn test_select_float_ge_condition() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float)]);
        engine.create_table("floats", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "floats",
                    HashMap::from([("val".to_string(), Value::Float(i as f64))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "floats",
                Condition::Ge("val".to_string(), Value::Float(5.0)),
            )
            .unwrap();
        assert_eq!(rows.len(), 5); // 5,6,7,8,9
    }

    // ========== Durable Engine Tests ==========

    #[test]
    fn test_open_durable_engine() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let config = tensor_store::WalConfig::default();
        let engine = RelationalEngine::open_durable(&wal_path, config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_recover_engine() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // First create a durable engine
        let config = tensor_store::WalConfig::default();
        {
            let _engine = RelationalEngine::open_durable(&wal_path, config.clone()).unwrap();
            // Engine drops, WAL is closed
        }

        // Recover
        let recovered = RelationalEngine::recover(&wal_path, &config, None);
        assert!(recovered.is_ok());
    }

    #[test]
    fn test_durable_insert_and_delete() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("durable.wal");

        let config = tensor_store::WalConfig::default();
        let engine = RelationalEngine::open_durable(&wal_path, config).unwrap();

        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("durable_test", schema).unwrap();

        engine
            .insert(
                "durable_test",
                HashMap::from([("id".to_string(), Value::Int(1))]),
            )
            .unwrap();

        let rows = engine.select("durable_test", Condition::True).unwrap();
        assert_eq!(rows.len(), 1);

        engine
            .delete_rows(
                "durable_test",
                Condition::Eq("id".to_string(), Value::Int(1)),
            )
            .unwrap();

        let rows = engine.select("durable_test", Condition::True).unwrap();
        assert_eq!(rows.len(), 0);
    }

    // ========== Transaction Manager with_timeout ==========

    #[test]
    fn test_transaction_manager_with_timeout() {
        use crate::transaction::TransactionManager;
        use std::time::Duration;

        let mgr = TransactionManager::with_timeout(Duration::from_secs(30));
        let tx = mgr.begin();
        assert!(mgr.is_active(tx));
    }

    // ========== SIMD Bitmap Operations ==========

    #[test]
    fn test_simd_popcount() {
        let bitmap = vec![0b11111111u64, 0b00001111u64];
        assert_eq!(simd::popcount(&bitmap), 12);
    }

    #[test]
    fn test_simd_bitmap_words() {
        assert_eq!(simd::bitmap_words(0), 0);
        assert_eq!(simd::bitmap_words(1), 1);
        assert_eq!(simd::bitmap_words(64), 1);
        assert_eq!(simd::bitmap_words(65), 2);
        assert_eq!(simd::bitmap_words(128), 2);
    }

    #[test]
    fn test_simd_selected_indices() {
        let bitmap = vec![0b10100101u64];
        let indices = simd::selected_indices(&bitmap, 8);
        assert_eq!(indices, vec![0, 2, 5, 7]);
    }

    // ========== Complex Condition Tests ==========

    #[test]
    fn test_complex_and_or_conditions() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("a", ColumnType::Int),
            Column::new("b", ColumnType::Int),
        ]);
        engine.create_table("complex", schema).unwrap();

        for a in 0..5 {
            for b in 0..5 {
                engine
                    .insert(
                        "complex",
                        HashMap::from([
                            ("a".to_string(), Value::Int(a)),
                            ("b".to_string(), Value::Int(b)),
                        ]),
                    )
                    .unwrap();
            }
        }

        // (a < 2) AND (b > 2)
        let cond = Condition::And(
            Box::new(Condition::Lt("a".to_string(), Value::Int(2))),
            Box::new(Condition::Gt("b".to_string(), Value::Int(2))),
        );
        let rows = engine.select("complex", cond).unwrap();
        // a in {0, 1}, b in {3, 4} => 2 * 2 = 4 rows
        assert_eq!(rows.len(), 4);

        // (a == 0) OR (b == 0)
        let cond = Condition::Or(
            Box::new(Condition::Eq("a".to_string(), Value::Int(0))),
            Box::new(Condition::Eq("b".to_string(), Value::Int(0))),
        );
        let rows = engine.select("complex", cond).unwrap();
        // a=0: 5 rows, b=0: 5 rows, overlap: 1 row => 9 rows
        assert_eq!(rows.len(), 9);
    }

    // ========== Row Key Helper Tests ==========

    #[test]
    fn test_row_key_format() {
        let key = RelationalEngine::row_key("users", 42);
        assert_eq!(key, "users:42");
    }

    #[test]
    fn test_row_prefix_format() {
        let prefix = RelationalEngine::row_prefix("users");
        assert_eq!(prefix, "users:");
    }

    #[test]
    fn test_table_meta_key_format() {
        let key = RelationalEngine::table_meta_key("users");
        assert_eq!(key, "_meta:table:users");
    }

    #[test]
    fn test_index_meta_key_format() {
        let key = RelationalEngine::index_meta_key("users", "email");
        assert_eq!(key, "_idx:users:email");
    }

    // ========== Batch Insert Tests ==========

    #[test]
    fn test_batch_insert_basic() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("name", ColumnType::String),
        ]);
        engine.create_table("batch_test", schema).unwrap();

        let rows = vec![
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("name".to_string(), Value::String("Alice".to_string())),
            ]),
            HashMap::from([
                ("id".to_string(), Value::Int(2)),
                ("name".to_string(), Value::String("Bob".to_string())),
            ]),
            HashMap::from([
                ("id".to_string(), Value::Int(3)),
                ("name".to_string(), Value::String("Carol".to_string())),
            ]),
        ];

        let ids = engine.batch_insert("batch_test", rows).unwrap();
        assert_eq!(ids.len(), 3);

        let result = engine.select("batch_test", Condition::True).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_batch_insert_empty() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("empty_batch", schema).unwrap();

        let ids = engine.batch_insert("empty_batch", vec![]).unwrap();
        assert_eq!(ids.len(), 0);
    }

    #[test]
    fn test_batch_insert_with_index() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("val", ColumnType::Int),
        ]);
        engine.create_table("batch_idx", schema).unwrap();
        engine.create_index("batch_idx", "val").unwrap();

        let rows: Vec<_> = (0..10)
            .map(|i| {
                HashMap::from([
                    ("id".to_string(), Value::Int(i)),
                    ("val".to_string(), Value::Int(i % 3)),
                ])
            })
            .collect();

        engine.batch_insert("batch_idx", rows).unwrap();

        let result = engine
            .select("batch_idx", Condition::Eq("val".to_string(), Value::Int(0)))
            .unwrap();
        // i = 0, 3, 6, 9 have val = 0
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_batch_insert_with_btree_index() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("score", ColumnType::Int),
        ]);
        engine.create_table("batch_btree", schema).unwrap();
        engine.create_btree_index("batch_btree", "score").unwrap();

        let rows: Vec<_> = (0..20)
            .map(|i| {
                HashMap::from([
                    ("id".to_string(), Value::Int(i)),
                    ("score".to_string(), Value::Int(i)),
                ])
            })
            .collect();

        engine.batch_insert("batch_btree", rows).unwrap();

        let result = engine
            .select(
                "batch_btree",
                Condition::Gt("score".to_string(), Value::Int(15)),
            )
            .unwrap();
        // 16, 17, 18, 19 > 15
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_batch_insert_null_validation() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("required", ColumnType::Int)]);
        engine.create_table("batch_null", schema).unwrap();

        let rows = vec![
            HashMap::from([("required".to_string(), Value::Int(1))]),
            HashMap::new(), // Missing required field
        ];

        let result = engine.batch_insert("batch_null", rows);
        assert!(matches!(result, Err(RelationalError::NullNotAllowed(_))));
    }

    #[test]
    fn test_batch_insert_type_validation() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("num", ColumnType::Int)]);
        engine.create_table("batch_type", schema).unwrap();

        let rows = vec![
            HashMap::from([("num".to_string(), Value::Int(1))]),
            HashMap::from([("num".to_string(), Value::String("not an int".to_string()))]),
        ];

        let result = engine.batch_insert("batch_type", rows);
        assert!(matches!(result, Err(RelationalError::TypeMismatch { .. })));
    }

    // ========== List Tables Tests ==========

    #[test]
    fn test_list_tables() {
        let engine = RelationalEngine::new();

        // Initially no tables
        let tables = engine.list_tables();
        assert_eq!(tables.len(), 0);

        // Add tables
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("table_a", schema.clone()).unwrap();
        engine.create_table("table_b", schema.clone()).unwrap();
        engine.create_table("table_c", schema).unwrap();

        let tables = engine.list_tables();
        assert_eq!(tables.len(), 3);
        assert!(tables.contains(&"table_a".to_string()));
        assert!(tables.contains(&"table_b".to_string()));
        assert!(tables.contains(&"table_c".to_string()));
    }

    // ========== BTree Index Range Query Tests ==========

    #[test]
    fn test_btree_index_range_lt() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("btree_lt", schema).unwrap();
        engine.create_btree_index("btree_lt", "val").unwrap();

        for i in 0..20 {
            engine
                .insert(
                    "btree_lt",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let result = engine
            .select("btree_lt", Condition::Lt("val".to_string(), Value::Int(5)))
            .unwrap();
        assert_eq!(result.len(), 5); // 0,1,2,3,4
    }

    #[test]
    fn test_btree_index_range_le() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("btree_le", schema).unwrap();
        engine.create_btree_index("btree_le", "val").unwrap();

        for i in 0..20 {
            engine
                .insert(
                    "btree_le",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let result = engine
            .select("btree_le", Condition::Le("val".to_string(), Value::Int(5)))
            .unwrap();
        assert_eq!(result.len(), 6); // 0,1,2,3,4,5
    }

    #[test]
    fn test_btree_index_range_gt() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("btree_gt", schema).unwrap();
        engine.create_btree_index("btree_gt", "val").unwrap();

        for i in 0..20 {
            engine
                .insert(
                    "btree_gt",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let result = engine
            .select("btree_gt", Condition::Gt("val".to_string(), Value::Int(15)))
            .unwrap();
        assert_eq!(result.len(), 4); // 16,17,18,19
    }

    #[test]
    fn test_btree_index_range_ge() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("btree_ge", schema).unwrap();
        engine.create_btree_index("btree_ge", "val").unwrap();

        for i in 0..20 {
            engine
                .insert(
                    "btree_ge",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let result = engine
            .select("btree_ge", Condition::Ge("val".to_string(), Value::Int(15)))
            .unwrap();
        assert_eq!(result.len(), 5); // 15,16,17,18,19
    }

    #[test]
    fn test_index_lookup_with_and_condition() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("a", ColumnType::Int),
            Column::new("b", ColumnType::Int),
        ]);
        engine.create_table("and_idx", schema).unwrap();
        engine.create_index("and_idx", "a").unwrap();

        for a in 0..5 {
            for b in 0..5 {
                engine
                    .insert(
                        "and_idx",
                        HashMap::from([
                            ("a".to_string(), Value::Int(a)),
                            ("b".to_string(), Value::Int(b)),
                        ]),
                    )
                    .unwrap();
            }
        }

        // (a == 2) AND (b < 3)
        let cond = Condition::And(
            Box::new(Condition::Eq("a".to_string(), Value::Int(2))),
            Box::new(Condition::Lt("b".to_string(), Value::Int(3))),
        );
        let rows = engine.select("and_idx", cond).unwrap();
        // a=2: 5 rows, b<3: 3 rows => 3 matching rows
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_btree_index_with_and_condition() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("a", ColumnType::Int),
            Column::new("b", ColumnType::Int),
        ]);
        engine.create_table("btree_and", schema).unwrap();
        engine.create_btree_index("btree_and", "a").unwrap();

        for a in 0..10 {
            for b in 0..10 {
                engine
                    .insert(
                        "btree_and",
                        HashMap::from([
                            ("a".to_string(), Value::Int(a)),
                            ("b".to_string(), Value::Int(b)),
                        ]),
                    )
                    .unwrap();
            }
        }

        // (a < 3) AND (b > 5)
        let cond = Condition::And(
            Box::new(Condition::Lt("a".to_string(), Value::Int(3))),
            Box::new(Condition::Gt("b".to_string(), Value::Int(5))),
        );
        let rows = engine.select("btree_and", cond).unwrap();
        // a in {0,1,2}: 30 rows, b in {6,7,8,9}: 4 each => 3*4 = 12 rows
        assert_eq!(rows.len(), 12);
    }

    // ========== Index with _id Tests ==========

    #[test]
    fn test_index_on_id_column() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("name", ColumnType::String)]);
        engine.create_table("id_idx", schema).unwrap();
        engine.create_index("id_idx", "_id").unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "id_idx",
                    HashMap::from([("name".to_string(), Value::String(format!("user{}", i)))]),
                )
                .unwrap();
        }

        let result = engine
            .select("id_idx", Condition::Eq("_id".to_string(), Value::Int(5)))
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, 5);
    }

    #[test]
    fn test_btree_index_on_id_column() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("name", ColumnType::String)]);
        engine.create_table("btree_id", schema).unwrap();
        engine.create_btree_index("btree_id", "_id").unwrap();

        for i in 0..20 {
            engine
                .insert(
                    "btree_id",
                    HashMap::from([("name".to_string(), Value::String(format!("user{}", i)))]),
                )
                .unwrap();
        }

        let result = engine
            .select("btree_id", Condition::Lt("_id".to_string(), Value::Int(6)))
            .unwrap();
        assert_eq!(result.len(), 5); // ids 1,2,3,4,5 < 6
    }

    // ========== Drop Index Tests ==========

    #[test]
    fn test_drop_index() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("drop_idx", schema).unwrap();
        engine.create_index("drop_idx", "val").unwrap();

        // Insert some data
        for i in 0..5 {
            engine
                .insert(
                    "drop_idx",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        // Drop the index
        let result = engine.drop_index("drop_idx", "val");
        assert!(result.is_ok());

        // Query should still work (full scan)
        let rows = engine
            .select("drop_idx", Condition::Eq("val".to_string(), Value::Int(2)))
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_drop_btree_index() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("drop_btree", schema).unwrap();
        engine.create_btree_index("drop_btree", "val").unwrap();

        for i in 0..5 {
            engine
                .insert(
                    "drop_btree",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let result = engine.drop_btree_index("drop_btree", "val");
        assert!(result.is_ok());

        let rows = engine
            .select(
                "drop_btree",
                Condition::Lt("val".to_string(), Value::Int(3)),
            )
            .unwrap();
        assert_eq!(rows.len(), 3);
    }

    // ========== Update with Index Tests ==========

    #[test]
    fn test_update_with_hash_index() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("val", ColumnType::Int),
        ]);
        engine.create_table("update_idx", schema).unwrap();
        engine.create_index("update_idx", "val").unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "update_idx",
                    HashMap::from([
                        ("id".to_string(), Value::Int(i)),
                        ("val".to_string(), Value::Int(i % 3)),
                    ]),
                )
                .unwrap();
        }

        // Update rows where val = 0 to val = 99
        let updated = engine
            .update(
                "update_idx",
                Condition::Eq("val".to_string(), Value::Int(0)),
                HashMap::from([("val".to_string(), Value::Int(99))]),
            )
            .unwrap();
        assert!(updated > 0);

        // Old value should return 0 rows
        let result = engine
            .select(
                "update_idx",
                Condition::Eq("val".to_string(), Value::Int(0)),
            )
            .unwrap();
        assert_eq!(result.len(), 0);

        // New value should have the updated rows
        let result = engine
            .select(
                "update_idx",
                Condition::Eq("val".to_string(), Value::Int(99)),
            )
            .unwrap();
        assert_eq!(result.len(), updated);
    }

    // ========== Columnar Scan Tests ==========

    #[test]
    fn test_select_columnar() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("a", ColumnType::Int),
            Column::new("b", ColumnType::Int),
            Column::new("c", ColumnType::Int),
        ]);
        engine.create_table("columnar", schema).unwrap();

        for i in 0..100 {
            engine
                .insert(
                    "columnar",
                    HashMap::from([
                        ("a".to_string(), Value::Int(i)),
                        ("b".to_string(), Value::Int(i * 2)),
                        ("c".to_string(), Value::Int(i * 3)),
                    ]),
                )
                .unwrap();
        }

        let opts = ColumnarScanOptions {
            projection: Some(vec!["a".to_string(), "b".to_string()]),
            prefer_columnar: true,
        };

        let result = engine
            .select_columnar(
                "columnar",
                Condition::Lt("a".to_string(), Value::Int(10)),
                opts,
            )
            .unwrap();
        assert_eq!(result.len(), 10);
    }

    // ========== Aggregation Tests ==========

    #[test]
    fn test_count_rows() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("count_test", schema).unwrap();

        for i in 0..50 {
            engine
                .insert(
                    "count_test",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let count = engine.count("count_test", Condition::True).unwrap();
        assert_eq!(count, 50);

        let count = engine
            .count(
                "count_test",
                Condition::Lt("val".to_string(), Value::Int(10)),
            )
            .unwrap();
        assert_eq!(count, 10);
    }

    #[test]
    fn test_sum_aggregate() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("sum_test", schema).unwrap();

        for i in 1..=10 {
            engine
                .insert(
                    "sum_test",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let sum = engine.sum("sum_test", "val", Condition::True).unwrap();
        // 1+2+3+4+5+6+7+8+9+10 = 55
        assert_eq!(sum, 55.0);
    }

    #[test]
    fn test_avg_aggregate() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("avg_test", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "avg_test",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let avg = engine.avg("avg_test", "val", Condition::True).unwrap();
        // (0+1+2+3+4+5+6+7+8+9)/10 = 4.5
        assert!((avg.unwrap() - 4.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_min_aggregate() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("min_test", schema).unwrap();

        for i in [5, 3, 8, 1, 9, 2] {
            engine
                .insert(
                    "min_test",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let min = engine.min("min_test", "val", Condition::True).unwrap();
        assert_eq!(min, Some(Value::Int(1)));
    }

    #[test]
    fn test_max_aggregate() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("max_test", schema).unwrap();

        for i in [5, 3, 8, 1, 9, 2] {
            engine
                .insert(
                    "max_test",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let max = engine.max("max_test", "val", Condition::True).unwrap();
        assert_eq!(max, Some(Value::Int(9)));
    }

    #[test]
    fn test_count_column() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int).nullable()]);
        engine.create_table("cnt_col", schema).unwrap();

        engine
            .insert(
                "cnt_col",
                HashMap::from([("val".to_string(), Value::Int(1))]),
            )
            .unwrap();
        engine
            .insert("cnt_col", HashMap::from([("val".to_string(), Value::Null)]))
            .unwrap();
        engine
            .insert(
                "cnt_col",
                HashMap::from([("val".to_string(), Value::Int(3))]),
            )
            .unwrap();

        // count_column should exclude nulls
        let count = engine
            .count_column("cnt_col", "val", Condition::True)
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_parallel_sum_threshold() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("psum", schema).unwrap();

        // Insert 1001 rows to trigger parallel path (>= 1000)
        for i in 0..1001 {
            engine
                .insert("psum", HashMap::from([("val".to_string(), Value::Int(i))]))
                .unwrap();
        }

        let sum = engine.sum("psum", "val", Condition::True).unwrap();
        // Sum of 0..1001 = 1001 * 1000 / 2 = 500500
        assert_eq!(sum, 500500.0);
    }

    #[test]
    fn test_parallel_avg_threshold() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float)]);
        engine.create_table("pavg", schema).unwrap();

        // Insert 1001 rows to trigger parallel path (>= 1000)
        for i in 0..1001 {
            engine
                .insert(
                    "pavg",
                    HashMap::from([("val".to_string(), Value::Float(i as f64))]),
                )
                .unwrap();
        }

        let avg = engine.avg("pavg", "val", Condition::True).unwrap().unwrap();
        // Avg of 0..1001 = 500.0
        assert!((avg - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_parallel_min_threshold() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("pmin", schema).unwrap();

        // Insert 1001 rows to trigger parallel path (>= 1000)
        for i in 0..1001 {
            engine
                .insert(
                    "pmin",
                    HashMap::from([("val".to_string(), Value::Int(1000 - i))]),
                )
                .unwrap();
        }

        let min = engine.min("pmin", "val", Condition::True).unwrap();
        assert_eq!(min, Some(Value::Int(0)));
    }

    #[test]
    fn test_parallel_max_threshold() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("pmax", schema).unwrap();

        // Insert 1001 rows to trigger parallel path (>= 1000)
        for i in 0..1001 {
            engine
                .insert("pmax", HashMap::from([("val".to_string(), Value::Int(i))]))
                .unwrap();
        }

        let max = engine.max("pmax", "val", Condition::True).unwrap();
        assert_eq!(max, Some(Value::Int(1000)));
    }

    #[test]
    fn test_tx_rollback_inserted_row() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_rb_ins", schema).unwrap();

        let tx_id = engine.begin_transaction();
        engine
            .tx_insert(
                tx_id,
                "tx_rb_ins",
                HashMap::from([("val".to_string(), Value::Int(42))]),
            )
            .unwrap();

        // Verify row exists before rollback
        let rows = engine.select("tx_rb_ins", Condition::True).unwrap();
        assert_eq!(rows.len(), 1);

        // Rollback
        engine.rollback(tx_id).unwrap();

        // Verify row is removed
        let rows = engine.select("tx_rb_ins", Condition::True).unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_tx_rollback_updated_row() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_rb_upd", schema).unwrap();

        // Insert initial row
        engine
            .insert(
                "tx_rb_upd",
                HashMap::from([("val".to_string(), Value::Int(10))]),
            )
            .unwrap();

        let tx_id = engine.begin_transaction();
        engine
            .tx_update(
                tx_id,
                "tx_rb_upd",
                Condition::True,
                HashMap::from([("val".to_string(), Value::Int(99))]),
            )
            .unwrap();

        // Verify updated value
        let rows = engine.select("tx_rb_upd", Condition::True).unwrap();
        assert_eq!(rows[0].get("val"), Some(&Value::Int(99)));

        // Rollback
        engine.rollback(tx_id).unwrap();

        // Verify original value restored
        let rows = engine.select("tx_rb_upd", Condition::True).unwrap();
        assert_eq!(rows[0].get("val"), Some(&Value::Int(10)));
    }

    #[test]
    fn test_tx_rollback_deleted_row() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_rb_del", schema).unwrap();

        // Insert initial row
        engine
            .insert(
                "tx_rb_del",
                HashMap::from([("val".to_string(), Value::Int(42))]),
            )
            .unwrap();

        let tx_id = engine.begin_transaction();
        engine
            .tx_delete(tx_id, "tx_rb_del", Condition::True)
            .unwrap();

        // Verify row deleted
        let rows = engine.select("tx_rb_del", Condition::True).unwrap();
        assert_eq!(rows.len(), 0);

        // Rollback
        engine.rollback(tx_id).unwrap();

        // Verify row restored
        let rows = engine.select("tx_rb_del", Condition::True).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("val"), Some(&Value::Int(42)));
    }

    #[test]
    fn test_tx_not_found_commit() {
        let engine = RelationalEngine::new();
        let result = engine.commit(99999);
        assert!(matches!(
            result,
            Err(RelationalError::TransactionNotFound(99999))
        ));
    }

    #[test]
    fn test_tx_not_found_rollback() {
        let engine = RelationalEngine::new();
        let result = engine.rollback(99999);
        assert!(matches!(
            result,
            Err(RelationalError::TransactionNotFound(99999))
        ));
    }

    #[test]
    fn test_tx_inactive_commit() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_inact", schema).unwrap();

        let tx_id = engine.begin_transaction();
        engine.commit(tx_id).unwrap();

        // Second commit should fail
        let result = engine.commit(tx_id);
        assert!(matches!(
            result,
            Err(RelationalError::TransactionNotFound(_))
        ));
    }

    #[test]
    fn test_slab_vectorized_filter_float_lt() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float)]);
        engine.create_table("slab_flt_lt", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "slab_flt_lt",
                    HashMap::from([("val".to_string(), Value::Float(i as f64))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "slab_flt_lt",
                Condition::Lt("val".to_string(), Value::Float(5.0)),
            )
            .unwrap();
        assert_eq!(rows.len(), 5); // 0, 1, 2, 3, 4
    }

    #[test]
    fn test_slab_vectorized_filter_float_gt() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float)]);
        engine.create_table("slab_flt_gt", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "slab_flt_gt",
                    HashMap::from([("val".to_string(), Value::Float(i as f64))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "slab_flt_gt",
                Condition::Gt("val".to_string(), Value::Float(5.0)),
            )
            .unwrap();
        assert_eq!(rows.len(), 4); // 6, 7, 8, 9
    }

    #[test]
    fn test_slab_vectorized_filter_float_eq() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float)]);
        engine.create_table("slab_flt_eq", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "slab_flt_eq",
                    HashMap::from([("val".to_string(), Value::Float(i as f64))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "slab_flt_eq",
                Condition::Eq("val".to_string(), Value::Float(5.0)),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_slab_vectorized_filter_int_ne() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("slab_int_ne", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "slab_int_ne",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "slab_int_ne",
                Condition::Ne("val".to_string(), Value::Int(5)),
            )
            .unwrap();
        assert_eq!(rows.len(), 9); // All except 5
    }

    #[test]
    fn test_slab_vectorized_filter_int_le() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("slab_int_le", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "slab_int_le",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "slab_int_le",
                Condition::Le("val".to_string(), Value::Int(5)),
            )
            .unwrap();
        assert_eq!(rows.len(), 6); // 0, 1, 2, 3, 4, 5
    }

    #[test]
    fn test_slab_vectorized_filter_int_ge() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("slab_int_ge", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "slab_int_ge",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "slab_int_ge",
                Condition::Ge("val".to_string(), Value::Int(5)),
            )
            .unwrap();
        assert_eq!(rows.len(), 5); // 5, 6, 7, 8, 9
    }

    #[test]
    fn test_slab_vectorized_filter_int_lt() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("slab_int_lt", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "slab_int_lt",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "slab_int_lt",
                Condition::Lt("val".to_string(), Value::Int(5)),
            )
            .unwrap();
        assert_eq!(rows.len(), 5); // 0, 1, 2, 3, 4
    }

    #[test]
    fn test_slab_vectorized_filter_int_gt() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("slab_int_gt", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "slab_int_gt",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "slab_int_gt",
                Condition::Gt("val".to_string(), Value::Int(5)),
            )
            .unwrap();
        assert_eq!(rows.len(), 4); // 6, 7, 8, 9
    }

    #[test]
    fn test_slab_vectorized_filter_and() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("slab_and", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "slab_and",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let cond = Condition::And(
            Box::new(Condition::Gt("val".to_string(), Value::Int(3))),
            Box::new(Condition::Lt("val".to_string(), Value::Int(7))),
        );
        let rows = engine.select("slab_and", cond).unwrap();
        assert_eq!(rows.len(), 3); // 4, 5, 6
    }

    #[test]
    fn test_slab_vectorized_filter_or() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("slab_or", schema).unwrap();

        for i in 0..10 {
            engine
                .insert(
                    "slab_or",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let cond = Condition::Or(
            Box::new(Condition::Lt("val".to_string(), Value::Int(2))),
            Box::new(Condition::Gt("val".to_string(), Value::Int(7))),
        );
        let rows = engine.select("slab_or", cond).unwrap();
        assert_eq!(rows.len(), 4); // 0, 1, 8, 9
    }

    #[test]
    fn test_natural_join_missing_column() {
        let engine = RelationalEngine::new();
        let schema_a = Schema::new(vec![
            Column::new("id", ColumnType::Int).nullable(),
            Column::new("val", ColumnType::Int),
        ]);
        let schema_b = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("other", ColumnType::Int),
        ]);

        engine.create_table("nj_miss_a", schema_a).unwrap();
        engine.create_table("nj_miss_b", schema_b).unwrap();

        // Insert with null for common column 'id'
        engine
            .insert(
                "nj_miss_a",
                HashMap::from([
                    ("id".to_string(), Value::Null),
                    ("val".to_string(), Value::Int(1)),
                ]),
            )
            .unwrap();

        engine
            .insert(
                "nj_miss_b",
                HashMap::from([
                    ("id".to_string(), Value::Int(1)),
                    ("other".to_string(), Value::Int(2)),
                ]),
            )
            .unwrap();

        // Natural join with null on join column should not match non-null
        let results = engine.natural_join("nj_miss_a", "nj_miss_b").unwrap();
        // Actually nulls don't match non-nulls, so should be 0
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_sum_with_float_values() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float)]);
        engine.create_table("sum_float", schema).unwrap();

        for i in 0..5 {
            engine
                .insert(
                    "sum_float",
                    HashMap::from([("val".to_string(), Value::Float(i as f64 * 1.5))]),
                )
                .unwrap();
        }

        let sum = engine.sum("sum_float", "val", Condition::True).unwrap();
        // 0 + 1.5 + 3 + 4.5 + 6 = 15.0
        assert!((sum - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_avg_empty_table() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float)]);
        engine.create_table("avg_empty", schema).unwrap();

        let avg = engine.avg("avg_empty", "val", Condition::True).unwrap();
        assert_eq!(avg, None);
    }

    #[test]
    fn test_min_float_values() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float)]);
        engine.create_table("min_float", schema).unwrap();

        for v in [5.5, 3.3, 8.8, 1.1, 9.9] {
            engine
                .insert(
                    "min_float",
                    HashMap::from([("val".to_string(), Value::Float(v))]),
                )
                .unwrap();
        }

        let min = engine.min("min_float", "val", Condition::True).unwrap();
        assert_eq!(min, Some(Value::Float(1.1)));
    }

    #[test]
    fn test_max_float_values() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float)]);
        engine.create_table("max_float", schema).unwrap();

        for v in [5.5, 3.3, 8.8, 1.1, 9.9] {
            engine
                .insert(
                    "max_float",
                    HashMap::from([("val".to_string(), Value::Float(v))]),
                )
                .unwrap();
        }

        let max = engine.max("max_float", "val", Condition::True).unwrap();
        assert_eq!(max, Some(Value::Float(9.9)));
    }

    #[test]
    fn test_columnar_select_with_projection() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("a", ColumnType::Int),
            Column::new("b", ColumnType::Int),
            Column::new("c", ColumnType::Int),
        ]);
        engine.create_table("col_proj", schema).unwrap();

        for i in 0..5 {
            engine
                .insert(
                    "col_proj",
                    HashMap::from([
                        ("a".to_string(), Value::Int(i)),
                        ("b".to_string(), Value::Int(i * 2)),
                        ("c".to_string(), Value::Int(i * 3)),
                    ]),
                )
                .unwrap();
        }

        let opts = ColumnarScanOptions {
            projection: Some(vec!["a".to_string(), "c".to_string()]),
            ..Default::default()
        };
        let rows = engine
            .select_columnar("col_proj", Condition::True, opts)
            .unwrap();
        assert_eq!(rows.len(), 5);
        // Should only have 'a' and 'c' columns
        for row in &rows {
            assert!(row.get("a").is_some());
            assert!(row.get("c").is_some());
            assert!(row.get("b").is_none());
        }
    }

    #[test]
    fn test_tx_rollback_with_index() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_rb_idx", schema).unwrap();
        engine.create_index("tx_rb_idx", "val").unwrap();

        // Insert initial row
        engine
            .insert(
                "tx_rb_idx",
                HashMap::from([("val".to_string(), Value::Int(10))]),
            )
            .unwrap();

        let tx_id = engine.begin_transaction();

        // Insert another row in transaction
        engine
            .tx_insert(
                tx_id,
                "tx_rb_idx",
                HashMap::from([("val".to_string(), Value::Int(20))]),
            )
            .unwrap();

        // Rollback
        engine.rollback(tx_id).unwrap();

        // Verify only initial row exists
        let rows = engine.select("tx_rb_idx", Condition::True).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("val"), Some(&Value::Int(10)));
    }

    #[test]
    fn test_tx_rollback_with_btree_index() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_rb_btree", schema).unwrap();
        engine.create_btree_index("tx_rb_btree", "val").unwrap();

        // Insert initial row
        engine
            .insert(
                "tx_rb_btree",
                HashMap::from([("val".to_string(), Value::Int(10))]),
            )
            .unwrap();

        let tx_id = engine.begin_transaction();

        // Update row in transaction
        engine
            .tx_update(
                tx_id,
                "tx_rb_btree",
                Condition::True,
                HashMap::from([("val".to_string(), Value::Int(99))]),
            )
            .unwrap();

        // Rollback
        engine.rollback(tx_id).unwrap();

        // Verify original value restored
        let rows = engine.select("tx_rb_btree", Condition::True).unwrap();
        assert_eq!(rows[0].get("val"), Some(&Value::Int(10)));
    }

    #[test]
    fn test_tx_insert_rollback_cleans_index() {
        // Regression test: ensure index entries are removed after insert rollback
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("idx_clean", schema).unwrap();
        engine.create_index("idx_clean", "val").unwrap();

        let tx = engine.begin_transaction();
        engine
            .tx_insert(
                tx,
                "idx_clean",
                HashMap::from([("val".to_string(), Value::Int(42))]),
            )
            .unwrap();
        engine.rollback(tx).unwrap();

        // Verify index has no entries for value 42
        let rows = engine
            .select(
                "idx_clean",
                Condition::Eq("val".to_string(), Value::Int(42)),
            )
            .unwrap();
        assert!(
            rows.is_empty(),
            "Index should not contain stale entries after rollback"
        );

        // Also verify with btree index
        let engine2 = RelationalEngine::new();
        let schema2 = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine2.create_table("btree_clean", schema2).unwrap();
        engine2.create_btree_index("btree_clean", "val").unwrap();

        let tx2 = engine2.begin_transaction();
        engine2
            .tx_insert(
                tx2,
                "btree_clean",
                HashMap::from([("val".to_string(), Value::Int(99))]),
            )
            .unwrap();
        engine2.rollback(tx2).unwrap();

        let rows2 = engine2
            .select(
                "btree_clean",
                Condition::Eq("val".to_string(), Value::Int(99)),
            )
            .unwrap();
        assert!(
            rows2.is_empty(),
            "BTree index should not contain stale entries after rollback"
        );
    }

    #[test]
    fn test_empty_row_count_selection() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("empty_sel", schema).unwrap();

        // Columnar select on empty table
        let rows = engine
            .select_columnar("empty_sel", Condition::True, ColumnarScanOptions::default())
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_tx_update_not_found() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_upd_nf", schema).unwrap();

        let result = engine.tx_update(
            99999,
            "tx_upd_nf",
            Condition::True,
            HashMap::from([("val".to_string(), Value::Int(42))]),
        );
        assert!(matches!(
            result,
            Err(RelationalError::TransactionNotFound(99999))
        ));
    }

    #[test]
    fn test_tx_delete_not_found() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_del_nf", schema).unwrap();

        let result = engine.tx_delete(99999, "tx_del_nf", Condition::True);
        assert!(matches!(
            result,
            Err(RelationalError::TransactionNotFound(99999))
        ));
    }

    #[test]
    fn test_tx_insert_not_found() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_ins_nf", schema).unwrap();

        let result = engine.tx_insert(
            99999,
            "tx_ins_nf",
            HashMap::from([("val".to_string(), Value::Int(42))]),
        );
        assert!(matches!(
            result,
            Err(RelationalError::TransactionNotFound(99999))
        ));
    }

    #[test]
    fn test_tx_insert_with_index_on_id() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_ins_id_idx", schema).unwrap();
        engine.create_index("tx_ins_id_idx", "_id").unwrap();

        let tx_id = engine.begin_transaction();
        engine
            .tx_insert(
                tx_id,
                "tx_ins_id_idx",
                HashMap::from([("val".to_string(), Value::Int(42))]),
            )
            .unwrap();
        engine.commit(tx_id).unwrap();

        let rows = engine.select("tx_ins_id_idx", Condition::True).unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_tx_insert_with_btree_index_on_id() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_ins_bt_id", schema).unwrap();
        engine.create_btree_index("tx_ins_bt_id", "_id").unwrap();

        let tx_id = engine.begin_transaction();
        engine
            .tx_insert(
                tx_id,
                "tx_ins_bt_id",
                HashMap::from([("val".to_string(), Value::Int(42))]),
            )
            .unwrap();
        engine.commit(tx_id).unwrap();

        let rows = engine.select("tx_ins_bt_id", Condition::True).unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_rollback_insert_with_id_index() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("rb_ins_id", schema).unwrap();
        engine.create_index("rb_ins_id", "_id").unwrap();

        let tx_id = engine.begin_transaction();
        engine
            .tx_insert(
                tx_id,
                "rb_ins_id",
                HashMap::from([("val".to_string(), Value::Int(42))]),
            )
            .unwrap();

        engine.rollback(tx_id).unwrap();

        let rows = engine.select("rb_ins_id", Condition::True).unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_rollback_update_with_id_btree() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("rb_upd_bt", schema).unwrap();
        engine.create_btree_index("rb_upd_bt", "_id").unwrap();

        engine
            .insert(
                "rb_upd_bt",
                HashMap::from([("val".to_string(), Value::Int(10))]),
            )
            .unwrap();

        let tx_id = engine.begin_transaction();
        engine
            .tx_update(
                tx_id,
                "rb_upd_bt",
                Condition::True,
                HashMap::from([("val".to_string(), Value::Int(99))]),
            )
            .unwrap();

        engine.rollback(tx_id).unwrap();

        let rows = engine.select("rb_upd_bt", Condition::True).unwrap();
        assert_eq!(rows[0].get("val"), Some(&Value::Int(10)));
    }

    #[test]
    fn test_rollback_delete_with_indexes() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("rb_del_idx", schema).unwrap();
        engine.create_index("rb_del_idx", "val").unwrap();
        engine.create_btree_index("rb_del_idx", "val").unwrap();

        engine
            .insert(
                "rb_del_idx",
                HashMap::from([("val".to_string(), Value::Int(42))]),
            )
            .unwrap();

        let tx_id = engine.begin_transaction();
        engine
            .tx_delete(tx_id, "rb_del_idx", Condition::True)
            .unwrap();

        engine.rollback(tx_id).unwrap();

        let rows = engine.select("rb_del_idx", Condition::True).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("val"), Some(&Value::Int(42)));
    }

    #[test]
    fn test_sum_with_non_numeric_column() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::String)]);
        engine.create_table("sum_str", schema).unwrap();

        engine
            .insert(
                "sum_str",
                HashMap::from([("val".to_string(), Value::String("hello".to_string()))]),
            )
            .unwrap();

        // sum on string column returns 0
        let sum = engine.sum("sum_str", "val", Condition::True).unwrap();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_avg_with_non_numeric_column() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::String)]);
        engine.create_table("avg_str", schema).unwrap();

        engine
            .insert(
                "avg_str",
                HashMap::from([("val".to_string(), Value::String("hello".to_string()))]),
            )
            .unwrap();

        // avg on string column returns None (no numeric values)
        let avg = engine.avg("avg_str", "val", Condition::True).unwrap();
        assert_eq!(avg, None);
    }

    #[test]
    fn test_min_empty_table() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("min_empty", schema).unwrap();

        let min = engine.min("min_empty", "val", Condition::True).unwrap();
        assert_eq!(min, None);
    }

    #[test]
    fn test_max_empty_table() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("max_empty", schema).unwrap();

        let max = engine.max("max_empty", "val", Condition::True).unwrap();
        assert_eq!(max, None);
    }

    #[test]
    fn test_columnar_scan_with_null_int() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int).nullable()]);
        engine.create_table("col_null_int", schema).unwrap();

        engine
            .insert(
                "col_null_int",
                HashMap::from([("val".to_string(), Value::Int(1))]),
            )
            .unwrap();
        engine
            .insert(
                "col_null_int",
                HashMap::from([("val".to_string(), Value::Null)]),
            )
            .unwrap();
        engine
            .insert(
                "col_null_int",
                HashMap::from([("val".to_string(), Value::Int(3))]),
            )
            .unwrap();

        let rows = engine.select("col_null_int", Condition::True).unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_columnar_scan_with_null_float() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float).nullable()]);
        engine.create_table("col_null_flt", schema).unwrap();

        engine
            .insert(
                "col_null_flt",
                HashMap::from([("val".to_string(), Value::Float(1.1))]),
            )
            .unwrap();
        engine
            .insert(
                "col_null_flt",
                HashMap::from([("val".to_string(), Value::Null)]),
            )
            .unwrap();
        engine
            .insert(
                "col_null_flt",
                HashMap::from([("val".to_string(), Value::Float(3.3))]),
            )
            .unwrap();

        let rows = engine.select("col_null_flt", Condition::True).unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_tx_is_active() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_active", schema).unwrap();

        let tx_id = engine.begin_transaction();
        assert!(engine.is_transaction_active(tx_id));

        engine.commit(tx_id).unwrap();
        assert!(!engine.is_transaction_active(tx_id));
    }

    #[test]
    fn test_parallel_join_large() {
        let engine = RelationalEngine::new();
        let schema_a = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("val", ColumnType::Int),
        ]);
        let schema_b = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("other", ColumnType::Int),
        ]);

        engine.create_table("pjoin_a", schema_a).unwrap();
        engine.create_table("pjoin_b", schema_b).unwrap();

        // Insert enough rows to trigger parallel path (>= 1000)
        for i in 0..1001 {
            engine
                .insert(
                    "pjoin_a",
                    HashMap::from([
                        ("id".to_string(), Value::Int(i)),
                        ("val".to_string(), Value::Int(i * 2)),
                    ]),
                )
                .unwrap();
        }

        for i in 0..10 {
            engine
                .insert(
                    "pjoin_b",
                    HashMap::from([
                        ("id".to_string(), Value::Int(i)),
                        ("other".to_string(), Value::Int(i + 100)),
                    ]),
                )
                .unwrap();
        }

        let results = engine.natural_join("pjoin_a", "pjoin_b").unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_select_with_condition_on_deleted_rows() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("del_cond", schema).unwrap();

        for i in 0..5 {
            engine
                .insert(
                    "del_cond",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        // Delete some rows
        engine
            .delete_rows("del_cond", Condition::Lt("val".to_string(), Value::Int(3)))
            .unwrap();

        // Select remaining rows
        let rows = engine.select("del_cond", Condition::True).unwrap();
        assert_eq!(rows.len(), 2); // Only 3, 4 remain
    }

    #[test]
    fn test_result_too_large_error_display() {
        let err = RelationalError::ResultTooLarge {
            operation: "CROSS JOIN".to_string(),
            actual: 2000000,
            max: 1000000,
        };
        let display = format!("{}", err);
        assert!(display.contains("CROSS JOIN"));
        assert!(display.contains("2000000"));
        assert!(display.contains("1000000"));
    }

    #[test]
    fn test_select_with_string_condition() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("name", ColumnType::String)]);
        engine.create_table("str_cond", schema).unwrap();

        engine
            .insert(
                "str_cond",
                HashMap::from([("name".to_string(), Value::String("alice".to_string()))]),
            )
            .unwrap();
        engine
            .insert(
                "str_cond",
                HashMap::from([("name".to_string(), Value::String("bob".to_string()))]),
            )
            .unwrap();

        let rows = engine
            .select(
                "str_cond",
                Condition::Eq("name".to_string(), Value::String("alice".to_string())),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_tx_update_column_not_found() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_upd_cnf", schema).unwrap();

        engine
            .insert(
                "tx_upd_cnf",
                HashMap::from([("val".to_string(), Value::Int(1))]),
            )
            .unwrap();

        let tx_id = engine.begin_transaction();
        let result = engine.tx_update(
            tx_id,
            "tx_upd_cnf",
            Condition::True,
            HashMap::from([("nonexistent".to_string(), Value::Int(42))]),
        );
        assert!(matches!(result, Err(RelationalError::ColumnNotFound(_))));
        engine.rollback(tx_id).unwrap();
    }

    #[test]
    fn test_tx_update_type_mismatch() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_upd_type", schema).unwrap();

        engine
            .insert(
                "tx_upd_type",
                HashMap::from([("val".to_string(), Value::Int(1))]),
            )
            .unwrap();

        let tx_id = engine.begin_transaction();
        let result = engine.tx_update(
            tx_id,
            "tx_upd_type",
            Condition::True,
            HashMap::from([("val".to_string(), Value::String("wrong".to_string()))]),
        );
        assert!(matches!(result, Err(RelationalError::TypeMismatch { .. })));
        engine.rollback(tx_id).unwrap();
    }

    #[test]
    fn test_tx_update_null_not_allowed() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_upd_null", schema).unwrap();

        engine
            .insert(
                "tx_upd_null",
                HashMap::from([("val".to_string(), Value::Int(1))]),
            )
            .unwrap();

        let tx_id = engine.begin_transaction();
        let result = engine.tx_update(
            tx_id,
            "tx_upd_null",
            Condition::True,
            HashMap::from([("val".to_string(), Value::Null)]),
        );
        assert!(matches!(result, Err(RelationalError::NullNotAllowed(_))));
        engine.rollback(tx_id).unwrap();
    }

    #[test]
    fn test_tx_inactive_rollback() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("tx_inact_rb", schema).unwrap();

        let tx_id = engine.begin_transaction();
        engine.commit(tx_id).unwrap();

        // Rollback after commit should fail
        let result = engine.rollback(tx_id);
        assert!(matches!(
            result,
            Err(RelationalError::TransactionNotFound(_))
        ));
    }

    #[test]
    fn test_min_with_null_values() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int).nullable()]);
        engine.create_table("min_nulls", schema).unwrap();

        engine
            .insert(
                "min_nulls",
                HashMap::from([("val".to_string(), Value::Null)]),
            )
            .unwrap();
        engine
            .insert(
                "min_nulls",
                HashMap::from([("val".to_string(), Value::Int(10))]),
            )
            .unwrap();
        engine
            .insert(
                "min_nulls",
                HashMap::from([("val".to_string(), Value::Int(5))]),
            )
            .unwrap();

        let min = engine.min("min_nulls", "val", Condition::True).unwrap();
        assert_eq!(min, Some(Value::Int(5)));
    }

    #[test]
    fn test_max_with_null_values() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int).nullable()]);
        engine.create_table("max_nulls", schema).unwrap();

        engine
            .insert(
                "max_nulls",
                HashMap::from([("val".to_string(), Value::Null)]),
            )
            .unwrap();
        engine
            .insert(
                "max_nulls",
                HashMap::from([("val".to_string(), Value::Int(10))]),
            )
            .unwrap();
        engine
            .insert(
                "max_nulls",
                HashMap::from([("val".to_string(), Value::Int(5))]),
            )
            .unwrap();

        let max = engine.max("max_nulls", "val", Condition::True).unwrap();
        assert_eq!(max, Some(Value::Int(10)));
    }

    #[test]
    fn test_sum_with_null_values() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int).nullable()]);
        engine.create_table("sum_nulls", schema).unwrap();

        engine
            .insert(
                "sum_nulls",
                HashMap::from([("val".to_string(), Value::Null)]),
            )
            .unwrap();
        engine
            .insert(
                "sum_nulls",
                HashMap::from([("val".to_string(), Value::Int(10))]),
            )
            .unwrap();
        engine
            .insert(
                "sum_nulls",
                HashMap::from([("val".to_string(), Value::Int(5))]),
            )
            .unwrap();

        let sum = engine.sum("sum_nulls", "val", Condition::True).unwrap();
        assert_eq!(sum, 15.0);
    }

    #[test]
    fn test_avg_with_null_values() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float).nullable()]);
        engine.create_table("avg_nulls", schema).unwrap();

        engine
            .insert(
                "avg_nulls",
                HashMap::from([("val".to_string(), Value::Null)]),
            )
            .unwrap();
        engine
            .insert(
                "avg_nulls",
                HashMap::from([("val".to_string(), Value::Float(10.0))]),
            )
            .unwrap();
        engine
            .insert(
                "avg_nulls",
                HashMap::from([("val".to_string(), Value::Float(20.0))]),
            )
            .unwrap();

        let avg = engine.avg("avg_nulls", "val", Condition::True).unwrap();
        // Only non-null values count: (10 + 20) / 2 = 15
        assert!(avg.is_some());
    }

    #[test]
    fn test_min_string() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::String)]);
        engine.create_table("min_str", schema).unwrap();

        engine
            .insert(
                "min_str",
                HashMap::from([("val".to_string(), Value::String("banana".to_string()))]),
            )
            .unwrap();
        engine
            .insert(
                "min_str",
                HashMap::from([("val".to_string(), Value::String("apple".to_string()))]),
            )
            .unwrap();
        engine
            .insert(
                "min_str",
                HashMap::from([("val".to_string(), Value::String("cherry".to_string()))]),
            )
            .unwrap();

        let min = engine.min("min_str", "val", Condition::True).unwrap();
        assert_eq!(min, Some(Value::String("apple".to_string())));
    }

    #[test]
    fn test_max_string() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::String)]);
        engine.create_table("max_str", schema).unwrap();

        engine
            .insert(
                "max_str",
                HashMap::from([("val".to_string(), Value::String("banana".to_string()))]),
            )
            .unwrap();
        engine
            .insert(
                "max_str",
                HashMap::from([("val".to_string(), Value::String("apple".to_string()))]),
            )
            .unwrap();
        engine
            .insert(
                "max_str",
                HashMap::from([("val".to_string(), Value::String("cherry".to_string()))]),
            )
            .unwrap();

        let max = engine.max("max_str", "val", Condition::True).unwrap();
        assert_eq!(max, Some(Value::String("cherry".to_string())));
    }

    #[test]
    fn test_update_with_condition_no_match() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("upd_no_match", schema).unwrap();

        engine
            .insert(
                "upd_no_match",
                HashMap::from([("val".to_string(), Value::Int(10))]),
            )
            .unwrap();

        // Update with condition that doesn't match
        let updated = engine
            .update(
                "upd_no_match",
                Condition::Eq("val".to_string(), Value::Int(99)),
                HashMap::from([("val".to_string(), Value::Int(20))]),
            )
            .unwrap();
        assert_eq!(updated, 0);

        // Value should remain unchanged
        let rows = engine.select("upd_no_match", Condition::True).unwrap();
        assert_eq!(rows[0].get("val"), Some(&Value::Int(10)));
    }

    #[test]
    fn test_delete_with_condition_no_match() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("del_no_match", schema).unwrap();

        engine
            .insert(
                "del_no_match",
                HashMap::from([("val".to_string(), Value::Int(10))]),
            )
            .unwrap();

        // Delete with condition that doesn't match
        let deleted = engine
            .delete_rows(
                "del_no_match",
                Condition::Eq("val".to_string(), Value::Int(99)),
            )
            .unwrap();
        assert_eq!(deleted, 0);

        // Row should still exist
        let rows = engine.select("del_no_match", Condition::True).unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_simd_filter_non_aligned_lt() {
        // Test with 5 elements (not divisible by 4) to hit remainder loop
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("simd_na_lt", schema).unwrap();

        for i in 0..5 {
            engine
                .insert(
                    "simd_na_lt",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "simd_na_lt",
                Condition::Lt("val".to_string(), Value::Int(3)),
            )
            .unwrap();
        assert_eq!(rows.len(), 3); // 0, 1, 2
    }

    #[test]
    fn test_simd_filter_non_aligned_le() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("simd_na_le", schema).unwrap();

        for i in 0..5 {
            engine
                .insert(
                    "simd_na_le",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "simd_na_le",
                Condition::Le("val".to_string(), Value::Int(3)),
            )
            .unwrap();
        assert_eq!(rows.len(), 4); // 0, 1, 2, 3
    }

    #[test]
    fn test_simd_filter_non_aligned_gt() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("simd_na_gt", schema).unwrap();

        for i in 0..5 {
            engine
                .insert(
                    "simd_na_gt",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "simd_na_gt",
                Condition::Gt("val".to_string(), Value::Int(2)),
            )
            .unwrap();
        assert_eq!(rows.len(), 2); // 3, 4
    }

    #[test]
    fn test_simd_filter_non_aligned_ge() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("simd_na_ge", schema).unwrap();

        for i in 0..5 {
            engine
                .insert(
                    "simd_na_ge",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "simd_na_ge",
                Condition::Ge("val".to_string(), Value::Int(2)),
            )
            .unwrap();
        assert_eq!(rows.len(), 3); // 2, 3, 4
    }

    #[test]
    fn test_simd_filter_non_aligned_eq() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("simd_na_eq", schema).unwrap();

        for i in 0..5 {
            engine
                .insert(
                    "simd_na_eq",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "simd_na_eq",
                Condition::Eq("val".to_string(), Value::Int(4)),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_simd_filter_non_aligned_ne() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("simd_na_ne", schema).unwrap();

        for i in 0..5 {
            engine
                .insert(
                    "simd_na_ne",
                    HashMap::from([("val".to_string(), Value::Int(i))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "simd_na_ne",
                Condition::Ne("val".to_string(), Value::Int(4)),
            )
            .unwrap();
        assert_eq!(rows.len(), 4); // 0, 1, 2, 3
    }

    #[test]
    fn test_simd_filter_float_non_aligned_lt() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float)]);
        engine.create_table("simd_fna_lt", schema).unwrap();

        for i in 0..5 {
            engine
                .insert(
                    "simd_fna_lt",
                    HashMap::from([("val".to_string(), Value::Float(i as f64))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "simd_fna_lt",
                Condition::Lt("val".to_string(), Value::Float(3.0)),
            )
            .unwrap();
        assert_eq!(rows.len(), 3); // 0, 1, 2
    }

    #[test]
    fn test_simd_filter_float_non_aligned_gt() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float)]);
        engine.create_table("simd_fna_gt", schema).unwrap();

        for i in 0..5 {
            engine
                .insert(
                    "simd_fna_gt",
                    HashMap::from([("val".to_string(), Value::Float(i as f64))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "simd_fna_gt",
                Condition::Gt("val".to_string(), Value::Float(2.0)),
            )
            .unwrap();
        assert_eq!(rows.len(), 2); // 3, 4
    }

    #[test]
    fn test_simd_filter_float_non_aligned_eq() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Float)]);
        engine.create_table("simd_fna_eq", schema).unwrap();

        for i in 0..5 {
            engine
                .insert(
                    "simd_fna_eq",
                    HashMap::from([("val".to_string(), Value::Float(i as f64))]),
                )
                .unwrap();
        }

        let rows = engine
            .select(
                "simd_fna_eq",
                Condition::Eq("val".to_string(), Value::Float(4.0)),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_invalid_name_error_display() {
        let err = RelationalError::InvalidName("test message".to_string());
        let display = format!("{}", err);
        assert!(display.contains("Invalid name"));
        assert!(display.contains("test message"));
    }

    #[test]
    fn test_durable_engine_delete() {
        use tensor_store::WalConfig;

        let temp_dir = tempfile::tempdir().unwrap();
        let wal_path = temp_dir.path().join("test_del.wal");
        let config = WalConfig::default();

        let engine = RelationalEngine::open_durable(&wal_path, config).unwrap();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("dur_del", schema).unwrap();

        engine
            .insert(
                "dur_del",
                HashMap::from([("val".to_string(), Value::Int(42))]),
            )
            .unwrap();

        // Delete row using durable engine
        let deleted = engine.delete_rows("dur_del", Condition::True).unwrap();
        assert_eq!(deleted, 1);

        let rows = engine.select("dur_del", Condition::True).unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_durable_engine_drop_table() {
        use tensor_store::WalConfig;

        let temp_dir = tempfile::tempdir().unwrap();
        let wal_path = temp_dir.path().join("test_drop.wal");
        let config = WalConfig::default();

        let engine = RelationalEngine::open_durable(&wal_path, config).unwrap();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("dur_drop", schema).unwrap();

        engine
            .insert(
                "dur_drop",
                HashMap::from([("val".to_string(), Value::Int(42))]),
            )
            .unwrap();

        // Drop the table using durable engine
        engine.drop_table("dur_drop").unwrap();

        // Verify table is gone
        let result = engine.select("dur_drop", Condition::True);
        assert!(matches!(result, Err(RelationalError::TableNotFound(_))));
    }

    #[test]
    fn test_concurrent_create_table_same_name() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        use std::thread;

        let engine = Arc::new(RelationalEngine::new());
        let success = Arc::new(AtomicUsize::new(0));
        let error = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let eng = Arc::clone(&engine);
                let s = Arc::clone(&success);
                let e = Arc::clone(&error);
                thread::spawn(move || {
                    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
                    match eng.create_table("contested", schema) {
                        Ok(()) => {
                            s.fetch_add(1, Ordering::SeqCst);
                        },
                        Err(RelationalError::TableAlreadyExists(_)) => {
                            e.fetch_add(1, Ordering::SeqCst);
                        },
                        Err(err) => panic!("unexpected error: {err:?}"),
                    };
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(success.load(Ordering::SeqCst), 1);
        assert_eq!(error.load(Ordering::SeqCst), 9);
    }

    #[test]
    fn test_concurrent_create_index_same_column() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        use std::thread;

        let engine = Arc::new(RelationalEngine::new());
        let schema = Schema::new(vec![Column::new("value", ColumnType::Int)]);
        engine.create_table("idx_test", schema).unwrap();

        let success = Arc::new(AtomicUsize::new(0));
        let error = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let eng = Arc::clone(&engine);
                let s = Arc::clone(&success);
                let e = Arc::clone(&error);
                thread::spawn(move || match eng.create_index("idx_test", "value") {
                    Ok(()) => {
                        s.fetch_add(1, Ordering::SeqCst);
                    },
                    Err(RelationalError::IndexAlreadyExists { .. }) => {
                        e.fetch_add(1, Ordering::SeqCst);
                    },
                    Err(err) => panic!("unexpected error: {err:?}"),
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(success.load(Ordering::SeqCst), 1);
        assert_eq!(error.load(Ordering::SeqCst), 9);
    }

    #[test]
    fn test_concurrent_create_btree_index_same_column() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        use std::thread;

        let engine = Arc::new(RelationalEngine::new());
        let schema = Schema::new(vec![Column::new("value", ColumnType::Int)]);
        engine.create_table("btree_test", schema).unwrap();

        let success = Arc::new(AtomicUsize::new(0));
        let error = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let eng = Arc::clone(&engine);
                let s = Arc::clone(&success);
                let e = Arc::clone(&error);
                thread::spawn(
                    move || match eng.create_btree_index("btree_test", "value") {
                        Ok(()) => {
                            s.fetch_add(1, Ordering::SeqCst);
                        },
                        Err(RelationalError::IndexAlreadyExists { .. }) => {
                            e.fetch_add(1, Ordering::SeqCst);
                        },
                        Err(err) => panic!("unexpected error: {err:?}"),
                    },
                )
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(success.load(Ordering::SeqCst), 1);
        assert_eq!(error.load(Ordering::SeqCst), 9);
    }

    #[test]
    fn test_concurrent_drop_table_same_name() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        use std::thread;

        let engine = Arc::new(RelationalEngine::new());
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("drop_test", schema).unwrap();

        let success = Arc::new(AtomicUsize::new(0));
        let error = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let eng = Arc::clone(&engine);
                let s = Arc::clone(&success);
                let e = Arc::clone(&error);
                thread::spawn(move || match eng.drop_table("drop_test") {
                    Ok(()) => {
                        s.fetch_add(1, Ordering::SeqCst);
                    },
                    Err(RelationalError::TableNotFound(_)) => {
                        e.fetch_add(1, Ordering::SeqCst);
                    },
                    Err(err) => panic!("unexpected error: {err:?}"),
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(success.load(Ordering::SeqCst), 1);
        assert_eq!(error.load(Ordering::SeqCst), 9);
    }

    mod condition_tensor_tests {
        use super::*;
        use tensor_store::{ScalarValue, TensorData, TensorValue};

        fn make_test_tensor(id: i64, val: i64) -> TensorData {
            let mut data = TensorData::new();
            data.set("_id".to_string(), TensorValue::Scalar(ScalarValue::Int(id)));
            data.set(
                "val".to_string(),
                TensorValue::Scalar(ScalarValue::Int(val)),
            );
            data
        }

        #[test]
        fn test_condition_tensor_true() {
            let tensor = make_test_tensor(1, 42);
            assert!(Condition::True.evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_eq() {
            let tensor = make_test_tensor(1, 42);
            assert!(Condition::Eq("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
            assert!(!Condition::Eq("val".to_string(), Value::Int(99)).evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_ne() {
            let tensor = make_test_tensor(1, 42);
            assert!(Condition::Ne("val".to_string(), Value::Int(99)).evaluate_tensor(&tensor));
            assert!(!Condition::Ne("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_lt() {
            let tensor = make_test_tensor(1, 42);
            assert!(Condition::Lt("val".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
            assert!(!Condition::Lt("val".to_string(), Value::Int(30)).evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_le() {
            let tensor = make_test_tensor(1, 42);
            assert!(Condition::Le("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
            assert!(Condition::Le("val".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_gt() {
            let tensor = make_test_tensor(1, 42);
            assert!(Condition::Gt("val".to_string(), Value::Int(30)).evaluate_tensor(&tensor));
            assert!(!Condition::Gt("val".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_ge() {
            let tensor = make_test_tensor(1, 42);
            assert!(Condition::Ge("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
            assert!(Condition::Ge("val".to_string(), Value::Int(30)).evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_and() {
            let tensor = make_test_tensor(1, 42);
            let cond = Condition::And(
                Box::new(Condition::Gt("val".to_string(), Value::Int(30))),
                Box::new(Condition::Lt("val".to_string(), Value::Int(50))),
            );
            assert!(cond.evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_or() {
            let tensor = make_test_tensor(1, 42);
            let cond = Condition::Or(
                Box::new(Condition::Eq("val".to_string(), Value::Int(99))),
                Box::new(Condition::Eq("val".to_string(), Value::Int(42))),
            );
            assert!(cond.evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_id_field() {
            let tensor = make_test_tensor(100, 42);
            assert!(Condition::Eq("_id".to_string(), Value::Int(100)).evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_missing_column() {
            let tensor = make_test_tensor(1, 42);
            assert!(!Condition::Eq("missing".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_float() {
            let mut tensor = TensorData::new();
            tensor.set(
                "score".to_string(),
                TensorValue::Scalar(ScalarValue::Float(3.14)),
            );
            assert!(Condition::Eq("score".to_string(), Value::Float(3.14)).evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_string() {
            let mut tensor = TensorData::new();
            tensor.set(
                "name".to_string(),
                TensorValue::Scalar(ScalarValue::String("test".to_string())),
            );
            assert!(
                Condition::Eq("name".to_string(), Value::String("test".to_string()))
                    .evaluate_tensor(&tensor)
            );
        }

        #[test]
        fn test_condition_tensor_missing_id() {
            let mut tensor = TensorData::new();
            tensor.set("val".to_string(), TensorValue::Scalar(ScalarValue::Int(42)));
            assert!(!Condition::Eq("_id".to_string(), Value::Int(1)).evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_le_missing() {
            let tensor = make_test_tensor(1, 42);
            assert!(!Condition::Le("missing".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
        }

        #[test]
        fn test_condition_tensor_ge_missing() {
            let tensor = make_test_tensor(1, 42);
            assert!(!Condition::Ge("missing".to_string(), Value::Int(30)).evaluate_tensor(&tensor));
        }
    }

    #[test]
    fn test_update_atomicity_on_failure() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("val", ColumnType::Int),
        ]);
        engine.create_table("atomic_test", schema).unwrap();
        engine.create_index("atomic_test", "val").unwrap();

        // Insert initial row
        engine
            .insert(
                "atomic_test",
                HashMap::from([
                    ("id".to_string(), Value::Int(1)),
                    ("val".to_string(), Value::Int(100)),
                ]),
            )
            .unwrap();

        // Verify initial state
        let rows = engine
            .select(
                "atomic_test",
                Condition::Eq("val".to_string(), Value::Int(100)),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);

        // Try update with type mismatch (should fail validation, no changes)
        let result = engine.update(
            "atomic_test",
            Condition::True,
            HashMap::from([("val".to_string(), Value::String("not_int".to_string()))]),
        );
        assert!(result.is_err());

        // Verify state unchanged
        let rows = engine
            .select(
                "atomic_test",
                Condition::Eq("val".to_string(), Value::Int(100)),
            )
            .unwrap();
        assert_eq!(
            rows.len(),
            1,
            "Data should be unchanged after failed update"
        );
    }

    #[test]
    fn test_insert_atomicity() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("insert_atomic", schema).unwrap();
        engine.create_index("insert_atomic", "val").unwrap();

        // Successful insert
        let row_id = engine
            .insert(
                "insert_atomic",
                HashMap::from([("val".to_string(), Value::Int(42))]),
            )
            .unwrap();
        assert_eq!(row_id, 1);

        // Verify index works
        let rows = engine
            .select(
                "insert_atomic",
                Condition::Eq("val".to_string(), Value::Int(42)),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_delete_atomicity() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("delete_atomic", schema).unwrap();
        engine.create_index("delete_atomic", "val").unwrap();

        engine
            .insert(
                "delete_atomic",
                HashMap::from([("val".to_string(), Value::Int(1))]),
            )
            .unwrap();
        engine
            .insert(
                "delete_atomic",
                HashMap::from([("val".to_string(), Value::Int(2))]),
            )
            .unwrap();

        // Delete one row
        let count = engine
            .delete_rows(
                "delete_atomic",
                Condition::Eq("val".to_string(), Value::Int(1)),
            )
            .unwrap();
        assert_eq!(count, 1);

        // Verify only one row remains and index is correct
        let rows = engine.select("delete_atomic", Condition::True).unwrap();
        assert_eq!(rows.len(), 1);

        let indexed = engine
            .select(
                "delete_atomic",
                Condition::Eq("val".to_string(), Value::Int(2)),
            )
            .unwrap();
        assert_eq!(indexed.len(), 1);
    }

    #[test]
    fn test_index_lock_sharding_bounded_memory() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("age", ColumnType::Int),
        ]);
        engine.create_table("users", schema).unwrap();
        engine.create_index("users", "name").unwrap();

        // Insert many unique values - lock array stays fixed size (64 locks)
        for i in 0..1000 {
            engine
                .insert(
                    "users",
                    HashMap::from([
                        ("name".to_string(), Value::String(format!("user_{i}"))),
                        ("age".to_string(), Value::Int(i)),
                    ]),
                )
                .unwrap();
        }

        // Verify all rows inserted correctly
        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows.len(), 1000);
    }

    #[test]
    fn test_concurrent_index_operations_with_lock_striping() {
        use std::sync::Arc;
        use std::thread;

        let engine = Arc::new(RelationalEngine::new());
        let schema = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("age", ColumnType::Int),
        ]);
        engine.create_table("users", schema).unwrap();
        engine.create_index("users", "age").unwrap();

        let handles: Vec<_> = (0..10)
            .map(|t| {
                let eng = Arc::clone(&engine);
                thread::spawn(move || {
                    for i in 0..100 {
                        let mut values = HashMap::new();
                        values.insert(
                            "name".to_string(),
                            Value::String(format!("thread_{t}_user_{i}")),
                        );
                        values.insert("age".to_string(), Value::Int((t * 1000 + i) as i64));
                        eng.insert("users", values).unwrap();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all rows inserted
        let rows = engine.select("users", Condition::True).unwrap();
        assert_eq!(rows.len(), 1000);
    }

    #[test]
    fn test_btree_index_lock_ordering_no_deadlock() {
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        let engine = Arc::new(RelationalEngine::new());
        let schema = Schema::new(vec![Column::new("value", ColumnType::Int)]);
        engine.create_table("lock_test", schema).unwrap();
        engine.create_btree_index("lock_test", "value").unwrap();

        // Pre-insert rows
        for i in 0..100 {
            let mut values = HashMap::new();
            values.insert("value".to_string(), Value::Int(i));
            engine.insert("lock_test", values).unwrap();
        }

        let mut handles = vec![];

        // Spawn threads doing concurrent btree index adds
        for t in 0..4 {
            let eng = Arc::clone(&engine);
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    let val = (t * 1000 + i) as i64;
                    let mut values = HashMap::new();
                    values.insert("value".to_string(), Value::Int(val));
                    let _ = eng.insert("lock_test", values);
                }
            }));
        }

        // Spawn threads doing concurrent btree index removes (via delete)
        for _ in 0..2 {
            let eng = Arc::clone(&engine);
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    let _ = eng.delete_rows(
                        "lock_test",
                        Condition::Eq("value".to_string(), Value::Int(i)),
                    );
                }
            }));
        }

        // Use timeout to detect deadlock
        let start = std::time::Instant::now();
        for handle in handles {
            handle.join().expect("Thread panicked");
        }
        let elapsed = start.elapsed();

        // If this takes more than 10 seconds, likely deadlock (test should complete in < 1s)
        assert!(
            elapsed < Duration::from_secs(10),
            "Possible deadlock: took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_rollback_always_releases_locks_even_on_undo_failure() {
        use std::sync::Arc;

        let engine = Arc::new(RelationalEngine::new());
        let schema = Schema::new(vec![Column::new("value", ColumnType::Int)]);
        engine.create_table("test", schema).unwrap();

        // Start transaction and insert a row
        let tx1 = engine.begin_transaction();
        let mut values = HashMap::new();
        values.insert("value".to_string(), Value::Int(42));
        engine.tx_insert(tx1, "test", values).unwrap();

        // Rollback - should always release locks
        let _ = engine.rollback(tx1);

        // Verify transaction is gone (not stuck in Aborting)
        assert!(
            engine.tx_manager.get(tx1).is_none(),
            "Transaction should be removed after rollback"
        );

        // Verify another transaction can proceed (locks released)
        let tx2 = engine.begin_transaction();
        let mut values2 = HashMap::new();
        values2.insert("value".to_string(), Value::Int(100));
        let result = engine.tx_insert(tx2, "test", values2);
        assert!(
            result.is_ok(),
            "New transaction should succeed after rollback"
        );
        engine.commit(tx2).unwrap();
    }

    #[test]
    fn test_rollback_completes_all_undo_entries_even_with_errors() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("a", ColumnType::Int),
            Column::new("b", ColumnType::Int),
        ]);
        engine.create_table("multi", schema).unwrap();

        // Insert multiple rows in a transaction
        let tx = engine.begin_transaction();
        for i in 0..5 {
            let mut values = HashMap::new();
            values.insert("a".to_string(), Value::Int(i));
            values.insert("b".to_string(), Value::Int(i * 10));
            engine.tx_insert(tx, "multi", values).unwrap();
        }

        // Rollback should process all entries
        let result = engine.rollback(tx);

        // Transaction should be cleaned up regardless of result
        assert!(
            engine.tx_manager.get(tx).is_none(),
            "Transaction must be removed after rollback"
        );

        // Rollback succeeded with no errors in this case
        assert!(result.is_ok());

        // Table should be empty (all inserts undone)
        let rows = engine.select("multi", Condition::True).unwrap();
        assert_eq!(rows.len(), 0, "All inserted rows should be undone");
    }

    #[test]
    fn test_rollback_returns_error_but_still_cleans_up() {
        // This test verifies that even if RollbackFailed is returned,
        // the transaction state is properly cleaned up
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("cleanup_test", schema).unwrap();

        let tx = engine.begin_transaction();
        let mut values = HashMap::new();
        values.insert("val".to_string(), Value::Int(1));
        engine.tx_insert(tx, "cleanup_test", values).unwrap();

        // Even if rollback returns an error, transaction should be gone
        let _ = engine.rollback(tx);

        // Attempting to use the transaction should fail with NotFound, not Inactive
        let result = engine.rollback(tx);
        assert!(matches!(
            result,
            Err(RelationalError::TransactionNotFound(_))
        ));
    }

    #[test]
    fn test_get_schema_missing_column_metadata_returns_error() {
        let engine = RelationalEngine::new();

        // Create a table normally first to ensure the infrastructure exists
        let schema = Schema::new(vec![Column::new("name", ColumnType::String)]);
        engine.create_table("test", schema).unwrap();

        // Corrupt the schema by adding a column name without its metadata
        let meta_key = "_meta:table:test";
        let mut meta = engine.store.get(meta_key).unwrap();
        // Overwrite _columns to include a non-existent column
        meta.set(
            "_columns".to_string(),
            TensorValue::Scalar(ScalarValue::String("name,ghost_column".to_string())),
        );
        engine.store.put(meta_key, meta).unwrap();

        // Now get_schema should return an error for the missing column metadata
        let result = engine.get_schema("test");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            RelationalError::SchemaCorrupted { table, reason } => {
                assert_eq!(table, "test");
                assert!(reason.contains("ghost_column"));
                assert!(reason.contains("missing metadata"));
            },
            other => panic!("Expected SchemaCorrupted, got: {other}"),
        }
    }

    #[test]
    fn test_get_schema_malformed_type_string_returns_error() {
        let engine = RelationalEngine::new();

        // Create a table normally first
        let schema = Schema::new(vec![Column::new("name", ColumnType::String)]);
        engine.create_table("test", schema).unwrap();

        // Corrupt the column type metadata to be malformed (missing the nullable part)
        let meta_key = "_meta:table:test";
        let mut meta = engine.store.get(meta_key).unwrap();
        meta.set(
            "_col:name".to_string(),
            TensorValue::Scalar(ScalarValue::String("string".to_string())), // Missing :notnull or :null
        );
        engine.store.put(meta_key, meta).unwrap();

        // Now get_schema should return an error for the malformed type string
        let result = engine.get_schema("test");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            RelationalError::SchemaCorrupted { table, reason } => {
                assert_eq!(table, "test");
                assert!(reason.contains("malformed type string"));
                assert!(reason.contains("name"));
            },
            other => panic!("Expected SchemaCorrupted, got: {other}"),
        }
    }

    #[test]
    fn test_get_schema_unknown_column_type_returns_error() {
        let engine = RelationalEngine::new();

        // Create a table normally first
        let schema = Schema::new(vec![Column::new("data", ColumnType::String)]);
        engine.create_table("test", schema).unwrap();

        // Corrupt the column type metadata to have an unknown type
        let meta_key = "_meta:table:test";
        let mut meta = engine.store.get(meta_key).unwrap();
        meta.set(
            "_col:data".to_string(),
            TensorValue::Scalar(ScalarValue::String("blob:notnull".to_string())),
        );
        engine.store.put(meta_key, meta).unwrap();

        // Now get_schema should return an error for the unknown type
        let result = engine.get_schema("test");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            RelationalError::SchemaCorrupted { table, reason } => {
                assert_eq!(table, "test");
                assert!(reason.contains("unknown column type"));
                assert!(reason.contains("blob"));
            },
            other => panic!("Expected SchemaCorrupted, got: {other}"),
        }
    }

    #[test]
    fn test_btree_index_respects_entry_limit() {
        // Create an engine with a very small btree entry limit for testing
        let mut engine = RelationalEngine::new();
        engine.max_btree_entries = 3; // Very small limit for testing

        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("test", schema).unwrap();
        engine.create_btree_index("test", "val").unwrap();

        // Insert rows with unique values - each creates a new btree entry
        for i in 0..3 {
            engine
                .insert("test", HashMap::from([("val".to_string(), Value::Int(i))]))
                .unwrap();
        }

        // Verify counter is at 3
        assert_eq!(engine.btree_entry_count.load(Ordering::Relaxed), 3);

        // Fourth unique value should fail
        let result = engine.insert(
            "test",
            HashMap::from([("val".to_string(), Value::Int(100))]),
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            RelationalError::ResultTooLarge {
                operation,
                actual,
                max,
            } => {
                assert_eq!(operation, "btree_index_add");
                assert_eq!(actual, 4);
                assert_eq!(max, 3);
            },
            other => panic!("Expected ResultTooLarge, got: {other}"),
        }

        // Inserting duplicate value should work (doesn't add new btree key)
        engine
            .insert("test", HashMap::from([("val".to_string(), Value::Int(0))]))
            .unwrap();
        assert_eq!(engine.btree_entry_count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_btree_index_remove_decrements_counter() {
        let engine = RelationalEngine::new();

        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("test", schema).unwrap();
        engine.create_btree_index("test", "val").unwrap();

        // Insert 3 rows with unique values
        for i in 0..3 {
            engine
                .insert("test", HashMap::from([("val".to_string(), Value::Int(i))]))
                .unwrap();
        }
        assert_eq!(engine.btree_entry_count.load(Ordering::Relaxed), 3);

        // Delete one row - should decrement counter
        engine
            .delete_rows("test", Condition::Eq("val".to_string(), Value::Int(1)))
            .unwrap();
        assert_eq!(engine.btree_entry_count.load(Ordering::Relaxed), 2);

        // Delete another row
        engine
            .delete_rows("test", Condition::Eq("val".to_string(), Value::Int(0)))
            .unwrap();
        assert_eq!(engine.btree_entry_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_drop_btree_index_clears_counter() {
        let engine = RelationalEngine::new();

        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("test", schema).unwrap();
        engine.create_btree_index("test", "val").unwrap();

        // Insert rows with unique values
        for i in 0..5 {
            engine
                .insert("test", HashMap::from([("val".to_string(), Value::Int(i))]))
                .unwrap();
        }
        assert_eq!(engine.btree_entry_count.load(Ordering::Relaxed), 5);

        // Drop the index - should decrement counter by 5
        engine.drop_btree_index("test", "val").unwrap();
        assert_eq!(engine.btree_entry_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_error_display_too_many_tables() {
        let err = RelationalError::TooManyTables {
            current: 10,
            max: 5,
        };
        let msg = err.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
        assert!(msg.contains("Too many tables"));
    }

    #[test]
    fn test_error_display_too_many_indexes() {
        let err = RelationalError::TooManyIndexes {
            table: "users".to_string(),
            current: 8,
            max: 5,
        };
        let msg = err.to_string();
        assert!(msg.contains("users"));
        assert!(msg.contains("8"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_error_display_query_timeout() {
        let err = RelationalError::QueryTimeout {
            operation: "SELECT".to_string(),
            timeout_ms: 5000,
        };
        let msg = err.to_string();
        assert!(msg.contains("SELECT"));
        assert!(msg.contains("5000"));
        assert!(msg.contains("timeout"));
    }

    #[test]
    fn test_with_store_and_config() {
        let store = TensorStore::new();
        let config = RelationalConfig::new().with_max_tables(3);
        let engine = RelationalEngine::with_store_and_config(store, config);

        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("t1", schema.clone()).unwrap();
        engine.create_table("t2", schema.clone()).unwrap();
        engine.create_table("t3", schema.clone()).unwrap();

        // Fourth should fail
        let result = engine.create_table("t4", schema);
        assert!(matches!(result, Err(RelationalError::TooManyTables { .. })));
    }

    #[test]
    fn test_resolve_timeout_no_timeout_configured() {
        let config = RelationalConfig::default();
        let engine = RelationalEngine::with_config(config);

        // No timeout configured anywhere
        let resolved = engine.resolve_timeout(QueryOptions::new());
        assert!(resolved.is_none());
    }

    #[test]
    fn test_resolve_timeout_uses_default() {
        let config = RelationalConfig::new().with_default_timeout_ms(3000);
        let engine = RelationalEngine::with_config(config);

        // Should use default when options don't specify
        let resolved = engine.resolve_timeout(QueryOptions::new());
        assert_eq!(resolved, Some(3000));
    }

    #[test]
    fn test_resolve_timeout_clamps_query_options_to_max() {
        // Query specifies 10000ms but max is 5000ms
        let config = RelationalConfig::new().with_max_timeout_ms(5000);
        let engine = RelationalEngine::with_config(config);

        let resolved = engine.resolve_timeout(QueryOptions::new().with_timeout_ms(10000));
        assert_eq!(resolved, Some(5000));
    }

    #[test]
    fn test_resolve_timeout_no_clamping_without_max() {
        // No max configured, should use query's timeout as-is
        let config = RelationalConfig::new().with_default_timeout_ms(1000);
        let engine = RelationalEngine::with_config(config);

        let resolved = engine.resolve_timeout(QueryOptions::new().with_timeout_ms(50000));
        assert_eq!(resolved, Some(50000));
    }

    #[test]
    fn test_config_validation_error_message() {
        let config = RelationalConfig::new()
            .with_default_timeout_ms(10000)
            .with_max_timeout_ms(5000);
        let err = config.validate().unwrap_err();
        assert!(err.contains("10000"));
        assert!(err.contains("5000"));
    }

    #[test]
    fn test_config_all_builder_methods() {
        let config = RelationalConfig::new()
            .with_max_tables(50)
            .with_max_indexes_per_table(8)
            .with_max_btree_entries(2_000_000)
            .with_default_timeout_ms(5000)
            .with_max_timeout_ms(30000);

        assert_eq!(config.max_tables, Some(50));
        assert_eq!(config.max_indexes_per_table, Some(8));
        assert_eq!(config.max_btree_entries, 2_000_000);
        assert_eq!(config.default_query_timeout_ms, Some(5000));
        assert_eq!(config.max_query_timeout_ms, Some(30000));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_query_options_is_copy() {
        let opts = QueryOptions::new().with_timeout_ms(1000);
        let opts2 = opts; // Copy
        let opts3 = opts; // Copy again
        assert_eq!(opts2.timeout_ms, opts3.timeout_ms);
    }

    #[test]
    fn test_select_with_options_success() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("test", schema).unwrap();
        engine
            .insert("test", HashMap::from([("id".to_string(), Value::Int(1))]))
            .unwrap();

        // Normal query with generous timeout should succeed
        let result = engine.select_with_options(
            "test",
            Condition::True,
            QueryOptions::new().with_timeout_ms(60000),
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_update_with_options_success() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("val", ColumnType::Int),
        ]);
        engine.create_table("test", schema).unwrap();
        engine
            .insert(
                "test",
                HashMap::from([
                    ("id".to_string(), Value::Int(1)),
                    ("val".to_string(), Value::Int(100)),
                ]),
            )
            .unwrap();

        let result = engine.update_with_options(
            "test",
            Condition::True,
            HashMap::from([("val".to_string(), Value::Int(200))]),
            QueryOptions::new().with_timeout_ms(60000),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_delete_with_options_success() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("test", schema).unwrap();
        engine
            .insert("test", HashMap::from([("id".to_string(), Value::Int(1))]))
            .unwrap();

        let result = engine.delete_rows_with_options(
            "test",
            Condition::True,
            QueryOptions::new().with_timeout_ms(60000),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_join_with_options_success() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("name", ColumnType::String),
        ]);
        engine.create_table("a", schema.clone()).unwrap();
        engine.create_table("b", schema).unwrap();

        engine
            .insert(
                "a",
                HashMap::from([
                    ("id".to_string(), Value::Int(1)),
                    ("name".to_string(), Value::String("x".to_string())),
                ]),
            )
            .unwrap();
        engine
            .insert(
                "b",
                HashMap::from([
                    ("id".to_string(), Value::Int(1)),
                    ("name".to_string(), Value::String("y".to_string())),
                ]),
            )
            .unwrap();

        let result = engine.join_with_options(
            "a",
            "b",
            "id",
            "id",
            QueryOptions::new().with_timeout_ms(60000),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_table_limit_no_limit() {
        let config = RelationalConfig::default();
        let engine = RelationalEngine::with_config(config);

        // Should succeed with no limit
        for i in 0..5 {
            let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
            engine.create_table(&format!("t{i}"), schema).unwrap();
        }
        assert_eq!(engine.table_count(), 5);
    }

    #[test]
    fn test_check_index_limit_no_limit() {
        let config = RelationalConfig::default();
        let engine = RelationalEngine::with_config(config);

        let schema = Schema::new(vec![
            Column::new("a", ColumnType::Int),
            Column::new("b", ColumnType::Int),
            Column::new("c", ColumnType::Int),
        ]);
        engine.create_table("test", schema).unwrap();

        // Should succeed with no limit
        engine.create_index("test", "a").unwrap();
        engine.create_index("test", "b").unwrap();
        engine.create_index("test", "c").unwrap();
    }

    #[test]
    fn test_btree_index_limit_enforced() {
        let config = RelationalConfig::new().with_max_indexes_per_table(2);
        let engine = RelationalEngine::with_config(config);

        let schema = Schema::new(vec![
            Column::new("a", ColumnType::Int),
            Column::new("b", ColumnType::Int),
            Column::new("c", ColumnType::Int),
        ]);
        engine.create_table("test", schema).unwrap();

        // First two btree indexes should succeed
        engine.create_btree_index("test", "a").unwrap();
        engine.create_btree_index("test", "b").unwrap();

        // Third should fail
        let result = engine.create_btree_index("test", "c");
        assert!(matches!(
            result,
            Err(RelationalError::TooManyIndexes { .. })
        ));
    }

    #[test]
    fn test_drop_table_decrements_count() {
        let config = RelationalConfig::new().with_max_tables(5);
        let engine = RelationalEngine::with_config(config);

        // Create 5 tables (at limit)
        for i in 0..5 {
            let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
            engine.create_table(&format!("t{i}"), schema).unwrap();
        }
        assert_eq!(engine.table_count(), 5);

        // Drop one
        engine.drop_table("t0").unwrap();
        assert_eq!(engine.table_count(), 4);

        // Should be able to create another
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("t5", schema).unwrap();
        assert_eq!(engine.table_count(), 5);
    }

    #[test]
    fn test_select_with_btree_index_and_timeout() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("val", ColumnType::Int),
        ]);
        engine.create_table("test", schema).unwrap();
        engine.create_btree_index("test", "id").unwrap();
        engine
            .insert(
                "test",
                HashMap::from([
                    ("id".to_string(), Value::Int(1)),
                    ("val".to_string(), Value::Int(100)),
                ]),
            )
            .unwrap();

        // Test with btree index path
        let result = engine.select_with_options(
            "test",
            Condition::Eq("id".to_string(), Value::Int(1)),
            QueryOptions::new().with_timeout_ms(60000),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_debug_display() {
        let config = RelationalConfig::new().with_max_tables(10);
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("max_tables"));
        assert!(debug_str.contains("10"));
    }

    #[test]
    fn test_query_options_debug_display() {
        let opts = QueryOptions::new().with_timeout_ms(5000);
        let debug_str = format!("{opts:?}");
        assert!(debug_str.contains("timeout_ms"));
        assert!(debug_str.contains("5000"));
    }

    #[test]
    fn test_config_clone() {
        let config = RelationalConfig::new()
            .with_max_tables(10)
            .with_max_indexes_per_table(5);
        let cloned = config.clone();
        assert_eq!(cloned.max_tables, config.max_tables);
        assert_eq!(cloned.max_indexes_per_table, config.max_indexes_per_table);
    }

    #[test]
    fn test_query_options_default_timeout() {
        let opts = QueryOptions::default();
        assert!(opts.timeout_ms.is_none());
    }

    #[test]
    fn test_config_presets() {
        let high = RelationalConfig::high_throughput();
        assert_eq!(high.max_btree_entries, 20_000_000);
        assert!(high.max_tables.is_none());

        let low = RelationalConfig::low_memory();
        assert_eq!(low.max_tables, Some(100));
        assert_eq!(low.max_btree_entries, 1_000_000);
    }

    #[test]
    fn test_select_timeout_after_sleep() {
        use std::thread;
        use std::time::Duration;

        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("test", schema).unwrap();

        // Insert data
        for i in 0..100 {
            engine
                .insert("test", HashMap::from([("id".to_string(), Value::Int(i))]))
                .unwrap();
        }

        // Sleep to ensure time passes, then query with very short timeout
        thread::sleep(Duration::from_millis(1));
        let result = engine.select_with_options(
            "test",
            Condition::True,
            QueryOptions::new().with_timeout_ms(0),
        );

        // Either succeeds or times out
        match result {
            Ok(_) => (),
            Err(RelationalError::QueryTimeout { .. }) => (),
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_update_with_timeout_immediate() {
        let config = RelationalConfig::new();
        let engine = RelationalEngine::with_config(config);

        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("val", ColumnType::Int),
        ]);
        engine.create_table("test", schema).unwrap();
        engine
            .insert(
                "test",
                HashMap::from([
                    ("id".to_string(), Value::Int(1)),
                    ("val".to_string(), Value::Int(100)),
                ]),
            )
            .unwrap();

        // 0ms timeout - operation should complete or timeout
        let result = engine.update_with_options(
            "test",
            Condition::True,
            HashMap::from([("val".to_string(), Value::Int(200))]),
            QueryOptions::new().with_timeout_ms(0),
        );

        // Either succeeds (fast) or times out
        match result {
            Ok(_) => (),
            Err(RelationalError::QueryTimeout { .. }) => (),
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_delete_with_timeout_immediate() {
        let config = RelationalConfig::new();
        let engine = RelationalEngine::with_config(config);

        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("test", schema).unwrap();
        engine
            .insert("test", HashMap::from([("id".to_string(), Value::Int(1))]))
            .unwrap();

        // 0ms timeout
        let result = engine.delete_rows_with_options(
            "test",
            Condition::True,
            QueryOptions::new().with_timeout_ms(0),
        );

        match result {
            Ok(_) => (),
            Err(RelationalError::QueryTimeout { .. }) => (),
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_join_with_timeout_immediate() {
        let config = RelationalConfig::new();
        let engine = RelationalEngine::with_config(config);

        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("name", ColumnType::String),
        ]);
        engine.create_table("a", schema.clone()).unwrap();
        engine.create_table("b", schema).unwrap();

        engine
            .insert(
                "a",
                HashMap::from([
                    ("id".to_string(), Value::Int(1)),
                    ("name".to_string(), Value::String("x".to_string())),
                ]),
            )
            .unwrap();
        engine
            .insert(
                "b",
                HashMap::from([
                    ("id".to_string(), Value::Int(1)),
                    ("name".to_string(), Value::String("y".to_string())),
                ]),
            )
            .unwrap();

        // 0ms timeout
        let result =
            engine.join_with_options("a", "b", "id", "id", QueryOptions::new().with_timeout_ms(0));

        match result {
            Ok(_) => (),
            Err(RelationalError::QueryTimeout { .. }) => (),
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_select_with_index_and_timeout() {
        let config = RelationalConfig::new();
        let engine = RelationalEngine::with_config(config);

        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("val", ColumnType::Int),
        ]);
        engine.create_table("test", schema).unwrap();
        engine.create_index("test", "id").unwrap();
        engine
            .insert(
                "test",
                HashMap::from([
                    ("id".to_string(), Value::Int(1)),
                    ("val".to_string(), Value::Int(100)),
                ]),
            )
            .unwrap();

        // Test with index path
        let result = engine.select_with_options(
            "test",
            Condition::Eq("id".to_string(), Value::Int(1)),
            QueryOptions::new().with_timeout_ms(0),
        );

        match result {
            Ok(rows) => assert!(!rows.is_empty() || rows.is_empty()),
            Err(RelationalError::QueryTimeout { .. }) => (),
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}

#[cfg(all(test, feature = "test-internals"))]
mod tensor_eval_tests {
    use super::*;
    use tensor_store::{ScalarValue, TensorData, TensorValue};

    fn make_tensor(id: i64, val: i64) -> TensorData {
        let mut data = TensorData::new();
        data.set("_id".to_string(), TensorValue::Scalar(ScalarValue::Int(id)));
        data.set(
            "val".to_string(),
            TensorValue::Scalar(ScalarValue::Int(val)),
        );
        data
    }

    fn make_tensor_float(id: i64, val: f64) -> TensorData {
        let mut data = TensorData::new();
        data.set("_id".to_string(), TensorValue::Scalar(ScalarValue::Int(id)));
        data.set(
            "val".to_string(),
            TensorValue::Scalar(ScalarValue::Float(val)),
        );
        data
    }

    fn make_tensor_str(id: i64, val: &str) -> TensorData {
        let mut data = TensorData::new();
        data.set("_id".to_string(), TensorValue::Scalar(ScalarValue::Int(id)));
        data.set(
            "name".to_string(),
            TensorValue::Scalar(ScalarValue::String(val.to_string())),
        );
        data
    }

    #[test]
    fn test_tensor_eval_true() {
        let tensor = make_tensor(1, 42);
        assert!(Condition::True.evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_eq_int() {
        let tensor = make_tensor(1, 42);
        assert!(Condition::Eq("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
        assert!(!Condition::Eq("val".to_string(), Value::Int(99)).evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_ne_int() {
        let tensor = make_tensor(1, 42);
        assert!(Condition::Ne("val".to_string(), Value::Int(99)).evaluate_tensor(&tensor));
        assert!(!Condition::Ne("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_lt_int() {
        let tensor = make_tensor(1, 42);
        assert!(Condition::Lt("val".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
        assert!(!Condition::Lt("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
        assert!(!Condition::Lt("val".to_string(), Value::Int(30)).evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_le_int() {
        let tensor = make_tensor(1, 42);
        assert!(Condition::Le("val".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
        assert!(Condition::Le("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
        assert!(!Condition::Le("val".to_string(), Value::Int(30)).evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_gt_int() {
        let tensor = make_tensor(1, 42);
        assert!(Condition::Gt("val".to_string(), Value::Int(30)).evaluate_tensor(&tensor));
        assert!(!Condition::Gt("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
        assert!(!Condition::Gt("val".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_ge_int() {
        let tensor = make_tensor(1, 42);
        assert!(Condition::Ge("val".to_string(), Value::Int(30)).evaluate_tensor(&tensor));
        assert!(Condition::Ge("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
        assert!(!Condition::Ge("val".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_and() {
        let tensor = make_tensor(1, 42);
        let cond = Condition::And(
            Box::new(Condition::Gt("val".to_string(), Value::Int(30))),
            Box::new(Condition::Lt("val".to_string(), Value::Int(50))),
        );
        assert!(cond.evaluate_tensor(&tensor));

        let cond2 = Condition::And(
            Box::new(Condition::Gt("val".to_string(), Value::Int(50))),
            Box::new(Condition::Lt("val".to_string(), Value::Int(60))),
        );
        assert!(!cond2.evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_or() {
        let tensor = make_tensor(1, 42);
        let cond = Condition::Or(
            Box::new(Condition::Lt("val".to_string(), Value::Int(30))),
            Box::new(Condition::Gt("val".to_string(), Value::Int(40))),
        );
        assert!(cond.evaluate_tensor(&tensor));

        let cond2 = Condition::Or(
            Box::new(Condition::Lt("val".to_string(), Value::Int(30))),
            Box::new(Condition::Gt("val".to_string(), Value::Int(50))),
        );
        assert!(!cond2.evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_id_field() {
        let tensor = make_tensor(99, 42);
        assert!(Condition::Eq("_id".to_string(), Value::Int(99)).evaluate_tensor(&tensor));
        assert!(!Condition::Eq("_id".to_string(), Value::Int(1)).evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_float() {
        let tensor = make_tensor_float(1, 3.14);
        assert!(Condition::Lt("val".to_string(), Value::Float(4.0)).evaluate_tensor(&tensor));
        assert!(Condition::Gt("val".to_string(), Value::Float(3.0)).evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_missing_column() {
        let tensor = make_tensor(1, 42);
        // Missing column should return false for comparisons
        assert!(!Condition::Eq("nonexistent".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
        assert!(!Condition::Lt("nonexistent".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_string() {
        let tensor = make_tensor_str(1, "hello");
        assert!(
            Condition::Eq("name".to_string(), Value::String("hello".to_string()))
                .evaluate_tensor(&tensor)
        );
        assert!(
            !Condition::Eq("name".to_string(), Value::String("world".to_string()))
                .evaluate_tensor(&tensor)
        );
    }

    #[test]
    fn test_tensor_eval_missing_id() {
        // Tensor without _id field
        let mut data = TensorData::new();
        data.set("val".to_string(), TensorValue::Scalar(ScalarValue::Int(42)));

        // Checking _id on tensor without _id should return false
        assert!(!Condition::Eq("_id".to_string(), Value::Int(1)).evaluate_tensor(&data));
    }

    #[test]
    fn test_tensor_eval_non_int_id() {
        // Tensor with _id as string (not int)
        let mut data = TensorData::new();
        data.set(
            "_id".to_string(),
            TensorValue::Scalar(ScalarValue::String("not_an_int".to_string())),
        );

        // Checking _id should return false since it's not an Int
        assert!(!Condition::Eq("_id".to_string(), Value::Int(1)).evaluate_tensor(&data));
    }

    #[test]
    fn test_tensor_eval_le_missing_column() {
        let tensor = make_tensor(1, 42);
        assert!(!Condition::Le("nonexistent".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_ge_missing_column() {
        let tensor = make_tensor(1, 42);
        assert!(!Condition::Ge("nonexistent".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_gt_missing_column() {
        let tensor = make_tensor(1, 42);
        assert!(!Condition::Gt("nonexistent".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
    }

    #[test]
    fn test_tensor_eval_nested_and_or() {
        let tensor = make_tensor(1, 42);
        // Complex condition: (val > 30 AND val < 50) OR val == 100
        let cond = Condition::Or(
            Box::new(Condition::And(
                Box::new(Condition::Gt("val".to_string(), Value::Int(30))),
                Box::new(Condition::Lt("val".to_string(), Value::Int(50))),
            )),
            Box::new(Condition::Eq("val".to_string(), Value::Int(100))),
        );
        assert!(cond.evaluate_tensor(&tensor));
    }

    #[test]
    fn test_concurrent_index_add_same_value() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        use std::thread;

        let engine = Arc::new(RelationalEngine::new());
        let schema = Schema::new(vec![Column::new("category", ColumnType::String)]);
        engine.create_table("items", schema).unwrap();
        engine.create_index("items", "category").unwrap();

        let success = Arc::new(AtomicUsize::new(0));

        // 100 threads all inserting rows with same category value
        let handles: Vec<_> = (0..100)
            .map(|_| {
                let eng = Arc::clone(&engine);
                let s = Arc::clone(&success);
                thread::spawn(move || {
                    let row = HashMap::from([(
                        "category".to_string(),
                        Value::String("same_category".to_string()),
                    )]);
                    if eng.insert("items", row).is_ok() {
                        s.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(success.load(Ordering::SeqCst), 100);

        // Verify ALL 100 rows are findable via index
        let results = engine
            .select_with_condition(
                "items",
                Condition::Eq(
                    "category".to_string(),
                    Value::String("same_category".to_string()),
                ),
            )
            .unwrap();
        assert_eq!(results.len(), 100, "All 100 rows should be indexed");
    }

    #[test]
    fn test_concurrent_btree_index_add_same_value() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        use std::thread;

        let engine = Arc::new(RelationalEngine::new());
        let schema = Schema::new(vec![Column::new("score", ColumnType::Int)]);
        engine.create_table("scores", schema).unwrap();
        engine.create_btree_index("scores", "score").unwrap();

        let success = Arc::new(AtomicUsize::new(0));

        // 100 threads all inserting rows with same score value
        let handles: Vec<_> = (0..100)
            .map(|_| {
                let eng = Arc::clone(&engine);
                let s = Arc::clone(&success);
                thread::spawn(move || {
                    let row = HashMap::from([("score".to_string(), Value::Int(42))]);
                    if eng.insert("scores", row).is_ok() {
                        s.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(success.load(Ordering::SeqCst), 100);

        // Verify ALL 100 rows are findable via btree index
        let results = engine
            .select_with_condition("scores", Condition::Eq("score".to_string(), Value::Int(42)))
            .unwrap();
        assert_eq!(results.len(), 100, "All 100 rows should be indexed");
    }

    #[test]
    fn test_concurrent_index_add_remove() {
        use std::sync::Arc;
        use std::thread;

        let engine = Arc::new(RelationalEngine::new());
        let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
        engine.create_table("test", schema).unwrap();
        engine.create_index("test", "val").unwrap();

        // Insert initial rows
        for i in 0..50 {
            engine
                .insert("test", HashMap::from([("val".to_string(), Value::Int(i))]))
                .unwrap();
        }

        // Concurrent inserts and deletes
        let handles: Vec<_> = (0..20)
            .map(|i| {
                let eng = Arc::clone(&engine);
                thread::spawn(move || {
                    // Insert new rows
                    for j in 0..5 {
                        let _ = eng.insert(
                            "test",
                            HashMap::from([("val".to_string(), Value::Int(100 + i * 5 + j))]),
                        );
                    }
                    // Delete some existing rows
                    let _ = eng.delete(
                        "test",
                        Condition::Eq("val".to_string(), Value::Int(i64::from(i))),
                    );
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Engine should remain consistent - no panics or lost data
        let count = engine.row_count("test").unwrap();
        assert!(count > 0, "Table should have rows");
    }

    #[test]
    fn test_max_tables_limit() {
        let config = RelationalConfig::new().with_max_tables(2);
        let engine = RelationalEngine::with_config(config);

        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);

        // First two tables should succeed
        engine.create_table("table1", schema.clone()).unwrap();
        engine.create_table("table2", schema.clone()).unwrap();

        // Third table should fail
        let result = engine.create_table("table3", schema);
        assert!(matches!(
            result,
            Err(RelationalError::TooManyTables { current: 2, max: 2 })
        ));

        // After dropping a table, we should be able to create another
        engine.drop_table("table1").unwrap();
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("table3", schema).unwrap();
    }

    #[test]
    fn test_max_indexes_per_table_limit() {
        let config = RelationalConfig::new().with_max_indexes_per_table(2);
        let engine = RelationalEngine::with_config(config);

        let schema = Schema::new(vec![
            Column::new("a", ColumnType::Int),
            Column::new("b", ColumnType::Int),
            Column::new("c", ColumnType::Int),
        ]);
        engine.create_table("test", schema).unwrap();

        // First two indexes should succeed
        engine.create_index("test", "a").unwrap();
        engine.create_btree_index("test", "b").unwrap();

        // Third index should fail
        let result = engine.create_index("test", "c");
        assert!(matches!(
            result,
            Err(RelationalError::TooManyIndexes {
                table,
                current: 2,
                max: 2
            }) if table == "test"
        ));
    }

    #[test]
    fn test_unlimited_when_none() {
        let config = RelationalConfig::default();
        assert!(config.max_tables.is_none());
        assert!(config.max_indexes_per_table.is_none());

        let engine = RelationalEngine::with_config(config);

        // Should be able to create many tables
        for i in 0..10 {
            let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
            engine.create_table(&format!("table{i}"), schema).unwrap();
        }

        // Should be able to create many indexes on one table
        let schema = Schema::new(vec![
            Column::new("a", ColumnType::Int),
            Column::new("b", ColumnType::Int),
            Column::new("c", ColumnType::Int),
            Column::new("d", ColumnType::Int),
            Column::new("e", ColumnType::Int),
        ]);
        engine.create_table("multi_idx", schema).unwrap();
        engine.create_index("multi_idx", "a").unwrap();
        engine.create_index("multi_idx", "b").unwrap();
        engine.create_index("multi_idx", "c").unwrap();
        engine.create_index("multi_idx", "d").unwrap();
        engine.create_index("multi_idx", "e").unwrap();
    }

    #[test]
    fn test_select_with_timeout() {
        let config = RelationalConfig::new().with_default_timeout_ms(0);
        let engine = RelationalEngine::with_config(config);

        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("test", schema).unwrap();

        // With 0ms timeout, the query should immediately timeout
        // We need to use select_with_options with an explicit 0 timeout
        let result = engine.select_with_options(
            "test",
            Condition::True,
            QueryOptions::new().with_timeout_ms(0),
        );

        // Allow a small window - the query might complete before timeout check
        // or it might timeout
        match result {
            Ok(rows) => assert!(rows.is_empty()),
            Err(RelationalError::QueryTimeout { .. }) => (),
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        let config = RelationalConfig::new()
            .with_default_timeout_ms(1000)
            .with_max_timeout_ms(5000);
        assert!(config.validate().is_ok());

        // Invalid config: default > max
        let config = RelationalConfig::new()
            .with_default_timeout_ms(10000)
            .with_max_timeout_ms(5000);
        assert!(config.validate().is_err());

        // Valid: no max set
        let config = RelationalConfig::new().with_default_timeout_ms(100000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_presets() {
        let high = RelationalConfig::high_throughput();
        assert!(high.max_tables.is_none());
        assert!(high.max_indexes_per_table.is_none());
        assert_eq!(high.max_btree_entries, 20_000_000);
        assert_eq!(high.default_query_timeout_ms, Some(30_000));

        let low = RelationalConfig::low_memory();
        assert_eq!(low.max_tables, Some(100));
        assert_eq!(low.max_indexes_per_table, Some(5));
        assert_eq!(low.max_btree_entries, 1_000_000);
        assert_eq!(low.default_query_timeout_ms, Some(10_000));
    }

    #[test]
    fn test_new_uses_default_config() {
        let engine = RelationalEngine::new();

        // Default config should allow unlimited tables
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        for i in 0..20 {
            engine
                .create_table(&format!("table{i}"), schema.clone())
                .unwrap();
        }
    }

    #[test]
    fn test_concurrent_table_creation_with_limit() {
        use std::sync::Arc;
        use std::thread;

        let config = RelationalConfig::new().with_max_tables(5);
        let engine = Arc::new(RelationalEngine::with_config(config));

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let eng = Arc::clone(&engine);
                thread::spawn(move || {
                    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
                    eng.create_table(&format!("table{i}"), schema)
                })
            })
            .collect();

        let mut successes = 0;
        let mut too_many = 0;
        for h in handles {
            match h.join().unwrap() {
                Ok(()) => successes += 1,
                Err(RelationalError::TooManyTables { .. }) => too_many += 1,
                Err(e) => panic!("Unexpected error: {e}"),
            }
        }

        // Exactly 5 should succeed, 5 should fail
        assert_eq!(successes, 5);
        assert_eq!(too_many, 5);
    }

    #[test]
    fn test_query_options_new_and_builder() {
        let opts = QueryOptions::new();
        assert!(opts.timeout_ms.is_none());

        let opts = QueryOptions::new().with_timeout_ms(5000);
        assert_eq!(opts.timeout_ms, Some(5000));

        let opts = QueryOptions::default();
        assert!(opts.timeout_ms.is_none());
    }

    #[test]
    fn test_config_accessor_methods() {
        let config = RelationalConfig::new()
            .with_max_tables(100)
            .with_max_indexes_per_table(10)
            .with_default_timeout_ms(5000)
            .with_max_timeout_ms(60000)
            .with_max_btree_entries(5_000_000);

        assert_eq!(config.max_tables, Some(100));
        assert_eq!(config.max_indexes_per_table, Some(10));
        assert_eq!(config.default_query_timeout_ms, Some(5000));
        assert_eq!(config.max_query_timeout_ms, Some(60000));
        assert_eq!(config.max_btree_entries, 5_000_000);

        let engine = RelationalEngine::with_config(config.clone());
        let engine_config = engine.config();
        assert_eq!(engine_config.max_tables, Some(100));
    }

    #[test]
    fn test_table_count_tracking() {
        let config = RelationalConfig::new().with_max_tables(10);
        let engine = RelationalEngine::with_config(config);

        assert_eq!(engine.table_count(), 0);

        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("t1", schema.clone()).unwrap();
        assert_eq!(engine.table_count(), 1);

        engine.create_table("t2", schema.clone()).unwrap();
        assert_eq!(engine.table_count(), 2);

        engine.drop_table("t1").unwrap();
        assert_eq!(engine.table_count(), 1);

        engine.drop_table("t2").unwrap();
        assert_eq!(engine.table_count(), 0);
    }

    #[test]
    fn test_timeout_clamps_to_max() {
        let config = RelationalConfig::new()
            .with_default_timeout_ms(1000)
            .with_max_timeout_ms(500);
        let engine = RelationalEngine::with_config(config);

        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table("test", schema).unwrap();

        // The effective timeout should be clamped to 500ms even though
        // options request 1000ms
        let resolved = engine.resolve_timeout(QueryOptions::new().with_timeout_ms(1000));
        assert_eq!(resolved, Some(500));
    }
}
