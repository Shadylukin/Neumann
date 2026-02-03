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
//! **Persisted and auto-recovered:**
//! - Table schemas and metadata
//! - Row data (columnar storage)
//! - B-tree indexes (automatically rebuilt on recovery)
//! - Row counters (automatically rebuilt from scanning data)
//!
//! **NOT persisted (must recreate after recovery):**
//! - Hash indexes: call [`RelationalEngine::create_index`] after recovery
//! - Transaction state: transactions are aborted on crash
//!
//! # Recovery Example
//!
//! ```ignore
//! // Open durable engine - B-tree indexes are automatically rebuilt
//! let engine = RelationalEngine::recover(
//!     "data/wal",
//!     &WalConfig::default(),
//!     Some(Path::new("data/snapshot")),
//! )?;
//!
//! // Only hash indexes need to be recreated after recovery
//! for table in engine.list_tables() {
//!     engine.create_index(&table, "user_id")?;
//!     // B-tree indexes are already restored!
//! }
//! ```

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    hash::{Hash, Hasher},
    path::Path,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use dashmap::DashMap;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tensor_store::RelationalSlab;
pub use tensor_store::WalConfig;
pub(crate) use tensor_store::{
    ColumnDef as SlabColumnDef, ColumnType as SlabColumnType, ColumnValue as SlabColumnValue,
    RowId as SlabRowId, TableSchema as SlabTableSchema,
};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorStoreError, TensorValue};
use tracing::{debug, info, instrument, warn};

pub mod transaction;
pub(crate) use transaction::{Deadline, IndexChange, UndoEntry};
pub use transaction::{Transaction, TransactionManager, TxPhase};

mod simd;

pub mod cursor;
pub mod observability;

pub use cursor::{CursorBuilder, StreamingCursor};
pub use observability::{IndexMissReport, IndexTracker, QueryMetrics};

/// Column data type.
#[non_exhaustive]
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
    /// Binary data.
    Bytes,
    /// JSON value.
    Json,
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
    /// Table constraints (primary key, unique, foreign key, not null).
    #[serde(default)]
    pub constraints: Vec<Constraint>,
}

impl Schema {
    /// Creates a schema from columns with no constraints.
    #[must_use]
    pub const fn new(columns: Vec<Column>) -> Self {
        Self {
            columns,
            constraints: Vec::new(),
        }
    }

    /// Creates a schema with columns and constraints.
    #[must_use]
    pub const fn with_constraints(columns: Vec<Column>, constraints: Vec<Constraint>) -> Self {
        Self {
            columns,
            constraints,
        }
    }

    /// Finds a column by name.
    #[must_use]
    pub fn get_column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Returns a reference to the constraints.
    #[must_use]
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Adds a constraint to the schema.
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }
}

/// Referential action for foreign key constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ReferentialAction {
    /// Prevent the operation if it would violate the constraint.
    #[default]
    Restrict,
    /// Cascade the operation to referencing rows.
    Cascade,
    /// Set the referencing column(s) to NULL.
    SetNull,
    /// Set the referencing column(s) to their default value.
    SetDefault,
    /// Same as Restrict but checked at transaction commit.
    NoAction,
}

/// Foreign key constraint definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKeyConstraint {
    /// Constraint name.
    pub name: String,
    /// Columns in this table that reference another table.
    pub columns: Vec<String>,
    /// Name of the referenced table.
    pub referenced_table: String,
    /// Columns in the referenced table.
    pub referenced_columns: Vec<String>,
    /// Action when referenced row is deleted.
    pub on_delete: ReferentialAction,
    /// Action when referenced row is updated.
    pub on_update: ReferentialAction,
}

impl ForeignKeyConstraint {
    /// Creates a new foreign key constraint.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        columns: Vec<String>,
        referenced_table: impl Into<String>,
        referenced_columns: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            columns,
            referenced_table: referenced_table.into(),
            referenced_columns,
            on_delete: ReferentialAction::default(),
            on_update: ReferentialAction::default(),
        }
    }

    /// Sets the ON DELETE action.
    #[must_use]
    pub const fn on_delete(mut self, action: ReferentialAction) -> Self {
        self.on_delete = action;
        self
    }

    /// Sets the ON UPDATE action.
    #[must_use]
    pub const fn on_update(mut self, action: ReferentialAction) -> Self {
        self.on_update = action;
        self
    }
}

/// Table constraint definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Primary key constraint (unique + not null).
    PrimaryKey {
        /// Constraint name.
        name: String,
        /// Columns that form the primary key.
        columns: Vec<String>,
    },
    /// Unique constraint.
    Unique {
        /// Constraint name.
        name: String,
        /// Columns that must be unique together.
        columns: Vec<String>,
    },
    /// Foreign key constraint.
    ForeignKey(ForeignKeyConstraint),
    /// Not null constraint on a single column.
    NotNull {
        /// Constraint name.
        name: String,
        /// Column that cannot be null.
        column: String,
    },
}

impl Constraint {
    /// Returns the constraint name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::PrimaryKey { name, .. }
            | Self::Unique { name, .. }
            | Self::NotNull { name, .. } => name,
            Self::ForeignKey(fk) => &fk.name,
        }
    }

    /// Creates a primary key constraint.
    #[must_use]
    pub fn primary_key(name: impl Into<String>, columns: Vec<String>) -> Self {
        Self::PrimaryKey {
            name: name.into(),
            columns,
        }
    }

    /// Creates a unique constraint.
    #[must_use]
    pub fn unique(name: impl Into<String>, columns: Vec<String>) -> Self {
        Self::Unique {
            name: name.into(),
            columns,
        }
    }

    /// Creates a not null constraint.
    #[must_use]
    pub fn not_null(name: impl Into<String>, column: impl Into<String>) -> Self {
        Self::NotNull {
            name: name.into(),
            column: column.into(),
        }
    }

    /// Creates a foreign key constraint.
    #[must_use]
    pub const fn foreign_key(fk: ForeignKeyConstraint) -> Self {
        Self::ForeignKey(fk)
    }
}

/// Query value type.
#[non_exhaustive]
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
    /// Binary data.
    Bytes(Vec<u8>),
    /// JSON value.
    Json(serde_json::Value),
}

impl Value {
    fn from_scalar(scalar: &ScalarValue) -> Self {
        match scalar {
            ScalarValue::Int(v) => Self::Int(*v),
            ScalarValue::Float(v) => Self::Float(*v),
            ScalarValue::String(v) => Self::String(v.clone()),
            ScalarValue::Bool(v) => Self::Bool(*v),
            ScalarValue::Bytes(v) => Self::Bytes(v.clone()),
            ScalarValue::Null => Self::Null,
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
                | (Self::Bytes(_), ColumnType::Bytes)
                | (Self::Json(_), ColumnType::Json)
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
            Self::Bytes(v) => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                v.hash(&mut hasher);
                format!("y:{:x}", hasher.finish())
            },
            Self::Json(v) => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                v.to_string().hash(&mut hasher);
                format!("j:{:x}", hasher.finish())
            },
        }
    }

    fn partial_cmp_value(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => Some(a.cmp(b)),
            (Self::Float(a), Self::Float(b)) => a.partial_cmp(b),
            (Self::String(a), Self::String(b)) => Some(a.cmp(b)),
            (Self::Bytes(a), Self::Bytes(b)) => Some(a.cmp(b)),
            (Self::Json(a), Self::Json(b)) => Some(a.to_string().cmp(&b.to_string())),
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
            Self::Bytes(v) => {
                // Hex-encode bytes for lexicographic ordering
                format!("y{}", hex::encode(v))
            },
            Self::Json(v) => {
                // Use compact JSON string representation
                format!("j{v}")
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
            Self::Bytes(b) => !b.is_empty(),
            Self::Json(j) => !j.is_null(),
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
            ColumnType::Bytes => Self::Bytes,
            ColumnType::Json => Self::Json,
        }
    }
}

impl From<&SlabColumnType> for ColumnType {
    fn from(ct: &SlabColumnType) -> Self {
        match ct {
            SlabColumnType::Int => Self::Int,
            SlabColumnType::Float => Self::Float,
            SlabColumnType::Bool => Self::Bool,
            SlabColumnType::String => Self::String,
            SlabColumnType::Bytes => Self::Bytes,
            SlabColumnType::Json => Self::Json,
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
            Value::Bytes(b) => Self::Bytes(b.clone()),
            Value::Json(j) => Self::Json(j.to_string()),
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
            SlabColumnValue::Bytes(b) => Self::Bytes(b),
            SlabColumnValue::Json(j) => Self::Json(serde_json::from_str(&j).unwrap_or_else(|e| {
                warn!(error = %e, "Malformed JSON in column value, storing as string");
                serde_json::Value::String(j)
            })),
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
    pub fn get_with_id(&self, column: &str) -> Option<Value> {
        if column == "_id" {
            return Some(Value::Int(i64::try_from(self.id).unwrap_or(i64::MAX)));
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
#[non_exhaustive]
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

    /// Evaluates with depth tracking to prevent stack overflow.
    ///
    /// # Errors
    ///
    /// Returns `ConditionTooDeep` if the condition tree exceeds `max_depth`.
    pub fn evaluate_with_depth(&self, row: &Row, depth: usize, max_depth: usize) -> Result<bool> {
        use std::cmp::Ordering;

        if depth > max_depth {
            return Err(RelationalError::ConditionTooDeep { max_depth });
        }
        match self {
            Self::True => Ok(true),
            Self::Eq(col, val) => Ok(row.get_with_id(col).as_ref() == Some(val)),
            Self::Ne(col, val) => Ok(row.get_with_id(col).as_ref() != Some(val)),
            Self::Lt(col, val) => Ok(Self::compare_ord(row, col, val, Ordering::Less)),
            Self::Le(col, val) => Ok(Self::compare_ord_le(row, col, val)),
            Self::Gt(col, val) => Ok(Self::compare_ord(row, col, val, Ordering::Greater)),
            Self::Ge(col, val) => Ok(Self::compare_ord_ge(row, col, val)),
            Self::And(a, b) => Ok(a.evaluate_with_depth(row, depth + 1, max_depth)?
                && b.evaluate_with_depth(row, depth + 1, max_depth)?),
            Self::Or(a, b) => Ok(a.evaluate_with_depth(row, depth + 1, max_depth)?
                || b.evaluate_with_depth(row, depth + 1, max_depth)?),
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
    #[allow(dead_code)] // Used via test-internals feature by integration/fuzz tests
    pub(crate) fn evaluate_tensor(&self, tensor: &TensorData) -> bool {
        self.evaluate_tensor_impl(tensor)
    }

    #[allow(dead_code)] // Called by evaluate_tensor which is used via test-internals feature
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

    #[allow(dead_code)] // Helper for tensor evaluation methods
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

    #[allow(dead_code)] // Helper for tensor evaluation methods
    fn tensor_field_eq(tensor: &TensorData, col: &str, val: &Value) -> bool {
        Self::tensor_get_value(tensor, col).as_ref() == Some(val)
    }

    #[allow(dead_code)] // Helper for tensor evaluation methods
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

    #[allow(dead_code)] // Helper for tensor evaluation methods
    fn tensor_compare_le(tensor: &TensorData, col: &str, val: &Value) -> bool {
        Self::tensor_get_value(tensor, col)
            .and_then(|v| v.partial_cmp_value(val))
            .is_some_and(|o| o != std::cmp::Ordering::Greater)
    }

    #[allow(dead_code)] // Helper for tensor evaluation methods
    fn tensor_compare_ge(tensor: &TensorData, col: &str, val: &Value) -> bool {
        Self::tensor_get_value(tensor, col)
            .and_then(|v| v.partial_cmp_value(val))
            .is_some_and(|o| o != std::cmp::Ordering::Less)
    }
}

/// Aggregate expression for GROUP BY queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregateExpr {
    /// COUNT(*) - counts all rows.
    CountAll,
    /// COUNT(column) - counts non-null values.
    Count(String),
    /// SUM(column) - sum of numeric values.
    Sum(String),
    /// AVG(column) - average of numeric values.
    Avg(String),
    /// MIN(column) - minimum value.
    Min(String),
    /// MAX(column) - maximum value.
    Max(String),
}

impl AggregateExpr {
    fn column_name(&self) -> Option<&str> {
        match self {
            Self::CountAll => None,
            Self::Count(c) | Self::Sum(c) | Self::Avg(c) | Self::Min(c) | Self::Max(c) => Some(c),
        }
    }

    fn result_name(&self) -> String {
        match self {
            Self::CountAll => "count_all".to_string(),
            Self::Count(c) => format!("count_{c}"),
            Self::Sum(c) => format!("sum_{c}"),
            Self::Avg(c) => format!("avg_{c}"),
            Self::Min(c) => format!("min_{c}"),
            Self::Max(c) => format!("max_{c}"),
        }
    }
}

/// Typed aggregate result value.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateValue {
    /// Count result.
    Count(u64),
    /// Sum result (numeric).
    Sum(f64),
    /// Average result (None if no values).
    Avg(Option<f64>),
    /// Minimum value (None if no values).
    Min(Option<Value>),
    /// Maximum value (None if no values).
    Max(Option<Value>),
}

impl AggregateValue {
    #[allow(clippy::cast_possible_wrap)] // Count values are always positive and within i64 range
    fn to_value(&self) -> Value {
        match self {
            Self::Count(c) => Value::Int(*c as i64),
            Self::Sum(s) => Value::Float(*s),
            Self::Avg(Some(a)) => Value::Float(*a),
            Self::Avg(None) | Self::Min(None) | Self::Max(None) => Value::Null,
            Self::Min(Some(v)) | Self::Max(Some(v)) => v.clone(),
        }
    }
}

/// Result of a grouped aggregate query.
#[derive(Debug, Clone)]
pub struct GroupedRow {
    /// The group key values as (column name, value) pairs.
    pub group_key: Vec<(String, Value)>,
    /// The computed aggregates as (name, value) pairs.
    pub aggregates: Vec<(String, AggregateValue)>,
}

impl GroupedRow {
    /// Gets a group key value by column name.
    #[must_use]
    pub fn get_key(&self, column: &str) -> Option<&Value> {
        self.group_key
            .iter()
            .find(|(name, _)| name == column)
            .map(|(_, v)| v)
    }

    /// Gets an aggregate value by name.
    #[must_use]
    pub fn get_aggregate(&self, name: &str) -> Option<&AggregateValue> {
        self.aggregates
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
    }
}

/// Reference to an aggregate in HAVING clause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregateRef {
    /// COUNT(*)
    CountAll,
    /// COUNT(column)
    Count(String),
    /// SUM(column)
    Sum(String),
    /// AVG(column)
    Avg(String),
    /// MIN(column)
    Min(String),
    /// MAX(column)
    Max(String),
}

impl AggregateRef {
    fn result_name(&self) -> String {
        match self {
            Self::CountAll => "count_all".to_string(),
            Self::Count(c) => format!("count_{c}"),
            Self::Sum(c) => format!("sum_{c}"),
            Self::Avg(c) => format!("avg_{c}"),
            Self::Min(c) => format!("min_{c}"),
            Self::Max(c) => format!("max_{c}"),
        }
    }
}

/// HAVING condition for filtering groups after aggregation.
#[derive(Debug, Clone)]
pub enum HavingCondition {
    /// Greater than comparison.
    Gt(AggregateRef, Value),
    /// Greater than or equal comparison.
    Ge(AggregateRef, Value),
    /// Less than comparison.
    Lt(AggregateRef, Value),
    /// Less than or equal comparison.
    Le(AggregateRef, Value),
    /// Equality comparison.
    Eq(AggregateRef, Value),
    /// Not equal comparison.
    Ne(AggregateRef, Value),
    /// Logical AND of two conditions.
    And(Box<Self>, Box<Self>),
    /// Logical OR of two conditions.
    Or(Box<Self>, Box<Self>),
}

impl HavingCondition {
    /// Evaluates with depth tracking to prevent stack overflow.
    #[allow(dead_code)] // Defensive API for depth-limited HAVING queries
    fn evaluate_with_depth(
        &self,
        row: &GroupedRow,
        depth: usize,
        max_depth: usize,
    ) -> Result<bool> {
        if depth > max_depth {
            return Err(RelationalError::ConditionTooDeep { max_depth });
        }
        match self {
            Self::Gt(agg_ref, val) => Ok(Self::compare_agg(
                row,
                agg_ref,
                val,
                std::cmp::Ordering::Greater,
            )),
            Self::Ge(agg_ref, val) => {
                Ok(
                    Self::compare_agg(row, agg_ref, val, std::cmp::Ordering::Greater)
                        || Self::compare_agg(row, agg_ref, val, std::cmp::Ordering::Equal),
                )
            },
            Self::Lt(agg_ref, val) => Ok(Self::compare_agg(
                row,
                agg_ref,
                val,
                std::cmp::Ordering::Less,
            )),
            Self::Le(agg_ref, val) => {
                Ok(
                    Self::compare_agg(row, agg_ref, val, std::cmp::Ordering::Less)
                        || Self::compare_agg(row, agg_ref, val, std::cmp::Ordering::Equal),
                )
            },
            Self::Eq(agg_ref, val) => Ok(Self::compare_agg(
                row,
                agg_ref,
                val,
                std::cmp::Ordering::Equal,
            )),
            Self::Ne(agg_ref, val) => Ok(!Self::compare_agg(
                row,
                agg_ref,
                val,
                std::cmp::Ordering::Equal,
            )),
            Self::And(a, b) => Ok(a.evaluate_with_depth(row, depth + 1, max_depth)?
                && b.evaluate_with_depth(row, depth + 1, max_depth)?),
            Self::Or(a, b) => Ok(a.evaluate_with_depth(row, depth + 1, max_depth)?
                || b.evaluate_with_depth(row, depth + 1, max_depth)?),
        }
    }

    fn compare_agg(
        row: &GroupedRow,
        agg_ref: &AggregateRef,
        val: &Value,
        ord: std::cmp::Ordering,
    ) -> bool {
        let name = agg_ref.result_name();
        row.get_aggregate(&name)
            .map(AggregateValue::to_value)
            .and_then(|v| v.partial_cmp_value(val))
            .is_some_and(|o| o == ord)
    }
}

/// Wrapper for Value that implements Hash and Eq for use in HashSets/HashMaps.
#[derive(Debug, Clone)]
struct HashableValue(Value);

impl Hash for HashableValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash_key().hash(state);
    }
}

impl PartialEq for HashableValue {
    fn eq(&self, other: &Self) -> bool {
        self.0.hash_key() == other.0.hash_key()
    }
}

impl Eq for HashableValue {}

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
    /// Slow query warning threshold in milliseconds. Default: 100ms.
    pub slow_query_threshold_ms: u64,
    /// Maximum rows returned from a single query. `None` means unlimited.
    pub max_query_result_rows: Option<usize>,
    /// Default transaction timeout in seconds. Default: 60s.
    pub transaction_timeout_secs: u64,
    /// Default lock timeout in seconds. Default: 30s.
    pub lock_timeout_secs: u64,
    /// Maximum nesting depth for condition trees. Default: 64.
    pub max_condition_depth: usize,
}

impl Default for RelationalConfig {
    fn default() -> Self {
        Self {
            max_tables: None,
            max_indexes_per_table: None,
            max_btree_entries: 10_000_000,
            default_query_timeout_ms: Some(30_000), // 30 seconds
            max_query_timeout_ms: Some(300_000),    // 5 minutes
            slow_query_threshold_ms: 100,
            max_query_result_rows: None,
            transaction_timeout_secs: 60,
            lock_timeout_secs: 30,
            max_condition_depth: 64,
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

    /// Sets the slow query warning threshold in milliseconds.
    #[must_use]
    pub const fn with_slow_query_threshold_ms(mut self, threshold_ms: u64) -> Self {
        self.slow_query_threshold_ms = threshold_ms;
        self
    }

    /// Sets the maximum rows returned from a single query.
    #[must_use]
    pub const fn with_max_query_result_rows(mut self, max: usize) -> Self {
        self.max_query_result_rows = Some(max);
        self
    }

    /// Sets the transaction timeout in seconds.
    #[must_use]
    pub const fn with_transaction_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.transaction_timeout_secs = timeout_secs;
        self
    }

    /// Sets the lock timeout in seconds.
    #[must_use]
    pub const fn with_lock_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.lock_timeout_secs = timeout_secs;
        self
    }

    /// Sets the maximum condition nesting depth.
    #[must_use]
    pub const fn with_max_condition_depth(mut self, depth: usize) -> Self {
        self.max_condition_depth = depth;
        self
    }

    /// Configuration preset for high-throughput workloads.
    ///
    /// - No table or index limits
    /// - 30 second default query timeout
    /// - 20M B-tree entries allowed
    /// - 50ms slow query threshold
    #[must_use]
    pub const fn high_throughput() -> Self {
        Self {
            max_tables: None,
            max_indexes_per_table: None,
            max_btree_entries: 20_000_000,
            default_query_timeout_ms: Some(30_000),
            max_query_timeout_ms: Some(600_000), // 10 minutes
            slow_query_threshold_ms: 50,
            max_query_result_rows: None,
            transaction_timeout_secs: 120,
            lock_timeout_secs: 60,
            max_condition_depth: 64,
        }
    }

    /// Configuration preset for low-memory environments.
    ///
    /// - Maximum 100 tables
    /// - Maximum 5 indexes per table
    /// - 1M B-tree entries
    /// - 10 second default query timeout
    /// - 10K max result rows
    #[must_use]
    pub const fn low_memory() -> Self {
        Self {
            max_tables: Some(100),
            max_indexes_per_table: Some(5),
            max_btree_entries: 1_000_000,
            default_query_timeout_ms: Some(10_000),
            max_query_timeout_ms: Some(60_000), // 1 minute
            slow_query_threshold_ms: 100,
            max_query_result_rows: Some(10_000),
            transaction_timeout_secs: 30,
            lock_timeout_secs: 15,
            max_condition_depth: 64,
        }
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the default timeout exceeds the maximum timeout.
    #[instrument(level = "debug", skip(self))]
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

/// Options for cursor-based iteration over query results.
#[derive(Debug, Clone, Copy)]
pub struct CursorOptions {
    /// Number of rows to fetch per batch (reserved for future streaming). Default: 1000.
    pub batch_size: usize,
    /// Starting row offset. Default: 0.
    pub offset: usize,
    /// Maximum rows to return. Default: None (unlimited).
    pub limit: Option<usize>,
}

impl Default for CursorOptions {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            offset: 0,
            limit: None,
        }
    }
}

impl CursorOptions {
    /// Creates new cursor options with default batch size.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            batch_size: 1000,
            offset: 0,
            limit: None,
        }
    }

    /// Sets the batch size for fetching rows.
    #[must_use]
    pub const fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the starting offset.
    #[must_use]
    pub const fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Sets the maximum number of rows to return.
    #[must_use]
    pub const fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Iterator over query results.
///
/// Created by [`RelationalEngine::select_iter`]. Provides an iterator interface
/// for processing query results one row at a time.
pub struct RowCursor<'a> {
    #[allow(dead_code)] // Reserved for future lazy loading of additional batches
    engine: &'a RelationalEngine,
    rows: std::vec::IntoIter<Row>,
    rows_processed: usize,
    total_rows: usize,
}

impl RowCursor<'_> {
    /// Returns the number of rows processed so far.
    #[must_use]
    pub const fn rows_processed(&self) -> usize {
        self.rows_processed
    }

    /// Returns the total number of rows in the result set.
    #[must_use]
    pub const fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// Returns true if all rows have been consumed.
    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        self.rows_processed >= self.total_rows
    }
}

impl Iterator for RowCursor<'_> {
    type Item = Result<Row>;

    fn next(&mut self) -> Option<Self::Item> {
        self.rows.next().map(|row| {
            self.rows_processed += 1;
            Ok(row)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_rows.saturating_sub(self.rows_processed);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for RowCursor<'_> {}

impl std::fmt::Debug for RowCursor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowCursor")
            .field("rows_processed", &self.rows_processed)
            .field("total_rows", &self.total_rows)
            .finish()
    }
}

/// Errors from relational engine operations.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationalError {
    /// Table does not exist.
    TableNotFound(String),
    /// Table already exists.
    TableAlreadyExists(String),
    /// Column does not exist in table.
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
    /// Primary key constraint violation.
    PrimaryKeyViolation {
        /// Table name.
        table: String,
        /// Columns in the primary key.
        columns: Vec<String>,
        /// Duplicate value.
        value: String,
    },
    /// Unique constraint violation.
    UniqueViolation {
        /// Constraint name.
        constraint_name: String,
        /// Columns in the unique constraint.
        columns: Vec<String>,
        /// Duplicate value.
        value: String,
    },
    /// Foreign key constraint violation on insert/update.
    ForeignKeyViolation {
        /// Constraint name.
        constraint_name: String,
        /// Table with the foreign key.
        table: String,
        /// Referenced table.
        referenced_table: String,
    },
    /// Foreign key restrict prevents delete/update of referenced row.
    ForeignKeyRestrict {
        /// Constraint name.
        constraint_name: String,
        /// Table containing the referenced row.
        table: String,
        /// Table with the referencing foreign key.
        referencing_table: String,
        /// Number of rows that reference this row.
        row_count: usize,
    },
    /// Constraint not found.
    ConstraintNotFound {
        /// Table name.
        table: String,
        /// Constraint name.
        constraint_name: String,
    },
    /// Constraint already exists.
    ConstraintAlreadyExists {
        /// Table name.
        table: String,
        /// Constraint name.
        constraint_name: String,
    },
    /// Column has a constraint that prevents the operation.
    ColumnHasConstraint {
        /// Column name.
        column: String,
        /// Constraint name.
        constraint_name: String,
    },
    /// Cannot add column due to constraint or data issue.
    CannotAddColumn {
        /// Column name.
        column: String,
        /// Reason the column cannot be added.
        reason: String,
    },
    /// Column already exists in the table.
    ColumnAlreadyExists {
        /// Table name.
        table: String,
        /// Column name.
        column: String,
    },
    /// Condition tree exceeds maximum nesting depth.
    ConditionTooDeep {
        /// Maximum allowed depth.
        max_depth: usize,
    },
}

impl std::fmt::Display for RelationalError {
    #[allow(clippy::too_many_lines)]
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
            Self::PrimaryKeyViolation {
                table,
                columns,
                value,
            } => {
                write!(
                    f,
                    "Primary key violation on '{table}' columns [{}]: duplicate value '{value}'",
                    columns.join(", ")
                )
            },
            Self::UniqueViolation {
                constraint_name,
                columns,
                value,
            } => {
                write!(
                    f,
                    "Unique constraint '{constraint_name}' violation on columns [{}]: duplicate value '{value}'",
                    columns.join(", ")
                )
            },
            Self::ForeignKeyViolation {
                constraint_name,
                table,
                referenced_table,
            } => {
                write!(
                    f,
                    "Foreign key constraint '{constraint_name}' violation: value in '{table}' not found in '{referenced_table}'"
                )
            },
            Self::ForeignKeyRestrict {
                constraint_name,
                table,
                referencing_table,
                row_count,
            } => {
                write!(
                    f,
                    "Foreign key constraint '{constraint_name}' restricts operation: {row_count} row(s) in '{referencing_table}' reference '{table}'"
                )
            },
            Self::ConstraintNotFound {
                table,
                constraint_name,
            } => {
                write!(
                    f,
                    "Constraint '{constraint_name}' not found on table '{table}'"
                )
            },
            Self::ConstraintAlreadyExists {
                table,
                constraint_name,
            } => {
                write!(
                    f,
                    "Constraint '{constraint_name}' already exists on table '{table}'"
                )
            },
            Self::ColumnHasConstraint {
                column,
                constraint_name,
            } => {
                write!(
                    f,
                    "Column '{column}' is part of constraint '{constraint_name}'"
                )
            },
            Self::CannotAddColumn { column, reason } => {
                write!(f, "Cannot add column '{column}': {reason}")
            },
            Self::ColumnAlreadyExists { table, column } => {
                write!(f, "Column '{column}' already exists in table '{table}'")
            },
            Self::ConditionTooDeep { max_depth } => {
                write!(f, "Condition nesting exceeds maximum depth of {max_depth}")
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
    Bytes {
        dict: Vec<Vec<u8>>,
        indices: Vec<u32>,
    },
    Json {
        dict: Vec<String>,
        indices: Vec<u32>,
    },
}

#[allow(dead_code)] // Used internally for columnar operations and in tests
impl ColumnValues {
    pub const fn len(&self) -> usize {
        match self {
            Self::Int(v) => v.len(),
            Self::Float(v) => v.len(),
            Self::String { indices, .. }
            | Self::Bytes { indices, .. }
            | Self::Json { indices, .. } => indices.len(),
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
            ColumnValues::Bytes { dict, indices } => indices
                .get(idx)
                .and_then(|&i| dict.get(i as usize).map(|b| Value::Bytes(b.clone()))),
            ColumnValues::Json { dict, indices } => indices.get(idx).and_then(|&i| {
                dict.get(i as usize).map(|s| {
                    Value::Json(
                        serde_json::from_str(s)
                            .unwrap_or_else(|_| serde_json::Value::String(s.clone())),
                    )
                })
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SelectionVector {
    bitmap: Vec<u64>,
    row_count: usize,
}

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

    #[allow(dead_code)] // Used by tests for SIMD filtering verification
    pub fn bitmap_mut(&mut self) -> &mut [u64] {
        &mut self.bitmap
    }

    #[allow(dead_code)] // Used by tests for SIMD filtering verification
    pub fn bitmap(&self) -> &[u64] {
        &self.bitmap
    }

    #[allow(dead_code)] // Used by tests for SIMD filtering verification
    pub fn count(&self) -> usize {
        simd::popcount(&self.bitmap)
    }

    #[allow(dead_code)] // Used by tests for SIMD filtering verification
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
    /// Constraint cache: table name -> constraints.
    constraint_cache: DashMap<String, Vec<Constraint>>,
    /// Foreign key reference graph: `referenced_table` -> [(`referencing_table`, FK constraint)].
    fk_references: RwLock<HashMap<String, Vec<(String, ForeignKeyConstraint)>>>,
}

/// Ordered key for B-tree indexes with correct comparison semantics.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum OrderedKey {
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
    /// Binary data (lexicographic ordering).
    Bytes(Vec<u8>),
    /// JSON value (string representation for ordering).
    Json(String),
}

/// Wrapper for f64 that implements total ordering (NaN is less than all values).
#[derive(Debug, Clone, Copy)]
pub(crate) struct OrderedFloat(pub(crate) f64);

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
            Value::Bytes(b) => Self::Bytes(b.clone()),
            Value::Json(j) => Self::Json(j.to_string()),
        }
    }

    /// Parses a sortable key string back to an `OrderedKey`.
    ///
    /// Returns `None` if the format is invalid.
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)] // Math is correct for i64 range
    fn from_sortable_key(s: &str) -> Option<Self> {
        if s == "0" {
            return Some(Self::Null);
        }

        // Handle empty string case: "s" with nothing after
        if s == "s" {
            return Some(Self::String(String::new()));
        }

        // Handle empty bytes case: "y" with nothing after
        if s == "y" {
            return Some(Self::Bytes(Vec::new()));
        }

        // Handle empty JSON case: "j" with nothing after
        if s == "j" {
            return Some(Self::Json(String::new()));
        }

        if s.len() < 2 {
            return None;
        }

        let prefix = &s[..1];
        let value = &s[1..];

        match prefix {
            "i" => {
                // Parse hex and reverse the offset encoding
                let unsigned = u64::from_str_radix(value, 16).ok()?;
                let signed = (i128::from(unsigned) - i128::from(i64::MAX) - 1) as i64;
                Some(Self::Int(signed))
            },
            "f" => {
                // Parse hex and reverse the IEEE 754 encoding
                let sortable = u64::from_str_radix(value, 16).ok()?;
                // Check if original was positive (high bit set in sortable)
                let bits = if sortable & 0x8000_0000_0000_0000 != 0 {
                    sortable ^ 0x8000_0000_0000_0000 // Was positive
                } else {
                    !sortable // Was negative
                };
                Some(Self::Float(OrderedFloat(f64::from_bits(bits))))
            },
            "s" => Some(Self::String(value.to_string())),
            "b" => match value {
                "1" => Some(Self::Bool(true)),
                "0" => Some(Self::Bool(false)),
                _ => None,
            },
            "y" => {
                // Decode hex to bytes
                hex::decode(value).ok().map(Self::Bytes)
            },
            "j" => Some(Self::Json(value.to_string())),
            _ => None,
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
    #[instrument(level = "info", skip_all)]
    pub fn new() -> Self {
        info!("creating in-memory relational engine");
        Self::with_config(RelationalConfig::default())
    }

    /// Creates a new in-memory relational engine with custom configuration.
    #[must_use]
    #[instrument(level = "info", skip_all, fields(max_tables = ?config.max_tables))]
    pub fn with_config(config: RelationalConfig) -> Self {
        info!(
            max_btree_entries = config.max_btree_entries,
            slow_query_threshold_ms = config.slow_query_threshold_ms,
            transaction_timeout_secs = config.transaction_timeout_secs,
            lock_timeout_secs = config.lock_timeout_secs,
            "creating relational engine with config"
        );
        let tx_manager = TransactionManager::with_timeouts(
            Duration::from_secs(config.transaction_timeout_secs),
            Duration::from_secs(config.lock_timeout_secs),
        );
        Self {
            store: TensorStore::new(),
            row_counters: DashMap::new(),
            btree_indexes: RwLock::new(HashMap::new()),
            is_durable: false,
            tx_manager,
            ddl_lock: RwLock::new(()),
            index_locks: std::array::from_fn(|_| RwLock::new(())),
            max_btree_entries: config.max_btree_entries,
            btree_entry_count: AtomicUsize::new(0),
            config,
            table_count: AtomicUsize::new(0),
            constraint_cache: DashMap::new(),
            fk_references: RwLock::new(HashMap::new()),
        }
    }

    /// Creates an engine with an existing store and default configuration.
    #[must_use]
    #[instrument(level = "info", skip_all)]
    pub fn with_store(store: TensorStore) -> Self {
        info!("creating relational engine with existing store");
        Self::with_store_and_config(store, RelationalConfig::default())
    }

    /// Creates an engine with an existing store and custom configuration.
    #[must_use]
    #[instrument(level = "info", skip_all, fields(max_tables = ?config.max_tables))]
    pub fn with_store_and_config(store: TensorStore, config: RelationalConfig) -> Self {
        info!(
            max_btree_entries = config.max_btree_entries,
            "creating relational engine with store and config"
        );
        let tx_manager = TransactionManager::with_timeouts(
            Duration::from_secs(config.transaction_timeout_secs),
            Duration::from_secs(config.lock_timeout_secs),
        );
        Self {
            store,
            row_counters: DashMap::new(),
            btree_indexes: RwLock::new(HashMap::new()),
            is_durable: false,
            tx_manager,
            ddl_lock: RwLock::new(()),
            index_locks: std::array::from_fn(|_| RwLock::new(())),
            max_btree_entries: config.max_btree_entries,
            btree_entry_count: AtomicUsize::new(0),
            config,
            table_count: AtomicUsize::new(0),
            constraint_cache: DashMap::new(),
            fk_references: RwLock::new(HashMap::new()),
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
    #[instrument(level = "info", skip_all, fields(wal_path = ?wal_path.as_ref()))]
    pub fn open_durable<P: AsRef<Path>>(
        wal_path: P,
        wal_config: WalConfig,
    ) -> std::io::Result<Self> {
        info!("opening durable relational engine");
        Self::open_durable_with_config(wal_path, wal_config, RelationalConfig::default())
    }

    /// Create a durable engine with Write-Ahead Log and custom configuration.
    ///
    /// # Errors
    /// Returns an I/O error if the WAL file cannot be created or opened.
    #[instrument(level = "info", skip_all, fields(wal_path = ?wal_path.as_ref()))]
    pub fn open_durable_with_config<P: AsRef<Path>>(
        wal_path: P,
        wal_config: WalConfig,
        config: RelationalConfig,
    ) -> std::io::Result<Self> {
        info!(
            max_btree_entries = config.max_btree_entries,
            "opening durable relational engine with config"
        );
        let store = TensorStore::open_durable(wal_path, wal_config)?;
        let tx_manager = TransactionManager::with_timeouts(
            Duration::from_secs(config.transaction_timeout_secs),
            Duration::from_secs(config.lock_timeout_secs),
        );
        Ok(Self {
            store,
            row_counters: DashMap::new(),
            btree_indexes: RwLock::new(HashMap::new()),
            is_durable: true,
            tx_manager,
            ddl_lock: RwLock::new(()),
            index_locks: std::array::from_fn(|_| RwLock::new(())),
            max_btree_entries: config.max_btree_entries,
            btree_entry_count: AtomicUsize::new(0),
            config,
            table_count: AtomicUsize::new(0),
            constraint_cache: DashMap::new(),
            fk_references: RwLock::new(HashMap::new()),
        })
    }

    /// Recover a durable engine from WAL after crash.
    ///
    /// Recovers all table metadata, row data, and B-tree indexes from the WAL.
    /// The following must be recreated manually after recovery:
    /// - Hash indexes: call `create_index()` for each needed index
    ///
    /// # Errors
    /// Returns `RelationalError::StorageError` if WAL recovery fails.
    #[instrument(level = "info", skip_all, fields(wal_path = ?wal_path.as_ref()))]
    pub fn recover<P: AsRef<Path>>(
        wal_path: P,
        wal_config: &WalConfig,
        snapshot_path: Option<&Path>,
    ) -> std::result::Result<Self, RelationalError> {
        info!("recovering relational engine from WAL");
        Self::recover_with_config(
            wal_path,
            wal_config,
            snapshot_path,
            RelationalConfig::default(),
        )
    }

    /// Recover a durable engine from WAL with custom configuration.
    ///
    /// B-tree indexes are automatically rebuilt from persisted data.
    ///
    /// # Errors
    /// Returns `RelationalError::StorageError` if WAL recovery fails.
    #[instrument(level = "info", skip_all, fields(wal_path = ?wal_path.as_ref()))]
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
        info!(
            existing_tables = existing_tables,
            "recovered relational engine"
        );
        let tx_manager = TransactionManager::with_timeouts(
            Duration::from_secs(config.transaction_timeout_secs),
            Duration::from_secs(config.lock_timeout_secs),
        );
        let engine = Self {
            store,
            row_counters: DashMap::new(),
            btree_indexes: RwLock::new(HashMap::new()),
            is_durable: true,
            tx_manager,
            ddl_lock: RwLock::new(()),
            index_locks: std::array::from_fn(|_| RwLock::new(())),
            max_btree_entries: config.max_btree_entries,
            btree_entry_count: AtomicUsize::new(0),
            config,
            table_count: AtomicUsize::new(existing_tables),
            constraint_cache: DashMap::new(),
            fk_references: RwLock::new(HashMap::new()),
        };

        // Rebuild RelationalSlab tables from persisted metadata
        engine.rebuild_slab_tables()?;

        // Validate hash indexes from persisted data
        engine.rebuild_hash_indexes()?;

        // Rebuild B-tree indexes from persisted data
        engine.rebuild_btree_indexes()?;

        Ok(engine)
    }

    /// Rebuilds `RelationalSlab` table structures from persisted metadata.
    ///
    /// After WAL recovery, the `TensorStore` has table metadata but the
    /// `RelationalSlab` is empty. This method scans for table metadata
    /// and recreates the table structures in the slab.
    ///
    /// Note: This only recreates table structure (schema), not row data.
    /// Row data is not persisted to WAL and must be repopulated separately.
    ///
    /// # Errors
    /// Returns `RelationalError::StorageError` if reading or parsing metadata fails.
    #[instrument(skip(self))]
    fn rebuild_slab_tables(&self) -> Result<()> {
        let table_keys = self.store.scan("_meta:table:");
        let mut rebuilt_count = 0usize;

        for key in table_keys {
            // Extract table name from key: _meta:table:NAME
            let parts: Vec<&str> = key.split(':').collect();
            if parts.len() != 3 {
                continue;
            }
            let table_name = parts[2];

            // Get schema from metadata
            let schema = match self.get_schema(table_name) {
                Ok(s) => s,
                Err(e) => {
                    debug!(table = %table_name, error = ?e, "failed to parse schema during recovery");
                    continue;
                },
            };

            // Create table in slab (if not already exists)
            let slab_schema: tensor_store::TableSchema = (&schema).into();
            if !self.slab().table_exists(table_name) {
                if let Err(e) = self.slab().create_table(table_name, slab_schema) {
                    debug!(table = %table_name, error = ?e, "failed to create slab table during recovery");
                    continue;
                }
                rebuilt_count += 1;
                debug!(table = %table_name, "rebuilt slab table from metadata");
            }

            // Rebuild row counter from stored row data (if any)
            // Note: Row data doesn't survive WAL recovery, so this is usually 0
            self.row_counters
                .entry(table_name.to_string())
                .or_insert_with(|| AtomicU64::new(0));
        }

        info!(
            rebuilt_tables = rebuilt_count,
            "rebuilt slab tables from persistent metadata"
        );

        Ok(())
    }

    /// Validates hash indexes from persistent storage.
    ///
    /// Hash indexes are stored directly in `TensorStore` and don't require
    /// in-memory reconstruction like B-tree indexes. This method validates
    /// and logs recovered indexes for observability.
    #[instrument(skip(self))]
    fn rebuild_hash_indexes(&self) -> Result<()> {
        let all_keys = self.store.scan("_idx:");
        let mut index_count = 0usize;
        let mut tables_with_indexes: HashSet<String> = HashSet::new();

        for key in all_keys {
            let colon_count = key.matches(':').count();
            // Meta keys have 2 colons: _idx:table:column
            // Entry keys have 3 colons: _idx:table:column:hash
            if colon_count == 2 {
                // This is a meta key
                let parts: Vec<&str> = key.split(':').collect();
                if parts.len() == 3 {
                    let table = parts[1];
                    let column = parts[2];
                    tables_with_indexes.insert(table.to_string());
                    index_count += 1;
                    debug!(table = %table, column = %column, "recovered hash index");
                }
            }
        }

        info!(
            hash_index_count = index_count,
            tables = tables_with_indexes.len(),
            "validated hash indexes from persistent storage"
        );

        Ok(())
    }

    /// Rebuilds all B-tree indexes from persisted data after recovery.
    ///
    /// This method scans for all B-tree entry keys in the store and reconstructs
    /// the in-memory `BTreeMap` indexes. It should be called automatically after
    /// recovery, but can also be called manually if needed.
    ///
    /// # Errors
    /// Returns `RelationalError::StorageError` if reading persisted index data fails.
    #[instrument(skip(self))]
    fn rebuild_btree_indexes(&self) -> Result<()> {
        // Scan for all btree keys
        let btree_keys = self.store.scan("_btree:");

        let mut indexes: HashMap<BTreeIndexKey, BTreeMap<OrderedKey, Vec<u64>>> = HashMap::new();
        let mut entry_count = 0usize;

        for key in btree_keys {
            let parts: Vec<&str> = key.split(':').collect();

            // Skip meta keys (3 parts: _btree:table:column)
            // Process entry keys (4 parts: _btree:table:column:sortable_value)
            if parts.len() != 4 {
                continue;
            }

            let table = parts[1];
            let column = parts[2];
            let sortable_value = parts[3];

            // Parse the sortable key back to OrderedKey
            let Some(ordered_key) = OrderedKey::from_sortable_key(sortable_value) else {
                debug!(
                    key = %key,
                    "skipping btree entry with unparseable sortable key"
                );
                continue;
            };

            // Get the row IDs from the store
            let tensor = match self.store.get(&key) {
                Ok(t) => t,
                Err(e) => {
                    debug!(key = %key, error = %e, "failed to read btree entry");
                    continue;
                },
            };

            let ids = match Self::tensor_to_id_list(&tensor) {
                Ok(ids) => ids,
                Err(e) => {
                    debug!(key = %key, error = ?e, "failed to parse btree entry ids");
                    continue;
                },
            };

            // Add to the in-memory index
            let btree_key = (table.to_string(), column.to_string());
            let btree = indexes.entry(btree_key).or_default();
            let existing_ids = btree.entry(ordered_key).or_default();
            existing_ids.extend(ids);
            entry_count += 1;
        }

        // Update the engine's btree_indexes
        let index_count = indexes.len();
        *self.btree_indexes.write() = indexes;
        self.btree_entry_count.store(entry_count, Ordering::Relaxed);

        info!(
            btree_count = index_count,
            entry_count = entry_count,
            "rebuilt B-tree indexes from persistent storage"
        );

        Ok(())
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
    #[instrument(skip(self))]
    pub fn table_count(&self) -> usize {
        self.table_count.load(Ordering::Acquire)
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

    /// Saturating subtract for atomic counter to prevent underflow.
    fn saturating_sub_atomic(counter: &AtomicUsize, amount: usize) {
        let mut current = counter.load(Ordering::Relaxed);
        loop {
            let new_val = current.saturating_sub(amount);
            match counter.compare_exchange_weak(
                current,
                new_val,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
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

    /// Creates a new table with the given schema.
    ///
    /// # Examples
    ///
    /// ```
    /// use relational_engine::{RelationalEngine, Schema, Column, ColumnType};
    ///
    /// let engine = RelationalEngine::new();
    /// let schema = Schema::new(vec![
    ///     Column::new("id", ColumnType::Int),
    ///     Column::new("name", ColumnType::String),
    ///     Column::new("active", ColumnType::Bool),
    /// ]);
    /// engine.create_table("users", schema).unwrap();
    /// ```
    ///
    /// # Errors
    /// Returns `InvalidName` if table/column names exceed 255 chars or are empty.
    /// Returns `TableAlreadyExists` if the table already exists.
    /// Returns `StorageError` if the underlying storage fails.
    #[allow(clippy::needless_pass_by_value)] // Public API takes ownership for ergonomics
    #[instrument(skip(self, schema), fields(table = %name))]
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
                ColumnType::Bytes => "bytes",
                ColumnType::Json => "json",
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

        // Serialize constraints as JSON if present
        if !schema.constraints.is_empty() {
            let constraints_json = serde_json::to_string(&schema.constraints)
                .map_err(|e| RelationalError::StorageError(e.to_string()))?;
            meta.set(
                "_constraints",
                TensorValue::Scalar(ScalarValue::String(constraints_json)),
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
    #[instrument(skip(self), fields(table = %table))]
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
                "bytes" => ColumnType::Bytes,
                "json" => ColumnType::Json,
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

        // Load constraints if present
        let constraints = if let Some(TensorValue::Scalar(ScalarValue::String(constraints_json))) =
            meta.get("_constraints")
        {
            serde_json::from_str(constraints_json).map_err(|e| {
                RelationalError::SchemaCorrupted {
                    table: table.to_string(),
                    reason: format!("invalid constraints JSON: {e}"),
                }
            })?
        } else {
            Vec::new()
        };

        Ok(Schema::with_constraints(columns, constraints))
    }

    /// Returns a list of all table names.
    #[instrument(skip(self))]
    pub fn list_tables(&self) -> Vec<String> {
        self.store
            .scan("_meta:table:")
            .into_iter()
            .filter_map(|key| key.strip_prefix("_meta:table:").map(String::from))
            .collect()
    }

    /// Inserts a row into the table and returns the row ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use relational_engine::{RelationalEngine, Schema, Column, ColumnType, Value};
    /// use std::collections::HashMap;
    ///
    /// let engine = RelationalEngine::new();
    /// let schema = Schema::new(vec![
    ///     Column::new("name", ColumnType::String),
    ///     Column::new("age", ColumnType::Int),
    /// ]);
    /// engine.create_table("users", schema).unwrap();
    ///
    /// let row_id = engine.insert("users", HashMap::from([
    ///     ("name".to_string(), Value::String("Alice".into())),
    ///     ("age".to_string(), Value::Int(30)),
    /// ])).unwrap();
    /// assert_eq!(row_id, 1);
    /// ```
    ///
    /// # Errors
    /// Returns `TableNotFound`, `NullNotAllowed`, `TypeMismatch`, or `StorageError`.
    #[allow(clippy::needless_pass_by_value)] // Public API takes ownership for ergonomics
    #[instrument(skip(self, values), fields(table = %table))]
    pub fn insert(&self, table: &str, values: HashMap<String, Value>) -> Result<u64> {
        let start = Instant::now();

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
                self.log_slow_query("insert", table, start, 1);
                Ok(row_id)
            },
            Err(e) => {
                if let Err(rollback_err) = self.rollback(tx_id) {
                    warn!(tx_id = tx_id, error = %rollback_err, "Rollback failed after insert error");
                }
                Err(e)
            },
        }
    }

    /// Inserts multiple rows into the table atomically and returns the row IDs.
    ///
    /// # Examples
    ///
    /// ```
    /// use relational_engine::{RelationalEngine, Schema, Column, ColumnType, Value};
    /// use std::collections::HashMap;
    ///
    /// let engine = RelationalEngine::new();
    /// let schema = Schema::new(vec![
    ///     Column::new("name", ColumnType::String),
    ///     Column::new("age", ColumnType::Int),
    /// ]);
    /// engine.create_table("users", schema).unwrap();
    ///
    /// let rows = vec![
    ///     HashMap::from([
    ///         ("name".to_string(), Value::String("Alice".into())),
    ///         ("age".to_string(), Value::Int(30)),
    ///     ]),
    ///     HashMap::from([
    ///         ("name".to_string(), Value::String("Bob".into())),
    ///         ("age".to_string(), Value::Int(25)),
    ///     ]),
    /// ];
    /// let ids = engine.batch_insert("users", rows).unwrap();
    /// assert_eq!(ids, vec![1, 2]);
    /// ```
    ///
    /// # Errors
    /// Returns `TableNotFound`, `NullNotAllowed`, `TypeMismatch`, or `StorageError`.
    #[must_use = "batch insert results contain row IDs that should be used"]
    #[allow(clippy::cast_possible_wrap)] // Row IDs are monotonic from 1, won't exceed i64::MAX
    #[instrument(skip(self, rows), fields(table = %table, row_count = rows.len()))]
    pub fn batch_insert(&self, table: &str, rows: Vec<HashMap<String, Value>>) -> Result<Vec<u64>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let start = Instant::now();
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

        self.log_slow_query("batch_insert", table, start, row_ids.len());
        Ok(row_ids)
    }

    /// Selects rows from the table that match the condition.
    ///
    /// # Examples
    ///
    /// ```
    /// use relational_engine::{RelationalEngine, Schema, Column, ColumnType, Value, Condition};
    /// use std::collections::HashMap;
    ///
    /// let engine = RelationalEngine::new();
    /// let schema = Schema::new(vec![
    ///     Column::new("name", ColumnType::String),
    ///     Column::new("age", ColumnType::Int),
    /// ]);
    /// engine.create_table("users", schema).unwrap();
    /// engine.insert("users", HashMap::from([
    ///     ("name".to_string(), Value::String("Alice".into())),
    ///     ("age".to_string(), Value::Int(30)),
    /// ])).unwrap();
    ///
    /// // Select all rows
    /// let all = engine.select("users", Condition::True).unwrap();
    /// assert_eq!(all.len(), 1);
    ///
    /// // Select with condition
    /// let adults = engine.select("users", Condition::Ge("age".into(), Value::Int(18))).unwrap();
    /// assert_eq!(adults.len(), 1);
    /// ```
    ///
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
    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    #[allow(clippy::needless_pass_by_value)]
    #[instrument(skip(self, condition, options), fields(table = %table))]
    pub fn select_with_options(
        &self,
        table: &str,
        condition: Condition,
        options: QueryOptions,
    ) -> Result<Vec<Row>> {
        let start = Instant::now();
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

            // Convert 1-based row_ids to 0-based slab indices (saturating to avoid overflow)
            let indices: Vec<usize> = row_ids
                .iter()
                .filter_map(|id| usize::try_from(id.saturating_sub(1)).ok())
                .collect();
            let slab_rows = self
                .slab()
                .get_rows_by_indices(table, &indices)
                .map_err(|e| RelationalError::StorageError(e.to_string()))?;

            let max_depth = self.config.max_condition_depth;
            let rows: Result<Vec<Row>> = slab_rows
                .into_iter()
                .filter_map(|(row_id, slab_row)| {
                    let row = Self::slab_row_to_engine_row(&schema, row_id, slab_row);
                    match condition.evaluate_with_depth(&row, 0, max_depth) {
                        Ok(true) => Some(Ok(row)),
                        Ok(false) => None,
                        Err(e) => Some(Err(e)),
                    }
                })
                .collect();
            let mut rows = rows?;
            rows.sort_by_key(|r| r.id);

            // Check result limit
            if let Some(max) = self.config.max_query_result_rows {
                if rows.len() > max {
                    return Err(RelationalError::ResultTooLarge {
                        operation: "select".to_string(),
                        actual: rows.len(),
                        max,
                    });
                }
            }

            // Log slow query warning
            let elapsed = start.elapsed();
            let elapsed_ms = elapsed.as_millis() as u64;
            if elapsed_ms > self.config.slow_query_threshold_ms {
                warn!(
                    table = %table,
                    elapsed_ms = elapsed_ms,
                    threshold_ms = self.config.slow_query_threshold_ms,
                    row_count = rows.len(),
                    "slow query detected (indexed)"
                );
            }

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
        debug!(
            table = %table,
            "index miss: falling back to full table scan"
        );
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

        let max_depth = self.config.max_condition_depth;
        let rows: Result<Vec<Row>> = slab_rows
            .into_iter()
            .filter_map(|(row_id, slab_row)| {
                let row = Self::slab_row_to_engine_row(&schema, row_id, slab_row);
                match condition.evaluate_with_depth(&row, 0, max_depth) {
                    Ok(true) => Some(Ok(row)),
                    Ok(false) => None,
                    Err(e) => Some(Err(e)),
                }
            })
            .collect();
        let mut rows = rows?;

        rows.sort_by_key(|r| r.id);

        // Check result limit
        if let Some(max) = self.config.max_query_result_rows {
            if rows.len() > max {
                return Err(RelationalError::ResultTooLarge {
                    operation: "select".to_string(),
                    actual: rows.len(),
                    max,
                });
            }
        }

        // Log slow query warning
        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_millis() as u64;
        if elapsed_ms > self.config.slow_query_threshold_ms {
            warn!(
                table = %table,
                elapsed_ms = elapsed_ms,
                threshold_ms = self.config.slow_query_threshold_ms,
                row_count = rows.len(),
                "slow query detected (full scan)"
            );
        }

        Ok(rows)
    }

    /// Selects rows with LIMIT and OFFSET, with early termination.
    ///
    /// More efficient than `select()` for pagination as scanning stops
    /// once `offset + limit` rows are found matching the condition.
    ///
    /// # Arguments
    /// * `table` - Table name
    /// * `condition` - Filter condition
    /// * `limit` - Maximum rows to return
    /// * `offset` - Rows to skip before returning
    ///
    /// # Errors
    /// Returns `TableNotFound` if the table does not exist, or `QueryTimeout` if the
    /// query exceeds the configured timeout.
    #[must_use = "query results should be used"]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::needless_pass_by_value)]
    #[instrument(skip(self, condition), fields(table = %table, limit, offset))]
    pub fn select_with_limit(
        &self,
        table: &str,
        condition: Condition,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Row>> {
        let start = Instant::now();
        let timeout = self.resolve_timeout(QueryOptions::default());
        let deadline = Deadline::from_timeout_ms(timeout);
        let schema = self.get_schema(table)?;

        // Early return for limit 0
        if limit == 0 {
            return Ok(Vec::new());
        }

        let target_count = offset.saturating_add(limit);

        // Try index path first
        if let Some(row_ids) = self.try_index_lookup(table, &condition)? {
            // Check timeout after index lookup
            if deadline.is_expired() {
                return Err(RelationalError::QueryTimeout {
                    operation: "select_with_limit (index lookup)".to_string(),
                    timeout_ms: timeout.unwrap_or(0),
                });
            }

            // Index gives us row IDs - take only what we need
            let limited_ids: Vec<u64> = row_ids.into_iter().take(target_count).collect();

            let indices: Vec<usize> = limited_ids
                .iter()
                .filter_map(|id| usize::try_from(id.saturating_sub(1)).ok())
                .collect();

            let slab_rows = self
                .slab()
                .get_rows_by_indices(table, &indices)
                .map_err(|e| RelationalError::StorageError(e.to_string()))?;

            let max_depth = self.config.max_condition_depth;
            let rows: Result<Vec<Row>> = slab_rows
                .into_iter()
                .filter_map(|(row_id, slab_row)| {
                    let row = Self::slab_row_to_engine_row(&schema, row_id, slab_row);
                    match condition.evaluate_with_depth(&row, 0, max_depth) {
                        Ok(true) => Some(Ok(row)),
                        Ok(false) => None,
                        Err(e) => Some(Err(e)),
                    }
                })
                .collect();
            let mut rows = rows?;

            rows.sort_by_key(|r| r.id);

            // Apply offset and limit
            let result: Vec<Row> = rows.into_iter().skip(offset).take(limit).collect();

            self.log_slow_query("select_with_limit (indexed)", table, start, result.len());
            return Ok(result);
        }

        // Full scan with early termination
        let slab_rows = self
            .slab()
            .scan_all(table)
            .map_err(|e| RelationalError::StorageError(e.to_string()))?;

        let max_depth = self.config.max_condition_depth;
        let mut collected = 0usize;
        let mut rows: Vec<Row> = Vec::with_capacity(target_count.min(1000));

        for (row_id, slab_row) in slab_rows {
            if deadline.is_expired() {
                return Err(RelationalError::QueryTimeout {
                    operation: "select_with_limit (scan)".to_string(),
                    timeout_ms: timeout.unwrap_or(0),
                });
            }

            let row = Self::slab_row_to_engine_row(&schema, row_id, slab_row);

            if condition.evaluate_with_depth(&row, 0, max_depth)? {
                rows.push(row);
                collected += 1;

                // Early termination: we have enough rows
                if collected >= target_count {
                    break;
                }
            }
        }

        rows.sort_by_key(|r| r.id);

        // Apply offset and limit
        let result: Vec<Row> = rows.into_iter().skip(offset).take(limit).collect();

        self.log_slow_query("select_with_limit", table, start, result.len());
        Ok(result)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn log_slow_query(&self, operation: &str, table: &str, start: Instant, row_count: usize) {
        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_millis() as u64;
        if elapsed_ms > self.config.slow_query_threshold_ms {
            warn!(
                table = %table,
                operation = %operation,
                elapsed_ms = elapsed_ms,
                threshold_ms = self.config.slow_query_threshold_ms,
                row_count = row_count,
                "slow query detected"
            );
        }
    }

    /// Returns an iterator over query results.
    ///
    /// This method provides an iterator interface for processing query results
    /// one row at a time. Use `CursorOptions` to specify a starting offset and/or limit.
    ///
    /// # Examples
    ///
    /// ```
    /// use relational_engine::{RelationalEngine, Schema, Column, ColumnType, Value, Condition, CursorOptions};
    /// use std::collections::HashMap;
    ///
    /// let engine = RelationalEngine::new();
    /// let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
    /// engine.create_table("nums", schema).unwrap();
    ///
    /// for i in 0..10 {
    ///     engine.insert("nums", HashMap::from([("val".to_string(), Value::Int(i))])).unwrap();
    /// }
    ///
    /// // Iterate over all rows
    /// let mut cursor = engine.select_iter("nums", Condition::True, CursorOptions::default()).unwrap();
    /// let mut count = 0;
    /// while let Some(row) = cursor.next() {
    ///     count += 1;
    /// }
    /// assert_eq!(count, 10);
    ///
    /// // Resume iteration from a specific offset
    /// let cursor = engine.select_iter("nums", Condition::True, CursorOptions::new().with_offset(5)).unwrap();
    /// assert_eq!(cursor.count(), 5);
    /// ```
    ///
    /// # Errors
    /// Returns `TableNotFound` if the table does not exist.
    #[instrument(skip(self, condition, options), fields(table = %table))]
    pub fn select_iter(
        &self,
        table: &str,
        condition: Condition,
        options: CursorOptions,
    ) -> Result<RowCursor<'_>> {
        let start = Instant::now();
        let rows = if let Some(limit) = options.limit {
            // Use efficient limited select with early termination
            self.select_with_limit(table, condition, limit, options.offset)?
        } else if options.offset > 0 {
            // Offset only - materialize then skip
            let rows = self.select(table, condition)?;
            if options.offset >= rows.len() {
                Vec::new()
            } else {
                rows.into_iter().skip(options.offset).collect()
            }
        } else {
            // No limit or offset
            self.select(table, condition)?
        };

        let total_rows = rows.len();
        self.log_slow_query("select_iter", table, start, total_rows);
        Ok(RowCursor {
            engine: self,
            rows: rows.into_iter(),
            rows_processed: 0,
            total_rows,
        })
    }

    /// Creates a streaming cursor for memory-efficient iteration over large result sets.
    ///
    /// Unlike `select_iter`, which loads all results upfront, `select_streaming` fetches
    /// rows in configurable batches, reducing memory usage for large tables.
    ///
    /// # Arguments
    /// * `table` - Table name
    /// * `condition` - Filter condition
    ///
    /// # Returns
    /// A `StreamingCursor` that yields rows one at a time, fetching in batches internally.
    ///
    /// # Errors
    /// Returns `TableNotFound` if the table does not exist.
    #[must_use]
    #[instrument(skip(self, condition), fields(table = %table))]
    pub fn select_streaming(&self, table: &str, condition: Condition) -> StreamingCursor<'_> {
        StreamingCursor::new(self, table, condition)
    }

    /// Creates a cursor builder for streaming queries with custom options.
    ///
    /// # Arguments
    /// * `table` - Table name
    /// * `condition` - Filter condition
    ///
    /// # Returns
    /// A `CursorBuilder` that can be configured with batch size and max rows.
    #[must_use]
    #[instrument(skip(self, condition), fields(table = %table))]
    pub fn select_streaming_builder(&self, table: &str, condition: Condition) -> CursorBuilder<'_> {
        CursorBuilder::new(self, table, condition)
    }

    /// Selects distinct rows, removing duplicates based on selected columns.
    ///
    /// If `columns` is `None`, all columns are used for deduplication.
    /// If `columns` is `Some`, only those columns are compared for uniqueness.
    ///
    /// # Examples
    ///
    /// ```
    /// use relational_engine::{RelationalEngine, Schema, Column, ColumnType, Value, Condition};
    /// use std::collections::HashMap;
    ///
    /// let engine = RelationalEngine::new();
    /// let schema = Schema::new(vec![
    ///     Column::new("name", ColumnType::String),
    ///     Column::new("dept", ColumnType::String),
    /// ]);
    /// engine.create_table("employees", schema).unwrap();
    ///
    /// // Insert some duplicate departments
    /// engine.insert("employees", HashMap::from([
    ///     ("name".to_string(), Value::String("Alice".into())),
    ///     ("dept".to_string(), Value::String("Engineering".into())),
    /// ])).unwrap();
    /// engine.insert("employees", HashMap::from([
    ///     ("name".to_string(), Value::String("Bob".into())),
    ///     ("dept".to_string(), Value::String("Engineering".into())),
    /// ])).unwrap();
    ///
    /// // Get distinct departments
    /// let distinct = engine.select_distinct("employees", Condition::True, Some(&["dept".to_string()])).unwrap();
    /// assert_eq!(distinct.len(), 1);
    /// ```
    ///
    /// # Errors
    /// Returns `TableNotFound`, `ColumnNotFound`, or `StorageError`.
    #[must_use = "query results should be used"]
    #[allow(clippy::needless_pass_by_value)]
    #[instrument(skip(self, condition, columns), fields(table = %table))]
    pub fn select_distinct(
        &self,
        table: &str,
        condition: Condition,
        columns: Option<&[String]>,
    ) -> Result<Vec<Row>> {
        let start = Instant::now();
        let schema = self.get_schema(table)?;

        // Validate columns if specified
        if let Some(cols) = columns {
            for col in cols {
                if schema.get_column(col).is_none() {
                    return Err(RelationalError::ColumnNotFound(col.clone()));
                }
            }
        }

        let rows = self.select(table, condition)?;

        // Build set of seen value keys
        let mut seen: HashSet<Vec<HashableValue>> = HashSet::new();
        let mut distinct_rows = Vec::new();

        for row in rows {
            let key: Vec<HashableValue> = if let Some(cols) = columns {
                cols.iter()
                    .filter_map(|col| row.get(col).map(|v| HashableValue(v.clone())))
                    .collect()
            } else {
                row.values
                    .iter()
                    .map(|(_, v)| HashableValue(v.clone()))
                    .collect()
            };

            if seen.insert(key) {
                distinct_rows.push(row);
            }
        }

        self.log_slow_query("select_distinct", table, start, distinct_rows.len());
        Ok(distinct_rows)
    }

    /// Selects rows with GROUP BY aggregation.
    ///
    /// Groups rows by the specified columns and computes aggregates for each group.
    /// An optional HAVING condition can filter groups after aggregation.
    ///
    /// # Examples
    ///
    /// ```
    /// use relational_engine::{
    ///     RelationalEngine, Schema, Column, ColumnType, Value, Condition,
    ///     AggregateExpr, AggregateValue, HavingCondition, AggregateRef,
    /// };
    /// use std::collections::HashMap;
    ///
    /// let engine = RelationalEngine::new();
    /// let schema = Schema::new(vec![
    ///     Column::new("dept", ColumnType::String),
    ///     Column::new("salary", ColumnType::Int),
    /// ]);
    /// engine.create_table("employees", schema).unwrap();
    ///
    /// engine.insert("employees", HashMap::from([
    ///     ("dept".to_string(), Value::String("Engineering".into())),
    ///     ("salary".to_string(), Value::Int(100000)),
    /// ])).unwrap();
    /// engine.insert("employees", HashMap::from([
    ///     ("dept".to_string(), Value::String("Engineering".into())),
    ///     ("salary".to_string(), Value::Int(120000)),
    /// ])).unwrap();
    /// engine.insert("employees", HashMap::from([
    ///     ("dept".to_string(), Value::String("Sales".into())),
    ///     ("salary".to_string(), Value::Int(80000)),
    /// ])).unwrap();
    ///
    /// // Group by department and compute aggregates
    /// let groups = engine.select_grouped(
    ///     "employees",
    ///     Condition::True,
    ///     &["dept".to_string()],
    ///     &[AggregateExpr::CountAll, AggregateExpr::Sum("salary".to_string())],
    ///     None,
    /// ).unwrap();
    ///
    /// assert_eq!(groups.len(), 2);
    /// ```
    ///
    /// # Errors
    /// Returns `TableNotFound`, `ColumnNotFound`, or `StorageError`.
    #[must_use = "query results should be used"]
    #[allow(clippy::needless_pass_by_value)]
    #[instrument(skip(self, condition, group_by, aggregates, having), fields(table = %table))]
    pub fn select_grouped(
        &self,
        table: &str,
        condition: Condition,
        group_by: &[String],
        aggregates: &[AggregateExpr],
        having: Option<HavingCondition>,
    ) -> Result<Vec<GroupedRow>> {
        let start = Instant::now();
        let schema = self.get_schema(table)?;

        // Validate group_by columns
        for col in group_by {
            if schema.get_column(col).is_none() {
                return Err(RelationalError::ColumnNotFound(col.clone()));
            }
        }

        // Validate aggregate columns
        for agg in aggregates {
            if let Some(col) = agg.column_name() {
                if schema.get_column(col).is_none() {
                    return Err(RelationalError::ColumnNotFound(col.to_string()));
                }
            }
        }

        let rows = self.select(table, condition)?;

        // Group rows by key
        let mut groups: HashMap<Vec<HashableValue>, Vec<Row>> = HashMap::new();

        for row in rows {
            let key: Vec<HashableValue> = group_by
                .iter()
                .map(|col| HashableValue(row.get(col).cloned().unwrap_or(Value::Null)))
                .collect();

            groups.entry(key).or_default().push(row);
        }

        // Compute aggregates for each group
        let mut results = Vec::with_capacity(groups.len());

        for (key, group_rows) in groups {
            let group_key: Vec<(String, Value)> = group_by
                .iter()
                .zip(key.iter())
                .map(|(col, hv)| (col.clone(), hv.0.clone()))
                .collect();

            let computed_aggregates: Vec<(String, AggregateValue)> = aggregates
                .iter()
                .map(|agg| {
                    let name = agg.result_name();
                    let value = Self::compute_aggregate(agg, &group_rows);
                    (name, value)
                })
                .collect();

            let grouped_row = GroupedRow {
                group_key,
                aggregates: computed_aggregates,
            };

            // Apply HAVING filter
            if let Some(ref having_cond) = having {
                if !having_cond.evaluate_with_depth(
                    &grouped_row,
                    0,
                    self.config.max_condition_depth,
                )? {
                    continue;
                }
            }

            results.push(grouped_row);
        }

        // Sort by group key for deterministic output
        results.sort_by(|a, b| {
            for (av, bv) in a.group_key.iter().zip(b.group_key.iter()) {
                if let Some(ord) = av.1.partial_cmp_value(&bv.1) {
                    if ord != std::cmp::Ordering::Equal {
                        return ord;
                    }
                }
            }
            std::cmp::Ordering::Equal
        });

        self.log_slow_query("select_grouped", table, start, results.len());
        Ok(results)
    }

    #[allow(clippy::cast_precision_loss)] // Standard SQL behavior: int->float conversion in aggregates
    fn compute_aggregate(agg: &AggregateExpr, rows: &[Row]) -> AggregateValue {
        match agg {
            AggregateExpr::CountAll => AggregateValue::Count(rows.len() as u64),
            AggregateExpr::Count(col) => {
                let count = rows
                    .iter()
                    .filter(|r| r.get(col).is_some_and(|v| !matches!(v, Value::Null)))
                    .count();
                AggregateValue::Count(count as u64)
            },
            AggregateExpr::Sum(col) => {
                let mut sum = 0.0;
                for row in rows {
                    if let Some(val) = row.get(col) {
                        match val {
                            Value::Int(i) => sum += *i as f64,
                            Value::Float(f) => sum += f,
                            _ => {},
                        }
                    }
                }
                AggregateValue::Sum(sum)
            },
            AggregateExpr::Avg(col) => {
                let mut sum = 0.0;
                let mut count = 0u64;
                for row in rows {
                    if let Some(val) = row.get(col) {
                        match val {
                            Value::Int(i) => {
                                sum += *i as f64;
                                count += 1;
                            },
                            Value::Float(f) => {
                                sum += f;
                                count += 1;
                            },
                            _ => {},
                        }
                    }
                }
                if count == 0 {
                    AggregateValue::Avg(None)
                } else {
                    AggregateValue::Avg(Some(sum / count as f64))
                }
            },
            AggregateExpr::Min(col) => {
                let mut min: Option<Value> = None;
                for row in rows {
                    if let Some(val) = row.get(col) {
                        if matches!(val, Value::Null) {
                            continue;
                        }
                        match &min {
                            None => min = Some(val.clone()),
                            Some(current) => {
                                if val.partial_cmp_value(current) == Some(std::cmp::Ordering::Less)
                                {
                                    min = Some(val.clone());
                                }
                            },
                        }
                    }
                }
                AggregateValue::Min(min)
            },
            AggregateExpr::Max(col) => {
                let mut max: Option<Value> = None;
                for row in rows {
                    if let Some(val) = row.get(col) {
                        if matches!(val, Value::Null) {
                            continue;
                        }
                        match &max {
                            None => max = Some(val.clone()),
                            Some(current) => {
                                if val.partial_cmp_value(current)
                                    == Some(std::cmp::Ordering::Greater)
                                {
                                    max = Some(val.clone());
                                }
                            },
                        }
                    }
                }
                AggregateValue::Max(max)
            },
        }
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

    /// Updates rows that match the condition and returns the count of updated rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use relational_engine::{RelationalEngine, Schema, Column, ColumnType, Value, Condition};
    /// use std::collections::HashMap;
    ///
    /// let engine = RelationalEngine::new();
    /// let schema = Schema::new(vec![
    ///     Column::new("name", ColumnType::String),
    ///     Column::new("age", ColumnType::Int),
    /// ]);
    /// engine.create_table("users", schema).unwrap();
    /// engine.insert("users", HashMap::from([
    ///     ("name".to_string(), Value::String("Alice".into())),
    ///     ("age".to_string(), Value::Int(30)),
    /// ])).unwrap();
    ///
    /// let updated = engine.update(
    ///     "users",
    ///     Condition::Eq("name".into(), Value::String("Alice".into())),
    ///     HashMap::from([("age".to_string(), Value::Int(31))]),
    /// ).unwrap();
    /// assert_eq!(updated, 1);
    /// ```
    ///
    /// # Errors
    /// Returns `TableNotFound`, `ColumnNotFound`, `TypeMismatch`, `NullNotAllowed`, or `StorageError`.
    #[allow(clippy::needless_pass_by_value)] // Public API takes ownership for ergonomics
    #[instrument(skip(self, condition, updates), fields(table = %table))]
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
    #[allow(clippy::needless_pass_by_value, clippy::cast_possible_truncation)]
    #[instrument(skip(self, condition, updates, options), fields(table = %table))]
    pub fn update_with_options(
        &self,
        table: &str,
        condition: Condition,
        updates: HashMap<String, Value>,
        options: QueryOptions,
    ) -> Result<usize> {
        let start = Instant::now();
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
        let result = match self.tx_update(tx_id, table, condition, updates) {
            Ok(count) => {
                self.commit(tx_id)?;
                Ok(count)
            },
            Err(e) => {
                if let Err(rollback_err) = self.rollback(tx_id) {
                    warn!(tx_id = tx_id, error = %rollback_err, "Rollback failed after update error");
                }
                Err(e)
            },
        };

        // Log slow query warning
        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_millis() as u64;
        if elapsed_ms > self.config.slow_query_threshold_ms {
            warn!(
                table = %table,
                elapsed_ms = elapsed_ms,
                threshold_ms = self.config.slow_query_threshold_ms,
                rows_updated = result.as_ref().copied().unwrap_or(0),
                "slow update detected"
            );
        }

        result
    }

    /// Deletes rows that match the condition and returns the count of deleted rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use relational_engine::{RelationalEngine, Schema, Column, ColumnType, Value, Condition};
    /// use std::collections::HashMap;
    ///
    /// let engine = RelationalEngine::new();
    /// let schema = Schema::new(vec![
    ///     Column::new("name", ColumnType::String),
    ///     Column::new("age", ColumnType::Int),
    /// ]);
    /// engine.create_table("users", schema).unwrap();
    /// engine.insert("users", HashMap::from([
    ///     ("name".to_string(), Value::String("Alice".into())),
    ///     ("age".to_string(), Value::Int(30)),
    /// ])).unwrap();
    ///
    /// let deleted = engine.delete_rows("users", Condition::True).unwrap();
    /// assert_eq!(deleted, 1);
    /// ```
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[allow(clippy::needless_pass_by_value)] // Public API takes ownership for ergonomics
    #[instrument(skip(self, condition), fields(table = %table))]
    pub fn delete_rows(&self, table: &str, condition: Condition) -> Result<usize> {
        self.delete_rows_with_options(table, condition, QueryOptions::default())
    }

    /// Delete rows with query options including timeout.
    ///
    /// # Errors
    /// Returns `TableNotFound`, `StorageError`, or `QueryTimeout`.
    #[allow(clippy::needless_pass_by_value, clippy::cast_possible_truncation)]
    #[instrument(skip(self, condition, options), fields(table = %table))]
    pub fn delete_rows_with_options(
        &self,
        table: &str,
        condition: Condition,
        options: QueryOptions,
    ) -> Result<usize> {
        let start = Instant::now();
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
        let result = match self.tx_delete(tx_id, table, condition) {
            Ok(count) => {
                self.commit(tx_id)?;
                Ok(count)
            },
            Err(e) => {
                if let Err(rollback_err) = self.rollback(tx_id) {
                    warn!(tx_id = tx_id, error = %rollback_err, "Rollback failed after delete error");
                }
                Err(e)
            },
        };

        // Log slow query warning
        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_millis() as u64;
        if elapsed_ms > self.config.slow_query_threshold_ms {
            warn!(
                table = %table,
                elapsed_ms = elapsed_ms,
                threshold_ms = self.config.slow_query_threshold_ms,
                rows_deleted = result.as_ref().copied().unwrap_or(0),
                "slow delete detected"
            );
        }

        result
    }

    /// Hash join: O(n+m) instead of O(n*m) nested loop join.
    /// Builds a hash index on the right table, then probes from the left.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[instrument(skip(self), fields(table_a = %table_a, table_b = %table_b, on_a = %on_a, on_b = %on_b))]
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
    #[allow(clippy::cast_possible_truncation)]
    #[instrument(skip(self, options), fields(table_a = %table_a, table_b = %table_b, on_a = %on_a, on_b = %on_b))]
    pub fn join_with_options(
        &self,
        table_a: &str,
        table_b: &str,
        on_a: &str,
        on_b: &str,
        options: QueryOptions,
    ) -> Result<Vec<(Row, Row)>> {
        let start = Instant::now();
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

        // Log slow query warning
        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_millis() as u64;
        if elapsed_ms > self.config.slow_query_threshold_ms {
            warn!(
                table_a = %table_a,
                table_b = %table_b,
                elapsed_ms = elapsed_ms,
                threshold_ms = self.config.slow_query_threshold_ms,
                result_count = results.len(),
                "slow join detected"
            );
        }

        Ok(results)
    }

    /// LEFT JOIN: Returns all rows from `table_a`, with matching rows from `table_b`.
    /// If no match, the `table_b` side is `None`.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[instrument(skip(self), fields(table_a = %table_a, table_b = %table_b, on_a = %on_a, on_b = %on_b))]
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
    #[instrument(skip(self), fields(table_a = %table_a, table_b = %table_b, on_a = %on_a, on_b = %on_b))]
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
    #[instrument(skip(self), fields(table_a = %table_a, table_b = %table_b, on_a = %on_a, on_b = %on_b))]
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
    #[instrument(skip(self), fields(table_a = %table_a, table_b = %table_b))]
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
    #[instrument(skip(self), fields(table_a = %table_a, table_b = %table_b))]
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

    /// Counts rows matching the condition.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[instrument(skip(self, condition), fields(table = %table))]
    pub fn count(&self, table: &str, condition: Condition) -> Result<u64> {
        let rows = self.select(table, condition)?;
        Ok(rows.len() as u64)
    }

    /// Counts non-null values in a column matching the condition.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[instrument(skip(self, condition), fields(table = %table, column = %column))]
    pub fn count_column(&self, table: &str, column: &str, condition: Condition) -> Result<u64> {
        let rows = self.select(table, condition)?;
        let count = rows
            .iter()
            .filter(|row| row.get(column).is_some_and(|v| !matches!(v, Value::Null)))
            .count();
        Ok(count as u64)
    }

    /// Sums numeric values in a column matching the condition.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[allow(clippy::cast_precision_loss)] // Aggregate functions accept f64 precision loss
    #[instrument(skip(self, condition), fields(table = %table, column = %column))]
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

    /// Calculates the average of numeric values in a column matching the condition.
    ///
    /// Returns `None` if no numeric values are found.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[allow(clippy::cast_precision_loss)] // Aggregate functions accept f64 precision loss
    #[instrument(skip(self, condition), fields(table = %table, column = %column))]
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

    /// Finds the minimum value in a column matching the condition.
    ///
    /// Returns `None` if no comparable values are found.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[instrument(skip(self, condition), fields(table = %table, column = %column))]
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

    /// Finds the maximum value in a column matching the condition.
    ///
    /// Returns `None` if no comparable values are found.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[instrument(skip(self, condition), fields(table = %table, column = %column))]
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
    #[instrument(skip(self), fields(table = %table))]
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

        // Clean up constraint cache and FK references
        self.constraint_cache.remove(table);
        {
            let mut fk_refs = self.fk_references.write();
            fk_refs.remove(table);
            // Remove any FK references pointing to this table
            for refs in fk_refs.values_mut() {
                refs.retain(|(ref_table, _): &(String, ForeignKeyConstraint)| ref_table != table);
            }
        }

        Ok(())
    }

    // ==================== ALTER TABLE Operations ====================

    /// Adds a new column to an existing table.
    ///
    /// All existing rows will have NULL for the new column (if nullable) or the operation
    /// will fail if the column is not nullable and no default value mechanism exists.
    ///
    /// # Errors
    /// Returns `TableNotFound` if the table doesn't exist, `ColumnAlreadyExists` if
    /// a column with the same name exists, or `CannotAddColumn` if the column is
    /// not nullable.
    pub fn add_column(&self, table: &str, column: Column) -> Result<()> {
        let _ddl_guard = self.ddl_lock.write();

        let mut schema = self.get_schema(table)?;

        // Check if column already exists
        if schema.get_column(&column.name).is_some() {
            return Err(RelationalError::ColumnAlreadyExists {
                table: table.to_string(),
                column: column.name,
            });
        }

        // Non-nullable columns cannot be added to tables with existing rows
        // (unless we support DEFAULT, which we don't yet)
        if !column.nullable {
            let count = self.row_count(table)?;
            if count > 0 {
                return Err(RelationalError::CannotAddColumn {
                    column: column.name,
                    reason: "Cannot add non-nullable column to table with existing rows".into(),
                });
            }
        }

        schema.columns.push(column);
        self.save_schema(table, &schema)?;

        Ok(())
    }

    /// Drops a column from an existing table.
    ///
    /// # Errors
    /// Returns `TableNotFound` if the table doesn't exist, `ColumnNotFound` if
    /// the column doesn't exist, or `ColumnHasConstraint` if the column is part
    /// of a constraint.
    pub fn drop_column(&self, table: &str, column: &str) -> Result<()> {
        let _ddl_guard = self.ddl_lock.write();

        let mut schema = self.get_schema(table)?;

        // Check if column exists
        if schema.get_column(column).is_none() {
            return Err(RelationalError::ColumnNotFound(column.to_string()));
        }

        // Check if column is part of any constraint
        for constraint in &schema.constraints {
            match constraint {
                Constraint::PrimaryKey { columns, name } | Constraint::Unique { columns, name } => {
                    if columns.contains(&column.to_string()) {
                        return Err(RelationalError::ColumnHasConstraint {
                            column: column.to_string(),
                            constraint_name: name.clone(),
                        });
                    }
                },
                Constraint::ForeignKey(fk) => {
                    if fk.columns.contains(&column.to_string()) {
                        return Err(RelationalError::ColumnHasConstraint {
                            column: column.to_string(),
                            constraint_name: fk.name.clone(),
                        });
                    }
                },
                Constraint::NotNull { column: c, name } => {
                    if c == column {
                        return Err(RelationalError::ColumnHasConstraint {
                            column: column.to_string(),
                            constraint_name: name.clone(),
                        });
                    }
                },
            }
        }

        // Remove the column from schema
        schema.columns.retain(|c| c.name != column);
        self.save_schema(table, &schema)?;

        // Drop any indexes on this column (inline to avoid deadlock - we already hold ddl_lock)
        // Hash index cleanup
        let hash_meta_key = Self::index_meta_key(table, column);
        if self.store.exists(&hash_meta_key) {
            let prefix = Self::index_prefix(table, column);
            for key in self.store.scan(&prefix) {
                if let Err(e) = self.store.delete(&key) {
                    warn!(key = %key, error = %e, "Failed to delete hash index entry");
                }
            }
            if let Err(e) = self.store.delete(&hash_meta_key) {
                warn!(key = %hash_meta_key, error = %e, "Failed to delete hash index metadata");
            }
        }

        // B-tree index cleanup
        let btree_meta_key = Self::btree_meta_key(table, column);
        if self.store.exists(&btree_meta_key) {
            let entries_removed = {
                let key = (table.to_string(), column.to_string());
                let mut indexes = self.btree_indexes.write();
                indexes.remove(&key).map_or(0, |btree| btree.len())
            };
            if entries_removed > 0 {
                // Saturating subtract to prevent underflow
                Self::saturating_sub_atomic(&self.btree_entry_count, entries_removed);
            }
            let prefix = Self::btree_prefix(table, column);
            for key in self.store.scan(&prefix) {
                if let Err(e) = self.store.delete(&key) {
                    warn!(key = %key, error = %e, "Failed to delete btree index entry");
                }
            }
            if let Err(e) = self.store.delete(&btree_meta_key) {
                warn!(key = %btree_meta_key, error = %e, "Failed to delete btree index metadata");
            }
        }

        Ok(())
    }

    /// Renames a column in an existing table.
    ///
    /// # Errors
    /// Returns `TableNotFound` if the table doesn't exist, `ColumnNotFound` if
    /// the old column doesn't exist, or `ColumnAlreadyExists` if the new name
    /// already exists.
    pub fn rename_column(&self, table: &str, old_name: &str, new_name: &str) -> Result<()> {
        let _ddl_guard = self.ddl_lock.write();

        let mut schema = self.get_schema(table)?;

        // Check if old column exists
        if schema.get_column(old_name).is_none() {
            return Err(RelationalError::ColumnNotFound(old_name.to_string()));
        }

        // Check if new name already exists
        if schema.get_column(new_name).is_some() {
            return Err(RelationalError::ColumnAlreadyExists {
                table: table.to_string(),
                column: new_name.to_string(),
            });
        }

        // Rename the column
        for col in &mut schema.columns {
            if col.name == old_name {
                col.name = new_name.to_string();
                break;
            }
        }

        // Update column references in constraints
        for constraint in &mut schema.constraints {
            match constraint {
                Constraint::PrimaryKey { columns, .. } | Constraint::Unique { columns, .. } => {
                    for col in columns {
                        if col == old_name {
                            *col = new_name.to_string();
                        }
                    }
                },
                Constraint::ForeignKey(fk) => {
                    for col in &mut fk.columns {
                        if col == old_name {
                            *col = new_name.to_string();
                        }
                    }
                },
                Constraint::NotNull { column, .. } => {
                    if column == old_name {
                        *column = new_name.to_string();
                    }
                },
            }
        }

        self.save_schema(table, &schema)?;

        // Update constraint cache
        self.constraint_cache
            .insert(table.to_string(), schema.constraints);

        Ok(())
    }

    /// Adds a constraint to an existing table.
    ///
    /// For PRIMARY KEY and UNIQUE constraints, validates that existing data
    /// satisfies the constraint before adding it.
    ///
    /// # Errors
    /// Returns `TableNotFound`, `ColumnNotFound`, `ConstraintAlreadyExists`,
    /// `PrimaryKeyViolation`, `UniqueViolation`, or `ForeignKeyViolation`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn add_constraint(&self, table: &str, constraint: Constraint) -> Result<()> {
        let _ddl_guard = self.ddl_lock.write();

        let mut schema = self.get_schema(table)?;

        // Check if constraint name already exists
        let constraint_name = constraint.name();
        if schema
            .constraints
            .iter()
            .any(|c| c.name() == constraint_name)
        {
            return Err(RelationalError::ConstraintAlreadyExists {
                table: table.to_string(),
                constraint_name: constraint_name.to_string(),
            });
        }

        // Validate columns exist
        match &constraint {
            Constraint::PrimaryKey { columns, .. } | Constraint::Unique { columns, .. } => {
                for col in columns {
                    if schema.get_column(col).is_none() {
                        return Err(RelationalError::ColumnNotFound(col.clone()));
                    }
                }
            },
            Constraint::ForeignKey(fk) => {
                for col in &fk.columns {
                    if schema.get_column(col).is_none() {
                        return Err(RelationalError::ColumnNotFound(col.clone()));
                    }
                }
                // Validate referenced table and columns exist
                let ref_schema = self.get_schema(&fk.referenced_table)?;
                for col in &fk.referenced_columns {
                    if ref_schema.get_column(col).is_none() {
                        return Err(RelationalError::ColumnNotFound(col.clone()));
                    }
                }
            },
            Constraint::NotNull { column, .. } => {
                if schema.get_column(column).is_none() {
                    return Err(RelationalError::ColumnNotFound(column.clone()));
                }
            },
        }

        // Validate existing data satisfies the constraint
        self.validate_constraint_on_existing_data(table, &schema, &constraint)?;

        // Add constraint to schema
        schema.constraints.push(constraint.clone());
        self.save_schema(table, &schema)?;

        // Update constraint cache
        self.constraint_cache
            .entry(table.to_string())
            .or_default()
            .push(constraint.clone());

        // Update FK reference graph for foreign keys
        if let Constraint::ForeignKey(fk) = &constraint {
            let mut fk_refs = self.fk_references.write();
            fk_refs
                .entry(fk.referenced_table.clone())
                .or_default()
                .push((table.to_string(), fk.clone()));
        }

        Ok(())
    }

    /// Drops a constraint from an existing table.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `ConstraintNotFound`.
    pub fn drop_constraint(&self, table: &str, constraint_name: &str) -> Result<()> {
        let _ddl_guard = self.ddl_lock.write();

        let mut schema = self.get_schema(table)?;

        // Find and remove the constraint
        let idx = schema
            .constraints
            .iter()
            .position(|c| c.name() == constraint_name);

        let Some(idx) = idx else {
            return Err(RelationalError::ConstraintNotFound {
                table: table.to_string(),
                constraint_name: constraint_name.to_string(),
            });
        };

        let removed = schema.constraints.remove(idx);
        self.save_schema(table, &schema)?;

        // Update constraint cache
        if let Some(mut entry) = self.constraint_cache.get_mut(table) {
            entry.retain(|c| c.name() != constraint_name);
        }

        // Update FK reference graph
        if let Constraint::ForeignKey(fk) = &removed {
            let mut fk_refs = self.fk_references.write();
            if let Some(refs) = fk_refs.get_mut(&fk.referenced_table) {
                refs.retain(|(t, c): &(String, ForeignKeyConstraint)| {
                    !(t == table && c.name == constraint_name)
                });
            }
        }

        Ok(())
    }

    /// Returns the constraints for a table.
    ///
    /// # Errors
    /// Returns `TableNotFound` if the table does not exist.
    pub fn get_constraints(&self, table: &str) -> Result<Vec<Constraint>> {
        // Check cache first
        if let Some(constraints) = self.constraint_cache.get(table) {
            return Ok(constraints.clone());
        }

        // Load from schema
        let schema = self.get_schema(table)?;
        let constraints = schema.constraints;

        // Cache for future use
        self.constraint_cache
            .insert(table.to_string(), constraints.clone());

        Ok(constraints)
    }

    /// Validates that existing data satisfies a constraint.
    fn validate_constraint_on_existing_data(
        &self,
        table: &str,
        _schema: &Schema,
        constraint: &Constraint,
    ) -> Result<()> {
        match constraint {
            Constraint::PrimaryKey { columns, .. } | Constraint::Unique { columns, .. } => {
                // Check for duplicate values
                let rows = self.select(table, Condition::True)?;
                let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

                for row in &rows {
                    let key = Self::make_constraint_key(row, columns);
                    if !seen.insert(key.clone()) {
                        return Err(if matches!(constraint, Constraint::PrimaryKey { .. }) {
                            RelationalError::PrimaryKeyViolation {
                                table: table.to_string(),
                                columns: columns.clone(),
                                value: key,
                            }
                        } else {
                            RelationalError::UniqueViolation {
                                constraint_name: constraint.name().to_string(),
                                columns: columns.clone(),
                                value: key,
                            }
                        });
                    }
                }
            },
            Constraint::ForeignKey(fk) => {
                // Check that all FK values exist in referenced table
                let rows = self.select(table, Condition::True)?;
                for row in &rows {
                    self.validate_fk_reference(row, fk)?;
                }
            },
            Constraint::NotNull { column, name } => {
                // Check for NULL values
                let rows = self.select(table, Condition::True)?;
                for row in &rows {
                    if matches!(row.get(column), Some(Value::Null)) {
                        return Err(RelationalError::ColumnHasConstraint {
                            column: column.clone(),
                            constraint_name: name.clone(),
                        });
                    }
                }
            },
        }
        Ok(())
    }

    /// Creates a composite key string from row values for constraint checking.
    fn make_constraint_key(row: &Row, columns: &[String]) -> String {
        columns
            .iter()
            .map(|col| {
                row.get(col)
                    .map_or_else(|| "null".to_string(), Value::hash_key)
            })
            .collect::<Vec<_>>()
            .join(":")
    }

    /// Validates that a foreign key reference exists in the referenced table.
    fn validate_fk_reference(&self, row: &Row, fk: &ForeignKeyConstraint) -> Result<()> {
        // Build condition to check if referenced row exists
        let mut conditions = Vec::new();
        for (fk_col, ref_col) in fk.columns.iter().zip(fk.referenced_columns.iter()) {
            if let Some(value) = row.get(fk_col) {
                // Skip NULL values (NULLs don't violate FK)
                if matches!(value, Value::Null) {
                    return Ok(());
                }
                conditions.push(Condition::Eq(ref_col.clone(), value.clone()));
            }
        }

        if conditions.is_empty() {
            return Ok(());
        }

        // Combine conditions with AND
        let condition = conditions
            .into_iter()
            .reduce(|a, b| Condition::And(Box::new(a), Box::new(b)))
            .unwrap_or(Condition::True);

        let referenced_rows = self.select(&fk.referenced_table, condition)?;
        if referenced_rows.is_empty() {
            return Err(RelationalError::ForeignKeyViolation {
                constraint_name: fk.name.clone(),
                table: fk.columns.join(", "),
                referenced_table: fk.referenced_table.clone(),
            });
        }

        Ok(())
    }

    /// Saves the schema to storage.
    fn save_schema(&self, table: &str, schema: &Schema) -> Result<()> {
        let meta_key = Self::table_meta_key(table);

        let mut meta = TensorData::new();
        meta.set(
            "_name",
            TensorValue::Scalar(ScalarValue::String(table.to_string())),
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
                ColumnType::Bytes => "bytes",
                ColumnType::Json => "json",
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

        // Serialize constraints as JSON
        if !schema.constraints.is_empty() {
            let constraints_json = serde_json::to_string(&schema.constraints)
                .map_err(|e| RelationalError::StorageError(e.to_string()))?;
            meta.set(
                "_constraints",
                TensorValue::Scalar(ScalarValue::String(constraints_json)),
            );
        }

        self.put_maybe_durable(meta_key, meta)
    }

    /// Returns true if the table exists.
    #[instrument(skip(self), fields(table = %table))]
    pub fn table_exists(&self, table: &str) -> bool {
        let meta_key = Self::table_meta_key(table);
        self.store.exists(&meta_key)
    }

    /// Returns the number of rows in the table.
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[instrument(skip(self), fields(table = %table))]
    pub fn row_count(&self, table: &str) -> Result<usize> {
        let _ = self.get_schema(table)?;
        self.slab()
            .row_count(table)
            .map_err(|e| RelationalError::StorageError(e.to_string()))
    }

    /// Creates a hash index on a column for fast equality lookups.
    ///
    /// Hash indexes accelerate `Eq` conditions with O(1) lookup time.
    ///
    /// # Examples
    ///
    /// ```
    /// use relational_engine::{RelationalEngine, Schema, Column, ColumnType, Value, Condition};
    /// use std::collections::HashMap;
    ///
    /// let engine = RelationalEngine::new();
    /// let schema = Schema::new(vec![
    ///     Column::new("email", ColumnType::String),
    ///     Column::new("name", ColumnType::String),
    /// ]);
    /// engine.create_table("users", schema).unwrap();
    /// engine.create_index("users", "email").unwrap();
    ///
    /// // Inserts and equality lookups on 'email' now use the hash index
    /// engine.insert("users", HashMap::from([
    ///     ("email".to_string(), Value::String("alice@example.com".into())),
    ///     ("name".to_string(), Value::String("Alice".into())),
    /// ])).unwrap();
    /// ```
    ///
    /// # Errors
    /// Returns `InvalidName`, `TableNotFound`, `ColumnNotFound`, `IndexAlreadyExists`, or `StorageError`.
    #[allow(clippy::cast_possible_wrap)] // Row IDs won't exceed i64::MAX
    #[instrument(skip(self), fields(table = %table, column = %column))]
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
        self.put_maybe_durable(&meta_key, meta)?;

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

    /// Creates a B-tree index on a column for fast range queries.
    ///
    /// B-tree indexes accelerate `Lt`, `Le`, `Gt`, `Ge` conditions with O(log n) lookup.
    ///
    /// # Examples
    ///
    /// ```
    /// use relational_engine::{RelationalEngine, Schema, Column, ColumnType, Value, Condition};
    /// use std::collections::HashMap;
    ///
    /// let engine = RelationalEngine::new();
    /// let schema = Schema::new(vec![
    ///     Column::new("name", ColumnType::String),
    ///     Column::new("age", ColumnType::Int),
    /// ]);
    /// engine.create_table("users", schema).unwrap();
    /// engine.create_btree_index("users", "age").unwrap();
    ///
    /// engine.insert("users", HashMap::from([
    ///     ("name".to_string(), Value::String("Alice".into())),
    ///     ("age".to_string(), Value::Int(30)),
    /// ])).unwrap();
    ///
    /// // Range queries on 'age' now use the B-tree index
    /// let adults = engine.select("users", Condition::Ge("age".into(), Value::Int(18))).unwrap();
    /// ```
    ///
    /// # Errors
    /// Returns `InvalidName`, `TableNotFound`, `ColumnNotFound`, `IndexAlreadyExists`, or `StorageError`.
    #[allow(clippy::cast_possible_wrap)] // Row IDs won't exceed i64::MAX
    #[instrument(skip(self), fields(table = %table, column = %column))]
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
        self.put_maybe_durable(&meta_key, meta)?;

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
    #[instrument(skip(self), fields(table = %table, column = %column))]
    pub fn has_btree_index(&self, table: &str, column: &str) -> bool {
        let meta_key = Self::btree_meta_key(table, column);
        self.store.exists(&meta_key)
    }

    /// Drops a B-tree index from a column.
    ///
    /// # Errors
    /// Returns `TableNotFound`, `IndexNotFound`, or `StorageError`.
    #[instrument(skip(self), fields(table = %table, column = %column))]
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

        // Decrement entry counter (saturating to prevent underflow)
        if entries_removed > 0 {
            Self::saturating_sub_atomic(&self.btree_entry_count, entries_removed);
        }

        let prefix = Self::btree_prefix(table, column);
        let keys = self.store.scan(&prefix);
        for key in keys {
            self.delete_maybe_durable(&key)?;
        }

        self.delete_maybe_durable(&meta_key)?;

        Ok(())
    }

    /// Returns columns that have B-tree indexes.
    #[instrument(skip(self), fields(table = %table))]
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
    #[instrument(skip(self), fields(table = %table, column = %column))]
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
            self.delete_maybe_durable(&key)?;
        }

        self.delete_maybe_durable(&meta_key)?;

        Ok(())
    }

    /// Returns true if a hash index exists on the column.
    #[instrument(skip(self), fields(table = %table, column = %column))]
    pub fn has_index(&self, table: &str, column: &str) -> bool {
        let meta_key = Self::index_meta_key(table, column);
        self.store.exists(&meta_key)
    }

    /// Returns columns that have hash indexes.
    #[instrument(skip(self), fields(table = %table))]
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
            self.put_maybe_durable(&key, Self::id_list_to_tensor(&ids))?;
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
                self.delete_maybe_durable(&key)?;
            } else {
                self.put_maybe_durable(&key, Self::id_list_to_tensor(&ids))?;
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
            self.put_maybe_durable(&store_key, Self::id_list_to_tensor(&ids))?;
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

        // Decrement counter if key was removed (saturating to prevent underflow)
        if removed_key {
            Self::saturating_sub_atomic(&self.btree_entry_count, 1);
        }

        // Per-key lock for TensorStore update (after global lock released)
        let lock = self.acquire_index_lock(&store_key);
        let _guard = lock.write();

        // Update TensorStore (now protected by per-key lock)
        if let Ok(tensor) = self.store.get(&store_key) {
            let mut ids = Self::tensor_to_id_list(&tensor)?;
            ids.retain(|&id| id != row_id);

            if ids.is_empty() {
                self.delete_maybe_durable(&store_key)?;
            } else {
                self.put_maybe_durable(&store_key, Self::id_list_to_tensor(&ids))?;
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
    #[instrument(skip(self), fields(table = %table, column = %column))]
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
    #[instrument(skip(self, columns), fields(table = %table))]
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
    #[allow(clippy::too_many_lines)] // Exhaustive per-type columnar extraction
    #[instrument(skip(self), fields(table = %table, column = %column))]
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
            ColumnType::Bytes => {
                let mut dict: Vec<Vec<u8>> = Vec::new();
                let mut indices = Vec::with_capacity(row_count);
                let mut dict_map: std::collections::HashMap<Vec<u8>, u32> =
                    std::collections::HashMap::new();
                for (idx, (row_id, row)) in slab_rows.iter().enumerate() {
                    row_ids.push(row_id.as_u64() + 1);
                    match &row[col_idx] {
                        SlabColumnValue::Bytes(b) => {
                            let dict_idx = *dict_map.entry(b.clone()).or_insert_with(|| {
                                let idx = dict.len() as u32;
                                dict.push(b.clone());
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
                ColumnValues::Bytes { dict, indices }
            },
            ColumnType::Json => {
                let mut dict: Vec<String> = Vec::new();
                let mut indices = Vec::with_capacity(row_count);
                let mut dict_map: std::collections::HashMap<String, u32> =
                    std::collections::HashMap::new();
                for (idx, (row_id, row)) in slab_rows.iter().enumerate() {
                    row_ids.push(row_id.as_u64() + 1);
                    match &row[col_idx] {
                        SlabColumnValue::Json(j) => {
                            let dict_idx = *dict_map.entry(j.clone()).or_insert_with(|| {
                                let idx = dict.len() as u32;
                                dict.push(j.clone());
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
                ColumnValues::Json { dict, indices }
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
    #[instrument(skip(self), fields(table = %table, column = %column))]
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
        // Iterative implementation using explicit stack to avoid stack overflow on deep trees
        let mut stack = vec![condition];
        while let Some(cond) = stack.pop() {
            match cond {
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
                    stack.push(b);
                    stack.push(a);
                },
            }
        }
    }

    fn select_columnar_impl(
        &self,
        table: &str,
        condition: Condition,
        options: ColumnarScanOptions,
    ) -> Result<Vec<Row>> {
        // Try slab-based SIMD filtering first
        if let Some(rows) = self.try_slab_select(table, &condition, &options) {
            return Ok(rows);
        }

        // Fall back to row-based path for unsupported conditions
        self.select_with_projection(table, condition, options.projection)
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

    /// Row-based select with projection (fallback path).
    ///
    /// # Errors
    /// Returns `TableNotFound` or `StorageError`.
    #[must_use = "query results should be used"]
    #[instrument(skip(self, condition, projection), fields(table = %table))]
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
    #[instrument(skip(self))]
    pub fn begin_transaction(&self) -> u64 {
        self.tx_manager.begin()
    }

    /// Get the transaction manager for advanced use cases.
    pub const fn tx_manager(&self) -> &TransactionManager {
        &self.tx_manager
    }

    /// Check if a transaction is active.
    #[instrument(skip(self), fields(tx_id))]
    pub fn is_transaction_active(&self, tx_id: u64) -> bool {
        self.tx_manager.is_active(tx_id)
    }

    /// Commit a transaction, making all changes permanent.
    ///
    /// # Errors
    /// Returns `TransactionNotFound` or `TransactionInactive`.
    #[instrument(skip(self), fields(tx_id))]
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
    #[instrument(skip(self), fields(tx_id))]
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
    #[instrument(skip(self, values), fields(tx_id, table = %table))]
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
    #[instrument(skip(self, condition, updates), fields(tx_id, table = %table))]
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

        let max_depth = self.config.max_condition_depth;
        let matching_rows: Result<Vec<(SlabRowId, Row, Vec<SlabColumnValue>)>> = slab_rows
            .into_iter()
            .filter_map(|(row_id, slab_row)| {
                let row = Self::slab_row_to_engine_row(&schema, row_id, slab_row.clone());
                match condition.evaluate_with_depth(&row, 0, max_depth) {
                    Ok(true) => Some(Ok((row_id, row, slab_row))),
                    Ok(false) => None,
                    Err(e) => Some(Err(e)),
                }
            })
            .collect();
        let matching_rows = matching_rows?;

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
    #[instrument(skip(self, condition), fields(tx_id, table = %table))]
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

        let max_depth = self.config.max_condition_depth;
        let to_delete: Result<Vec<(SlabRowId, Row, Vec<SlabColumnValue>)>> = slab_rows
            .into_iter()
            .filter_map(|(row_id, slab_row)| {
                let row = Self::slab_row_to_engine_row(&schema, row_id, slab_row.clone());
                match condition.evaluate_with_depth(&row, 0, max_depth) {
                    Ok(true) => Some(Ok((row_id, row, slab_row))),
                    Ok(false) => None,
                    Err(e) => Some(Err(e)),
                }
            })
            .collect();
        let to_delete = to_delete?;

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
    #[instrument(skip(self, condition), fields(tx_id, table = %table))]
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
    #[instrument(skip(self))]
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
mod tests;

#[cfg(all(test, feature = "test-internals"))]
mod tensor_eval_tests;
