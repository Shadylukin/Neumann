// SPDX-License-Identifier: MIT OR Apache-2.0
//! Columnar storage for relational data.
//!
//! `RelationalSlab` stores table rows in a column-oriented format, optimizing
//! for scan operations and batch inserts. Each column is stored as a separate
//! vector, providing cache-friendly access patterns for analytical queries.
//!
//! # Design Philosophy
//!
//! - Column-oriented storage for efficient scans
//! - Deletion via bitmap (no data movement)
//! - B-tree indexes that split nodes (no resize stalls)
//! - Row IDs are stable (append-only)

use std::{
    collections::{BTreeMap, HashMap},
    sync::atomic::{AtomicU64, Ordering},
};

use bitvec::prelude::*;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Range operation for B-tree index queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeOp {
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

/// Row identifier within a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RowId(pub u64);

impl RowId {
    /// Creates a new row ID.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying u64 value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Convert to array index. Truncates on 32-bit platforms (acceptable since
    /// `Vec` capacity is also limited to `usize::MAX`).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn as_index(self) -> usize {
        self.0 as usize
    }
}

impl From<u64> for RowId {
    fn from(id: u64) -> Self {
        Self(id)
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
    /// Binary data.
    Bytes,
    /// JSON string.
    Json,
}

/// Column definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    /// Column name.
    pub name: String,
    /// Column data type.
    pub col_type: ColumnType,
    /// Whether null values are allowed.
    pub nullable: bool,
}

impl ColumnDef {
    /// Creates a new column definition.
    #[must_use]
    pub fn new(name: &str, col_type: ColumnType, nullable: bool) -> Self {
        Self {
            name: name.to_string(),
            col_type,
            nullable,
        }
    }
}

/// Table schema definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    /// Column definitions.
    pub columns: Vec<ColumnDef>,
    /// Primary key column name.
    pub primary_key: Option<String>,
}

impl TableSchema {
    /// Creates a schema with no primary key.
    #[must_use]
    pub const fn new(columns: Vec<ColumnDef>) -> Self {
        Self {
            columns,
            primary_key: None,
        }
    }

    /// Sets the primary key column.
    #[must_use]
    pub fn with_primary_key(mut self, column: &str) -> Self {
        self.primary_key = Some(column.to_string());
        self
    }

    /// Finds the index of a column by name.
    #[must_use]
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }
}

/// Column value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColumnValue {
    /// Null/missing value.
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
    /// JSON string.
    Json(String),
}

impl ColumnValue {
    /// Returns true if this is a null value.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}

/// A row of column values.
pub type Row = Vec<ColumnValue>;

/// Columnar storage for a single column.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum ColumnStorage {
    Int(Vec<i64>),
    Float(Vec<f64>),
    String(Vec<Option<String>>),
    Bool(BitVec),
    Bytes(Vec<Option<Vec<u8>>>),
    Json(Vec<Option<String>>),
}

impl ColumnStorage {
    fn new(col_type: &ColumnType, capacity: usize) -> Self {
        match col_type {
            ColumnType::Int => Self::Int(Vec::with_capacity(capacity)),
            ColumnType::Float => Self::Float(Vec::with_capacity(capacity)),
            ColumnType::String => Self::String(Vec::with_capacity(capacity)),
            ColumnType::Bool => Self::Bool(BitVec::with_capacity(capacity)),
            ColumnType::Bytes => Self::Bytes(Vec::with_capacity(capacity)),
            ColumnType::Json => Self::Json(Vec::with_capacity(capacity)),
        }
    }

    #[allow(clippy::match_same_arms)]
    fn push(&mut self, value: &ColumnValue) {
        match (self, value) {
            (Self::Int(v), ColumnValue::Int(i)) => v.push(*i),
            (Self::Int(v), ColumnValue::Null) => v.push(0),
            (Self::Float(v), ColumnValue::Float(f)) => v.push(*f),
            (Self::Float(v), ColumnValue::Null) => v.push(0.0),
            (Self::String(v), ColumnValue::String(s)) => v.push(Some(s.clone())),
            (Self::String(v), ColumnValue::Null) => v.push(None),
            (Self::Bool(v), ColumnValue::Bool(b)) => v.push(*b),
            (Self::Bool(v), ColumnValue::Null) => v.push(false),
            (Self::Bytes(v), ColumnValue::Bytes(b)) => v.push(Some(b.clone())),
            (Self::Bytes(v), ColumnValue::Null) => v.push(None),
            (Self::Json(v), ColumnValue::Json(j)) => v.push(Some(j.clone())),
            (Self::Json(v), ColumnValue::Null) => v.push(None),
            _ => {}, // Type mismatch - should be validated before
        }
    }

    fn get(&self, idx: usize) -> ColumnValue {
        match self {
            Self::Int(v) => v
                .get(idx)
                .map_or(ColumnValue::Null, |i| ColumnValue::Int(*i)),
            Self::Float(v) => v
                .get(idx)
                .map_or(ColumnValue::Null, |f| ColumnValue::Float(*f)),
            Self::String(v) => v
                .get(idx)
                .and_then(Clone::clone)
                .map_or(ColumnValue::Null, ColumnValue::String),
            Self::Bool(v) => v
                .get(idx)
                .map_or(ColumnValue::Null, |b| ColumnValue::Bool(*b)),
            Self::Bytes(v) => v
                .get(idx)
                .and_then(Clone::clone)
                .map_or(ColumnValue::Null, ColumnValue::Bytes),
            Self::Json(v) => v
                .get(idx)
                .and_then(Clone::clone)
                .map_or(ColumnValue::Null, ColumnValue::Json),
        }
    }

    #[allow(clippy::match_same_arms)]
    fn set(&mut self, idx: usize, value: &ColumnValue) {
        match (self, value) {
            (Self::Int(v), ColumnValue::Int(i)) => {
                if idx < v.len() {
                    v[idx] = *i;
                }
            },
            (Self::Float(v), ColumnValue::Float(f)) => {
                if idx < v.len() {
                    v[idx] = *f;
                }
            },
            (Self::String(v), ColumnValue::String(s)) => {
                if idx < v.len() {
                    v[idx] = Some(s.clone());
                }
            },
            (Self::String(v), ColumnValue::Null) => {
                if idx < v.len() {
                    v[idx] = None;
                }
            },
            (Self::Bool(v), ColumnValue::Bool(b)) => {
                if idx < v.len() {
                    v.set(idx, *b);
                }
            },
            (Self::Bytes(v), ColumnValue::Bytes(b)) => {
                if idx < v.len() {
                    v[idx] = Some(b.clone());
                }
            },
            (Self::Bytes(v), ColumnValue::Null) => {
                if idx < v.len() {
                    v[idx] = None;
                }
            },
            (Self::Json(v), ColumnValue::Json(j)) => {
                if idx < v.len() {
                    v[idx] = Some(j.clone());
                }
            },
            (Self::Json(v), ColumnValue::Null) => {
                if idx < v.len() {
                    v[idx] = None;
                }
            },
            _ => {}, // Type mismatch - should be validated before
        }
    }
}

/// Per-table storage with columnar layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TableStorage {
    schema: TableSchema,
    columns: Vec<ColumnStorage>,
    alive: BitVec,
    null_bitmaps: Vec<BitVec>,
    total_rows: usize,
    live_rows: usize,
    hash_indexes: HashMap<String, BTreeMap<i64, Vec<RowId>>>,
}

impl TableStorage {
    fn new(schema: TableSchema) -> Self {
        let num_cols = schema.columns.len();
        let columns: Vec<ColumnStorage> = schema
            .columns
            .iter()
            .map(|c| ColumnStorage::new(&c.col_type, 100))
            .collect();
        let null_bitmaps = vec![BitVec::new(); num_cols];

        Self {
            schema,
            columns,
            alive: BitVec::new(),
            null_bitmaps,
            total_rows: 0,
            live_rows: 0,
            hash_indexes: HashMap::new(),
        }
    }

    fn insert(&mut self, row: &Row) -> RowId {
        let row_id = RowId::new(self.total_rows as u64);

        for (col_idx, value) in row.iter().enumerate() {
            self.columns[col_idx].push(value);
            self.null_bitmaps[col_idx].push(value.is_null());
        }

        self.alive.push(true);
        self.total_rows += 1;
        self.live_rows += 1;

        // Update indexes
        for (col_name, index) in &mut self.hash_indexes {
            if let Some(col_idx) = self.schema.column_index(col_name) {
                if let ColumnValue::Int(key) = &row[col_idx] {
                    index.entry(*key).or_default().push(row_id);
                }
            }
        }

        row_id
    }

    fn get(&self, row_id: RowId) -> Option<Row> {
        let idx = row_id.as_index();
        if idx >= self.total_rows || !self.alive[idx] {
            return None;
        }

        let row: Row = self
            .columns
            .iter()
            .enumerate()
            .map(|(col_idx, col)| {
                if self.null_bitmaps[col_idx][idx] {
                    ColumnValue::Null
                } else {
                    col.get(idx)
                }
            })
            .collect();

        Some(row)
    }

    fn delete(&mut self, row_id: RowId) -> bool {
        let idx = row_id.as_index();
        if idx >= self.total_rows || !self.alive[idx] {
            return false;
        }

        self.alive.set(idx, false);
        self.live_rows -= 1;
        true
    }

    fn scan<F>(&self, predicate: F) -> Vec<(RowId, Row)>
    where
        F: Fn(&Row) -> bool,
    {
        let mut results = Vec::new();

        for idx in 0..self.total_rows {
            if !self.alive[idx] {
                continue;
            }

            let row: Row = self
                .columns
                .iter()
                .enumerate()
                .map(|(col_idx, col)| {
                    if self.null_bitmaps[col_idx][idx] {
                        ColumnValue::Null
                    } else {
                        col.get(idx)
                    }
                })
                .collect();

            if predicate(&row) {
                results.push((RowId::new(idx as u64), row));
            }
        }

        results
    }

    fn create_index(&mut self, column: &str) -> bool {
        if self.hash_indexes.contains_key(column) {
            return false;
        }

        let Some(col_idx) = self.schema.column_index(column) else {
            return false;
        };

        // Only support int indexes for now
        if self.schema.columns[col_idx].col_type != ColumnType::Int {
            return false;
        }

        let mut index = BTreeMap::new();

        if let ColumnStorage::Int(values) = &self.columns[col_idx] {
            for (idx, value) in values.iter().enumerate() {
                if self.alive[idx] {
                    index
                        .entry(*value)
                        .or_insert_with(Vec::new)
                        .push(RowId::new(idx as u64));
                }
            }
        }

        self.hash_indexes.insert(column.to_string(), index);
        true
    }

    fn index_lookup(&self, column: &str, key: i64) -> Vec<RowId> {
        self.hash_indexes
            .get(column)
            .and_then(|idx| idx.get(&key))
            .map(|rows| {
                rows.iter()
                    .filter(|r| self.alive[r.as_index()])
                    .copied()
                    .collect()
            })
            .unwrap_or_default()
    }

    fn index_range(&self, column: &str, op: RangeOp, key: i64) -> Vec<RowId> {
        let Some(index) = self.hash_indexes.get(column) else {
            return Vec::new();
        };

        let alive = &self.alive;
        let filter_alive = |rows: &Vec<RowId>| -> Vec<RowId> {
            rows.iter()
                .filter(|r| alive[r.as_index()])
                .copied()
                .collect()
        };

        let mut result = Vec::new();

        match op {
            RangeOp::Lt => {
                for (_, rows) in index.range(..key) {
                    result.extend(filter_alive(rows));
                }
            },
            RangeOp::Le => {
                for (_, rows) in index.range(..=key) {
                    result.extend(filter_alive(rows));
                }
            },
            RangeOp::Gt => {
                for (_, rows) in
                    index.range((std::ops::Bound::Excluded(key), std::ops::Bound::Unbounded))
                {
                    result.extend(filter_alive(rows));
                }
            },
            RangeOp::Ge => {
                for (_, rows) in index.range(key..) {
                    result.extend(filter_alive(rows));
                }
            },
        }

        result
    }

    fn index_between(&self, column: &str, min: i64, max: i64) -> Vec<RowId> {
        let Some(index) = self.hash_indexes.get(column) else {
            return Vec::new();
        };

        let mut result = Vec::new();
        for (_, rows) in index.range(min..=max) {
            for row_id in rows {
                if self.alive[row_id.as_index()] {
                    result.push(*row_id);
                }
            }
        }
        result
    }

    fn update(&mut self, row_id: RowId, updates: &[(String, ColumnValue)]) -> bool {
        let idx = row_id.as_index();
        if idx >= self.total_rows || !self.alive[idx] {
            return false;
        }

        for (col_name, value) in updates {
            if let Some(col_idx) = self.schema.column_index(col_name) {
                self.columns[col_idx].set(idx, value);
                self.null_bitmaps[col_idx].set(idx, value.is_null());
            }
        }
        true
    }

    fn restore_row(&mut self, row_id: RowId, values: &[ColumnValue]) -> bool {
        let idx = row_id.as_index();
        if idx >= self.total_rows || !self.alive[idx] {
            return false;
        }
        if values.len() != self.schema.columns.len() {
            return false;
        }

        for (col_idx, value) in values.iter().enumerate() {
            self.columns[col_idx].set(idx, value);
            self.null_bitmaps[col_idx].set(idx, value.is_null());
        }
        true
    }

    fn restore_deleted_row(&mut self, row_id: RowId, values: &[ColumnValue]) -> bool {
        let idx = row_id.as_index();
        if idx >= self.total_rows || self.alive[idx] {
            // Row doesn't exist or is already alive
            return false;
        }
        if values.len() != self.schema.columns.len() {
            return false;
        }

        // Mark as alive
        self.alive.set(idx, true);
        self.live_rows += 1;

        // Restore values
        for (col_idx, value) in values.iter().enumerate() {
            self.columns[col_idx].set(idx, value);
            self.null_bitmaps[col_idx].set(idx, value.is_null());
        }
        true
    }

    fn get_rows_by_indices(&self, indices: &[usize]) -> Vec<(RowId, Row)> {
        let mut results = Vec::with_capacity(indices.len());
        for &idx in indices {
            if idx < self.total_rows && self.alive[idx] {
                let row: Row = self
                    .columns
                    .iter()
                    .enumerate()
                    .map(|(col_idx, col)| {
                        if self.null_bitmaps[col_idx][idx] {
                            ColumnValue::Null
                        } else {
                            col.get(idx)
                        }
                    })
                    .collect();
                results.push((RowId::new(idx as u64), row));
            }
        }
        results
    }

    fn get_int_column(&self, column: &str) -> Option<(&[i64], Vec<u64>, Vec<u64>)> {
        let col_idx = self.schema.column_index(column)?;
        if let ColumnStorage::Int(values) = &self.columns[col_idx] {
            let alive_words = self.alive_as_words();
            let null_words = self.null_bitmap_as_words(col_idx);
            Some((values.as_slice(), alive_words, null_words))
        } else {
            None
        }
    }

    fn get_float_column(&self, column: &str) -> Option<(&[f64], Vec<u64>, Vec<u64>)> {
        let col_idx = self.schema.column_index(column)?;
        if let ColumnStorage::Float(values) = &self.columns[col_idx] {
            let alive_words = self.alive_as_words();
            let null_words = self.null_bitmap_as_words(col_idx);
            Some((values.as_slice(), alive_words, null_words))
        } else {
            None
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn alive_as_words(&self) -> Vec<u64> {
        // BitVec uses usize storage; on 64-bit we can cast directly
        self.alive
            .as_raw_slice()
            .iter()
            .map(|&word| word as u64)
            .collect()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn null_bitmap_as_words(&self, col_idx: usize) -> Vec<u64> {
        // BitVec uses usize storage; on 64-bit we can cast directly
        self.null_bitmaps[col_idx]
            .as_raw_slice()
            .iter()
            .map(|&word| word as u64)
            .collect()
    }

    fn add_column(&mut self, col_def: ColumnDef, default: Option<&ColumnValue>) -> bool {
        // Check if column already exists
        if self.schema.column_index(&col_def.name).is_some() {
            return false;
        }

        // For non-nullable columns without default, table must be empty
        if !col_def.nullable && default.is_none() && self.live_rows > 0 {
            return false;
        }

        // Create storage for new column
        let mut col_storage = ColumnStorage::new(&col_def.col_type, self.total_rows.max(100));

        // Create null bitmap for new column
        let mut null_bitmap = BitVec::with_capacity(self.total_rows);

        // Fill with default/null for existing rows
        for idx in 0..self.total_rows {
            if self.alive[idx] {
                if let Some(val) = default {
                    col_storage.push(val);
                    null_bitmap.push(val.is_null());
                } else {
                    col_storage.push(&ColumnValue::Null);
                    null_bitmap.push(true);
                }
            } else {
                // Deleted rows also need placeholder values
                col_storage.push(&ColumnValue::Null);
                null_bitmap.push(true);
            }
        }

        // Update schema
        self.schema.columns.push(col_def);
        self.columns.push(col_storage);
        self.null_bitmaps.push(null_bitmap);

        true
    }

    fn drop_column(&mut self, column: &str) -> bool {
        let Some(col_idx) = self.schema.column_index(column) else {
            return false;
        };

        // Remove from schema
        self.schema.columns.remove(col_idx);

        // Remove column storage
        self.columns.remove(col_idx);

        // Remove null bitmap
        self.null_bitmaps.remove(col_idx);

        // Remove any index on this column
        self.hash_indexes.remove(column);

        true
    }
}

/// Columnar storage for relational tables.
///
/// # Thread Safety
///
/// Uses `parking_lot::RwLock` for concurrent access.
pub struct RelationalSlab {
    tables: RwLock<BTreeMap<String, TableStorage>>,
    next_table_id: AtomicU64,
}

impl RelationalSlab {
    /// Creates an empty relational slab.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            tables: RwLock::new(BTreeMap::new()),
            next_table_id: AtomicU64::new(0),
        }
    }

    /// Create a new table.
    ///
    /// # Errors
    ///
    /// Returns an error if a table with the given name already exists.
    pub fn create_table(&self, name: &str, schema: TableSchema) -> Result<(), RelationalError> {
        let mut tables = self.tables.write();
        if tables.contains_key(name) {
            return Err(RelationalError::TableExists(name.to_string()));
        }
        tables.insert(name.to_string(), TableStorage::new(schema));
        drop(tables);
        self.next_table_id.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Drop a table.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist.
    pub fn drop_table(&self, name: &str) -> Result<(), RelationalError> {
        let mut tables = self.tables.write();
        let removed = tables.remove(name).is_some();
        drop(tables);
        if !removed {
            return Err(RelationalError::TableNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Check if a table exists.
    pub fn table_exists(&self, name: &str) -> bool {
        self.tables.read().contains_key(name)
    }

    /// Get table schema.
    pub fn get_schema(&self, name: &str) -> Option<TableSchema> {
        self.tables.read().get(name).map(|t| t.schema.clone())
    }

    /// Add a column to an existing table.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or if the column already exists.
    #[allow(clippy::significant_drop_tightening)] // Lock needed for entire operation
    pub fn add_column(
        &self,
        table: &str,
        col_def: ColumnDef,
        default: Option<&ColumnValue>,
    ) -> Result<(), RelationalError> {
        let mut tables = self.tables.write();
        let storage = tables
            .get_mut(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        if !storage.add_column(col_def.clone(), default) {
            return Err(RelationalError::ColumnExists(col_def.name));
        }
        Ok(())
    }

    /// Drop a column from a table.
    ///
    /// # Errors
    ///
    /// Returns an error if the table or column does not exist.
    #[allow(clippy::significant_drop_tightening)] // Lock needed for entire operation
    pub fn drop_column(&self, table: &str, column: &str) -> Result<(), RelationalError> {
        let mut tables = self.tables.write();
        let storage = tables
            .get_mut(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        if !storage.drop_column(column) {
            return Err(RelationalError::ColumnNotFound(column.to_string()));
        }
        Ok(())
    }

    /// Insert a row.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or if the row has the wrong number of columns.
    #[allow(clippy::needless_pass_by_value)] // API consistency with batch_insert
    pub fn insert(&self, table: &str, row: Row) -> Result<RowId, RelationalError> {
        let mut tables = self.tables.write();
        let storage = tables
            .get_mut(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        if row.len() != storage.schema.columns.len() {
            return Err(RelationalError::ColumnMismatch {
                expected: storage.schema.columns.len(),
                actual: row.len(),
            });
        }

        let row_id = storage.insert(&row);
        drop(tables);
        Ok(row_id)
    }

    /// Batch insert rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or if any row has the wrong number of columns.
    pub fn batch_insert(&self, table: &str, rows: Vec<Row>) -> Result<Vec<RowId>, RelationalError> {
        let mut tables = self.tables.write();
        let storage = tables
            .get_mut(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        let mut row_ids = Vec::with_capacity(rows.len());
        for row in rows {
            if row.len() != storage.schema.columns.len() {
                return Err(RelationalError::ColumnMismatch {
                    expected: storage.schema.columns.len(),
                    actual: row.len(),
                });
            }
            row_ids.push(storage.insert(&row));
        }
        drop(tables);

        Ok(row_ids)
    }

    /// Get a row by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist.
    pub fn get(&self, table: &str, row_id: RowId) -> Result<Option<Row>, RelationalError> {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        let result = storage.get(row_id);
        drop(tables);
        Ok(result)
    }

    /// Delete a row.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist.
    pub fn delete(&self, table: &str, row_id: RowId) -> Result<bool, RelationalError> {
        let mut tables = self.tables.write();
        let storage = tables
            .get_mut(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        let deleted = storage.delete(row_id);
        drop(tables);
        Ok(deleted)
    }

    /// Scan a table with a predicate.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist.
    pub fn scan<F>(&self, table: &str, predicate: F) -> Result<Vec<(RowId, Row)>, RelationalError>
    where
        F: Fn(&Row) -> bool,
    {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        let results = storage.scan(predicate);
        drop(tables);
        Ok(results)
    }

    /// Get all rows from a table.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist.
    pub fn scan_all(&self, table: &str) -> Result<Vec<(RowId, Row)>, RelationalError> {
        self.scan(table, |_| true)
    }

    /// Create an index on a column.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or if index creation fails.
    pub fn create_index(&self, table: &str, column: &str) -> Result<(), RelationalError> {
        let mut tables = self.tables.write();
        let storage = tables
            .get_mut(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        let created = storage.create_index(column);
        drop(tables);
        if !created {
            return Err(RelationalError::IndexCreationFailed(column.to_string()));
        }

        Ok(())
    }

    /// Look up rows by indexed column.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist.
    pub fn index_lookup(
        &self,
        table: &str,
        column: &str,
        key: i64,
    ) -> Result<Vec<RowId>, RelationalError> {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        let result = storage.index_lookup(column, key);
        drop(tables);
        Ok(result)
    }

    /// Range query on indexed column using B-tree's O(log n) range operations.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist.
    pub fn index_range(
        &self,
        table: &str,
        column: &str,
        op: RangeOp,
        key: i64,
    ) -> Result<Vec<RowId>, RelationalError> {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        let result = storage.index_range(column, op, key);
        drop(tables);
        Ok(result)
    }

    /// Range query for values between min and max (inclusive).
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist.
    pub fn index_between(
        &self,
        table: &str,
        column: &str,
        min: i64,
        max: i64,
    ) -> Result<Vec<RowId>, RelationalError> {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        let result = storage.index_between(column, min, max);
        drop(tables);
        Ok(result)
    }

    /// Get the number of live rows in a table.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist.
    pub fn row_count(&self, table: &str) -> Result<usize, RelationalError> {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        let count = storage.live_rows;
        drop(tables);
        Ok(count)
    }

    /// Get the number of tables.
    pub fn table_count(&self) -> usize {
        self.tables.read().len()
    }

    /// Get all table names.
    pub fn table_names(&self) -> Vec<String> {
        self.tables.read().keys().cloned().collect()
    }

    /// Clear all tables.
    pub fn clear(&self) {
        self.tables.write().clear();
    }

    /// Get serializable state for snapshots.
    pub fn snapshot(&self) -> RelationalSlabSnapshot {
        RelationalSlabSnapshot {
            tables: self.tables.read().clone(),
        }
    }

    /// Restores from a snapshot.
    #[must_use]
    pub fn restore(snapshot: RelationalSlabSnapshot) -> Self {
        Self {
            tables: RwLock::new(snapshot.tables),
            next_table_id: AtomicU64::new(0),
        }
    }

    /// Update a row's columns.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or the row was not found.
    pub fn update_row(
        &self,
        table: &str,
        row_id: RowId,
        updates: &[(String, ColumnValue)],
    ) -> Result<(), RelationalError> {
        let mut tables = self.tables.write();
        let storage = tables
            .get_mut(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        if !storage.update(row_id, updates) {
            return Err(RelationalError::RowNotFound(row_id));
        }
        drop(tables);
        Ok(())
    }

    /// Restore all columns of a row to given values (for transaction rollback).
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist, the row was not found,
    /// or the values count doesn't match the schema.
    pub fn restore_row(
        &self,
        table: &str,
        row_id: RowId,
        values: &[ColumnValue],
    ) -> Result<(), RelationalError> {
        let mut tables = self.tables.write();
        let storage = tables
            .get_mut(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        if !storage.restore_row(row_id, values) {
            return Err(RelationalError::RowNotFound(row_id));
        }
        drop(tables);
        Ok(())
    }

    /// Restore a deleted row (for transaction rollback).
    ///
    /// Marks the row as alive and restores its values.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist, the row was not deleted,
    /// or the values count doesn't match the schema.
    pub fn restore_deleted_row(
        &self,
        table: &str,
        row_id: RowId,
        values: &[ColumnValue],
    ) -> Result<(), RelationalError> {
        let mut tables = self.tables.write();
        let storage = tables
            .get_mut(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        if !storage.restore_deleted_row(row_id, values) {
            return Err(RelationalError::RowNotFound(row_id));
        }
        drop(tables);
        Ok(())
    }

    /// Get multiple rows by their indices.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist.
    pub fn get_rows_by_indices(
        &self,
        table: &str,
        indices: &[usize],
    ) -> Result<Vec<(RowId, Row)>, RelationalError> {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        let result = storage.get_rows_by_indices(indices);
        drop(tables);
        Ok(result)
    }

    /// Get an integer column's raw data along with alive and null bitmaps.
    ///
    /// Returns `(values, alive_bitmap_words, null_bitmap_words)` where bitmaps
    /// are packed as 64-bit words for SIMD operations.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or the column is not an int type.
    #[allow(clippy::type_complexity, clippy::significant_drop_tightening)]
    pub fn get_int_column(
        &self,
        table: &str,
        column: &str,
    ) -> Result<(Vec<i64>, Vec<u64>, Vec<u64>), RelationalError> {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        storage
            .get_int_column(column)
            .map(|(values, alive, nulls)| (values.to_vec(), alive, nulls))
            .ok_or_else(|| RelationalError::ColumnNotFound(column.to_string()))
    }

    /// Get a float column's raw data along with alive and null bitmaps.
    ///
    /// Returns `(values, alive_bitmap_words, null_bitmap_words)` where bitmaps
    /// are packed as 64-bit words for SIMD operations.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or the column is not a float type.
    #[allow(clippy::type_complexity, clippy::significant_drop_tightening)]
    pub fn get_float_column(
        &self,
        table: &str,
        column: &str,
    ) -> Result<(Vec<f64>, Vec<u64>, Vec<u64>), RelationalError> {
        let tables = self.tables.read();
        let storage = tables
            .get(table)
            .ok_or_else(|| RelationalError::TableNotFound(table.to_string()))?;

        storage
            .get_float_column(column)
            .map(|(values, alive, nulls)| (values.to_vec(), alive, nulls))
            .ok_or_else(|| RelationalError::ColumnNotFound(column.to_string()))
    }
}

impl Default for RelationalSlab {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors from `RelationalSlab` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationalError {
    /// Table does not exist.
    TableNotFound(String),
    /// Table already exists.
    TableExists(String),
    /// Row has wrong number of columns.
    ColumnMismatch {
        /// Expected column count.
        expected: usize,
        /// Actual column count.
        actual: usize,
    },
    /// Index creation failed.
    IndexCreationFailed(String),
    /// Row does not exist or was deleted.
    RowNotFound(RowId),
    /// Column does not exist or wrong type.
    ColumnNotFound(String),
    /// Column already exists.
    ColumnExists(String),
}

impl std::fmt::Display for RelationalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TableNotFound(name) => write!(f, "table not found: {name}"),
            Self::TableExists(name) => write!(f, "table already exists: {name}"),
            Self::ColumnMismatch { expected, actual } => {
                write!(f, "column mismatch: expected {expected}, got {actual}")
            },
            Self::IndexCreationFailed(col) => {
                write!(f, "index creation failed for column: {col}")
            },
            Self::RowNotFound(row_id) => write!(f, "row not found: {}", row_id.0),
            Self::ColumnNotFound(col) => write!(f, "column not found: {col}"),
            Self::ColumnExists(col) => write!(f, "column already exists: {col}"),
        }
    }
}

impl std::error::Error for RelationalError {}

/// Serializable snapshot of `RelationalSlab` state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationalSlabSnapshot {
    tables: BTreeMap<String, TableStorage>,
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread, time::Instant};

    use super::*;

    fn create_test_schema() -> TableSchema {
        TableSchema::new(vec![
            ColumnDef::new("id", ColumnType::Int, false),
            ColumnDef::new("name", ColumnType::String, true),
            ColumnDef::new("age", ColumnType::Int, true),
        ])
    }

    fn create_test_row(id: i64, name: &str, age: i64) -> Row {
        vec![
            ColumnValue::Int(id),
            ColumnValue::String(name.to_string()),
            ColumnValue::Int(age),
        ]
    }

    #[test]
    fn test_new() {
        let slab = RelationalSlab::new();
        assert_eq!(slab.table_count(), 0);
    }

    #[test]
    fn test_create_table() {
        let slab = RelationalSlab::new();
        let schema = create_test_schema();

        slab.create_table("users", schema).unwrap();

        assert!(slab.table_exists("users"));
        assert_eq!(slab.table_count(), 1);
    }

    #[test]
    fn test_create_duplicate_table() {
        let slab = RelationalSlab::new();
        let schema = create_test_schema();

        slab.create_table("users", schema.clone()).unwrap();
        let result = slab.create_table("users", schema);

        assert!(matches!(result, Err(RelationalError::TableExists(_))));
    }

    #[test]
    fn test_drop_table() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        slab.drop_table("users").unwrap();

        assert!(!slab.table_exists("users"));
    }

    #[test]
    fn test_drop_nonexistent_table() {
        let slab = RelationalSlab::new();
        let result = slab.drop_table("users");

        assert!(matches!(result, Err(RelationalError::TableNotFound(_))));
    }

    #[test]
    fn test_insert_get() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row = create_test_row(1, "Alice", 30);
        let row_id = slab.insert("users", row.clone()).unwrap();

        let retrieved = slab.get("users", row_id).unwrap().unwrap();
        assert_eq!(retrieved, row);
    }

    #[test]
    fn test_insert_column_mismatch() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let bad_row = vec![
            ColumnValue::Int(1),
            ColumnValue::String("Alice".to_string()),
        ];
        let result = slab.insert("users", bad_row);

        assert!(matches!(
            result,
            Err(RelationalError::ColumnMismatch { .. })
        ));
    }

    #[test]
    fn test_batch_insert() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let rows = vec![
            create_test_row(1, "Alice", 30),
            create_test_row(2, "Bob", 25),
            create_test_row(3, "Carol", 35),
        ];

        let row_ids = slab.batch_insert("users", rows).unwrap();
        assert_eq!(row_ids.len(), 3);
        assert_eq!(slab.row_count("users").unwrap(), 3);
    }

    #[test]
    fn test_delete() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();
        assert_eq!(slab.row_count("users").unwrap(), 1);

        slab.delete("users", row_id).unwrap();
        assert_eq!(slab.row_count("users").unwrap(), 0);

        let retrieved = slab.get("users", row_id).unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_scan() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20 + i))
                .unwrap();
        }

        // Scan for age > 25
        let results = slab
            .scan(
                "users",
                |row| matches!(&row[2], ColumnValue::Int(age) if *age > 25),
            )
            .unwrap();

        assert_eq!(results.len(), 4); // ages 26, 27, 28, 29
    }

    #[test]
    fn test_scan_all() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..5 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20 + i))
                .unwrap();
        }

        let results = slab.scan_all("users").unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_create_index() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        let results = slab.index_lookup("users", "id", 5).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_index_lookup_after_delete() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();
        slab.create_index("users", "id").unwrap();

        // Before delete
        let results = slab.index_lookup("users", "id", 1).unwrap();
        assert_eq!(results.len(), 1);

        // After delete
        slab.delete("users", row_id).unwrap();
        let results = slab.index_lookup("users", "id", 1).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_get_schema() {
        let slab = RelationalSlab::new();
        let schema = create_test_schema();
        slab.create_table("users", schema.clone()).unwrap();

        let retrieved = slab.get_schema("users").unwrap();
        assert_eq!(retrieved.columns.len(), 3);
    }

    #[test]
    fn test_table_names() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();
        slab.create_table("posts", create_test_schema()).unwrap();

        let names = slab.table_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"users".to_string()));
        assert!(names.contains(&"posts".to_string()));
    }

    #[test]
    fn test_clear() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();
        slab.create_table("posts", create_test_schema()).unwrap();

        slab.clear();

        assert_eq!(slab.table_count(), 0);
    }

    #[test]
    fn test_snapshot_restore() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();
        slab.insert("users", create_test_row(1, "Alice", 30))
            .unwrap();

        let snapshot = slab.snapshot();
        let restored = RelationalSlab::restore(snapshot);

        assert!(restored.table_exists("users"));
        assert_eq!(restored.row_count("users").unwrap(), 1);
    }

    #[test]
    fn test_concurrent_reads_writes() {
        let slab = Arc::new(RelationalSlab::new());
        slab.create_table("users", create_test_schema()).unwrap();

        let mut handles = vec![];

        // Writer threads
        for t in 0..4 {
            let s = Arc::clone(&slab);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let _ = s.insert(
                        "users",
                        create_test_row(t * 1000 + i, &format!("User{}", i), 20),
                    );
                }
            }));
        }

        // Reader threads
        for _ in 0..4 {
            let s = Arc::clone(&slab);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let _ = s.row_count("users");
                    let _ = s.scan_all("users");
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(slab.row_count("users").unwrap(), 400);
    }

    #[test]
    fn test_no_resize_stall() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let count = 10_000;
        let start = Instant::now();
        let mut max_op_time = std::time::Duration::ZERO;

        for i in 0..count as i64 {
            let op_start = Instant::now();
            let _ = slab.insert("users", create_test_row(i, &format!("User{}", i), 20));
            let op_time = op_start.elapsed();
            if op_time > max_op_time {
                max_op_time = op_time;
            }
        }

        let total_time = start.elapsed();

        assert!(
            max_op_time.as_millis() < 100,
            "Max operation time {:?} exceeded 100ms threshold",
            max_op_time
        );

        let ops_per_sec = count as f64 / total_time.as_secs_f64();
        assert!(
            ops_per_sec > 5_000.0, // Lower threshold accounts for coverage overhead
            "Throughput {:.0} ops/sec too low",
            ops_per_sec
        );
    }

    #[test]
    fn test_null_values() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row = vec![ColumnValue::Int(1), ColumnValue::Null, ColumnValue::Null];
        let row_id = slab.insert("users", row.clone()).unwrap();

        let retrieved = slab.get("users", row_id).unwrap().unwrap();
        assert_eq!(retrieved[1], ColumnValue::Null);
        assert_eq!(retrieved[2], ColumnValue::Null);
    }

    #[test]
    fn test_error_display() {
        let err = RelationalError::TableNotFound("users".to_string());
        assert!(err.to_string().contains("users"));

        let err = RelationalError::ColumnMismatch {
            expected: 3,
            actual: 2,
        };
        assert!(err.to_string().contains("3"));
    }

    #[test]
    fn test_default() {
        let slab = RelationalSlab::default();
        assert_eq!(slab.table_count(), 0);
    }

    #[test]
    fn test_column_value_is_null() {
        assert!(ColumnValue::Null.is_null());
        assert!(!ColumnValue::Int(1).is_null());
    }

    #[test]
    fn test_column_def_new() {
        let def = ColumnDef::new("name", ColumnType::String, true);
        assert_eq!(def.name, "name");
        assert_eq!(def.col_type, ColumnType::String);
        assert!(def.nullable);
    }

    #[test]
    fn test_table_schema_column_index() {
        let schema = create_test_schema();
        assert_eq!(schema.column_index("id"), Some(0));
        assert_eq!(schema.column_index("name"), Some(1));
        assert_eq!(schema.column_index("nonexistent"), None);
    }

    #[test]
    fn test_row_id_from_u64() {
        let row_id: RowId = 42u64.into();
        assert_eq!(row_id.as_u64(), 42);
    }

    #[test]
    fn test_with_primary_key() {
        let schema = TableSchema::new(vec![ColumnDef::new("id", ColumnType::Int, false)])
            .with_primary_key("id");

        assert_eq!(schema.primary_key, Some("id".to_string()));
    }

    #[test]
    fn test_float_and_bool_columns() {
        let schema = TableSchema::new(vec![
            ColumnDef::new("score", ColumnType::Float, false),
            ColumnDef::new("active", ColumnType::Bool, false),
        ]);

        let slab = RelationalSlab::new();
        slab.create_table("data", schema).unwrap();

        // Insert row with float and bool
        let row = vec![ColumnValue::Float(3.14), ColumnValue::Bool(true)];
        let row_id = slab.insert("data", row).unwrap();

        // Retrieve and verify
        let retrieved = slab.get("data", row_id).unwrap().unwrap();
        assert_eq!(retrieved[0], ColumnValue::Float(3.14));
        assert_eq!(retrieved[1], ColumnValue::Bool(true));
    }

    #[test]
    fn test_float_and_bool_null_values() {
        let schema = TableSchema::new(vec![
            ColumnDef::new("score", ColumnType::Float, true),
            ColumnDef::new("active", ColumnType::Bool, true),
        ]);

        let slab = RelationalSlab::new();
        slab.create_table("data", schema).unwrap();

        // Insert row with null float and bool
        let row = vec![ColumnValue::Null, ColumnValue::Null];
        let row_id = slab.insert("data", row).unwrap();

        // Retrieve and verify nulls
        let retrieved = slab.get("data", row_id).unwrap().unwrap();
        assert_eq!(retrieved[0], ColumnValue::Null);
        assert_eq!(retrieved[1], ColumnValue::Null);
    }

    #[test]
    fn test_index_update_on_insert() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        // Create index BEFORE inserting
        slab.create_index("users", "id").unwrap();

        // Now insert - this triggers the index update code path
        let row_id = slab
            .insert("users", create_test_row(100, "Alice", 30))
            .unwrap();

        // Verify index works
        let results = slab.index_lookup("users", "id", 100).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], row_id);
    }

    #[test]
    fn test_batch_insert_column_mismatch() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let bad_rows = vec![
            create_test_row(1, "Alice", 30),
            vec![ColumnValue::Int(2)], // Wrong number of columns
        ];

        let result = slab.batch_insert("users", bad_rows);
        assert!(matches!(
            result,
            Err(RelationalError::ColumnMismatch { .. })
        ));
    }

    #[test]
    fn test_create_index_on_non_int_column() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        // Try to create index on string column (not supported)
        let result = slab.create_index("users", "name");
        assert!(matches!(
            result,
            Err(RelationalError::IndexCreationFailed(_))
        ));
    }

    #[test]
    fn test_create_index_on_nonexistent_column() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let result = slab.create_index("users", "nonexistent");
        assert!(matches!(
            result,
            Err(RelationalError::IndexCreationFailed(_))
        ));
    }

    #[test]
    fn test_create_duplicate_index() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        slab.create_index("users", "id").unwrap();
        let result = slab.create_index("users", "id");
        assert!(matches!(
            result,
            Err(RelationalError::IndexCreationFailed(_))
        ));
    }

    #[test]
    fn test_scan_with_deleted_rows() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        // Insert multiple rows
        let id1 = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();
        let _id2 = slab.insert("users", create_test_row(2, "Bob", 25)).unwrap();
        let id3 = slab
            .insert("users", create_test_row(3, "Carol", 35))
            .unwrap();

        // Delete middle row
        slab.delete("users", id1).unwrap();

        // Scan should skip deleted row
        let results = slab.scan_all("users").unwrap();
        assert_eq!(results.len(), 2);

        // Verify we got the right rows (ids 2 and 3)
        let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
        assert!(ids.contains(&1)); // row_id 1 is the second insert
        assert!(ids.contains(&2)); // row_id 2 is the third insert

        // Verify row_id 0 (first insert) is not in results
        assert!(!ids.contains(&(id1.as_u64()))); // id1 is 0
        assert!(ids.contains(&(id3.as_u64() - 1))); // id2 is 1
    }

    #[test]
    fn test_delete_already_deleted() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();

        // Delete once
        assert!(slab.delete("users", row_id).unwrap());

        // Delete again - should return false
        assert!(!slab.delete("users", row_id).unwrap());
    }

    #[test]
    fn test_delete_out_of_bounds() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        // Try to delete row that was never inserted
        assert!(!slab.delete("users", RowId::new(999)).unwrap());
    }

    #[test]
    fn test_get_deleted_row() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();
        slab.delete("users", row_id).unwrap();

        // Get deleted row should return None
        assert!(slab.get("users", row_id).unwrap().is_none());
    }

    #[test]
    fn test_get_out_of_bounds() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        // Get row that was never inserted
        assert!(slab.get("users", RowId::new(999)).unwrap().is_none());
    }

    #[test]
    fn test_error_display_all_variants() {
        let e1 = RelationalError::TableNotFound("t".to_string());
        assert!(e1.to_string().contains("not found"));

        let e2 = RelationalError::TableExists("t".to_string());
        assert!(e2.to_string().contains("already exists"));

        let e3 = RelationalError::ColumnMismatch {
            expected: 3,
            actual: 2,
        };
        assert!(e3.to_string().contains("mismatch"));

        let e4 = RelationalError::IndexCreationFailed("c".to_string());
        assert!(e4.to_string().contains("index creation failed"));
    }

    #[test]
    fn test_scan_with_null_columns() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        // Insert row with null values
        let row = vec![ColumnValue::Int(1), ColumnValue::Null, ColumnValue::Null];
        slab.insert("users", row).unwrap();

        // Scan and verify nulls are preserved
        let results = slab.scan_all("users").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1[1], ColumnValue::Null);
        assert_eq!(results[0].1[2], ColumnValue::Null);
    }

    #[test]
    fn test_column_value_variants() {
        assert!(!ColumnValue::Float(1.0).is_null());
        assert!(!ColumnValue::Bool(true).is_null());
        assert!(!ColumnValue::String("test".to_string()).is_null());
    }

    #[test]
    fn test_index_range_lt() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        let results = slab.index_range("users", "id", RangeOp::Lt, 5).unwrap();
        assert_eq!(results.len(), 5); // ids 0, 1, 2, 3, 4
    }

    #[test]
    fn test_index_range_le() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        let results = slab.index_range("users", "id", RangeOp::Le, 5).unwrap();
        assert_eq!(results.len(), 6); // ids 0, 1, 2, 3, 4, 5
    }

    #[test]
    fn test_index_range_gt() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        let results = slab.index_range("users", "id", RangeOp::Gt, 5).unwrap();
        assert_eq!(results.len(), 4); // ids 6, 7, 8, 9
    }

    #[test]
    fn test_index_range_ge() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        let results = slab.index_range("users", "id", RangeOp::Ge, 5).unwrap();
        assert_eq!(results.len(), 5); // ids 5, 6, 7, 8, 9
    }

    #[test]
    fn test_index_between() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        let results = slab.index_between("users", "id", 3, 7).unwrap();
        assert_eq!(results.len(), 5); // ids 3, 4, 5, 6, 7
    }

    #[test]
    fn test_index_range_with_deleted_rows() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..10 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("users", "id").unwrap();

        // Delete some rows
        slab.delete("users", RowId::new(3)).unwrap();
        slab.delete("users", RowId::new(5)).unwrap();

        let results = slab.index_range("users", "id", RangeOp::Lt, 7).unwrap();
        assert_eq!(results.len(), 5); // ids 0, 1, 2, 4, 6 (3, 5 deleted)
    }

    #[test]
    fn test_index_range_no_index() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..5 {
            slab.insert("users", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        // No index created
        let results = slab.index_range("users", "id", RangeOp::Lt, 3).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_index_range_performance() {
        let slab = RelationalSlab::new();
        slab.create_table("perf", create_test_schema()).unwrap();

        let count = 10_000;
        for i in 0..count {
            slab.insert("perf", create_test_row(i, &format!("User{}", i), 20))
                .unwrap();
        }

        slab.create_index("perf", "id").unwrap();

        let start = Instant::now();
        let results = slab.index_range("perf", "id", RangeOp::Lt, 100).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 100);
        assert!(
            elapsed.as_micros() < 5000,
            "Range query took {:?}, expected < 5ms",
            elapsed
        );
    }

    // Tests for new column types (Bytes, Json)

    #[test]
    fn test_bytes_column() {
        let slab = RelationalSlab::new();
        let schema = TableSchema::new(vec![
            ColumnDef::new("id", ColumnType::Int, false),
            ColumnDef::new("data", ColumnType::Bytes, true),
        ]);
        slab.create_table("bindata", schema).unwrap();

        let row1 = vec![
            ColumnValue::Int(1),
            ColumnValue::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]),
        ];
        let row2 = vec![ColumnValue::Int(2), ColumnValue::Null];

        let id1 = slab.insert("bindata", row1).unwrap();
        let id2 = slab.insert("bindata", row2).unwrap();

        let retrieved1 = slab.get("bindata", id1).unwrap().unwrap();
        assert_eq!(
            retrieved1[1],
            ColumnValue::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF])
        );

        let retrieved2 = slab.get("bindata", id2).unwrap().unwrap();
        assert_eq!(retrieved2[1], ColumnValue::Null);
    }

    #[test]
    fn test_json_column() {
        let slab = RelationalSlab::new();
        let schema = TableSchema::new(vec![
            ColumnDef::new("id", ColumnType::Int, false),
            ColumnDef::new("metadata", ColumnType::Json, true),
        ]);
        slab.create_table("jsondata", schema).unwrap();

        let json_str = r#"{"name": "test", "value": 42}"#.to_string();
        let row1 = vec![ColumnValue::Int(1), ColumnValue::Json(json_str.clone())];
        let row2 = vec![ColumnValue::Int(2), ColumnValue::Null];

        let id1 = slab.insert("jsondata", row1).unwrap();
        let id2 = slab.insert("jsondata", row2).unwrap();

        let retrieved1 = slab.get("jsondata", id1).unwrap().unwrap();
        assert_eq!(retrieved1[1], ColumnValue::Json(json_str));

        let retrieved2 = slab.get("jsondata", id2).unwrap().unwrap();
        assert_eq!(retrieved2[1], ColumnValue::Null);
    }

    // Tests for update_row

    #[test]
    fn test_update_row() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();

        // Update name and age
        let updates = vec![
            ("name".to_string(), ColumnValue::String("Bob".to_string())),
            ("age".to_string(), ColumnValue::Int(35)),
        ];
        slab.update_row("users", row_id, &updates).unwrap();

        let retrieved = slab.get("users", row_id).unwrap().unwrap();
        assert_eq!(retrieved[0], ColumnValue::Int(1)); // id unchanged
        assert_eq!(retrieved[1], ColumnValue::String("Bob".to_string()));
        assert_eq!(retrieved[2], ColumnValue::Int(35));
    }

    #[test]
    fn test_update_row_to_null() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();

        // Update name to null
        let updates = vec![("name".to_string(), ColumnValue::Null)];
        slab.update_row("users", row_id, &updates).unwrap();

        let retrieved = slab.get("users", row_id).unwrap().unwrap();
        assert_eq!(retrieved[1], ColumnValue::Null);
    }

    #[test]
    fn test_update_row_not_found() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let updates = vec![("name".to_string(), ColumnValue::String("Bob".to_string()))];
        let result = slab.update_row("users", RowId::new(999), &updates);

        assert!(matches!(result, Err(RelationalError::RowNotFound(_))));
    }

    #[test]
    fn test_update_deleted_row() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();
        slab.delete("users", row_id).unwrap();

        let updates = vec![("name".to_string(), ColumnValue::String("Bob".to_string()))];
        let result = slab.update_row("users", row_id, &updates);

        assert!(matches!(result, Err(RelationalError::RowNotFound(_))));
    }

    // Tests for restore_row

    #[test]
    fn test_restore_row() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();

        // Modify the row
        let updates = vec![
            ("name".to_string(), ColumnValue::String("Bob".to_string())),
            ("age".to_string(), ColumnValue::Int(35)),
        ];
        slab.update_row("users", row_id, &updates).unwrap();

        // Restore to original values
        let original = vec![
            ColumnValue::Int(1),
            ColumnValue::String("Alice".to_string()),
            ColumnValue::Int(30),
        ];
        slab.restore_row("users", row_id, &original).unwrap();

        let retrieved = slab.get("users", row_id).unwrap().unwrap();
        assert_eq!(retrieved[0], ColumnValue::Int(1));
        assert_eq!(retrieved[1], ColumnValue::String("Alice".to_string()));
        assert_eq!(retrieved[2], ColumnValue::Int(30));
    }

    #[test]
    fn test_restore_row_table_not_found() {
        let slab = RelationalSlab::new();
        let values = vec![ColumnValue::Int(1)];
        let result = slab.restore_row("nonexistent", RowId::new(0), &values);
        assert!(matches!(result, Err(RelationalError::TableNotFound(_))));
    }

    #[test]
    fn test_restore_row_not_found() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let values = vec![
            ColumnValue::Int(1),
            ColumnValue::String("Alice".to_string()),
            ColumnValue::Int(30),
        ];
        let result = slab.restore_row("users", RowId::new(999), &values);
        assert!(matches!(result, Err(RelationalError::RowNotFound(_))));
    }

    #[test]
    fn test_restore_row_wrong_column_count() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();

        // Try to restore with wrong number of columns
        let values = vec![ColumnValue::Int(1), ColumnValue::Int(2)]; // Only 2 instead of 3
        let result = slab.restore_row("users", row_id, &values);
        assert!(matches!(result, Err(RelationalError::RowNotFound(_))));
    }

    // Tests for restore_deleted_row

    #[test]
    fn test_restore_deleted_row() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();

        // Delete the row
        slab.delete("users", row_id).unwrap();
        assert!(slab.get("users", row_id).unwrap().is_none());

        // Restore it
        let values = vec![
            ColumnValue::Int(1),
            ColumnValue::String("Alice".to_string()),
            ColumnValue::Int(30),
        ];
        slab.restore_deleted_row("users", row_id, &values).unwrap();

        // Row should be back
        let retrieved = slab.get("users", row_id).unwrap().unwrap();
        assert_eq!(retrieved[0], ColumnValue::Int(1));
        assert_eq!(retrieved[1], ColumnValue::String("Alice".to_string()));
        assert_eq!(retrieved[2], ColumnValue::Int(30));
    }

    #[test]
    fn test_restore_deleted_row_table_not_found() {
        let slab = RelationalSlab::new();
        let values = vec![ColumnValue::Int(1)];
        let result = slab.restore_deleted_row("nonexistent", RowId::new(0), &values);
        assert!(matches!(result, Err(RelationalError::TableNotFound(_))));
    }

    #[test]
    fn test_restore_deleted_row_not_deleted() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();

        // Try to restore a non-deleted row
        let values = vec![
            ColumnValue::Int(1),
            ColumnValue::String("Alice".to_string()),
            ColumnValue::Int(30),
        ];
        let result = slab.restore_deleted_row("users", row_id, &values);
        assert!(matches!(result, Err(RelationalError::RowNotFound(_))));
    }

    #[test]
    fn test_restore_deleted_row_wrong_column_count() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();
        slab.delete("users", row_id).unwrap();

        // Try to restore with wrong number of columns
        let values = vec![ColumnValue::Int(1)]; // Only 1 instead of 3
        let result = slab.restore_deleted_row("users", row_id, &values);
        assert!(matches!(result, Err(RelationalError::RowNotFound(_))));
    }

    #[test]
    fn test_restore_deleted_row_updates_live_count() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let row_id = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();
        assert_eq!(slab.row_count("users").unwrap(), 1);

        slab.delete("users", row_id).unwrap();
        assert_eq!(slab.row_count("users").unwrap(), 0);

        let values = vec![
            ColumnValue::Int(1),
            ColumnValue::String("Alice".to_string()),
            ColumnValue::Int(30),
        ];
        slab.restore_deleted_row("users", row_id, &values).unwrap();
        assert_eq!(slab.row_count("users").unwrap(), 1);
    }

    // Tests for get_rows_by_indices

    #[test]
    fn test_get_rows_by_indices() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        slab.insert("users", create_test_row(1, "Alice", 30))
            .unwrap();
        slab.insert("users", create_test_row(2, "Bob", 25)).unwrap();
        slab.insert("users", create_test_row(3, "Charlie", 35))
            .unwrap();

        let results = slab.get_rows_by_indices("users", &[0, 2]).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, RowId::new(0));
        assert_eq!(results[0].1[0], ColumnValue::Int(1));
        assert_eq!(results[1].0, RowId::new(2));
        assert_eq!(results[1].1[0], ColumnValue::Int(3));
    }

    #[test]
    fn test_get_rows_by_indices_with_deleted() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let id1 = slab
            .insert("users", create_test_row(1, "Alice", 30))
            .unwrap();
        slab.insert("users", create_test_row(2, "Bob", 25)).unwrap();
        slab.insert("users", create_test_row(3, "Charlie", 35))
            .unwrap();

        slab.delete("users", id1).unwrap();

        let results = slab.get_rows_by_indices("users", &[0, 1, 2]).unwrap();

        // Should only return 2 rows (index 0 was deleted)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_get_rows_by_indices_empty() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let results = slab.get_rows_by_indices("users", &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_get_rows_by_indices_out_of_bounds() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        slab.insert("users", create_test_row(1, "Alice", 30))
            .unwrap();

        let results = slab.get_rows_by_indices("users", &[0, 100, 200]).unwrap();

        // Should only return the valid index
        assert_eq!(results.len(), 1);
    }

    // Tests for get_int_column

    #[test]
    fn test_get_int_column() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        slab.insert("users", create_test_row(10, "Alice", 30))
            .unwrap();
        slab.insert("users", create_test_row(20, "Bob", 25))
            .unwrap();
        slab.insert("users", create_test_row(30, "Charlie", 35))
            .unwrap();

        let (values, alive, _nulls) = slab.get_int_column("users", "id").unwrap();

        assert_eq!(values.len(), 3);
        assert_eq!(values[0], 10);
        assert_eq!(values[1], 20);
        assert_eq!(values[2], 30);
        assert!(!alive.is_empty());
    }

    #[test]
    fn test_get_int_column_with_deleted() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let id1 = slab
            .insert("users", create_test_row(10, "Alice", 30))
            .unwrap();
        slab.insert("users", create_test_row(20, "Bob", 25))
            .unwrap();

        slab.delete("users", id1).unwrap();

        let (values, alive, _nulls) = slab.get_int_column("users", "id").unwrap();

        // Values still there, but alive bitmap reflects deletion
        assert_eq!(values.len(), 2);
        // First bit in alive should be 0 (deleted)
        assert_eq!(alive[0] & 1, 0);
    }

    #[test]
    fn test_get_int_column_wrong_type() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        slab.insert("users", create_test_row(1, "Alice", 30))
            .unwrap();

        let result = slab.get_int_column("users", "name");
        assert!(matches!(result, Err(RelationalError::ColumnNotFound(_))));
    }

    #[test]
    fn test_get_int_column_not_found() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        let result = slab.get_int_column("users", "nonexistent");
        assert!(matches!(result, Err(RelationalError::ColumnNotFound(_))));
    }

    // Tests for get_float_column

    #[test]
    fn test_get_float_column() {
        let slab = RelationalSlab::new();
        let schema = TableSchema::new(vec![
            ColumnDef::new("id", ColumnType::Int, false),
            ColumnDef::new("score", ColumnType::Float, false),
        ]);
        slab.create_table("scores", schema).unwrap();

        slab.insert(
            "scores",
            vec![ColumnValue::Int(1), ColumnValue::Float(95.5)],
        )
        .unwrap();
        slab.insert(
            "scores",
            vec![ColumnValue::Int(2), ColumnValue::Float(87.3)],
        )
        .unwrap();

        let (values, alive, _nulls) = slab.get_float_column("scores", "score").unwrap();

        assert_eq!(values.len(), 2);
        assert!((values[0] - 95.5).abs() < f64::EPSILON);
        assert!((values[1] - 87.3).abs() < f64::EPSILON);
        assert!(!alive.is_empty());
    }

    #[test]
    fn test_get_float_column_wrong_type() {
        let slab = RelationalSlab::new();
        slab.create_table("users", create_test_schema()).unwrap();

        slab.insert("users", create_test_row(1, "Alice", 30))
            .unwrap();

        let result = slab.get_float_column("users", "id");
        assert!(matches!(result, Err(RelationalError::ColumnNotFound(_))));
    }

    // Tests for error types

    #[test]
    fn test_row_not_found_error_display() {
        let err = RelationalError::RowNotFound(RowId::new(42));
        assert_eq!(format!("{err}"), "row not found: 42");
    }

    #[test]
    fn test_column_not_found_error_display() {
        let err = RelationalError::ColumnNotFound("missing".to_string());
        assert_eq!(format!("{err}"), "column not found: missing");
    }

    // Tests for bytes/json update

    #[test]
    fn test_update_bytes_column() {
        let slab = RelationalSlab::new();
        let schema = TableSchema::new(vec![
            ColumnDef::new("id", ColumnType::Int, false),
            ColumnDef::new("data", ColumnType::Bytes, true),
        ]);
        slab.create_table("bindata", schema).unwrap();

        let row_id = slab
            .insert(
                "bindata",
                vec![ColumnValue::Int(1), ColumnValue::Bytes(vec![1, 2, 3])],
            )
            .unwrap();

        let updates = vec![("data".to_string(), ColumnValue::Bytes(vec![4, 5, 6, 7]))];
        slab.update_row("bindata", row_id, &updates).unwrap();

        let retrieved = slab.get("bindata", row_id).unwrap().unwrap();
        assert_eq!(retrieved[1], ColumnValue::Bytes(vec![4, 5, 6, 7]));
    }

    #[test]
    fn test_update_json_column() {
        let slab = RelationalSlab::new();
        let schema = TableSchema::new(vec![
            ColumnDef::new("id", ColumnType::Int, false),
            ColumnDef::new("meta", ColumnType::Json, true),
        ]);
        slab.create_table("jsondata", schema).unwrap();

        let row_id = slab
            .insert(
                "jsondata",
                vec![
                    ColumnValue::Int(1),
                    ColumnValue::Json(r#"{"a":1}"#.to_string()),
                ],
            )
            .unwrap();

        let updates = vec![(
            "meta".to_string(),
            ColumnValue::Json(r#"{"b":2}"#.to_string()),
        )];
        slab.update_row("jsondata", row_id, &updates).unwrap();

        let retrieved = slab.get("jsondata", row_id).unwrap().unwrap();
        assert_eq!(retrieved[1], ColumnValue::Json(r#"{"b":2}"#.to_string()));
    }

    // Concurrent tests for new methods

    #[test]
    fn test_concurrent_update_row() {
        let slab = Arc::new(RelationalSlab::new());
        slab.create_table("users", create_test_schema()).unwrap();

        // Insert 100 rows
        for i in 0..100 {
            slab.insert("users", create_test_row(i, &format!("User{i}"), 20))
                .unwrap();
        }

        // Spawn threads to update rows concurrently
        let handles: Vec<_> = (0..10)
            .map(|t| {
                let slab = Arc::clone(&slab);
                thread::spawn(move || {
                    for i in 0..10 {
                        let row_id = RowId::new((t * 10 + i) as u64);
                        let updates = vec![(
                            "age".to_string(),
                            ColumnValue::Int(100 + (t * 10 + i) as i64),
                        )];
                        slab.update_row("users", row_id, &updates).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Verify all updates
        for i in 0..100 {
            let row = slab.get("users", RowId::new(i)).unwrap().unwrap();
            assert_eq!(row[2], ColumnValue::Int(100 + i as i64));
        }
    }

    #[test]
    fn test_concurrent_get_rows_by_indices() {
        let slab = Arc::new(RelationalSlab::new());
        slab.create_table("users", create_test_schema()).unwrap();

        for i in 0..100 {
            slab.insert("users", create_test_row(i, &format!("User{i}"), 20))
                .unwrap();
        }

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let slab = Arc::clone(&slab);
                thread::spawn(move || {
                    let indices: Vec<usize> = (0..100).step_by(2).collect();
                    let results = slab.get_rows_by_indices("users", &indices).unwrap();
                    assert_eq!(results.len(), 50);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }
}
